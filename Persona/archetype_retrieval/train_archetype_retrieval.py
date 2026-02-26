#!/usr/bin/env python3
"""
Train and evaluate archetype retrieval models per tag.

Pipeline:
1) Merge tag-level examples with demographics (D) and environment config (E).
2) Train embedding regressors from D+E -> embedding using group CV by gameId.
3) Evaluate retrieval quality by nearest-neighbor lookup in training-fold banks.
4) Fit final models on full tag data and export retrieval artifacts.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, MultiTaskElasticNet, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from retrieval_common import (
    CONFIG_FEATURE_COLUMNS,
    DEFAULT_TAGS,
    DEMOGRAPHIC_KEY_COLUMNS,
    coerce_feature_frame,
    is_tag_active,
    load_jsonl,
    normalize_key,
    normalize_rows,
    parse_bool,
    pick_embedding_file,
    pick_sections_input_jsonl,
    write_jsonl,
)


DEFAULT_MODELS = ["mean", "linear", "ridge", "elastic_net", "mlp"]


def to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def build_estimator(
    model_name: str,
    random_state: int,
    mlp_max_iter: int,
    mlp_hidden_layers: Tuple[int, ...],
) -> Pipeline:
    if model_name == "linear":
        model = LinearRegression()
    elif model_name == "ridge":
        model = Ridge(alpha=10.0, random_state=random_state)
    elif model_name == "elastic_net":
        model = MultiTaskElasticNet(
            alpha=0.01,
            l1_ratio=0.2,
            max_iter=300,
            tol=1e-3,
            random_state=random_state,
        )
    elif model_name == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=mlp_hidden_layers,
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=mlp_max_iter,
            early_stopping=True,
            batch_size=64,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def retrieval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bank_vectors: np.ndarray,
    top_k: int,
) -> Dict[str, float]:
    if y_true.shape[0] == 0 or bank_vectors.shape[0] == 0:
        raise ValueError("Empty y_true or bank_vectors in retrieval_metrics.")

    y_true_n = normalize_rows(y_true)
    y_pred_n = normalize_rows(y_pred)
    bank_n = normalize_rows(bank_vectors)

    sim_pred_bank = y_pred_n @ bank_n.T
    pred_idx = np.argmax(sim_pred_bank, axis=1)

    sim_true_bank = y_true_n @ bank_n.T
    oracle_idx = np.argmax(sim_true_bank, axis=1)

    k = min(top_k, bank_n.shape[0])
    if k <= 1:
        topk_idx = pred_idx.reshape(-1, 1)
    else:
        topk_idx = np.argpartition(-sim_pred_bank, kth=k - 1, axis=1)[:, :k]

    hit_at_1 = float(np.mean(pred_idx == oracle_idx))
    hit_at_k = float(
        np.mean([oracle_idx[i] in topk_idx[i] for i in range(len(oracle_idx))])
    )

    pred_true_cos = np.sum(y_pred_n * y_true_n, axis=1)
    retrieved_true_cos = np.sum(bank_n[pred_idx] * y_true_n, axis=1)
    oracle_true_cos = sim_true_bank[np.arange(sim_true_bank.shape[0]), oracle_idx]
    mse = np.mean((y_true - y_pred) ** 2, axis=1)

    return {
        "pred_true_cosine_mean": float(np.mean(pred_true_cos)),
        "retrieved_true_cosine_mean": float(np.mean(retrieved_true_cos)),
        "oracle_true_cosine_mean": float(np.mean(oracle_true_cos)),
        "retrieval_cosine_regret_mean": float(
            np.mean(oracle_true_cos - retrieved_true_cos)
        ),
        "oracle_hit_at_1": hit_at_1,
        f"oracle_hit_at_{k}": hit_at_k,
        "embedding_mse_mean": float(np.mean(mse)),
    }


def load_feature_tables(
    demographics_csv: Path,
    environment_csv: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], Dict[str, Any]]:
    demo = pd.read_csv(demographics_csv)
    env = pd.read_csv(environment_csv)

    for key in DEMOGRAPHIC_KEY_COLUMNS:
        demo[key] = demo[key].map(normalize_key)
    env["gameId"] = env["gameId"].map(normalize_key)

    demo_before = len(demo)
    demo = demo.drop_duplicates(subset=DEMOGRAPHIC_KEY_COLUMNS, keep="first")

    env_before = len(env)
    env = env.drop_duplicates(subset=["gameId"], keep="first")

    demo_feature_cols = [
        c for c in demo.columns if c not in set(DEMOGRAPHIC_KEY_COLUMNS)
    ]
    env_feature_cols = [c for c in CONFIG_FEATURE_COLUMNS if c in env.columns]
    env = env[["gameId", *env_feature_cols]].copy()

    stats = {
        "demographics_rows_raw": demo_before,
        "demographics_rows_dedup": len(demo),
        "environment_rows_raw": env_before,
        "environment_rows_dedup": len(env),
        "demographic_feature_count": len(demo_feature_cols),
        "environment_feature_count": len(env_feature_cols),
    }
    return demo, env, demo_feature_cols, env_feature_cols, stats


def load_tag_dataset(
    learning_wave_root: Path,
    tag: str,
    demo: pd.DataFrame,
    env: pd.DataFrame,
    feature_columns: List[str],
    max_rows: Optional[int],
    random_state: int,
) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, pd.DataFrame, Dict[str, Any]]:
    tag_dir = learning_wave_root / tag
    sections_path = pick_sections_input_jsonl(tag_dir)
    embedding_path = pick_embedding_file(tag_dir)

    rows = load_jsonl(sections_path)
    embeddings = np.load(embedding_path)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    if len(rows) != embeddings.shape[0]:
        raise ValueError(
            f"{tag}: rows in {sections_path.name} ({len(rows)}) do not match "
            f"embedding vectors ({embeddings.shape[0]})."
        )

    def row_id(row: Dict[str, Any], primary: str, alias: str) -> str:
        value = row.get(primary)
        if value is None or (isinstance(value, float) and np.isnan(value)) or str(value) == "":
            value = row.get(alias)
        return normalize_key(value)

    base = pd.DataFrame(
        {
            "gameId": [row_id(r, "gameId", "experiment") for r in rows],
            "playerId": [row_id(r, "playerId", "participant") for r in rows],
            "game_finished": [parse_bool(r.get("game_finished")) for r in rows],
            "section_text": [
                str(r.get("section_text") or r.get("text") or "").strip() for r in rows
            ],
        }
    )
    base["source_row_idx"] = np.arange(len(base), dtype=int)

    merged = base.merge(demo, on=["gameId", "playerId"], how="left")
    merged = merged.merge(env, on="gameId", how="left")

    missing_demo = int(
        merged[[c for c in demo.columns if c not in {"gameId", "playerId"}]]
        .isna()
        .all(axis=1)
        .sum()
    )
    missing_env = int(
        merged[[c for c in env.columns if c != "gameId"]].isna().all(axis=1).sum()
    )

    is_active = merged.apply(lambda r: is_tag_active(tag, r.to_dict()), axis=1)
    dropped_inactive = int((~is_active).sum())
    if dropped_inactive > 0:
        merged = merged.loc[is_active].reset_index(drop=True)
        embeddings = embeddings[is_active.to_numpy()]

    if max_rows is not None and len(merged) > max_rows:
        sampled = merged.sample(n=max_rows, random_state=random_state).sort_values(
            "source_row_idx"
        )
        keep_idx = sampled.index.to_numpy()
        merged = sampled.reset_index(drop=True)
        embeddings = embeddings[keep_idx]

    X = coerce_feature_frame(merged[feature_columns], feature_columns)
    y = embeddings.astype(np.float32, copy=False)
    groups = merged["gameId"].copy()

    metadata = merged[
        ["gameId", "playerId", "section_text", "game_finished", "source_row_idx"]
    ].copy()
    metadata["tag"] = tag

    stats = {
        "sections_file": to_repo_rel(sections_path),
        "embedding_file": to_repo_rel(embedding_path),
        "rows_raw": len(rows),
        "rows_after_activation_filter": len(merged),
        "dropped_inactive_rows": dropped_inactive,
        "missing_demographics_rows": missing_demo,
        "missing_environment_rows": missing_env,
        "unique_games": int(groups.nunique()),
        "embedding_dim": int(y.shape[1]),
    }
    return X, y, groups, metadata, stats


def run_group_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    model_names: List[str],
    cv_splits: int,
    random_state: int,
    top_k: int,
    mlp_max_iter: int,
    mlp_hidden_layers: Tuple[int, ...],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_games = groups.nunique()
    if unique_games < 2:
        raise ValueError("Need at least 2 unique gameId values for grouped CV.")

    n_splits = min(cv_splits, unique_games)
    splitter = GroupKFold(n_splits=n_splits)

    fold_rows: List[Dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=groups), start=1
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        for model_name in model_names:
            row: Dict[str, Any] = {
                "fold": fold_idx,
                "model": model_name,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_train_games": int(groups.iloc[train_idx].nunique()),
                "n_test_games": int(groups.iloc[test_idx].nunique()),
                "status": "ok",
            }
            t0 = time.perf_counter()
            try:
                if model_name == "mean":
                    mean_vec = y_train.mean(axis=0, keepdims=True)
                    y_pred = np.repeat(mean_vec, repeats=len(test_idx), axis=0)
                else:
                    est = build_estimator(
                        model_name, random_state, mlp_max_iter, mlp_hidden_layers
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=ConvergenceWarning)
                        est.fit(X_train, y_train)
                    y_pred = est.predict(X_test)
                    if y_pred.ndim == 1:
                        y_pred = y_pred.reshape(1, -1)

                metrics = retrieval_metrics(
                    y_true=y_test,
                    y_pred=y_pred.astype(np.float32, copy=False),
                    bank_vectors=y_train,
                    top_k=top_k,
                )
                row.update(metrics)
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
            row["fit_eval_seconds"] = round(time.perf_counter() - t0, 4)
            fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)

    metric_cols = [
        c
        for c in fold_df.columns
        if c
        not in {
            "fold",
            "model",
            "n_train",
            "n_test",
            "n_train_games",
            "n_test_games",
            "status",
            "error",
            "fit_eval_seconds",
        }
    ]

    ok_df = fold_df[fold_df["status"] == "ok"].copy()
    summary_rows: List[Dict[str, Any]] = []
    for model_name, g in ok_df.groupby("model"):
        out: Dict[str, Any] = {
            "model": model_name,
            "folds_ok": int(len(g)),
            "fit_eval_seconds_mean": float(g["fit_eval_seconds"].mean()),
        }
        for m in metric_cols:
            out[f"{m}_mean"] = float(g[m].mean())
            out[f"{m}_std"] = float(g[m].std(ddof=0))
        summary_rows.append(out)

    summary_df = pd.DataFrame(summary_rows).sort_values("model").reset_index(drop=True)
    return fold_df, summary_df


def fit_full_models(
    X: pd.DataFrame,
    y: np.ndarray,
    model_names: List[str],
    feature_columns: List[str],
    tag: str,
    out_models_dir: Path,
    random_state: int,
    mlp_max_iter: int,
    mlp_hidden_layers: Tuple[int, ...],
) -> List[Dict[str, Any]]:
    out_models_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    for model_name in model_names:
        t0 = time.perf_counter()
        path = out_models_dir / f"{model_name}.joblib"
        rec: Dict[str, Any] = {"model": model_name, "path": to_repo_rel(path)}
        try:
            if model_name == "mean":
                artifact = {
                    "artifact_version": 1,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "tag": tag,
                    "model_name": model_name,
                    "feature_columns": feature_columns,
                    "baseline_mean_vector": y.mean(axis=0).astype(np.float32),
                }
            else:
                est = build_estimator(
                    model_name, random_state, mlp_max_iter, mlp_hidden_layers
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    est.fit(X, y)
                artifact = {
                    "artifact_version": 1,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "tag": tag,
                    "model_name": model_name,
                    "feature_columns": feature_columns,
                    "estimator": est,
                }
            joblib.dump(artifact, path)
            rec["status"] = "ok"
        except Exception as exc:
            rec["status"] = "error"
            rec["error"] = str(exc)
        rec["fit_seconds"] = round(time.perf_counter() - t0, 4)
        records.append(rec)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train archetype retrieval models with grouped CV by gameId."
    )
    parser.add_argument(
        "--learning-wave-root",
        type=Path,
        default=Path("Persona/archetype_retrieval/learning_wave"),
    )
    parser.add_argument(
        "--demographics-csv",
        type=Path,
        default=Path("demographics/demographics_numeric_learn.csv"),
    )
    parser.add_argument(
        "--environment-csv",
        type=Path,
        default=Path("data/processed_data/df_analysis_learn.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Persona/archetype_retrieval/model_runs"),
    )
    parser.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--mlp-max-iter", type=int, default=250)
    parser.add_argument(
        "--mlp-hidden-layers",
        type=str,
        default="64",
        help="Comma-separated hidden layer sizes for MLP (e.g., '128,64').",
    )
    parser.add_argument("--max-rows-per-tag", type=int, default=None)
    parser.add_argument(
        "--skip-full-fit",
        action="store_true",
        help="Run CV only; do not fit/export final models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tags = [str(t).strip().upper() for t in args.tags if str(t).strip()]
    models = [str(m).strip().lower() for m in args.models if str(m).strip()]
    allowed = {"mean", "linear", "ridge", "elastic_net", "mlp"}
    bad = [m for m in models if m not in allowed]
    if bad:
        raise ValueError(f"Unsupported models: {bad}. Allowed: {sorted(allowed)}")

    mlp_hidden_layers = tuple(
        int(x.strip())
        for x in str(args.mlp_hidden_layers).split(",")
        if x.strip()
    )
    if not mlp_hidden_layers:
        raise ValueError("mlp-hidden-layers must contain at least one positive integer.")
    if any(x <= 0 for x in mlp_hidden_layers):
        raise ValueError("mlp-hidden-layers values must be positive integers.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_dir = args.output_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    demo, env, demo_cols, env_cols, table_stats = load_feature_tables(
        args.demographics_csv, args.environment_csv
    )
    feature_columns = [*demo_cols, *env_cols]

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": {
            "learning_wave_root": to_repo_rel(args.learning_wave_root),
            "demographics_csv": to_repo_rel(args.demographics_csv),
            "environment_csv": to_repo_rel(args.environment_csv),
            "output_root": to_repo_rel(args.output_root),
            "tags": tags,
            "models": models,
            "cv_splits": args.cv_splits,
            "top_k": args.top_k,
            "random_state": args.random_state,
            "mlp_max_iter": args.mlp_max_iter,
            "mlp_hidden_layers": list(mlp_hidden_layers),
            "max_rows_per_tag": args.max_rows_per_tag,
            "skip_full_fit": args.skip_full_fit,
        },
        "feature_table_stats": table_stats,
        "feature_columns": feature_columns,
        "tag_results": {},
        "errors": [],
    }

    all_summary_frames: List[pd.DataFrame] = []

    for tag in tags:
        tag_out = run_dir / tag
        tag_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[{tag}] building dataset...")
        try:
            X, y, groups, metadata, data_stats = load_tag_dataset(
                learning_wave_root=args.learning_wave_root,
                tag=tag,
                demo=demo,
                env=env,
                feature_columns=feature_columns,
                max_rows=args.max_rows_per_tag,
                random_state=args.random_state,
            )

            print(
                f"[{tag}] rows={len(X)} games={groups.nunique()} emb_dim={y.shape[1]} "
                f"(dropped_inactive={data_stats['dropped_inactive_rows']})"
            )

            fold_df, summary_df = run_group_cv(
                X=X,
                y=y,
                groups=groups,
                model_names=models,
                cv_splits=args.cv_splits,
                random_state=args.random_state,
                top_k=args.top_k,
                mlp_max_iter=args.mlp_max_iter,
                mlp_hidden_layers=mlp_hidden_layers,
            )

            fold_path = tag_out / "cv_fold_metrics.csv"
            summary_path = tag_out / "cv_summary.csv"
            fold_df.to_csv(fold_path, index=False)
            summary_df.to_csv(summary_path, index=False)

            emb_path = tag_out / "bank_embeddings.npy"
            np.save(emb_path, y.astype(np.float32))
            meta_path = tag_out / "bank_metadata.jsonl"
            write_jsonl(meta_path, metadata.to_dict(orient="records"))

            stats_path = tag_out / "data_stats.json"
            with stats_path.open("w", encoding="utf-8") as f:
                json.dump(data_stats, f, indent=2)
                f.write("\n")

            full_fit_records: List[Dict[str, Any]] = []
            if not args.skip_full_fit:
                full_fit_records = fit_full_models(
                    X=X,
                    y=y,
                    model_names=models,
                    feature_columns=feature_columns,
                    tag=tag,
                    out_models_dir=tag_out / "models",
                    random_state=args.random_state,
                    mlp_max_iter=args.mlp_max_iter,
                    mlp_hidden_layers=mlp_hidden_layers,
                )

            tag_manifest = {
                "status": "ok",
                "rows": int(len(X)),
                "unique_games": int(groups.nunique()),
                "embedding_dim": int(y.shape[1]),
                "files": {
                    "cv_fold_metrics_csv": to_repo_rel(fold_path),
                    "cv_summary_csv": to_repo_rel(summary_path),
                    "bank_embeddings_npy": to_repo_rel(emb_path),
                    "bank_metadata_jsonl": to_repo_rel(meta_path),
                    "data_stats_json": to_repo_rel(stats_path),
                },
                "full_fit_records": full_fit_records,
            }
            manifest["tag_results"][tag] = tag_manifest

            if not summary_df.empty:
                tmp = summary_df.copy()
                tmp.insert(0, "tag", tag)
                all_summary_frames.append(tmp)

        except Exception as exc:
            msg = f"{tag}: {exc}"
            print(f"[{tag}] ERROR: {exc}")
            manifest["tag_results"][tag] = {"status": "error", "error": str(exc)}
            manifest["errors"].append(msg)

    if all_summary_frames:
        all_summary = pd.concat(all_summary_frames, ignore_index=True)
        all_summary_path = run_dir / "cv_summary_all_tags.csv"
        all_summary.to_csv(all_summary_path, index=False)
        manifest["files"] = {"cv_summary_all_tags_csv": to_repo_rel(all_summary_path)}

    manifest_path = run_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    latest_path = args.output_root / "latest_run.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")

    print(f"\nRun directory: {to_repo_rel(run_dir)}")
    print(f"Manifest: {to_repo_rel(manifest_path)}")
    print(f"Latest pointer: {to_repo_rel(latest_path)}")

    if manifest["errors"]:
        print("\nCompleted with errors:")
        for err in manifest["errors"]:
            print(f"- {err}")
        return 1
    print("\nCompleted successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
