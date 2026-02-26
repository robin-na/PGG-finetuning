#!/usr/bin/env python3
"""
Validate trained archetype-retrieval models on a separate wave.

This script expects:
- A trained run directory from train_archetype_retrieval.py (learning wave).
- Validation-wave tag folders with *_sections_input.jsonl + embedding .npy files.
- Validation demographics and environment CSV files.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from retrieval_common import (
    DEMOGRAPHIC_KEY_COLUMNS,
    coerce_feature_frame,
    is_tag_active,
    load_jsonl,
    normalize_key,
    normalize_rows,
    parse_bool,
    pick_embedding_file,
    pick_sections_input_jsonl,
)


def to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def load_latest_run(output_root: Path) -> Path:
    latest = output_root / "latest_run.txt"
    if not latest.exists():
        raise FileNotFoundError(f"No latest run pointer found at {latest}")
    return Path(latest.read_text(encoding="utf-8").strip())


def _row_id(row: Dict[str, Any], primary: str, alias: str) -> str:
    value = row.get(primary)
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value) == "":
        value = row.get(alias)
    return normalize_key(value)


def load_feature_tables(
    demographics_csv: Path,
    environment_csv: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    demo = pd.read_csv(demographics_csv)
    env = pd.read_csv(environment_csv)

    for key in DEMOGRAPHIC_KEY_COLUMNS:
        demo[key] = demo[key].map(normalize_key)
    env["gameId"] = env["gameId"].map(normalize_key)

    demo = demo.drop_duplicates(subset=DEMOGRAPHIC_KEY_COLUMNS, keep="first")
    env = env.drop_duplicates(subset=["gameId"], keep="first")
    return demo, env


def load_validation_tag_dataset(
    validation_wave_root: Path,
    tag: str,
    demo: pd.DataFrame,
    env: pd.DataFrame,
    feature_columns: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    tag_dir = validation_wave_root / tag
    sections_path = pick_sections_input_jsonl(tag_dir)
    embedding_path = pick_embedding_file(tag_dir)

    rows = load_jsonl(sections_path)
    embeddings = np.load(embedding_path)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if len(rows) != embeddings.shape[0]:
        raise ValueError(
            f"{tag}: validation rows in {sections_path.name} ({len(rows)}) do not match "
            f"embedding vectors ({embeddings.shape[0]})."
        )

    base = pd.DataFrame(
        {
            "gameId": [_row_id(r, "gameId", "experiment") for r in rows],
            "playerId": [_row_id(r, "playerId", "participant") for r in rows],
            "game_finished": [parse_bool(r.get("game_finished")) for r in rows],
            "section_text": [
                str(r.get("section_text") or r.get("text") or "").strip() for r in rows
            ],
        }
    )

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

    X = coerce_feature_frame(merged, feature_columns)
    y = embeddings.astype(np.float32, copy=False)
    metadata = merged[["gameId", "playerId", "section_text", "game_finished"]].copy()
    metadata["tag"] = tag

    stats = {
        "validation_sections_file": to_repo_rel(sections_path),
        "validation_embedding_file": to_repo_rel(embedding_path),
        "rows_raw": len(rows),
        "rows_after_activation_filter": int(len(merged)),
        "dropped_inactive_rows": dropped_inactive,
        "missing_demographics_rows": missing_demo,
        "missing_environment_rows": missing_env,
        "unique_games": int(merged["gameId"].nunique()),
        "embedding_dim": int(y.shape[1]) if len(y) else None,
    }
    return X, y, metadata, stats


def retrieval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bank_vectors: np.ndarray,
    top_k: int,
) -> Dict[str, float]:
    if y_true.shape[0] == 0 or bank_vectors.shape[0] == 0:
        raise ValueError("Empty y_true or bank_vectors for validation metrics.")

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


def infer_models_for_tag(tag_run_dir: Path) -> List[str]:
    models_dir = tag_run_dir / "models"
    if not models_dir.is_dir():
        return []
    return sorted([p.stem for p in models_dir.glob("*.joblib")])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate learning-wave trained models on validation wave."
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Persona/archetype_retrieval/model_runs"),
        help="Used only when --run-dir is omitted (reads latest_run.txt).",
    )
    parser.add_argument(
        "--validation-wave-root",
        type=Path,
        default=Path("Persona/archetype_retrieval/validation_wave"),
    )
    parser.add_argument(
        "--demographics-csv",
        type=Path,
        default=Path("demographics/demographics_numeric_val.csv"),
    )
    parser.add_argument(
        "--environment-csv",
        type=Path,
        default=Path("data/processed_data/df_analysis_val.csv"),
    )
    parser.add_argument("--tags", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir is not None else load_latest_run(args.output_root)

    out_dir = run_dir / "validation_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    demo, env = load_feature_tables(args.demographics_csv, args.environment_csv)

    tag_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and (p / "bank_embeddings.npy").exists()])
    default_tags = [p.name for p in tag_dirs]
    tags = [t.strip().upper() for t in (args.tags or default_tags) if str(t).strip()]

    results_by_tag: List[Dict[str, Any]] = []
    errors: List[str] = []

    for tag in tags:
        print(f"[{tag}] loading validation data...")
        tag_run_dir = run_dir / tag
        if not tag_run_dir.is_dir():
            errors.append(f"{tag}: missing tag directory in run {to_repo_rel(run_dir)}")
            continue

        model_names = [m.strip().lower() for m in (args.models or infer_models_for_tag(tag_run_dir))]
        if not model_names:
            errors.append(f"{tag}: no models found for evaluation.")
            continue

        bank_embeddings = np.load(tag_run_dir / "bank_embeddings.npy")

        feature_columns: Optional[List[str]] = None
        artifacts: Dict[str, Dict[str, Any]] = {}
        for model_name in model_names:
            model_path = tag_run_dir / "models" / f"{model_name}.joblib"
            if not model_path.exists():
                errors.append(f"{tag}: missing model artifact {to_repo_rel(model_path)}")
                continue
            artifact = joblib.load(model_path)
            artifacts[model_name] = artifact
            if feature_columns is None:
                feature_columns = artifact.get("feature_columns")

        if not feature_columns:
            errors.append(f"{tag}: could not determine feature columns from model artifacts.")
            continue

        try:
            X_val, y_val, val_meta, val_stats = load_validation_tag_dataset(
                validation_wave_root=args.validation_wave_root,
                tag=tag,
                demo=demo,
                env=env,
                feature_columns=feature_columns,
            )
        except Exception as exc:
            errors.append(f"{tag}: {exc}")
            continue

        tag_metrics_rows: List[Dict[str, Any]] = []
        for model_name in model_names:
            if model_name not in artifacts:
                continue
            artifact = artifacts[model_name]
            row: Dict[str, Any] = {
                "tag": tag,
                "model": model_name,
                "n_validation_rows": int(len(X_val)),
                "n_validation_games": int(val_meta["gameId"].nunique()) if len(val_meta) else 0,
                "status": "ok",
            }
            try:
                if len(X_val) == 0:
                    raise ValueError("No validation rows after activation filter.")
                if model_name == "mean":
                    y_pred = np.repeat(
                        artifact["baseline_mean_vector"].reshape(1, -1),
                        repeats=len(X_val),
                        axis=0,
                    ).astype(np.float32)
                else:
                    est = artifact["estimator"]
                    y_pred = est.predict(X_val)
                    if y_pred.ndim == 1:
                        y_pred = y_pred.reshape(1, -1)
                    y_pred = y_pred.astype(np.float32, copy=False)

                metrics = retrieval_metrics(
                    y_true=y_val,
                    y_pred=y_pred,
                    bank_vectors=bank_embeddings,
                    top_k=args.top_k,
                )
                row.update(metrics)
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
            tag_metrics_rows.append(row)
            results_by_tag.append(row)

        tag_metrics_df = pd.DataFrame(tag_metrics_rows)
        tag_metrics_path = out_dir / f"{tag}_validation_metrics.csv"
        tag_metrics_df.to_csv(tag_metrics_path, index=False)
        with (out_dir / f"{tag}_validation_data_stats.json").open("w", encoding="utf-8") as f:
            json.dump(val_stats, f, ensure_ascii=False, indent=2)
            f.write("\n")

    all_df = pd.DataFrame(results_by_tag)
    all_path = out_dir / "validation_metrics_all_tags.csv"
    all_df.to_csv(all_path, index=False)

    ok_df = all_df[all_df.get("status") == "ok"] if not all_df.empty else pd.DataFrame()
    if not ok_df.empty:
        metric_cols = [
            c
            for c in ok_df.columns
            if c not in {"tag", "model", "n_validation_rows", "n_validation_games", "status", "error"}
        ]
        summary = (
            ok_df.groupby("model", as_index=False)[metric_cols]
            .mean(numeric_only=True)
            .sort_values("retrieval_cosine_regret_mean")
        )
    else:
        summary = pd.DataFrame()
    summary_path = out_dir / "validation_metrics_summary_by_model.csv"
    summary.to_csv(summary_path, index=False)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": to_repo_rel(run_dir),
        "validation_wave_root": to_repo_rel(args.validation_wave_root),
        "demographics_csv": to_repo_rel(args.demographics_csv),
        "environment_csv": to_repo_rel(args.environment_csv),
        "tags": tags,
        "models_override": args.models,
        "top_k": args.top_k,
        "files": {
            "validation_metrics_all_tags_csv": to_repo_rel(all_path),
            "validation_metrics_summary_by_model_csv": to_repo_rel(summary_path),
        },
        "errors": errors,
    }
    manifest_path = out_dir / "validation_eval_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Validation eval written to: {to_repo_rel(out_dir)}")
    if not summary.empty:
        print(summary.to_string(index=False))
    if errors:
        print("Completed with errors:")
        for e in errors:
            print(f"- {e}")
        return 1
    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

