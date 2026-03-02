#!/usr/bin/env python3
"""
Plot validation-wave retrieval performance with a random baseline.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retrieval_common import normalize_rows, pick_embedding_file
from validate_archetype_retrieval import load_latest_run
from validate_archetype_retrieval import (
    load_feature_tables,
    load_validation_tag_dataset,
)


def to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def infer_top_k(metrics_df: pd.DataFrame) -> int:
    ks = []
    for col in metrics_df.columns:
        m = re.match(r"oracle_hit_at_(\d+)", str(col))
        if m:
            ks.append(int(m.group(1)))
    return max(ks) if ks else 5


def random_baseline_metrics(
    bank_embeddings: np.ndarray,
    validation_embeddings: np.ndarray,
    top_k: int,
    repeats: int,
    seed: int,
) -> Dict[str, float]:
    bank_n = normalize_rows(bank_embeddings.astype(np.float32, copy=False))
    val_n = normalize_rows(validation_embeddings.astype(np.float32, copy=False))

    sim_true_bank = val_n @ bank_n.T
    oracle_idx = np.argmax(sim_true_bank, axis=1)
    oracle_true_cos = sim_true_bank[np.arange(len(val_n)), oracle_idx]

    rng = np.random.default_rng(seed)
    retrieved_means = []
    regret_means = []
    hit1_means = []
    for _ in range(repeats):
        pred_idx = rng.integers(0, len(bank_n), size=len(val_n))
        retrieved = np.sum(bank_n[pred_idx] * val_n, axis=1)
        retrieved_means.append(float(np.mean(retrieved)))
        regret_means.append(float(np.mean(oracle_true_cos - retrieved)))
        hit1_means.append(float(np.mean(pred_idx == oracle_idx)))

    k = min(top_k, len(bank_n))
    hitk = float(k / len(bank_n))
    ret = {
        "pred_true_cosine_mean": np.nan,
        "retrieved_true_cosine_mean": float(np.mean(retrieved_means)),
        "retrieved_true_cosine_std": float(np.std(retrieved_means, ddof=0)),
        "oracle_true_cosine_mean": float(np.mean(oracle_true_cos)),
        "retrieval_cosine_regret_mean": float(np.mean(regret_means)),
        "retrieval_cosine_regret_std": float(np.std(regret_means, ddof=0)),
        "oracle_hit_at_1": float(np.mean(hit1_means)),
        "oracle_hit_at_1_std": float(np.std(hit1_means, ddof=0)),
        f"oracle_hit_at_{k}": hitk,
        f"oracle_hit_at_{k}_std": 0.0,
        "embedding_mse_mean": np.nan,
    }
    return ret


def build_random_rows(
    run_dir: Path,
    validation_wave_root: Path,
    metrics_df: pd.DataFrame,
    top_k: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for tag in sorted(metrics_df["tag"].unique()):
        bank_path = run_dir / tag / "bank_embeddings.npy"
        val_emb_path = pick_embedding_file(validation_wave_root / tag)
        if not bank_path.exists():
            continue
        bank = np.load(bank_path)
        val = np.load(val_emb_path)
        if val.ndim == 1:
            val = val.reshape(1, -1)

        m = random_baseline_metrics(
            bank_embeddings=bank,
            validation_embeddings=val,
            top_k=top_k,
            repeats=repeats,
            seed=seed,
        )
        row = {
            "tag": tag,
            "model": "random",
            "n_validation_rows": int(len(val)),
            "status": "ok",
            **m,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def model_order(models: List[str]) -> List[str]:
    preferred = ["ridge", "linear", "elastic_net", "mlp", "mean", "random"]
    out = [m for m in preferred if m in models]
    out.extend([m for m in models if m not in out])
    return out


def _sample_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bank_vectors: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_true_n = normalize_rows(y_true.astype(np.float32, copy=False))
    y_pred_n = normalize_rows(y_pred.astype(np.float32, copy=False))
    bank_n = normalize_rows(bank_vectors.astype(np.float32, copy=False))

    sim_pred_bank = y_pred_n @ bank_n.T
    pred_idx = np.argmax(sim_pred_bank, axis=1)

    sim_true_bank = y_true_n @ bank_n.T
    oracle_idx = np.argmax(sim_true_bank, axis=1)

    k = min(top_k, bank_n.shape[0])
    if k <= 1:
        topk_idx = pred_idx.reshape(-1, 1)
    else:
        topk_idx = np.argpartition(-sim_pred_bank, kth=k - 1, axis=1)[:, :k]

    retrieved_true_cos = np.sum(bank_n[pred_idx] * y_true_n, axis=1)
    oracle_true_cos = sim_true_bank[np.arange(sim_true_bank.shape[0]), oracle_idx]
    regret = oracle_true_cos - retrieved_true_cos
    hit1 = (pred_idx == oracle_idx).astype(np.float32)
    hitk = np.array(
        [oracle_idx[i] in topk_idx[i] for i in range(len(oracle_idx))], dtype=np.float32
    )
    return retrieved_true_cos, regret, hit1, hitk


def _bootstrap_std(
    metric_values: np.ndarray,
    repeats: int,
    seed: int,
) -> float:
    n = len(metric_values)
    if n <= 1:
        return 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(metric_values[idx]))
    return float(np.std(means, ddof=0))


def add_better_arrow(ax: plt.Axes, higher_is_better: bool) -> None:
    x = 1.02
    y_low = 0.08
    y_high = 0.92
    if higher_is_better:
        xy = (x, y_high)
        xytext = (x, y_low)
        text_rotation = 90
    else:
        xy = (x, y_low)
        xytext = (x, y_high)
        text_rotation = -90

    ax.annotate(
        "",
        xy=xy,
        xytext=xytext,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={"arrowstyle": "-|>", "linewidth": 1.6, "color": "black"},
        annotation_clip=False,
    )
    ax.text(
        x + 0.035,
        0.5,
        "better",
        rotation=text_rotation,
        va="center",
        ha="center",
        transform=ax.transAxes,
    )


def add_model_error_bars(
    combined: pd.DataFrame,
    run_dir: Path,
    validation_wave_root: Path,
    demographics_csv: Path,
    environment_csv: Path,
    top_k: int,
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    demo, env = load_feature_tables(demographics_csv, environment_csv)

    out_rows = []
    cache: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = {}
    for _, row in combined.iterrows():
        model = str(row["model"])
        tag = str(row["tag"])
        row_out = row.to_dict()

        if model == "random":
            out_rows.append(row_out)
            continue

        key = tag
        if key not in cache:
            model_path = run_dir / tag / "models" / f"{model}.joblib"
            if not model_path.exists():
                out_rows.append(row_out)
                continue
            artifact = joblib.load(model_path)
            feature_columns = artifact.get("feature_columns", [])
            if not feature_columns:
                out_rows.append(row_out)
                continue
            X_val, y_val, _, _ = load_validation_tag_dataset(
                validation_wave_root=validation_wave_root,
                tag=tag,
                demo=demo,
                env=env,
                feature_columns=feature_columns,
            )
            cache[key] = (X_val, y_val)

        X_val, y_val = cache[key]
        if len(X_val) == 0:
            row_out["retrieval_cosine_regret_std"] = 0.0
            row_out["oracle_hit_at_1_std"] = 0.0
            row_out[f"oracle_hit_at_{top_k}_std"] = 0.0
            out_rows.append(row_out)
            continue

        bank = np.load(run_dir / tag / "bank_embeddings.npy")
        artifact = joblib.load(run_dir / tag / "models" / f"{model}.joblib")
        if model == "mean":
            y_pred = np.repeat(
                artifact["baseline_mean_vector"].reshape(1, -1), repeats=len(X_val), axis=0
            ).astype(np.float32)
        else:
            est = artifact["estimator"]
            y_pred = est.predict(X_val)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(1, -1)
            y_pred = y_pred.astype(np.float32, copy=False)

        _, regret_vals, hit1_vals, hitk_vals = _sample_level_metrics(
            y_true=y_val,
            y_pred=y_pred,
            bank_vectors=bank,
            top_k=top_k,
        )
        row_out["retrieval_cosine_regret_std"] = _bootstrap_std(
            regret_vals, repeats=bootstrap_repeats, seed=bootstrap_seed
        )
        row_out["oracle_hit_at_1_std"] = _bootstrap_std(
            hit1_vals, repeats=bootstrap_repeats, seed=bootstrap_seed + 1
        )
        row_out[f"oracle_hit_at_{top_k}_std"] = _bootstrap_std(
            hitk_vals, repeats=bootstrap_repeats, seed=bootstrap_seed + 2
        )
        out_rows.append(row_out)

    return pd.DataFrame(out_rows)


def plot_grouped_bar(
    df: pd.DataFrame,
    metric: str,
    metric_std: str | None,
    ylabel: str,
    title: str,
    out_path: Path,
    higher_is_better: bool,
    top_k: int,
) -> None:
    pivot = df.pivot(index="tag", columns="model", values=metric).sort_index()
    tags = list(pivot.index)
    models = model_order(list(pivot.columns))
    x = np.arange(len(tags), dtype=float)
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, model in enumerate(models):
        vals = pivot[model].to_numpy()
        if metric_std and metric_std in df.columns:
            err_pivot = df.pivot(index="tag", columns="model", values=metric_std).sort_index()
            yerr = err_pivot[model].to_numpy()
        else:
            yerr = None
        ax.bar(
            x + (i - (len(models) - 1) / 2) * width,
            vals,
            width=width,
            label=model,
            yerr=yerr,
            capsize=2 if yerr is not None else 0,
        )
    ax.set_xticks(x, tags, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title.replace("{k}", str(top_k)))
    ax.legend(ncol=min(6, len(models)), frameon=False)
    add_better_arrow(ax, higher_is_better=higher_is_better)
    fig.tight_layout(rect=[0, 0, 0.96, 1])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_aggregate(
    df: pd.DataFrame,
    hitk_col: str,
    out_path: Path,
) -> None:
    ok = df[df["status"] == "ok"].copy()
    agg_mean = (
        ok.groupby("model", as_index=False)[
            ["retrieval_cosine_regret_mean", "oracle_hit_at_1", hitk_col]
        ]
        .mean(numeric_only=True)
    )
    agg_std = (
        ok.groupby("model", as_index=False)[
            ["retrieval_cosine_regret_mean", "oracle_hit_at_1", hitk_col]
        ]
        .std(ddof=0, numeric_only=True)
    )
    agg = agg_mean.merge(agg_std, on="model", suffixes=("_mean", "_std")).sort_values(
        "retrieval_cosine_regret_mean_mean"
    )
    models = list(agg["model"])
    x = np.arange(len(models), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(
        x,
        agg["retrieval_cosine_regret_mean_mean"],
        yerr=agg["retrieval_cosine_regret_mean_std"],
        capsize=3,
    )
    axes[0].set_xticks(x, models, rotation=30, ha="right")
    axes[0].set_ylabel("Mean Regret (lower better)")
    axes[0].set_title("Across Tags: Regret (error bars=tag std)")
    add_better_arrow(axes[0], higher_is_better=False)

    axes[1].bar(
        x,
        agg[f"{hitk_col}_mean"],
        yerr=agg[f"{hitk_col}_std"],
        capsize=3,
    )
    axes[1].set_xticks(x, models, rotation=30, ha="right")
    axes[1].set_ylabel(f"Mean {hitk_col} (higher better)")
    axes[1].set_title("Across Tags: Hit@K (error bars=tag std)")
    add_better_arrow(axes[1], higher_is_better=True)

    fig.tight_layout(rect=[0, 0, 0.96, 1])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot validation performance and random baseline."
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Persona/archetype_retrieval/model_runs_learning_for_validation"),
        help="Used only if --run-dir is omitted.",
    )
    parser.add_argument(
        "--validation-wave-root",
        type=Path,
        default=Path("Persona/archetype_retrieval/validation_wave"),
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--exclude-models",
        nargs="+",
        default=["mean"],
        help="Models to exclude from the plotted comparison.",
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
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=200,
        help="Bootstrap repeats for model error bars.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir is not None else load_latest_run(args.output_root)
    eval_dir = run_dir / "validation_eval"
    metrics_path = eval_dir / "validation_metrics_all_tags.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing validation metrics file: {metrics_path}")

    metrics_df = pd.read_csv(metrics_path)
    excluded = {m.strip().lower() for m in args.exclude_models}
    metrics_df = metrics_df[~metrics_df["model"].str.lower().isin(excluded)].copy()
    top_k = args.top_k or infer_top_k(metrics_df)
    hitk_col = f"oracle_hit_at_{top_k}"
    if hitk_col not in metrics_df.columns:
        raise ValueError(f"Could not find {hitk_col} in {metrics_path}")

    random_df = build_random_rows(
        run_dir=run_dir,
        validation_wave_root=args.validation_wave_root,
        metrics_df=metrics_df,
        top_k=top_k,
        repeats=args.random_repeats,
        seed=args.random_seed,
    )

    combined = pd.concat([metrics_df, random_df], ignore_index=True, sort=False)
    combined = add_model_error_bars(
        combined=combined,
        run_dir=run_dir,
        validation_wave_root=args.validation_wave_root,
        demographics_csv=args.demographics_csv,
        environment_csv=args.environment_csv,
        top_k=top_k,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.random_seed,
    )
    combined_path = eval_dir / "validation_metrics_with_random.csv"
    combined.to_csv(combined_path, index=False)

    summary = (
        combined[combined["status"] == "ok"]
        .groupby("model", as_index=False)[
            ["retrieval_cosine_regret_mean", "oracle_hit_at_1", hitk_col]
        ]
        .mean()
        .sort_values("retrieval_cosine_regret_mean")
    )
    summary_path = eval_dir / "validation_metrics_summary_with_random.csv"
    summary.to_csv(summary_path, index=False)

    fig_dir = eval_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig1 = fig_dir / "per_tag_regret_with_random.png"
    plot_grouped_bar(
        df=combined[combined["status"] == "ok"],
        metric="retrieval_cosine_regret_mean",
        metric_std="retrieval_cosine_regret_std",
        ylabel="Mean Regret (lower better)",
        title="Per Tag: Retrieval Regret (includes random baseline)",
        out_path=fig1,
        higher_is_better=False,
        top_k=top_k,
    )

    fig2 = fig_dir / f"per_tag_hit_at_{top_k}_with_random.png"
    plot_grouped_bar(
        df=combined[combined["status"] == "ok"],
        metric=hitk_col,
        metric_std=f"{hitk_col}_std",
        ylabel=f"{hitk_col} (higher better)",
        title="Per Tag: Hit@{k} (includes random baseline)",
        out_path=fig2,
        higher_is_better=True,
        top_k=top_k,
    )

    fig3 = fig_dir / "across_tags_summary_with_random.png"
    plot_aggregate(
        df=combined,
        hitk_col=hitk_col,
        out_path=fig3,
    )

    print(f"Wrote: {to_repo_rel(combined_path)}")
    print(f"Wrote: {to_repo_rel(summary_path)}")
    print(f"Wrote: {to_repo_rel(fig1)}")
    print(f"Wrote: {to_repo_rel(fig2)}")
    print(f"Wrote: {to_repo_rel(fig3)}")
    print()
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
