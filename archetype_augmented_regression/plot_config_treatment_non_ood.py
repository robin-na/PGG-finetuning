from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from archetype_augmented_regression.bootstrap_inference import bootstrap_methods_by_groups
from archetype_augmented_regression.config_treatment_eval import (
    fit_config_treatment_predictions,
    infer_best_k_from_results,
)


METHOD_ORDER = ["config_only", "synthetic", "oracle", "noise_ceiling"]
METHOD_LABEL = {
    "config_only": "CONFIG only",
    "synthetic": "Synthetic style",
    "oracle": "Oracle style",
    "noise_ceiling": "Noise ceiling",
}
METHOD_COLOR = {
    "config_only": "#6b7280",
    "synthetic": "#2563eb",
    "oracle": "#f59e0b",
    "noise_ceiling": "#dc2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-run CONFIG-treatment plot with paired bootstrap CIs/tests "
            "(CONFIG-only vs synthetic vs oracle + sampling noise ceiling)."
        )
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("reports/archetype_augmented_regression/results_itt_efficiency_both_granularity.csv"),
        help="Single-run evaluator results CSV used to infer best k for oracle/synthetic.",
    )
    parser.add_argument(
        "--learn-analysis-csv",
        type=Path,
        default=Path("benchmark/data/processed_data/df_analysis_learn.csv"),
    )
    parser.add_argument(
        "--val-analysis-csv",
        type=Path,
        default=Path("benchmark/data/processed_data/df_analysis_val.csv"),
    )
    parser.add_argument(
        "--learn-persona-jsonl",
        type=Path,
        default=Path("Persona/archetype_oracle_gpt51_learn.jsonl"),
    )
    parser.add_argument(
        "--val-persona-jsonl",
        type=Path,
        default=Path("Persona/archetype_oracle_gpt51_val.jsonl"),
    )
    parser.add_argument("--target", type=str, default="itt_efficiency")
    parser.add_argument("--model", type=str, default="ridge", choices=["ridge", "linear"])
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--style-ridge-alpha", type=float, default=1.0)
    parser.add_argument("--style-oof-folds", type=int, default=5)
    parser.add_argument("--group-col", type=str, default="CONFIG_treatmentName")
    parser.add_argument(
        "--exclude-config-cols",
        nargs="+",
        default=["CONFIG_configId", "CONFIG_treatmentName"],
    )
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/archetype_augmented_regression/figures/non_ood_single/ridge"),
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _ci_table_to_plot(method_ci_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = method_ci_df[method_ci_df["metric"] == metric].copy()
    sub["method"] = pd.Categorical(sub["method"], categories=METHOD_ORDER, ordered=True)
    return sub.sort_values("method")


def plot_method_ci(method_ci_df: pd.DataFrame, out_png: Path, dpi: int, model: str, n_boot: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    metrics = [
        ("r2", "R2 (test-mean denominator)"),
        ("r2_oos_train_mean", "R2 (train-mean denominator)"),
        ("rmse", "RMSE"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        sub = _ci_table_to_plot(method_ci_df, metric)
        x = np.arange(len(sub))
        colors = [METHOD_COLOR[m] for m in sub["method"]]
        labels = [METHOD_LABEL[m] for m in sub["method"]]
        vals = sub["point"].to_numpy(dtype=float)
        lo = sub["ci_low"].to_numpy(dtype=float)
        hi = sub["ci_high"].to_numpy(dtype=float)
        yerr = np.vstack([vals - lo, hi - vals])
        ax.bar(x, vals, yerr=yerr, capsize=3, color=colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(metric)
    fig.suptitle(
        f"Non-OOD CONFIG-treatment ({model}) with paired bootstrap 95% CI (B={n_boot})",
        y=1.03,
    )
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_delta_ci(delta_df: pd.DataFrame, out_png: Path, dpi: int, model: str, n_boot: int) -> None:
    focus_pairs = [
        ("oracle", "config_only"),
        ("synthetic", "config_only"),
        ("oracle", "synthetic"),
    ]
    metrics = ["r2", "r2_oos_train_mean", "rmse"]
    n_rows = len(metrics)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 9), constrained_layout=True)
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sub = delta_df[delta_df["metric"] == metric].copy()
        rows: list[pd.Series] = []
        labels: list[str] = []
        for a, b in focus_pairs:
            r = sub[(sub["model_a"] == a) & (sub["model_b"] == b)]
            if r.empty:
                continue
            rows.append(r.iloc[0])
            if metric == "rmse":
                labels.append(f"{a} better than {b} (RMSE drop)")
            else:
                labels.append(f"{a} - {b}")
        if not rows:
            continue
        use = pd.DataFrame(rows)
        y = np.arange(len(use))
        x = use["delta_point"].to_numpy(dtype=float)
        xerr = np.vstack(
            [
                x - use["ci_low"].to_numpy(dtype=float),
                use["ci_high"].to_numpy(dtype=float) - x,
            ]
        )
        ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=3, color="#1f2937")
        ax.axvline(0.0, color="#9ca3af", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(metric)
    fig.suptitle(
        f"Paired-bootstrap delta CIs ({model}, CONFIG treatment, B={n_boot})",
        y=1.02,
    )
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_k = infer_best_k_from_results(args.results_csv, model=args.model)
    pred_pack = fit_config_treatment_predictions(
        learn_analysis_csv=args.learn_analysis_csv,
        val_analysis_csv=args.val_analysis_csv,
        learn_persona_jsonl=args.learn_persona_jsonl,
        val_persona_jsonl=args.val_persona_jsonl,
        target_requested=args.target,
        model=args.model,
        ridge_alpha=args.ridge_alpha,
        style_ridge_alpha=args.style_ridge_alpha,
        style_oof_folds=args.style_oof_folds,
        best_k_oracle=best_k["oracle"],
        best_k_synthetic=best_k["synthetic"],
        group_col=args.group_col,
        exclude_config_cols=args.exclude_config_cols,
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_max=args.ngram_max,
    )

    method_ci_df, delta_df, _, _ = bootstrap_methods_by_groups(
        y_true=pred_pack["y_true"],
        pred_by_method=pred_pack["pred_by_method"],
        train_mean_target=pred_pack["train_mean_target"],
        sampling_terms=pred_pack["sampling_terms"],
        n_boot=args.n_boot,
        seed=args.seed,
        comparison_pairs=[
            ("oracle", "config_only"),
            ("synthetic", "config_only"),
            ("oracle", "synthetic"),
        ],
    )

    groups_df = pd.DataFrame(
        {
            "group": pred_pack["groups"],
            "y_true": pred_pack["y_true"],
            "pred_config_only": pred_pack["pred_by_method"]["config_only"],
            "pred_synthetic": pred_pack["pred_by_method"]["synthetic"],
            "pred_oracle": pred_pack["pred_by_method"]["oracle"],
            "sampling_term": pred_pack["sampling_terms"],
        }
    )

    out_methods = args.output_dir / f"{args.model}_non_ood_config_treatment_bootstrap_method_ci.csv"
    out_delta = args.output_dir / f"{args.model}_non_ood_config_treatment_bootstrap_pairwise_delta.csv"
    out_groups = args.output_dir / f"{args.model}_non_ood_config_treatment_group_predictions.csv"
    out_plot_methods = args.output_dir / f"{args.model}_non_ood_config_treatment_method_comparison_bootstrap_ci.png"
    out_plot_delta = args.output_dir / f"{args.model}_non_ood_config_treatment_pairwise_delta_bootstrap_ci.png"

    method_ci_df.to_csv(out_methods, index=False)
    delta_df.to_csv(out_delta, index=False)
    groups_df.to_csv(out_groups, index=False)
    plot_method_ci(method_ci_df, out_plot_methods, args.dpi, args.model, args.n_boot)
    plot_delta_ci(delta_df, out_plot_delta, args.dpi, args.model, args.n_boot)

    print(f"best_k_oracle={best_k['oracle']} best_k_synthetic={best_k['synthetic']}")
    print(f"Saved: {out_methods}")
    print(f"Saved: {out_delta}")
    print(f"Saved: {out_groups}")
    print(f"Saved: {out_plot_methods}")
    print(f"Saved: {out_plot_delta}")


if __name__ == "__main__":
    main()
