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
from archetype_augmented_regression.config_treatment_eval import fit_config_treatment_predictions


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
            "OOD split-wise CONFIG-treatment bootstrap inference and granular plots "
            "(error bars and pairwise deltas)."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "outputs/archetype_augmented_regression/ood_wave_anchored/runs/run_full_oos_noise_itt_efficiency"
        ),
    )
    parser.add_argument(
        "--ood-root",
        type=Path,
        default=Path("benchmark/data_ood_splits_wave_anchored"),
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
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to reports/archetype_augmented_regression/figures/<run_id>/<model>/config_treatment_bootstrap",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def _best_k_for_split(split_results_csv: Path, model: str) -> dict[str, int]:
    df = pd.read_csv(split_results_csv)
    out: dict[str, int] = {}
    for style in ("oracle", "synthetic"):
        sub = df[
            (df["eval_granularity"] == "config_treatment")
            & (df["style_source"] == style)
            & (df["model"] == model)
        ].copy()
        if sub.empty:
            raise ValueError(f"No config_treatment rows in {split_results_csv} for style={style} model={model}")
        row = sub.sort_values(["delta_rmse", "delta_mae"], ascending=[True, True]).iloc[0]
        out[style] = int(row["k_clusters"])
    return out


def _plot_split_metric_errorbars(method_df: pd.DataFrame, out_png: Path, dpi: int, model: str) -> None:
    splits = sorted(method_df["split"].unique())
    metrics = ["r2", "r2_oos_train_mean", "rmse"]
    ylabels = {
        "r2": "R2 (test-mean denominator)",
        "r2_oos_train_mean": "R2 (train-mean denominator)",
        "rmse": "RMSE",
    }
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        base_x = np.arange(len(splits), dtype=float)
        width = 0.2
        for i, method in enumerate(METHOD_ORDER):
            sub = method_df[(method_df["method"] == method) & (method_df["metric"] == metric)].copy()
            sub = sub.set_index("split").reindex(splits)
            x = base_x + (i - 1.5) * width
            point = sub["point"].to_numpy(dtype=float)
            lo = sub["ci_low"].to_numpy(dtype=float)
            hi = sub["ci_high"].to_numpy(dtype=float)
            yerr = np.vstack([point - lo, hi - point])
            ax.errorbar(
                x,
                point,
                yerr=yerr,
                fmt="o",
                markersize=4,
                capsize=2,
                color=METHOD_COLOR[method],
                label=METHOD_LABEL[method],
                alpha=0.9,
            )
        ax.set_xticks(base_x)
        ax.set_xticklabels(splits, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabels[metric])
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"OOD split-wise bootstrap CIs ({model}, CONFIG treatment)", y=1.05)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_split_pairwise_deltas(delta_df: pd.DataFrame, out_png: Path, dpi: int, model: str) -> None:
    pairs = [("oracle", "config_only"), ("synthetic", "config_only")]
    metrics = ["r2", "r2_oos_train_mean", "rmse"]
    ylabels = {
        "r2": "Delta R2 (A - B)",
        "r2_oos_train_mean": "Delta R2 train-mean (A - B)",
        "rmse": "Delta RMSE drop (B - A)",
    }
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        sub = delta_df[delta_df["metric"] == metric].copy()
        splits = sorted(sub["split"].unique())
        base_x = np.arange(len(splits), dtype=float)
        width = 0.22
        for i, (a, b) in enumerate(pairs):
            p = sub[(sub["model_a"] == a) & (sub["model_b"] == b)].copy()
            p = p.set_index("split").reindex(splits)
            x = base_x + (i - 0.5) * width
            point = p["delta_point"].to_numpy(dtype=float)
            lo = p["ci_low"].to_numpy(dtype=float)
            hi = p["ci_high"].to_numpy(dtype=float)
            yerr = np.vstack([point - lo, hi - point])
            label = f"{a} vs {b}"
            color = METHOD_COLOR["oracle"] if a == "oracle" else METHOD_COLOR["synthetic"]
            ax.errorbar(
                x,
                point,
                yerr=yerr,
                fmt="o",
                markersize=4,
                capsize=2,
                color=color,
                label=label,
                alpha=0.95,
            )
        ax.axhline(0.0, color="#9ca3af", linestyle="--", linewidth=1)
        ax.set_xticks(base_x)
        ax.set_xticklabels(splits, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabels[metric])
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"OOD split-wise paired-bootstrap deltas ({model}, CONFIG treatment)", y=1.05)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_id = args.run_root.name
    output_dir = args.output_dir or Path(
        f"reports/archetype_augmented_regression/figures/{run_id}/{args.model}/config_treatment_bootstrap"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    split_dirs = sorted([p for p in args.run_root.glob("*/*") if p.is_dir()])
    if not split_dirs:
        raise SystemExit(f"No split directories under run_root: {args.run_root}")

    method_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    group_rows: list[pd.DataFrame] = []

    for i, split_dir in enumerate(split_dirs, 1):
        factor = split_dir.parent.name
        direction = split_dir.name
        split = f"{factor}/{direction}"
        print(f"[{i}/{len(split_dirs)}] {split}")

        split_results_csv = _require(split_dir / "results.csv")
        best_k = _best_k_for_split(split_results_csv, model=args.model)

        learn_csv = _require(args.ood_root / factor / direction / "processed_data" / "df_analysis_learn.csv")
        val_csv = _require(args.ood_root / factor / direction / "processed_data" / "df_analysis_val.csv")

        pred_pack = fit_config_treatment_predictions(
            learn_analysis_csv=learn_csv,
            val_analysis_csv=val_csv,
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
        method_ci_df["split"] = split
        method_ci_df["factor"] = factor
        method_ci_df["direction"] = direction
        method_ci_df["best_k_oracle"] = int(best_k["oracle"])
        method_ci_df["best_k_synthetic"] = int(best_k["synthetic"])
        delta_df["split"] = split
        delta_df["factor"] = factor
        delta_df["direction"] = direction
        delta_df["best_k_oracle"] = int(best_k["oracle"])
        delta_df["best_k_synthetic"] = int(best_k["synthetic"])
        method_rows.append(method_ci_df)
        delta_rows.append(delta_df)

        groups_df = pd.DataFrame(
            {
                "split": split,
                "factor": factor,
                "direction": direction,
                "group": pred_pack["groups"],
                "y_true": pred_pack["y_true"],
                "pred_config_only": pred_pack["pred_by_method"]["config_only"],
                "pred_synthetic": pred_pack["pred_by_method"]["synthetic"],
                "pred_oracle": pred_pack["pred_by_method"]["oracle"],
                "sampling_term": pred_pack["sampling_terms"],
            }
        )
        group_rows.append(groups_df)

    method_all = pd.concat(method_rows, ignore_index=True)
    delta_all = pd.concat(delta_rows, ignore_index=True)
    groups_all = pd.concat(group_rows, ignore_index=True)

    out_methods = output_dir / f"{args.model}_ood_config_treatment_split_bootstrap_method_ci.csv"
    out_deltas = output_dir / f"{args.model}_ood_config_treatment_split_bootstrap_pairwise_delta.csv"
    out_groups = output_dir / f"{args.model}_ood_config_treatment_split_group_predictions.csv"
    out_plot_methods = output_dir / f"{args.model}_ood_config_treatment_split_method_errorbars.png"
    out_plot_deltas = output_dir / f"{args.model}_ood_config_treatment_split_pairwise_delta_errorbars.png"

    method_all.to_csv(out_methods, index=False)
    delta_all.to_csv(out_deltas, index=False)
    groups_all.to_csv(out_groups, index=False)
    _plot_split_metric_errorbars(method_all, out_plot_methods, args.dpi, args.model)
    _plot_split_pairwise_deltas(delta_all, out_plot_deltas, args.dpi, args.model)

    print(f"Saved: {out_methods}")
    print(f"Saved: {out_deltas}")
    print(f"Saved: {out_groups}")
    print(f"Saved: {out_plot_methods}")
    print(f"Saved: {out_plot_deltas}")


if __name__ == "__main__":
    main()
