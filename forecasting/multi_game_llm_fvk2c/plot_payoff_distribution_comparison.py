from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import write_csv, write_json


RUN_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
PRETTY_METRIC_LABELS = {
    "sender_expected_payoff": "TG Sender Expected Payoff",
    "receiver_expected_payoff": "TG Receiver Expected Payoff",
    "pd_expected_payoff": "PD Expected Payoff",
    "sh_expected_payoff": "SH Expected Payoff",
    "c_expected_payoff": "Coordination Expected Payoff",
    "proposer_expected_payoff": "UG Proposer Expected Payoff",
    "responder_expected_payoff": "UG Responder Expected Payoff",
    "mean_expected_payoff": "Mean Expected Payoff",
}
HEADLINE_METRICS = [
    "sender_expected_payoff",
    "receiver_expected_payoff",
    "pd_expected_payoff",
    "sh_expected_payoff",
    "c_expected_payoff",
    "proposer_expected_payoff",
    "responder_expected_payoff",
]
FULL_METRICS = HEADLINE_METRICS + ["mean_expected_payoff"]


def _pretty_run_label(run_name: str) -> str:
    mapping = {
        "baseline_gpt_5_mini": "Baseline",
        "demographic_only_row_resampled_seed_0_gpt_5_mini": "Demographic Only",
        "twin_sampled_seed_0_gpt_5_mini": "Twin-Sampled",
        "twin_sampled_unadjusted_seed_0_gpt_5_mini": "Twin Unadjusted",
        "baseline_gpt_5_1": "Baseline",
        "demographic_only_row_resampled_seed_0_gpt_5_1": "Demographic Only",
        "twin_sampled_seed_0_gpt_5_1": "Twin-Sampled",
        "twin_sampled_unadjusted_seed_0_gpt_5_1": "Twin Unadjusted",
    }
    return mapping.get(run_name, run_name.replace("_", " "))


def _load_run_summary(forecasting_root: Path, run_name: str) -> pd.DataFrame:
    path = forecasting_root / "results" / f"{run_name}__payoff_analysis" / "payoff_primary_summary.csv"
    frame = pd.read_csv(path)
    frame["run_name"] = run_name
    frame["run_label"] = _pretty_run_label(run_name)
    return frame


def _load_relative_error_summary(forecasting_root: Path, run_name: str) -> pd.DataFrame:
    path = forecasting_root / "results" / f"{run_name}__payoff_analysis" / "relative_payoff_difference_error_summary.csv"
    frame = pd.read_csv(path)
    frame["run_name"] = run_name
    frame["run_label"] = _pretty_run_label(run_name)
    return frame


def _plot_metric_grid(
    *,
    run_df: pd.DataFrame,
    metric_names: list[str],
    output_path: Path,
) -> None:
    n_metrics = len(metric_names)
    ncols = 4
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.3 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for axis, metric_name in zip(axes_flat, metric_names):
        metric_frame = run_df[run_df["metric"] == metric_name].copy()
        values = metric_frame["mean_value"].tolist()
        errors = metric_frame["stderr"].fillna(0.0).tolist()
        labels = metric_frame["run_label"].tolist()

        x_positions = list(range(len(labels)))
        colors = [RUN_COLORS[idx % len(RUN_COLORS)] for idx in range(len(labels))]
        axis.bar(
            x_positions,
            values,
            yerr=errors,
            color=colors,
            alpha=0.92,
            capsize=4,
            edgecolor="black",
            linewidth=0.4,
        )
        axis.set_title(PRETTY_METRIC_LABELS.get(metric_name, metric_name), fontsize=11)
        axis.set_ylabel("Wasserstein Distance", fontsize=10)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        axis.grid(axis="y", alpha=0.25, linestyle=":")
        axis.set_axisbelow(True)

    for axis in axes_flat[n_metrics:]:
        axis.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot multi-game expected-payoff distribution alignment across multiple runs."
    )
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--run-names", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        suffix = "__vs__".join(args.run_names) + "__payoff_plots"
        args.output_dir = args.forecasting_root / "results" / suffix

    run_frames = [_load_run_summary(args.forecasting_root, run_name) for run_name in args.run_names]
    combined = pd.concat(run_frames, ignore_index=True)
    combined["metric_order"] = combined["metric"].map({name: idx for idx, name in enumerate(FULL_METRICS)}).fillna(99)
    combined["run_order"] = combined["run_name"].map({name: idx for idx, name in enumerate(args.run_names)}).fillna(99)
    combined = combined.sort_values(["metric_order", "run_order"], kind="stable").drop(columns=["metric_order", "run_order"])

    relative_frames = [_load_relative_error_summary(args.forecasting_root, run_name) for run_name in args.run_names]
    relative_error = pd.concat(relative_frames, ignore_index=True)

    _plot_metric_grid(
        run_df=combined,
        metric_names=HEADLINE_METRICS,
        output_path=args.output_dir / "headline_payoff_distribution_comparison.png",
    )
    _plot_metric_grid(
        run_df=combined,
        metric_names=FULL_METRICS,
        output_path=args.output_dir / "all_payoff_distribution_metrics.png",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "run_payoff_distribution_comparison.csv", combined)
    write_csv(args.output_dir / "run_relative_payoff_error_summary.csv", relative_error)
    write_json(
        args.output_dir / "manifest.json",
        {
            "run_names": args.run_names,
            "headline_metrics": HEADLINE_METRICS,
            "full_metrics": FULL_METRICS,
            "output_dir": str(args.output_dir),
        },
    )


if __name__ == "__main__":
    main()
