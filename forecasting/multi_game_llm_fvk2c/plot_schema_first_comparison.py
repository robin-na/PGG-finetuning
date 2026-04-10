from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUN_ORDER = [
    "baseline_gpt_5_mini",
    "demographic_only_row_resampled_seed_0_gpt_5_mini",
    "twin_sampled_seed_0_gpt_5_mini",
    "twin_sampled_unadjusted_seed_0_gpt_5_mini",
]
RUN_LABELS = {
    "baseline_gpt_5_mini": "Baseline",
    "demographic_only_row_resampled_seed_0_gpt_5_mini": "Demographic Only",
    "twin_sampled_seed_0_gpt_5_mini": "Twin-Sampled",
    "twin_sampled_unadjusted_seed_0_gpt_5_mini": "Twin Unadjusted",
}
RUN_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
NOISE_CEILING_COLOR = "#2CA02C"

METRICS = [
    ("mean_delegation_wd", "Mean Delegation WD", "Distance"),
    ("mean_role_state_tv", "Mean Role State TV", "Distance"),
    ("mean_numeric_direct_value_wd", "Mean Numeric Direct Value WD", "Distance"),
]


def main() -> None:
    forecasting_root = Path(__file__).resolve().parent
    plot_dir = (
        forecasting_root
        / "results"
        / "baseline_gpt_5_mini__vs__demographic_only_row_resampled_seed_0_gpt_5_mini__vs__twin_sampled_seed_0_gpt_5_mini__vs__twin_sampled_unadjusted_seed_0_gpt_5_mini__plots"
    )
    comparison_df = pd.read_csv(plot_dir / "run_primary_distribution_comparison.csv")
    noise_df = pd.read_csv(plot_dir / "noise_ceiling_primary_distribution_summary.csv")
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.1 * len(METRICS), 4.4), squeeze=False)
    axes_flat = axes.flatten()

    for axis, (metric_name, title, ylabel) in zip(axes_flat, METRICS):
        metric_frame = (
            comparison_df[comparison_df["metric"] == metric_name]
            .copy()
            .assign(run_order=lambda df: df["run_name"].map({name: idx for idx, name in enumerate(RUN_ORDER)}))
            .sort_values("run_order")
        )

        llm_x = list(range(len(RUN_ORDER)))
        llm_vals = [float(metric_frame.loc[metric_frame["run_name"] == run_name, "mean_value"].iloc[0]) for run_name in RUN_ORDER]
        llm_errs = [float(metric_frame.loc[metric_frame["run_name"] == run_name, "stderr"].iloc[0]) for run_name in RUN_ORDER]
        llm_labels = [RUN_LABELS[run_name] for run_name in RUN_ORDER]

        noise_row = noise_df[noise_df["metric"] == metric_name].iloc[0]
        noise_x = len(RUN_ORDER)
        noise_mean = float(noise_row["bootstrap_mean"])
        noise_err_low = noise_mean - float(noise_row["bootstrap_p05"])
        noise_err_high = float(noise_row["bootstrap_p95"]) - noise_mean

        axis.bar(
            llm_x,
            llm_vals,
            yerr=llm_errs,
            color=RUN_COLORS,
            alpha=0.92,
            capsize=4,
            edgecolor="black",
            linewidth=0.4,
        )
        axis.bar(
            [noise_x],
            [noise_mean],
            yerr=[[noise_err_low], [noise_err_high]],
            color=NOISE_CEILING_COLOR,
            alpha=0.92,
            capsize=4,
            edgecolor="black",
            linewidth=0.4,
        )
        axis.set_title(title, fontsize=11)
        axis.set_ylabel(ylabel, fontsize=10)
        axis.set_xticks(llm_x + [noise_x])
        axis.set_xticklabels(llm_labels + ["Human Ceiling"], rotation=20, ha="right", fontsize=9)
        axis.grid(axis="y", alpha=0.25, linestyle=":")
        axis.set_axisbelow(True)

        ymax = max(llm_vals + [noise_mean])
        axis.set_ylim(0, ymax * 1.22 if ymax > 0 else 1)

    fig.tight_layout()
    output_path = plot_dir / "schema_first_primary_distribution_comparison.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
