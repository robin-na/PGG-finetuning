from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FORECASTING_ROOT = Path(__file__).resolve().parent
RUN_FAMILIES = {
    "gpt-5-mini": {
        "runs": [
            "baseline_gpt_5_mini",
            "demographic_only_row_resampled_seed_0_gpt_5_mini",
            "twin_sampled_seed_0_gpt_5_mini",
            "twin_sampled_unadjusted_seed_0_gpt_5_mini",
        ],
        "comparison_csv": FORECASTING_ROOT
        / "results"
        / "baseline_gpt_5_mini__vs__demographic_only_row_resampled_seed_0_gpt_5_mini__vs__twin_sampled_seed_0_gpt_5_mini__vs__twin_sampled_unadjusted_seed_0_gpt_5_mini__plots"
        / "run_primary_distribution_comparison.csv",
        "noise_csv": FORECASTING_ROOT / "results" / "baseline_gpt_5_mini__noise_ceiling" / "primary_noise_ceiling_summary.csv",
    },
    "gpt-5.1": {
        "runs": [
            "baseline_gpt_5_1",
            "demographic_only_row_resampled_seed_0_gpt_5_1",
            "twin_sampled_seed_0_gpt_5_1",
            "twin_sampled_unadjusted_seed_0_gpt_5_1",
        ],
        "comparison_csv": FORECASTING_ROOT
        / "results"
        / "baseline_gpt_5_1__vs__demographic_only_row_resampled_seed_0_gpt_5_1__vs__twin_sampled_seed_0_gpt_5_1__vs__twin_sampled_unadjusted_seed_0_gpt_5_1__plots"
        / "run_primary_distribution_comparison.csv",
        "noise_csv": FORECASTING_ROOT / "results" / "baseline_gpt_5_1__noise_ceiling" / "primary_noise_ceiling_summary.csv",
    },
}
RUN_LABELS = {
    "baseline_gpt_5_mini": "Baseline",
    "demographic_only_row_resampled_seed_0_gpt_5_mini": "Demographic Only",
    "twin_sampled_seed_0_gpt_5_mini": "Twin-Sampled",
    "twin_sampled_unadjusted_seed_0_gpt_5_mini": "Twin Unadjusted",
    "baseline_gpt_5_1": "Baseline",
    "demographic_only_row_resampled_seed_0_gpt_5_1": "Demographic Only",
    "twin_sampled_seed_0_gpt_5_1": "Twin-Sampled",
    "twin_sampled_unadjusted_seed_0_gpt_5_1": "Twin Unadjusted",
}
RUN_COLORS = ["#4C78A8", "#9ECAE1", "#D62728", "#F4A3A3"]
NOISE_CEILING_COLOR = "#8C8C8C"
METRICS = [
    ("mean_delegation_wd", "Mean Delegation"),
    ("prosociality_index_wd", "Trust Proxy"),
]


def _load_family_tables(family_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = RUN_FAMILIES[family_name]
    comparison_df = pd.read_csv(family["comparison_csv"])
    noise_df = pd.read_csv(family["noise_csv"])
    return comparison_df, noise_df


def _row_ylim(metric_name: str, family_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> float:
    upper = 0.0
    for family_name, (comparison_df, noise_df) in family_tables.items():
        runs = RUN_FAMILIES[family_name]["runs"]
        metric_df = comparison_df[comparison_df["metric"] == metric_name].set_index("run_name")
        for run_name in runs:
            row = metric_df.loc[run_name]
            mean_value = float(row["mean_value"])
            stderr = float(row["stderr"]) if pd.notna(row["stderr"]) else 0.0
            upper = max(upper, mean_value + stderr)
        noise_row = noise_df[noise_df["metric"] == metric_name].iloc[0]
        upper = max(upper, float(noise_row["bootstrap_p95"]))
    return upper * 1.15 if upper > 0 else 1.0


def _plot_panel(
    ax: plt.Axes,
    *,
    family_name: str,
    metric_name: str,
    metric_label: str,
    comparison_df: pd.DataFrame,
    noise_df: pd.DataFrame,
    ylim: float,
    show_ylabel: bool,
) -> tuple[list, list]:
    runs = RUN_FAMILIES[family_name]["runs"]
    metric_df = comparison_df[comparison_df["metric"] == metric_name].set_index("run_name")
    noise_row = noise_df[noise_df["metric"] == metric_name].iloc[0]

    x = np.arange(len(runs) + 1)
    values = []
    errors = []
    for run_name in runs:
        row = metric_df.loc[run_name]
        values.append(float(row["mean_value"]))
        errors.append(float(row["stderr"]) if pd.notna(row["stderr"]) else 0.0)

    noise_mean = float(noise_row["bootstrap_mean"])
    noise_low = max(noise_mean - float(noise_row["bootstrap_p05"]), 0.0)
    noise_high = max(float(noise_row["bootstrap_p95"]) - noise_mean, 0.0)

    bars = []
    for idx, (value, err) in enumerate(zip(values, errors)):
        bar = ax.bar(
            x[idx],
            value,
            width=0.72,
            color=RUN_COLORS[idx],
            edgecolor="black",
            linewidth=0.4,
            yerr=[[err], [err]],
            capsize=4,
            alpha=0.92,
            label=RUN_LABELS[runs[idx]],
        )
        bars.append(bar[0])
    noise_bar = ax.bar(
        x[-1],
        noise_mean,
        width=0.72,
        color=NOISE_CEILING_COLOR,
        edgecolor="black",
        linewidth=0.4,
        yerr=[[noise_low], [noise_high]],
        capsize=4,
        alpha=0.92,
        label="Human Ceiling",
    )
    bars.append(noise_bar[0])

    ax.set_ylim(0, ylim)
    ax.set_xticks(x)
    ax.set_xticklabels([RUN_LABELS[run_name] for run_name in runs] + ["Human Ceiling"], rotation=24, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    if show_ylabel:
        ax.set_ylabel(f"{metric_label}\nWasserstein Distance", fontsize=10)
    return bars, [RUN_LABELS[run_name] for run_name in runs] + ["Human Ceiling"]


def main() -> None:
    family_tables = {family_name: _load_family_tables(family_name) for family_name in RUN_FAMILIES}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), constrained_layout=False)
    fig.subplots_adjust(top=0.79, bottom=0.16, left=0.11, right=0.99, hspace=0.34, wspace=0.12)

    legend_handles = None
    legend_labels = None
    family_names = list(RUN_FAMILIES.keys())

    for row_idx, (metric_name, metric_label) in enumerate(METRICS):
        ylim = _row_ylim(metric_name, family_tables)
        for col_idx, family_name in enumerate(family_names):
            comparison_df, noise_df = family_tables[family_name]
            handles, labels = _plot_panel(
                axes[row_idx, col_idx],
                family_name=family_name,
                metric_name=metric_name,
                metric_label=metric_label,
                comparison_df=comparison_df,
                noise_df=noise_df,
                ylim=ylim,
                show_ylabel=(col_idx == 0),
            )
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(family_name, fontsize=12)
            if legend_handles is None:
                legend_handles, legend_labels = handles, labels

    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=5, frameon=False)

    fig.suptitle("Multi-Game LLM Delegation", fontsize=14, y=0.985)
    output_path = FORECASTING_ROOT / "results" / "headline_model_family_panels.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
