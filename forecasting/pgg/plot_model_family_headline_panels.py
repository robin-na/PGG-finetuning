from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FORECASTING_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = FORECASTING_ROOT / "results"
RUN_FAMILIES = {
    "gpt-5-mini": {
        "runs": [
            "baseline_gpt_5_mini",
            "demographic_only_row_resampled_seed_0_gpt_5_mini",
            "twin_sampled_seed_0_gpt_5_mini",
            "twin_sampled_unadjusted_seed_0_gpt_5_mini",
        ],
    },
    "gpt-5.1": {
        "runs": [
            "baseline_gpt_5_1",
            "demographic_only_row_resampled_seed_0_gpt_5_1",
            "twin_sampled_seed_0_gpt_5_1",
            "twin_sampled_unadjusted_seed_0_gpt_5_1",
        ],
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
    ("mean_round_normalized_efficiency", "Mean Normalized Efficiency"),
    ("mean_total_contribution_rate", "Mean Contribution Rate"),
]
NOISE_CEILING_CSV = RESULTS_ROOT / "model_comparison__noise_ceiling" / "noise_ceiling_summary.csv"


def _load_run_metric_table(run_name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_ROOT / f"{run_name}__vs_human_treatments" / "treatment_wasserstein_distance.csv")


def _load_family_table(family_name: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for run_name in RUN_FAMILIES[family_name]["runs"]:
        frame = _load_run_metric_table(run_name).copy()
        frame["run_name"] = run_name
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def _summarize_run_metric(comparison_df: pd.DataFrame, metric_name: str, run_name: str) -> tuple[float, float]:
    values = (
        comparison_df[(comparison_df["metric"] == metric_name) & (comparison_df["run_name"] == run_name)]["wasserstein_distance"]
        .dropna()
        .astype(float)
    )
    mean_value = float(values.mean())
    stderr = float(values.std(ddof=1) / np.sqrt(values.shape[0])) if values.shape[0] > 1 else 0.0
    return mean_value, stderr


def _row_ylim(metric_name: str, family_tables: dict[str, pd.DataFrame], noise_df: pd.DataFrame) -> float:
    upper = 0.0
    for family_name, comparison_df in family_tables.items():
        for run_name in RUN_FAMILIES[family_name]["runs"]:
            mean_value, stderr = _summarize_run_metric(comparison_df, metric_name, run_name)
            upper = max(upper, mean_value + stderr)
    noise_row = noise_df[(noise_df["score_family"] == "mean_wasserstein_distance") & (noise_df["metric"] == metric_name)].iloc[0]
    upper = max(upper, float(noise_row["noise_ceiling_p95"]))
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
    x = np.arange(len(runs) + 1)

    bars = []
    labels = []
    for idx, run_name in enumerate(runs):
        mean_value, stderr = _summarize_run_metric(comparison_df, metric_name, run_name)
        bar = ax.bar(
            x[idx],
            mean_value,
            width=0.72,
            color=RUN_COLORS[idx],
            edgecolor="black",
            linewidth=0.4,
            yerr=[[stderr], [stderr]],
            capsize=4,
            alpha=0.92,
            label=RUN_LABELS[run_name],
        )
        bars.append(bar[0])
        labels.append(RUN_LABELS[run_name])

    noise_row = noise_df[(noise_df["score_family"] == "mean_wasserstein_distance") & (noise_df["metric"] == metric_name)].iloc[0]
    noise_mean = float(noise_row["noise_ceiling_mean"])
    noise_low = max(noise_mean - float(noise_row["noise_ceiling_p05"]), 0.0)
    noise_high = max(float(noise_row["noise_ceiling_p95"]) - noise_mean, 0.0)
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
    labels.append("Human Ceiling")

    ax.set_ylim(0, ylim)
    ax.set_xticks(x)
    ax.set_xticklabels([RUN_LABELS[run_name] for run_name in runs] + ["Human Ceiling"], rotation=24, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    if show_ylabel:
        ax.set_ylabel(f"{metric_label}\nWasserstein Distance", fontsize=10)
    return bars, labels


def main() -> None:
    family_tables = {family_name: _load_family_table(family_name) for family_name in RUN_FAMILIES}
    noise_df = pd.read_csv(NOISE_CEILING_CSV)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), constrained_layout=False)
    fig.subplots_adjust(top=0.79, bottom=0.16, left=0.11, right=0.99, hspace=0.34, wspace=0.12)

    legend_handles = None
    legend_labels = None
    family_names = list(RUN_FAMILIES.keys())

    for row_idx, (metric_name, metric_label) in enumerate(METRICS):
        ylim = _row_ylim(metric_name, family_tables, noise_df)
        for col_idx, family_name in enumerate(family_names):
            handles, labels = _plot_panel(
                axes[row_idx, col_idx],
                family_name=family_name,
                metric_name=metric_name,
                metric_label=metric_label,
                comparison_df=family_tables[family_name],
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

    fig.suptitle("Public Goods Game", fontsize=14, y=0.985)
    output_path = RESULTS_ROOT / "headline_model_family_panels.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
