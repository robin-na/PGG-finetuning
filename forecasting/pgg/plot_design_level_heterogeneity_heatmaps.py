from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FORECASTING_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = FORECASTING_ROOT / "results"
RUN_FAMILIES = {
    "gpt-5-mini": [
        "baseline_gpt_5_mini",
        "demographic_only_row_resampled_seed_0_gpt_5_mini",
        "twin_sampled_seed_0_gpt_5_mini",
        "twin_sampled_unadjusted_seed_0_gpt_5_mini",
    ],
    "gpt-5.1": [
        "baseline_gpt_5_1",
        "demographic_only_row_resampled_seed_0_gpt_5_1",
        "twin_sampled_seed_0_gpt_5_1",
        "twin_sampled_unadjusted_seed_0_gpt_5_1",
    ],
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
METRICS = [
    ("mean_round_normalized_efficiency", "Mean Normalized Efficiency"),
    ("mean_total_contribution_rate", "Mean Contribution Rate"),
]


def _treatment_sort_key(treatment_name: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"VALIDATION_(\d+)_([A-Z])", treatment_name)
    if match is None:
        return (10**9, 10**9, treatment_name)
    treatment_id = int(match.group(1))
    condition = match.group(2)
    condition_rank = {"C": 0, "T": 1}.get(condition, 99)
    return (treatment_id, condition_rank, treatment_name)


def _display_treatment_name(treatment_name: str) -> str:
    match = re.fullmatch(r"VALIDATION_(\d+)_([A-Z])", treatment_name)
    if match is None:
        return treatment_name
    return f"{match.group(1)}-{match.group(2)}"


def _load_run_metric_frame(run_name: str) -> pd.DataFrame:
    path = RESULTS_ROOT / f"{run_name}__vs_human_treatments" / "treatment_wasserstein_distance.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["metric"].isin([metric_name for metric_name, _ in METRICS])].copy()
    frame["run_name"] = run_name
    frame["run_label"] = frame["run_name"].map(RUN_LABELS)
    frame["treatment_label"] = frame["treatment_name"].map(_display_treatment_name)
    frame["score"] = frame["wasserstein_distance"].astype(float)
    return frame


def main() -> None:
    all_frames = [
        _load_run_metric_frame(run_name)
        for run_names in RUN_FAMILIES.values()
        for run_name in run_names
    ]
    combined = pd.concat(all_frames, ignore_index=True)
    combined["family_name"] = combined["run_name"].map(
        {
            run_name: family_name
            for family_name, run_names in RUN_FAMILIES.items()
            for run_name in run_names
        }
    )
    combined.to_csv(RESULTS_ROOT / "design_level_heterogeneity_heatmaps_summary.csv", index=False)

    treatment_order = [
        _display_treatment_name(treatment_name)
        for treatment_name in sorted(combined["treatment_name"].unique(), key=_treatment_sort_key)
    ]

    metric_ranges = {
        metric_name: (
            float(combined.loc[combined["metric"] == metric_name, "score"].min()),
            float(combined.loc[combined["metric"] == metric_name, "score"].max()),
        )
        for metric_name, _ in METRICS
    }

    fig, axes = plt.subplots(len(METRICS), len(RUN_FAMILIES), figsize=(12.4, 18.5), constrained_layout=False)
    fig.subplots_adjust(top=0.93, bottom=0.04, left=0.18, right=0.96, hspace=0.12, wspace=0.18)

    family_names = list(RUN_FAMILIES.keys())
    for col_idx, family_name in enumerate(family_names):
        family_runs = RUN_FAMILIES[family_name]
        family_frame = combined[combined["run_name"].isin(family_runs)].copy()
        for row_idx, (metric_name, metric_label) in enumerate(METRICS):
            metric_frame = family_frame[family_frame["metric"] == metric_name].copy()
            metric_frame["run_label"] = pd.Categorical(
                metric_frame["run_label"],
                categories=[RUN_LABELS[run_name] for run_name in family_runs],
                ordered=True,
            )
            metric_frame["treatment_label"] = pd.Categorical(
                metric_frame["treatment_label"],
                categories=treatment_order,
                ordered=True,
            )
            pivot = metric_frame.pivot(index="treatment_label", columns="run_label", values="score")
            vmin, vmax = metric_ranges[metric_name]

            ax = axes[row_idx, col_idx]
            sns.heatmap(
                pivot,
                ax=ax,
                cmap="YlOrRd_r",
                vmin=vmin,
                vmax=vmax,
                cbar=(col_idx == len(family_names) - 1),
                linewidths=0.3,
                linecolor="white",
            )
            if row_idx == 0:
                ax.set_title(family_name, fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=10)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=25, labelsize=9)
            ax.tick_params(axis="y", rotation=0, labelsize=7)

    fig.suptitle("Public Goods Game Design-Level Heterogeneity", fontsize=14, y=0.975)
    fig.savefig(RESULTS_ROOT / "design_level_heterogeneity_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
