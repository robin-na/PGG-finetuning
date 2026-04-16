from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RESULTS_ROOT = Path(__file__).resolve().parent / "results"
RUN_FAMILIES = {
    "gpt-5-mini": [
        "baseline_gpt_5_mini_pgg_aligned_v3",
        "twin_sampled_unadjusted_seed_0_gpt_5_mini_pgg_aligned_v3",
        "twin_sampled_unadjusted_seed_0_gpt_5_mini_bargain_card_v1",
    ],
    "gpt-5.1": [
        "baseline_gpt_5_1_pgg_aligned_v3",
        "twin_sampled_unadjusted_seed_0_gpt_5_1_pgg_aligned_v3",
        "twin_sampled_unadjusted_seed_0_gpt_5_1_bargain_card_v1",
    ],
}
RUN_LABELS = {
    "baseline_gpt_5_mini_pgg_aligned_v3": "Baseline",
    "twin_sampled_unadjusted_seed_0_gpt_5_mini_pgg_aligned_v3": "Twin Unadjusted",
    "twin_sampled_unadjusted_seed_0_gpt_5_mini_bargain_card_v1": "Twin Bargain Card",
    "baseline_gpt_5_1_pgg_aligned_v3": "Baseline",
    "twin_sampled_unadjusted_seed_0_gpt_5_1_pgg_aligned_v3": "Twin Unadjusted",
    "twin_sampled_unadjusted_seed_0_gpt_5_1_bargain_card_v1": "Twin Bargain Card",
}
CELL_LABELS = {
    "CHIP2__GAME_1": "Chip2 / Game 1",
    "CHIP2__GAME_2_ALT_PROFILE": "Chip2 / Alt",
    "CHIP3__GAME_1": "Chip3 / Game 1",
    "CHIP3__GAME_2_ALT_PROFILE": "Chip3 / Alt",
    "CHIP4__GAME_1": "Chip4 / Game 1",
    "CHIP4__GAME_2_ALT_PROFILE": "Chip4 / Alt",
}
METRICS = [
    ("final_surplus_ratio", "Final Surplus Ratio"),
    ("trade_ratio", "Trade Ratio"),
    ("acceptance_rate", "Acceptance Rate"),
]


def _load_cell_metric_frame(run_name: str) -> pd.DataFrame:
    path = RESULTS_ROOT / f"{run_name}__vs_human_treatments" / "distribution_distance.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["metric"].isin([metric for metric, _ in METRICS])].copy()
    frame["run_name"] = run_name
    frame["run_label"] = frame["run_name"].map(RUN_LABELS)
    frame["cell_label"] = frame["treatment_name"].map(CELL_LABELS)
    return frame


def main() -> None:
    all_frames = [
        _load_cell_metric_frame(run_name)
        for run_names in RUN_FAMILIES.values()
        for run_name in run_names
    ]
    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(RESULTS_ROOT / "cell_level_bargain_card_heatmaps_summary.csv", index=False)

    fig, axes = plt.subplots(len(METRICS), len(RUN_FAMILIES), figsize=(11.5, 9.4), constrained_layout=False)
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.13, right=0.98, hspace=0.36, wspace=0.20)

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
            metric_frame["cell_label"] = pd.Categorical(
                metric_frame["cell_label"],
                categories=[CELL_LABELS[cell] for cell in CELL_LABELS],
                ordered=True,
            )
            pivot = metric_frame.pivot(index="cell_label", columns="run_label", values="score").sort_index()
            ax = axes[row_idx, col_idx]
            sns.heatmap(
                pivot,
                ax=ax,
                cmap="YlOrRd_r",
                annot=True,
                fmt=".3f",
                cbar=(col_idx == len(family_names) - 1),
                linewidths=0.5,
                linecolor="white",
                annot_kws={"fontsize": 8},
            )
            if row_idx == 0:
                ax.set_title(family_name, fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=10)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=25, labelsize=9)
            ax.tick_params(axis="y", rotation=0, labelsize=9)

    fig.suptitle("Chip Bargaining Cell-Level Heterogeneity", fontsize=14, y=0.98)
    fig.savefig(RESULTS_ROOT / "cell_level_bargain_card_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
