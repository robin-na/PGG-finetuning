from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE_ORDER = ["persistence", "ewma", "within_game_ar"]
BASELINE_LABELS = {
    "persistence": "Persistence",
    "ewma": "EWMA",
    "within_game_ar": "Within-game AR",
}
BASELINE_COLORS = {
    "persistence": "#0b6e4f",
    "ewma": "#c84c09",
    "within_game_ar": "#2f4b7c",
}
BASELINE_MARKERS = {
    "persistence": "o",
    "ewma": "s",
    "within_game_ar": "^",
}

PLOT_SPECS = [
    ("contribution_rate_mae", "Contribution Rate MAE", "Lower is better"),
    ("total_contribution_rate_mae", "Total Contribution Rate MAE", "Lower is better"),
    ("round_normalized_efficiency_mae", "Round Normalized Efficiency MAE", "Lower is better"),
    ("future_normalized_efficiency_abs_error", "Future Normalized Efficiency Error", "Lower is better"),
    ("punish_target_f1", "Punish Target F1", "Higher is better"),
    ("reward_target_f1", "Reward Target F1", "Higher is better"),
]


def _summarize_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["baseline", "k"], as_index=False)
        .agg(
            num_games=("game_id", "nunique"),
            contribution_rate_mae_mean=("contribution_rate_mae", "mean"),
            contribution_rate_mae_std=("contribution_rate_mae", "std"),
            total_contribution_rate_mae_mean=("total_contribution_rate_mae", "mean"),
            total_contribution_rate_mae_std=("total_contribution_rate_mae", "std"),
            round_normalized_efficiency_mae_mean=("round_normalized_efficiency_mae", "mean"),
            round_normalized_efficiency_mae_std=("round_normalized_efficiency_mae", "std"),
            future_normalized_efficiency_abs_error_mean=("future_normalized_efficiency_abs_error", "mean"),
            future_normalized_efficiency_abs_error_std=("future_normalized_efficiency_abs_error", "std"),
            punish_target_f1_mean=("punish_target_f1", "mean"),
            punish_target_f1_std=("punish_target_f1", "std"),
            reward_target_f1_mean=("reward_target_f1", "mean"),
            reward_target_f1_std=("reward_target_f1", "std"),
        )
        .sort_values(["baseline", "k"])
        .reset_index(drop=True)
    )

    for metric, _, _ in PLOT_SPECS:
        std_col = f"{metric}_std"
        se_col = f"{metric}_se"
        grouped[std_col] = grouped[std_col].fillna(0.0)
        grouped[se_col] = grouped[std_col] / np.sqrt(grouped["num_games"].clip(lower=1))

    return grouped


def _plot_metric(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str, subtitle: str) -> None:
    mean_col = f"{metric}_mean"
    se_col = f"{metric}_se"
    for baseline in BASELINE_ORDER:
        baseline_df = df[df["baseline"] == baseline].sort_values("k")
        if baseline_df.empty:
            continue
        ax.errorbar(
            baseline_df["k"],
            baseline_df[mean_col],
            yerr=baseline_df[se_col],
            color=BASELINE_COLORS[baseline],
            marker=BASELINE_MARKERS[baseline],
            linewidth=2.2,
            markersize=6,
            capsize=4,
            elinewidth=1.4,
            label=BASELINE_LABELS[baseline],
        )

    ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Observed prefix k")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_xticks(sorted(df["k"].unique()))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_figure(df: pd.DataFrame, output_path: Path, dataset_label: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.patch.set_facecolor("#f5f1e8")

    for ax, (metric, title, subtitle) in zip(axes.flat, PLOT_SPECS):
        _plot_metric(ax, df, metric, title, subtitle)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        f"Trajectory Completion Baselines\n{dataset_label}",
        fontsize=15,
        fontweight="bold",
        y=1.06,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot trajectory completion evaluation results.")
    parser.add_argument(
        "--game-summary-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "learning_wave_complete_gt10_k1358" / "game_summary.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "learning_wave_complete_gt10_k1358" / "trajectory_completion_summary.png",
    )
    parser.add_argument(
        "--stats-output-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "learning_wave_complete_gt10_k1358" / "plot_metric_summary.csv",
    )
    parser.add_argument(
        "--dataset-label",
        type=str,
        default="Learning-wave complete games, num_rounds > 10",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.game_summary_csv)
    plot_df = _summarize_for_plot(df)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(args.stats_output_csv, index=False)
    build_figure(plot_df, args.output_path, args.dataset_label)
    print(f"Wrote plot to {args.output_path}")
    print(f"Wrote plot stats to {args.stats_output_csv}")


if __name__ == "__main__":
    main()
