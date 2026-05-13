"""Create manuscript figures for the persona-transfer audit."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_LABELS = {
    "mean_contribution_rate": "Mean contribution",
    "full_contribution_rate": "Full contribution",
    "zero_contribution_rate": "Zero contribution",
    "contribution_sd": "Contribution variability",
    "messages_per_round": "Messages per round",
    "reward_given_round_rate": "Reward giving",
    "punish_given_round_rate": "Punishment giving",
    "punish_received_round_rate": "Punishment received",
}

METRIC_ORDER = [
    "mean_contribution_rate",
    "full_contribution_rate",
    "zero_contribution_rate",
    "contribution_sd",
    "messages_per_round",
    "reward_given_round_rate",
    "punish_given_round_rate",
    "punish_received_round_rate",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def _modal_share_null(n: int, k: int, *, iterations: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed + n * 1009 + k * 9173)
    values: list[float] = []
    for _ in range(iterations):
        counts = [0] * k
        for __ in range(n):
            counts[rng.randrange(k)] += 1
        values.append(max(counts) / n)
    return _percentile(values, 0.025), sum(values) / len(values), _percentile(values, 0.975)


def _plot_behavior_panel(
    ax: plt.Axes,
    significance_rows: list[dict[str, str]],
    comparison_rows: list[dict[str, str]],
) -> None:
    by_metric = {row["metric"]: row for row in significance_rows}
    sd_by_metric = {
        row["metric"]: float(row["candidate_uniform_sd"])
        for row in comparison_rows
        if float(row["candidate_uniform_sd"]) > 0
    }
    ordered = [by_metric[metric] for metric in METRIC_ORDER if metric in by_metric and metric in sd_by_metric]
    y_positions = list(range(len(ordered)))
    values = [float(row["mean_diff"]) / sd_by_metric[row["metric"]] for row in ordered]
    lower = [float(row["game_id_cluster_ci_low"]) / sd_by_metric[row["metric"]] for row in ordered]
    upper = [float(row["game_id_cluster_ci_high"]) / sd_by_metric[row["metric"]] for row in ordered]
    xerr = [[value - low for value, low in zip(values, lower)], [high - value for value, high in zip(values, upper)]]
    colors = ["#1f6f8b" if value >= 0 else "#a23b3b" for value in values]

    ax.axvline(0, color="#4d4d4d", linewidth=1)
    ax.errorbar(values, y_positions, xerr=xerr, fmt="none", ecolor="#303030", elinewidth=1.2, capsize=3)
    ax.scatter(values, y_positions, s=42, color=colors, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([METRIC_LABELS[row["metric"]] for row in ordered])
    ax.invert_yaxis()
    ax.set_xlabel("Standardized matched-minus-baseline difference")
    ax.set_title("A. Matched players skew toward cooperation and reward")
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)


def _plot_within_game_uniform_null_panel(ax: plt.Axes, game_rows: list[dict[str, str]]) -> None:
    rows = sorted(
        [row for row in game_rows if float(row.get("top1_total", 0.0)) > 0],
        key=lambda row: float(row["top1_top_player_share"])
        - _modal_share_null(int(float(row["top1_total"])), int(row["n_players"]))[1],
        reverse=True,
    )
    y_positions = list(range(len(rows)))

    observed_values = []
    null_means = []
    for y, row in zip(y_positions, rows):
        n = int(float(row["top1_total"]))
        k = int(row["n_players"])
        observed = float(row["top1_top_player_share"])
        low, null_mean, high = _modal_share_null(n, k)
        observed_values.append(observed)
        null_means.append(null_mean)
        ax.plot([low, high], [y, y], color="#9a9a9a", linewidth=4, solid_capstyle="round", zorder=1)
        ax.scatter([null_mean], [y], facecolor="white", edgecolor="#4d4d4d", s=28, zorder=2)
        ax.scatter([observed], [y], color="#9a6b54", s=32, zorder=3)
    ax.set_yticks([])
    ax.set_xlabel("Share of personas selecting modal player")
    ax.set_title("B. Local collapse exceeds uniform null")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    if observed_values:
        median = sorted(observed_values)[len(observed_values) // 2] if len(observed_values) % 2 else (
            sorted(observed_values)[len(observed_values) // 2 - 1] + sorted(observed_values)[len(observed_values) // 2]
        ) / 2
        ax.text(
            0.03,
            0.04,
            f"Median: {median:.0%}",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "pad": 3},
        )
    ax.scatter([], [], color="#9a6b54", s=32, label="Observed")
    ax.scatter([], [], facecolor="white", edgecolor="#4d4d4d", s=28, label="Null mean")
    ax.plot([], [], color="#9a9a9a", linewidth=4, label="95% null interval")
    ax.legend(frameon=False, loc="upper right", fontsize=8)


def _plot_within_game_effective_n_scatter(ax: plt.Axes, game_rows: list[dict[str, str]]) -> None:
    rows = [row for row in game_rows if float(row.get("top1_total", 0.0)) > 0]
    x_values = [float(row["n_players"]) for row in rows]
    y_values = [float(row["top1_effective_n"]) for row in rows]
    colors = [float(row["top1_top_player_share"]) for row in rows]
    max_axis = max(x_values + y_values) if rows else 1.0

    scatter = ax.scatter(
        x_values,
        y_values,
        c=colors,
        cmap="YlOrBr",
        vmin=0,
        vmax=1,
        s=54,
        edgecolor="#303030",
        linewidth=0.5,
        zorder=3,
    )
    ax.plot([0, max_axis], [0, max_axis], color="#4d4d4d", linewidth=1, linestyle="--")
    ax.set_xlim(0, max_axis * 1.05)
    ax.set_ylim(0, max_axis * 1.05)
    ax.set_xlabel("Players in game")
    ax.set_ylabel("Top-1 effective N across personas")
    ax.set_title("B. Effective coverage falls below uniform")
    ax.grid(color="#e6e6e6", linewidth=0.8)
    cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Modal player share")


def main(args: argparse.Namespace) -> None:
    metadata_dir = args.metadata_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    significance_rows = _read_csv(metadata_dir / "comprehensive_significance_checks.csv")
    comparison_rows = _read_csv(metadata_dir / "comprehensive_behavior_metric_comparison.csv")
    game_rows = _read_csv(metadata_dir / "comprehensive_game_concentration.csv")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    _plot_behavior_panel(axes[0], significance_rows, comparison_rows)
    if args.panel_b == "effective_n_scatter":
        _plot_within_game_effective_n_scatter(axes[1], game_rows)
        panel_b_note = (
            "Panel B plots, for each fixed game, the number of players against the effective number "
            "of top-1 selected players across personas; the dashed line is uniform coverage. "
        )
    else:
        _plot_within_game_uniform_null_panel(axes[1], game_rows)
        panel_b_note = (
            "Panel B asks, within each fixed game, what share of personas selected the same modal top-1 player. "
        )
    fig.suptitle("Twin persona summaries matched to revealed public-goods-game behavior", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"{args.output_stem}.{suffix}", bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("forecasting/persona_transfer_audit/figures"),
    )
    parser.add_argument(
        "--panel-b",
        choices=["modal_share", "effective_n_scatter"],
        default="modal_share",
    )
    parser.add_argument("--output-stem", default="figure_1_transfer_audit_main")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
