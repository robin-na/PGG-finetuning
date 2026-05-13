"""Create chip-bargaining behavior-skew figures for the persona-transfer audit."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_LABELS = {
    "final_surplus": "Final surplus",
    "final_welfare": "Final welfare",
    "proposer_mean_net_surplus": "Proposer net surplus",
    "proposer_acceptance_rate": "Proposal accepted",
    "proposer_mean_trade_ratio": "Trade ratio offered",
    "response_acceptance_rate": "Responder acceptance",
    "response_mean_net_surplus_if_accepted": "Responder surplus if accepted",
    "received_trade_rate": "Received trade",
}

METRIC_ORDER = [
    "final_surplus",
    "final_welfare",
    "proposer_mean_net_surplus",
    "proposer_acceptance_rate",
    "proposer_mean_trade_ratio",
    "response_acceptance_rate",
    "response_mean_net_surplus_if_accepted",
    "received_trade_rate",
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


def _modal_share_null(n: int, k: int, *, iterations: int = 10000, seed: int = 10) -> tuple[float, float, float]:
    rng = random.Random(seed + n * 1009 + k * 9173)
    values: list[float] = []
    for _ in range(iterations):
        counts = [0] * k
        for __ in range(n):
            counts[rng.randrange(k)] += 1
        values.append(max(counts) / n)
    return _percentile(values, 0.025), sum(values) / len(values), _percentile(values, 0.975)


def _weighted_sd(rows: list[dict[str, str]], metric: str, weight_column: str) -> float:
    values: list[tuple[float, float]] = []
    for row in rows:
        value = float(row.get(metric, "nan"))
        weight = float(row.get(weight_column, 0.0))
        if not math.isnan(value) and weight > 0:
            values.append((value, weight))
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return float("nan")
    mean = sum(value * weight for value, weight in values) / total_weight
    variance = sum(weight * (value - mean) ** 2 for value, weight in values) / total_weight
    return math.sqrt(max(variance, 0.0))


def _plot_behavior_skew(metadata_dir: Path, output_dir: Path, output_stem: str) -> None:
    significance_rows = _read_csv(metadata_dir / "chip_significance_checks.csv")
    candidate_rows = _read_csv(metadata_dir / "chip_candidate_uniform_behavior_long.csv")
    by_metric = {row["metric"]: row for row in significance_rows}
    sd_by_metric = {
        metric: _weighted_sd(candidate_rows, metric, "candidate_uniform_weight")
        for metric in METRIC_ORDER
    }

    rows = [
        by_metric[metric]
        for metric in METRIC_ORDER
        if metric in by_metric and sd_by_metric.get(metric, 0.0) > 0
    ]
    values = [float(row["mean_diff"]) / sd_by_metric[row["metric"]] for row in rows]
    lower = [float(row["record_low"]) / sd_by_metric[row["metric"]] for row in rows]
    upper = [float(row["record_high"]) / sd_by_metric[row["metric"]] for row in rows]
    xerr = [
        [value - low for value, low in zip(values, lower)],
        [high - value for value, high in zip(values, upper)],
    ]
    y_positions = list(range(len(rows)))
    colors = ["#1f6f8b" if value >= 0 else "#a23b3b" for value in values]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axvline(0, color="#4d4d4d", linewidth=1)
    ax.errorbar(values, y_positions, xerr=xerr, fmt="none", ecolor="#303030", elinewidth=1.2, capsize=3)
    ax.scatter(values, y_positions, s=48, color=colors, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([METRIC_LABELS[row["metric"]] for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Standardized matched-minus-baseline difference")
    ax.set_title("Twin persona summaries matched to chip-bargaining behavior")
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    fig.tight_layout(rect=[0, 0, 1, 1])

    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"{output_stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def _median(values: list[float]) -> float:
    values = sorted(values)
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2


def _plot_modal_share_panel(ax: plt.Axes, game_rows: list[dict[str, str]]) -> None:
    rows = sorted(
        [row for row in game_rows if float(row.get("top1_total", 0.0)) > 0],
        key=lambda row: float(row["top1_top_player_share"])
        - _modal_share_null(int(float(row["top1_total"])), int(row["n_players"]))[1],
        reverse=True,
    )
    y_positions = list(range(len(rows)))

    values = []
    for y, row in zip(y_positions, rows):
        n = int(float(row["top1_total"]))
        k = int(row["n_players"])
        observed = float(row["top1_top_player_share"])
        low, null_mean, high = _modal_share_null(n, k)
        values.append(observed)
        ax.plot([low, high], [y, y], color="#9a9a9a", linewidth=4, solid_capstyle="round", zorder=1)
        ax.scatter([null_mean], [y], facecolor="white", edgecolor="#4d4d4d", s=28, zorder=2)
        ax.scatter([observed], [y], color="#9a6b54", s=32, zorder=3)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel("Share of personas selecting modal player")
    ax.set_title("A. Local collapse exceeds uniform null")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    if values:
        ax.text(
            0.03,
            0.04,
            f"Median: {_median(values):.0%}",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "pad": 3},
        )
    ax.scatter([], [], color="#9a6b54", s=32, label="Observed")
    ax.scatter([], [], facecolor="white", edgecolor="#4d4d4d", s=28, label="Null mean")
    ax.plot([], [], color="#9a9a9a", linewidth=4, label="95% null interval")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=8)


def _plot_effective_n_panel(ax: plt.Axes, game_rows: list[dict[str, str]]) -> None:
    rows = [row for row in game_rows if float(row.get("top1_total", 0.0)) > 0]
    x_values = [float(row["n_players"]) for row in rows]
    y_values = [float(row["top1_effective_n"]) for row in rows]
    colors = [float(row["top1_top_player_share"]) for row in rows]
    max_axis = max(x_values + y_values) if rows else 3.0

    scatter = ax.scatter(
        x_values,
        y_values,
        c=colors,
        cmap="YlOrBr",
        vmin=0,
        vmax=1,
        s=58,
        edgecolor="#303030",
        linewidth=0.5,
        zorder=3,
    )
    ax.plot([0, max_axis], [0, max_axis], color="#4d4d4d", linewidth=1, linestyle="--")
    ax.set_xlim(0, max_axis * 1.05)
    ax.set_ylim(0, max_axis * 1.05)
    ax.set_xlabel("Players in game")
    ax.set_ylabel("Top-1 effective N across personas")
    ax.set_title("B. Effective coverage vs uniform")
    ax.grid(color="#e6e6e6", linewidth=0.8)
    cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Modal player share")


def _plot_coverage(metadata_dir: Path, output_dir: Path, output_stem: str) -> None:
    game_rows = _read_csv(metadata_dir / "chip_game_concentration.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [1.05, 1]})
    _plot_modal_share_panel(axes[0], game_rows)
    _plot_effective_n_panel(axes[1], game_rows)
    fig.suptitle("Within-game coverage in chip-bargaining persona matches", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"{output_stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def _plot_main(metadata_dir: Path, output_dir: Path, output_stem: str) -> None:
    significance_rows = _read_csv(metadata_dir / "chip_significance_checks.csv")
    candidate_rows = _read_csv(metadata_dir / "chip_candidate_uniform_behavior_long.csv")
    game_rows = _read_csv(metadata_dir / "chip_game_concentration.csv")
    by_metric = {row["metric"]: row for row in significance_rows}
    sd_by_metric = {
        metric: _weighted_sd(candidate_rows, metric, "candidate_uniform_weight")
        for metric in METRIC_ORDER
    }
    rows = [
        by_metric[metric]
        for metric in METRIC_ORDER
        if metric in by_metric and sd_by_metric.get(metric, 0.0) > 0
    ]
    values = [float(row["mean_diff"]) / sd_by_metric[row["metric"]] for row in rows]
    lower = [float(row["record_low"]) / sd_by_metric[row["metric"]] for row in rows]
    upper = [float(row["record_high"]) / sd_by_metric[row["metric"]] for row in rows]
    xerr = [
        [value - low for value, low in zip(values, lower)],
        [high - value for value, high in zip(values, upper)],
    ]
    y_positions = list(range(len(rows)))
    colors = ["#1f6f8b" if value >= 0 else "#a23b3b" for value in values]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.axvline(0, color="#4d4d4d", linewidth=1)
    ax.errorbar(values, y_positions, xerr=xerr, fmt="none", ecolor="#303030", elinewidth=1.2, capsize=3)
    ax.scatter(values, y_positions, s=42, color=colors, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([METRIC_LABELS[row["metric"]] for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Standardized matched-minus-baseline difference")
    ax.set_title("A. Matched players skew toward accepted proposals")
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)

    _plot_modal_share_panel(axes[1], game_rows)
    axes[1].set_title("B. Local collapse exceeds uniform null")
    fig.suptitle("Twin persona summaries matched to chip-bargaining behavior", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"{output_stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    metadata_dir = args.metadata_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )

    if args.figure == "behavior":
        _plot_behavior_skew(metadata_dir, output_dir, args.output_stem)
    elif args.figure == "coverage":
        _plot_coverage(metadata_dir, output_dir, args.output_stem)
    elif args.figure == "main":
        _plot_main(metadata_dir, output_dir, args.output_stem)
    else:
        raise ValueError(f"Unsupported figure: {args.figure}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("forecasting/persona_transfer_audit/figures"),
    )
    parser.add_argument("--output-stem", default="figure_chip_bargain_behavior_skew")
    parser.add_argument("--figure", choices=["behavior", "coverage", "main"], default="behavior")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
