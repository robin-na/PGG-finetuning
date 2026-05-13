"""Plot PGG demographic skew in persona-transfer matches."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_METADATA_DIR = Path(
    "forecasting/persona_transfer_audit/metadata/"
    "twin_direct_summary_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2"
)

DEMOGRAPHIC_ORDER = [
    ("gender_self_report", "Man", "Man"),
    ("gender_self_report", "Woman", "Woman"),
    ("education_self_report", "High school", "High school"),
    ("education_self_report", "Bachelor", "Bachelor"),
    ("education_self_report", "Master", "Master"),
    ("education_self_report", "Other", "Other education"),
    ("country_of_residence", "United States", "U.S. residence"),
    ("country_of_residence", "United Kingdom", "U.K. residence"),
    ("nationality", "United States", "U.S. nationality"),
    ("nationality", "United Kingdom", "U.K. nationality"),
    ("employment_status", "Full-Time", "Full-time"),
    ("employment_status", "Part-Time", "Part-time"),
    (
        "employment_status",
        "Not in paid work (e.g. homemaker', 'retired or disabled)",
        "Not in paid work",
    ),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["distribution"], row["field"], row["category"])


def _plot_panel(ax: plt.Axes, rows_by_key: dict[tuple[str, str, str], dict[str, str]], distribution: str) -> None:
    rows = [
        (label, rows_by_key[(distribution, field, category)])
        for field, category, label in DEMOGRAPHIC_ORDER
        if (distribution, field, category) in rows_by_key
    ]
    y_positions = list(range(len(rows)))
    values = [100 * float(row["mean_diff"]) for _, row in rows]
    lower = [100 * float(row["crossed_cluster_ci_low"]) for _, row in rows]
    upper = [100 * float(row["crossed_cluster_ci_high"]) for _, row in rows]
    xerr = [
        [value - low for value, low in zip(values, lower)],
        [high - value for value, high in zip(values, upper)],
    ]
    colors = ["#1f6f8b" if value >= 0 else "#a23b3b" for value in values]

    ax.axvline(0, color="#4d4d4d", linewidth=1)
    ax.errorbar(values, y_positions, xerr=xerr, fmt="none", ecolor="#303030", elinewidth=1.2, capsize=3)
    ax.scatter(
        values,
        y_positions,
        s=48,
        color=colors,
        edgecolor="#ffffff",
        linewidth=0.4,
        zorder=3,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for label, _ in rows])
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)


def main(args: argparse.Namespace) -> None:
    metadata_dir = args.metadata_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(metadata_dir / "pgg_demographic_cluster_significance.csv")
    rows = [row for row in rows if row["metric_type"] == "categorical"]
    rows_by_key = {_row_key(row): row for row in rows}

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1]},
    )
    _plot_panel(axes[0], rows_by_key, "matched_probability")
    _plot_panel(axes[1], rows_by_key, "matched_top1")
    axes[0].invert_yaxis()
    axes[0].set_title("A. Probability-weighted matches")
    axes[1].set_title("B. Top-1 matches")
    axes[0].set_xlabel("Matched minus within-game baseline (percentage points)")
    axes[1].set_xlabel("Matched minus within-game baseline (percentage points)")

    all_bounds = []
    for row in rows:
        if row["distribution"] in {"matched_probability", "matched_top1"}:
            all_bounds.extend(
                [
                    100 * float(row["crossed_cluster_ci_low"]),
                    100 * float(row["crossed_cluster_ci_high"]),
                    100 * float(row["mean_diff"]),
                ]
            )
    limit = max(abs(value) for value in all_bounds) if all_bounds else 10
    limit = max(10, min(20, limit * 1.12))
    for ax in axes:
        ax.set_xlim(-limit, limit)

    fig.suptitle("Demographic skew in PGG persona-transfer matches", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"{args.output_stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("forecasting/persona_transfer_audit/figures"),
    )
    parser.add_argument("--output-stem", default="figure_pgg_demographic_skew")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
