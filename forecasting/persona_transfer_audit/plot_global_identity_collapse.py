"""Plot request-conditional global identity-collapse results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    ("selected_identity_share", "Selected identities"),
    ("entropy_effective_n_share", "Effective N"),
    ("top_5pct_share", "Top 5% mass"),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric_rows(metadata_dir: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(metadata_dir / "global_identity_collapse_summary.csv")
    return {row["metric"]: row for row in rows}


def main(args: argparse.Namespace) -> None:
    pgg_rows = _metric_rows(args.pgg_metadata_dir.expanduser().resolve())
    chip_rows = _metric_rows(args.chip_metadata_dir.expanduser().resolve())
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
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9), sharey=True)
    for ax, title, rows in [
        (axes[0], "Public goods game", pgg_rows),
        (axes[1], "Chip bargaining", chip_rows),
    ]:
        y_positions = list(range(len(METRICS)))
        for y, (metric, _) in zip(y_positions, METRICS):
            row = rows[metric]
            observed = float(row["observed"])
            null_mean = float(row["null_mean"])
            low = float(row["null_ci_low"])
            high = float(row["null_ci_high"])
            ax.plot([low, high], [y, y], color="#9a9a9a", linewidth=5, solid_capstyle="round", zorder=1)
            ax.scatter([null_mean], [y], facecolor="white", edgecolor="#4d4d4d", s=46, zorder=2)
            ax.scatter([observed], [y], color="#1f6f8b", s=52, zorder=3)
        ax.set_title(title)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Share of candidate identities / top-1 mass")
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([label for _, label in METRICS])
        ax.invert_yaxis()
    axes[0].scatter([], [], facecolor="white", edgecolor="#4d4d4d", s=46, label="Null mean")
    axes[0].scatter([], [], color="#1f6f8b", s=52, label="Observed")
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle("Global identity collapse relative to request-conditional uniform null", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"{args.output_stem}.{suffix}", bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgg-metadata-dir", type=Path, required=True)
    parser.add_argument("--chip-metadata-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("forecasting/persona_transfer_audit/figures"),
    )
    parser.add_argument("--output-stem", default="figure_global_identity_collapse")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
