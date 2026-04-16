#!/usr/bin/env python3
"""Analyze distributions of deterministic Twin profile-card traits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CARDS_CSV = THIS_DIR / "output" / "twin_extended_profile_cards" / "standard" / "twin_extended_profile_cards.csv"
DEFAULT_OUTPUT_DIR = THIS_DIR / "output" / "twin_extended_profile_cards" / "standard" / "analysis"

TRAIT_ORDER = [
    "cooperation_orientation",
    "conditional_cooperation",
    "norm_enforcement",
    "generosity_without_return",
    "exploitation_caution",
    "communication_coordination",
    "behavioral_stability",
]

TRAIT_LABELS = {
    "cooperation_orientation": "Cooperation\norientation",
    "conditional_cooperation": "Conditional\ncooperation",
    "norm_enforcement": "Norm\nenforcement",
    "generosity_without_return": "Generosity\nwithout return",
    "exploitation_caution": "Exploitation\ncaution",
    "communication_coordination": "Communication /\ncoordination",
    "behavioral_stability": "Behavioral\nstability",
}

LABEL_ORDER = ["very_low", "low", "mixed", "medium", "high", "very_high", "unknown"]
LABEL_COLORS = {
    "very_low": "#8b1e3f",
    "low": "#d96c75",
    "mixed": "#f0c987",
    "medium": "#c8d38f",
    "high": "#78b159",
    "very_high": "#2f7d32",
    "unknown": "#9a9a9a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards-csv", type=Path, default=DEFAULT_CARDS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_cards(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_score_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for trait in TRAIT_ORDER:
        series = pd.to_numeric(df[f"{trait}_score"], errors="coerce")
        rows.append(
            {
                "trait": trait,
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std(ddof=1)), 2),
                "min": round(float(series.min()), 2),
                "p10": round(float(series.quantile(0.10)), 2),
                "p25": round(float(series.quantile(0.25)), 2),
                "median": round(float(series.quantile(0.50)), 2),
                "p75": round(float(series.quantile(0.75)), 2),
                "p90": round(float(series.quantile(0.90)), 2),
                "max": round(float(series.max()), 2),
            }
        )
    return pd.DataFrame(rows)


def build_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    total = len(df)
    for trait in TRAIT_ORDER:
        counts = df[f"{trait}_label"].fillna("unknown").value_counts()
        for label in LABEL_ORDER:
            count = int(counts.get(label, 0))
            rows.append(
                {
                    "trait": trait,
                    "label": label,
                    "count": count,
                    "share": round((count / total) * 100, 2) if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_high_low_summary(label_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for trait in TRAIT_ORDER:
        subset = label_df[label_df["trait"] == trait].set_index("label")
        rows.append(
            {
                "trait": trait,
                "very_low_or_low_share": round(float(subset.loc[["very_low", "low"], "share"].sum()), 2),
                "mixed_or_medium_share": round(float(subset.loc[["mixed", "medium"], "share"].sum()), 2),
                "high_or_very_high_share": round(float(subset.loc[["high", "very_high"], "share"].sum()), 2),
            }
        )
    return pd.DataFrame(rows)


def plot_score_distributions(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.8))
    axes = axes.flatten()
    bins = list(range(0, 105, 5))
    for idx, trait in enumerate(TRAIT_ORDER):
        ax = axes[idx]
        series = pd.to_numeric(df[f"{trait}_score"], errors="coerce")
        ax.hist(series, bins=bins, color="#4c78a8", edgecolor="white", linewidth=0.6)
        ax.axvline(series.mean(), color="#b22222", linestyle="--", linewidth=1.2)
        ax.set_title(TRAIT_LABELS[trait], fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Score", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[-1].axis("off")
    fig.suptitle("Twin Profile Card Trait Score Distributions", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.10, wspace=0.28, hspace=0.42)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_label_distributions(label_df: pd.DataFrame, output_path: Path) -> None:
    pivot = label_df.pivot(index="trait", columns="label", values="share").fillna(0.0)
    pivot = pivot.loc[TRAIT_ORDER, LABEL_ORDER]
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    left = pd.Series([0.0] * len(pivot), index=pivot.index)
    for label in LABEL_ORDER:
        values = pivot[label]
        ax.barh(
            [TRAIT_LABELS[idx] for idx in pivot.index],
            values,
            left=left,
            color=LABEL_COLORS[label],
            edgecolor="white",
            linewidth=0.6,
            label=label.replace("_", " "),
        )
        left += values
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of participants (%)", fontsize=10)
    ax.set_title("Twin Profile Card Trait Label Distributions", fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=9)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.77, bottom=0.12)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_mean_scores(score_df: pd.DataFrame, output_path: Path) -> None:
    labels = [TRAIT_LABELS[trait].replace("\n", " ") for trait in score_df["trait"]]
    means = score_df["mean"]
    p25 = score_df["p25"]
    p75 = score_df["p75"]
    med = score_df["median"]

    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    positions = range(len(labels))
    ax.bar(positions, means, color="#7aa6c2", width=0.7)
    ax.errorbar(
        positions,
        med,
        yerr=[med - p25, p75 - med],
        fmt="o",
        color="#1f1f1f",
        capsize=4,
        linewidth=1.2,
        markersize=4,
    )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Twin Profile Card Trait Means with IQR", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.28)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    df = load_cards(args.cards_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    score_df = build_score_summary(df)
    label_df = build_label_distribution(df)
    grouped_df = build_high_low_summary(label_df)

    score_df.to_csv(args.output_dir / "trait_score_summary.csv", index=False)
    label_df.to_csv(args.output_dir / "trait_label_distribution.csv", index=False)
    grouped_df.to_csv(args.output_dir / "trait_grouped_label_summary.csv", index=False)

    plot_score_distributions(df, args.output_dir / "trait_score_distributions.png")
    plot_label_distributions(label_df, args.output_dir / "trait_label_distributions.png")
    plot_mean_scores(score_df, args.output_dir / "trait_mean_iqr.png")

    print(f"Cards analyzed: {len(df)}")
    print(f"Wrote: {args.output_dir / 'trait_score_summary.csv'}")
    print(f"Wrote: {args.output_dir / 'trait_label_distribution.csv'}")
    print(f"Wrote: {args.output_dir / 'trait_grouped_label_summary.csv'}")
    print(f"Wrote: {args.output_dir / 'trait_score_distributions.png'}")
    print(f"Wrote: {args.output_dir / 'trait_label_distributions.png'}")
    print(f"Wrote: {args.output_dir / 'trait_mean_iqr.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
