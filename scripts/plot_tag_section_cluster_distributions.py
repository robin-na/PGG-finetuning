#!/usr/bin/env python3
"""
Plot cluster/config distributions for tag-section clustering outputs.

Input expected:
- cluster_config_distribution_long.csv from
  Persona/misc/tag_section_clusters_openai/analysis_config_distributions
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BINARY_COLS = [
    "CONFIG_chat",
    "CONFIG_punishmentExists",
    "CONFIG_showRewardId",
    "CONFIG_showNRounds",
    "CONFIG_showPunishmentId",
    "CONFIG_rewardExists",
    "CONFIG_showOtherSummaries",
    "CONFIG_allOrNothing",
    "CONFIG_defaultContribProp",
]


def _is_true_value(v) -> bool:
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    return s in {"1", "1.0", "true", "t", "yes", "y"}


def _axes_grid(n: int, ncols: int = 2):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 4.0 * nrows), constrained_layout=True)
    if isinstance(axes, np.ndarray):
        axes_list = axes.flatten().tolist()
    else:
        axes_list = [axes]
    return fig, axes_list


def _cluster_order(df: pd.DataFrame, tag: str) -> List[int]:
    sub = df[df["tag"] == tag][["cluster_id"]].drop_duplicates().sort_values("cluster_id")
    return sub["cluster_id"].astype(int).tolist()


def plot_cluster_sizes(df_long: pd.DataFrame, output_path: Path) -> None:
    sizes = (
        df_long[["tag", "cluster_id", "cluster_title", "cluster_n"]]
        .drop_duplicates()
        .sort_values(["tag", "cluster_id"])
    )
    tags = sorted(sizes["tag"].unique().tolist())
    fig, axes = _axes_grid(len(tags), ncols=2)

    for i, tag in enumerate(tags):
        ax = axes[i]
        sub = sizes[sizes["tag"] == tag].sort_values("cluster_id")
        x = sub["cluster_id"].astype(int).to_numpy()
        y = sub["cluster_n"].astype(int).to_numpy()
        ax.bar(x, y, color="#4C78A8")
        ax.set_title(tag, fontsize=10)
        ax.set_xlabel("cluster_id")
        ax.set_ylabel("n")
        ax.set_xticks(x)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.25)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Cluster Sizes by Tag", fontsize=14)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_binary_true_rates(df_long: pd.DataFrame, output_path: Path) -> None:
    tmp = df_long[df_long["config_column"].isin(BINARY_COLS)].copy()
    tmp["is_true"] = tmp["config_value"].map(_is_true_value)

    true_rates = (
        tmp[tmp["is_true"]]
        .groupby(["tag", "cluster_id", "config_column"], as_index=False)["pct_within_cluster"]
        .sum()
    )

    tags = sorted(df_long["tag"].unique().tolist())
    fig, axes = _axes_grid(len(tags), ncols=2)
    vmin, vmax = 0.0, 1.0
    im = None

    for i, tag in enumerate(tags):
        ax = axes[i]
        cluster_ids = _cluster_order(df_long, tag)
        pivot = (
            true_rates[true_rates["tag"] == tag]
            .pivot(index="cluster_id", columns="config_column", values="pct_within_cluster")
            .reindex(index=cluster_ids, columns=BINARY_COLS)
            .fillna(0.0)
        )
        mat = pivot.to_numpy(dtype=float)
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(tag, fontsize=10)
        ax.set_xlabel("CONFIG column")
        ax.set_ylabel("cluster_id")
        ax.set_xticks(np.arange(len(BINARY_COLS)))
        ax.set_xticklabels(BINARY_COLS, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(cluster_ids)))
        ax.set_yticklabels(cluster_ids, fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes[: len(tags)], shrink=0.85)
        cbar.set_label("P(value=True | cluster)", fontsize=9)

    fig.suptitle("Binary CONFIG True Rates by Tag/Cluster", fontsize=14)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_mode_purity(df_long: pd.DataFrame, output_path: Path) -> None:
    mode_purity = (
        df_long.groupby(["tag", "cluster_id", "config_column"], as_index=False)["pct_within_cluster"]
        .max()
        .rename(columns={"pct_within_cluster": "mode_purity"})
    )
    all_cols = sorted(df_long["config_column"].unique().tolist())
    tags = sorted(df_long["tag"].unique().tolist())
    fig, axes = _axes_grid(len(tags), ncols=2)
    vmin, vmax = 0.0, 1.0
    im = None

    for i, tag in enumerate(tags):
        ax = axes[i]
        cluster_ids = _cluster_order(df_long, tag)
        pivot = (
            mode_purity[mode_purity["tag"] == tag]
            .pivot(index="cluster_id", columns="config_column", values="mode_purity")
            .reindex(index=cluster_ids, columns=all_cols)
            .fillna(0.0)
        )
        mat = pivot.to_numpy(dtype=float)
        im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(tag, fontsize=10)
        ax.set_xlabel("CONFIG column")
        ax.set_ylabel("cluster_id")
        ax.set_xticks(np.arange(len(all_cols)))
        ax.set_xticklabels(all_cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(cluster_ids)))
        ax.set_yticklabels(cluster_ids, fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes[: len(tags)], shrink=0.85)
        cbar.set_label("Mode purity: max P(value | cluster)", fontsize=9)

    fig.suptitle("Config Mode Purity by Tag/Cluster", fontsize=14)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tag-section cluster/config distributions")
    parser.add_argument(
        "--input-long-csv",
        type=Path,
        default=Path("Persona/misc/tag_section_clusters_openai/analysis_config_distributions/cluster_config_distribution_long.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Persona/misc/tag_section_clusters_openai/analysis_config_distributions/plots"),
    )
    args = parser.parse_args()

    df_long = pd.read_csv(args.input_long_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_cluster_sizes(df_long, args.output_dir / "cluster_sizes_by_tag.png")
    plot_binary_true_rates(df_long, args.output_dir / "binary_true_rate_by_tag_cluster.png")
    plot_mode_purity(df_long, args.output_dir / "config_mode_purity_by_tag_cluster.png")

    print(f"Saved plots to: {args.output_dir}")


if __name__ == "__main__":
    main()

