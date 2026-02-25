#!/usr/bin/env python3
"""
Plot proportional stacked bars of cluster distributions by CONFIG values.

For each tag and each CONFIG column:
- x-axis: CONFIG value
- y-axis: P(cluster_id | CONFIG value, tag)
- bars are stacked by cluster_id
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CLUSTER_ROOT = Path("Persona/misc/tag_section_clusters_openai")
DEFAULT_CONFIG_CSV = Path("data/processed_data/df_analysis_learn.csv")
DEFAULT_OUTPUT_DIR = Path("Persona/misc/tag_section_clusters_openai/analysis_config_distributions/plots_stacked_by_config")

CONFIG_COLUMNS = [
    "CONFIG_chat",
    "CONFIG_punishmentExists",
    "CONFIG_showRewardId",
    "CONFIG_rewardMagnitude",
    "CONFIG_punishmentMagnitude",
    "CONFIG_showNRounds",
    "CONFIG_showPunishmentId",
    "CONFIG_punishmentCost",
    "CONFIG_rewardExists",
    "CONFIG_MPCR",
    "CONFIG_showOtherSummaries",
    "CONFIG_numRounds",
    "CONFIG_allOrNothing",
    "CONFIG_defaultContribProp",
    "CONFIG_playerCount",
    "CONFIG_rewardCost",
]


def load_cluster_rows(cluster_root: Path) -> pd.DataFrame:
    files = sorted(cluster_root.glob("*/*_clustered.jsonl"))
    if not files:
        raise FileNotFoundError(f"No *_clustered.jsonl files found under {cluster_root}")

    rows: List[dict] = []
    for p in files:
        tag = p.parent.name
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                obj["tag_folder"] = tag
                rows.append(obj)
    if not rows:
        raise ValueError("No rows found in clustered files")
    return pd.DataFrame(rows)


def value_sort_key(v) -> Tuple[int, float, str]:
    if pd.isna(v):
        return (2, 0.0, "NA")
    s = str(v).strip()
    low = s.lower()
    if low in {"false", "0", "0.0"}:
        return (0, 0.0, "false")
    if low in {"true", "1", "1.0"}:
        return (0, 1.0, "true")
    try:
        return (1, float(s), s)
    except Exception:
        return (3, 0.0, s)


def format_value_label(v) -> str:
    if pd.isna(v):
        return "NA"
    s = str(v)
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - int(f)) < 1e-9:
            return str(int(f))
        return f"{f:g}"
    except Exception:
        return s


def build_cluster_order(df_tag: pd.DataFrame) -> List[int]:
    ids = sorted(df_tag["cluster_id"].dropna().astype(int).unique().tolist())
    return ids


def build_cluster_color_map(cluster_ids: List[int]) -> Dict[int, tuple]:
    cmap = plt.get_cmap("tab20")
    colors: Dict[int, tuple] = {}
    for i, cid in enumerate(cluster_ids):
        colors[cid] = cmap(i % 20)
    return colors


def plot_stacked_for_tag_config(
    df_tag: pd.DataFrame,
    tag: str,
    config_col: str,
    output_path: Path,
) -> Dict[str, int]:
    sub = df_tag[["cluster_id", config_col]].copy()
    sub = sub.dropna(subset=["cluster_id", config_col])
    if sub.empty:
        return {"n_rows": 0, "n_values": 0, "n_clusters": 0}

    cluster_ids = build_cluster_order(df_tag)
    if not cluster_ids:
        return {"n_rows": 0, "n_values": 0, "n_clusters": 0}

    values = sorted(sub[config_col].unique().tolist(), key=value_sort_key)
    if not values:
        return {"n_rows": 0, "n_values": 0, "n_clusters": len(cluster_ids)}

    counts = (
        sub.groupby([config_col, "cluster_id"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=values, columns=cluster_ids, fill_value=0)
    )
    props = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)

    x = np.arange(len(values))
    width = max(10.0, 0.45 * len(values))
    fig_height = 6.0
    fig, ax = plt.subplots(figsize=(width, fig_height), constrained_layout=True)

    color_map = build_cluster_color_map(cluster_ids)
    bottom = np.zeros(len(values), dtype=float)
    for cid in cluster_ids:
        y = props[cid].to_numpy(dtype=float)
        ax.bar(
            x,
            y,
            bottom=bottom,
            color=color_map[cid],
            width=0.9,
            label=f"cluster {cid}",
            linewidth=0.0,
        )
        bottom += y

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion within CONFIG value")
    ax.set_xlabel(config_col)
    ax.set_title(f"{tag}: cluster distribution by {config_col}")
    ax.set_xticks(x)
    ax.set_xticklabels([format_value_label(v) for v in values], rotation=60, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        ncol=1,
        fontsize=8,
    )

    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    return {"n_rows": int(len(sub)), "n_values": int(len(values)), "n_clusters": int(len(cluster_ids))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot stacked cluster proportions by CONFIG value")
    parser.add_argument("--cluster-root", type=Path, default=DEFAULT_CLUSTER_ROOT)
    parser.add_argument("--config-csv", type=Path, default=DEFAULT_CONFIG_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    df_cluster = load_cluster_rows(args.cluster_root)
    df_cluster["experiment"] = df_cluster["experiment"].astype(str)
    df_cluster["cluster_id"] = df_cluster["cluster_id"].astype(int)

    cfg = pd.read_csv(args.config_csv)
    needed = ["gameId"] + CONFIG_COLUMNS
    missing = [c for c in needed if c not in cfg.columns]
    if missing:
        raise ValueError(f"Missing CONFIG columns in {args.config_csv}: {missing}")

    cfg_game = cfg[needed].drop_duplicates(subset=["gameId"], keep="first").copy()
    cfg_game["gameId"] = cfg_game["gameId"].astype(str)

    merged = df_cluster.merge(cfg_game, how="left", left_on="experiment", right_on="gameId")
    unmatched = int(merged["gameId"].isna().sum())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cluster_root": str(args.cluster_root),
        "config_csv": str(args.config_csv),
        "output_dir": str(args.output_dir),
        "n_rows": int(len(merged)),
        "n_unmatched_rows": unmatched,
        "tags": {},
    }

    tags = sorted(merged["tag_folder"].dropna().unique().tolist())
    for tag in tags:
        tag_dir = args.output_dir / tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        df_tag = merged[merged["tag_folder"] == tag].copy()
        tag_meta = {
            "n_rows": int(len(df_tag)),
            "plots": {},
        }

        for col in CONFIG_COLUMNS:
            out_png = tag_dir / f"{col}.png"
            stats = plot_stacked_for_tag_config(df_tag, tag, col, out_png)
            tag_meta["plots"][col] = {
                "file": str(out_png),
                **stats,
            }
        manifest["tags"][tag] = tag_meta

    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved stacked plots to: {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Unmatched rows on experiment->gameId: {unmatched}")


if __name__ == "__main__":
    main()

