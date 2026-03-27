#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_PLAYER_ROUNDS = Path("data/raw_data/learning_wave/player-rounds.csv")
DEFAULT_EXTREME_FLAGS = Path("reports/learning_extreme_contributors/player_game_extreme_strategy_flags.csv")
DEFAULT_TAG_BASE = Path("Persona/archetype_retrieval/learning_wave")
DEFAULT_OUTPUT_DIR = Path("reports/learning_noncontribution_cluster_contribution_levels")

DEFAULT_TAGS = [
    "COMMUNICATION",
    "RESPONSE_TO_END_GAME",
    "RESPONSE_TO_OTHERS_OUTCOME",
    "PUNISHMENT",
    "REWARD",
    "RESPONSE_TO_PUNISHER",
    "RESPONSE_TO_REWARDER",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize average contribution levels by non-contribution persona cluster "
            "for valid complete player-games."
        )
    )
    parser.add_argument("--player-rounds", type=Path, default=DEFAULT_PLAYER_ROUNDS)
    parser.add_argument("--extreme-flags", type=Path, default=DEFAULT_EXTREME_FLAGS)
    parser.add_argument("--tag-base", type=Path, default=DEFAULT_TAG_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tags", nargs="+", default=list(DEFAULT_TAGS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rounds = pd.read_csv(
        args.player_rounds,
        usecols=["gameId", "playerId", "data.contribution"],
    )
    rounds["gameId"] = rounds["gameId"].astype(str)
    rounds["playerId"] = rounds["playerId"].astype(str)
    rounds["contribution"] = pd.to_numeric(rounds["data.contribution"], errors="coerce")
    rounds = rounds.dropna(subset=["contribution"]).copy()

    player_summary = (
        rounds.groupby(["gameId", "playerId"], as_index=False)
        .agg(
            mean_contribution=("contribution", "mean"),
            median_contribution=("contribution", "median"),
            mean_frac_zero=("contribution", lambda s: float((s == 0).mean())),
            mean_frac_full=("contribution", lambda s: float((s == 20).mean())),
        )
    )

    flags = pd.read_csv(args.extreme_flags)
    flags = flags[
        flags["valid_number_of_starting_players"].astype(bool)
        & flags["complete_round_coverage"].astype(bool)
    ][["gameId", "playerId"]].copy()
    flags["gameId"] = flags["gameId"].astype(str)
    flags["playerId"] = flags["playerId"].astype(str)

    base = player_summary.merge(flags, on=["gameId", "playerId"], how="inner")

    cluster_rows: list[pd.DataFrame] = []
    spread_rows: list[dict[str, object]] = []

    for tag in args.tags:
        cluster_paths = sorted((args.tag_base / tag).glob("*_clustered.jsonl"))
        if not cluster_paths:
            continue
        cluster_df = pd.read_json(cluster_paths[0], lines=True)[
            ["gameId", "playerId", "cluster_id", "cluster_title"]
        ].copy()
        cluster_df["gameId"] = cluster_df["gameId"].astype(str)
        cluster_df["playerId"] = cluster_df["playerId"].astype(str)

        merged = base.merge(cluster_df, on=["gameId", "playerId"], how="inner")
        if merged.empty:
            continue

        summary_df = (
            merged.groupby(["cluster_id", "cluster_title"], as_index=False)
            .agg(
                n_player_game_rows=("playerId", "size"),
                mean_contribution=("mean_contribution", "mean"),
                median_contribution=("mean_contribution", "median"),
                mean_frac_zero=("mean_frac_zero", "mean"),
                mean_frac_full=("mean_frac_full", "mean"),
            )
            .sort_values("mean_contribution", ascending=False)
        )
        summary_df["tag"] = tag
        cluster_rows.append(summary_df)

        mean_by_cluster = summary_df.set_index("cluster_id")["mean_contribution"]
        spread_rows.append(
            {
                "tag": tag,
                "n_player_game_rows": int(len(merged)),
                "n_clusters": int(summary_df["cluster_id"].nunique()),
                "max_cluster_mean_contribution": float(mean_by_cluster.max()),
                "min_cluster_mean_contribution": float(mean_by_cluster.min()),
                "mean_contribution_gap": float(mean_by_cluster.max() - mean_by_cluster.min()),
                "top_cluster_id": int(mean_by_cluster.idxmax()),
                "bottom_cluster_id": int(mean_by_cluster.idxmin()),
            }
        )

    cluster_summary_df = pd.concat(cluster_rows, ignore_index=True) if cluster_rows else pd.DataFrame()
    spread_df = pd.DataFrame(spread_rows).sort_values("mean_contribution_gap", ascending=False)

    cluster_summary_df.to_csv(args.output_dir / "cluster_contribution_summary.csv", index=False)
    spread_df.to_csv(args.output_dir / "tag_contribution_spread_summary.csv", index=False)

    manifest = {
        "player_rounds": str(args.player_rounds),
        "extreme_flags": str(args.extreme_flags),
        "tag_base": str(args.tag_base),
        "output_dir": str(args.output_dir),
        "tags": args.tags,
        "notes": [
            "Contribution summaries are computed at the player-game level, then averaged within each non-contribution persona cluster.",
            "Only valid_only players with complete round coverage are included.",
            "mean_contribution is the player's average contribution over all observed rounds in that game.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
