#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PLAYER_ROUNDS = Path("data/raw_data/learning_wave/player-rounds.csv")
DEFAULT_EXTREME_FLAGS = Path("reports/learning_extreme_contributors/player_game_extreme_strategy_flags.csv")
DEFAULT_TAG_BASE = Path("Persona/archetype_retrieval/learning_wave")
DEFAULT_OUTPUT_DIR = Path("reports/learning_noncontribution_cluster_behavior_by_config")

DEFAULT_BINARY_CONFIGS = [
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

DEFAULT_TAGS = [
    "COMMUNICATION",
    "PUNISHMENT",
    "REWARD",
    "RESPONSE_TO_END_GAME",
    "RESPONSE_TO_OTHERS_OUTCOME",
    "RESPONSE_TO_PUNISHER",
    "RESPONSE_TO_REWARDER",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose config-dependent changes in contribution / punishment / reward "
            "through shifts in non-contribution persona cluster mixtures."
        )
    )
    parser.add_argument("--player-rounds", type=Path, default=DEFAULT_PLAYER_ROUNDS)
    parser.add_argument("--extreme-flags", type=Path, default=DEFAULT_EXTREME_FLAGS)
    parser.add_argument("--tag-base", type=Path, default=DEFAULT_TAG_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_BINARY_CONFIGS))
    parser.add_argument("--tags", nargs="+", default=list(DEFAULT_TAGS))
    return parser.parse_args()


def parse_action_dict(value: Any) -> dict[str, float]:
    if pd.isna(value):
        return {}
    text = str(value).strip()
    if not text or text == "{}" or text.lower() == "nan":
        return {}
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    cleaned: dict[str, float] = {}
    for key, raw in parsed.items():
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            cleaned[str(key)] = numeric
    return cleaned


def normalize_config_value(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isfinite(numeric) and numeric.is_integer():
        return int(numeric)
    return float(numeric)


def value_sort_key(value: Any) -> tuple[int, Any]:
    if pd.isna(value):
        return (3, "")
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float, np.number)):
        return (1, float(value))
    return (2, str(value))


def build_player_game_behavior(
    *,
    player_rounds_path: Path,
    extreme_flags_path: Path,
    config_cols: list[str],
) -> pd.DataFrame:
    rounds = pd.read_csv(
        player_rounds_path,
        usecols=[
            "gameId",
            "playerId",
            "data.contribution",
            "data.punished",
            "data.rewarded",
        ],
    )
    rounds["gameId"] = rounds["gameId"].astype(str)
    rounds["playerId"] = rounds["playerId"].astype(str)
    rounds["contribution"] = pd.to_numeric(rounds["data.contribution"], errors="coerce")
    rounds["punish_units"] = rounds["data.punished"].apply(
        lambda value: float(sum(parse_action_dict(value).values()))
    )
    rounds["reward_units"] = rounds["data.rewarded"].apply(
        lambda value: float(sum(parse_action_dict(value).values()))
    )
    rounds = rounds.dropna(subset=["contribution"]).copy()
    rounds["punish_any"] = (rounds["punish_units"] > 0).astype(float)
    rounds["reward_any"] = (rounds["reward_units"] > 0).astype(float)

    player_game = (
        rounds.groupby(["gameId", "playerId"], as_index=False)
        .agg(
            mean_contribution=("contribution", "mean"),
            mean_punish_units=("punish_units", "mean"),
            mean_reward_units=("reward_units", "mean"),
            punish_any_round_rate=("punish_any", "mean"),
            reward_any_round_rate=("reward_any", "mean"),
        )
    )

    flags = pd.read_csv(extreme_flags_path)
    flags = flags[
        flags["valid_number_of_starting_players"].astype(bool)
        & flags["complete_round_coverage"].astype(bool)
    ][["gameId", "playerId"] + config_cols].copy()
    flags["gameId"] = flags["gameId"].astype(str)
    flags["playerId"] = flags["playerId"].astype(str)
    for col in config_cols:
        if col in flags.columns:
            flags[col] = flags[col].map(normalize_config_value)

    return player_game.merge(flags, on=["gameId", "playerId"], how="inner")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_cols = list(dict.fromkeys(args.configs))
    behavior_df = build_player_game_behavior(
        player_rounds_path=args.player_rounds,
        extreme_flags_path=args.extreme_flags,
        config_cols=config_cols,
    )

    behavior_cols = [
        "mean_contribution",
        "mean_punish_units",
        "mean_reward_units",
        "punish_any_round_rate",
        "reward_any_round_rate",
    ]

    profile_rows: list[pd.DataFrame] = []
    config_value_rows: list[dict[str, Any]] = []
    config_gap_rows: list[dict[str, Any]] = []

    for tag in args.tags:
        cluster_paths = sorted((args.tag_base / tag).glob("*_clustered.jsonl"))
        if not cluster_paths:
            continue
        cluster_df = pd.read_json(cluster_paths[0], lines=True)[
            ["gameId", "playerId", "cluster_id", "cluster_title"]
        ].copy()
        cluster_df["gameId"] = cluster_df["gameId"].astype(str)
        cluster_df["playerId"] = cluster_df["playerId"].astype(str)

        merged = behavior_df.merge(cluster_df, on=["gameId", "playerId"], how="inner")
        if merged.empty:
            continue

        cluster_profile = (
            merged.groupby(["cluster_id", "cluster_title"], as_index=False)
            .agg(
                n_player_game_rows=("playerId", "size"),
                mean_contribution=("mean_contribution", "mean"),
                mean_punish_units=("mean_punish_units", "mean"),
                mean_reward_units=("mean_reward_units", "mean"),
                punish_any_round_rate=("punish_any_round_rate", "mean"),
                reward_any_round_rate=("reward_any_round_rate", "mean"),
            )
            .sort_values("mean_contribution", ascending=False)
        )
        cluster_profile["tag"] = tag
        profile_rows.append(cluster_profile)

        cluster_behavior_map = (
            cluster_profile.set_index("cluster_id")[behavior_cols].to_dict(orient="index")
        )

        for config_col in config_cols:
            if config_col not in merged.columns:
                continue
            sub = merged.dropna(subset=[config_col]).copy()
            if sub[config_col].nunique() != 2:
                continue

            value_details: dict[Any, dict[str, Any]] = {}
            for config_value, value_group in sorted(
                sub.groupby(config_col),
                key=lambda item: value_sort_key(item[0]),
            ):
                cluster_share = (
                    value_group["cluster_id"]
                    .value_counts(normalize=True)
                    .sort_index()
                )
                top_cluster_id = int(cluster_share.index[0])
                top_cluster_title = str(
                    cluster_profile.loc[
                        cluster_profile["cluster_id"] == top_cluster_id,
                        "cluster_title",
                    ].iloc[0]
                )
                implied = {
                    metric: float(
                        sum(
                            float(share) * float(cluster_behavior_map[int(cluster_id)][metric])
                            for cluster_id, share in cluster_share.items()
                        )
                    )
                    for metric in behavior_cols
                }
                actual = {
                    metric: float(value_group[metric].mean())
                    for metric in behavior_cols
                }
                value_details[config_value] = {
                    "n_player_game_rows": int(len(value_group)),
                    "n_games": int(value_group["gameId"].nunique()),
                    "cluster_share": {int(k): float(v) for k, v in cluster_share.items()},
                    "top_cluster_id": top_cluster_id,
                    "top_cluster_title": top_cluster_title,
                    "top_cluster_share": float(cluster_share.iloc[0]),
                    "actual": actual,
                    "implied": implied,
                }

                config_value_rows.append(
                    {
                        "tag": tag,
                        "config_feature": config_col,
                        "config_value": str(config_value),
                        "n_player_game_rows": int(len(value_group)),
                        "n_games": int(value_group["gameId"].nunique()),
                        "top_cluster_id": top_cluster_id,
                        "top_cluster_title": top_cluster_title,
                        "top_cluster_share": float(cluster_share.iloc[0]),
                        **{f"actual_{metric}": value for metric, value in actual.items()},
                        **{f"implied_{metric}": value for metric, value in implied.items()},
                    }
                )

            values = sorted(value_details.keys(), key=value_sort_key)
            value0, value1 = values
            share0 = value_details[value0]["cluster_share"]
            share1 = value_details[value1]["cluster_share"]
            cluster_ids = sorted(set(share0) | set(share1))
            tvd = 0.5 * sum(
                abs(float(share0.get(cluster_id, 0.0)) - float(share1.get(cluster_id, 0.0)))
                for cluster_id in cluster_ids
            )

            config_gap_rows.append(
                {
                    "tag": tag,
                    "config_feature": config_col,
                    "config_value_0": str(value0),
                    "config_value_1": str(value1),
                    "cluster_mix_total_variation_distance": float(tvd),
                    "dominant_cluster_changes": bool(
                        value_details[value0]["top_cluster_id"] != value_details[value1]["top_cluster_id"]
                    ),
                    "actual_mean_contribution_gap": float(
                        value_details[value1]["actual"]["mean_contribution"]
                        - value_details[value0]["actual"]["mean_contribution"]
                    ),
                    "implied_mean_contribution_gap": float(
                        value_details[value1]["implied"]["mean_contribution"]
                        - value_details[value0]["implied"]["mean_contribution"]
                    ),
                    "actual_mean_punish_units_gap": float(
                        value_details[value1]["actual"]["mean_punish_units"]
                        - value_details[value0]["actual"]["mean_punish_units"]
                    ),
                    "implied_mean_punish_units_gap": float(
                        value_details[value1]["implied"]["mean_punish_units"]
                        - value_details[value0]["implied"]["mean_punish_units"]
                    ),
                    "actual_mean_reward_units_gap": float(
                        value_details[value1]["actual"]["mean_reward_units"]
                        - value_details[value0]["actual"]["mean_reward_units"]
                    ),
                    "implied_mean_reward_units_gap": float(
                        value_details[value1]["implied"]["mean_reward_units"]
                        - value_details[value0]["implied"]["mean_reward_units"]
                    ),
                    "actual_punish_any_round_rate_gap": float(
                        value_details[value1]["actual"]["punish_any_round_rate"]
                        - value_details[value0]["actual"]["punish_any_round_rate"]
                    ),
                    "actual_reward_any_round_rate_gap": float(
                        value_details[value1]["actual"]["reward_any_round_rate"]
                        - value_details[value0]["actual"]["reward_any_round_rate"]
                    ),
                }
            )

    profile_df = pd.concat(profile_rows, ignore_index=True) if profile_rows else pd.DataFrame()
    config_value_df = pd.DataFrame(config_value_rows)
    config_gap_df = pd.DataFrame(config_gap_rows)
    if not config_gap_df.empty:
        config_gap_df = config_gap_df.sort_values(
            [
                "cluster_mix_total_variation_distance",
                "dominant_cluster_changes",
                "tag",
            ],
            ascending=[False, False, True],
        )

    profile_df.to_csv(args.output_dir / "tag_cluster_behavior_profiles.csv", index=False)
    config_value_df.to_csv(args.output_dir / "tag_config_value_behavior_summary.csv", index=False)
    config_gap_df.to_csv(args.output_dir / "tag_config_behavior_gap_summary.csv", index=False)

    manifest = {
        "player_rounds": str(args.player_rounds),
        "extreme_flags": str(args.extreme_flags),
        "tag_base": str(args.tag_base),
        "output_dir": str(args.output_dir),
        "tags": args.tags,
        "binary_configs": config_cols,
        "notes": [
            "Behavior metrics are aggregated at the player-game level, then averaged within clusters.",
            "Only valid_only players with complete round coverage are included.",
            "implied_* metrics use overall cluster behavior profiles combined with the cluster shares under each config value.",
            "Cluster-mix total variation distance measures how much the cluster distribution shifts between the two config values.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
