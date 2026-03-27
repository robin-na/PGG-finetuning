#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PLAYER_ROUNDS = Path("data/raw_data/learning_wave/player-rounds.csv")
DEFAULT_CONFIG_CSV = Path("data/processed_data/df_analysis_learn.csv")
DEFAULT_OUTPUT_DIR = Path("reports/learning_extreme_contributors")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize players who always fully contribute or always fully defect "
            "throughout a game, and compare those rates across binary config values."
        )
    )
    parser.add_argument("--player-rounds", type=Path, default=DEFAULT_PLAYER_ROUNDS)
    parser.add_argument("--config-csv", type=Path, default=DEFAULT_CONFIG_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_BINARY_CONFIGS))
    return parser.parse_args()


def coerce_bool_like(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return value


def normalize_value(value: Any) -> Any:
    value = coerce_bool_like(value)
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return value
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


def value_label(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):g}"
    return str(value)


def build_scope_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    scopes: dict[str, pd.DataFrame] = {
        "matched_all": df.copy(),
    }
    if "valid_number_of_starting_players" in df.columns:
        valid_mask = (
            df["valid_number_of_starting_players"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
        scopes["valid_only"] = df.loc[valid_mask].copy()
    return scopes


def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    n_total = int(len(df))
    n_complete = int(df["complete_round_coverage"].sum())
    n_always_full = int(df["always_full"].sum())
    n_always_zero = int(df["always_zero"].sum())
    full_share = float(df["always_full"].mean()) if n_total else np.nan
    zero_share = float(df["always_zero"].mean()) if n_total else np.nan
    ratio = float(n_always_full / n_always_zero) if n_always_zero > 0 else np.nan
    return pd.DataFrame(
        [
            {
                "n_player_game_rows": n_total,
                "n_complete_round_coverage": n_complete,
                "share_complete_round_coverage": float(n_complete / n_total) if n_total else np.nan,
                "n_always_full": n_always_full,
                "share_always_full": full_share,
                "n_always_zero": n_always_zero,
                "share_always_zero": zero_share,
                "full_to_zero_count_ratio": ratio,
                "n_neither": int(n_total - n_always_full - n_always_zero),
                "share_neither": float(
                    1.0 - full_share - zero_share
                ) if n_total else np.nan,
            }
        ]
    )


def summarize_by_config(
    df: pd.DataFrame,
    config_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []

    overall_full = float(df["always_full"].mean())
    overall_zero = float(df["always_zero"].mean())

    for col in config_cols:
        if col not in df.columns:
            continue
        sub = df[[col, "gameId", "always_full", "always_zero"]].copy()
        sub["config_value"] = sub[col].map(normalize_value)
        sub = sub.dropna(subset=["config_value"]).copy()
        if sub.empty:
            continue

        full_shares: list[float] = []
        zero_shares: list[float] = []
        group_sizes: list[int] = []
        feature_rows: list[dict[str, Any]] = []

        for config_value, grp in sorted(
            sub.groupby("config_value", dropna=False),
            key=lambda item: value_sort_key(item[0]),
        ):
            n_rows = int(len(grp))
            n_full = int(grp["always_full"].sum())
            n_zero = int(grp["always_zero"].sum())
            full_share = float(grp["always_full"].mean())
            zero_share = float(grp["always_zero"].mean())
            full_shares.append(full_share)
            zero_shares.append(zero_share)
            group_sizes.append(n_rows)
            feature_rows.append(
                {
                    "config_feature": col,
                    "config_value": value_label(config_value),
                    "n_games": int(grp["gameId"].nunique()),
                    "n_player_game_rows": n_rows,
                    "n_always_full": n_full,
                    "share_always_full": full_share,
                    "delta_vs_overall_full_share": full_share - overall_full,
                    "n_always_zero": n_zero,
                    "share_always_zero": zero_share,
                    "delta_vs_overall_zero_share": zero_share - overall_zero,
                    "full_to_zero_count_ratio": float(n_full / n_zero) if n_zero > 0 else np.nan,
                }
            )

        if len(feature_rows) < 2:
            continue

        summary_rows.extend(feature_rows)
        effect_rows.append(
            {
                "config_feature": col,
                "n_values": int(len(feature_rows)),
                "n_player_game_rows": int(sum(group_sizes)),
                "overall_full_share": overall_full,
                "overall_zero_share": overall_zero,
                "max_full_share_gap": float(max(full_shares) - min(full_shares)),
                "max_zero_share_gap": float(max(zero_shares) - min(zero_shares)),
                "min_group_size": int(min(group_sizes)),
                "max_group_size": int(max(group_sizes)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    effect_df = pd.DataFrame(effect_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["config_feature", "config_value"])
    if not effect_df.empty:
        effect_df = effect_df.sort_values(
            ["max_full_share_gap", "max_zero_share_gap"],
            ascending=[False, False],
        )
    return summary_df, effect_df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_cols = list(dict.fromkeys(args.configs))

    rounds = pd.read_csv(
        args.player_rounds,
        usecols=["gameId", "playerId", "data.contribution"],
    )
    rounds["gameId"] = rounds["gameId"].astype(str)
    rounds["playerId"] = rounds["playerId"].astype(str)
    rounds["contribution"] = pd.to_numeric(rounds["data.contribution"], errors="coerce")
    rounds = rounds.dropna(subset=["contribution"]).copy()

    cfg_cols = ["gameId", "CONFIG_numRounds", "CONFIG_endowment", "valid_number_of_starting_players"] + config_cols
    cfg_cols = list(dict.fromkeys(cfg_cols))
    cfg = pd.read_csv(args.config_csv, usecols=cfg_cols)
    cfg["gameId"] = cfg["gameId"].astype(str)
    cfg = cfg.drop_duplicates(subset=["gameId"], keep="first").copy()

    merged = rounds.merge(cfg, on="gameId", how="left")
    merged = merged.loc[merged[config_cols].notna().all(axis=1)].copy()

    player_game = (
        merged.groupby(["gameId", "playerId"], as_index=False)
        .agg(
            n_observed_rounds=("contribution", "size"),
            min_contribution=("contribution", "min"),
            max_contribution=("contribution", "max"),
            expected_rounds=("CONFIG_numRounds", "first"),
            endowment=("CONFIG_endowment", "first"),
            valid_number_of_starting_players=("valid_number_of_starting_players", "first"),
            **{col: (col, "first") for col in config_cols},
        )
    )

    player_game["expected_rounds"] = pd.to_numeric(player_game["expected_rounds"], errors="coerce")
    player_game["endowment"] = pd.to_numeric(player_game["endowment"], errors="coerce")
    player_game["complete_round_coverage"] = (
        player_game["expected_rounds"].notna()
        & (player_game["n_observed_rounds"] == player_game["expected_rounds"])
    )
    player_game["always_full"] = (
        player_game["complete_round_coverage"]
        & player_game["endowment"].notna()
        & (player_game["min_contribution"] == player_game["endowment"])
        & (player_game["max_contribution"] == player_game["endowment"])
    )
    player_game["always_zero"] = (
        player_game["complete_round_coverage"]
        & (player_game["min_contribution"] == 0)
        & (player_game["max_contribution"] == 0)
    )

    scopes = build_scope_frames(player_game)

    manifest: dict[str, Any] = {
        "player_rounds": str(args.player_rounds),
        "config_csv": str(args.config_csv),
        "config_columns": config_cols,
        "output_dir": str(args.output_dir),
        "definition": {
            "always_full": "player contributed exactly CONFIG_endowment in every observed round and had complete round coverage",
            "always_zero": "player contributed 0 in every observed round and had complete round coverage",
            "complete_round_coverage": "n_observed_rounds == CONFIG_numRounds",
        },
    }

    for scope_name, scope_df in scopes.items():
        overall_df = summarize_overall(scope_df)
        by_config_df, effect_df = summarize_by_config(scope_df, config_cols)

        overall_df.to_csv(args.output_dir / f"{scope_name}_overall_summary.csv", index=False)
        by_config_df.to_csv(args.output_dir / f"{scope_name}_config_value_summary.csv", index=False)
        effect_df.to_csv(args.output_dir / f"{scope_name}_config_effect_ranking.csv", index=False)

        manifest[scope_name] = {
            "n_player_game_rows": int(len(scope_df)),
            "n_games": int(scope_df["gameId"].nunique()),
        }

    player_game.to_csv(args.output_dir / "player_game_extreme_strategy_flags.csv", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
