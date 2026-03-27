#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PLAYER_ROUNDS = Path("data/raw_data/learning_wave/player-rounds.csv")
DEFAULT_EXTREME_FLAGS = Path("reports/learning_extreme_contributors/player_game_extreme_strategy_flags.csv")
DEFAULT_OUTPUT_DIR = Path("reports/learning_nonextreme_patterns")
FULL_CONTRIBUTION = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize common round-by-round contribution patterns among players "
            "who are neither always-full contributors nor always-zero contributors."
        )
    )
    parser.add_argument("--player-rounds", type=Path, default=DEFAULT_PLAYER_ROUNDS)
    parser.add_argument("--extreme-flags", type=Path, default=DEFAULT_EXTREME_FLAGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def classify_behavior(values: pd.Series, full_value: float = FULL_CONTRIBUTION) -> str:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if vals.empty:
        return "unknown"

    uniq = sorted(vals.unique())
    mean_v = float(vals.mean())
    std_v = float(vals.std(ddof=0))
    frac_zero = float((vals == 0).mean())
    frac_full = float((vals == full_value).mean())

    if (vals == full_value).all():
        return "always_full"
    if (vals == 0).all():
        return "always_zero"
    if set(uniq).issubset({0.0, full_value}) and len(uniq) > 1:
        return "binary_0_20"
    if mean_v >= 15 and frac_full >= 0.5:
        return "high_cooperator"
    if mean_v <= 5 and frac_zero >= 0.5:
        return "low_contributor"
    if std_v <= 2:
        return "stable_mid"
    return "variable_mid"


def summarize_direction(first_value: float, last_value: float) -> str:
    if last_value > first_value:
        return "up"
    if last_value < first_value:
        return "down"
    return "flat"


def slope_from_values(values: np.ndarray) -> float:
    x = np.arange(1, len(values) + 1, dtype=float)
    if len(values) < 2:
        return np.nan
    return float(np.polyfit(x, values.astype(float), 1)[0])


def summarize_scope(
    rounds: pd.DataFrame,
    flags: pd.DataFrame,
    scope_name: str,
    output_dir: Path,
) -> None:
    if flags.empty:
        return

    scope_rounds = rounds.merge(flags[["gameId", "playerId"]], on=["gameId", "playerId"], how="inner")
    if scope_rounds.empty:
        return

    player_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []

    for (game_id, player_id), group in scope_rounds.groupby(["gameId", "playerId"], sort=False):
        values = group["contribution"].to_numpy(dtype=float)
        first_value = float(values[0])
        last_value = float(values[-1])
        switches = int(np.sum(np.diff(values) != 0))
        pattern = classify_behavior(group["contribution"])

        player_rows.append(
            {
                "gameId": game_id,
                "playerId": player_id,
                "pattern": pattern,
                "n_rounds": int(len(values)),
                "mean_contribution": float(values.mean()),
                "std_contribution": float(values.std(ddof=0)),
                "min_contribution": float(values.min()),
                "max_contribution": float(values.max()),
                "frac_zero": float((values == 0).mean()),
                "frac_full": float((values == FULL_CONTRIBUTION).mean()),
                "first_contribution": first_value,
                "last_contribution": last_value,
                "direction": summarize_direction(first_value, last_value),
                "n_switches": switches,
                "slope_per_round": slope_from_values(values),
            }
        )

        example_rows.append(
            {
                "gameId": game_id,
                "playerId": player_id,
                "pattern": pattern,
                "trajectory": json.dumps([float(v) for v in values.tolist()]),
            }
        )

    player_df = pd.DataFrame(player_rows)
    example_df = pd.DataFrame(example_rows)

    summary_df = (
        player_df.groupby("pattern", as_index=False)
        .agg(
            n_players=("playerId", "size"),
            mean_of_means=("mean_contribution", "mean"),
            median_of_means=("mean_contribution", "median"),
            mean_std=("std_contribution", "mean"),
            mean_frac_zero=("frac_zero", "mean"),
            mean_frac_full=("frac_full", "mean"),
            mean_first=("first_contribution", "mean"),
            mean_last=("last_contribution", "mean"),
            mean_slope=("slope_per_round", "mean"),
            mean_switches=("n_switches", "mean"),
            median_switches=("n_switches", "median"),
            share_up=("direction", lambda s: float((s == "up").mean())),
            share_down=("direction", lambda s: float((s == "down").mean())),
            share_flat=("direction", lambda s: float((s == "flat").mean())),
        )
        .sort_values("n_players", ascending=False)
    )
    summary_df["share"] = summary_df["n_players"] / summary_df["n_players"].sum()

    direction_df = (
        player_df.groupby(["pattern", "direction"], as_index=False)
        .agg(n_players=("playerId", "size"))
        .sort_values(["pattern", "n_players"], ascending=[True, False])
    )
    direction_df["share_within_pattern"] = direction_df["n_players"] / direction_df.groupby("pattern")["n_players"].transform("sum")

    example_df = (
        example_df.groupby("pattern", as_index=False)
        .head(5)
        .reset_index(drop=True)
    )

    player_df.to_csv(output_dir / f"{scope_name}_player_game_pattern_labels.csv", index=False)
    summary_df.to_csv(output_dir / f"{scope_name}_pattern_summary.csv", index=False)
    direction_df.to_csv(output_dir / f"{scope_name}_pattern_direction_summary.csv", index=False)
    example_df.to_csv(output_dir / f"{scope_name}_example_trajectories.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rounds = pd.read_csv(
        args.player_rounds,
        usecols=["gameId", "playerId", "createdAt", "data.contribution"],
    )
    rounds["gameId"] = rounds["gameId"].astype(str)
    rounds["playerId"] = rounds["playerId"].astype(str)
    rounds["createdAt"] = pd.to_datetime(rounds["createdAt"], errors="coerce")
    rounds["contribution"] = pd.to_numeric(rounds["data.contribution"], errors="coerce")
    rounds = rounds.dropna(subset=["createdAt", "contribution"]).copy()
    rounds = rounds.sort_values(["gameId", "playerId", "createdAt"]).copy()

    flags = pd.read_csv(args.extreme_flags)
    flags["gameId"] = flags["gameId"].astype(str)
    flags["playerId"] = flags["playerId"].astype(str)
    flags = flags[
        flags["complete_round_coverage"].astype(bool)
        & ~flags["always_full"].astype(bool)
        & ~flags["always_zero"].astype(bool)
    ].copy()

    scopes = {
        "matched_all": flags.copy(),
        "valid_only": flags.loc[flags["valid_number_of_starting_players"].astype(bool)].copy(),
    }

    for scope_name, scope_flags in scopes.items():
        summarize_scope(rounds=rounds, flags=scope_flags, scope_name=scope_name, output_dir=args.output_dir)

    manifest = {
        "player_rounds": str(args.player_rounds),
        "extreme_flags": str(args.extreme_flags),
        "output_dir": str(args.output_dir),
        "notes": [
            "Trajectories are ordered within player-game by createdAt timestamp.",
            "Only complete-coverage player-games are included.",
            "Always-full and always-zero players are excluded before pattern classification.",
            "Pattern labels reuse the simple interpretable classifier from scripts/analyze_learning_match_contributions.py.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
