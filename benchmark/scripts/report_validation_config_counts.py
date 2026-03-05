#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Set

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "t", "yes"})
    )


def as_str_set(values: Iterable[object]) -> Set[str]:
    return {str(v) for v in values if pd.notna(v)}


def has_any_demo(df: pd.DataFrame) -> pd.Series:
    return df["age"].notna() | df["gender_code"].notna() | df["education_code"].notna()


def games_all_players_have_any_demo(
    player_rounds: pd.DataFrame,
    demographics: pd.DataFrame,
    candidate_games: Set[str],
) -> Set[str]:
    pr = player_rounds[player_rounds["gameId"].isin(candidate_games)][["gameId", "playerId"]].drop_duplicates()
    demo = demographics.copy()
    demo["has_any_demo"] = has_any_demo(demo)
    demo = demo[demo["gameId"].isin(candidate_games)][["gameId", "playerId", "has_any_demo"]].drop_duplicates(
        subset=["gameId", "playerId"], keep="first"
    )
    merged = pr.merge(demo, on=["gameId", "playerId"], how="left")
    ok = (
        merged.groupby("gameId")["has_any_demo"]
        .apply(
            lambda s: (
                lambda v: bool(v.notna().all() and v.fillna(False).all())
            )(s.astype("boolean"))
        )
        .loc[lambda s: s]
    )
    return as_str_set(ok.index)


def games_all_players_completed(player_rounds: pd.DataFrame, candidate_games: Set[str]) -> Set[str]:
    pr = player_rounds[player_rounds["gameId"].isin(candidate_games)].copy()
    if "data.roundPayoff" not in pr.columns:
        raise KeyError("player-rounds.csv missing required column: data.roundPayoff")
    ok = pr.groupby("gameId")["data.roundPayoff"].apply(lambda s: s.notna().all()).loc[lambda s: s]
    return as_str_set(ok.index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count validation games per CONFIG_configId after strict filters: "
            "valid_number_of_starting_players, all players completed, and all players have "
            "at least one demographic field."
        )
    )
    parser.add_argument(
        "--analysis-csv",
        type=Path,
        default=Path("data/processed_data/df_analysis_val.csv"),
    )
    parser.add_argument(
        "--player-rounds-csv",
        type=Path,
        default=Path("data/raw_data/validation_wave/player-rounds.csv"),
    )
    parser.add_argument(
        "--demographics-csv",
        type=Path,
        default=Path("demographics/demographics_numeric_val.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/benchmark/validation_config_counts_strict.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("reports/benchmark/validation_config_counts_strict_summary.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_csv = resolve_repo_path(args.analysis_csv)
    player_rounds_csv = resolve_repo_path(args.player_rounds_csv)
    demographics_csv = resolve_repo_path(args.demographics_csv)
    output_csv = resolve_repo_path(args.output_csv)
    summary_json = resolve_repo_path(args.summary_json)

    analysis = pd.read_csv(analysis_csv)
    player_rounds = pd.read_csv(player_rounds_csv)
    demographics = pd.read_csv(demographics_csv)

    analysis["gameId"] = analysis["gameId"].astype(str)
    if "CONFIG_configId" not in analysis.columns:
        raise KeyError("df_analysis_val.csv missing CONFIG_configId")
    analysis["CONFIG_configId"] = analysis["CONFIG_configId"].astype(str)

    player_rounds["gameId"] = player_rounds["gameId"].astype(str)
    player_rounds["playerId"] = player_rounds["playerId"].astype(str)
    demographics["gameId"] = demographics["gameId"].astype(str)
    demographics["playerId"] = demographics["playerId"].astype(str)

    valid_start_games = as_str_set(
        analysis.loc[
            normalize_bool(analysis["valid_number_of_starting_players"]),
            "gameId",
        ]
    )
    completed_games = games_all_players_completed(player_rounds, valid_start_games)
    demo_games = games_all_players_have_any_demo(player_rounds, demographics, valid_start_games)
    eligible_games = valid_start_games & completed_games & demo_games

    analysis_eligible = analysis[analysis["gameId"].isin(eligible_games)].copy()
    counts = (
        analysis_eligible.groupby("CONFIG_configId", as_index=False)["gameId"]
        .nunique()
        .rename(columns={"gameId": "n_games"})
        .sort_values(["n_games", "CONFIG_configId"], ascending=[False, True])
        .reset_index(drop=True)
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(output_csv, index=False)

    summary = {
        "analysis_csv": str(analysis_csv),
        "player_rounds_csv": str(player_rounds_csv),
        "demographics_csv": str(demographics_csv),
        "n_total_games": int(analysis["gameId"].nunique()),
        "n_valid_start_games": int(len(valid_start_games)),
        "n_completed_games_within_valid_start": int(len(completed_games)),
        "n_any_demo_games_within_valid_start": int(len(demo_games)),
        "n_eligible_games": int(len(eligible_games)),
        "n_unique_configs_with_eligible_games": int(counts["CONFIG_configId"].nunique()),
        "output_csv": str(output_csv),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
