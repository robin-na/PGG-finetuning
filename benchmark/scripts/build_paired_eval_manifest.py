#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Set

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


def compute_strict_eligible_games(
    analysis_df: pd.DataFrame,
    player_rounds_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
) -> Set[str]:
    valid_games = as_str_set(
        analysis_df.loc[
            normalize_bool(analysis_df["valid_number_of_starting_players"]),
            "gameId",
        ]
    )

    rounds_valid = player_rounds_df[player_rounds_df["gameId"].isin(valid_games)].copy()
    complete_games = as_str_set(
        rounds_valid.groupby("gameId")["data.roundPayoff"]
        .apply(lambda s: bool(s.notna().all()))
        .loc[lambda s: s]
        .index
    )

    gp = rounds_valid[["gameId", "playerId"]].drop_duplicates()
    demo = demographics_df.copy()
    demo["has_any_demo"] = has_any_demo(demo)
    demo = demo[["gameId", "playerId", "has_any_demo"]].drop_duplicates(
        subset=["gameId", "playerId"], keep="first"
    )
    merged = gp.merge(demo, on=["gameId", "playerId"], how="left")
    demo_games = as_str_set(
        merged.groupby("gameId")["has_any_demo"]
        .apply(
            lambda s: (
                lambda v: bool(v.notna().all() and v.fillna(False).all())
            )(s.astype("boolean"))
        )
        .loc[lambda s: s]
        .index
    )

    return valid_games & complete_games & demo_games


def pick_one(rng: random.Random, game_ids: List[str]) -> str:
    return game_ids[rng.randrange(len(game_ids))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build paired eval manifest with one game per CONFIG_configId per punishment condition "
            "(punishmentExists=False and punishmentExists=True)."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("benchmark/data"),
        help="Dataset root containing raw_data/processed_data/demographics.",
    )
    parser.add_argument(
        "--wave",
        choices=["validation_wave", "learning_wave"],
        default="validation_wave",
        help="Wave to build manifest from.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help=(
            "Apply strict filters: valid start players + all players complete + "
            "all players have at least one demographic field."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for game selection inside each pair cell.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output manifest CSV. Default is under reports/benchmark/manifests.",
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=None,
        help="Output summary JSON. Default is next to --output-csv.",
    )
    parser.add_argument(
        "--output-game-ids-txt",
        type=Path,
        default=None,
        help="Optional text file with comma-separated game IDs for direct --game_ids usage.",
    )
    parser.add_argument(
        "--strict-require-pairs",
        action="store_true",
        default=True,
        help="Fail if any CONFIG_configId lacks either punishment=False or punishment=True candidates.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = resolve_repo_path(args.data_root)
    wave = str(args.wave)
    split = "val" if wave == "validation_wave" else "learn"

    analysis_csv = data_root / "processed_data" / f"df_analysis_{split}.csv"
    rounds_csv = data_root / "raw_data" / wave / "player-rounds.csv"
    demographics_csv = data_root / "demographics" / f"demographics_numeric_{split}.csv"

    analysis = pd.read_csv(analysis_csv)
    rounds = pd.read_csv(rounds_csv, usecols=["gameId", "playerId", "data.roundPayoff"])
    demographics = pd.read_csv(demographics_csv)

    analysis["gameId"] = analysis["gameId"].astype(str)
    analysis["CONFIG_configId"] = analysis["CONFIG_configId"].astype(str)
    analysis["CONFIG_punishmentExists"] = normalize_bool(analysis["CONFIG_punishmentExists"])
    rounds["gameId"] = rounds["gameId"].astype(str)
    rounds["playerId"] = rounds["playerId"].astype(str)
    demographics["gameId"] = demographics["gameId"].astype(str)
    demographics["playerId"] = demographics["playerId"].astype(str)

    if args.strict:
        eligible_games = compute_strict_eligible_games(analysis, rounds, demographics)
    else:
        eligible_games = as_str_set(analysis["gameId"])

    candidates = analysis[analysis["gameId"].isin(eligible_games)].copy()
    candidates = candidates.drop_duplicates(subset=["gameId"], keep="first")

    grouped = (
        candidates.groupby(["CONFIG_configId", "CONFIG_punishmentExists"])["gameId"]
        .apply(lambda s: sorted(as_str_set(s)))
        .to_dict()
    )

    config_ids = sorted(as_str_set(candidates["CONFIG_configId"]), key=lambda x: (int(x) if x.isdigit() else x))
    missing_pairs: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []
    rng = random.Random(int(args.seed))

    for cfg in config_ids:
        false_games = list(grouped.get((cfg, False), []))
        true_games = list(grouped.get((cfg, True), []))
        if not false_games or not true_games:
            missing_pairs.append(
                {
                    "CONFIG_configId": cfg,
                    "n_false_candidates": len(false_games),
                    "n_true_candidates": len(true_games),
                }
            )
            continue
        selected_false = pick_one(rng, false_games)
        selected_true = pick_one(rng, true_games)
        rows.append(
            {
                "CONFIG_configId": cfg,
                "CONFIG_punishmentExists": False,
                "gameId": selected_false,
                "n_candidates_in_cell": len(false_games),
                "selection_seed": int(args.seed),
            }
        )
        rows.append(
            {
                "CONFIG_configId": cfg,
                "CONFIG_punishmentExists": True,
                "gameId": selected_true,
                "n_candidates_in_cell": len(true_games),
                "selection_seed": int(args.seed),
            }
        )

    if missing_pairs and args.strict_require_pairs:
        raise ValueError(
            "Some CONFIG_configId values do not have both punishment conditions. "
            f"Missing details: {missing_pairs}"
        )

    manifest = pd.DataFrame(rows).sort_values(
        by=["CONFIG_configId", "CONFIG_punishmentExists"], ascending=[True, True]
    )
    manifest["CONFIG_punishmentExists"] = manifest["CONFIG_punishmentExists"].astype(bool)

    try:
        data_root_label = str(data_root.resolve().relative_to(REPO_ROOT.resolve())).replace("/", "__")
    except Exception:
        data_root_label = data_root.name or "data_root"
    default_base = (
        Path("reports/benchmark/manifests")
        / f"{data_root_label}__{wave}__strict_{int(bool(args.strict))}__seed_{int(args.seed)}"
    )
    output_csv = resolve_repo_path(args.output_csv) if args.output_csv else resolve_repo_path(default_base.with_suffix(".csv"))
    output_summary = (
        resolve_repo_path(args.output_summary_json)
        if args.output_summary_json
        else output_csv.with_suffix(".summary.json")
    )
    output_game_ids = (
        resolve_repo_path(args.output_game_ids_txt)
        if args.output_game_ids_txt
        else output_csv.with_suffix(".game_ids.txt")
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_csv, index=False)
    game_ids = manifest["gameId"].astype(str).tolist()
    output_game_ids.write_text(",".join(game_ids) + "\n", encoding="utf-8")

    summary = {
        "data_root": str(data_root),
        "wave": wave,
        "strict_filters_applied": bool(args.strict),
        "seed": int(args.seed),
        "n_candidate_games_after_filters": int(candidates["gameId"].nunique()),
        "n_candidate_config_ids": int(len(config_ids)),
        "n_selected_rows": int(len(manifest)),
        "n_selected_unique_games": int(manifest["gameId"].nunique()),
        "n_selected_config_ids": int(manifest["CONFIG_configId"].nunique()),
        "missing_pair_config_ids": missing_pairs,
        "output_csv": str(output_csv),
        "output_game_ids_txt": str(output_game_ids),
    }
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
