#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.common.profiles import load_twin_personas, sample_twin_profiles
from forecasting.datasets.chip_bargain import build_bundle as build_chip_bargain_bundle


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "twin_to_chip_bargain_player_sampling_unadjusted"
DEFAULT_CARDS_DIR_NAME = "chip_bargain_prompt_min"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample one unadjusted Twin persona per unique chip-bargain player, then expand to per-game assignments."
    )
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cards-dir-name",
        type=str,
        default=DEFAULT_CARDS_DIR_NAME,
        help="Twin profile-card directory name under non-PGG_generalization/twin_profiles/output/twin_extended_profile_cards/.",
    )
    return parser.parse_args()


def _twin_profiles_jsonl(repo_root: Path) -> Path:
    return (
        repo_root
        / "non-PGG_generalization"
        / "twin_profiles"
        / "output"
        / "twin_extended_profiles"
        / "twin_extended_profiles.jsonl"
    )


def _twin_cards_jsonl(repo_root: Path, cards_dir_name: str) -> Path:
    return (
        repo_root
        / "non-PGG_generalization"
        / "twin_profiles"
        / "output"
        / "twin_extended_profile_cards"
        / cards_dir_name
        / "twin_extended_profile_cards.jsonl"
    )


def _player_unit_id(chip_family: str, player_id: str) -> str:
    return f"{chip_family}::{player_id}"


def main() -> None:
    args = parse_args()
    bundle = build_chip_bargain_bundle(args.repo_root)
    records = bundle.records.copy()

    player_rows: list[dict[str, Any]] = []
    for row in records.itertuples(index=False):
        players = json.loads(str(row.players_json))
        for player_id in players:
            player_rows.append(
                {
                    "unit_id": _player_unit_id(str(row.chip_family), str(player_id)),
                    "chip_family": str(row.chip_family),
                    "player_id": str(player_id),
                }
            )
    player_units = (
        pd.DataFrame(player_rows)
        .drop_duplicates("unit_id")
        .sort_values(["chip_family", "player_id"])
        .reset_index(drop=True)
    )

    twin_profiles_jsonl = _twin_profiles_jsonl(args.repo_root)
    twin_cards_jsonl = _twin_cards_jsonl(args.repo_root, args.cards_dir_name)
    twin_personas, _ = load_twin_personas(twin_profiles_jsonl, twin_cards_jsonl)

    seed_dir = args.output_dir / f"seed_{args.seed}"
    unit_to_assignment, unit_assignments_path = sample_twin_profiles(
        units=player_units[["unit_id"]],
        demographic_source=pd.DataFrame(),
        twin_personas=twin_personas,
        match_fields=[],
        seed=args.seed,
        output_dir=seed_dir,
        dataset_key="chip_bargain",
        corrected=False,
        cards_path=twin_cards_jsonl,
        profiles_path=twin_profiles_jsonl,
    )

    game_assignment_rows: list[dict[str, Any]] = []
    for row in records.itertuples(index=False):
        players = json.loads(str(row.players_json))
        assignments = []
        for seat_index, player_id in enumerate(players, start=1):
            unit_id = _player_unit_id(str(row.chip_family), str(player_id))
            assignment = unit_to_assignment[unit_id]
            assignments.append(
                {
                    "seat_index": int(seat_index),
                    "chip_bargain_player_id": str(player_id),
                    "player_unit_id": unit_id,
                    "twin_pid": assignment["twin_pid"],
                    "twin_profile_headline": assignment["headline"],
                    "twin_profile_summary": assignment["summary"],
                    "target_source": "unadjusted_random_full_pool",
                }
            )
        game_assignment_rows.append(
            {
                "gameId": str(row.record_id),
                "chip_family": str(row.chip_family),
                "cohort_name": str(row.cohort_name),
                "stage_name": str(row.stage_name),
                "treatment_name": str(row.treatment_name),
                "assignments": assignments,
            }
        )

    game_assignments_path = seed_dir / "game_assignments.jsonl"
    with game_assignments_path.open("w", encoding="utf-8") as handle:
        for row in game_assignment_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    summary = {
        "dataset_key": "chip_bargain",
        "variant": "twin_sampled_unadjusted_seed_0",
        "seed": args.seed,
        "game_count": int(len(game_assignment_rows)),
        "player_unit_count": int(len(player_units)),
        "unit_assignments_file": str(unit_assignments_path),
        "game_assignments_file": str(game_assignments_path),
        "source_profile_cards_file": str(twin_cards_jsonl),
        "source_profiles_file": str(twin_profiles_jsonl),
    }
    (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
