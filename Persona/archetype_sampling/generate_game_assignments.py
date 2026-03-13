#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Macro_simulation_eval.simulator import build_game_contexts
from Persona.archetype_sampling.runtime import (
    CONFIG_BANK_MODE,
    DEFAULT_LEARN_SUMMARY_POOL,
    SoftBankSummarySampler,
    assign_archetypes_for_game,
    load_finished_summary_pool,
)


DEFAULT_ROUNDS_CSV = "data/raw_data/validation_wave/player-rounds.csv"
DEFAULT_ANALYSIS_CSV = "data/processed_data/df_analysis_val.csv"
DEFAULT_DEMOGRAPHICS_CSV = "demographics/demographics_numeric_val.csv"
DEFAULT_PLAYERS_CSV = "data/raw_data/validation_wave/players.csv"


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_seeds(raw: str) -> List[int]:
    seeds: List[int] = []
    for token in str(raw or "").split(","):
        value = token.strip()
        if not value:
            continue
        seeds.append(int(value))
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def _apply_data_root_paths(args: argparse.Namespace) -> None:
    data_root = str(getattr(args, "data_root", "") or "").strip()
    if not data_root:
        return

    wave = str(getattr(args, "wave", "validation_wave") or "validation_wave").strip()
    if wave not in {"validation_wave", "learning_wave"}:
        raise ValueError(f"Unsupported --wave '{wave}'. Allowed values: validation_wave, learning_wave.")

    analysis_name = "df_analysis_val.csv" if wave == "validation_wave" else "df_analysis_learn.csv"
    demographics_name = (
        "demographics_numeric_val.csv"
        if wave == "validation_wave"
        else "demographics_numeric_learn.csv"
    )
    defaults = {
        "analysis_csv": DEFAULT_ANALYSIS_CSV,
        "rounds_csv": DEFAULT_ROUNDS_CSV,
        "players_csv": DEFAULT_PLAYERS_CSV,
        "demographics_csv": DEFAULT_DEMOGRAPHICS_CSV,
    }
    derived = {
        "analysis_csv": str(Path(data_root) / "processed_data" / analysis_name),
        "rounds_csv": str(Path(data_root) / "raw_data" / wave / "player-rounds.csv"),
        "players_csv": str(Path(data_root) / "raw_data" / wave / "players.csv"),
        "demographics_csv": str(Path(data_root) / "demographics" / demographics_name),
    }
    for key, path in derived.items():
        current = str(getattr(args, key, "") or "").strip()
        if (not current) or (current == defaults[key]):
            setattr(args, key, path)


def _select_contexts(contexts: List[Any], args: argparse.Namespace) -> List[Any]:
    selected = list(contexts)
    if args.game_ids:
        wanted = {token.strip() for token in str(args.game_ids).split(",") if token.strip()}
        selected = [ctx for ctx in selected if ctx.game_id in wanted or ctx.game_name in wanted]
    if args.max_games is not None:
        selected = selected[: int(args.max_games)]
    if not selected:
        raise ValueError("No games selected for assignment generation.")
    return selected


def generate_assignments(args: argparse.Namespace) -> Dict[str, Any]:
    _apply_data_root_paths(args)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df_analysis = pd.read_csv(args.analysis_csv)
    df_rounds = pd.read_csv(args.rounds_csv)
    df_players = pd.read_csv(args.players_csv) if args.players_csv and Path(args.players_csv).exists() else pd.DataFrame()
    df_demographics = (
        pd.read_csv(args.demographics_csv)
        if args.demographics_csv and Path(args.demographics_csv).exists()
        else pd.DataFrame()
    )
    contexts = _select_contexts(
        build_game_contexts(
            df_analysis=df_analysis,
            df_rounds=df_rounds,
            df_players=df_players,
            df_demographics=df_demographics,
        ),
        args,
    )

    summary_pool = load_finished_summary_pool(str(args.summary_pool))
    sampler = SoftBankSummarySampler(
        summary_pool_path=str(args.summary_pool),
        temperature=float(args.temperature),
    )
    seeds = _parse_seeds(args.seeds)

    seed_outputs: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        assignment_rows: List[Dict[str, Any]] = []
        for ctx in contexts:
            batch = assign_archetypes_for_game(
                mode=CONFIG_BANK_MODE,
                game_id=ctx.game_id,
                player_ids=ctx.player_ids,
                env=ctx.env,
                seed=int(seed),
                summary_pool=summary_pool,
                summary_pool_path=str(args.summary_pool),
                soft_bank_sampler=sampler,
            )
            assignment_rows.extend(batch.manifest_rows)
            summary_rows.append(
                {
                    "seed": int(seed),
                    "target_gameId": ctx.game_id,
                    "target_gameName": ctx.game_name,
                    "player_count": int(len(ctx.player_ids)),
                    "assignment_mode": str(batch.summary.get("mode") or ""),
                    "bank_size": int(batch.summary.get("bank_size") or 0),
                    "temperature": float(batch.summary.get("temperature") or args.temperature),
                    "effective_support": float(batch.summary.get("effective_support") or 0.0),
                    "max_weight": float(batch.summary.get("max_weight") or 0.0),
                }
            )

        assignment_path = output_dir / f"game_assignments_{CONFIG_BANK_MODE}_seed{int(seed)}.jsonl"
        _write_jsonl(assignment_path, assignment_rows)
        seed_outputs.append(
            {
                "seed": int(seed),
                "assignment_jsonl": str(assignment_path),
                "num_games": int(len(contexts)),
                "num_assignment_rows": int(len(assignment_rows)),
            }
        )

    summary_path = output_dir / "game_assignment_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    manifest = {
        "mode": CONFIG_BANK_MODE,
        "summary_pool": str(args.summary_pool),
        "analysis_csv": str(args.analysis_csv),
        "rounds_csv": str(args.rounds_csv),
        "players_csv": str(args.players_csv),
        "demographics_csv": str(args.demographics_csv),
        "temperature": float(args.temperature),
        "seeds": seeds,
        "num_games": int(len(contexts)),
        "game_ids": [ctx.game_id for ctx in contexts],
        "seed_outputs": seed_outputs,
        "summary_csv": str(summary_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "manifest_json": str(manifest_path),
        "summary_csv": str(summary_path),
        "seed_outputs": seed_outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute config-bank archetype assignments for selected games.")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--wave", type=str, default="validation_wave")
    parser.add_argument("--analysis_csv", type=str, default=DEFAULT_ANALYSIS_CSV)
    parser.add_argument("--rounds_csv", type=str, default=DEFAULT_ROUNDS_CSV)
    parser.add_argument("--players_csv", type=str, default=DEFAULT_PLAYERS_CSV)
    parser.add_argument("--demographics_csv", type=str, default=DEFAULT_DEMOGRAPHICS_CSV)
    parser.add_argument("--summary_pool", type=Path, default=Path(DEFAULT_LEARN_SUMMARY_POOL))
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--game_ids", type=str, default=None)
    parser.add_argument("--max_games", type=int, default=None)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "game_assignment_manifests" / CONFIG_BANK_MODE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_assignments(args)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
