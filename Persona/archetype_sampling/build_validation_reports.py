#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Persona.archetype_sampling.runtime import (
    CONFIG_BANK_MODE,
    DEFAULT_LEARN_SUMMARY_POOL,
    DEFAULT_VAL_PLAYER_TABLE,
    SOFT_BANK_FEATURE_COLUMNS,
    SoftBankSummarySampler,
    build_validation_treatment_contexts,
)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _player_slots(player_count: int) -> List[str]:
    return [f"slot_{idx}" for idx in range(1, int(player_count) + 1)]


def _treatment_env(row: pd.Series) -> Dict[str, Any]:
    env = {col: row.get(col) for col in SOFT_BANK_FEATURE_COLUMNS}
    env["CONFIG_treatmentName"] = row.get("CONFIG_treatmentName")
    return env


def _weighted_sample_positions(weights: List[float], k: int, seed: int, game_id: str) -> List[int]:
    rng = random.Random(f"{seed}|{game_id}|{CONFIG_BANK_MODE}")
    remaining_positions = list(range(len(weights)))
    remaining_weights = [max(0.0, float(weight)) for weight in weights]
    chosen: List[int] = []
    draws = min(int(k), len(remaining_positions))
    for _ in range(draws):
        total = sum(remaining_weights)
        if total <= 0.0:
            break
        threshold = rng.random() * total
        cumulative = 0.0
        picked_pos = len(remaining_weights) - 1
        for pos, weight in enumerate(remaining_weights):
            cumulative += weight
            if cumulative >= threshold:
                picked_pos = pos
                break
        chosen.append(remaining_positions.pop(picked_pos))
        remaining_weights.pop(picked_pos)
    return chosen


def _manifest_rows_from_ranked(
    ranked: pd.DataFrame,
    *,
    reference_game_id: str,
    treatment_name: str,
    slots: List[str],
    seed: int,
    summary_pool_path: str,
) -> List[Dict[str, Any]]:
    chosen_positions = _weighted_sample_positions(
        ranked["source_weight"].tolist(),
        len(slots),
        seed,
        reference_game_id,
    )
    rows: List[Dict[str, Any]] = []
    for slot_idx, (slot_name, pos) in enumerate(zip(slots, chosen_positions), start=1):
        rec = ranked.iloc[pos]
        rows.append(
            {
                "target_gameId": reference_game_id,
                "target_treatmentName": treatment_name,
                "target_playerId": slot_name,
                "target_player_slot": int(slot_idx),
                "archetype_mode": CONFIG_BANK_MODE,
                "summary_pool_path": summary_pool_path,
                "assignment_seed": int(seed),
                "source_gameId": str(rec["experiment"]),
                "source_playerId": str(rec["participant"]),
                "source_text": str(rec["text"]),
                "source_score": float(rec["source_score"]),
                "source_weight": float(rec["source_weight"]),
                "source_rank": int(rec["source_rank"]),
                "bank_size": int(rec["bank_size"]),
                "temperature": float(rec["temperature"]),
                "target_scope": "validation_treatment_reference_game",
            }
        )
    return rows


def build_reports(args: argparse.Namespace) -> Dict[str, str]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = SoftBankSummarySampler(
        summary_pool_path=str(args.summary_pool),
        temperature=float(args.temperature),
    )
    treatment_df = build_validation_treatment_contexts(str(args.val_player_table))

    single_sample_rows: List[Dict[str, Any]] = []
    distribution_rows: List[Dict[str, Any]] = []
    resample_summary_rows: List[Dict[str, Any]] = []
    top_persona_rows: List[Dict[str, Any]] = []

    for _, row in treatment_df.iterrows():
        treatment_name = str(row["CONFIG_treatmentName"])
        reference_game_id = str(row["reference_game_id"])
        env = _treatment_env(row)
        player_count = int(row["CONFIG_playerCount"])
        slots = _player_slots(player_count)

        ranked = sampler.score_bank(env)
        single_sample_rows.extend(
            _manifest_rows_from_ranked(
                ranked,
                reference_game_id=reference_game_id,
                treatment_name=treatment_name,
                slots=slots,
                seed=int(args.seed),
                summary_pool_path=str(args.summary_pool),
            )
        )
        distribution_frame = ranked[
            [
                "experiment",
                "participant",
                "source_score",
                "source_weight",
                "source_rank",
                "bank_size",
                "temperature",
            ]
        ].copy()
        distribution_frame["target_treatmentName"] = treatment_name
        distribution_frame["reference_game_id"] = reference_game_id
        distribution_rows.extend(distribution_frame.to_dict(orient="records"))

        selection_counts: Dict[Tuple[str, str], int] = {}
        for draw_idx in range(int(args.n_samples)):
            for rec in _manifest_rows_from_ranked(
                ranked,
                reference_game_id=reference_game_id,
                treatment_name=treatment_name,
                slots=slots,
                seed=int(args.seed) + draw_idx,
                summary_pool_path=str(args.summary_pool),
            ):
                key = (str(rec["source_gameId"]), str(rec["source_playerId"]))
                selection_counts[key] = selection_counts.get(key, 0) + 1

        effective_support = float(1.0 / ranked["source_weight"].pow(2).sum())
        max_weight = float(ranked["source_weight"].max())
        top_selection_rate = 0.0
        if selection_counts:
            top_selection_rate = max(selection_counts.values()) / float(args.n_samples)
        resample_summary_rows.append(
            {
                "CONFIG_treatmentName": treatment_name,
                "reference_game_id": reference_game_id,
                "player_count": player_count,
                "n_samples": int(args.n_samples),
                "bank_size": int(len(ranked)),
                "effective_support": effective_support,
                "max_weight": max_weight,
                "top_selected_persona_rate": top_selection_rate,
                "num_personas_selected_at_least_once": int(len(selection_counts)),
            }
        )

        ranked_lookup = {
            (str(rec["experiment"]), str(rec["participant"])): rec
            for rec in ranked.to_dict(orient="records")
        }
        top_selected = sorted(
            selection_counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )[: int(args.top_k_personas)]
        for (source_game_id, source_player_id), count in top_selected:
            ranked_rec = ranked_lookup[(source_game_id, source_player_id)]
            top_persona_rows.append(
                {
                    "CONFIG_treatmentName": treatment_name,
                    "reference_game_id": reference_game_id,
                    "selection_rate": float(count) / float(args.n_samples),
                    "selection_count": int(count),
                    "source_gameId": source_game_id,
                    "source_playerId": source_player_id,
                    "source_rank": int(ranked_rec["source_rank"]),
                    "source_score": float(ranked_rec["source_score"]),
                    "source_weight": float(ranked_rec["source_weight"]),
                    "source_text": str(ranked_rec["text"]),
                }
            )

    single_sample_path = output_dir / f"validation_treatment_single_sample_seed{int(args.seed)}.jsonl"
    distribution_path = output_dir / "validation_treatment_bank_distribution.csv"
    resample_summary_path = output_dir / "validation_treatment_resample_summary.csv"
    top_personas_path = output_dir / "validation_treatment_top_personas.jsonl"

    _write_jsonl(single_sample_path, single_sample_rows)
    pd.DataFrame(distribution_rows).to_csv(distribution_path, index=False)
    pd.DataFrame(resample_summary_rows).to_csv(resample_summary_path, index=False)
    _write_jsonl(top_personas_path, top_persona_rows)

    manifest = {
        "summary_pool": str(args.summary_pool),
        "val_player_table": str(args.val_player_table),
        "temperature": float(args.temperature),
        "seed": int(args.seed),
        "n_samples": int(args.n_samples),
        "top_k_personas": int(args.top_k_personas),
        "outputs": {
            "single_sample_jsonl": str(single_sample_path),
            "bank_distribution_csv": str(distribution_path),
            "resample_summary_csv": str(resample_summary_path),
            "top_personas_jsonl": str(top_personas_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "single_sample_jsonl": str(single_sample_path),
        "bank_distribution_csv": str(distribution_path),
        "resample_summary_csv": str(resample_summary_path),
        "top_personas_jsonl": str(top_personas_path),
        "manifest_json": str(manifest_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build validation-treatment soft-bank archetype reports.")
    parser.add_argument(
        "--summary-pool",
        type=Path,
        default=Path(DEFAULT_LEARN_SUMMARY_POOL),
        help="Finished learn-wave archetype summary pool used as the soft-bank source.",
    )
    parser.add_argument(
        "--val-player-table",
        type=Path,
        default=Path(DEFAULT_VAL_PLAYER_TABLE),
        help="Validation player-game table used to enumerate the 40 treatment configs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Softmax temperature for bank weights.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed used for the one-shot treatment sample.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of repeated treatment resamples for the empirical selection summary.",
    )
    parser.add_argument(
        "--top-k-personas",
        type=int,
        default=20,
        help="How many highest-frequency personas per treatment to keep in the top-personas JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Persona/archetype_sampling/outputs/validation_config_bank"),
        help="Directory for report artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    outputs = build_reports(parse_args())
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
