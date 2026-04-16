from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecasting.common.profiles import (
    load_twin_cards,
    render_pgg_persona_block,
)
from forecasting.common.runs.non_pgg import (
    ArchiveContext,
    MODEL_SLUGS,
    _estimate_input_tokens,
    _write_dataframe_csv_with_archive,
    _write_jsonl_with_archive,
    _write_text_with_archive,
)
from forecasting.datasets.chip_bargain import build_bundle as build_chip_bargain_bundle
from forecasting.prompts.chip_bargain import build_prompt


VARIANT_BASELINE = "baseline"
VARIANT_TWIN_UNADJUSTED = "twin_sampled_unadjusted_seed_0"
ALL_VARIANTS = [VARIANT_BASELINE, VARIANT_TWIN_UNADJUSTED]
ALL_MODELS = ["gpt-5.1", "gpt-5-mini"]


@dataclass(frozen=True)
class PersonaAssignment:
    profile_id: str
    headline: str
    summary: str


def _sanitize_token(value: str) -> str:
    import re

    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return sanitized.strip("_").lower()


def _batch_entry(custom_id: str, model: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
    }


def _shared_twin_cards_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "non-PGG_generalization"
        / "twin_profiles"
        / "output"
        / "twin_extended_profile_cards"
        / "chip_bargain_prompt_min"
        / "twin_extended_profile_cards.jsonl"
    )


def _shared_twin_notes_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "non-PGG_generalization"
        / "twin_profiles"
        / "output"
        / "twin_extended_profile_cards"
        / "chip_bargain_prompt_min"
        / "shared_prompt_notes.md"
    )


def _twin_assignment_path_for_variant(
    forecasting_root: Path,
    variant: str,
    assignment_dir_name: str,
) -> Path | None:
    if variant == VARIANT_TWIN_UNADJUSTED:
        return (
            forecasting_root
            / "profile_sampling"
            / "output"
            / assignment_dir_name
            / "seed_0"
            / "game_assignments.jsonl"
        )
    return None


def _load_game_assignments(assignments_path: Path) -> dict[str, dict[str, PersonaAssignment]]:
    assignments_by_game: dict[str, dict[str, PersonaAssignment]] = {}
    with assignments_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            game_id = str(row["gameId"])
            player_assignments: dict[str, PersonaAssignment] = {}
            for assignment in row.get("assignments", []):
                player_id = str(assignment["chip_bargain_player_id"])
                profile_id = str(assignment["twin_pid"])
                player_assignments[player_id] = PersonaAssignment(
                    profile_id=profile_id,
                    headline=str(assignment.get("twin_profile_headline", "")).strip(),
                    summary=str(assignment.get("twin_profile_summary", "")).strip(),
                )
            assignments_by_game[game_id] = player_assignments
    return assignments_by_game


def _build_persona_block(
    *,
    players: list[str],
    persona_assignments: dict[str, PersonaAssignment],
    twin_profile_cards: dict[str, dict[str, Any]],
    shared_prompt_notes: str,
) -> str:
    return render_pgg_persona_block(
        variant_slug=VARIANT_TWIN_UNADJUSTED,
        shared_prompt_notes=shared_prompt_notes,
        raw_player_order=players,
        avatar_order=players,
        persona_assignments=persona_assignments,
        twin_profile_cards=twin_profile_cards,
        is_demographic_only_variant=False,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chip-bargain forecasting batch inputs with multi-player baseline and Twin-unadjusted variants."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--forecasting-root", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--models", type=str, default=",".join(ALL_MODELS))
    parser.add_argument("--variants", type=str, default=",".join(ALL_VARIANTS))
    parser.add_argument("--max-records-per-treatment", type=int, default=None)
    parser.add_argument(
        "--run-suffix",
        type=str,
        default="",
        help="Optional suffix appended to each run name, e.g. mechanism_v2.",
    )
    parser.add_argument(
        "--assignment-dir-name",
        type=str,
        default="twin_to_chip_bargain_player_sampling_unadjusted",
        help="Profile-sampling output directory name under forecasting/chip_bargain/profile_sampling/output/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown_models = [model for model in models if model not in MODEL_SLUGS]
    if unknown_models:
        raise ValueError(f"Unsupported model names: {unknown_models}")
    unsupported_variants = [variant for variant in variants if variant not in ALL_VARIANTS]
    if unsupported_variants:
        raise ValueError(f"Unsupported variant names: {unsupported_variants}")

    run_suffix = _sanitize_token(args.run_suffix) if args.run_suffix else ""
    forecasting_root = args.forecasting_root
    archive_context = ArchiveContext(base_root=forecasting_root)
    batch_input_dir = forecasting_root / "batch_input"
    batch_output_dir = forecasting_root / "batch_output"
    metadata_root = forecasting_root / "metadata"
    results_root = forecasting_root / "results"
    for directory in [batch_input_dir, batch_output_dir, metadata_root, results_root]:
        directory.mkdir(parents=True, exist_ok=True)

    bundle = build_chip_bargain_bundle(args.repo_root)
    if args.max_records_per_treatment is not None:
        sampled_groups: list[pd.DataFrame] = []
        for _, group in bundle.records.groupby("treatment_name", sort=True):
            if len(group) <= args.max_records_per_treatment:
                sampled_groups.append(group.copy())
            else:
                sampled_groups.append(group.sample(n=args.max_records_per_treatment, random_state=0))
        records = (
            pd.concat(sampled_groups, ignore_index=True)
            .sort_values(["treatment_name", "record_id"])
            .reset_index(drop=True)
        )
    else:
        records = bundle.records.copy()

    twin_cards = load_twin_cards(_shared_twin_cards_path(args.repo_root))
    shared_prompt_notes = _shared_twin_notes_path(args.repo_root).read_text(encoding="utf-8").strip()

    variant_assignments: dict[str, dict[str, dict[str, PersonaAssignment]]] = {}
    for variant in variants:
        assignments_path = _twin_assignment_path_for_variant(
            forecasting_root,
            variant,
            args.assignment_dir_name,
        )
        if assignments_path is not None:
            if not assignments_path.exists():
                raise FileNotFoundError(
                    f"Missing chip-bargain assignment file for {variant}: {assignments_path}. "
                    "Run forecasting/chip_bargain/profile_sampling/sample_twin_personas_for_chip_bargain.py first."
                )
            variant_assignments[variant] = _load_game_assignments(assignments_path)
        else:
            variant_assignments[variant] = {}

    for variant in variants:
        for model in models:
            base_run_name = f"{variant}_{MODEL_SLUGS[model]}"
            run_name = f"{base_run_name}_{run_suffix}" if run_suffix else base_run_name
            metadata_dir = metadata_root / run_name
            metadata_dir.mkdir(parents=True, exist_ok=True)
            sample_dir = metadata_dir / "sample_prompts"
            sample_dir.mkdir(parents=True, exist_ok=True)

            batch_rows: list[dict[str, Any]] = []
            gold_rows: list[dict[str, Any]] = []
            request_manifest_rows: list[dict[str, Any]] = []
            token_rows: list[dict[str, Any]] = []
            sample_written = False

            for row in records.itertuples(index=False):
                record_series = pd.Series(row._asdict())
                profile_block = None
                if variant == VARIANT_TWIN_UNADJUSTED:
                    players = json.loads(str(row.players_json))
                    persona_assignments = variant_assignments[variant].get(str(row.record_id))
                    if persona_assignments is None:
                        raise ValueError(f"Missing Twin assignments for chip-bargain game {row.record_id}")
                    missing_players = [player_id for player_id in players if player_id not in persona_assignments]
                    if missing_players:
                        raise ValueError(
                            f"Game {row.record_id} is missing persona assignments for players: {missing_players}"
                        )
                    missing_profile_ids = [
                        persona_assignments[player_id].profile_id
                        for player_id in players
                        if persona_assignments[player_id].profile_id not in twin_cards
                    ]
                    if missing_profile_ids:
                        raise ValueError(
                            f"Game {row.record_id} references missing Twin cards: {sorted(set(missing_profile_ids))}"
                        )
                    profile_block = _build_persona_block(
                        players=players,
                        persona_assignments=persona_assignments,
                        twin_profile_cards=twin_cards,
                        shared_prompt_notes=shared_prompt_notes,
                    )

                system_prompt, user_prompt = build_prompt(record_series, profile_block)
                custom_id = f"chip_bargain__{_sanitize_token(str(row.record_id))}"
                batch_row = _batch_entry(custom_id, model, system_prompt, user_prompt)
                batch_rows.append(batch_row)
                gold_rows.append(
                    {
                        "custom_id": custom_id,
                        "record_id": str(row.record_id),
                        "unit_id": str(row.unit_id),
                        "treatment_name": str(row.treatment_name),
                        "gold_target": json.loads(str(row.gold_target_json)),
                    }
                )

                manifest_row = {
                    "custom_id": custom_id,
                    "record_id": str(row.record_id),
                    "unit_id": str(row.unit_id),
                    "treatment_name": str(row.treatment_name),
                    "variant": variant,
                    "model": model,
                }
                for column in records.columns:
                    if column in {"record_id", "unit_id", "treatment_name", "gold_target_json"}:
                        continue
                    manifest_row[column] = record_series[column]
                request_manifest_rows.append(manifest_row)
                token_rows.append(
                    {
                        "custom_id": custom_id,
                        "input_token_estimate": _estimate_input_tokens(batch_row["body"]["messages"]),
                    }
                )

                if not sample_written:
                    sample_text = "\n\n".join(
                        [
                            "=== SYSTEM MESSAGE ===",
                            system_prompt,
                            "=== USER MESSAGE ===",
                            user_prompt,
                        ]
                    )
                    _write_text_with_archive(sample_dir / "sample_prompt.txt", sample_text, archive_context)
                    sample_written = True

            batch_path = batch_input_dir / f"{run_name}.jsonl"
            _write_jsonl_with_archive(batch_path, batch_rows, archive_context)
            _write_jsonl_with_archive(metadata_dir / "gold_targets.jsonl", gold_rows, archive_context)
            _write_dataframe_csv_with_archive(metadata_dir / "selected_records.csv", records, archive_context)
            _write_dataframe_csv_with_archive(
                metadata_dir / "request_manifest.csv",
                pd.DataFrame(request_manifest_rows),
                archive_context,
            )
            token_df = pd.DataFrame(token_rows)
            _write_dataframe_csv_with_archive(
                metadata_dir / "request_token_estimates.csv", token_df, archive_context
            )
            token_summary = {
                "request_file": str(batch_path),
                "num_requests": int(len(token_df)),
                "total_input_tokens_estimate": int(token_df["input_token_estimate"].sum()),
                "mean_input_tokens_estimate": float(token_df["input_token_estimate"].mean()),
                "median_input_tokens_estimate": float(token_df["input_token_estimate"].median()),
                "max_input_tokens_estimate": int(token_df["input_token_estimate"].max()),
                "min_input_tokens_estimate": int(token_df["input_token_estimate"].min()),
            }
            _write_text_with_archive(
                metadata_dir / "request_token_estimates.json",
                json.dumps(token_summary, indent=2),
                archive_context,
            )

            manifest = {
                "dataset_key": "chip_bargain",
                "display_name": "Chip Bargaining",
                "run_name": run_name,
                "variant_name": variant,
                "model": model,
                "record_count": int(len(records)),
                "unit_count": int(len(records)),
                "max_records_per_treatment": args.max_records_per_treatment,
                "batch_input_file": str(batch_path),
                "expected_batch_output_file": str(batch_output_dir / f"{run_name}.jsonl"),
                "metadata_dir": str(metadata_dir),
                "profile_assignment_file": (
                    str(
                        _twin_assignment_path_for_variant(
                            forecasting_root,
                            variant,
                            args.assignment_dir_name,
                        )
                    )
                    if variant == VARIANT_TWIN_UNADJUSTED
                    else None
                ),
                "profile_cards_file": (
                    str(_shared_twin_cards_path(args.repo_root))
                    if variant == VARIANT_TWIN_UNADJUSTED
                    else None
                ),
                "profile_shared_notes_file": (
                    str(_shared_twin_notes_path(args.repo_root))
                    if variant == VARIANT_TWIN_UNADJUSTED
                    else None
                ),
            }
            _write_text_with_archive(
                metadata_dir / "manifest.json",
                json.dumps(manifest, indent=2),
                archive_context,
            )


if __name__ == "__main__":
    main()
