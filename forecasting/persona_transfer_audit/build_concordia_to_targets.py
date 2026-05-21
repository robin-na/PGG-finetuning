"""Build persona-to-player matching batches from Concordia persona JSON."""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.persona_transfer_audit import build_argyle_to_targets as argyle_builder
from forecasting.persona_transfer_audit import build_twin_to_chip_bargain as chip_builder
from forecasting.persona_transfer_audit import build_twin_to_pgg_pilot as pgg_builder


DEFAULT_OUTPUT_ROOT = THIS_DIR
DEFAULT_PGG_SOURCE_RUN = (
    THIS_DIR / "metadata" / "twin_direct_summary_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2"
)
DEFAULT_CHIP_SOURCE_RUN = (
    THIS_DIR / "metadata" / "twin_direct_summary_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2"
)

SYSTEM_PROMPT = (
    "You are behaving as a person with the given profile. Identify which player in the "
    "provided social interaction matches most closely with your personality."
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _unique_values(rows: list[dict[str, str]], key: str) -> list[str]:
    seen = set()
    values = []
    for row in rows:
        value = str(row[key])
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def _load_concordia_personas(path: Path, condition: str, profile_mode: str) -> list[dict[str, Any]]:
    raw = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of Concordia personas in {path}")
    personas = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Persona {index} is not a JSON object.")
        profile_payload = _profile_payload(item, profile_mode)
        raw_profile = json.dumps(profile_payload, indent=2, ensure_ascii=False, sort_keys=True)
        name = str(item.get("name") or item.get("characteristics", {}).get("name") or f"persona_{index}")
        source_key = f"{condition}:{index:03d}:{name}"
        personas.append(
            {
                "persona_pid": "concordia_" + hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:12],
                "persona_source": "concordia_persona_generator",
                "condition": condition,
                "profile_mode": profile_mode,
                "persona_index": index,
                "source_key": source_key,
                "name": name,
                "raw_profile": raw_profile,
                "raw_profile_chars": len(raw_profile),
                "raw_profile_sha1": hashlib.sha1(raw_profile.encode("utf-8")).hexdigest(),
            }
        )
    return personas


def _profile_payload(item: dict[str, Any], profile_mode: str) -> dict[str, Any]:
    if profile_mode == "compact":
        return _compact_profile_payload(item)

    payload = copy.deepcopy(item)
    characteristics = payload.get("characteristics")
    if isinstance(characteristics, dict):
        # Concordia adds the generation context to each persona before memory
        # generation. It is not part of the persona profile and includes
        # generator instructions, so keep it out of the matching prompt.
        characteristics.pop("initial_context", None)
    return payload


def _compact_profile_payload(item: dict[str, Any]) -> dict[str, Any]:
    characteristics = item.get("characteristics")
    if not isinstance(characteristics, dict):
        characteristics = {}

    raw_name = str(item.get("name") or characteristics.get("name") or "").strip()
    compact_name = _compact_name(raw_name)
    profile = {
        "core_motivation": characteristics.get("core_motivation", ""),
        "defining_experience": characteristics.get("defining_experience", ""),
        "description": characteristics.get("description", ""),
    }
    return {
        "name": compact_name,
        "profile": {key: value for key, value in profile.items() if value},
    }


def _compact_name(raw_name: str) -> str:
    name = raw_name.strip()
    if not name:
        return ""
    name = name.split("(", 1)[0].strip()
    name = re.split(r"\s+the\s+", name, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    name = name.split(",", 1)[0].strip()
    return name or raw_name.strip()


def _render_profile(persona: dict[str, Any], max_chars: int) -> str:
    text = str(persona["raw_profile"]).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[TRUNCATED]"


def _render_pgg_user_prompt(
    persona: dict[str, Any],
    gold_row: dict[str, Any],
    metadata: Any,
    top_k: int,
    max_profile_chars: int,
) -> str:
    players = list(gold_row["players"])
    rendered_top_k = min(top_k, len(players))
    return "\n\n".join(
        [
            "Below is information about yourself.",
            _render_profile(persona, max_profile_chars),
            pgg_builder._render_game_rules(metadata, players),
            gold_row["gold_continuation_text"],
            (
                f"Rank up to {rendered_top_k} observed players by how closely their revealed behavior matches "
                "your personality, the behavior you would most plausibly have produced in this game."
            ),
            (
                "Return only valid JSON. Put the closest match first. Assign each listed player a probability "
                "from 0 to 1, and make the probabilities across the listed players sum to exactly 1. "
                "Any player not listed is treated as having probability 0."
            ),
        ]
    )


def _render_chip_user_prompt(
    persona: dict[str, Any],
    target: dict[str, Any],
    top_k: int,
    max_profile_chars: int,
) -> str:
    players = list(target["players"])
    rendered_top_k = min(top_k, len(players))
    return "\n\n".join(
        [
            "Below is information about yourself.",
            _render_profile(persona, max_profile_chars),
            chip_builder._render_game_context(target),
            chip_builder._render_transcript(target),
            (
                f"Rank up to {rendered_top_k} observed players by how closely their revealed behavior matches "
                "your personality, the behavior you would most plausibly have produced in this game."
            ),
            (
                "Return only valid JSON. Put the closest match first. Assign each listed player a probability "
                "from 0 to 1, and make the probabilities across the listed players sum to exactly 1. "
                "Any player not listed is treated as having probability 0."
            ),
        ]
    )


def _manifest_persona_fields(persona: dict[str, Any]) -> dict[str, Any]:
    return {
        "persona_pid": persona["persona_pid"],
        "persona_source": persona["persona_source"],
        "condition": persona["condition"],
        "profile_mode": persona["profile_mode"],
        "persona_index": persona["persona_index"],
        "source_key": persona["source_key"],
        "name": persona["name"],
        "raw_profile_chars": persona["raw_profile_chars"],
        "raw_profile_sha1": persona["raw_profile_sha1"],
    }


def _default_run_name(args: argparse.Namespace, target: str) -> str:
    game_part = "pgg_stratified_32x40" if target == "pgg" else "chip_bargain_stratified_32x48"
    condition = args.condition
    if args.profile_mode != "full":
        condition = f"{condition}_{args.profile_mode}"
    return f"{condition}_to_{game_part}_top{args.top_k}_{args.model.replace('-', '_')}"


def _custom_id_prefix(args: argparse.Namespace, target: str) -> str:
    if args.profile_mode == "full":
        return f"concordia_{target}_match"
    return f"concordia_{args.profile_mode}_{target}_match"


def _build_pgg(args: argparse.Namespace, personas: list[dict[str, Any]]) -> dict[str, Any]:
    source_run = args.pgg_source_run.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_name = args.pgg_run_name or _default_run_name(args, "pgg")
    metadata_dir = output_root / "metadata" / run_name
    batch_input_file = output_root / "batch_input" / f"{run_name}.jsonl"
    expected_batch_output_file = output_root / "batch_output" / f"{run_name}.jsonl"

    source_manifest_rows = _read_csv(source_run / "request_manifest.csv")
    game_ids = _unique_values(source_manifest_rows, "game_id")
    source_gold = {str(row["game_id"]): row for row in _read_jsonl(args.pgg_source_gold)}
    prompt_metadata = pgg_builder._load_prompt_metadata(PROJECT_ROOT / "data/processed_data/df_analysis_val.csv")

    requests = []
    manifest_rows = []
    for persona_index, persona in enumerate(personas, start=1):
        for game_index, game_id in enumerate(game_ids, start=1):
            game = source_gold[game_id]
            metadata = prompt_metadata[game_id]
            custom_id = (
                f"{_custom_id_prefix(args, 'pgg')}__pid_{persona['persona_pid']}__game_{game_id}__"
                f"p{persona_index:03d}__g{game_index:03d}"
            )
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _render_pgg_user_prompt(
                                persona,
                                game,
                                metadata,
                                args.top_k,
                                args.max_profile_chars,
                            ),
                        },
                    ],
                    "response_format": pgg_builder._response_schema(game["players"], args.top_k),
                },
            }
            if args.temperature is not None:
                request["body"]["temperature"] = args.temperature
            requests.append(request)
            manifest_rows.append(
                {
                    "custom_id": custom_id,
                    **_manifest_persona_fields(persona),
                    "game_id": game["game_id"],
                    "treatment_name": game["treatment_name"],
                    "config_id": game.get("config_id", ""),
                    "num_rounds": len(game.get("gold_rounds", [])),
                    "players": json.dumps(game["players"]),
                    "model": args.model,
                    "top_k": min(args.top_k, len(game["players"])),
                }
            )

    _write_jsonl(batch_input_file, requests)
    _write_csv(metadata_dir / "request_manifest.csv", manifest_rows)
    _write_jsonl(metadata_dir / "selected_personas.jsonl", personas)
    token_summary = argyle_builder._write_token_estimates(metadata_dir, requests)
    (metadata_dir / "sample_prompt.txt").write_text(
        requests[0]["body"]["messages"][1]["content"] if requests else "",
        encoding="utf-8",
    )
    manifest = {
        "dataset_key": "concordia_persona_generator_transfer_audit_pgg",
        "run_name": run_name,
        "created_at": _utc_now_iso(),
        "model": args.model,
        "source_persona_library": "Concordia Persona Generators",
        "condition": args.condition,
        "profile_mode": args.profile_mode,
        "personas_json": str(args.personas_json.expanduser().resolve()),
        "num_personas": len(personas),
        "num_games": len(game_ids),
        "num_requests": len(requests),
        "source_run": str(source_run),
        "source_gold": str(args.pgg_source_gold.expanduser().resolve()),
        "top_k": args.top_k,
        "max_profile_chars": args.max_profile_chars,
        "batch_input_file": str(batch_input_file),
        "expected_batch_output_file": str(expected_batch_output_file),
        "metadata_dir": str(metadata_dir),
        "request_manifest": str(metadata_dir / "request_manifest.csv"),
        "selected_personas": str(metadata_dir / "selected_personas.jsonl"),
        "sample_prompt": str(metadata_dir / "sample_prompt.txt"),
        "request_token_estimates": str(metadata_dir / "request_token_estimates.csv"),
        "token_summary": token_summary,
    }
    (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "run_name": run_name,
        "num_requests": len(requests),
        "batch_input_file": str(batch_input_file),
        "metadata_dir": str(metadata_dir),
        "token_summary": token_summary,
    }


def _build_chip(args: argparse.Namespace, personas: list[dict[str, Any]]) -> dict[str, Any]:
    source_run = args.chip_source_run.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_name = args.chip_run_name or _default_run_name(args, "chip")
    metadata_dir = output_root / "metadata" / run_name
    batch_input_file = output_root / "batch_input" / f"{run_name}.jsonl"
    expected_batch_output_file = output_root / "batch_output" / f"{run_name}.jsonl"

    source_manifest_by_record = {}
    for row in _read_csv(source_run / "request_manifest.csv"):
        source_manifest_by_record.setdefault(str(row["record_id"]), row)
    gold_by_record = {}
    for row in _read_jsonl(source_run / "gold_targets.jsonl"):
        gold_by_record.setdefault(str(row["record_id"]), row)

    requests = []
    manifest_rows = []
    gold_rows = []
    for persona_index, persona in enumerate(personas, start=1):
        for game_index, record_id in enumerate(gold_by_record, start=1):
            gold_row = gold_by_record[record_id]
            target = gold_row["gold_target"]
            source_manifest = source_manifest_by_record[record_id]
            custom_id = (
                f"{_custom_id_prefix(args, 'chip')}__pid_{persona['persona_pid']}__record_{record_id}__"
                f"p{persona_index:03d}__g{game_index:03d}"
            )
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _render_chip_user_prompt(
                                persona,
                                target,
                                args.top_k,
                                args.max_profile_chars,
                            ),
                        },
                    ],
                    "response_format": chip_builder._response_schema(list(target["players"]), args.top_k),
                },
            }
            if args.temperature is not None:
                request["body"]["temperature"] = args.temperature
            requests.append(request)
            manifest_rows.append(
                {
                    "custom_id": custom_id,
                    **_manifest_persona_fields(persona),
                    "record_id": record_id,
                    "unit_id": source_manifest.get("unit_id", ""),
                    "treatment_name": source_manifest.get("treatment_name", ""),
                    "chip_family": source_manifest.get("chip_family", ""),
                    "chip_family_display": source_manifest.get("chip_family_display", ""),
                    "cohort_name": source_manifest.get("cohort_name", ""),
                    "stage_name": source_manifest.get("stage_name", ""),
                    "stage_code": source_manifest.get("stage_code", ""),
                    "experiment_name": source_manifest.get("experiment_name", ""),
                    "players": json.dumps(target["players"]),
                    "model": args.model,
                    "top_k": min(args.top_k, len(target["players"])),
                }
            )
            gold_rows.append(
                {
                    "custom_id": custom_id,
                    "record_id": record_id,
                    "treatment_name": source_manifest.get("treatment_name", ""),
                    "gold_target": target,
                }
            )

    _write_jsonl(batch_input_file, requests)
    _write_csv(metadata_dir / "request_manifest.csv", manifest_rows)
    _write_jsonl(metadata_dir / "selected_personas.jsonl", personas)
    _write_jsonl(metadata_dir / "gold_targets.jsonl", gold_rows)
    token_summary = argyle_builder._write_token_estimates(metadata_dir, requests)
    (metadata_dir / "sample_prompt.txt").write_text(
        requests[0]["body"]["messages"][1]["content"] if requests else "",
        encoding="utf-8",
    )
    manifest = {
        "dataset_key": "concordia_persona_generator_transfer_audit_chip_bargain",
        "run_name": run_name,
        "created_at": _utc_now_iso(),
        "model": args.model,
        "source_persona_library": "Concordia Persona Generators",
        "condition": args.condition,
        "profile_mode": args.profile_mode,
        "personas_json": str(args.personas_json.expanduser().resolve()),
        "num_personas": len(personas),
        "num_games": len(gold_by_record),
        "num_requests": len(requests),
        "source_run": str(source_run),
        "top_k": args.top_k,
        "max_profile_chars": args.max_profile_chars,
        "batch_input_file": str(batch_input_file),
        "expected_batch_output_file": str(expected_batch_output_file),
        "metadata_dir": str(metadata_dir),
        "request_manifest": str(metadata_dir / "request_manifest.csv"),
        "selected_personas": str(metadata_dir / "selected_personas.jsonl"),
        "gold_targets": str(metadata_dir / "gold_targets.jsonl"),
        "sample_prompt": str(metadata_dir / "sample_prompt.txt"),
        "request_token_estimates": str(metadata_dir / "request_token_estimates.csv"),
        "token_summary": token_summary,
    }
    (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "run_name": run_name,
        "num_requests": len(requests),
        "batch_input_file": str(batch_input_file),
        "metadata_dir": str(metadata_dir),
        "token_summary": token_summary,
    }


def run(args: argparse.Namespace) -> None:
    personas = _load_concordia_personas(args.personas_json, args.condition, args.profile_mode)
    results = []
    if args.target in {"pgg", "both"}:
        results.append(_build_pgg(args, personas))
    if args.target in {"chip", "both"}:
        results.append(_build_chip(args, personas))
    print(json.dumps({"personas_json": str(args.personas_json.expanduser().resolve()), "built": results}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas-json", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--target", choices=["pgg", "chip", "both"], default="both")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-profile-chars", type=int, default=0)
    parser.add_argument("--profile-mode", choices=["full", "compact"], default="full")
    parser.add_argument("--pgg-run-name", default=None)
    parser.add_argument("--chip-run-name", default=None)
    parser.add_argument("--pgg-source-run", type=Path, default=DEFAULT_PGG_SOURCE_RUN)
    parser.add_argument("--chip-source-run", type=Path, default=DEFAULT_CHIP_SOURCE_RUN)
    parser.add_argument("--pgg-source-gold", type=Path, default=pgg_builder.DEFAULT_SOURCE_GOLD)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
