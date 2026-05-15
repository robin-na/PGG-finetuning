"""Build Argyle-style survey-record persona matching batches for PGG and chip."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.persona_transfer_audit import build_twin_to_chip_bargain as chip_builder
from forecasting.persona_transfer_audit import build_twin_to_pgg_pilot as pgg_builder


DEFAULT_OUTPUT_ROOT = THIS_DIR
DEFAULT_ARGYLE_2016 = (
    THIS_DIR
    / "external"
    / "argyle_out_of_one_many"
    / "full_results_2016_2.tab"
)
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

FIPS_STATE_MAP = {
    1: "Alabama",
    2: "Alaska",
    4: "Arizona",
    5: "Arkansas",
    6: "California",
    8: "Colorado",
    9: "Connecticut",
    10: "Delaware",
    12: "Florida",
    13: "Georgia",
    15: "Hawaii",
    16: "Idaho",
    17: "Illinois",
    18: "Indiana",
    19: "Iowa",
    20: "Kansas",
    21: "Kentucky",
    22: "Louisiana",
    23: "Maine",
    24: "Maryland",
    25: "Massachusetts",
    26: "Michigan",
    27: "Minnesota",
    28: "Mississippi",
    29: "Missouri",
    30: "Montana",
    31: "Nebraska",
    32: "Nevada",
    33: "New Hampshire",
    34: "New Jersey",
    35: "New Mexico",
    36: "New York",
    37: "North Carolina",
    38: "North Dakota",
    39: "Ohio",
    40: "Oklahoma",
    41: "Oregon",
    42: "Pennsylvania",
    44: "Rhode Island",
    45: "South Carolina",
    46: "South Dakota",
    47: "Tennessee",
    48: "Texas",
    49: "Utah",
    50: "Vermont",
    51: "Virginia",
    53: "Washington",
    54: "West Virginia",
    55: "Wisconsin",
    56: "Wyoming",
}

ARGYLE_2016_FIELDS: list[tuple[str, str, dict[int, str] | None]] = [
    (
        "V161310x",
        "Racially, I am XXX.",
        {1: "white", 2: "black", 3: "asian", 4: "native American", 5: "hispanic"},
    ),
    (
        "V162174",
        "XXX",
        {
            1: "I like to discuss politics with my family and friends.",
            2: "I never discuss politics with my family or friends.",
        },
    ),
    (
        "V161126",
        "Ideologically, I am XXX.",
        {
            1: "extremely liberal",
            2: "liberal",
            3: "slightly liberal",
            4: "moderate",
            5: "slightly conservative",
            6: "conservative",
            7: "extremely conservative",
        },
    ),
    (
        "V161158x",
        "Politically, I am XXX.",
        {
            1: "a strong democrat",
            2: "a weak Democrat",
            3: "an independent who leans Democratic",
            4: "an independent",
            5: "an independent who leans Republican",
            6: "a weak Republican",
            7: "a strong Republican",
        },
    ),
    ("V161244", "I XXX.", {1: "attend church", 2: "do not attend church"}),
    ("V161267", "I am XXX years old.", None),
    ("V161342", "I am a XXX.", {1: "man", 2: "woman"}),
    (
        "V162256",
        "I am XXX interested in politics.",
        {1: "very", 2: "somewhat", 3: "not very", 4: "not at all"},
    ),
    (
        "V162125x",
        "It makes me feel XXX to see the American flag.",
        {
            1: "extremely good",
            2: "moderately good",
            3: "a little good",
            4: "neither good nor bad",
            5: "a little bad",
            6: "moderately bad",
            7: "extremely bad",
        },
    ),
    ("V161010d", "I am from XXX.", FIPS_STATE_MAP),
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


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
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_int(value: str) -> int | None:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def _render_argyle_backstory(row: dict[str, str]) -> tuple[str, dict[str, Any]]:
    statements = []
    used_fields = []
    for field, template, value_map in ARGYLE_2016_FIELDS:
        value = _parse_int(row.get(field, ""))
        if value is None:
            continue
        if value_map is None:
            if 18 <= value <= 120:
                statements.append(template.replace("XXX", str(value)))
                used_fields.append(field)
        elif value in value_map:
            statements.append(template.replace("XXX", value_map[value]))
            used_fields.append(field)
    backstory = " ".join(statements).strip()
    metadata = {
        "num_backstory_statements": len(statements),
        "used_fields": "|".join(used_fields),
        "backstory_sha1": hashlib.sha1(backstory.encode("utf-8")).hexdigest(),
    }
    return backstory, metadata


def _load_argyle_personas(path: Path, min_statements: int) -> list[dict[str, Any]]:
    rows = _read_csv(path, delimiter="\t")
    personas = []
    for row in rows:
        backstory, backstory_metadata = _render_argyle_backstory(row)
        if backstory_metadata["num_backstory_statements"] < min_statements:
            continue
        source_id = str(row["V160001_orig"])
        personas.append(
            {
                "persona_pid": f"argyle_anes2016_{source_id}",
                "source_record_id": source_id,
                "persona_summary": backstory,
                **backstory_metadata,
            }
        )
    if not personas:
        raise ValueError(f"No eligible Argyle personas found in {path}")
    return personas


def _sample_rows(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if n <= 0 or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    return [rows[index] for index in sorted(rng.sample(range(len(rows)), n))]


def _unique_values(rows: list[dict[str, str]], key: str) -> list[str]:
    seen = set()
    values = []
    for row in rows:
        value = str(row[key])
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def _token_encoding(model: str) -> Any | None:
    try:
        import tiktoken
    except Exception:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("o200k_base")


def _request_input_tokens_tiktoken(request: dict[str, Any]) -> int | None:
    encoding = _token_encoding(str(request["body"]["model"]))
    if encoding is None:
        return None
    total = 0
    for message in request["body"].get("messages", []):
        total += 4
        total += len(encoding.encode(str(message.get("role", ""))))
        total += len(encoding.encode(str(message.get("content", ""))))
    total += 2
    total += len(encoding.encode(json.dumps(request["body"].get("response_format", {}), ensure_ascii=False)))
    return total


def _write_token_estimates(metadata_dir: Path, requests: list[dict[str, Any]]) -> dict[str, Any]:
    char_rows = []
    token_rows = []
    tiktoken_available = True
    for request in requests:
        input_chars = pgg_builder._request_input_char_count(request)
        char_rows.append(
            {
                "custom_id": request["custom_id"],
                "estimated_input_chars": input_chars,
                "estimated_input_tokens_char4": pgg_builder._estimate_tokens_from_chars(input_chars),
            }
        )
        input_tokens = _request_input_tokens_tiktoken(request)
        if input_tokens is None:
            tiktoken_available = False
        else:
            token_rows.append({"custom_id": request["custom_id"], "input_tokens_tiktoken": input_tokens})
    _write_csv(metadata_dir / "request_token_estimates.csv", char_rows)
    char_values = [int(row["estimated_input_tokens_char4"]) for row in char_rows]
    char_summary = {
        "method": "character_count_divided_by_4",
        "num_requests": len(char_values),
        "total_estimated_input_tokens": sum(char_values),
        "mean_estimated_input_tokens": sum(char_values) / len(char_values) if char_values else 0,
        "min_estimated_input_tokens": min(char_values) if char_values else 0,
        "max_estimated_input_tokens": max(char_values) if char_values else 0,
    }
    (metadata_dir / "request_token_estimates.json").write_text(
        json.dumps(char_summary, indent=2),
        encoding="utf-8",
    )
    if not tiktoken_available:
        return char_summary
    _write_csv(metadata_dir / "request_token_estimates_tiktoken.csv", token_rows)
    token_values = [int(row["input_tokens_tiktoken"]) for row in token_rows]
    token_summary = {
        "method": "tiktoken; o200k_base fallback; chat role/content + response_format schema with small chat framing allowance",
        "num_requests": len(token_values),
        "total_input_tokens_tiktoken": sum(token_values),
        "mean_input_tokens_tiktoken": sum(token_values) / len(token_values) if token_values else 0,
        "median_input_tokens_tiktoken": statistics.median(token_values) if token_values else 0,
        "min_input_tokens_tiktoken": min(token_values) if token_values else 0,
        "max_input_tokens_tiktoken": max(token_values) if token_values else 0,
    }
    (metadata_dir / "request_token_estimates_tiktoken.json").write_text(
        json.dumps(token_summary, indent=2),
        encoding="utf-8",
    )
    return token_summary


def _render_pgg_user_prompt(
    persona: dict[str, Any],
    gold_row: dict[str, Any],
    metadata: Any,
    top_k: int,
) -> str:
    players = list(gold_row["players"])
    rendered_top_k = min(top_k, len(players))
    return "\n\n".join(
        [
            "Below is information about yourself.",
            str(persona["persona_summary"]).strip(),
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


def _render_chip_user_prompt(persona: dict[str, Any], target: dict[str, Any], top_k: int) -> str:
    players = list(target["players"])
    rendered_top_k = min(top_k, len(players))
    return "\n\n".join(
        [
            "Below is information about yourself.",
            str(persona["persona_summary"]).strip(),
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


def _build_pgg(args: argparse.Namespace, personas: list[dict[str, Any]]) -> dict[str, Any]:
    source_run = args.pgg_source_run.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    metadata_dir = output_root / "metadata" / args.pgg_run_name
    batch_input_file = output_root / "batch_input" / f"{args.pgg_run_name}.jsonl"
    expected_batch_output_file = output_root / "batch_output" / f"{args.pgg_run_name}.jsonl"

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
                f"argyle_pgg_match__pid_{persona['persona_pid']}__game_{game_id}__"
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
                            "content": _render_pgg_user_prompt(persona, game, metadata, args.top_k),
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
                    "persona_pid": persona["persona_pid"],
                    "persona_source": "argyle_anes2016_first_person_backstory",
                    "source_record_id": persona["source_record_id"],
                    "num_backstory_statements": persona["num_backstory_statements"],
                    "backstory_sha1": persona["backstory_sha1"],
                    "game_id": game["game_id"],
                    "treatment_name": game["treatment_name"],
                    "config_id": game.get("config_id", ""),
                    "num_rounds": len(game.get("gold_rounds", [])),
                    "players": json.dumps(game["players"]),
                    "model": args.model,
                    "max_persona_chars": 0,
                    "top_k": min(args.top_k, len(game["players"])),
                }
            )

    _write_jsonl(batch_input_file, requests)
    _write_csv(metadata_dir / "request_manifest.csv", manifest_rows)
    _write_csv(metadata_dir / "selected_personas.csv", personas)
    token_summary = _write_token_estimates(metadata_dir, requests)
    (metadata_dir / "sample_prompt.txt").write_text(
        requests[0]["body"]["messages"][1]["content"] if requests else "",
        encoding="utf-8",
    )
    manifest = {
        "dataset_key": "argyle_anes2016_transfer_audit_pgg",
        "run_name": args.pgg_run_name,
        "created_at": _utc_now_iso(),
        "model": args.model,
        "seed": args.seed,
        "source_persona_library": "Argyle et al. replication data, ANES 2016 first-person backstory fields",
        "argyle_data_file": str(args.argyle_data.expanduser().resolve()),
        "num_personas": len(personas),
        "num_games": len(game_ids),
        "num_requests": len(requests),
        "source_run": str(source_run),
        "source_gold": str(args.pgg_source_gold.expanduser().resolve()),
        "top_k": args.top_k,
        "batch_input_file": str(batch_input_file),
        "expected_batch_output_file": str(expected_batch_output_file),
        "metadata_dir": str(metadata_dir),
        "request_manifest": str(metadata_dir / "request_manifest.csv"),
        "selected_personas": str(metadata_dir / "selected_personas.csv"),
        "sample_prompt": str(metadata_dir / "sample_prompt.txt"),
        "request_token_estimates": str(metadata_dir / "request_token_estimates.csv"),
    }
    (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "run_name": args.pgg_run_name,
        "num_requests": len(requests),
        "batch_input_file": str(batch_input_file),
        "metadata_dir": str(metadata_dir),
        "token_summary": token_summary,
    }


def _build_chip(args: argparse.Namespace, personas: list[dict[str, Any]]) -> dict[str, Any]:
    source_run = args.chip_source_run.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    metadata_dir = output_root / "metadata" / args.chip_run_name
    batch_input_file = output_root / "batch_input" / f"{args.chip_run_name}.jsonl"
    expected_batch_output_file = output_root / "batch_output" / f"{args.chip_run_name}.jsonl"

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
                f"argyle_chip_match__pid_{persona['persona_pid']}__record_{record_id}__"
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
                            "content": _render_chip_user_prompt(persona, target, args.top_k),
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
                    "persona_pid": persona["persona_pid"],
                    "persona_source": "argyle_anes2016_first_person_backstory",
                    "source_record_id": persona["source_record_id"],
                    "num_backstory_statements": persona["num_backstory_statements"],
                    "backstory_sha1": persona["backstory_sha1"],
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
                    "max_persona_chars": 0,
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
    _write_csv(metadata_dir / "selected_personas.csv", personas)
    _write_jsonl(metadata_dir / "gold_targets.jsonl", gold_rows)
    token_summary = _write_token_estimates(metadata_dir, requests)
    (metadata_dir / "sample_prompt.txt").write_text(
        requests[0]["body"]["messages"][1]["content"] if requests else "",
        encoding="utf-8",
    )
    manifest = {
        "dataset_key": "argyle_anes2016_transfer_audit_chip_bargain",
        "run_name": args.chip_run_name,
        "created_at": _utc_now_iso(),
        "model": args.model,
        "seed": args.seed,
        "source_persona_library": "Argyle et al. replication data, ANES 2016 first-person backstory fields",
        "argyle_data_file": str(args.argyle_data.expanduser().resolve()),
        "num_personas": len(personas),
        "num_games": len(gold_by_record),
        "num_requests": len(requests),
        "source_run": str(source_run),
        "top_k": args.top_k,
        "batch_input_file": str(batch_input_file),
        "expected_batch_output_file": str(expected_batch_output_file),
        "metadata_dir": str(metadata_dir),
        "request_manifest": str(metadata_dir / "request_manifest.csv"),
        "selected_personas": str(metadata_dir / "selected_personas.csv"),
        "gold_targets": str(metadata_dir / "gold_targets.jsonl"),
        "sample_prompt": str(metadata_dir / "sample_prompt.txt"),
        "request_token_estimates": str(metadata_dir / "request_token_estimates.csv"),
    }
    (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "run_name": args.chip_run_name,
        "num_requests": len(requests),
        "batch_input_file": str(batch_input_file),
        "metadata_dir": str(metadata_dir),
        "token_summary": token_summary,
    }


def run(args: argparse.Namespace) -> None:
    argyle_path = args.argyle_data.expanduser().resolve()
    if not argyle_path.is_file():
        raise FileNotFoundError(f"Argyle data file not found: {argyle_path}")
    persona_pool = _load_argyle_personas(argyle_path, args.min_backstory_statements)
    personas = _sample_rows(persona_pool, args.num_personas, args.seed)
    results = []
    if args.target in {"pgg", "both"}:
        results.append(_build_pgg(args, personas))
    if args.target in {"chip", "both"}:
        results.append(_build_chip(args, personas))
    print(json.dumps({"eligible_personas": len(persona_pool), "built": results}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["pgg", "chip", "both"], default="both")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--argyle-data", type=Path, default=DEFAULT_ARGYLE_2016)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--num-personas", type=int, default=32)
    parser.add_argument("--min-backstory-statements", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--pgg-run-name",
        default="argyle_anes2016_backstory_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2",
    )
    parser.add_argument(
        "--chip-run-name",
        default="argyle_anes2016_backstory_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2",
    )
    parser.add_argument("--pgg-source-run", type=Path, default=DEFAULT_PGG_SOURCE_RUN)
    parser.add_argument("--chip-source-run", type=Path, default=DEFAULT_CHIP_SOURCE_RUN)
    parser.add_argument("--pgg-source-gold", type=Path, default=pgg_builder.DEFAULT_SOURCE_GOLD)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
