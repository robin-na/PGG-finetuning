"""Build NVIDIA Nemotron-Personas-USA full-persona matching batches."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
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
DEFAULT_SAMPLE_FILE = THIS_DIR / "external" / "nemotron" / "nemotron_full_persona_adult_seed_2_n32.jsonl"
DEFAULT_PGG_SOURCE_RUN = (
    THIS_DIR / "metadata" / "twin_direct_summary_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2"
)
DEFAULT_CHIP_SOURCE_RUN = (
    THIS_DIR / "metadata" / "twin_direct_summary_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2"
)

HF_DATASET_ID = "nvidia/Nemotron-Personas-USA"
HF_CONFIG = "default"
HF_SPLIT = "train"
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_STATISTICS_URL = "https://datasets-server.huggingface.co/statistics"

SYSTEM_PROMPT = (
    "You are behaving as a person with the given profile. Identify which player in the "
    "provided social interaction matches most closely with your personality."
)

NARRATIVE_FIELDS: list[tuple[str, str]] = [
    ("persona", ""),
    ("professional_persona", ""),
    ("cultural_background", ""),
    ("skills_and_expertise", ""),
    ("skills_and_expertise_list", "Specific skills include:"),
    ("hobbies_and_interests", ""),
    ("hobbies_and_interests_list", "Specific hobbies and interests include:"),
    ("sports_persona", ""),
    ("arts_persona", ""),
    ("travel_persona", ""),
    ("culinary_persona", ""),
    ("career_goals_and_ambitions", ""),
]
RAW_PROFILE_FIELDS = [
    "age",
    "sex",
    "marital_status",
    "education_level",
    "bachelors_field",
    "occupation",
    "city",
    "state",
    "zipcode",
    "country",
    "persona",
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "cultural_background",
    "skills_and_expertise",
    "skills_and_expertise_list",
    "hobbies_and_interests",
    "hobbies_and_interests_list",
    "career_goals_and_ambitions",
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


def _fetch_json(url: str, params: dict[str, Any], retries: int = 8) -> dict[str, Any]:
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(full_url, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt >= retries:
                raise
            retry_after = error.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = float(retry_after)
            else:
                delay = min(60.0, 2.0 * (2**attempt))
            time.sleep(delay)
    raise RuntimeError(f"Unable to fetch {full_url}")


def _num_examples() -> int:
    data = _fetch_json(
        HF_STATISTICS_URL,
        {"dataset": HF_DATASET_ID, "config": HF_CONFIG, "split": HF_SPLIT},
    )
    return int(data["num_examples"])


def _fetch_row(offset: int, delay_seconds: float) -> dict[str, Any]:
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    data = _fetch_json(
        HF_ROWS_URL,
        {
            "dataset": HF_DATASET_ID,
            "config": HF_CONFIG,
            "split": HF_SPLIT,
            "offset": offset,
            "length": 1,
        },
    )
    rows = data.get("rows", [])
    if not rows:
        raise ValueError(f"No Nemotron row returned for offset {offset}")
    result = dict(rows[0]["row"])
    result["_row_idx"] = int(rows[0]["row_idx"])
    return result


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _humanize_code(value: Any) -> str:
    return _as_text(value).replace("_", " ").strip()


def _parse_list_text(value: Any) -> str:
    text = _as_text(value)
    if not text:
        return ""
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return text
    if isinstance(parsed, list):
        return ", ".join(str(item).strip() for item in parsed if str(item).strip())
    return text


def _valid_adult(row: dict[str, Any]) -> bool:
    try:
        return int(row.get("age", -1)) >= 18
    except Exception:
        return False


def _render_demographic_sentence(row: dict[str, Any]) -> str:
    age = row.get("age")
    sex = _humanize_code(row.get("sex", "")).lower()
    age_sex = []
    if isinstance(age, int) or str(age).isdigit():
        age_sex.append(f"{int(age)} years old")
    if sex:
        age_sex.append(sex)
    residence = ", ".join(
        part for part in [_humanize_code(row.get("city", "")), _humanize_code(row.get("state", "")), _humanize_code(row.get("country", ""))] if part
    )
    clauses = []
    if age_sex:
        clauses.append("You are " + " and ".join(age_sex))
    if residence:
        clauses.append(f"you live in {residence}")
    marital_status = _humanize_code(row.get("marital_status", ""))
    education = _humanize_code(row.get("education_level", ""))
    bachelors_field = _humanize_code(row.get("bachelors_field", ""))
    occupation = _humanize_code(row.get("occupation", ""))
    if marital_status:
        clauses.append(f"your marital status is {marital_status}")
    if education:
        education_text = education
        if bachelors_field:
            education_text = f"{education_text} with a background in {bachelors_field}"
        clauses.append(f"your education level is {education_text}")
    if occupation:
        clauses.append(f"your occupation is {occupation}")
    if not clauses:
        return ""
    return "; ".join(clauses) + "."


def _render_full_persona(row: dict[str, Any]) -> str:
    if row.get("persona_summary"):
        return str(row["persona_summary"]).strip()
    paragraphs = []
    demographic = _render_demographic_sentence(row)
    if demographic:
        paragraphs.append(demographic)
    for field, prefix in NARRATIVE_FIELDS:
        value = row.get(field, "")
        text = _parse_list_text(value) if field.endswith("_list") else _as_text(value)
        if not text:
            continue
        paragraphs.append(f"{prefix} {text}".strip() if prefix else text)
    return "\n\n".join(paragraphs).strip()


def _render_raw_field_profile(row: dict[str, Any]) -> str:
    if row.get("raw_profile"):
        return str(row["raw_profile"]).strip()
    lines = []
    for field in RAW_PROFILE_FIELDS:
        value = row.get(field, "")
        text = "" if value is None else str(value).strip()
        if text:
            lines.append(f"{field}: {text}")
    return "\n".join(lines).strip()


def _sample_personas(args: argparse.Namespace) -> list[dict[str, Any]]:
    sample_file = args.persona_sample_file.expanduser().resolve()
    if sample_file.is_file() and not args.refresh_sample:
        return _read_jsonl(sample_file)

    total_rows = _num_examples()
    rng = random.Random(args.seed)
    selected = []
    seen_offsets = set()
    draws = 0
    while len(selected) < args.num_personas:
        offset = rng.randrange(total_rows)
        if offset in seen_offsets:
            continue
        seen_offsets.add(offset)
        row = _fetch_row(offset, args.row_fetch_delay)
        draws += 1
        if args.adults_only and not _valid_adult(row):
            continue
        profile = _render_full_persona(row)
        if not profile:
            continue
        uuid = _as_text(row.get("uuid", "")) or hashlib.sha1(str(offset).encode("utf-8")).hexdigest()
        selected.append(
            {
                "persona_pid": f"nemotron_{uuid[:12]}",
                "source_dataset": HF_DATASET_ID,
                "source_row_idx": row["_row_idx"],
                "uuid": uuid,
                "persona_summary": profile,
                "raw_profile": _render_raw_field_profile(row),
                "persona_chars": len(profile),
                "raw_profile_chars": len(_render_raw_field_profile(row)),
                "profile_sha1": hashlib.sha1(profile.encode("utf-8")).hexdigest(),
                "raw_profile_sha1": hashlib.sha1(_render_raw_field_profile(row).encode("utf-8")).hexdigest(),
                "age": row.get("age", ""),
                "sex": row.get("sex", ""),
                "marital_status": row.get("marital_status", ""),
                "education_level": row.get("education_level", ""),
                "bachelors_field": row.get("bachelors_field", ""),
                "occupation": row.get("occupation", ""),
                "city": row.get("city", ""),
                "state": row.get("state", ""),
                "country": row.get("country", ""),
            }
        )
    _write_jsonl(sample_file, selected)
    lengths = [int(row["persona_chars"]) for row in selected]
    summary = {
        "created_at": _utc_now_iso(),
        "source": HF_DATASET_ID,
        "num_dataset_rows": total_rows,
        "num_sampled": len(selected),
        "seed": args.seed,
        "sampling_method": "uniform random row offsets from the Hugging Face dataset-server train split",
        "sampling_filter": "age >= 18" if args.adults_only else "none",
        "draws_until_sample_complete": draws,
        "sampled_persona_chars": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "mean": statistics.mean(lengths),
            "max": max(lengths),
        },
        "sampled_raw_profile_chars": {
            "min": min(int(row["raw_profile_chars"]) for row in selected),
            "median": statistics.median(int(row["raw_profile_chars"]) for row in selected),
            "mean": statistics.mean(int(row["raw_profile_chars"]) for row in selected),
            "max": max(int(row["raw_profile_chars"]) for row in selected),
        },
    }
    sample_file.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return selected


def _unique_values(rows: list[dict[str, str]], key: str) -> list[str]:
    seen = set()
    values = []
    for row in rows:
        value = str(row[key])
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


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
            _render_raw_field_profile(persona),
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
            _render_raw_field_profile(persona),
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
        "persona_source": "nemotron_personas_usa_full_profile",
        "source_dataset": persona["source_dataset"],
        "source_row_idx": persona["source_row_idx"],
        "uuid": persona["uuid"],
        "persona_chars": persona["persona_chars"],
        "raw_profile_chars": persona.get("raw_profile_chars", ""),
        "profile_sha1": persona["profile_sha1"],
        "raw_profile_sha1": persona.get("raw_profile_sha1", ""),
        "age": persona.get("age", ""),
        "sex": persona.get("sex", ""),
        "marital_status": persona.get("marital_status", ""),
        "education_level": persona.get("education_level", ""),
        "bachelors_field": persona.get("bachelors_field", ""),
        "occupation": persona.get("occupation", ""),
        "city": persona.get("city", ""),
        "state": persona.get("state", ""),
        "country": persona.get("country", ""),
    }


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
                f"nemotron_pgg_match__pid_{persona['persona_pid']}__game_{game_id}__"
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
                        {"role": "user", "content": _render_pgg_user_prompt(persona, game, metadata, args.top_k)},
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
        "dataset_key": "nemotron_personas_usa_transfer_audit_pgg",
        "run_name": args.pgg_run_name,
        "created_at": _utc_now_iso(),
        "model": args.model,
        "seed": args.seed,
        "source_persona_library": HF_DATASET_ID,
        "persona_sample_file": str(args.persona_sample_file.expanduser().resolve()),
        "sampling_filter": "age >= 18" if args.adults_only else "none",
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
        "selected_personas": str(metadata_dir / "selected_personas.jsonl"),
        "sample_prompt": str(metadata_dir / "sample_prompt.txt"),
        "request_token_estimates": str(metadata_dir / "request_token_estimates.csv"),
        "token_summary": token_summary,
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
                f"nemotron_chip_match__pid_{persona['persona_pid']}__record_{record_id}__"
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
                        {"role": "user", "content": _render_chip_user_prompt(persona, target, args.top_k)},
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
        "dataset_key": "nemotron_personas_usa_transfer_audit_chip_bargain",
        "run_name": args.chip_run_name,
        "created_at": _utc_now_iso(),
        "model": args.model,
        "seed": args.seed,
        "source_persona_library": HF_DATASET_ID,
        "persona_sample_file": str(args.persona_sample_file.expanduser().resolve()),
        "sampling_filter": "age >= 18" if args.adults_only else "none",
        "num_personas": len(personas),
        "num_games": len(gold_by_record),
        "num_requests": len(requests),
        "source_run": str(source_run),
        "top_k": args.top_k,
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
        "run_name": args.chip_run_name,
        "num_requests": len(requests),
        "batch_input_file": str(batch_input_file),
        "metadata_dir": str(metadata_dir),
        "token_summary": token_summary,
    }


def run(args: argparse.Namespace) -> None:
    personas = _sample_personas(args)
    results = []
    if args.target in {"pgg", "both"}:
        results.append(_build_pgg(args, personas))
    if args.target in {"chip", "both"}:
        results.append(_build_chip(args, personas))
    print(json.dumps({"sample_file": str(args.persona_sample_file.expanduser().resolve()), "built": results}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["pgg", "chip", "both"], default="both")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--persona-sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--refresh-sample", action="store_true")
    parser.add_argument("--include-minors", action="store_true")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--row-fetch-delay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--num-personas", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--pgg-run-name",
        default="nemotron_full_persona_adult_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2",
    )
    parser.add_argument(
        "--chip-run-name",
        default="nemotron_full_persona_adult_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2",
    )
    parser.add_argument("--pgg-source-run", type=Path, default=DEFAULT_PGG_SOURCE_RUN)
    parser.add_argument("--chip-source-run", type=Path, default=DEFAULT_CHIP_SOURCE_RUN)
    parser.add_argument("--pgg-source-gold", type=Path, default=pgg_builder.DEFAULT_SOURCE_GOLD)
    args = parser.parse_args()
    args.adults_only = not args.include_minors
    return args


if __name__ == "__main__":
    run(parse_args())
