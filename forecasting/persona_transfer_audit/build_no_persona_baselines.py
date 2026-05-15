"""Build no-persona baseline matching batches for PGG and chip bargaining."""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_PGG_SOURCE_RUN = (
    THIS_DIR / "metadata" / "twin_direct_summary_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2"
)
DEFAULT_CHIP_SOURCE_RUN = (
    THIS_DIR / "metadata" / "twin_direct_summary_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2"
)

NO_PERSONA_SYSTEM_PROMPT = (
    "You are an online participant of an experiment. Identify which player in the provided social interaction "
    "matches most closely with how you would behave."
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
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
    body = request["body"]
    encoding = _token_encoding(str(body["model"]))
    if encoding is None:
        return None
    total = 0
    for message in body.get("messages", []):
        total += 4
        total += len(encoding.encode(str(message.get("role", ""))))
        total += len(encoding.encode(str(message.get("content", ""))))
    total += 2
    total += len(encoding.encode(json.dumps(body.get("response_format", {}), ensure_ascii=False)))
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

    if tiktoken_available:
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

    return {
        "method": "tiktoken unavailable; see request_token_estimates.json for char/4 estimate",
        "num_requests": len(requests),
    }


def _render_pgg_user_prompt(gold_row: dict[str, Any], metadata: Any, top_k: int) -> str:
    players = list(gold_row["players"])
    rendered_top_k = min(top_k, len(players))
    return "\n\n".join(
        [
            pgg_builder._render_game_rules(metadata, players),
            gold_row["gold_continuation_text"],
            (
                f"Rank up to {rendered_top_k} observed players by how closely their revealed behavior matches "
                "the behavior you would most plausibly have produced in this game."
            ),
            (
                "Return only valid JSON. Put the closest match first. Assign each listed player a probability "
                "from 0 to 1, and make the probabilities across the listed players sum to exactly 1. "
                "Any player not listed is treated as having probability 0."
            ),
        ]
    )


def _render_chip_user_prompt(target: dict[str, Any], top_k: int) -> str:
    players = list(target["players"])
    rendered_top_k = min(top_k, len(players))
    return "\n\n".join(
        [
            chip_builder._render_game_context(target),
            chip_builder._render_transcript(target),
            (
                f"Rank up to {rendered_top_k} observed players by how closely their revealed behavior matches "
                "the behavior you would most plausibly have produced in this game."
            ),
            (
                "Return only valid JSON. Put the closest match first. Assign each listed player a probability "
                "from 0 to 1, and make the probabilities across the listed players sum to exactly 1. "
                "Any player not listed is treated as having probability 0."
            ),
        ]
    )


def _build_pgg(args: argparse.Namespace) -> dict[str, Any]:
    source_run = args.pgg_source_run.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_name = args.pgg_run_name
    metadata_dir = output_root / "metadata" / run_name
    batch_input_file = output_root / "batch_input" / f"{run_name}.jsonl"
    expected_batch_output_file = output_root / "batch_output" / f"{run_name}.jsonl"

    source_manifest_rows = _read_csv(source_run / "request_manifest.csv")
    game_ids = _unique_values(source_manifest_rows, "game_id")
    source_gold = {str(row["game_id"]): row for row in _read_jsonl(args.pgg_source_gold)}
    prompt_metadata = pgg_builder._load_prompt_metadata(PROJECT_ROOT / "data/processed_data/df_analysis_val.csv")

    requests = []
    manifest_rows = []
    for game_index, game_id in enumerate(game_ids, start=1):
        game = source_gold[game_id]
        metadata = prompt_metadata[game_id]
        custom_id = f"no_persona_pgg_match__game_{game_id}__g{game_index:03d}"
        user_prompt = _render_pgg_user_prompt(game, metadata, args.top_k)
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": NO_PERSONA_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
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
                "persona_pid": "no_persona",
                "persona_source": "none",
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
    token_summary = _write_token_estimates(metadata_dir, requests)
    (metadata_dir / "sample_prompt.txt").write_text(
        requests[0]["body"]["messages"][1]["content"] if requests else "",
        encoding="utf-8",
    )
    manifest = {
        "dataset_key": "no_persona_transfer_audit_pgg",
        "run_name": run_name,
        "created_at": _utc_now_iso(),
        "model": args.model,
        "num_personas": 0,
        "num_games": len(game_ids),
        "num_requests": len(requests),
        "source_run": str(source_run),
        "source_gold": str(args.pgg_source_gold.expanduser().resolve()),
        "top_k": args.top_k,
        "batch_input_file": str(batch_input_file),
        "expected_batch_output_file": str(expected_batch_output_file),
        "metadata_dir": str(metadata_dir),
        "request_manifest": str(metadata_dir / "request_manifest.csv"),
        "sample_prompt": str(metadata_dir / "sample_prompt.txt"),
        "request_token_estimates": str(metadata_dir / "request_token_estimates.csv"),
    }
    (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "run_name": run_name,
        "num_requests": len(requests),
        "batch_input_file": str(batch_input_file),
        "metadata_dir": str(metadata_dir),
        "token_summary": token_summary,
    }


def _build_chip(args: argparse.Namespace) -> dict[str, Any]:
    source_run = args.chip_source_run.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_name = args.chip_run_name
    metadata_dir = output_root / "metadata" / run_name
    batch_input_file = output_root / "batch_input" / f"{run_name}.jsonl"
    expected_batch_output_file = output_root / "batch_output" / f"{run_name}.jsonl"

    source_manifest_by_record = {}
    for row in _read_csv(source_run / "request_manifest.csv"):
        source_manifest_by_record.setdefault(str(row["record_id"]), row)

    gold_by_record = {}
    for row in _read_jsonl(source_run / "gold_targets.jsonl"):
        gold_by_record.setdefault(str(row["record_id"]), row)
    record_ids = list(gold_by_record)

    requests = []
    manifest_rows = []
    gold_rows = []
    for game_index, record_id in enumerate(record_ids, start=1):
        gold_row = gold_by_record[record_id]
        target = gold_row["gold_target"]
        source_manifest = source_manifest_by_record[record_id]
        custom_id = f"no_persona_chip_match__record_{record_id}__g{game_index:03d}"
        user_prompt = _render_chip_user_prompt(target, args.top_k)
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": NO_PERSONA_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
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
                "persona_pid": "no_persona",
                "persona_source": "none",
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
    _write_jsonl(metadata_dir / "gold_targets.jsonl", gold_rows)
    token_summary = _write_token_estimates(metadata_dir, requests)
    (metadata_dir / "sample_prompt.txt").write_text(
        requests[0]["body"]["messages"][1]["content"] if requests else "",
        encoding="utf-8",
    )
    manifest = {
        "dataset_key": "no_persona_transfer_audit_chip_bargain",
        "run_name": run_name,
        "created_at": _utc_now_iso(),
        "model": args.model,
        "num_personas": 0,
        "num_games": len(record_ids),
        "num_requests": len(requests),
        "source_run": str(source_run),
        "top_k": args.top_k,
        "batch_input_file": str(batch_input_file),
        "expected_batch_output_file": str(expected_batch_output_file),
        "metadata_dir": str(metadata_dir),
        "request_manifest": str(metadata_dir / "request_manifest.csv"),
        "gold_targets": str(metadata_dir / "gold_targets.jsonl"),
        "sample_prompt": str(metadata_dir / "sample_prompt.txt"),
        "request_token_estimates": str(metadata_dir / "request_token_estimates.csv"),
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
    results = []
    if args.target in {"pgg", "both"}:
        results.append(_build_pgg(args))
    if args.target in {"chip", "both"}:
        results.append(_build_chip(args))
    print(json.dumps({"built": results}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["pgg", "chip", "both"], default="both")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--pgg-run-name",
        default="no_persona_to_pgg_stratified_40_top3_gpt_5_mini_seed_2",
    )
    parser.add_argument(
        "--chip-run-name",
        default="no_persona_to_chip_bargain_stratified_48_top3_gpt_5_mini_seed_2",
    )
    parser.add_argument("--pgg-source-run", type=Path, default=DEFAULT_PGG_SOURCE_RUN)
    parser.add_argument("--chip-source-run", type=Path, default=DEFAULT_CHIP_SOURCE_RUN)
    parser.add_argument("--pgg-source-gold", type=Path, default=pgg_builder.DEFAULT_SOURCE_GOLD)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
