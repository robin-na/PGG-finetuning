"""Build PersonaHub matching batches for PGG and chip."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
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
DEFAULT_ELITE_PREVIEW_SAMPLE_FILE = (
    THIS_DIR / "external" / "personahub" / "elite_personas_preview_unfiltered_seed_2_n32.jsonl"
)
DEFAULT_PERSONA_JSONL_SAMPLE_FILE = (
    THIS_DIR / "external" / "personahub" / "persona_jsonl_unfiltered_seed_2_n32.jsonl"
)
DEFAULT_PGG_SOURCE_RUN = (
    THIS_DIR / "metadata" / "twin_direct_summary_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2"
)
DEFAULT_CHIP_SOURCE_RUN = (
    THIS_DIR / "metadata" / "twin_direct_summary_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2"
)
HF_API_URL = "https://huggingface.co/api/datasets/proj-persona/PersonaHub"
HF_RAW_PREFIX = "https://huggingface.co/datasets/proj-persona/PersonaHub/resolve/main"
HF_PERSONA_JSONL = f"{HF_RAW_PREFIX}/persona.jsonl"

SYSTEM_PROMPT = (
    "You are behaving as a person with the given profile. Identify which player in the "
    "provided social interaction matches most closely with your personality."
)


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


def _fetch_elite_filenames() -> list[str]:
    with urllib.request.urlopen(HF_API_URL, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    files = [
        str(item["rfilename"])
        for item in data.get("siblings", [])
        if str(item.get("rfilename", "")).startswith("ElitePersonas/")
        and str(item.get("rfilename", "")).endswith(".jsonl")
    ]
    return sorted(files, key=lambda name: int(name.split("part", 1)[1].split(".", 1)[0]))


def _passes_length_filter(text: str, min_chars: int, max_chars: int) -> bool:
    if not text:
        return False
    if len(text) < min_chars:
        return False
    if max_chars > 0 and len(text) > max_chars:
        return False
    return True


def _quantile(sorted_values: list[int], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = p * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _length_summary(lengths: list[int]) -> dict[str, Any]:
    values = sorted(lengths)
    return {
        "n": len(values),
        "min": values[0] if values else None,
        "mean": statistics.mean(values) if values else None,
        "sd": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p01": _quantile(values, 0.01),
        "p05": _quantile(values, 0.05),
        "p10": _quantile(values, 0.10),
        "p25": _quantile(values, 0.25),
        "p50": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "max": values[-1] if values else None,
    }


def _candidate_rows_from_elite_preview(
    max_lines_per_file: int,
    min_chars: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    rows = []
    for filename in _fetch_elite_filenames():
        url = f"{HF_RAW_PREFIX}/{filename}"
        with urllib.request.urlopen(url, timeout=90) as response:
            for line_index in range(max_lines_per_file):
                line = response.readline()
                if not line:
                    break
                try:
                    row = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                persona_text = str(row.get("persona", "")).strip()
                if not _passes_length_filter(persona_text, min_chars, max_chars):
                    continue
                source_key = f"{filename}:{line_index + 1}"
                rows.append(
                    {
                        "source_key": source_key,
                        "persona_pid": "personahub_elite_" + hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:12],
                        "persona_source": "personahub_elite_preview",
                        "source_frame": f"ElitePersonas first {max_lines_per_file:,} rows per shard",
                        "persona": persona_text,
                        "persona_chars": len(persona_text),
                        "general_domain_top_1pct": str(row.get("general domain (top 1 percent)", "")),
                        "specific_domain_top_1pct": str(row.get("specific domain (top 1 percent)", "")),
                        "general_domain_top_0_1pct": str(row.get("general domain (top 0.1 percent)", "")),
                        "specific_domain_top_0_1pct": str(row.get("specific domain (top 0.1 percent)", "")),
                    }
                )
    if not rows:
        raise ValueError("No eligible PersonaHub elite-preview personas found.")
    return rows


def _candidate_rows_from_persona_jsonl(min_chars: int, max_chars: int) -> list[dict[str, Any]]:
    rows = []
    with urllib.request.urlopen(HF_PERSONA_JSONL, timeout=90) as response:
        for line_index, line in enumerate(response, start=1):
            try:
                row = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            persona_text = str(row.get("persona", "")).strip()
            if not _passes_length_filter(persona_text, min_chars, max_chars):
                continue
            source_key = f"persona.jsonl:{line_index}"
            rows.append(
                {
                    "source_key": source_key,
                    "persona_pid": "personahub_persona_" + hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:12],
                    "persona_source": "personahub_persona_jsonl",
                    "source_frame": "persona.jsonl official 200k-row subset",
                    "persona": persona_text,
                    "persona_chars": len(persona_text),
                    "general_domain_top_1pct": "",
                    "specific_domain_top_1pct": "",
                    "general_domain_top_0_1pct": "",
                    "specific_domain_top_0_1pct": "",
                }
            )
    if not rows:
        raise ValueError("No eligible PersonaHub persona.jsonl personas found.")
    return rows


def _source_label(args: argparse.Namespace) -> str:
    if args.persona_source == "persona-jsonl":
        return "proj-persona/PersonaHub persona.jsonl official 200k-row subset"
    return "proj-persona/PersonaHub ElitePersonas preview frame"


def _source_slug(args: argparse.Namespace) -> str:
    if args.persona_source == "persona-jsonl":
        return "personahub_persona_jsonl"
    return "personahub_elite_preview"


def _default_sample_file(args: argparse.Namespace) -> Path:
    if args.persona_source == "persona-jsonl":
        return DEFAULT_PERSONA_JSONL_SAMPLE_FILE
    return DEFAULT_ELITE_PREVIEW_SAMPLE_FILE


def _default_run_name(args: argparse.Namespace, target: str) -> str:
    source = "personahub_persona_jsonl_unfiltered" if args.persona_source == "persona-jsonl" else "personahub_elite_preview_unfiltered"
    game_part = "pgg_stratified_32x40" if target == "pgg" else "chip_bargain_stratified_32x48"
    return f"{source}_to_{game_part}_top{args.top_k}_{args.model.replace('-', '_')}_seed_{args.seed}"


def _sample_personas(args: argparse.Namespace) -> list[dict[str, Any]]:
    sample_file = (args.persona_sample_file or _default_sample_file(args)).expanduser().resolve()
    args.persona_sample_file = sample_file
    if sample_file.is_file() and not args.refresh_sample:
        return _read_jsonl(sample_file)

    if args.persona_source == "persona-jsonl":
        candidates = _candidate_rows_from_persona_jsonl(args.min_persona_chars, args.max_persona_chars)
    else:
        candidates = _candidate_rows_from_elite_preview(
            args.max_lines_per_file,
            args.min_persona_chars,
            args.max_persona_chars,
        )
    rng = random.Random(args.seed)
    if args.num_personas <= 0 or args.num_personas >= len(candidates):
        sampled = list(candidates)
    else:
        sampled = [candidates[index] for index in sorted(rng.sample(range(len(candidates)), args.num_personas))]
    _write_jsonl(sample_file, sampled)
    summary = {
        "created_at": _utc_now_iso(),
        "source": _source_label(args),
        "persona_source": args.persona_source,
        "num_candidates_seen": len(candidates),
        "num_sampled": len(sampled),
        "seed": args.seed,
        "max_lines_per_file": args.max_lines_per_file if args.persona_source == "elite-preview" else None,
        "min_persona_chars": args.min_persona_chars,
        "max_persona_chars": args.max_persona_chars,
        "length_filter": (
            "none"
            if args.min_persona_chars == 0 and args.max_persona_chars <= 0
            else f"{args.min_persona_chars} to {args.max_persona_chars if args.max_persona_chars > 0 else 'unbounded'} chars"
        ),
        "candidate_persona_chars": _length_summary([row["persona_chars"] for row in candidates]),
        "sampled_persona_chars": _length_summary([row["persona_chars"] for row in sampled]),
    }
    sample_file.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return sampled


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
            str(persona["persona"]).strip(),
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
            str(persona["persona"]).strip(),
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
        "persona_source": persona.get("persona_source", "personahub"),
        "source_frame": persona.get("source_frame", ""),
        "source_key": persona["source_key"],
        "persona_chars": persona["persona_chars"],
        "persona_sha1": hashlib.sha1(str(persona["persona"]).encode("utf-8")).hexdigest(),
        "general_domain_top_1pct": persona.get("general_domain_top_1pct", ""),
        "specific_domain_top_1pct": persona.get("specific_domain_top_1pct", ""),
        "general_domain_top_0_1pct": persona.get("general_domain_top_0_1pct", ""),
        "specific_domain_top_0_1pct": persona.get("specific_domain_top_0_1pct", ""),
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
                f"{_source_slug(args)}_pgg_match__pid_{persona['persona_pid']}__game_{game_id}__"
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
                    "max_persona_chars": args.max_persona_chars,
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
        "dataset_key": "personahub_elite_transfer_audit_pgg",
        "run_name": args.pgg_run_name,
        "created_at": _utc_now_iso(),
        "model": args.model,
        "seed": args.seed,
        "source_persona_library": _source_label(args),
        "persona_source": args.persona_source,
        "persona_sample_file": str(args.persona_sample_file.expanduser().resolve()),
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
                f"{_source_slug(args)}_chip_match__pid_{persona['persona_pid']}__record_{record_id}__"
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
                    "max_persona_chars": args.max_persona_chars,
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
        "dataset_key": "personahub_elite_transfer_audit_chip_bargain",
        "run_name": args.chip_run_name,
        "created_at": _utc_now_iso(),
        "model": args.model,
        "seed": args.seed,
        "source_persona_library": _source_label(args),
        "persona_source": args.persona_source,
        "persona_sample_file": str(args.persona_sample_file.expanduser().resolve()),
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
    parser.add_argument("--persona-source", choices=["elite-preview", "persona-jsonl"], default="elite-preview")
    parser.add_argument("--persona-sample-file", type=Path, default=None)
    parser.add_argument("--refresh-sample", action="store_true")
    parser.add_argument("--max-lines-per-file", type=int, default=1000)
    parser.add_argument("--min-persona-chars", type=int, default=0)
    parser.add_argument("--max-persona-chars", type=int, default=0)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--num-personas", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--pgg-run-name",
        default=None,
    )
    parser.add_argument(
        "--chip-run-name",
        default=None,
    )
    parser.add_argument("--pgg-source-run", type=Path, default=DEFAULT_PGG_SOURCE_RUN)
    parser.add_argument("--chip-source-run", type=Path, default=DEFAULT_CHIP_SOURCE_RUN)
    parser.add_argument("--pgg-source-gold", type=Path, default=pgg_builder.DEFAULT_SOURCE_GOLD)
    args = parser.parse_args()
    if args.pgg_run_name is None:
        args.pgg_run_name = _default_run_name(args, "pgg")
    if args.chip_run_name is None:
        args.chip_run_name = _default_run_name(args, "chip")
    return args


if __name__ == "__main__":
    run(parse_args())
