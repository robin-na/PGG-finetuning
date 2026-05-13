from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MM_ROOT = SCRIPT_DIR.parent
DEFAULT_SOURCE_RUN = (
    "mm_individual_demographic_complete_5_per_scenario_n_gt_10000_actual_order_seed0_gpt_5_mini"
)
DEFAULT_RUN_NAME = (
    "mm_individual_demographic_complete_5_per_scenario_n_gt_10000_actual_order_seed0_"
    "gpt_5_mini_ab_no_logprobs"
)
DEFAULT_SYSTEM_PROMPT = (
    "You need to predict a participant's decision in the following moral dilemma scenario. "
    "Return only A or B."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an individual-level Moral Machine A/B batch from the existing demographic "
            "sample manifest, without logprobs."
        )
    )
    parser.add_argument("--source-run", default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument(
        "--custom-id-suffix",
        default="ab",
        help="Suffix appended to each source custom_id in the new batch.",
    )
    parser.add_argument(
        "--subset-manifest-csv",
        type=Path,
        default=None,
        help="Optional manifest whose IDs define a subset of source rows to keep.",
    )
    parser.add_argument(
        "--subset-custom-id-column",
        default="source_custom_id",
        help=(
            "Column in --subset-manifest-csv containing source-run custom_ids. "
            "For WVS-covered controls this is source_custom_id."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Optional temperature. This is intentionally omitted by default because OpenAI docs "
            "state gpt-5/gpt-5-mini/gpt-5-nano reject temperature/top_p/logprobs fields."
        ),
    )
    return parser.parse_args()


def estimate_tokens(text: str, *, chars_per_token: float) -> int:
    return math.ceil(len(text) / chars_per_token)


def request_input_text(system_prompt: str, user_prompt: str) -> str:
    return f"system: {system_prompt}\nuser: {user_prompt}\n"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {path}")
        return rows, list(reader.fieldnames)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_batch_entry(
    *,
    custom_id: str,
    model: str,
    system_prompt: str,
    prompt: str,
    temperature: float | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    if temperature is not None:
        body["temperature"] = temperature
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def output_fieldnames(source_fieldnames: list[str]) -> list[str]:
    fields: list[str] = []
    for field in source_fieldnames:
        if field == "source_custom_id":
            continue
        fields.append(field)
        if field == "custom_id" and "source_custom_id" not in fields:
            fields.append("source_custom_id")
    for field in ["source_run_name", "prompt_version", "temperature_included"]:
        if field not in fields:
            fields.append(field)
    return fields


def main() -> None:
    args = parse_args()
    source_metadata_dir = MM_ROOT / "metadata" / args.source_run
    source_manifest_path = source_metadata_dir / "manifest.json"
    source_manifest = load_json(source_manifest_path)
    source_csv_path = Path(source_manifest["sample_manifest_file"])
    source_rows, source_fieldnames = read_csv(source_csv_path)
    if not source_rows:
        raise ValueError(f"No rows found in {source_csv_path}")
    num_source_rows_before_subset = len(source_rows)
    subset_ids: set[str] | None = None
    if args.subset_manifest_csv is not None:
        subset_rows, _ = read_csv(args.subset_manifest_csv)
        if not subset_rows:
            raise ValueError(f"No rows found in subset manifest {args.subset_manifest_csv}")
        if args.subset_custom_id_column not in subset_rows[0]:
            raise ValueError(
                f"Subset manifest {args.subset_manifest_csv} does not contain "
                f"{args.subset_custom_id_column!r}."
            )
        subset_ids = {
            str(row.get(args.subset_custom_id_column) or "")
            for row in subset_rows
            if row.get(args.subset_custom_id_column)
        }
        source_rows = [row for row in source_rows if row["custom_id"] in subset_ids]
        if not source_rows:
            raise ValueError(
                "Subset filtering removed every source row. Check --source-run and "
                "--subset-custom-id-column."
            )

    system_prompt = str(source_manifest.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)
    batch_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    input_char_counts: list[int] = []
    estimated_input_tokens_4cpt: list[int] = []
    estimated_input_tokens_35cpt: list[int] = []

    for row in source_rows:
        source_custom_id = row["custom_id"]
        custom_id = f"{source_custom_id}_{args.custom_id_suffix}"
        prompt = row["prompt"]
        input_text = request_input_text(system_prompt, prompt)
        input_char_count = len(input_text)
        input_tokens_4 = estimate_tokens(input_text, chars_per_token=4.0)
        input_tokens_35 = estimate_tokens(input_text, chars_per_token=3.5)

        batch_rows.append(
            build_batch_entry(
                custom_id=custom_id,
                model=args.model,
                system_prompt=system_prompt,
                prompt=prompt,
                temperature=args.temperature,
            )
        )

        new_row: dict[str, Any] = dict(row)
        new_row["custom_id"] = custom_id
        new_row["source_custom_id"] = source_custom_id
        new_row["input_char_count"] = input_char_count
        new_row["estimated_input_tokens_4_chars_per_token"] = input_tokens_4
        new_row["estimated_input_tokens_3_5_chars_per_token"] = input_tokens_35
        new_row["source_run_name"] = args.source_run
        new_row["prompt_version"] = "individual_demographic_ab_no_logprobs_v1"
        new_row["temperature_included"] = args.temperature is not None
        manifest_rows.append(new_row)

        input_char_counts.append(input_char_count)
        estimated_input_tokens_4cpt.append(input_tokens_4)
        estimated_input_tokens_35cpt.append(input_tokens_35)

    batch_path = MM_ROOT / "batch_input" / f"{args.run_name}.jsonl"
    output_path = MM_ROOT / "batch_output" / f"{args.run_name}.jsonl"
    metadata_dir = MM_ROOT / "metadata" / args.run_name
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv_path = metadata_dir / "sample_manifest.csv"
    run_manifest_path = metadata_dir / "manifest.json"
    prompt_preview_path = metadata_dir / "sample_prompt.txt"

    write_jsonl(batch_path, batch_rows)
    write_csv(manifest_csv_path, manifest_rows, output_fieldnames(source_fieldnames))
    prompt_preview_path.write_text(manifest_rows[0]["prompt"] + "\n", encoding="utf-8")

    token_estimate = {
        "method": (
            "Approximate only: estimated from rendered system+user message text because no local "
            "model tokenizer is installed."
        ),
        "input_characters_total": sum(input_char_counts),
        "estimated_input_tokens_4_chars_per_token_total": sum(estimated_input_tokens_4cpt),
        "estimated_input_tokens_3_5_chars_per_token_total": sum(estimated_input_tokens_35cpt),
        "estimated_input_tokens_4_chars_per_token_mean": (
            sum(estimated_input_tokens_4cpt) / len(estimated_input_tokens_4cpt)
        ),
        "estimated_input_tokens_3_5_chars_per_token_mean": (
            sum(estimated_input_tokens_35cpt) / len(estimated_input_tokens_35cpt)
        ),
        "min_estimated_input_tokens_4_chars_per_token": min(estimated_input_tokens_4cpt),
        "max_estimated_input_tokens_4_chars_per_token": max(estimated_input_tokens_4cpt),
        "min_estimated_input_tokens_3_5_chars_per_token": min(estimated_input_tokens_35cpt),
        "max_estimated_input_tokens_3_5_chars_per_token": max(estimated_input_tokens_35cpt),
    }
    run_manifest = {
        "run_name": args.run_name,
        "source_run_name": args.source_run,
        "source_manifest_file": str(source_manifest_path),
        "source_request_manifest_file": str(source_csv_path),
        "model": args.model,
        "metadata_dir": str(metadata_dir),
        "batch_input_file": str(batch_path),
        "expected_batch_output_file": str(output_path),
        "sample_manifest_file": str(manifest_csv_path),
        "sample_prompt_file": str(prompt_preview_path),
        "endpoint": "/v1/chat/completions",
        "response_format": None,
        "logprobs": False,
        "top_logprobs": None,
        "max_completion_tokens": None,
        "temperature": args.temperature,
        "temperature_note": (
            "temperature was left unset for gpt-5-mini. Official OpenAI GPT-5-family guidance "
            "states temperature/top_p/logprobs raise errors for gpt-5, gpt-5-mini, and gpt-5-nano."
            if args.temperature is None
            else "temperature was explicitly included by user/config request."
        ),
        "condition": "individual_demographic_complete_actual_order_ab_no_logprobs",
        "prompt_version": "individual_demographic_ab_no_logprobs_v1",
        "num_source_rows_before_subset": num_source_rows_before_subset,
        "subset_manifest_file": str(args.subset_manifest_csv) if args.subset_manifest_csv else None,
        "subset_custom_id_column": args.subset_custom_id_column if args.subset_manifest_csv else None,
        "subset_ids_requested": len(subset_ids) if subset_ids is not None else None,
        "subset_filter": (
            "Kept source rows whose custom_id appeared in the subset manifest column."
            if subset_ids is not None
            else None
        ),
        "system_prompt": system_prompt,
        "threshold": source_manifest.get("threshold"),
        "num_scenarios": source_manifest.get("num_scenarios"),
        "num_requests": len(batch_rows),
        "samples_per_scenario": source_manifest.get("samples_per_scenario"),
        "seed": source_manifest.get("seed"),
        "answer_labels": source_manifest.get("answer_labels"),
        "option_order": source_manifest.get("option_order"),
        "left_action_counts": source_manifest.get("left_action_counts"),
        "gold_choice_counts": source_manifest.get("gold_choice_counts"),
        "demographic_filter": source_manifest.get("demographic_filter"),
        "candidate_cache_file": source_manifest.get("candidate_cache_file"),
        "candidate_cache_manifest_file": source_manifest.get("candidate_cache_manifest_file"),
        "not_submitted": True,
        "token_estimate": token_estimate,
    }
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_name": args.run_name,
                "batch_input_file": str(batch_path),
                "sample_manifest_file": str(manifest_csv_path),
                "sample_prompt_file": str(prompt_preview_path),
                "run_manifest_file": str(run_manifest_path),
                "num_requests": len(batch_rows),
                "temperature_included": args.temperature is not None,
                "estimated_input_tokens_4_chars": token_estimate[
                    "estimated_input_tokens_4_chars_per_token_total"
                ],
                "estimated_input_tokens_3_5_chars": token_estimate[
                    "estimated_input_tokens_3_5_chars_per_token_total"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
