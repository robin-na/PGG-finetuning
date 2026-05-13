from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MM_ROOT = SCRIPT_DIR.parent

DEFAULT_SCENARIO_SOURCE_RUN = "mm_global_distribution_scenario_only_n_gt_10000_gpt_5_mini"
DEFAULT_INDIVIDUAL_SOURCE_RUN = (
    "mm_individual_demographic_complete_5_per_scenario_n_gt_10000_actual_order_seed0_gpt_5_mini"
)
DEFAULT_SCENARIO_RUN = (
    "mm_global_distribution_scenario_only_n_gt_10000_gpt_5_mini_verbalized_probs"
)
DEFAULT_INDIVIDUAL_RUN = (
    "mm_individual_demographic_complete_5_per_scenario_n_gt_10000_actual_order_seed0_"
    "gpt_5_mini_verbalized_probs"
)

PROMPT_VERSION = "verbalized_probability_json_v1"

SCENARIO_SYSTEM_PROMPT = (
    "You are estimating the distribution of human responses to a Moral Machine dilemma. "
    "Return JSON only."
)
INDIVIDUAL_SYSTEM_PROMPT = (
    "You are estimating one participant's Moral Machine response probabilities. Return JSON only."
)

SCENARIO_ONE_WORD_TAIL = "\n\nAnswer with A or B only.\nAnswer:"
INDIVIDUAL_ONE_WORD_TAIL = (
    "\n\nWhich option did this participant choose?\nAnswer with A or B only.\nAnswer:"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert existing Moral Machine one-token A/B batch manifests into no-logprob "
            "verbalized probability JSON batches."
        )
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument(
        "--mode",
        choices=["scenario", "individual", "both"],
        default="both",
        help="Which replacement batch input(s) to build.",
    )
    parser.add_argument("--scenario-source-run", default=DEFAULT_SCENARIO_SOURCE_RUN)
    parser.add_argument("--individual-source-run", default=DEFAULT_INDIVIDUAL_SOURCE_RUN)
    parser.add_argument("--scenario-run-name", default=DEFAULT_SCENARIO_RUN)
    parser.add_argument("--individual-run-name", default=DEFAULT_INDIVIDUAL_RUN)
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


def probability_contract(task_line: str) -> str:
    return (
        f"{task_line}\n"
        "Return JSON only.\n"
        "Output contract:\n"
        '- Use exactly two keys: "A" and "B".\n'
        "- Use integer percentages from 0 to 100.\n"
        "- A and B must sum to exactly 100.\n"
        "- Do not include explanations, markdown, or extra keys.\n"
        f"- Example: {json.dumps({'A': 55, 'B': 45})}\n"
        "Answer:"
    )


SCENARIO_PROBABILITY_TAIL = probability_contract(
    "Estimate what percentage of respondents would choose each option if asked this dilemma once."
)
INDIVIDUAL_PROBABILITY_TAIL = probability_contract(
    "Estimate the probability, as integer percentages, that this participant would choose each option "
    "if asked this dilemma once."
)


def replace_prompt_tail(prompt: str, *, old_tail: str, new_tail: str) -> str:
    if not prompt.endswith(old_tail):
        preview = prompt[-200:].replace("\n", "\\n")
        raise ValueError(f"Prompt did not end with expected tail. Last 200 chars: {preview}")
    return prompt[: -len(old_tail)] + "\n\n" + new_tail


def build_batch_entry(
    *,
    custom_id: str,
    model: str,
    system_prompt: str,
    prompt: str,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        },
    }


def output_fieldnames(source_fieldnames: list[str]) -> list[str]:
    fields: list[str] = []
    for field in source_fieldnames:
        fields.append(field)
        if field == "custom_id":
            fields.append("source_custom_id")
    for field in ["source_run_name", "prompt_version", "response_format"]:
        if field not in fields:
            fields.append(field)
    return fields


def transform_manifest_rows(
    *,
    rows: list[dict[str, str]],
    source_run_name: str,
    custom_id_suffix: str,
    system_prompt: str,
    old_tail: str,
    new_tail: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    batch_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    input_char_counts: list[int] = []
    estimated_input_tokens_4cpt: list[int] = []
    estimated_input_tokens_35cpt: list[int] = []

    for row in rows:
        source_custom_id = row["custom_id"]
        custom_id = f"{source_custom_id}_{custom_id_suffix}"
        prompt = replace_prompt_tail(row["prompt"], old_tail=old_tail, new_tail=new_tail)
        input_text = request_input_text(system_prompt, prompt)
        input_char_count = len(input_text)
        input_tokens_4 = estimate_tokens(input_text, chars_per_token=4.0)
        input_tokens_35 = estimate_tokens(input_text, chars_per_token=3.5)

        new_row: dict[str, Any] = dict(row)
        new_row["custom_id"] = custom_id
        new_row["source_custom_id"] = source_custom_id
        new_row["input_char_count"] = input_char_count
        new_row["estimated_input_tokens_4_chars_per_token"] = input_tokens_4
        new_row["estimated_input_tokens_3_5_chars_per_token"] = input_tokens_35
        new_row["prompt"] = prompt
        new_row["source_run_name"] = source_run_name
        new_row["prompt_version"] = PROMPT_VERSION
        new_row["response_format"] = "json_object"
        manifest_rows.append(new_row)

        batch_rows.append(
            build_batch_entry(
                custom_id=custom_id,
                model="",  # Filled by caller to keep this transform independent of model choice.
                system_prompt=system_prompt,
                prompt=prompt,
            )
        )
        input_char_counts.append(input_char_count)
        estimated_input_tokens_4cpt.append(input_tokens_4)
        estimated_input_tokens_35cpt.append(input_tokens_35)

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
    return batch_rows, manifest_rows, token_estimate


def build_from_source_manifest(
    *,
    model: str,
    source_run_name: str,
    run_name: str,
    source_manifest_key: str,
    output_manifest_name: str,
    system_prompt: str,
    old_tail: str,
    new_tail: str,
    condition: str,
    custom_id_suffix: str,
) -> dict[str, Any]:
    source_metadata_dir = MM_ROOT / "metadata" / source_run_name
    source_manifest_path = source_metadata_dir / "manifest.json"
    source_manifest = load_json(source_manifest_path)
    source_csv_path = Path(source_manifest[source_manifest_key])
    source_rows, source_fieldnames = read_csv(source_csv_path)
    if not source_rows:
        raise ValueError(f"No rows found in {source_csv_path}")

    batch_rows, manifest_rows, token_estimate = transform_manifest_rows(
        rows=source_rows,
        source_run_name=source_run_name,
        custom_id_suffix=custom_id_suffix,
        system_prompt=system_prompt,
        old_tail=old_tail,
        new_tail=new_tail,
    )
    for batch_row in batch_rows:
        batch_row["body"]["model"] = model

    batch_path = MM_ROOT / "batch_input" / f"{run_name}.jsonl"
    output_path = MM_ROOT / "batch_output" / f"{run_name}.jsonl"
    metadata_dir = MM_ROOT / "metadata" / run_name
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv_path = metadata_dir / output_manifest_name
    run_manifest_path = metadata_dir / "manifest.json"
    prompt_preview_path = metadata_dir / "sample_prompt.txt"

    write_jsonl(batch_path, batch_rows)
    write_csv(manifest_csv_path, manifest_rows, output_fieldnames(source_fieldnames))
    prompt_preview_path.write_text(manifest_rows[0]["prompt"] + "\n", encoding="utf-8")

    run_manifest = {
        "run_name": run_name,
        "source_run_name": source_run_name,
        "source_manifest_file": str(source_manifest_path),
        "source_request_manifest_file": str(source_csv_path),
        "model": model,
        "metadata_dir": str(metadata_dir),
        "batch_input_file": str(batch_path),
        "expected_batch_output_file": str(output_path),
        source_manifest_key: str(manifest_csv_path),
        "sample_prompt_file": str(prompt_preview_path),
        "endpoint": "/v1/chat/completions",
        "response_format": {"type": "json_object"},
        "logprobs": False,
        "top_logprobs": None,
        "max_completion_tokens": None,
        "condition": condition,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": system_prompt,
        "output_contract": {
            "top_level_keys": ["A", "B"],
            "value_type": "integer_percentages",
            "sum": 100,
            "no_explanations_or_extra_keys": True,
        },
        "threshold": source_manifest.get("threshold"),
        "num_requests": len(batch_rows),
        "answer_labels": source_manifest.get("answer_labels"),
        "option_order": source_manifest.get("option_order"),
        "source_notes": {
            "preserved_prompt_content": (
                "Scenario text, demographic profile text, option labels, and option order were "
                "copied from the source run. Only the final answer instruction and system prompt "
                "were changed for verbalized probabilities."
            ),
            "not_submitted": True,
        },
        "token_estimate": token_estimate,
    }
    for passthrough_key in [
        "num_scenarios",
        "samples_per_scenario",
        "seed",
        "left_action_counts",
        "gold_choice_counts",
        "demographic_filter",
        "candidate_cache_file",
        "candidate_cache_manifest_file",
        "sampling_source",
    ]:
        if passthrough_key in source_manifest:
            run_manifest[passthrough_key] = source_manifest[passthrough_key]

    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "run_name": run_name,
        "batch_input_file": str(batch_path),
        "request_manifest_file": str(manifest_csv_path),
        "sample_prompt_file": str(prompt_preview_path),
        "run_manifest_file": str(run_manifest_path),
        "num_requests": len(batch_rows),
        "estimated_input_tokens_4_chars": token_estimate[
            "estimated_input_tokens_4_chars_per_token_total"
        ],
        "estimated_input_tokens_3_5_chars": token_estimate[
            "estimated_input_tokens_3_5_chars_per_token_total"
        ],
    }


def main() -> None:
    args = parse_args()
    outputs: list[dict[str, Any]] = []
    if args.mode in {"scenario", "both"}:
        outputs.append(
            build_from_source_manifest(
                model=args.model,
                source_run_name=args.scenario_source_run,
                run_name=args.scenario_run_name,
                source_manifest_key="scenario_manifest_file",
                output_manifest_name="scenario_manifest.csv",
                system_prompt=SCENARIO_SYSTEM_PROMPT,
                old_tail=SCENARIO_ONE_WORD_TAIL,
                new_tail=SCENARIO_PROBABILITY_TAIL,
                condition="scenario_only_global_distribution_alignment_verbalized_probabilities",
                custom_id_suffix="vprob",
            )
        )
    if args.mode in {"individual", "both"}:
        outputs.append(
            build_from_source_manifest(
                model=args.model,
                source_run_name=args.individual_source_run,
                run_name=args.individual_run_name,
                source_manifest_key="sample_manifest_file",
                output_manifest_name="sample_manifest.csv",
                system_prompt=INDIVIDUAL_SYSTEM_PROMPT,
                old_tail=INDIVIDUAL_ONE_WORD_TAIL,
                new_tail=INDIVIDUAL_PROBABILITY_TAIL,
                condition="individual_demographic_complete_actual_order_verbalized_probabilities",
                custom_id_suffix="vprob",
            )
        )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
