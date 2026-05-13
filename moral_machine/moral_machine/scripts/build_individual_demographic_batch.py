from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

from build_distribution_alignment_batch import (
    compact_outcome,
    estimate_tokens,
    request_input_text,
)
from build_ordered_distribution_alignment_batch import chosen_action, left_action
from build_stay_swerve_batch import (
    DEFAULT_FULL_INPUT,
    DEFAULT_SURVEY_INPUT,
    DEMOGRAPHIC_FIELDS,
    has_complete_demographics,
    is_default_or_blank,
    iter_csv_rows,
    parse_float,
)
from count_scenario_repeats import paired_rows
from count_unique_scenarios import value, visible_signature


SCRIPT_DIR = Path(__file__).resolve().parent
MM_ROOT = SCRIPT_DIR.parent
DEFAULT_DB = (
    MM_ROOT
    / "processed"
    / "high_repeat_scenario_cells"
    / "global_100_cell_100"
    / "scenario_cells.sqlite"
)
DEFAULT_RUN_NAME = (
    "mm_individual_demographic_complete_5_per_scenario_n_gt_10000_actual_order_seed0_gpt_5_mini"
)

PREDICTION_SYSTEM_PROMPT = (
    "You need to predict a participant's decision in the following moral dilemma scenario. "
    "Return only A or B."
)

EDUCATION_LABELS = {
    "underHigh": "Less than a high school diploma",
    "high": "High school diploma",
    "vocational": "Vocational training",
    "college": "Attended college",
    "bachelor": "Bachelor degree",
    "graduate": "Graduate degree",
    "other": "Other",
}

GENDER_LABELS = {
    "male": "Male",
    "female": "Female",
    "other": "Other",
}

INCOME_LABELS = {
    "under5000": "Under $5,000",
    "5000": "$5,000-$10,000",
    "10000": "$10,001-$15,000",
    "15000": "$15,001-$25,000",
    "25000": "$25,001-$35,000",
    "35000": "$35,001-$50,000",
    "50000": "$50,001-$80,000",
    "80000": "$80,001-$100,000",
    "above100000": "Over $100,000",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an individual-level Moral Machine batch with complete participant demographics, "
            "sampled per high-repeat exact scenario and rendered in actual left/right order."
        )
    )
    parser.add_argument("--survey-input", type=Path, default=DEFAULT_SURVEY_INPUT)
    parser.add_argument("--full-input", type=Path, default=DEFAULT_FULL_INPUT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--samples-per-scenario", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-global-count", type=int, default=10_000)
    parser.add_argument(
        "--strictly-greater",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use global_count > threshold instead of global_count >= threshold.",
    )
    parser.add_argument("--age-min", type=int, default=18)
    parser.add_argument("--age-max", type=int, default=75)
    parser.add_argument(
        "--exclude-slider-midpoints",
        action="store_true",
        help=(
            "Drop Review_political/Review_religious values of exactly 0.5. The MM codebook says "
            "0.5 is also the no-answer default, but it is indistinguishable from a true midpoint."
        ),
    )
    parser.add_argument(
        "--require-country",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require nonblank UserCountry3 for the sampled participant.",
    )
    parser.add_argument("--max-completion-tokens", type=int, default=1)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    return parser.parse_args()


def fetch_scenarios(db_path: Path, min_global_count: int, strictly_greater: bool) -> list[dict[str, Any]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    operator = ">" if strictly_greater else ">="
    try:
        rows = connection.execute(
            f"""
            SELECT *
            FROM scenarios
            WHERE global_count {operator} ?
            ORDER BY global_count DESC, scenario_hash
            """,
            (min_global_count,),
        )
        return [dict(row) for row in rows]
    finally:
        connection.close()


def valid_demographics(
    row: dict[str, str],
    *,
    age_min: int,
    age_max: int,
    require_country: bool,
    exclude_slider_midpoints: bool,
) -> bool:
    if require_country and is_default_or_blank(row.get("UserCountry3")):
        return False
    if not has_complete_demographics(row, age_min=age_min, age_max=age_max):
        return False
    if exclude_slider_midpoints:
        for field in ["Review_political", "Review_religious"]:
            value = parse_float(row.get(field, ""))
            if value == 0.5:
                return False
    return True


def load_complete_response_ids(
    *,
    survey_input: Path,
    age_min: int,
    age_max: int,
    require_country: bool,
    exclude_slider_midpoints: bool,
) -> tuple[set[str], dict[str, int]]:
    complete_ids: set[str] = set()
    stats = {
        "survey_rows_seen": 0,
        "survey_stay_rows_seen": 0,
        "complete_demographic_stay_rows_seen": 0,
    }
    for row in iter_csv_rows(survey_input):
        stats["survey_rows_seen"] += 1
        if row.get("Intervention") != "0":
            continue
        stats["survey_stay_rows_seen"] += 1
        if not valid_demographics(
            row,
            age_min=age_min,
            age_max=age_max,
            require_country=require_country,
            exclude_slider_midpoints=exclude_slider_midpoints,
        ):
            continue
        response_id = row.get("ResponseID", "")
        if not response_id:
            continue
        complete_ids.add(response_id)
        stats["complete_demographic_stay_rows_seen"] += 1
    stats["unique_complete_response_ids"] = len(complete_ids)
    return complete_ids, stats


def load_selected_demographics(
    *,
    survey_input: Path,
    response_ids: set[str],
    age_min: int,
    age_max: int,
    require_country: bool,
    exclude_slider_midpoints: bool,
) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for row in iter_csv_rows(survey_input):
        response_id = row.get("ResponseID", "")
        if (
            response_id not in response_ids
            or response_id in selected
            or row.get("Intervention") != "0"
        ):
            continue
        if not valid_demographics(
            row,
            age_min=age_min,
            age_max=age_max,
            require_country=require_country,
            exclude_slider_midpoints=exclude_slider_midpoints,
        ):
            continue
        selected[response_id] = {
            "UserCountry3": row.get("UserCountry3", ""),
            **{field: row.get(field, "") for field in DEMOGRAPHIC_FIELDS},
        }
        if len(selected) == len(response_ids):
            break
    missing = response_ids - set(selected)
    if missing:
        raise ValueError(f"Could not reload demographics for {len(missing)} selected responses.")
    return selected


def screen_order_from_left(left: str) -> list[dict[str, str]]:
    if left == "stay":
        return [
            {"label": "A", "prefix": "stay", "action": "Stay", "screen_side": "left"},
            {"label": "B", "prefix": "swerve", "action": "Swerve", "screen_side": "right"},
        ]
    if left == "swerve":
        return [
            {"label": "A", "prefix": "swerve", "action": "Swerve", "screen_side": "left"},
            {"label": "B", "prefix": "stay", "action": "Stay", "screen_side": "right"},
        ]
    raise ValueError(f"Unsupported left action: {left}")


def education_label(raw: str) -> str:
    return EDUCATION_LABELS.get(raw, raw)


def gender_label(raw: str) -> str:
    return GENDER_LABELS.get(raw, raw)


def income_label(raw: str) -> str:
    return INCOME_LABELS.get(raw, raw)


def slider_label(raw: str) -> str:
    value = parse_float(raw)
    if value is None:
        return raw
    return f"{value:.2f}"


def demographic_block(demographics: dict[str, str]) -> str:
    return "\n".join(
        [
            "Participant profile:",
            f"Age: {demographics['Review_age']}",
            f"Gender: {gender_label(demographics['Review_gender'])}",
            f"Education: {education_label(demographics['Review_education'])}",
            f"Annual income: {income_label(demographics['Review_income'])}",
            (
                "Political views: "
                f"{slider_label(demographics['Review_political'])} "
                "on a conservative (0) to progressive (1) scale"
            ),
            (
                "Religiosity: "
                f"{slider_label(demographics['Review_religious'])} "
                "on a not religious (0) to very religious (1) scale"
            ),
            f"Country: {demographics['UserCountry3']}",
        ]
    )


def build_prompt(
    scenario: dict[str, Any],
    *,
    demographics: dict[str, str],
    left: str,
) -> str:
    assignments = screen_order_from_left(left)
    return (
        "A participant with the following profile answered a Moral Machine dilemma.\n\n"
        f"{demographic_block(demographics)}\n\n"
        "A self-driving car's brakes suddenly fail, and you must choose one to either stay on course or swerve.\n\n"
        + "\n".join(
            compact_outcome(
                scenario,
                prefix=assignment["prefix"],
                label=assignment["label"],
                action=assignment["action"],
            )
            for assignment in assignments
        )
        + "\n\n"
        "Which option did this participant choose?\n"
        "Answer with A or B only.\n"
        "Answer:"
    )


def build_batch_entry(
    *,
    custom_id: str,
    model: str,
    prompt: str,
    max_completion_tokens: int,
    top_logprobs: int,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": PREDICTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": max_completion_tokens,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def reservoir_select_pairs(
    *,
    full_input: Path,
    complete_response_ids: set[str],
    scenarios_by_hash: dict[str, dict[str, Any]],
    samples_per_scenario: int,
    rng: random.Random,
    progress_every: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    target_hashes = {bytes.fromhex(scenario_hash) for scenario_hash in scenarios_by_hash}
    selected: dict[str, list[dict[str, Any]]] = {scenario_hash: [] for scenario_hash in scenarios_by_hash}
    eligible_counts: Counter[str] = Counter()
    anomalies: Counter[str] = Counter()
    matched_target_pairs = 0
    complete_target_pairs = 0
    start = time.time()

    for stay_row, swerve_row, index in paired_rows(full_input, progress_every=progress_every):
        scenario_hash_bytes = visible_signature(stay_row, swerve_row, index)
        if scenario_hash_bytes not in target_hashes:
            continue
        matched_target_pairs += 1
        response_id = value(stay_row, index, "ResponseID")
        if response_id not in complete_response_ids:
            continue
        complete_target_pairs += 1
        left = left_action(stay_row, swerve_row, index)
        if left is None:
            anomalies["invalid_left_right_order"] += 1
            continue
        choice = chosen_action(stay_row, swerve_row, index)
        if choice is None:
            anomalies["invalid_choice"] += 1
            continue

        scenario_hash = scenario_hash_bytes.hex()
        eligible_counts[scenario_hash] += 1
        seen = eligible_counts[scenario_hash]
        candidate = {
            "response_id": response_id,
            "extended_session_id": value(stay_row, index, "ExtendedSessionID"),
            "user_id": value(stay_row, index, "UserID"),
            "scenario_order": value(stay_row, index, "ScenarioOrder"),
            "user_country3_raw": value(stay_row, index, "UserCountry3"),
            "left_action": left,
            "right_action": "swerve" if left == "stay" else "stay",
            "chosen_action": choice,
            "gold_choice": "A" if choice == left else "B",
        }
        sample = selected[scenario_hash]
        if len(sample) < samples_per_scenario:
            sample.append(candidate)
        else:
            replacement_index = rng.randrange(seen)
            if replacement_index < samples_per_scenario:
                sample[replacement_index] = candidate

        if progress_every and matched_target_pairs % progress_every == 0:
            print(
                json.dumps(
                    {
                        "matched_target_pairs": matched_target_pairs,
                        "complete_target_pairs": complete_target_pairs,
                        "filled_scenarios": sum(
                            1 for rows in selected.values() if len(rows) >= samples_per_scenario
                        ),
                        "elapsed_seconds": round(time.time() - start, 1),
                    }
                ),
                flush=True,
            )

    scan_summary = dict(paired_rows.summary)
    underfilled = {
        scenario_hash: len(rows)
        for scenario_hash, rows in selected.items()
        if len(rows) < samples_per_scenario
    }
    stats = {
        "matched_target_pairs": matched_target_pairs,
        "complete_target_pairs": complete_target_pairs,
        "eligible_pairs_after_order_and_choice_checks": sum(eligible_counts.values()),
        "eligible_counts_min": min(eligible_counts.values()) if eligible_counts else 0,
        "eligible_counts_max": max(eligible_counts.values()) if eligible_counts else 0,
        "eligible_counts_by_scenario_hash": dict(sorted(eligible_counts.items())),
        "underfilled_scenarios": underfilled,
        "anomalies": dict(sorted(anomalies.items())),
        "full_scan_summary": scan_summary,
    }
    if underfilled:
        sample = dict(list(underfilled.items())[:10])
        raise ValueError(
            f"{len(underfilled)} scenarios had fewer than {samples_per_scenario} complete candidates: {sample}"
        )
    return selected, stats


def main() -> None:
    args = parse_args()
    if args.samples_per_scenario <= 0:
        raise ValueError("--samples-per-scenario must be positive.")

    scenarios = fetch_scenarios(args.db, args.min_global_count, args.strictly_greater)
    if not scenarios:
        raise ValueError("No scenarios matched the requested threshold.")
    scenarios_by_hash = {scenario["scenario_hash"]: scenario for scenario in scenarios}

    print(
        json.dumps(
            {
                "pass": "load_complete_survey_response_ids",
                "survey_input": str(args.survey_input),
                "target_scenarios": len(scenarios),
            }
        ),
        flush=True,
    )
    complete_response_ids, demographic_stats = load_complete_response_ids(
        survey_input=args.survey_input,
        age_min=args.age_min,
        age_max=args.age_max,
        require_country=args.require_country,
        exclude_slider_midpoints=args.exclude_slider_midpoints,
    )

    print(
        json.dumps(
            {
                "pass": "sample_full_response_pairs",
                "full_input": str(args.full_input),
                "complete_response_ids": len(complete_response_ids),
                "target_scenarios": len(scenarios),
                "samples_per_scenario": args.samples_per_scenario,
            }
        ),
        flush=True,
    )
    selected_by_hash, sampling_stats = reservoir_select_pairs(
        full_input=args.full_input,
        complete_response_ids=complete_response_ids,
        scenarios_by_hash=scenarios_by_hash,
        samples_per_scenario=args.samples_per_scenario,
        rng=random.Random(args.seed),
        progress_every=args.progress_every,
    )

    selected_response_ids = {
        row["response_id"] for rows in selected_by_hash.values() for row in rows
    }
    demographics_by_response_id = load_selected_demographics(
        survey_input=args.survey_input,
        response_ids=selected_response_ids,
        age_min=args.age_min,
        age_max=args.age_max,
        require_country=args.require_country,
        exclude_slider_midpoints=args.exclude_slider_midpoints,
    )

    batch_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    input_char_counts: list[int] = []
    estimated_input_tokens_4cpt: list[int] = []
    estimated_input_tokens_35cpt: list[int] = []

    request_index = 0
    for scenario_index, scenario in enumerate(scenarios, start=1):
        scenario_hash = scenario["scenario_hash"]
        selected_rows = list(selected_by_hash[scenario_hash])
        selected_rows.sort(key=lambda row: row["response_id"])
        for sample_index, selected in enumerate(selected_rows, start=1):
            request_index += 1
            custom_id = f"mm_individual_demo_{scenario_index:03d}_{sample_index:02d}"
            demographics = demographics_by_response_id[selected["response_id"]]
            prompt = build_prompt(
                scenario,
                demographics=demographics,
                left=selected["left_action"],
            )
            input_text = request_input_text(PREDICTION_SYSTEM_PROMPT, prompt)
            input_char_count = len(input_text)
            input_char_counts.append(input_char_count)
            estimated_input_tokens_4cpt.append(estimate_tokens(input_text, chars_per_token=4.0))
            estimated_input_tokens_35cpt.append(estimate_tokens(input_text, chars_per_token=3.5))
            batch_rows.append(
                build_batch_entry(
                    custom_id=custom_id,
                    model=args.model,
                    prompt=prompt,
                    max_completion_tokens=args.max_completion_tokens,
                    top_logprobs=args.top_logprobs,
                )
            )
            manifest_rows.append(
                {
                    "sample_index": request_index,
                    "scenario_index": scenario_index,
                    "sample_within_scenario": sample_index,
                    "custom_id": custom_id,
                    "response_id": selected["response_id"],
                    "extended_session_id": selected["extended_session_id"],
                    "user_id": selected["user_id"],
                    "scenario_order": selected["scenario_order"],
                    "scenario_hash": scenario_hash,
                    "global_count": scenario["global_count"],
                    "left_action": selected["left_action"],
                    "right_action": selected["right_action"],
                    "option_a_screen_side": "left",
                    "option_b_screen_side": "right",
                    "option_a_action": selected["left_action"],
                    "option_b_action": selected["right_action"],
                    "gold_choice": selected["gold_choice"],
                    "gold_action": selected["chosen_action"],
                    "user_country3": demographics["UserCountry3"],
                    "user_country3_raw_full_file": selected["user_country3_raw"],
                    "review_age": demographics["Review_age"],
                    "review_education": demographics["Review_education"],
                    "review_gender": demographics["Review_gender"],
                    "review_income": demographics["Review_income"],
                    "review_political": demographics["Review_political"],
                    "review_religious": demographics["Review_religious"],
                    "scenario_type": scenario["ScenarioType"],
                    "scenario_type_strict": scenario["ScenarioTypeStrict"],
                    "attribute_level": scenario["AttributeLevel"],
                    "pedped": scenario["PedPed"],
                    "input_char_count": input_char_count,
                    "estimated_input_tokens_4_chars_per_token": estimated_input_tokens_4cpt[-1],
                    "estimated_input_tokens_3_5_chars_per_token": estimated_input_tokens_35cpt[-1],
                    "prompt": prompt,
                }
            )

    batch_path = MM_ROOT / "batch_input" / f"{args.run_name}.jsonl"
    output_path = MM_ROOT / "batch_output" / f"{args.run_name}.jsonl"
    metadata_dir = MM_ROOT / "metadata" / args.run_name
    manifest_csv_path = metadata_dir / "sample_manifest.csv"
    run_manifest_path = metadata_dir / "manifest.json"
    prompt_preview_path = metadata_dir / "sample_prompt.txt"

    write_jsonl(batch_path, batch_rows)
    write_csv(manifest_csv_path, manifest_rows)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    prompt_preview_path.write_text(manifest_rows[0]["prompt"] + "\n", encoding="utf-8")

    token_estimate = {
        "method": (
            "Approximate only: estimated from rendered system+user message text because no local "
            "model tokenizer is installed."
        ),
        "input_characters_total": sum(input_char_counts),
        "estimated_input_tokens_4_chars_per_token_total": sum(estimated_input_tokens_4cpt),
        "estimated_input_tokens_3_5_chars_per_token_total": sum(estimated_input_tokens_35cpt),
        "estimated_input_tokens_4_chars_per_token_mean": sum(estimated_input_tokens_4cpt)
        / len(batch_rows),
        "estimated_input_tokens_3_5_chars_per_token_mean": sum(estimated_input_tokens_35cpt)
        / len(batch_rows),
        "min_estimated_input_tokens_4_chars_per_token": min(estimated_input_tokens_4cpt),
        "max_estimated_input_tokens_4_chars_per_token": max(estimated_input_tokens_4cpt),
        "min_estimated_input_tokens_3_5_chars_per_token": min(estimated_input_tokens_35cpt),
        "max_estimated_input_tokens_3_5_chars_per_token": max(estimated_input_tokens_35cpt),
    }

    left_counts = Counter(row["left_action"] for row in manifest_rows)
    gold_counts = Counter(row["gold_choice"] for row in manifest_rows)
    run_manifest = {
        "run_name": args.run_name,
        "model": args.model,
        "survey_source_file": str(args.survey_input),
        "full_outcome_source_file": str(args.full_input),
        "db": str(args.db),
        "metadata_dir": str(metadata_dir),
        "batch_input_file": str(batch_path),
        "expected_batch_output_file": str(output_path),
        "sample_manifest_file": str(manifest_csv_path),
        "sample_prompt_file": str(prompt_preview_path),
        "endpoint": "/v1/chat/completions",
        "max_completion_tokens": args.max_completion_tokens,
        "logprobs": True,
        "top_logprobs": args.top_logprobs,
        "condition": "individual_demographic_complete_actual_order",
        "system_prompt": PREDICTION_SYSTEM_PROMPT,
        "samples_per_scenario": args.samples_per_scenario,
        "seed": args.seed,
        "threshold": {
            "field": "global_count",
            "operator": ">" if args.strictly_greater else ">=",
            "value": args.min_global_count,
        },
        "num_scenarios": len(scenarios),
        "num_requests": len(batch_rows),
        "answer_labels": {
            "A": "actual left-hand option",
            "B": "actual right-hand option",
        },
        "option_order": "actual_left_right_from_LeftHand",
        "left_action_counts": dict(sorted(left_counts.items())),
        "gold_choice_counts": dict(sorted(gold_counts.items())),
        "demographic_filter": {
            "source": "SharedResponsesSurvey.csv",
            "required_fields": [*DEMOGRAPHIC_FIELDS, "UserCountry3"]
            if args.require_country
            else list(DEMOGRAPHIC_FIELDS),
            "age_min": args.age_min,
            "age_max": args.age_max,
            "education_gender_income": "nonblank and not default",
            "political_religious": (
                "valid 0-1 slider values; 0.5 retained because MM uses 0.5 both as midpoint "
                "and as the documented no-answer default"
                if not args.exclude_slider_midpoints
                else "valid 0-1 slider values excluding exactly 0.5, the documented no-answer default"
            ),
            "require_country": args.require_country,
        },
        "demographic_stats": demographic_stats,
        "sampling_stats": sampling_stats,
        "token_estimate": token_estimate,
    }
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "batch_input_file": str(batch_path),
                "sample_manifest_file": str(manifest_csv_path),
                "sample_prompt_file": str(prompt_preview_path),
                "run_manifest_file": str(run_manifest_path),
                "num_requests": len(batch_rows),
                "left_action_counts": dict(sorted(left_counts.items())),
                "gold_choice_counts": dict(sorted(gold_counts.items())),
                "estimated_input_tokens_4_chars": token_estimate[
                    "estimated_input_tokens_4_chars_per_token_total"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
