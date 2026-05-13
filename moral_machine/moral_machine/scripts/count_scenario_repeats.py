from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from count_unique_scenarios import DEFAULT_INPUT, OutcomeStream, next_or_none, value, visible_signature
from build_stay_swerve_batch import action_outcome_text, gold_choice


DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "metadata" / "scenario_repeat_counts.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count repeated exact Moral Machine Stay/Swerve scenario pairs globally and within country cells."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=int, default=532)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    return parser.parse_args()


def paired_rows(input_path: Path, progress_every: int = 0):
    stay_stream = OutcomeStream(input_path, "0")
    swerve_stream = OutcomeStream(input_path, "1")
    if stay_stream.header != swerve_stream.header:
        raise ValueError("Stay/Swerve stream headers differ.")
    index = stay_stream.index

    stay_iter = iter(stay_stream)
    swerve_iter = iter(swerve_stream)
    stay_item = next_or_none(stay_iter)
    swerve_item = next_or_none(swerve_iter)

    matched_pairs = 0
    stay_unpaired = 0
    swerve_unpaired = 0
    start = time.time()

    try:
        while stay_item is not None and swerve_item is not None:
            stay_id, stay_row = stay_item
            swerve_id, swerve_row = swerve_item
            if stay_id == swerve_id:
                matched_pairs += 1
                if progress_every and matched_pairs % progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "matched_pairs": matched_pairs,
                                "elapsed_seconds": round(time.time() - start, 1),
                            }
                        ),
                        flush=True,
                    )
                yield stay_row, swerve_row, index
                stay_item = next_or_none(stay_iter)
                swerve_item = next_or_none(swerve_iter)
            elif stay_id < swerve_id:
                stay_unpaired += 1
                stay_item = next_or_none(stay_iter)
            else:
                swerve_unpaired += 1
                swerve_item = next_or_none(swerve_iter)

        while stay_item is not None:
            stay_unpaired += 1
            stay_item = next_or_none(stay_iter)
        while swerve_item is not None:
            swerve_unpaired += 1
            swerve_item = next_or_none(swerve_iter)
    finally:
        stay_code, stay_stderr = stay_stream.close()
        swerve_code, swerve_stderr = swerve_stream.close()
        paired_rows.summary = {
            "matched_pairs": matched_pairs,
            "stay_rows_without_swerve_pair": stay_unpaired,
            "swerve_rows_without_stay_pair": swerve_unpaired,
            "stay_filtered_rows_seen": stay_stream.filtered_rows_seen,
            "swerve_filtered_rows_seen": swerve_stream.filtered_rows_seen,
            "stay_stream_unsorted_transitions": stay_stream.unsorted_transitions,
            "swerve_stream_unsorted_transitions": swerve_stream.unsorted_transitions,
            "stream_exit_codes": {"stay_stream": stay_code, "swerve_stream": swerve_code},
            "stream_stderr": {
                "stay_stream": stay_stderr.strip(),
                "swerve_stream": swerve_stderr.strip(),
            },
        }


paired_rows.summary = {}


def choice_label(stay_row: list[str], swerve_row: list[str], index: dict[str, int]) -> str | None:
    stay_dict = {name: value(stay_row, index, name) for name in ["Saved"]}
    swerve_dict = {name: value(swerve_row, index, name) for name in ["Saved"]}
    return gold_choice(stay_dict, swerve_dict)


def scenario_example(stay_row: list[str], swerve_row: list[str], index: dict[str, int]) -> dict[str, Any]:
    stay_dict = {name: value(stay_row, index, name) for name in index}
    swerve_dict = {name: value(swerve_row, index, name) for name in index}
    return {
        "scenario_type": stay_dict.get("ScenarioType", ""),
        "scenario_type_strict": stay_dict.get("ScenarioTypeStrict", ""),
        "pedped": stay_dict.get("PedPed", ""),
        "stay_outcome": action_outcome_text(stay_dict, action="stay"),
        "swerve_outcome": action_outcome_text(swerve_dict, action="swerve"),
    }


def top_entry(
    *,
    scenario_hash: bytes,
    count: int,
    examples: dict[bytes, dict[str, Any]],
    choice_counts: dict[bytes, Counter[str]],
    country_counts: Counter[tuple[bytes, str]],
) -> dict[str, Any]:
    countries = [
        {"country": country, "count": country_count}
        for (hash_value, country), country_count in country_counts.most_common()
        if hash_value == scenario_hash
    ][:10]
    return {
        "scenario_hash": scenario_hash.hex(),
        "count": count,
        "choice_counts": dict(sorted(choice_counts.get(scenario_hash, Counter()).items())),
        "top_countries": countries,
        "example": examples.get(scenario_hash, {}),
    }


def main() -> None:
    args = parse_args()
    start = time.time()

    scenario_counts: Counter[bytes] = Counter()
    for stay_row, swerve_row, index in paired_rows(args.input, args.progress_every):
        scenario_counts[visible_signature(stay_row, swerve_row, index)] += 1

    first_pass_summary = dict(paired_rows.summary)
    threshold = args.threshold
    high_repeat_hashes = {hash_value for hash_value, count in scenario_counts.items() if count > threshold}
    top_hashes = {hash_value for hash_value, _ in scenario_counts.most_common(args.top_n)}
    hashes_to_describe = high_repeat_hashes | top_hashes

    examples: dict[bytes, dict[str, Any]] = {}
    choice_counts: dict[bytes, Counter[str]] = {hash_value: Counter() for hash_value in hashes_to_describe}
    country_counts: Counter[tuple[bytes, str]] = Counter()

    for stay_row, swerve_row, index in paired_rows(args.input, args.progress_every):
        scenario_hash = visible_signature(stay_row, swerve_row, index)
        if scenario_hash not in hashes_to_describe:
            continue
        if scenario_hash not in examples:
            examples[scenario_hash] = scenario_example(stay_row, swerve_row, index)
        country = value(stay_row, index, "UserCountry3")
        if country:
            country_counts[(scenario_hash, country)] += 1
        choice = choice_label(stay_row, swerve_row, index)
        if choice:
            choice_counts[scenario_hash][choice] += 1

    second_pass_summary = dict(paired_rows.summary)
    top_global = [
        top_entry(
            scenario_hash=hash_value,
            count=count,
            examples=examples,
            choice_counts=choice_counts,
            country_counts=country_counts,
        )
        for hash_value, count in scenario_counts.most_common(args.top_n)
    ]
    top_country_cells = [
        {
            "scenario_hash": hash_value.hex(),
            "country": country,
            "count": count,
            "global_count": scenario_counts[hash_value],
            "choice_counts": dict(sorted(choice_counts.get(hash_value, Counter()).items())),
            "example": examples.get(hash_value, {}),
        }
        for (hash_value, country), count in country_counts.most_common(args.top_n)
    ]

    result = {
        "source_file": str(args.input),
        "definition": (
            "Exact visible Stay/Swerve pair: paired outcomes with the same ResponseID, using Barrier, "
            "CrossingSignal, and all character counts for the Stay and Swerve outcomes. Respondent/session/"
            "order/choice/presentation fields are excluded."
        ),
        "threshold_strictly_greater_than": threshold,
        "matched_pairs": first_pass_summary.get("matched_pairs"),
        "unique_visible_scenarios": len(scenario_counts),
        "global_repeat_thresholds": {
            f">{threshold}": sum(1 for count in scenario_counts.values() if count > threshold),
            ">=100": sum(1 for count in scenario_counts.values() if count >= 100),
            ">=500": sum(1 for count in scenario_counts.values() if count >= 500),
            ">=1000": sum(1 for count in scenario_counts.values() if count >= 1000),
        },
        "max_global_count": top_global[0]["count"] if top_global else 0,
        "max_country_cell_count": top_country_cells[0]["count"] if top_country_cells else 0,
        "top_global_scenarios": top_global,
        "top_country_scenario_cells": top_country_cells,
        "first_pass_summary": first_pass_summary,
        "second_pass_summary": second_pass_summary,
        "elapsed_seconds": time.time() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
