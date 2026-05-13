from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_stay_swerve_batch import action_outcome_text, build_prompt, gold_choice
from count_scenario_repeats import paired_rows
from count_unique_scenarios import CHARACTER_COLUMNS, DEFAULT_INPUT, PAIR_METADATA_COLUMNS, value, visible_signature


SCRIPT_DIR = Path(__file__).resolve().parent
MM_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = MM_ROOT / "processed" / "high_repeat_scenario_cells"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build benchmark-ready aggregate tables for repeated exact Moral Machine scenarios."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-global-count", type=int, default=100)
    parser.add_argument("--min-cell-count", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    parser.add_argument("--no-sqlite", action="store_true")
    parser.add_argument("--include-blank-country", action="store_true")
    return parser.parse_args()


def choice_label(stay_row: list[str], swerve_row: list[str], index: dict[str, int]) -> str | None:
    return gold_choice(
        {"Saved": value(stay_row, index, "Saved")},
        {"Saved": value(swerve_row, index, "Saved")},
    )


def row_dict(row: list[str], index: dict[str, int]) -> dict[str, str]:
    return {name: value(row, index, name) for name in index}


def share(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def scenario_record(
    *,
    scenario_hash: bytes,
    global_count: int,
    stay_row: list[str],
    swerve_row: list[str],
    index: dict[str, int],
) -> dict[str, Any]:
    stay = row_dict(stay_row, index)
    swerve = row_dict(swerve_row, index)
    record: dict[str, Any] = {
        "scenario_hash": scenario_hash.hex(),
        "global_count": global_count,
    }
    for column in PAIR_METADATA_COLUMNS:
        record[column] = stay.get(column, "")
    record["stay_outcome"] = action_outcome_text(stay, action="stay")
    record["swerve_outcome"] = action_outcome_text(swerve, action="swerve")
    record["prompt"] = build_prompt(stay, swerve)
    for action, source in [("stay", stay), ("swerve", swerve)]:
        record[f"{action}_Barrier"] = source.get("Barrier", "")
        record[f"{action}_CrossingSignal"] = source.get("CrossingSignal", "")
        for column in CHARACTER_COLUMNS:
            record[f"{action}_{column}"] = source.get(column, "")
    return record


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def sqlite_type(field: str) -> str:
    if field.endswith("_count") or field in {
        "global_count",
        "cell_count",
        "a_count",
        "b_count",
        "unknown_choice_count",
    }:
        return "INTEGER"
    if field.endswith("_share"):
        return "REAL"
    return "TEXT"


def write_sqlite(path: Path, scenario_rows: list[dict[str, Any]], cell_rows: list[dict[str, Any]]) -> None:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        for table_name, rows in [("scenarios", scenario_rows), ("cells", cell_rows)]:
            if not rows:
                continue
            fields = list(rows[0].keys())
            columns_sql = ", ".join(f'"{field}" {sqlite_type(field)}' for field in fields)
            connection.execute(f'CREATE TABLE "{table_name}" ({columns_sql})')
            placeholders = ", ".join("?" for _ in fields)
            fields_sql = ", ".join(f'"{field}"' for field in fields)
            connection.executemany(
                f'INSERT INTO "{table_name}" ({fields_sql}) VALUES ({placeholders})',
                ([row.get(field, "") for field in fields] for row in rows),
            )
        connection.execute("CREATE INDEX idx_scenarios_global_count ON scenarios(global_count DESC)")
        connection.execute("CREATE INDEX idx_scenarios_hash ON scenarios(scenario_hash)")
        connection.execute("CREATE INDEX idx_cells_count ON cells(cell_count DESC)")
        connection.execute("CREATE INDEX idx_cells_country_count ON cells(user_country3, cell_count DESC)")
        connection.execute("CREATE INDEX idx_cells_scenario ON cells(scenario_hash)")
        connection.commit()
    finally:
        connection.close()


def main() -> None:
    args = parse_args()
    if args.min_global_count <= 1:
        raise ValueError("--min-global-count should be > 1 for repeated scenario aggregation.")
    if args.min_cell_count <= 1:
        raise ValueError("--min-cell-count should be > 1 for repeated cell aggregation.")

    start = time.time()
    scenario_counts: Counter[bytes] = Counter()
    print("pass=1 counting exact scenario repeats", flush=True)
    for stay_row, swerve_row, index in paired_rows(args.input, args.progress_every):
        scenario_counts[visible_signature(stay_row, swerve_row, index)] += 1
    first_pass_summary = dict(paired_rows.summary)

    high_scenario_counts = {
        scenario_hash: count
        for scenario_hash, count in scenario_counts.items()
        if count >= args.min_global_count
    }
    high_scenario_hashes = set(high_scenario_counts)
    print(
        json.dumps(
            {
                "pass": 1,
                "unique_visible_scenarios": len(scenario_counts),
                "high_repeat_scenarios": len(high_scenario_hashes),
                "min_global_count": args.min_global_count,
            }
        ),
        flush=True,
    )

    scenario_choice_counts: dict[bytes, Counter[str]] = defaultdict(Counter)
    cell_choice_counts: dict[tuple[bytes, str], Counter[str]] = defaultdict(Counter)
    examples: dict[bytes, dict[str, Any]] = {}

    print("pass=2 aggregating country cells for high-repeat scenarios", flush=True)
    for stay_row, swerve_row, index in paired_rows(args.input, args.progress_every):
        scenario_hash = visible_signature(stay_row, swerve_row, index)
        if scenario_hash not in high_scenario_hashes:
            continue
        if scenario_hash not in examples:
            examples[scenario_hash] = scenario_record(
                scenario_hash=scenario_hash,
                global_count=high_scenario_counts[scenario_hash],
                stay_row=stay_row,
                swerve_row=swerve_row,
                index=index,
            )
        choice = choice_label(stay_row, swerve_row, index)
        if choice:
            scenario_choice_counts[scenario_hash][choice] += 1
        else:
            scenario_choice_counts[scenario_hash]["unknown"] += 1

        country = value(stay_row, index, "UserCountry3").strip()
        if not country and not args.include_blank_country:
            continue
        if not country:
            country = "UNKNOWN"
        if choice:
            cell_choice_counts[(scenario_hash, country)][choice] += 1
        else:
            cell_choice_counts[(scenario_hash, country)]["unknown"] += 1
    second_pass_summary = dict(paired_rows.summary)

    scenario_rows: list[dict[str, Any]] = []
    for scenario_hash, global_count in sorted(
        high_scenario_counts.items(),
        key=lambda item: (-item[1], item[0].hex()),
    ):
        counts = scenario_choice_counts[scenario_hash]
        a_count = counts.get("A", 0)
        b_count = counts.get("B", 0)
        unknown_count = counts.get("unknown", 0)
        row = dict(examples[scenario_hash])
        row.update(
            {
                "global_a_count": a_count,
                "global_b_count": b_count,
                "global_unknown_choice_count": unknown_count,
                "global_a_share": share(a_count, global_count),
                "global_b_share": share(b_count, global_count),
            }
        )
        scenario_rows.append(row)

    cell_rows: list[dict[str, Any]] = []
    for (scenario_hash, country), counts in cell_choice_counts.items():
        cell_count = sum(counts.values())
        if cell_count < args.min_cell_count:
            continue
        a_count = counts.get("A", 0)
        b_count = counts.get("B", 0)
        unknown_count = counts.get("unknown", 0)
        scenario = examples[scenario_hash]
        cell_rows.append(
            {
                "scenario_hash": scenario_hash.hex(),
                "user_country3": country,
                "cell_count": cell_count,
                "a_count": a_count,
                "b_count": b_count,
                "unknown_choice_count": unknown_count,
                "a_share": share(a_count, cell_count),
                "b_share": share(b_count, cell_count),
                "global_count": high_scenario_counts[scenario_hash],
                "ScenarioType": scenario.get("ScenarioType", ""),
                "ScenarioTypeStrict": scenario.get("ScenarioTypeStrict", ""),
                "PedPed": scenario.get("PedPed", ""),
                "AttributeLevel": scenario.get("AttributeLevel", ""),
            }
        )
    cell_rows.sort(key=lambda row: (-int(row["cell_count"]), row["user_country3"], row["scenario_hash"]))

    scenario_fieldnames = list(scenario_rows[0].keys()) if scenario_rows else []
    cell_fieldnames = list(cell_rows[0].keys()) if cell_rows else []
    output_dir = (
        args.output_root
        / f"global_{args.min_global_count}_cell_{args.min_cell_count}"
    )
    scenarios_csv = output_dir / "scenarios.csv"
    cells_csv = output_dir / "cells.csv"
    scenarios_jsonl = output_dir / "scenarios.jsonl"
    cells_jsonl = output_dir / "cells.jsonl"
    sqlite_path = output_dir / "scenario_cells.sqlite"
    manifest_path = output_dir / "manifest.json"
    readme_path = output_dir / "README.md"

    write_csv(scenarios_csv, scenario_rows, scenario_fieldnames)
    write_csv(cells_csv, cell_rows, cell_fieldnames)
    write_jsonl(scenarios_jsonl, scenario_rows)
    write_jsonl(cells_jsonl, cell_rows)
    if not args.no_sqlite:
        write_sqlite(sqlite_path, scenario_rows, cell_rows)

    country_counts = Counter(row["user_country3"] for row in cell_rows)
    country_participants = Counter()
    for row in cell_rows:
        country_participants[row["user_country3"]] += int(row["cell_count"])
    top_countries = [
        {
            "user_country3": country,
            "cell_count": country_counts[country],
            "summed_responses_across_cells": country_participants[country],
        }
        for country, _ in country_counts.most_common(20)
    ]
    manifest = {
        "source_file": str(args.input),
        "output_dir": str(output_dir),
        "min_global_count": args.min_global_count,
        "min_cell_count": args.min_cell_count,
        "definition": {
            "scenario": (
                "Exact visible Stay/Swerve pair, using Barrier, CrossingSignal, and all character counts "
                "for both outcomes; respondent/session/order/choice/presentation fields excluded."
            ),
            "cell": "Exact visible scenario crossed with UserCountry3.",
            "answer_labels": {"A": "stay", "B": "swerve"},
        },
        "matched_pairs": first_pass_summary.get("matched_pairs"),
        "unique_visible_scenarios": len(scenario_counts),
        "retained_scenarios": len(scenario_rows),
        "retained_cells": len(cell_rows),
        "max_global_count": max(high_scenario_counts.values()) if high_scenario_counts else 0,
        "max_cell_count": max((int(row["cell_count"]) for row in cell_rows), default=0),
        "top_cells": cell_rows[:20],
        "top_countries": top_countries,
        "files": {
            "scenarios_csv": str(scenarios_csv),
            "cells_csv": str(cells_csv),
            "scenarios_jsonl": str(scenarios_jsonl),
            "cells_jsonl": str(cells_jsonl),
            "sqlite": str(sqlite_path) if not args.no_sqlite else None,
            "manifest": str(manifest_path),
        },
        "first_pass_summary": first_pass_summary,
        "second_pass_summary": second_pass_summary,
        "elapsed_seconds": time.time() - start,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    readme_path.write_text(
        "\n".join(
            [
                "# High-Repeat Moral Machine Scenario Cells",
                "",
                "This directory contains benchmark-ready aggregate tables built from exact repeated",
                "Moral Machine Stay/Swerve scenarios.",
                "",
                f"- Minimum global scenario count: `{args.min_global_count}`",
                f"- Minimum scenario-country cell count: `{args.min_cell_count}`",
                "- `scenarios.csv`: one row per retained exact scenario, including prompt-ready text.",
                "- `cells.csv`: one row per retained `scenario_hash × UserCountry3` cell, including observed A/B shares.",
                "- `scenarios.jsonl` and `cells.jsonl`: line-oriented copies of the same tables.",
                "- `scenario_cells.sqlite`: indexed SQLite copy for fast sampling and joins.",
                "",
                "Answer labels: `A = stay`, `B = swerve`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
