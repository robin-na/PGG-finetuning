#!/usr/bin/env python3
"""Build the held-out-family benchmark spec for PGG transfer on Twin-2K-500."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from datasets import load_dataset, load_from_disk
from huggingface_hub import hf_hub_download


REPO_ID = "LLM-Digital-Twin/Twin-2K-500"
CONFIG = "wave_split"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
OUT_DIR = SCRIPT_DIR
INVENTORY_CSV = PROJECT_ROOT / "non-PGG_generalization" / "twin_profiles" / "twin_question_inventory.csv"
LOCAL_WAVE_SPLIT = PROJECT_ROOT / "non-PGG_generalization" / "data" / "Twin-2k-500" / "wave_split_dataset"
QUESTION_CATALOG_FILE = "question_catalog_and_human_response_csv/question_catalog.json"
LOCAL_QUESTION_CATALOG = (
    PROJECT_ROOT / "non-PGG_generalization" / "data" / "Twin-2k-500" / "snapshot" / QUESTION_CATALOG_FILE
)


PROFILE_FAMILIES = ["demographics", "personality", "cognitive_tests"]
PRIMARY_TARGET_FAMILIES = ["trust", "ultimatum", "dictator"]
SECONDARY_ECON_FAMILIES = [
    "mental_accounting",
    "time_preference",
    "risk_preference_gain",
    "risk_preference_loss",
]
SOCIAL_GAME_FAMILIES = ["trust", "ultimatum", "dictator"]
ALL_NON_TARGET_ECON_FAMILIES = SOCIAL_GAME_FAMILIES + SECONDARY_ECON_FAMILIES


CONDITIONS = {
    "main": {
        "label": "Main benchmark",
        "description": (
            "Use the full Twin profile plus all non-target economic families. "
            "Exclude the target family entirely."
        ),
    },
    "intermediate": {
        "label": "Intermediate ablation",
        "description": (
            "Use the full Twin profile plus only non-social economic families "
            "(mental accounting, time preference, risk preference)."
        ),
    },
    "strict": {
        "label": "Strict ablation",
        "description": (
            "Use the full Twin profile with no economic-preference items at all."
        ),
    },
}


def load_inventory(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_question_catalog() -> List[Dict[str, object]]:
    try:
        catalog_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=QUESTION_CATALOG_FILE,
            repo_type="dataset",
        )
    except Exception:
        if not LOCAL_QUESTION_CATALOG.exists():
            raise
        catalog_path = str(LOCAL_QUESTION_CATALOG)
    with Path(catalog_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_wave_split():
    try:
        return load_dataset(REPO_ID, CONFIG)["data"]
    except Exception:
        if not LOCAL_WAVE_SPLIT.exists():
            raise
        return load_from_disk(str(LOCAL_WAVE_SPLIT))["data"]


def ref_for_row(row: Dict[str, str]) -> str:
    return f"{row['block_name']}::{row['question_id']}"


def build_source_by_ref(catalog: Iterable[Dict[str, object]]) -> Dict[str, str]:
    source_by_ref: Dict[str, str] = {}
    for q in catalog:
        ref = f"{q.get('BlockName', '')}::{q.get('QuestionID', '')}"
        source_by_ref[ref] = str(q.get("source", ""))
    return source_by_ref


def families_for_condition(condition: str, target_family: str) -> List[str]:
    if condition == "main":
        econ_families = [
            family
            for family in ALL_NON_TARGET_ECON_FAMILIES
            if family != target_family
        ]
    elif condition == "intermediate":
        econ_families = list(SECONDARY_ECON_FAMILIES)
    elif condition == "strict":
        econ_families = []
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return PROFILE_FAMILIES + econ_families


def qids_for_family(
    rows: Iterable[Dict[str, str]],
    family: str,
    include_text: bool,
    source_by_ref: Dict[str, str],
) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        if row["family"] != family:
            continue
        if source_by_ref.get(ref_for_row(row)) != "wave1_3_persona_json":
            continue
        if row["question_type"] == "DB":
            continue
        if not include_text and row["question_type"] == "TE":
            continue
        out.append(row)
    return out


def target_rows(
    rows: Iterable[Dict[str, str]],
    target_family: str,
    source_by_ref: Dict[str, str],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    all_rows = qids_for_family(rows, target_family, include_text=True, source_by_ref=source_by_ref)
    choice_rows = [row for row in all_rows if row["question_type"] != "TE"]
    return all_rows, choice_rows


def build_ref_lists(
    inventory_rows: List[Dict[str, str]],
    target_family: str,
    condition: str,
    source_by_ref: Dict[str, str],
) -> Dict[str, List[Dict[str, str]]]:
    allowed_families = set(families_for_condition(condition, target_family))
    target_all_rows, target_choice_rows = target_rows(
        inventory_rows,
        target_family,
        source_by_ref=source_by_ref,
    )

    allowed_rows = [
        row
        for row in inventory_rows
        if row["family"] in allowed_families
        and source_by_ref.get(ref_for_row(row)) == "wave1_3_persona_json"
        and row["question_type"] != "DB"
        and int(row["n_csv_columns"] or 0) > 0
    ]
    excluded_rows = [
        row
        for row in inventory_rows
        if source_by_ref.get(ref_for_row(row)) == "wave1_3_persona_json"
        and row["family"] not in allowed_families
        and row["question_type"] != "DB"
        and int(row["n_csv_columns"] or 0) > 0
    ]

    return {
        "target_all": all_rows_to_refs(target_all_rows),
        "target_choice": all_rows_to_refs(target_choice_rows),
        "allowed": all_rows_to_refs(allowed_rows),
        "excluded": all_rows_to_refs(excluded_rows),
    }


def all_rows_to_refs(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    refs = []
    seen: Set[str] = set()
    for row in rows:
        ref = ref_for_row(row)
        if ref in seen:
            continue
        seen.add(ref)
        refs.append(
            {
                "ref": ref,
                "block_name": row["block_name"],
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "family": row["family"],
                "question_text_short": row["question_text_short"],
            }
        )
    refs.sort(key=lambda r: (r["family"], r["block_name"], r["question_id"]))
    return refs


def availability_by_ref(ds) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for example in ds:
        blocks = json.loads(example["wave1_3_persona_json"])
        present_refs: Set[str] = set()
        for block in blocks:
            block_name = block.get("BlockName", "")
            for q in block.get("Questions", []):
                ref = f"{block_name}::{q.get('QuestionID', '')}"
                answers = q.get("Answers", {})
                if not answers:
                    continue
                if answers.get("SelectedByPosition") is not None:
                    present_refs.add(ref)
                    continue
                if answers.get("SelectedText") is not None:
                    present_refs.add(ref)
                    continue
        for ref in present_refs:
            counts[ref] += 1
    return counts


def benchmark_cells(
    inventory_rows: List[Dict[str, str]],
    ref_availability: Dict[str, int],
    n_participants: int,
    source_by_ref: Dict[str, str],
) -> List[Dict[str, object]]:
    cells = []
    for target_family in PRIMARY_TARGET_FAMILIES:
        target_all_rows, target_choice_rows = target_rows(
            inventory_rows,
            target_family,
            source_by_ref=source_by_ref,
        )
        target_all_refs = {ref_for_row(row) for row in target_all_rows}
        target_choice_refs = {ref_for_row(row) for row in target_choice_rows}
        target_all_availability = sorted(ref_availability.get(ref, 0) for ref in target_all_refs)
        target_choice_availability = sorted(ref_availability.get(ref, 0) for ref in target_choice_refs)

        for condition in CONDITIONS:
            allowed_families = families_for_condition(condition, target_family)
            ref_lists = build_ref_lists(
                inventory_rows,
                target_family,
                condition,
                source_by_ref=source_by_ref,
            )
            cells.append(
                {
                    "target_family": target_family,
                    "condition": condition,
                    "condition_label": CONDITIONS[condition]["label"],
                    "allowed_input_families": allowed_families,
                    "target_question_count": len(ref_lists["target_all"]),
                    "target_choice_question_count": len(ref_lists["target_choice"]),
                    "allowed_input_question_count": len(ref_lists["allowed"]),
                    "excluded_input_question_count": len(ref_lists["excluded"]),
                    "target_all_refs": ref_lists["target_all"],
                    "target_choice_refs": ref_lists["target_choice"],
                    "allowed_input_refs": ref_lists["allowed"],
                    "excluded_input_refs": ref_lists["excluded"],
                    "target_all_min_availability": min(target_all_availability) if target_all_availability else 0,
                    "target_all_max_availability": max(target_all_availability) if target_all_availability else 0,
                    "target_choice_min_availability": min(target_choice_availability) if target_choice_availability else 0,
                    "target_choice_max_availability": max(target_choice_availability) if target_choice_availability else 0,
                    "n_participants_total": n_participants,
                }
            )
    return cells


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_cells_csv(path: Path, cells: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "target_family",
            "condition",
            "condition_label",
            "allowed_input_families",
            "target_question_count",
            "target_choice_question_count",
            "allowed_input_question_count",
            "excluded_input_question_count",
            "target_choice_min_availability",
            "target_choice_max_availability",
            "n_participants_total",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            row = dict(cell)
            row["allowed_input_families"] = ",".join(cell["allowed_input_families"])
            writer.writerow({k: row[k] for k in fieldnames})


def build_markdown(cells: List[Dict[str, object]], out_path: Path) -> None:
    by_target: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for cell in cells:
        by_target[cell["target_family"]].append(cell)

    lines = [
        "# PGG Transfer Benchmark",
        "",
        "## Benchmark Goal",
        "",
        "- Primary benchmark: predict held-out Trust, Ultimatum, or Dictator responses using the rest of the Twin profile.",
        "- Main condition: keep all non-target economic families in the prompt/profile.",
        "- Ablations: intermediate (only non-social economic families) and strict (no economic families).",
        "- Evaluation target: choice questions only in the primary scoreboard. Target-family thought-text items stay excluded from inputs and are not part of the main metric.",
        "",
        "## Why This Is Not The Public Wave-4 Benchmark",
        "",
        "- The released Twin `wave_split` benchmark block does not contain the economic-preference battery.",
        "- This benchmark therefore uses a held-out-family design over the `wave1_3_persona_json` profile instead of the public wave-4 target block.",
        "- If private wave-4 economic-game targets become available later, the same condition definitions can be reused unchanged.",
        "",
        "## Conditions",
        "",
    ]

    for condition, info in CONDITIONS.items():
        lines.append(f"### {info['label']}")
        lines.append("")
        lines.append(f"- `{condition}`: {info['description']}")
        lines.append("")

    lines.extend(
        [
            "## Target Families",
            "",
        ]
    )

    for target_family in PRIMARY_TARGET_FAMILIES:
        lines.append(f"### {target_family.title()}")
        lines.append("")
        target_cells = sorted(by_target[target_family], key=lambda c: ["main", "intermediate", "strict"].index(c["condition"]))
        main_cell = next(cell for cell in target_cells if cell["condition"] == "main")

        target_choice_refs = main_cell["target_choice_refs"]
        target_all_refs = main_cell["target_all_refs"]
        text_refs = [
            ref["question_id"]
            for ref in target_all_refs
            if ref["question_type"] == "TE"
        ]
        choice_qids = [ref["question_id"] for ref in target_choice_refs]

        lines.append(f"- Choice targets: `{', '.join(choice_qids)}`")
        if text_refs:
            lines.append(f"- Exclude target-family thought text from inputs: `{', '.join(text_refs)}`")
        lines.append(
            f"- Availability in cached Twin data: {main_cell['target_choice_min_availability']}/{main_cell['n_participants_total']} "
            f"to {main_cell['target_choice_max_availability']}/{main_cell['n_participants_total']} participants across target choice items."
        )
        lines.append("")
        lines.append("| Condition | Allowed Families | Allowed Input Questions | Excluded Input Questions |")
        lines.append("| --- | --- | ---: | ---: |")
        for cell in target_cells:
            allowed = ", ".join(cell["allowed_input_families"])
            lines.append(
                f"| {cell['condition_label']} | {allowed} | "
                f"{cell['allowed_input_question_count']} | {cell['excluded_input_question_count']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Recommended Scoreboard",
            "",
            "- Primary scoreboard: `main` condition for Trust, Ultimatum, and Dictator.",
            "- Secondary scoreboard: `intermediate` and `strict` ablations for the same three target families.",
            "- Recommended baselines per cell: Twin-only profile, Twin-only + non-target economic behavior, PGG-augmented retrieval, random-PGG retrieval.",
            "- Keep the benchmark keying on `(block_name, question_id)` internally to avoid QID collisions across Twin blocks.",
            "",
            "## Files",
            "",
            "- `pgg_transfer_benchmark_spec.json`",
            "- `pgg_transfer_benchmark_cells.csv`",
        ]
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    inventory_rows = load_inventory(INVENTORY_CSV)
    catalog = load_question_catalog()
    source_by_ref = build_source_by_ref(catalog)
    ds = load_wave_split()
    ref_availability = availability_by_ref(ds)
    cells = benchmark_cells(
        inventory_rows,
        ref_availability,
        len(ds),
        source_by_ref=source_by_ref,
    )

    spec = {
        "benchmark_name": "pgg_transfer_benchmark",
        "dataset": {
            "repo_id": REPO_ID,
            "config": CONFIG,
            "n_participants": len(ds),
            "note": (
                "Primary benchmark uses held-out-family prediction over wave1_3_persona_json "
                "because the public wave-4 target block does not include economic-preference items."
            ),
        },
        "target_families": PRIMARY_TARGET_FAMILIES,
        "profile_families": PROFILE_FAMILIES,
        "secondary_economic_families": SECONDARY_ECON_FAMILIES,
        "conditions": CONDITIONS,
        "benchmark_cells": cells,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "pgg_transfer_benchmark_spec.json", spec)
    write_cells_csv(OUT_DIR / "pgg_transfer_benchmark_cells.csv", cells)
    build_markdown(cells, OUT_DIR / "PGG_TRANSFER_BENCHMARK.md")

    print(f"Wrote {OUT_DIR / 'pgg_transfer_benchmark_spec.json'}")
    print(f"Wrote {OUT_DIR / 'pgg_transfer_benchmark_cells.csv'}")
    print(f"Wrote {OUT_DIR / 'PGG_TRANSFER_BENCHMARK.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
