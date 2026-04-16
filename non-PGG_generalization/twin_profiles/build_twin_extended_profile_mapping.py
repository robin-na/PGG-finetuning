#!/usr/bin/env python3
"""Build per-entry Twin extended-profile section mappings."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


THIS_DIR = Path(__file__).resolve().parent
CATALOG_PATH = THIS_DIR / "wave123_question_catalog.csv"
INVENTORY_PATH = THIS_DIR / "twin_question_inventory.csv"
OUT_PATH = THIS_DIR / "twin_extended_profile_mapping.csv"


def normalize_block_name(value: str) -> str:
    return " ".join((value or "").split())


def load_catalog_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_columns = json.loads(row["csv_columns_json"])
            row = dict(row)
            row["block_name_clean"] = normalize_block_name(row["block_name"])
            row["n_csv_columns"] = str(len(csv_columns))
            row["question_ref"] = f"{row['block_name_clean']}::{row['question_id']}"
            row["csv_columns_semicolon"] = ";".join(csv_columns)
            rows.append(row)
    return rows


def load_family_lookup(path: Path) -> Dict[Tuple[str, str, str, str], str]:
    lookup: Dict[Tuple[str, str, str, str], str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                normalize_block_name(row["block_name"]),
                row["question_id"],
                row["question_type"],
                row["n_csv_columns"],
            )
            lookup[key] = row["family"]
    return lookup


def family_for_row(row: Dict[str, str], family_lookup: Dict[Tuple[str, str, str, str], str]) -> str:
    key = (
        row["block_name_clean"],
        row["question_id"],
        row["question_type"],
        row["n_csv_columns"],
    )
    return family_lookup.get(key, "")


def classify_row(row: Dict[str, str], family: str) -> Dict[str, str]:
    block = row["block_name_clean"]
    n_csv_columns = int(row["n_csv_columns"])
    question_type = row["question_type"]

    if n_csv_columns == 0:
        return {
            "profile_section": "exclude",
            "profile_subsection": "scaffolding",
            "retention_mode": "drop",
            "derived_targets": "",
            "readable_card_priority": "exclude",
            "notes": "Instruction, section-intro, or transition page with no participant response.",
        }

    if block == "Demographics":
        return {
            "profile_section": "background_context",
            "profile_subsection": "demographics",
            "retention_mode": "raw_plus_harmonized",
            "derived_targets": "context_only",
            "readable_card_priority": "selective",
            "notes": "Keep as participant context. Do not over-interpret as deep trait evidence.",
        }

    if block == "Personality":
        if question_type == "TE":
            return {
                "profile_section": "observed_in_twin",
                "profile_subsection": "open_text_responses",
                "retention_mode": "raw_text_plus_summary",
                "derived_targets": (
                    "behavioral_signature;social_style;decision_style;"
                    "derived_dimensions.self_regulation_and_affect"
                ),
                "readable_card_priority": "high",
                "notes": "Open self-description text. Preserve raw excerpts and summarize cautiously.",
            }
        return {
            "profile_section": "observed_in_twin",
            "profile_subsection": "personality_and_self_report",
            "retention_mode": "raw_plus_block_summary",
            "derived_targets": (
                "behavioral_signature;social_style;decision_style;"
                "derived_dimensions.social_preferences;"
                "derived_dimensions.self_regulation_and_affect;"
                "pgg_relevant_cues"
            ),
            "readable_card_priority": "high",
            "notes": "Core social/personality evidence. Large matrices should be summarized into subscales plus diagnostic items.",
        }

    if family in {"trust", "ultimatum", "dictator"}:
        return {
            "profile_section": "observed_in_twin",
            "profile_subsection": "social_game_behavior",
            "retention_mode": "raw_plus_block_summary",
            "derived_targets": (
                "behavioral_signature;social_style;"
                "derived_dimensions.social_preferences;"
                "pgg_relevant_cues"
            ),
            "readable_card_priority": "high",
            "notes": "Direct social-preference game evidence. Keep observed behavior distinct from later transfer hypotheses.",
        }

    if family in {"mental_accounting", "time_preference", "risk_preference_gain", "risk_preference_loss"}:
        return {
            "profile_section": "observed_in_twin",
            "profile_subsection": "economic_preferences_non_social",
            "retention_mode": "raw_plus_block_summary",
            "derived_targets": (
                "decision_style;derived_dimensions.decision_style;pgg_relevant_cues"
            ),
            "readable_card_priority": "high",
            "notes": "Behavioral-econ evidence that may matter for later transfer, but should remain descriptive here.",
        }

    if family == "cognitive_tests":
        return {
            "profile_section": "observed_in_twin",
            "profile_subsection": "cognitive_performance",
            "retention_mode": "score_then_summarize",
            "derived_targets": "decision_style;derived_dimensions.decision_style",
            "readable_card_priority": "medium",
            "notes": "Prefer subscores and metacognitive gaps over item-by-item narration in the readable card.",
        }

    if block == "Product Preferences - Pricing":
        return {
            "profile_section": "observed_in_twin",
            "profile_subsection": "pricing_and_consumer_choice",
            "retention_mode": "score_then_summarize",
            "derived_targets": "decision_style;derived_dimensions.consumer_style",
            "readable_card_priority": "medium",
            "notes": "Large repetitive product-choice block. Summarize into price sensitivity, search willingness, and reference dependence.",
        }

    if block == "Forward Flow":
        return {
            "profile_section": "observed_in_twin",
            "profile_subsection": "open_text_responses",
            "retention_mode": "raw_text_plus_summary",
            "derived_targets": "behavioral_signature;decision_style",
            "readable_card_priority": "low",
            "notes": "Open free-association text. Keep available, but do not over-weight it relative to structured evidence.",
        }

    return {
        "profile_section": "observed_in_twin",
        "profile_subsection": "heuristics_and_biases",
        "retention_mode": "raw_plus_block_summary",
        "derived_targets": "decision_style;derived_dimensions.decision_style;derived_dimensions.consumer_style",
        "readable_card_priority": "medium",
        "notes": "Scenario-based heuristics/biases evidence. Use for decision-style summaries and a few diagnostic examples.",
    }


def build_rows(catalog_rows: Iterable[Dict[str, str]], family_lookup: Dict[Tuple[str, str, str, str], str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in catalog_rows:
        family = family_for_row(row, family_lookup)
        mapping = classify_row(row, family)
        out.append(
            {
                "question_ref": row["question_ref"],
                "question_id": row["question_id"],
                "block_name": row["block_name_clean"],
                "question_type": row["question_type"],
                "family": family or "unmapped",
                "csv_columns_json": row["csv_columns_json"],
                "n_csv_columns": row["n_csv_columns"],
                "profile_section": mapping["profile_section"],
                "profile_subsection": mapping["profile_subsection"],
                "retention_mode": mapping["retention_mode"],
                "derived_targets": mapping["derived_targets"],
                "readable_card_priority": mapping["readable_card_priority"],
                "question_text_short": row["question_text_full"][:180],
                "notes": mapping["notes"],
            }
        )
    return out


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    catalog_rows = load_catalog_rows(CATALOG_PATH)
    family_lookup = load_family_lookup(INVENTORY_PATH)
    rows = build_rows(catalog_rows, family_lookup)
    write_csv(OUT_PATH, rows)

    retained = sum(1 for row in rows if row["profile_section"] != "exclude")
    dropped = len(rows) - retained
    print(f"Wrote {OUT_PATH}")
    print(f"Rows: {len(rows)} | retained: {retained} | excluded scaffolding: {dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
