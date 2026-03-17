#!/usr/bin/env python3
"""Build a retrieval-ready oracle library from completed PGG participants."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Persona.misc.build_transfer_profile_requests import build_rule_summary
from Persona.misc.transfer_profile_data import build_raw_profiles


PERSONA_DIR = PROJECT_ROOT / "Persona"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "non-PGG_generalization" / "pgg_transfer_eval" / "output" / "oracle_library"
)
ORACLE_PATHS = {
    "learn": PERSONA_DIR / "archetype_oracle_gpt51_learn.jsonl",
    "val": PERSONA_DIR / "archetype_oracle_gpt51_val.jsonl",
}
DEMOGRAPHICS_PATHS = {
    "learn": PROJECT_ROOT / "demographics" / "demographics_numeric_learn.csv",
    "val": PROJECT_ROOT / "demographics" / "demographics_numeric_val.csv",
}
SPLITS = ("learn", "val")
ORACLE_SECTION_TITLES = {
    "CONTRIBUTION": "Contribution Pattern",
    "COMMUNICATION": "Communication Style",
    "RESPONSE_TO_END_GAME": "Response To End Game",
    "RESPONSE_TO_OTHERS_OUTCOME": "Response To Others' Outcome",
    "RESPONSE_TO_PUNISHER": "Response To Punisher",
    "RESPONSE_TO_REWARDER": "Response To Rewarder",
    "PUNISHMENT": "Punishment Use",
    "REWARD": "Reward Use",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def age_bucket(age: Optional[float]) -> str:
    if age is None or (isinstance(age, float) and math.isnan(age)):
        return "unknown"
    age_value = float(age)
    if age_value < 30:
        return "18-29"
    if age_value < 50:
        return "30-49"
    if age_value < 65:
        return "50-64"
    return "65+"


def decode_gender(row: Optional[Dict[str, Any]]) -> str:
    if not row:
        return "unknown"
    if int(float(row.get("gender_man", 0) or 0)) == 1:
        return "man"
    if int(float(row.get("gender_woman", 0) or 0)) == 1:
        return "woman"
    if int(float(row.get("gender_non_binary", 0) or 0)) == 1:
        return "non_binary"
    return "unknown"


def decode_education(row: Optional[Dict[str, Any]]) -> str:
    if not row:
        return "unknown"
    if int(float(row.get("education_high_school", 0) or 0)) == 1:
        return "high_school"
    if int(float(row.get("education_bachelor", 0) or 0)) == 1:
        return "bachelor"
    if int(float(row.get("education_master", 0) or 0)) == 1:
        return "master"
    if int(float(row.get("education_other", 0) or 0)) == 1:
        return "other"
    return "unknown"


def demographics_text(row: Optional[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
    if not row:
        return (
            [
                "Age: unknown",
                "Age bucket: unknown",
                "Gender: unknown",
                "Education: unknown",
            ],
            {
                "age": None,
                "age_bucket": "unknown",
                "gender": "unknown",
                "education": "unknown",
                "available": False,
            },
        )
    age = row.get("age")
    age_value = None if age is None or (isinstance(age, float) and math.isnan(age)) else float(age)
    age_bucket_value = age_bucket(age_value)
    gender = decode_gender(row)
    education = decode_education(row)
    lines = [
        f"Age: {int(age_value) if age_value is not None else 'unknown'}",
        f"Age bucket: {age_bucket_value}",
        f"Gender: {gender}",
        f"Education: {education}",
    ]
    return (
        lines,
        {
            "age": age_value,
            "age_bucket": age_bucket_value,
            "gender": gender,
            "education": education,
            "available": True,
        },
    )


def load_demographics(split: str) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    path = DEMOGRAPHICS_PATHS[split]
    rows: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (split, str(row["gameId"]), str(row["playerId"]))
            parsed: Dict[str, Any] = {}
            for k, v in row.items():
                if v == "":
                    parsed[k] = None
                    continue
                if k in {"gameId", "playerId"}:
                    parsed[k] = v
                    continue
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
            rows[key] = parsed
    return rows


def load_oracles(split: str) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    rows = load_jsonl(ORACLE_PATHS[split])
    return {
        (split, str(row["experiment"]), str(row["participant"])): row
        for row in rows
        if row.get("text")
    }


def parse_oracle_sections(text: str) -> List[Tuple[str, str]]:
    matches = list(re.finditer(r"<([A-Z_]+)>", text or ""))
    if not matches:
        cleaned = normalize_whitespace(text or "")
        return [("Archetype", cleaned)] if cleaned else []

    sections: List[Tuple[str, str]] = []
    for idx, match in enumerate(matches):
        tag = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = normalize_whitespace(text[start:end])
        if not body:
            continue
        title = ORACLE_SECTION_TITLES.get(tag, tag.replace("_", " ").title())
        sections.append((title, body))
    return sections


def oracle_filename(split: str, game_id: str, player_id: str) -> str:
    return f"oracle__{split}__{game_id}__{player_id}.md"


def oracle_custom_id(split: str, game_id: str, player_id: str) -> str:
    return f"oracle_pgg::{split}::{game_id}::{player_id}"


def build_attributes(
    split: str,
    profile: Dict[str, Any],
    demographic_summary: Dict[str, Any],
) -> Dict[str, Any]:
    config = profile.get("config", {})
    attrs: Dict[str, Any] = {
        "split": split,
        "complete": True,
        "age_bucket": demographic_summary["age_bucket"],
        "gender": demographic_summary["gender"],
        "education": demographic_summary["education"],
        "demographics_available": bool(demographic_summary["available"]),
        "chat_enabled": bool(config.get("CONFIG_chat") is True),
        "punishment_enabled": bool(config.get("CONFIG_punishmentExists") is True),
        "reward_enabled": bool(config.get("CONFIG_rewardExists") is True),
        "show_n_rounds": bool(config.get("CONFIG_showNRounds") is True),
        "all_or_nothing": bool(config.get("CONFIG_allOrNothing") is True),
    }
    if demographic_summary["age"] is not None:
        attrs["age"] = float(demographic_summary["age"])
    if config.get("CONFIG_playerCount") is not None:
        attrs["player_count"] = int(config["CONFIG_playerCount"])
    if config.get("CONFIG_numRounds") is not None:
        attrs["num_rounds"] = int(config["CONFIG_numRounds"])
    if config.get("CONFIG_multiplier") is not None:
        attrs["multiplier"] = float(config["CONFIG_multiplier"])
    if config.get("CONFIG_endowment") is not None:
        attrs["endowment"] = int(config["CONFIG_endowment"])
    return attrs


def build_document_text(
    split: str,
    profile: Dict[str, Any],
    oracle_text: str,
    demographic_lines: Sequence[str],
) -> str:
    config = profile.get("config", {})
    sections = parse_oracle_sections(oracle_text)
    lines: List[str] = [
        "# Oracle PGG Profile",
        "",
        "This card describes one completed participant observed in a repeated public-goods game.",
        "",
        "## Wave",
        f"- Split: {split}",
        "",
        "## Demographics",
    ]
    lines.extend(f"- {line}" for line in demographic_lines)
    lines.extend(
        [
            "",
            "## Repeated PGG Rules",
        ]
    )
    lines.extend(f"- {line}" for line in build_rule_summary(config))
    lines.extend(
        [
            "",
            "## Transfer Caveat",
            "- The behavior below was observed in this exact repeated public-goods-game environment under the rules above.",
            "- Use this as analogical evidence for other games, not as a claim that the person would behave identically elsewhere.",
            "",
            "## Oracle Archetype",
            "",
        ]
    )
    for title, body in sections:
        lines.append(f"### {title}")
        lines.append(body)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = args.output_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    raw_profiles = build_raw_profiles(SPLITS)
    raw_by_key = {
        (str(row["split"]), str(row["gameId"]), str(row["playerId"])): row
        for row in raw_profiles
    }
    demographics_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    oracle_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for split in SPLITS:
        demographics_by_key.update(load_demographics(split))
        oracle_by_key.update(load_oracles(split))

    manifest_path = args.output_dir / "oracle_library_manifest.jsonl"
    records_path = args.output_dir / "oracle_library_records.jsonl"
    summary_path = args.output_dir / "oracle_library_summary.json"
    preview_path = args.output_dir / "oracle_library_preview.md"

    total_raw = 0
    kept = 0
    skipped_not_complete = 0
    skipped_missing_oracle = 0
    missing_demographics = 0
    first_preview: Optional[str] = None

    with manifest_path.open("w", encoding="utf-8") as manifest_f, records_path.open(
        "w", encoding="utf-8"
    ) as records_f:
        for key in sorted(raw_by_key):
            total_raw += 1
            split, game_id, player_id = key
            profile = raw_by_key[key]
            if not profile.get("played_to_end"):
                skipped_not_complete += 1
                continue
            oracle_row = oracle_by_key.get(key)
            if oracle_row is None or oracle_row.get("game_finished") is not True:
                skipped_missing_oracle += 1
                continue

            demographic_row = demographics_by_key.get(key)
            demographic_lines, demographic_summary = demographics_text(demographic_row)
            if not demographic_summary["available"]:
                missing_demographics += 1

            filename = oracle_filename(split, game_id, player_id)
            custom_id = oracle_custom_id(split, game_id, player_id)
            doc_path = docs_dir / filename
            document_text = build_document_text(
                split=split,
                profile=profile,
                oracle_text=str(oracle_row["text"]),
                demographic_lines=demographic_lines,
            )
            doc_path.write_text(document_text, encoding="utf-8")

            attributes = build_attributes(split, profile, demographic_summary)
            manifest_row = {
                "custom_id": custom_id,
                "split": split,
                "gameId": game_id,
                "playerId": player_id,
                "filename": filename,
                "doc_path": str(doc_path),
                "attributes": attributes,
                "demographics": demographic_summary,
                "config": profile.get("config", {}),
                "rule_summary_lines": build_rule_summary(profile.get("config", {})),
                "oracle_text": str(oracle_row["text"]),
            }
            record_row = {
                **manifest_row,
                "document_text": document_text,
            }
            manifest_f.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
            records_f.write(json.dumps(record_row, ensure_ascii=False) + "\n")

            if first_preview is None:
                first_preview = document_text
            kept += 1
            if args.limit is not None and kept >= args.limit:
                break

    if first_preview is not None:
        preview_path.write_text(first_preview, encoding="utf-8")

    summary = {
        "raw_profile_count": total_raw,
        "kept_records": kept,
        "skipped_not_complete": skipped_not_complete,
        "skipped_missing_oracle": skipped_missing_oracle,
        "missing_demographics": missing_demographics,
        "docs_dir": str(docs_dir),
        "manifest_path": str(manifest_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
