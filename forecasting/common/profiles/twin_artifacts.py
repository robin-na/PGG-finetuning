from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


AGE_ORDER = ["18-29", "30-49", "50-64", "65+"]
EDUCATION_ORDER = ["high school", "college/postsecondary", "postgraduate"]
SEX_ORDER = ["male", "female"]


def _clean_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nan", "none", "na", "n/a", "data_expired", "consent_revoked"}:
        return None
    return text


def _canonical_sex(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in {"male", "man", "m"}:
        return "male"
    if lowered in {"female", "woman", "f"}:
        return "female"
    return None


def extract_harmonized_feature_map(profile: dict[str, Any]) -> dict[str, Any]:
    features = profile.get("background_context", {}).get("harmonized_features", [])
    out: dict[str, Any] = {}
    for feature in features:
        name = feature.get("name")
        value = feature.get("value", {})
        raw = value.get("raw") if isinstance(value, dict) else value
        if name:
            out[str(name)] = raw
    return out


def load_twin_cards(cards_path: Path) -> dict[str, dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    with cards_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            participant = row.get("participant", {})
            pid = str(participant.get("pid") or row.get("profile_id") or "").strip()
            if pid:
                cards[pid] = row
    return cards


def load_twin_personas(
    profiles_path: Path,
    cards_path: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    cards_by_pid = load_twin_cards(cards_path)
    rows: list[dict[str, Any]] = []
    with profiles_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            profile = json.loads(line)
            participant = profile.get("participant", {})
            pid = str(participant.get("pid") or "").strip()
            if not pid or pid not in cards_by_pid:
                continue
            features = extract_harmonized_feature_map(profile)
            age = features.get("age_bracket")
            education = features.get("education_completed_harmonized")
            sex = _canonical_sex(features.get("sex_assigned_at_birth"))
            if age not in AGE_ORDER or education not in EDUCATION_ORDER or sex not in SEX_ORDER:
                continue
            rows.append(
                {
                    "twin_pid": pid,
                    "age_bracket": age,
                    "education_harmonized": education,
                    "sex_assigned_at_birth": sex,
                    "matching_age_bracket": age,
                    "matching_education": education,
                    "matching_sex": sex,
                    "headline": str(cards_by_pid[pid].get("headline", "")).strip(),
                    "summary": str(cards_by_pid[pid].get("summary", "")).strip(),
                    "background_summary": str(
                        (cards_by_pid[pid].get("background") or {}).get("summary", "")
                    ).strip(),
                }
            )
    personas = pd.DataFrame(rows)
    if personas.empty:
        raise ValueError("No Twin personas were loaded from the shared profile artifacts.")
    return personas, cards_by_pid

