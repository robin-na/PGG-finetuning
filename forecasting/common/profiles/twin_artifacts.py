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


def _canonical_region(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered.startswith("midwest"):
        return "Midwest"
    if lowered.startswith("northeast"):
        return "Northeast"
    if lowered.startswith("south"):
        return "South"
    if lowered.startswith("west") or lowered.startswith("pacific"):
        return "West"
    return None


def _canonical_race(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in {"asian", "black", "hispanic", "white", "other"}:
        return lowered
    return None


def _canonical_relationship_status(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if "living with a partner" in lowered:
        return "living with a partner"
    if "married" in lowered:
        return "married"
    if "widowed" in lowered:
        return "widowed"
    if "divorc" in lowered:
        return "divorced"
    if "separat" in lowered:
        return "separated"
    if "never" in lowered:
        return "never been married"
    return None


def _canonical_religion(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    mapping = {
        "agnostic": "agnostic",
        "atheist": "atheist",
        "buddhist": "buddhist",
        "hindu": "hindu",
        "jewish": "jewish",
        "mormon": "mormon",
        "muslim": "muslim",
        "orthodox": "orthodox",
        "protestant": "protestant",
        "roman catholic": "roman catholic",
        "catholic": "roman catholic",
        "nothing in particular": "nothing in particular",
        "no religion": "nothing in particular",
    }
    for needle, canonical in mapping.items():
        if needle in lowered:
            return canonical
    return "other"


def _canonical_religious_attendance(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    allowed = {
        "never",
        "seldom",
        "a few times a year",
        "once or twice a month",
        "once a week",
        "more than once a week",
    }
    return lowered if lowered in allowed else None


def _canonical_party(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered == "democrat":
        return "Democrat"
    if lowered == "independent":
        return "Independent"
    if lowered == "republican":
        return "Republican"
    if lowered in {"other", "something else"}:
        return "Other"
    return None


def _canonical_political_views(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    mapping = {
        "very liberal": "very liberal",
        "liberal": "liberal",
        "moderate": "moderate",
        "conservative": "conservative",
        "very conservative": "very conservative",
    }
    return mapping.get(lowered)


def _canonical_ideology_broad(value: Any) -> str | None:
    exact = _canonical_political_views(value)
    if exact is None:
        return None
    if "liberal" in exact:
        return "liberal"
    if "conservative" in exact:
        return "conservative"
    return "moderate"


def _canonical_income(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    allowed = {
        "$100,000 or more",
        "$30,000-$50,000",
        "$50,000-$75,000",
        "$75,000-$100,000",
        "Less than $30,000",
    }
    return text if text in allowed else None


def _canonical_employment_broad(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if "retired" in lowered:
        return "retired"
    if "student" in lowered:
        return "student"
    if "home" in lowered:
        return "homemaker"
    if "self-employed" in lowered:
        return "employed"
    if "full-time" in lowered or "part-time" in lowered:
        return "employed"
    if "unemployed" in lowered:
        return "unemployed"
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
            region = _clean_text(features.get("region"))
            race_or_origin = _clean_text(features.get("race_or_origin"))
            relationship_status = _clean_text(features.get("relationship_status"))
            religion = _clean_text(features.get("religion"))
            religious_service_attendance = _clean_text(features.get("religious_service_attendance"))
            party_identification = _clean_text(features.get("party_identification"))
            income_bracket = _clean_text(features.get("income_bracket"))
            political_views = _clean_text(features.get("political_views"))
            employment_status = _clean_text(features.get("employment_status"))
            rows.append(
                {
                    "twin_pid": pid,
                    "age_bracket": age,
                    "education_harmonized": education,
                    "sex_assigned_at_birth": sex,
                    "matching_age_bracket": age,
                    "matching_education": education,
                    "matching_sex": sex,
                    "region": region,
                    "race_or_origin": race_or_origin,
                    "relationship_status": relationship_status,
                    "religion": religion,
                    "religious_service_attendance": religious_service_attendance,
                    "party_identification": party_identification,
                    "income_bracket": income_bracket,
                    "political_views": political_views,
                    "employment_status": employment_status,
                    "matching_region": _canonical_region(region),
                    "matching_race": _canonical_race(race_or_origin),
                    "matching_marital": _canonical_relationship_status(relationship_status),
                    "matching_religion": _canonical_religion(religion),
                    "matching_religattend": _canonical_religious_attendance(
                        religious_service_attendance
                    ),
                    "matching_party": _canonical_party(party_identification),
                    "matching_political_views": _canonical_political_views(political_views),
                    "matching_ideology_broad": _canonical_ideology_broad(political_views),
                    "matching_income": _canonical_income(income_bracket),
                    "matching_employment_broad": _canonical_employment_broad(employment_status),
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
