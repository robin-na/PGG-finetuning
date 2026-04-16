from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    dataset_key: str
    display_name: str
    records: pd.DataFrame
    units: pd.DataFrame
    demographic_source: pd.DataFrame
    twin_matching_fields: list[str]
    supported_variants: list[str] | None = None


def clean_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nan", "none", "na", "n/a", "data_expired", "consent_revoked"}:
        return None
    return text


def clean_numeric_string(value: Any) -> str | None:
    if pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return clean_text(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(round(numeric, 4))


def age_to_bracket(value: Any) -> str | None:
    if pd.isna(value):
        return None
    try:
        age = float(value)
    except (TypeError, ValueError):
        text = clean_text(value)
        if text is None:
            return None
        text = text.replace("years", "").strip()
        if text.endswith("+"):
            try:
                age = float(text[:-1])
            except ValueError:
                return None
        else:
            try:
                age = float(text)
            except ValueError:
                return None
    if age < 30:
        return "18-29"
    if age < 50:
        return "30-49"
    if age < 65:
        return "50-64"
    return "65+"


def canonical_sex(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in {"male", "man", "m"}:
        return "male"
    if lowered in {"female", "woman", "f"}:
        return "female"
    return None


def format_bullet_markdown(pairs: list[tuple[str, str | None]]) -> str:
    lines = [f"- {name}: {value}" for name, value in pairs if value is not None]
    return "\n".join(lines)


def simple_demographic_summary(
    *,
    age: str | None,
    sex_or_gender: str | None,
    education: str | None = None,
) -> tuple[str, str]:
    parts: list[str] = []
    if age is not None and sex_or_gender is not None:
        parts.append(f"{age}-year-old, {sex_or_gender.lower()}")
    elif age is not None:
        parts.append(f"{age}-year-old")
    elif sex_or_gender is not None:
        parts.append(sex_or_gender.lower())
    if education is not None:
        parts.append(education)
    summary = ", ".join(parts).strip()
    if summary:
        summary = summary[0].upper() + summary[1:] + "."
    markdown = format_bullet_markdown(
        [
            ("Age", age),
            ("Sex/gender", sex_or_gender),
            ("Education", education),
        ]
    )
    return summary, markdown


def yes_no(value: Any) -> str:
    return "YES" if int(value) == 1 else "NO"


def first_present_value(row: pd.Series | dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if isinstance(row, dict):
            if key in row:
                return row[key]
        else:
            if key in row.index:
                return row[key]
    raise KeyError(f"None of the requested keys were present: {keys}")
