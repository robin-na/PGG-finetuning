from __future__ import annotations

import difflib
import re

import pandas as pd

from simulation_statistical.archetype_distribution_embedding.utils.constants import CANONICAL_TAGS


_NORMALIZED_TAG_LOOKUP = {
    re.sub(r"[^A-Z]+", "", tag.upper()): tag for tag in CANONICAL_TAGS
}

_TAG_ALIASES = {
    "PUNISHMENTTYPESUMMARY": "PUNISHMENT",
    "PAINISHMENT": "PUNISHMENT",
    "PAINPT": "PUNISHMENT",
    "PUISHMENT": "PUNISHMENT",
    "PPUNISHMENT": "PUNISHMENT",
    "PUNISHMENTSUMMARY": "PUNISHMENT",
    "PUNSIHMENT": "PUNISHMENT",
    "R": "REWARD",
    "RERWARD": "REWARD",
}


def _normalize_token(value: str) -> str:
    return re.sub(r"[^A-Z]+", "", value.upper().replace("\ufeff", ""))


def canonicalize_tag_name(raw_tag: str) -> str | None:
    normalized = _normalize_token(raw_tag)
    if not normalized:
        return None
    if normalized in _NORMALIZED_TAG_LOOKUP:
        return _NORMALIZED_TAG_LOOKUP[normalized]
    if normalized in _TAG_ALIASES:
        return _TAG_ALIASES[normalized]

    for canonical_norm, canonical in _NORMALIZED_TAG_LOOKUP.items():
        if canonical_norm in normalized or normalized in canonical_norm:
            return canonical

    nearest = difflib.get_close_matches(normalized, list(_NORMALIZED_TAG_LOOKUP), n=1, cutoff=0.6)
    if nearest:
        return _NORMALIZED_TAG_LOOKUP[nearest[0]]
    return None


def normalize_text_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_tag_line(line: str) -> tuple[str | None, str]:
    stripped = line.strip()
    if not stripped.startswith("<"):
        return None, line

    tag_body = stripped[1:]
    remainder = ""
    if ">" in tag_body:
        tag_body, remainder = tag_body.split(">", 1)
        remainder = remainder.strip()
    canonical = canonicalize_tag_name(tag_body)
    if canonical is None:
        return None, line
    return canonical, remainder


def normalize_archetype_text(text: str) -> str:
    clean = normalize_text_whitespace(text or "")
    normalized_lines: list[str] = []
    for line in clean.split("\n"):
        canonical, remainder = _normalize_tag_line(line)
        if canonical is None:
            normalized_lines.append(line)
            continue
        normalized_lines.append(f"<{canonical}>")
        if remainder:
            normalized_lines.append(remainder)
    normalized = "\n".join(normalized_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def normalize_tag_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["archetype_text_clean"] = out["archetype_text_raw"].fillna("").astype(str).map(normalize_archetype_text)
    return out
