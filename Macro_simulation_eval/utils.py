from __future__ import annotations

import datetime
import math
import os
import re
from typing import Any, Dict, Mapping, Optional, Sequence


def log(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def timestamp_yymmddhhmm() -> str:
    return datetime.datetime.now().strftime("%y%m%d%H%M")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def relocate_output(path: Optional[str], directory: str) -> Optional[str]:
    if not path:
        return None
    base = os.path.basename(path)
    return os.path.join(directory, base) if base else directory


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n", "", "nan", "none", "null"}:
        return False
    return bool(s)


def is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except Exception:
        return False


def normalize_avatar(name: Any) -> str:
    if name is None:
        return ""
    return str(name).strip().upper()


def make_unique_avatar_map(
    player_ids: Sequence[str],
    raw_avatar_by_player: Mapping[str, Any],
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    used: Dict[str, str] = {}
    for i, pid in enumerate(player_ids, start=1):
        base = normalize_avatar(raw_avatar_by_player.get(pid))
        if not base:
            base = f"PLAYER_{i}"
        avatar = base
        k = 2
        while avatar in used and used[avatar] != pid:
            avatar = f"{base}_{k}"
            k += 1
        out[pid] = avatar
        used[avatar] = pid
    return out


def _decode_gender(row: Mapping[str, Any]) -> Optional[str]:
    for col, label in (
        ("gender_man", "man"),
        ("gender_woman", "woman"),
        ("gender_non_binary", "non-binary"),
    ):
        try:
            if int(float(row.get(col, 0) or 0)) == 1:
                return label
        except Exception:
            continue
    return None


def _decode_education(row: Mapping[str, Any]) -> Optional[str]:
    for col, label in (
        ("education_high_school", "high_school"),
        ("education_bachelor", "bachelor"),
        ("education_master", "master"),
    ):
        try:
            if int(float(row.get(col, 0) or 0)) == 1:
                return label
        except Exception:
            continue
    return None


def _article_for(next_word: str) -> str:
    if not next_word:
        return "a"
    return "an" if next_word[0].lower() in {"a", "e", "i", "o", "u"} else "a"


def _education_phrase(education: Optional[str]) -> Optional[str]:
    if education == "high_school":
        return "a high school educational background"
    if education == "bachelor":
        return "a bachelor's level educational background"
    if education == "master":
        return "a master's level educational background"
    return None


def demographics_line(row: Optional[Mapping[str, Any]]) -> str:
    if not row:
        return "Your demographic profile is unavailable."
    age_missing = as_bool(row.get("age_missing"))
    age_val = row.get("age")
    age: Optional[str] = None
    if not age_missing and age_val is not None and not is_nan(age_val):
        try:
            f = float(age_val)
            age = str(int(f)) if f.is_integer() else str(f)
        except Exception:
            age = str(age_val)
    gender = _decode_gender(row)
    edu_phrase = _education_phrase(_decode_education(row))

    identity_phrase: Optional[str] = None
    if age and gender in {"man", "woman"}:
        identity_phrase = f"a {age} year old {gender}"
    elif age and gender == "non-binary":
        identity_phrase = f"a {age} year old non-binary person"
    elif age:
        identity_phrase = f"{age} years old"
    elif gender in {"man", "woman"}:
        identity_phrase = f"{_article_for(gender)} {gender}"
    elif gender == "non-binary":
        identity_phrase = "a non-binary person"

    if identity_phrase and edu_phrase:
        return f"You are {identity_phrase} with {edu_phrase}."
    if identity_phrase:
        return f"You are {identity_phrase}."
    if edu_phrase:
        return f"You have {edu_phrase}."
    return "Your demographic profile is unavailable."
