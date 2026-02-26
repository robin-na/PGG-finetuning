from __future__ import annotations

import ast
import datetime
import json
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence


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
    base = path.split("/")[-1]
    return f"{directory}/{base}" if base else directory


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


def parse_dict(value: Any) -> Dict[str, int]:
    if isinstance(value, dict):
        out: Dict[str, int] = {}
        for k, v in value.items():
            try:
                iv = int(v)
            except Exception:
                continue
            out[str(k)] = iv
        return out
    if value is None:
        return {}
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return {}
        for loader in (ast.literal_eval, json.loads):
            try:
                obj = loader(s)
                if isinstance(obj, dict):
                    out: Dict[str, int] = {}
                    for k, v in obj.items():
                        try:
                            iv = int(v)
                        except Exception:
                            continue
                        out[str(k)] = iv
                    return out
            except Exception:
                continue
    return {}


def format_num(value: Any) -> str:
    if value is None:
        return "NA"
    if is_nan(value):
        return "NA"
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_avatar(name: Any) -> str:
    if name is None:
        return ""
    return str(name).strip().upper()


def make_unique_avatar_map(player_ids: Sequence[str], raw_avatar_by_player: Mapping[str, Any]) -> Dict[str, str]:
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


def _decode_gender(row: Mapping[str, Any]) -> str:
    for col, label in (
        ("gender_man", "man"),
        ("gender_woman", "woman"),
        ("gender_non_binary", "non-binary"),
        ("gender_unknown", "unknown"),
    ):
        try:
            if int(float(row.get(col, 0) or 0)) == 1:
                return label
        except Exception:
            continue
    code = row.get("gender_code")
    return f"code_{code}" if code is not None else "unknown"


def _decode_education(row: Mapping[str, Any]) -> str:
    for col, label in (
        ("education_high_school", "high_school"),
        ("education_bachelor", "bachelor"),
        ("education_master", "master"),
        ("education_other", "other"),
        ("education_unknown", "unknown"),
    ):
        try:
            if int(float(row.get(col, 0) or 0)) == 1:
                return label
        except Exception:
            continue
    code = row.get("education_code")
    return f"code_{code}" if code is not None else "unknown"


def demographics_line(row: Optional[Mapping[str, Any]]) -> str:
    if not row:
        return "Your demographic profile: unavailable."
    age_missing = as_bool(row.get("age_missing"))
    age_val = row.get("age")
    age = "unknown"
    if not age_missing and age_val is not None and not is_nan(age_val):
        try:
            f = float(age_val)
            age = str(int(f)) if f.is_integer() else str(f)
        except Exception:
            age = str(age_val)
    gender = _decode_gender(row)
    education = _decode_education(row)
    return f"Your demographic profile: age={age}; gender={gender}; education={education}."

