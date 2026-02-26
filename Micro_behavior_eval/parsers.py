from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

try:
    from .utils import log
except ImportError:
    from utils import log


def parse_json_response(s: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    if not isinstance(s, str):
        log("[parse] expected string for JSON output; defaulting to None")
        return None, False
    cleaned = s.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*", "", cleaned).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[: -len("```")].strip()
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(cleaned)
        if isinstance(obj, dict):
            return obj, True
    except Exception:
        pass
    if "{" not in cleaned or "}" not in cleaned:
        log("[parse] no JSON object found; defaulting to None")
        return None, False
    start = cleaned.find("{")
    try:
        obj, _ = decoder.raw_decode(cleaned[start:])
        if isinstance(obj, dict):
            return obj, True
    except Exception:
        log("[parse] failed to parse JSON object; defaulting to None")
        return None, False
    log("[parse] invalid JSON output; defaulting to None")
    return None, False
