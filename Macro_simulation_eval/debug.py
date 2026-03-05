from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional


def compact_text(text: Optional[str], limit: int = 200) -> Optional[str]:
    if text is None:
        return None
    return text[:limit]


def sha256_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_debug_record(
    *,
    game_id: str,
    round_idx: int,
    agent: str,
    phase: str,
    dt_sec: float,
    prompt: str,
    raw_output: str,
    debug_level: str,
    excerpt_chars: int = 200,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "game": game_id,
        "round": round_idx,
        "agent": agent,
        "phase": phase,
        "dt_sec": dt_sec,
    }
    if debug_level == "compact":
        record.update({
            "prompt_excerpt": compact_text(prompt, excerpt_chars),
            "prompt_sha256": sha256_text(prompt),
            "output_excerpt": compact_text(raw_output, excerpt_chars),
            "output_sha256": sha256_text(raw_output),
        })
    else:
        record.update({
            "prompt_full": prompt,
            "raw_output_full": raw_output,
        })
    return record


def build_full_debug_record(
    *,
    game_id: str,
    round_idx: int,
    agent: str,
    phase: str,
    dt_sec: float,
    prompt: str,
    raw_output: str,
) -> Dict[str, Any]:
    return {
        "game": game_id,
        "round": round_idx,
        "agent": agent,
        "phase": phase,
        "dt_sec": dt_sec,
        "prompt_full": prompt,
        "raw_output_full": raw_output,
    }
