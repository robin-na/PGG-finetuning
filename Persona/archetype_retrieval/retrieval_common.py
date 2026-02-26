#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


DEFAULT_TAGS = [
    "COMMUNICATION",
    "CONTRIBUTION",
    "PUNISHMENT",
    "RESPONSE_TO_END_GAME",
    "RESPONSE_TO_OTHERS_OUTCOME",
    "RESPONSE_TO_PUNISHER",
    "RESPONSE_TO_REWARDER",
    "REWARD",
]

DEMOGRAPHIC_KEY_COLUMNS = ["gameId", "playerId"]

CONFIG_FEATURE_COLUMNS = [
    "CONFIG_playerCount",
    "CONFIG_numRounds",
    "CONFIG_showNRounds",
    "CONFIG_endowment",
    "CONFIG_multiplier",
    "CONFIG_MPCR",
    "CONFIG_allOrNothing",
    "CONFIG_chat",
    "CONFIG_punishmentExists",
    "CONFIG_punishmentCost",
    "CONFIG_punishmentMagnitude",
    "CONFIG_punishmentTech",
    "CONFIG_rewardExists",
    "CONFIG_rewardCost",
    "CONFIG_rewardMagnitude",
    "CONFIG_rewardTech",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
]

TAG_ACTIVATION_REQUIREMENTS: Dict[str, List[str]] = {
    "CONTRIBUTION": [],
    "COMMUNICATION": ["CONFIG_chat"],
    "PUNISHMENT": ["CONFIG_punishmentExists"],
    "REWARD": ["CONFIG_rewardExists"],
    "RESPONSE_TO_END_GAME": ["CONFIG_showNRounds"],
    "RESPONSE_TO_OTHERS_OUTCOME": ["CONFIG_showOtherSummaries"],
    "RESPONSE_TO_PUNISHER": ["CONFIG_punishmentExists", "CONFIG_showPunishmentId"],
    "RESPONSE_TO_REWARDER": ["CONFIG_rewardExists", "CONFIG_showRewardId"],
}


def parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "1", "yes"}:
            return True
        if low in {"false", "0", "no"}:
            return False
    return None


def normalize_key(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    return str(v)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def pick_embedding_file(tag_dir: Path) -> Path:
    npy_files = sorted(tag_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy file found under {tag_dir}")
    preferred = [p for p in npy_files if "embedding" in p.name.lower()]
    if len(preferred) == 1:
        return preferred[0]
    if len(npy_files) == 1:
        return npy_files[0]
    raise ValueError(f"Could not uniquely identify embedding file under {tag_dir}")


def pick_sections_input_jsonl(tag_dir: Path) -> Path:
    matches = sorted(tag_dir.glob("*_sections_input.jsonl"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No *_sections_input.jsonl file found under {tag_dir}")
    raise ValueError(f"Expected one *_sections_input.jsonl under {tag_dir}, found {len(matches)}")


def series_to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    bool_parsed = series.map(parse_bool)
    if bool_parsed.notna().sum() == series.notna().sum():
        return bool_parsed.astype(float)

    return pd.to_numeric(series, errors="coerce")


def coerce_feature_frame(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_columns:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = series_to_numeric(out[col])
    return out[feature_columns]


def is_tag_active(tag: str, row: Dict[str, Any]) -> bool:
    req = TAG_ACTIVATION_REQUIREMENTS.get(tag.upper())
    if req is None:
        raise KeyError(f"Unknown tag: {tag}")
    for col in req:
        if parse_bool(row.get(col)) is not True:
            return False
    return True


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms

