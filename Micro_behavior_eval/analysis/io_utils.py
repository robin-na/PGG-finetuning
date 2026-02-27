from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd


REQUIRED_COLUMNS = [
    "gameId",
    "roundIndex",
    "playerId",
    "actual_available",
    "predicted_contribution",
    "predicted_contribution_parsed",
    "actual_contribution",
    "predicted_punished_pid",
    "predicted_rewarded_pid",
    "actual_punished_pid",
    "actual_rewarded_pid",
]

ACTION_DICT_COLUMNS = [
    "predicted_punished_pid",
    "predicted_rewarded_pid",
    "actual_punished_pid",
    "actual_rewarded_pid",
]


@dataclass
class FilterSummary:
    pre_filter_rows: int
    post_filter_rows: int
    dropped_rows: int
    min_round: Optional[int]
    max_round: Optional[int]
    skip_no_actual: bool


@dataclass
class ParseSummary:
    malformed_counts: Dict[str, int]
    malformed_rows: int
    total_rows: int


def load_eval_csv(eval_csv: str | Path) -> pd.DataFrame:
    path = Path(eval_csv)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {path}")
    return pd.read_csv(path)


def validate_required_columns(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _is_missing(value: Any) -> bool:
    try:
        return pd.isna(value)
    except Exception:
        return False


def parse_bool(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None or _is_missing(value):
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return default


def _coerce_action_map(obj: Any) -> Tuple[Dict[str, int], bool]:
    if obj is None or _is_missing(obj):
        return {}, False
    if not isinstance(obj, dict):
        return {}, True
    out: Dict[str, int] = {}
    malformed = False
    for key, value in obj.items():
        pid = str(key).strip()
        if not pid:
            malformed = True
            continue
        try:
            units = int(value)
        except Exception:
            malformed = True
            continue
        if units > 0:
            out[pid] = units
    return out, malformed


def parse_action_dict(value: Any) -> Tuple[Dict[str, int], bool]:
    if isinstance(value, dict):
        return _coerce_action_map(value)
    if value is None or _is_missing(value):
        return {}, False
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "{}"}:
        return {}, False

    parsed_obj = None
    try:
        parsed_obj = json.loads(text)
    except Exception:
        try:
            parsed_obj = ast.literal_eval(text)
        except Exception:
            return {}, True

    return _coerce_action_map(parsed_obj)


def _parsed_col_name(column: str) -> str:
    return f"{column}_dict"


def parse_action_columns(df: pd.DataFrame, columns: Iterable[str] = ACTION_DICT_COLUMNS) -> Tuple[pd.DataFrame, ParseSummary]:
    out = df.copy()
    malformed_counts = {column: 0 for column in columns}
    malformed_row_flags = pd.Series(False, index=out.index)

    for column in columns:
        parsed_col = _parsed_col_name(column)
        parsed_values = []
        malformed_flags = []
        for raw in out[column].tolist():
            parsed, malformed = parse_action_dict(raw)
            parsed_values.append(parsed)
            malformed_flags.append(malformed)
        malformed_series = pd.Series(malformed_flags, index=out.index)
        malformed_counts[column] = int(malformed_series.sum())
        malformed_row_flags = malformed_row_flags | malformed_series
        out[parsed_col] = parsed_values

    summary = ParseSummary(
        malformed_counts=malformed_counts,
        malformed_rows=int(malformed_row_flags.sum()),
        total_rows=int(len(out)),
    )
    return out, summary


def coerce_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gameId"] = out["gameId"].astype(str)
    out["playerId"] = out["playerId"].astype(str)
    out["roundIndex"] = pd.to_numeric(out["roundIndex"], errors="coerce").astype("Int64")
    out["predicted_contribution"] = pd.to_numeric(out["predicted_contribution"], errors="coerce")
    out["actual_contribution"] = pd.to_numeric(out["actual_contribution"], errors="coerce")
    out["actual_available_bool"] = out["actual_available"].map(lambda x: bool(parse_bool(x, default=False)))
    out["predicted_contribution_parsed_bool"] = out["predicted_contribution_parsed"].map(
        lambda x: bool(parse_bool(x, default=False))
    )
    return out


def apply_filters(
    df: pd.DataFrame,
    min_round: Optional[int],
    max_round: Optional[int],
    skip_no_actual: bool,
) -> Tuple[pd.DataFrame, FilterSummary]:
    out = df.copy()
    pre_rows = int(len(out))

    if min_round is not None:
        out = out[out["roundIndex"].notna() & (out["roundIndex"] >= int(min_round))]
    if max_round is not None:
        out = out[out["roundIndex"].notna() & (out["roundIndex"] <= int(max_round))]
    if skip_no_actual:
        out = out[out["actual_available_bool"]]

    out = out.reset_index(drop=True)
    post_rows = int(len(out))
    summary = FilterSummary(
        pre_filter_rows=pre_rows,
        post_filter_rows=post_rows,
        dropped_rows=pre_rows - post_rows,
        min_round=min_round,
        max_round=max_round,
        skip_no_actual=skip_no_actual,
    )
    return out, summary
