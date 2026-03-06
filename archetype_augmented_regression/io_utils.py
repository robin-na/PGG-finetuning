from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "experiment": str(obj.get("experiment", "")),
                    "participant": str(obj.get("participant", "")),
                    "text": str(obj.get("text", "")),
                }
            )
    return pd.DataFrame(rows)


def resolve_target(requested: str, learn_df: pd.DataFrame, val_df: pd.DataFrame) -> str:
    if requested in learn_df.columns and requested in val_df.columns:
        return requested
    if requested == "itt_normalized_efficiency":
        fallback = "itt_efficiency"
        if fallback in learn_df.columns and fallback in val_df.columns:
            print(
                "[warn] Requested target 'itt_normalized_efficiency' not found; "
                "using 'itt_efficiency' instead."
            )
            return fallback
    raise ValueError(
        f"Target '{requested}' not found in both analysis tables. "
        f"learn_has={requested in learn_df.columns}, val_has={requested in val_df.columns}"
    )


def coerce_config_col(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    if series.dtype.kind in "biufc":
        return series
    lowered = series.astype(str).str.strip().str.lower()
    if lowered.isin(["true", "false"]).all():
        return (lowered == "true").astype(float)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        return numeric
    return series.astype(str)


def aggregate_xy(
    x_df: pd.DataFrame,
    y_arr: np.ndarray,
    group_ids: pd.Series,
) -> tuple[pd.DataFrame, np.ndarray]:
    tmp = x_df.copy()
    tmp["__target__"] = y_arr
    tmp["__group__"] = group_ids.astype(str).to_numpy()

    numeric_cols = [col for col in x_df.columns if pd.api.types.is_numeric_dtype(x_df[col])]
    categorical_cols = [col for col in x_df.columns if col not in numeric_cols]

    agg_map: dict[str, str] = {col: "mean" for col in numeric_cols}
    agg_map.update({col: "first" for col in categorical_cols})
    agg_map["__target__"] = "mean"

    grouped = tmp.groupby("__group__", as_index=False).agg(agg_map)
    x_grouped = grouped[x_df.columns].copy()
    y_grouped = grouped["__target__"].to_numpy(dtype=float)
    return x_grouped, y_grouped


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value

