from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


@dataclass
class TreatmentConfig:
    treatment_name: str
    config: Dict[str, Any]
    n_players: int


def _clean_bool(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        if val == 1:
            return True
        if val == 0:
            return False
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
    return None


def _maybe_int(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, float):
        if pd.isna(val):
            return None
        if val.is_integer():
            return int(val)
        return val
    if isinstance(val, int):
        return int(val)
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() == "nan":
            return None
        try:
            f = float(s)
        except ValueError:
            return val
        if f.is_integer():
            return int(f)
        return f
    return val


def row_to_config(row: pd.Series) -> Dict[str, Any]:
    return {
        "playerCount": _maybe_int(row.get("CONFIG_playerCount")),
        "numRounds": _maybe_int(row.get("CONFIG_numRounds")),
        "showNRounds": _clean_bool(row.get("CONFIG_showNRounds")),
        "endowment": _maybe_int(row.get("CONFIG_endowment")),
        "multiplier": row.get("CONFIG_multiplier"),
        "MPCR": row.get("CONFIG_MPCR"),
        "allOrNothing": _clean_bool(row.get("CONFIG_allOrNothing")),
        "chat": _clean_bool(row.get("CONFIG_chat")),
        "punishmentExists": _clean_bool(row.get("CONFIG_punishmentExists")),
        "punishmentCost": row.get("CONFIG_punishmentCost"),
        "punishmentMagnitude": row.get("CONFIG_punishmentMagnitude"),
        "punishmentTech": row.get("CONFIG_punishmentTech"),
        "rewardExists": _clean_bool(row.get("CONFIG_rewardExists")),
        "rewardCost": row.get("CONFIG_rewardCost"),
        "rewardMagnitude": row.get("CONFIG_rewardMagnitude"),
        "rewardTech": row.get("CONFIG_rewardTech"),
        "showOtherSummaries": _clean_bool(row.get("CONFIG_showOtherSummaries")),
        "showPunishmentId": _clean_bool(row.get("CONFIG_showPunishmentId")),
        "showRewardId": _clean_bool(row.get("CONFIG_showRewardId")),
    }


def load_treatment_configs(
    csv_path: Path,
    treatment_col: str = "CONFIG_treatmentName",
) -> Tuple[List[TreatmentConfig], Dict[str, int]]:
    df = pd.read_csv(csv_path)
    if treatment_col not in df.columns:
        raise ValueError(f"Missing '{treatment_col}' column in {csv_path}")

    total_rows = len(df)
    df = df.drop_duplicates(subset=treatment_col, keep="first")
    dedup_rows = len(df)

    records: List[TreatmentConfig] = []
    for _, row in df.iterrows():
        treatment = row.get(treatment_col)
        if pd.isna(treatment):
            continue
        cfg = row_to_config(row)
        n_players = _maybe_int(cfg.get("playerCount"))
        if n_players is None:
            n_players = _maybe_int(row.get("CONFIG_playerCount"))
        if n_players is None:
            continue
        records.append(
            TreatmentConfig(
                treatment_name=str(treatment),
                config=cfg,
                n_players=int(n_players),
            )
        )

    stats = {
        "total_rows": total_rows,
        "dedup_rows": dedup_rows,
        "dropped_rows": total_rows - dedup_rows,
        "kept": len(records),
    }
    return records, stats
