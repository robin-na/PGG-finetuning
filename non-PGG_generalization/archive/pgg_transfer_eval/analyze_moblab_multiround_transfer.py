#!/usr/bin/env python3
"""Analyze MobLab multiround behavior and design a principled PGG retrieval score."""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Persona.misc.transfer_profile_data import build_raw_profiles


MOBLAB_DIR = PROJECT_ROOT / "non-PGG_generalization" / "data" / "MobLab"
PGG_DIR = PROJECT_ROOT / "non-PGG_generalization" / "data" / "PGG"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "pgg_transfer_eval"
    / "output"
    / "moblab_multiround_analysis"
)

SOURCE_FAMILY_TO_BASELINE = {
    "dictator": ["dictator"],
    "trust": ["trust_1", "trust_3"],
    "ultimatum": ["ultimatum_1", "ultimatum_2"],
    "pg": ["PG"],
}
TARGET_SPECS = {
    "dictator": {"family": "dictator", "label": "Dictator"},
    "trust_1": {"family": "trust", "label": "Trust Investor"},
    "trust_3": {"family": "trust", "label": "Trust Banker"},
    "ultimatum_1": {"family": "ultimatum", "label": "Ultimatum Proposer"},
    "ultimatum_2": {"family": "ultimatum", "label": "Ultimatum Responder"},
}
ROLE_LABELS = {
    "dictator": "Dictator",
    "trust_1": "Trust Investor",
    "trust_3": "Trust Banker",
    "ultimatum_1": "Ultimatum Proposer",
    "ultimatum_2": "Ultimatum Responder",
}
TARGET_ROLE_WEIGHTS = {
    "dictator": {
        "social": {"prosociality": 0.50, "fairness": 0.30, "caution": 0.05, "reciprocity": 0.00, "stability": 0.15},
        "component_weights": {"role_social": 0.42, "rule_stake": 0.13, "pattern": 0.15, "fairness_reciprocity": 0.30},
    },
    "trust_1": {
        "social": {"prosociality": 0.35, "fairness": 0.10, "caution": 0.30, "reciprocity": 0.10, "stability": 0.15},
        "component_weights": {"role_social": 0.34, "rule_stake": 0.16, "pattern": 0.20, "fairness_reciprocity": 0.30},
    },
    "trust_3": {
        "social": {"prosociality": 0.10, "fairness": 0.22, "caution": 0.08, "reciprocity": 0.42, "stability": 0.18},
        "component_weights": {"role_social": 0.34, "rule_stake": 0.10, "pattern": 0.18, "fairness_reciprocity": 0.38},
    },
    "ultimatum_1": {
        "social": {"prosociality": 0.26, "fairness": 0.38, "caution": 0.12, "reciprocity": 0.06, "stability": 0.18},
        "component_weights": {"role_social": 0.38, "rule_stake": 0.10, "pattern": 0.18, "fairness_reciprocity": 0.34},
    },
    "ultimatum_2": {
        "social": {"prosociality": 0.05, "fairness": 0.42, "caution": 0.26, "reciprocity": 0.10, "stability": 0.17},
        "component_weights": {"role_social": 0.35, "rule_stake": 0.08, "pattern": 0.17, "fairness_reciprocity": 0.40},
    },
}

MIN_SLOPE_DENOM = 1e-4
MAX_BEHAVIOR_SLOPE = 3.0


@dataclass
class EvalResult:
    source_family: str
    target: str
    n_users: int
    baseline_r2: float
    baseline_mae: float
    baseline_spearman: float
    rich_r2: float
    rich_mae: float
    rich_spearman: float
    delta_r2: float
    rich_preds: np.ndarray
    baseline_preds: np.ndarray
    target_values: np.ndarray
    user_ids: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def configure_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "#fbf7ef",
            "axes.facecolor": "#fbf7ef",
            "savefig.facecolor": "#fbf7ef",
            "axes.edgecolor": "#1d2a3a",
            "axes.labelcolor": "#1d2a3a",
            "text.color": "#1d2a3a",
            "xtick.color": "#1d2a3a",
            "ytick.color": "#1d2a3a",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "figure.titlesize": 18,
            "figure.titleweight": "bold",
            "grid.color": "#d8cfc2",
            "grid.alpha": 0.35,
            "axes.grid": True,
            "grid.linestyle": "--",
        }
    )


def robust_literal(value: object) -> Optional[object]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text == "None":
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def safe_float(value: object) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(value)
    except Exception:
        return None


def clipped(value: Optional[float], lo: float = 0.0, hi: float = 1.0) -> Optional[float]:
    if value is None or math.isnan(value):
        return None
    return float(min(max(value, lo), hi))


def safe_ratio(
    numerator: object,
    denominator: object,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> Optional[float]:
    numer = safe_float(numerator)
    denom = safe_float(denominator)
    if numer is None or denom is None or not math.isfinite(numer) or not math.isfinite(denom) or denom <= 0:
        return None
    value = numer / denom
    if lo is not None:
        value = max(value, lo)
    if hi is not None:
        value = min(value, hi)
    return float(value)


def total_components(value: object) -> Tuple[Optional[float], Optional[float]]:
    parsed = robust_literal(value)
    if isinstance(parsed, (list, tuple)):
        first = safe_float(parsed[0]) if len(parsed) >= 1 else None
        second = safe_float(parsed[1]) if len(parsed) >= 2 else None
        return first, second
    scalar = safe_float(parsed if parsed is not None else value)
    return scalar, None


def nanmean(values: Sequence[Optional[float]]) -> Optional[float]:
    arr = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def nanstd(values: Sequence[Optional[float]]) -> Optional[float]:
    arr = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return None
    return float(np.std(arr))


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    x_s = pd.Series(np.asarray(x, dtype=float)).rank(method="average")
    y_s = pd.Series(np.asarray(y, dtype=float)).rank(method="average")
    if x_s.nunique() <= 1 or y_s.nunique() <= 1:
        return float("nan")
    return float(x_s.corr(y_s))


def linear_slope(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    x = np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 0 or np.allclose(y_centered, 0):
        return 0.0
    return float(np.dot(x_centered, y_centered) / denom)


def rate_large_jumps(values: Sequence[float], threshold: float) -> Optional[float]:
    if len(values) < 2:
        return None
    diffs = np.abs(np.diff(np.asarray(values, dtype=float)))
    return float(np.mean(diffs >= threshold))


def response_slope(prev_values: Sequence[float], next_values: Sequence[float]) -> Optional[float]:
    if len(prev_values) < 2 or len(next_values) < 2:
        return None
    x = np.asarray(prev_values, dtype=float)
    y = np.asarray(next_values, dtype=float)
    if len(x) != len(y):
        return None
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = float(np.dot(x_centered, x_centered))
    if denom <= MIN_SLOPE_DENOM:
        return None
    slope = float(np.dot(x_centered, y_centered) / denom)
    if not math.isfinite(slope):
        return None
    return float(np.clip(slope, -MAX_BEHAVIOR_SLOPE, MAX_BEHAVIOR_SLOPE))


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X: object, y: Optional[object] = None) -> "QuantileClipper":
        frame = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan).astype(float)
        self.lower_bounds_ = frame.quantile(self.lower, numeric_only=True)
        self.upper_bounds_ = frame.quantile(self.upper, numeric_only=True)
        return self

    def transform(self, X: object) -> np.ndarray:
        frame = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan).astype(float)
        frame = frame.clip(self.lower_bounds_, self.upper_bounds_, axis=1)
        return frame.to_numpy(dtype=float)


def sessionize(df: pd.DataFrame, role_col: str = "Role") -> pd.DataFrame:
    df = df.reset_index().rename(columns={"index": "_row_order"}).copy()
    df["session_id"] = None
    for (user_id, role), group in df.groupby(["UserID", role_col], sort=False):
        group = group.sort_values("_row_order")
        session_idx = 0
        prev_round = None
        prev_total = None
        for row in group.itertuples():
            current_round = int(row.Round)
            current_total = int(row.totalRound)
            if prev_round is None or current_round <= prev_round or current_total != prev_total:
                session_idx += 1
            df.loc[row.Index, "session_id"] = f"{user_id}::{role}::{session_idx}"
            prev_round = current_round
            prev_total = current_total
    return df


def trajectory_feature_dict(values: Sequence[float], scale: float, prefix: str) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr_scaled = arr / scale if scale else arr
    diffs = np.diff(arr_scaled) if arr_scaled.size >= 2 else np.asarray([])
    k = max(1, arr_scaled.size // 3)
    return {
        f"{prefix}_rounds": float(arr_scaled.size),
        f"{prefix}_mean": float(np.mean(arr_scaled)),
        f"{prefix}_std": float(np.std(arr_scaled)),
        f"{prefix}_first": float(arr_scaled[0]),
        f"{prefix}_last": float(arr_scaled[-1]),
        f"{prefix}_min": float(np.min(arr_scaled)),
        f"{prefix}_max": float(np.max(arr_scaled)),
        f"{prefix}_range": float(np.max(arr_scaled) - np.min(arr_scaled)),
        f"{prefix}_trend": linear_slope(arr_scaled.tolist()) or 0.0,
        f"{prefix}_switch_rate": float(np.mean(diffs != 0)) if diffs.size else 0.0,
        f"{prefix}_mean_abs_jump": float(np.mean(np.abs(diffs))) if diffs.size else 0.0,
        f"{prefix}_large_jump_rate": rate_large_jumps(arr.tolist(), threshold=scale * 0.25) or 0.0,
        f"{prefix}_end_shift": float(np.mean(arr_scaled[-k:]) - np.mean(arr_scaled[:k])),
    }


def aggregate_feature_dicts(rows: Sequence[Dict[str, Optional[float]]], prefix: str) -> Dict[str, float]:
    if not rows:
        return {}
    all_keys = sorted({key for row in rows for key in row.keys()})
    out: Dict[str, float] = {
        f"{prefix}_session_count": float(len(rows)),
    }
    for key in all_keys:
        vals = [row.get(key) for row in rows]
        mean_value = nanmean(vals)
        if mean_value is None:
            continue
        out[f"{key}_agg_mean"] = mean_value
        std_value = nanstd(vals)
        if std_value is not None:
            out[f"{key}_agg_std"] = std_value
    return out


def load_joint_baseline() -> pd.DataFrame:
    df = pd.read_csv(MOBLAB_DIR / "joint.csv")
    trust_df = pd.read_csv(MOBLAB_DIR / "trust_investment.csv")
    trust_df = trust_df[(trust_df["gameType"] == "trust_investment") & (trust_df["Round"] == 1)].copy()
    trust_df["move_num"] = pd.to_numeric(trust_df["move"], errors="coerce")
    trust_df["round_tuple"] = trust_df["roundResult"].apply(robust_literal)
    trust_df["total_tuple"] = trust_df["Total"].apply(total_components)
    trust_df["trust_scale_100"] = trust_df["total_tuple"].apply(lambda x: safe_float(x[0]) == 100.0 if isinstance(x, tuple) else False)
    trust_df = trust_df[trust_df["trust_scale_100"]].copy()

    trust1_rows = trust_df[
        (trust_df["Role"] == "first") & trust_df["move_num"].between(0, 100, inclusive="both")
    ][["UserID", "move_num"]].drop_duplicates(subset=["UserID"], keep="first")

    trust3_records: List[Dict[str, float]] = []
    for row in trust_df[trust_df["Role"] == "second"].itertuples():
        pair = row.round_tuple if isinstance(row.round_tuple, (list, tuple)) else None
        if not pair or len(pair) < 2:
            continue
        invest = safe_float(pair[0])
        ret = safe_float(pair[1])
        if invest != 50 or ret is None or invest is None or ret < 0 or ret > invest * 3:
            continue
        trust3_records.append({"UserID": int(row.UserID), "trust_3": float(ret)})
    trust3_rows = pd.DataFrame(trust3_records).drop_duplicates(subset=["UserID"], keep="first")

    trust1_map = trust1_rows.rename(columns={"move_num": "trust_1"}).set_index("UserID")
    trust3_map = trust3_rows.set_index("UserID") if not trust3_rows.empty else pd.DataFrame(columns=["trust_3"]).set_index(pd.Index([], name="UserID"))
    df = df.set_index("UserID")
    df["trust_1"] = trust1_map["trust_1"]
    df["trust_3"] = trust3_map["trust_3"]
    df = df.reset_index()
    return df


def build_dictator_features() -> pd.DataFrame:
    df = pd.read_csv(MOBLAB_DIR / "dictator.csv")
    df = df[(df["gameType"] == "dictator") & (df["Role"] == "first") & (df["Total"] == 100)].copy()
    df["move"] = pd.to_numeric(df["move"], errors="coerce")
    df = df[df["move"].between(0, 100, inclusive="both")]
    df = sessionize(df)

    rows: List[Dict[str, float]] = []
    for user_id, user_df in df.groupby("UserID"):
        session_rows: List[Dict[str, float]] = []
        fairness_gaps: List[float] = []
        equal_rate: List[float] = []
        selfish_rate: List[float] = []
        for _, session_df in user_df.groupby("session_id", sort=False):
            offers = session_df.sort_values(["Round", "_row_order"])["move"].astype(float).tolist()
            if not offers:
                continue
            session_features = trajectory_feature_dict(offers, scale=100.0, prefix="dict_offer")
            session_rows.append(session_features)
            shares = np.asarray(offers, dtype=float) / 100.0
            fairness_gaps.append(float(np.mean(np.abs(shares - 0.5))))
            equal_rate.append(float(np.mean(np.abs(shares - 0.5) <= 0.05)))
            selfish_rate.append(float(np.mean(shares <= 0.1)))
        if not session_rows:
            continue
        record = {"UserID": int(user_id)}
        record.update(aggregate_feature_dicts(session_rows, prefix="dictator"))
        record["dictator_fairness_gap_mean"] = nanmean(fairness_gaps) or 0.0
        record["dictator_equal_split_rate"] = nanmean(equal_rate) or 0.0
        record["dictator_selfish_rate"] = nanmean(selfish_rate) or 0.0
        rows.append(record)
    return pd.DataFrame(rows)


def build_trust_features() -> pd.DataFrame:
    df = pd.read_csv(MOBLAB_DIR / "trust_investment.csv")
    df = df[df["gameType"] == "trust_investment"].copy()
    df["move_num"] = pd.to_numeric(df["move"], errors="coerce")
    df["round_tuple"] = df["roundResult"].apply(robust_literal)
    df["total_tuple"] = df["Total"].apply(total_components)
    df["trust_scale_100"] = df["total_tuple"].apply(lambda x: safe_float(x[0]) == 100.0 if isinstance(x, tuple) else False)
    df = df[df["trust_scale_100"]].copy()
    df = sessionize(df)

    rows: List[Dict[str, float]] = []
    for user_id, user_df in df.groupby("UserID"):
        investor_sessions: List[Dict[str, float]] = []
        banker_sessions: List[Dict[str, float]] = []
        investor_return_resp: List[Optional[float]] = []
        banker_reciprocity: List[Optional[float]] = []
        banker_fair_gap: List[float] = []
        for role, role_df in user_df.groupby("Role", sort=False):
            for _, session_df in role_df.groupby("session_id", sort=False):
                session_df = session_df.sort_values(["Round", "_row_order"])
                if role == "first":
                    session_valid = session_df[session_df["move_num"].between(0, 100, inclusive="both")].copy()
                    moves = session_valid["move_num"].astype(float).tolist()
                    return_rates: List[Optional[float]] = []
                    for row in session_valid.itertuples():
                        pair = row.round_tuple if isinstance(row.round_tuple, (list, tuple)) else None
                        if not pair or len(pair) < 2:
                            return_rates.append(None)
                            continue
                        invest = safe_float(pair[0])
                        ret = safe_float(pair[1])
                        return_rates.append(safe_ratio(ret, 3.0 * invest if invest is not None else None, lo=0.0, hi=2.0))
                    if not moves:
                        continue
                    session_features = trajectory_feature_dict(moves, scale=100.0, prefix="trust_invest")
                    valid_returns = [v for v in return_rates if v is not None]
                    if valid_returns:
                        session_features["trust_invest_received_return_mean"] = float(np.mean(valid_returns))
                    if len(return_rates) >= 2:
                        prev_ret: List[float] = []
                        next_inv_share: List[float] = []
                        for idx in range(len(return_rates) - 1):
                            if return_rates[idx] is None:
                                continue
                            prev_ret.append(float(return_rates[idx]))
                            next_inv_share.append(float(moves[idx + 1] / 100.0))
                        investor_return_resp.append(response_slope(prev_ret, next_inv_share))
                    investor_sessions.append(session_features)
                elif role == "second":
                    investments: List[float] = []
                    returns: List[float] = []
                    return_rates: List[float] = []
                    for row in session_df.itertuples():
                        pair = row.round_tuple if isinstance(row.round_tuple, (list, tuple)) else None
                        if not pair or len(pair) < 2:
                            continue
                        invest = safe_float(pair[0])
                        ret = safe_float(pair[1])
                        if invest is None or ret is None or invest < 0 or invest > 100 or ret < 0 or ret > invest * 3:
                            continue
                        return_rate = safe_ratio(ret, 3.0 * invest if invest is not None else None, lo=0.0, hi=2.0)
                        if return_rate is None:
                            continue
                        investments.append(invest)
                        returns.append(ret)
                        return_rates.append(return_rate)
                    if not returns:
                        continue
                    session_features = trajectory_feature_dict(returns, scale=150.0, prefix="trust_return")
                    rr = np.asarray(return_rates, dtype=float)
                    session_features["trust_return_rate_mean"] = float(np.mean(rr))
                    session_features["trust_return_rate_std"] = float(np.std(rr))
                    banker_fair_gap.append(float(np.mean(np.abs(rr - 0.5))))
                    if len(return_rates) >= 2:
                        banker_reciprocity.append(response_slope((np.asarray(investments, dtype=float) / 100.0).tolist(), return_rates))
                    banker_sessions.append(session_features)
        if not investor_sessions and not banker_sessions:
            continue
        record = {"UserID": int(user_id)}
        if investor_sessions:
            record.update(aggregate_feature_dicts(investor_sessions, prefix="trust_investor"))
            record["trust_investor_prev_return_response"] = nanmean(investor_return_resp) or 0.0
        if banker_sessions:
            record.update(aggregate_feature_dicts(banker_sessions, prefix="trust_banker"))
            record["trust_banker_reciprocity_slope"] = nanmean(banker_reciprocity) or 0.0
            record["trust_banker_fair_gap_mean"] = nanmean(banker_fair_gap) or 0.0
        rows.append(record)
    return pd.DataFrame(rows)


def build_ultimatum_features() -> pd.DataFrame:
    df = pd.read_csv(MOBLAB_DIR / "ultimatum_strategy.csv")
    df = df[(df["gameType"] == "ultimatum_strategy") & (df["Role"] == "player") & (df["Total"] == 100)].copy()
    df["move_tuple"] = df["move"].apply(robust_literal)
    df = df[df["move_tuple"].notna()].copy()
    df = sessionize(df)

    rows: List[Dict[str, float]] = []
    for user_id, user_df in df.groupby("UserID"):
        propose_sessions: List[Dict[str, float]] = []
        accept_sessions: List[Dict[str, float]] = []
        fairness_gap: List[float] = []
        consistency: List[float] = []
        for _, session_df in user_df.groupby("session_id", sort=False):
            session_df = session_df.sort_values(["Round", "_row_order"])
            propose: List[float] = []
            accept: List[float] = []
            for row in session_df.itertuples():
                pair = row.move_tuple if isinstance(row.move_tuple, (list, tuple)) else None
                if not pair or len(pair) < 2:
                    continue
                p = safe_float(pair[0])
                a = safe_float(pair[1])
                if p is None or a is None or not (0 <= p <= 100) or not (0 <= a <= 100):
                    continue
                propose.append(p)
                accept.append(a)
            if not propose:
                continue
            propose_sessions.append(trajectory_feature_dict(propose, scale=100.0, prefix="ult_propose"))
            accept_sessions.append(trajectory_feature_dict(accept, scale=100.0, prefix="ult_accept"))
            shares = np.asarray(propose, dtype=float) / 100.0
            fairness_gap.append(float(np.mean(np.abs(shares - 0.5))))
            consistency.append(float(np.mean(1.0 - np.abs((np.asarray(propose) - np.asarray(accept)) / 100.0))))
        if not propose_sessions:
            continue
        record = {"UserID": int(user_id)}
        record.update(aggregate_feature_dicts(propose_sessions, prefix="ultimatum_proposer"))
        record.update(aggregate_feature_dicts(accept_sessions, prefix="ultimatum_responder"))
        record["ultimatum_fairness_gap_mean"] = nanmean(fairness_gap) or 0.0
        record["ultimatum_internal_consistency"] = nanmean(consistency) or 0.0
        rows.append(record)
    return pd.DataFrame(rows)


def build_public_goods_features() -> pd.DataFrame:
    df = pd.read_csv(MOBLAB_DIR / "public_goods_linear_water.csv")
    df = df[(df["gameType"] == "public_goods_linear_water") & (df["Role"] == "contributor") & (df["Total"] == 20)].copy()
    df["move_num"] = pd.to_numeric(df["move"], errors="coerce")
    df["round_tuple"] = df["roundResult"].apply(robust_literal)
    df = df[df["move_num"].between(0, 20, inclusive="both")]
    df = sessionize(df)

    rows: List[Dict[str, float]] = []
    for user_id, user_df in df.groupby("UserID"):
        session_rows: List[Dict[str, float]] = []
        cond_slopes: List[Optional[float]] = []
        defection_responses: List[Optional[float]] = []
        for _, session_df in user_df.groupby("session_id", sort=False):
            session_df = session_df.sort_values(["Round", "_row_order"])
            contribs = session_df["move_num"].astype(float).tolist()
            session_features = trajectory_feature_dict(contribs, scale=20.0, prefix="pg_contrib")

            others_mean: List[float] = []
            next_contrib: List[float] = []
            deltas_after_low: List[float] = []
            for idx, row in enumerate(session_df.itertuples()):
                group_vals = row.round_tuple if isinstance(row.round_tuple, (list, tuple)) else None
                if group_vals and len(group_vals) >= 2:
                    values = [safe_float(v) for v in group_vals]
                    values = [v for v in values if v is not None]
                    if values:
                        mean_group = float(np.mean(values))
                        others_mean.append(mean_group / 20.0)
                        next_contrib.append(row.move_num / 20.0)
                        if idx + 1 < len(session_df):
                            if np.min(values) / 20.0 <= 0.25:
                                next_value = float(session_df.iloc[idx + 1]["move_num"])
                                deltas_after_low.append((next_value - float(row.move_num)) / 20.0)
            if len(others_mean) >= 2:
                cond_slopes.append(response_slope(others_mean, next_contrib))
            if deltas_after_low:
                defection_responses.append(float(np.mean(deltas_after_low)))
            session_rows.append(session_features)
        if not session_rows:
            continue
        record = {"UserID": int(user_id)}
        record.update(aggregate_feature_dicts(session_rows, prefix="pg"))
        record["pg_conditionality_slope"] = nanmean(cond_slopes) or 0.0
        record["pg_defection_response"] = nanmean(defection_responses) or 0.0
        rows.append(record)
    return pd.DataFrame(rows)


def family_tables() -> Dict[str, pd.DataFrame]:
    return {
        "dictator": build_dictator_features(),
        "trust": build_trust_features(),
        "ultimatum": build_ultimatum_features(),
        "pg": build_public_goods_features(),
    }


def evaluate_model(
    user_ids: np.ndarray,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> Tuple[np.ndarray, float, float, float]:
    X = X.replace([np.inf, -np.inf], np.nan).astype(float)
    usable_cols = [
        col
        for col in X.columns
        if X[col].notna().sum() >= max(5, int(len(X) * 0.02)) and X[col].dropna().nunique() > 1
    ]
    if not usable_cols:
        usable_cols = [col for col in X.columns if X[col].notna().sum() >= 2 and X[col].dropna().nunique() > 1]
    if not usable_cols:
        raise ValueError("No usable feature columns after sanitization.")
    X = X[usable_cols]
    model = Pipeline(
        [
            ("clip", QuantileClipper(lower=0.01, upper=0.99)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=3.0, solver="svd")),
        ]
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    preds = cross_val_predict(model, X, y, cv=cv, method="predict")
    r2 = float(r2_score(y, preds))
    mae = float(mean_absolute_error(y, preds))
    spear = spearman_corr(y, preds)
    return preds, r2, mae, spear


def compare_scalar_vs_rich(
    joint_df: pd.DataFrame,
    family_df_map: Dict[str, pd.DataFrame],
    seed: int,
) -> List[EvalResult]:
    results: List[EvalResult] = []
    joint_indexed = joint_df.set_index("UserID")
    for source_family, rich_df in family_df_map.items():
        rich_indexed = rich_df.set_index("UserID")
        baseline_cols = SOURCE_FAMILY_TO_BASELINE[source_family]
        baseline_df = joint_indexed[baseline_cols].copy()
        for target, target_meta in TARGET_SPECS.items():
            if target_meta["family"] == source_family:
                continue
            merged = pd.DataFrame(index=joint_indexed.index)
            merged["target"] = joint_indexed[target]
            for col in baseline_cols:
                merged[f"baseline__{col}"] = baseline_df[col]
            rich_cols = list(rich_indexed.columns)
            rich_subset = rich_indexed[rich_cols].copy()
            for col in rich_cols:
                merged[f"rich__{col}"] = rich_subset[col]

            baseline_mask = merged[[f"baseline__{col}" for col in baseline_cols]].notna().any(axis=1)
            rich_mask = merged[[f"rich__{col}" for col in rich_cols]].notna().any(axis=1)
            usable = merged[merged["target"].notna() & baseline_mask & rich_mask].copy()
            if len(usable) < 150:
                continue

            baseline_X = usable[[col for col in usable.columns if col.startswith("baseline__")]]
            rich_X = usable[[col for col in usable.columns if col.startswith("rich__")]]
            y = usable["target"].astype(float)
            user_ids = usable.index.to_numpy()

            baseline_preds, baseline_r2, baseline_mae, baseline_spearman = evaluate_model(
                user_ids=user_ids,
                X=baseline_X,
                y=y,
                seed=seed,
            )
            rich_preds, rich_r2, rich_mae, rich_spearman = evaluate_model(
                user_ids=user_ids,
                X=rich_X,
                y=y,
                seed=seed,
            )
            results.append(
                EvalResult(
                    source_family=source_family,
                    target=target,
                    n_users=len(usable),
                    baseline_r2=baseline_r2,
                    baseline_mae=baseline_mae,
                    baseline_spearman=baseline_spearman,
                    rich_r2=rich_r2,
                    rich_mae=rich_mae,
                    rich_spearman=rich_spearman,
                    delta_r2=rich_r2 - baseline_r2,
                    rich_preds=rich_preds,
                    baseline_preds=baseline_preds,
                    target_values=y.to_numpy(),
                    user_ids=user_ids,
                )
            )
    return results


def pgg_library_feature_table() -> pd.DataFrame:
    raw_profiles = build_raw_profiles(["learn", "val"])
    demo_df = pd.read_csv(PGG_DIR / "demographics_numeric_learn_val_consolidated.csv")
    demo_df["split"] = demo_df["wave"].map({"learning_wave": "learn", "validation_wave": "val"})
    demo_df = demo_df.rename(columns={"gameId": "gameId", "playerId": "playerId"})
    demo_map = {
        (str(row["split"]), str(row["gameId"]), str(row["playerId"])): row
        for _, row in demo_df.iterrows()
    }

    oracle_path = PGG_DIR / "archetype_oracle_gpt51_learn_val_union_finished.jsonl"
    oracle_map: Dict[Tuple[str, str, str], str] = {}
    with oracle_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split = "learn" if row.get("_wave") == "learning_wave" else "val"
            key = (split, str(row["experiment"]), str(row["participant"]))
            oracle_map[key] = str(row.get("text", ""))

    rows: List[Dict[str, object]] = []
    for row in raw_profiles:
        if not row.get("played_to_end"):
            continue
        split = str(row["split"])
        game_id = str(row["gameId"])
        player_id = str(row["playerId"])
        key = (split, game_id, player_id)

        config = row.get("config", {})
        summary = row.get("observed_summary", {})
        module_card = row.get("module_card", {})
        events = row.get("event_responses", [])
        contrib = summary.get("contribution", {})
        punishment_given = summary.get("punishment_given") or {}
        reward_given = summary.get("reward_given") or {}

        conditionality = ((module_card.get("conditionality") or {}).get("responds_to_others_mean") or {})
        cond_evidence = conditionality.get("evidence") or {}
        cond_corr = safe_float(cond_evidence.get("corr"))
        cond_slope = safe_float(cond_evidence.get("b_slope"))

        defection_deltas: List[float] = []
        punish_deltas: List[float] = []
        reward_deltas: List[float] = []
        for event in events:
            event_type = event.get("event_type")
            next_info = event.get("response_next") or {}
            delta = safe_float(next_info.get("delta_contribution_next"))
            if delta is None:
                continue
            endowment = safe_float(config.get("CONFIG_endowment")) or 20.0
            delta_norm = delta / endowment
            if event_type == "saw_defection":
                defection_deltas.append(delta_norm)
            elif event_type == "was_punished":
                punish_deltas.append(delta_norm)
            elif event_type == "was_rewarded":
                reward_deltas.append(delta_norm)

        demo = demo_map.get(key)
        oracle_text = oracle_map.get(key, "")

        contrib_mean = safe_float(contrib.get("mean"))
        contrib_std = safe_float(contrib.get("std"))
        switch_rate = safe_float(contrib.get("switch_rate"))
        endowment = safe_float(config.get("CONFIG_endowment")) or 20.0
        contrib_frac = (contrib_mean / endowment) if contrib_mean is not None else None
        contrib_std_frac = (contrib_std / endowment) if contrib_std is not None else None
        punishment_units = safe_float(punishment_given.get("mean_units_per_round"))
        reward_units = safe_float(reward_given.get("mean_units_per_round"))

        reciprocity = clipped(
            0.5
            + 0.35 * (cond_corr or 0.0)
            + 0.20 * ((reward_units or 0.0) / 2.0)
            - 0.10 * max(-(cond_corr or 0.0), 0.0)
        )
        fairness = clipped(
            0.45 * (contrib_frac or 0.0)
            + 0.30 * min((punishment_units or 0.0) / 2.0, 1.0)
            + 0.25 * max((nanmean(defection_deltas) or 0.0), 0.0)
        )
        caution = clipped(
            0.45 * max(-(cond_corr or 0.0), 0.0)
            + 0.35 * min((punishment_units or 0.0) / 2.0, 1.0)
            + 0.20 * max(-(nanmean(defection_deltas) or 0.0), 0.0)
        )
        stability = clipped(
            1.0 - (0.55 * (switch_rate or 0.0) + 0.45 * min((contrib_std_frac or 0.0) * 1.8, 1.0))
        )
        prosociality = clipped(contrib_frac)
        pattern_volatility = clipped(0.60 * (switch_rate or 0.0) + 0.40 * min((contrib_std_frac or 0.0) * 2.0, 1.0))
        pattern_responsiveness = clipped(
            0.50 * min(abs(cond_slope or 0.0) / 0.8, 1.0)
            + 0.25 * min(abs(nanmean(defection_deltas) or 0.0) / 0.5, 1.0)
            + 0.25 * min(abs(nanmean(punish_deltas) or 0.0) / 0.5, 1.0)
        )

        rows.append(
            {
                "split": split,
                "gameId": game_id,
                "playerId": player_id,
                "age": safe_float(demo["age"]) if demo is not None else None,
                "gender_man": safe_float(demo["gender_man"]) if demo is not None else None,
                "gender_woman": safe_float(demo["gender_woman"]) if demo is not None else None,
                "education_bachelor": safe_float(demo["education_bachelor"]) if demo is not None else None,
                "player_count": safe_float(config.get("CONFIG_playerCount")),
                "num_rounds": safe_float(config.get("CONFIG_numRounds")),
                "multiplier": safe_float(config.get("CONFIG_multiplier")),
                "endowment": safe_float(config.get("CONFIG_endowment")),
                "all_or_nothing": float(bool(config.get("CONFIG_allOrNothing") is True)),
                "chat_enabled": float(bool(config.get("CONFIG_chat") is True)),
                "punishment_enabled": float(bool(config.get("CONFIG_punishmentExists") is True)),
                "reward_enabled": float(bool(config.get("CONFIG_rewardExists") is True)),
                "pgg_prosociality": prosociality,
                "pgg_reciprocity": reciprocity,
                "pgg_fairness": fairness,
                "pgg_caution": caution,
                "pgg_stability": stability,
                "pgg_pattern_volatility": pattern_volatility,
                "pgg_pattern_responsiveness": pattern_responsiveness,
                "pgg_contrib_mean_frac": contrib_frac,
                "pgg_contrib_std_frac": contrib_std_frac,
                "pgg_switch_rate": switch_rate,
                "pgg_cond_corr": cond_corr,
                "pgg_cond_slope": cond_slope,
                "pgg_defection_delta": nanmean(defection_deltas),
                "pgg_punished_delta": nanmean(punish_deltas),
                "pgg_rewarded_delta": nanmean(reward_deltas),
                "pgg_punish_units_per_round": punishment_units,
                "pgg_reward_units_per_round": reward_units,
                "oracle_preview": oracle_text.split(".")[0][:220] if oracle_text else "",
            }
        )
    return pd.DataFrame(rows)


def family_to_query_environment(source_family: str, feature_row: pd.Series) -> Dict[str, Optional[float]]:
    if source_family == "dictator":
        rounds = safe_float(feature_row.get("dictator_session_count")) or 1.0
        return {"num_rounds": rounds * 3.0, "player_count": 2.0, "multiplier": 1.0, "all_or_nothing": 0.0, "chat_enabled": 0.0}
    if source_family == "trust":
        rounds = safe_float(feature_row.get("trust_investor_session_count")) or safe_float(feature_row.get("trust_banker_session_count")) or 1.0
        return {"num_rounds": rounds * 3.0, "player_count": 2.0, "multiplier": 3.0, "all_or_nothing": 0.0, "chat_enabled": 0.0}
    if source_family == "ultimatum":
        rounds = safe_float(feature_row.get("ultimatum_proposer_session_count")) or 1.0
        return {"num_rounds": rounds * 1.5, "player_count": 2.0, "multiplier": 1.0, "all_or_nothing": 0.0, "chat_enabled": 0.0}
    rounds = safe_float(feature_row.get("pg_session_count")) or 3.0
    return {"num_rounds": rounds * 5.0, "player_count": 4.0, "multiplier": 1.0, "all_or_nothing": 0.0, "chat_enabled": 0.0}


def family_to_query_axes(source_family: str, feature_row: pd.Series) -> Dict[str, Optional[float]]:
    if source_family == "dictator":
        mean_offer = safe_float(feature_row.get("dict_offer_mean_agg_mean"))
        fairness_gap = safe_float(feature_row.get("dictator_fairness_gap_mean"))
        stability = clipped(1.0 - (safe_float(feature_row.get("dict_offer_switch_rate_agg_mean")) or 0.0))
        return {
            "prosociality": mean_offer,
            "reciprocity": None,
            "fairness": clipped(1.0 - 2.0 * (fairness_gap or 0.5)),
            "caution": clipped(safe_float(feature_row.get("dictator_selfish_rate"))),
            "stability": stability,
            "pattern_volatility": clipped((safe_float(feature_row.get("dict_offer_std_agg_mean")) or 0.0) * 2.0),
            "pattern_responsiveness": clipped(abs(safe_float(feature_row.get("dict_offer_trend_agg_mean")) or 0.0) * 2.0),
        }
    if source_family == "trust":
        invest_mean = safe_float(feature_row.get("trust_invest_mean_agg_mean"))
        banker_return = safe_float(feature_row.get("trust_return_rate_mean_agg_mean"))
        banker_recip = safe_float(feature_row.get("trust_banker_reciprocity_slope"))
        invest_resp = safe_float(feature_row.get("trust_investor_prev_return_response"))
        fair_gap = safe_float(feature_row.get("trust_banker_fair_gap_mean"))
        stability = clipped(
            1.0
            - 0.5 * (safe_float(feature_row.get("trust_invest_switch_rate_agg_mean")) or 0.0)
            - 0.5 * min((safe_float(feature_row.get("trust_return_rate_std_agg_mean")) or 0.0) * 2.0, 1.0)
        )
        return {
            "prosociality": clipped(0.65 * (invest_mean or 0.0) + 0.35 * (banker_return or 0.0)),
            "reciprocity": clipped(0.60 * (banker_return or 0.0) + 0.25 * max(banker_recip or 0.0, 0.0) + 0.15 * max(invest_resp or 0.0, 0.0)),
            "fairness": clipped(0.65 * (1.0 - 2.0 * (fair_gap or 0.5)) + 0.35 * (banker_return or 0.0)),
            "caution": clipped(0.55 * (1.0 - (invest_mean or 0.0)) + 0.45 * max(-(invest_resp or 0.0), 0.0)),
            "stability": stability,
            "pattern_volatility": clipped(
                0.5 * (safe_float(feature_row.get("trust_invest_std_agg_mean")) or 0.0)
                + 0.5 * (safe_float(feature_row.get("trust_return_rate_std_agg_mean")) or 0.0)
            ),
            "pattern_responsiveness": clipped(
                0.55 * min(abs(banker_recip or 0.0), 1.0)
                + 0.45 * min(abs(invest_resp or 0.0), 1.0)
            ),
        }
    if source_family == "ultimatum":
        proposer_mean = safe_float(feature_row.get("ult_propose_mean_agg_mean"))
        accept_mean = safe_float(feature_row.get("ult_accept_mean_agg_mean"))
        fairness_gap = safe_float(feature_row.get("ultimatum_fairness_gap_mean"))
        consistency = safe_float(feature_row.get("ultimatum_internal_consistency"))
        stability = clipped(
            1.0
            - 0.5 * (safe_float(feature_row.get("ult_propose_switch_rate_agg_mean")) or 0.0)
            - 0.5 * (safe_float(feature_row.get("ult_accept_switch_rate_agg_mean")) or 0.0)
        )
        return {
            "prosociality": proposer_mean,
            "reciprocity": clipped(0.30 * (accept_mean or 0.0) + 0.70 * (consistency or 0.0)),
            "fairness": clipped(0.55 * (1.0 - 2.0 * (fairness_gap or 0.5)) + 0.45 * (accept_mean or 0.0)),
            "caution": accept_mean,
            "stability": stability,
            "pattern_volatility": clipped(
                0.5 * (safe_float(feature_row.get("ult_propose_std_agg_mean")) or 0.0)
                + 0.5 * (safe_float(feature_row.get("ult_accept_std_agg_mean")) or 0.0)
            ),
            "pattern_responsiveness": clipped(
                0.60 * abs(safe_float(feature_row.get("ult_propose_trend_agg_mean")) or 0.0)
                + 0.40 * abs(safe_float(feature_row.get("ult_accept_trend_agg_mean")) or 0.0)
            ),
        }
    stability = clipped(1.0 - (safe_float(feature_row.get("pg_contrib_switch_rate_agg_mean")) or 0.0))
    return {
        "prosociality": safe_float(feature_row.get("pg_contrib_mean_agg_mean")),
        "reciprocity": clipped(
            0.55 * max(safe_float(feature_row.get("pg_conditionality_slope")) or 0.0, 0.0)
            + 0.45 * max(safe_float(feature_row.get("pg_defection_response")) or 0.0, 0.0)
        ),
        "fairness": clipped(
            0.65 * (safe_float(feature_row.get("pg_contrib_mean_agg_mean")) or 0.0)
            + 0.35 * max(safe_float(feature_row.get("pg_defection_response")) or 0.0, 0.0)
        ),
        "caution": clipped(max(-(safe_float(feature_row.get("pg_defection_response")) or 0.0), 0.0)),
        "stability": stability,
        "pattern_volatility": clipped(
            0.60 * (safe_float(feature_row.get("pg_contrib_std_agg_mean")) or 0.0)
            + 0.40 * (safe_float(feature_row.get("pg_contrib_switch_rate_agg_mean")) or 0.0)
        ),
        "pattern_responsiveness": clipped(abs(safe_float(feature_row.get("pg_conditionality_slope")) or 0.0)),
    }


def similarity_from_pairs(values: Sequence[Tuple[Optional[float], Optional[float], float]]) -> float:
    total = 0.0
    weight_total = 0.0
    for left, right, weight in values:
        if weight <= 0:
            continue
        if left is None or right is None or math.isnan(left) or math.isnan(right):
            continue
        total += weight * (1.0 - min(abs(left - right), 1.0))
        weight_total += weight
    if weight_total == 0:
        return 0.5
    return float(total / weight_total)


def build_retrieval_candidates(
    source_family: str,
    feature_row: pd.Series,
    target_role: str,
    pgg_df: pd.DataFrame,
) -> pd.DataFrame:
    query_axes = family_to_query_axes(source_family, feature_row)
    query_env = family_to_query_environment(source_family, feature_row)
    weight_spec = TARGET_ROLE_WEIGHTS[target_role]
    role_weights = weight_spec["social"]
    comp_weights = weight_spec["component_weights"]

    candidates = pgg_df.copy()
    role_social_scores: List[float] = []
    rule_scores: List[float] = []
    pattern_scores: List[float] = []
    fairness_scores: List[float] = []
    total_scores: List[float] = []

    for row in candidates.itertuples():
        role_social = similarity_from_pairs(
            [
                (query_axes["prosociality"], row.pgg_prosociality, role_weights["prosociality"]),
                (query_axes["reciprocity"], row.pgg_reciprocity, role_weights["reciprocity"]),
                (query_axes["fairness"], row.pgg_fairness, role_weights["fairness"]),
                (query_axes["caution"], row.pgg_caution, role_weights["caution"]),
                (query_axes["stability"], row.pgg_stability, role_weights["stability"]),
            ]
        )
        rule_stake = similarity_from_pairs(
            [
                (clipped((query_env["num_rounds"] or 0.0) / 20.0), clipped((row.num_rounds or 0.0) / 20.0), 0.35),
                (clipped((query_env["player_count"] or 0.0) / 7.0), clipped((row.player_count or 0.0) / 7.0), 0.20),
                (clipped((query_env["multiplier"] or 0.0) / 4.0), clipped((row.multiplier or 0.0) / 4.0), 0.15),
                (query_env["all_or_nothing"], row.all_or_nothing, 0.10),
                (query_env["chat_enabled"], row.chat_enabled, 0.10),
                (0.0, row.punishment_enabled, 0.05),
                (0.0, row.reward_enabled, 0.05),
            ]
        )
        pattern = similarity_from_pairs(
            [
                (query_axes["pattern_volatility"], row.pgg_pattern_volatility, 0.42),
                (query_axes["pattern_responsiveness"], row.pgg_pattern_responsiveness, 0.38),
                (query_axes["stability"], row.pgg_stability, 0.20),
            ]
        )
        fairness_recip = similarity_from_pairs(
            [
                (query_axes["fairness"], row.pgg_fairness, 0.42),
                (query_axes["reciprocity"], row.pgg_reciprocity, 0.33),
                (query_axes["caution"], row.pgg_caution, 0.15),
                (query_axes["prosociality"], row.pgg_prosociality, 0.10),
            ]
        )
        total = (
            comp_weights["role_social"] * role_social
            + comp_weights["rule_stake"] * rule_stake
            + comp_weights["pattern"] * pattern
            + comp_weights["fairness_reciprocity"] * fairness_recip
        )
        role_social_scores.append(role_social)
        rule_scores.append(rule_stake)
        pattern_scores.append(pattern)
        fairness_scores.append(fairness_recip)
        total_scores.append(total)

    candidates["query_source_family"] = source_family
    candidates["target_role"] = target_role
    candidates["component_role_social"] = role_social_scores
    candidates["component_rule_stake"] = rule_scores
    candidates["component_pattern"] = pattern_scores
    candidates["component_fairness_reciprocity"] = fairness_scores
    candidates["retrieval_score"] = total_scores
    return candidates.sort_values("retrieval_score", ascending=False)


def select_example_queries(
    family_df_map: Dict[str, pd.DataFrame],
    eval_rows: List[EvalResult],
) -> List[Tuple[str, int, str]]:
    ranked = sorted(eval_rows, key=lambda row: row.delta_r2, reverse=True)
    selected: List[Tuple[str, int, str]] = []
    used_families: set[str] = set()
    for result in ranked:
        if result.source_family in used_families:
            continue
        source_df = family_df_map[result.source_family].set_index("UserID")
        overlap_ids = [uid for uid in result.user_ids if uid in source_df.index]
        if not overlap_ids:
            continue
        overlap_df = source_df.loc[overlap_ids].copy()
        target_values = pd.Series(result.target_values, index=result.user_ids)
        overlap_df["target_value"] = target_values.loc[overlap_ids]
        exemplar_id = int(overlap_df["target_value"].sort_values().index[len(overlap_df) // 2])
        selected.append((result.source_family, exemplar_id, result.target))
        used_families.add(result.source_family)
        if len(selected) >= 3:
            break
    return selected


def plot_overlap_heatmap(joint_df: pd.DataFrame, out_path: Path) -> None:
    families = ["dictator", "trust_1", "trust_3", "ultimatum_1", "ultimatum_2", "PG"]
    labels = [ROLE_LABELS.get(col, "PG") for col in families]
    matrix = np.zeros((len(families), len(families)))
    for i, left in enumerate(families):
        for j, right in enumerate(families):
            matrix[i, j] = int(joint_df[[left, right]].dropna().shape[0])

    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    cmap = LinearSegmentedColormap.from_list("sand_teal", ["#f3e9d7", "#d4b483", "#5e8b7e", "#1d3557"])
    im = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_title("MobLab User Overlap Across Role-Specific Targets")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{int(matrix[i, j]):,}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.82, label="Users with both measures")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_r2_heatmap(
    eval_df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: Path,
    fmt: str,
) -> None:
    rows = ["dictator", "trust", "ultimatum", "pg"]
    cols = ["dictator", "trust_1", "trust_3", "ultimatum_1", "ultimatum_2"]
    labels_rows = ["Dictator", "Trust", "Ultimatum", "Public Goods"]
    labels_cols = [ROLE_LABELS[col] for col in cols]
    matrix = np.full((len(rows), len(cols)), np.nan)
    for _, row in eval_df.iterrows():
        r = rows.index(row["source_family"])
        c = cols.index(row["target"])
        matrix[r, c] = row[value_col]

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    cmap = LinearSegmentedColormap.from_list("orange_blue", ["#d1495b", "#f7d08a", "#fbf7ef", "#6c91bf", "#1d3557"])
    im = ax.imshow(matrix, cmap=cmap, vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax.set_xticks(np.arange(len(labels_cols)), labels=labels_cols, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels_rows)), labels=labels_rows)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = "" if np.isnan(value) else format(value, fmt)
            ax.text(j, i, label, ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.82, label=value_col.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_improvement_bar(eval_df: pd.DataFrame, out_path: Path) -> None:
    ordered = eval_df.sort_values("delta_r2", ascending=False).copy()
    labels = [
        f"{row.source_family.title()} -> {ROLE_LABELS[row.target]}"
        for row in ordered.itertuples()
    ]
    colors = ["#1f6f8b" if val >= 0 else "#c44536" for val in ordered["delta_r2"]]
    fig, ax = plt.subplots(figsize=(10.6, 6.8))
    ax.barh(labels, ordered["delta_r2"], color=colors, edgecolor="#17324d")
    ax.axvline(0, color="#17324d", linewidth=1.3)
    ax.set_xlabel("CV $R^2$ improvement from rich features")
    ax.set_title("How Much Do Multiround Features Add Beyond Collapsed Scalars?")
    for idx, row in enumerate(ordered.itertuples()):
        ax.text(row.delta_r2 + (0.004 if row.delta_r2 >= 0 else -0.004), idx, f"{row.delta_r2:.3f}", va="center", ha="left" if row.delta_r2 >= 0 else "right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_best_pair_scatter(best_eval: EvalResult, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    ax.scatter(best_eval.target_values, best_eval.rich_preds, s=38, alpha=0.50, color="#1f6f8b", edgecolors="none")
    lo = float(min(np.min(best_eval.target_values), np.min(best_eval.rich_preds)))
    hi = float(max(np.max(best_eval.target_values), np.max(best_eval.rich_preds)))
    ax.plot([lo, hi], [lo, hi], color="#c44536", linewidth=1.6, linestyle="--")
    ax.set_xlabel(f"Observed {ROLE_LABELS[best_eval.target]}")
    ax.set_ylabel("OOF rich-feature prediction")
    ax.set_title(
        f"Best Cross-Game Pair: {best_eval.source_family.title()} -> {ROLE_LABELS[best_eval.target]}\n"
        f"CV $R^2$ {best_eval.rich_r2:.3f} vs baseline {best_eval.baseline_r2:.3f}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_pgg_feature_space(
    pgg_df: pd.DataFrame,
    family_df_map: Dict[str, pd.DataFrame],
    example_queries: List[Tuple[str, int, str]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    color_map = {
        (0.0, 0.0): "#6c91bf",
        (1.0, 0.0): "#d1495b",
        (0.0, 1.0): "#2a9d8f",
        (1.0, 1.0): "#8d5a97",
    }
    for key, group in pgg_df.groupby(["punishment_enabled", "reward_enabled"]):
        label = {
            (0.0, 0.0): "No sanction modules",
            (1.0, 0.0): "Punishment only",
            (0.0, 1.0): "Reward only",
            (1.0, 1.0): "Punishment + reward",
        }[key]
        ax.scatter(
            group["pgg_prosociality"],
            0.5 * group["pgg_reciprocity"] + 0.5 * group["pgg_fairness"],
            s=12,
            alpha=0.18,
            color=color_map[key],
            label=label,
            edgecolors="none",
        )

    marker_map = {"dictator": "D", "trust": "o", "ultimatum": "s", "pg": "^"}
    for source_family, user_id, target_role in example_queries:
        row = family_df_map[source_family].set_index("UserID").loc[user_id]
        axes = family_to_query_axes(source_family, row)
        x = axes["prosociality"]
        y = nanmean([axes["reciprocity"], axes["fairness"]])
        ax.scatter(x, y, s=180, marker=marker_map[source_family], color="#111111", edgecolors="#fbf7ef", linewidth=1.2)
        ax.text(
            x + 0.015,
            y + 0.012,
            f"{source_family.title()} query\nfor {ROLE_LABELS[target_role]}",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "fc": "#fff8ee", "ec": "#d4b483"},
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Generalized Prosociality")
    ax.set_ylabel("Reciprocity / Fairness Axis")
    ax.set_title("PGG Library Feature Space with Example MobLab Queries")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_retrieval_components(retrieval_examples: pd.DataFrame, out_path: Path) -> None:
    if retrieval_examples.empty:
        return
    retrieval_examples = retrieval_examples.copy()
    retrieval_examples["row_label"] = (
        retrieval_examples["query_label"] + " | rank " + retrieval_examples["rank"].astype(str)
    )
    component_cols = [
        "component_role_social",
        "component_rule_stake",
        "component_pattern",
        "component_fairness_reciprocity",
        "retrieval_score",
    ]
    matrix = retrieval_examples[component_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.2, 6.8))
    cmap = LinearSegmentedColormap.from_list("cream_teal", ["#f3e9d7", "#d4b483", "#76a5af", "#1d3557"])
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(component_cols)), labels=[c.replace("_", " ").title() for c in component_cols], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(retrieval_examples)), labels=retrieval_examples["row_label"].tolist())
    ax.set_title("Retrieval Score Decomposition for Example Queries")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.82, label="Component score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_target_role_weights(out_path: Path) -> None:
    rows = list(TARGET_ROLE_WEIGHTS.keys())
    cols = ["prosociality", "fairness", "caution", "reciprocity", "stability"]
    matrix = np.array(
        [[TARGET_ROLE_WEIGHTS[row]["social"][col] for col in cols] for row in rows],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    cmap = LinearSegmentedColormap.from_list("amber_navy", ["#f3e9d7", "#d4b483", "#457b9d", "#1d3557"])
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=np.nanmax(matrix), aspect="auto")
    ax.set_xticks(np.arange(len(cols)), labels=[col.title() for col in cols], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(rows)), labels=[ROLE_LABELS[row] for row in rows])
    ax.set_title("Role-Conditioned Social Weights in the PGG Retrieval Score")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.82, label="Weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_summary_markdown(
    eval_df: pd.DataFrame,
    example_queries: List[Tuple[str, int, str]],
    retrieval_examples: pd.DataFrame,
) -> str:
    top_gain = eval_df.sort_values("delta_r2", ascending=False).head(5)
    top_rich = eval_df.sort_values("rich_r2", ascending=False).head(5)
    lines = [
        "# MobLab Multiround Transfer Analysis",
        "",
        "## Main Takeaways",
        "",
        "- Rich multiround features generally add more signal than the collapsed `joint.csv` scalars, but the gains remain modest.",
        "- Cross-game predictability is still limited: even the best source-target pair only explains a small slice of variance out of sample.",
        "- This weak-but-nonzero cross-game structure supports a retrieval design that conditions on repeated-game dynamics instead of only demographics or a single scalar summary.",
        "",
        "## Best Delta-R2 Pairs",
        "",
        "| Source family | Target role | N | Baseline R2 | Rich R2 | Delta R2 | Rich Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_gain.itertuples():
        lines.append(
            f"| {row.source_family} | {ROLE_LABELS[row.target]} | {row.n_users} | "
            f"{row.baseline_r2:.3f} | {row.rich_r2:.3f} | {row.delta_r2:.3f} | {row.rich_spearman:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Strongest Rich-Feature Pairs",
            "",
            "| Source family | Target role | N | Rich R2 | Rich MAE | Rich Spearman |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_rich.itertuples():
        lines.append(
            f"| {row.source_family} | {ROLE_LABELS[row.target]} | {row.n_users} | "
            f"{row.rich_r2:.3f} | {row.rich_mae:.3f} | {row.rich_spearman:.3f} |"
        )

    if example_queries:
        lines.extend(["", "## Example Retrieval Queries", ""])
        for source_family, user_id, target_role in example_queries:
            lines.append(f"- `{source_family}` query user `{user_id}` scored for target `{ROLE_LABELS[target_role]}`")
    if not retrieval_examples.empty:
        lines.extend(
            [
                "",
                "## Retrieval Design Notes",
                "",
                "- Final score = role-conditioned social similarity + rule/stake similarity + repeated-pattern similarity + reciprocity/fairness similarity.",
                "- Role conditioning changes which behavioral axes matter most. For example, trust banker prioritizes reciprocity and fairness, while dictator prioritizes generosity and low exploitation of the counterpart.",
                "- Rule/stake similarity is intentionally lower-weighted than behavior similarity, so retrieval is not dominated by raw environment matching alone.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numpy\..*")
    configure_plot_style()

    output_dir = args.output_dir.resolve()
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    joint_df = load_joint_baseline()
    family_df_map = family_tables()
    eval_rows = compare_scalar_vs_rich(joint_df=joint_df, family_df_map=family_df_map, seed=args.seed)
    eval_df = pd.DataFrame(
        [
            {
                "source_family": row.source_family,
                "target": row.target,
                "n_users": row.n_users,
                "baseline_r2": row.baseline_r2,
                "baseline_mae": row.baseline_mae,
                "baseline_spearman": row.baseline_spearman,
                "rich_r2": row.rich_r2,
                "rich_mae": row.rich_mae,
                "rich_spearman": row.rich_spearman,
                "delta_r2": row.delta_r2,
            }
            for row in eval_rows
        ]
    )
    eval_df.to_csv(output_dir / "crossgame_rich_vs_scalar.csv", index=False)

    for family, df in family_df_map.items():
        df.to_csv(output_dir / f"{family}_multiround_features.csv", index=False)

    pgg_df = pgg_library_feature_table()
    pgg_df.to_csv(output_dir / "pgg_library_features.csv", index=False)

    example_queries = select_example_queries(family_df_map, eval_rows)
    retrieval_rows: List[Dict[str, object]] = []
    for source_family, user_id, target_role in example_queries:
        feature_row = family_df_map[source_family].set_index("UserID").loc[user_id]
        candidates = build_retrieval_candidates(
            source_family=source_family,
            feature_row=feature_row,
            target_role=target_role,
            pgg_df=pgg_df,
        ).head(5)
        for rank, row in enumerate(candidates.itertuples(), start=1):
            retrieval_rows.append(
                {
                    "query_label": f"{source_family.title()} -> {ROLE_LABELS[target_role]}",
                    "source_family": source_family,
                    "query_user_id": user_id,
                    "target_role": target_role,
                    "rank": rank,
                    "split": row.split,
                    "gameId": row.gameId,
                    "playerId": row.playerId,
                    "component_role_social": row.component_role_social,
                    "component_rule_stake": row.component_rule_stake,
                    "component_pattern": row.component_pattern,
                    "component_fairness_reciprocity": row.component_fairness_reciprocity,
                    "retrieval_score": row.retrieval_score,
                    "oracle_preview": row.oracle_preview,
                }
            )
    retrieval_examples = pd.DataFrame(retrieval_rows)
    retrieval_examples.to_csv(output_dir / "retrieval_examples.csv", index=False)

    plot_overlap_heatmap(joint_df, plots_dir / "moblab_overlap_heatmap.png")
    plot_r2_heatmap(
        eval_df,
        value_col="rich_r2",
        title="Cross-Game Prediction with Rich Multiround Features",
        out_path=plots_dir / "rich_r2_heatmap.png",
        fmt=".3f",
    )
    plot_r2_heatmap(
        eval_df,
        value_col="delta_r2",
        title="Improvement Over Collapsed Scalar Baselines",
        out_path=plots_dir / "delta_r2_heatmap.png",
        fmt="+.3f",
    )
    plot_improvement_bar(eval_df, plots_dir / "delta_r2_bar.png")
    best_eval = max(eval_rows, key=lambda row: row.delta_r2)
    plot_best_pair_scatter(best_eval, plots_dir / "best_pair_scatter.png")
    plot_pgg_feature_space(pgg_df, family_df_map, example_queries, plots_dir / "pgg_feature_space.png")
    plot_retrieval_components(retrieval_examples, plots_dir / "retrieval_component_heatmap.png")
    plot_target_role_weights(plots_dir / "retrieval_role_weights.png")

    summary_md = build_summary_markdown(eval_df, example_queries, retrieval_examples)
    (output_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    print(f"Wrote analysis artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
