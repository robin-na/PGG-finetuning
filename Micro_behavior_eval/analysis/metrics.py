from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


TypedTarget = Tuple[str, str]

ROW_SCORE_COLUMNS = [
    "contrib_error",
    "contrib_abs_error",
    "contrib_sq_error",
    "action_exact_match",
    "target_set_exact",
    "target_precision",
    "target_recall",
    "target_f1",
    "target_jaccard",
    "target_hit_any",
    "unit_mae_on_overlap",
    "unit_exact_on_overlap_rate",
    "actual_any_action",
    "pred_any_action",
    "actual_any_punish",
    "pred_any_punish",
    "actual_any_reward",
    "pred_any_reward",
]

AGG_METRIC_COLUMNS = [
    "n_rows",
    "predicted_contribution_parsed",
    "contrib_mae",
    "contrib_rmse",
    "contrib_bias",
    "contrib_corr",
    "action_exact_match",
    "target_set_exact",
    "target_precision",
    "target_recall",
    "target_f1",
    "target_jaccard",
    "target_hit_any",
    "unit_mae_on_overlap",
    "unit_exact_on_overlap_rate",
    "actual_any_action_rate",
    "pred_any_action_rate",
    "actual_any_punish_rate",
    "pred_any_punish_rate",
    "actual_any_reward_rate",
    "pred_any_reward_rate",
]

CONTRIB_REGIME_METRIC_COLUMNS = [
    "n_rows",
    "predicted_contribution_parsed",
    "contrib_mae",
    "contrib_mae_norm20",
    "contrib_rmse",
    "contrib_bias",
    "contrib_corr",
    "contrib_binary_n",
    "contrib_binary_accuracy",
    "contrib_binary_precision",
    "contrib_binary_recall",
    "contrib_binary_f1",
]


def _typed_actions(punished: Dict[str, int], rewarded: Dict[str, int]) -> Dict[TypedTarget, int]:
    out: Dict[TypedTarget, int] = {}
    for pid, units in punished.items():
        if int(units) > 0:
            out[("P", str(pid))] = int(units)
    for pid, units in rewarded.items():
        if int(units) > 0:
            out[("R", str(pid))] = int(units)
    return out


def _set_precision_recall_f1(pred: Set[TypedTarget], actual: Set[TypedTarget]) -> Tuple[float, float, float]:
    if not pred and not actual:
        return 1.0, 1.0, 1.0
    intersect = pred.intersection(actual)
    precision = float(len(intersect)) / float(len(pred)) if pred else 0.0
    recall = float(len(intersect)) / float(len(actual)) if actual else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def _set_jaccard(pred: Set[TypedTarget], actual: Set[TypedTarget]) -> float:
    union = pred.union(actual)
    if not union:
        return 1.0
    return float(len(pred.intersection(actual))) / float(len(union))


def _target_hit_any(pred: Set[TypedTarget], actual: Set[TypedTarget]) -> float:
    if not actual:
        return 1.0 if not pred else 0.0
    return 1.0 if pred.intersection(actual) else 0.0


def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()

    pred_contrib = pd.to_numeric(scored["predicted_contribution"], errors="coerce")
    actual_contrib = pd.to_numeric(scored["actual_contribution"], errors="coerce")
    contrib_error = pred_contrib - actual_contrib
    scored["contrib_error"] = contrib_error
    scored["contrib_abs_error"] = contrib_error.abs()
    scored["contrib_sq_error"] = contrib_error * contrib_error
    scored["contrib_abs_error_norm20"] = scored["contrib_abs_error"] / 20.0

    row_metrics: List[Dict[str, float]] = []
    for _, row in scored.iterrows():
        pred_punish = row.get("predicted_punished_pid_dict") or {}
        pred_reward = row.get("predicted_rewarded_pid_dict") or {}
        actual_punish = row.get("actual_punished_pid_dict") or {}
        actual_reward = row.get("actual_rewarded_pid_dict") or {}

        pred_typed = _typed_actions(pred_punish, pred_reward)
        actual_typed = _typed_actions(actual_punish, actual_reward)
        pred_targets = set(pred_typed.keys())
        actual_targets = set(actual_typed.keys())
        overlap = pred_targets.intersection(actual_targets)

        precision, recall, f1 = _set_precision_recall_f1(pred_targets, actual_targets)
        jaccard = _set_jaccard(pred_targets, actual_targets)
        target_hit_any = _target_hit_any(pred_targets, actual_targets)

        if overlap:
            diffs = [abs(int(pred_typed[key]) - int(actual_typed[key])) for key in overlap]
            unit_mae = float(np.mean(diffs))
            unit_exact = float(np.mean([1.0 if d == 0 else 0.0 for d in diffs]))
        else:
            unit_mae = float("nan")
            unit_exact = float("nan")

        row_metrics.append(
            {
                "action_exact_match": 1.0 if pred_typed == actual_typed else 0.0,
                "target_set_exact": 1.0 if pred_targets == actual_targets else 0.0,
                "target_precision": precision,
                "target_recall": recall,
                "target_f1": f1,
                "target_jaccard": jaccard,
                "target_hit_any": target_hit_any,
                "unit_mae_on_overlap": unit_mae,
                "unit_exact_on_overlap_rate": unit_exact,
                "actual_any_action": 1.0 if actual_targets else 0.0,
                "pred_any_action": 1.0 if pred_targets else 0.0,
                "actual_any_punish": 1.0 if any(k[0] == "P" for k in actual_targets) else 0.0,
                "pred_any_punish": 1.0 if any(k[0] == "P" for k in pred_targets) else 0.0,
                "actual_any_reward": 1.0 if any(k[0] == "R" for k in actual_targets) else 0.0,
                "pred_any_reward": 1.0 if any(k[0] == "R" for k in pred_targets) else 0.0,
            }
        )

    for col in ROW_SCORE_COLUMNS:
        if col in {"contrib_error", "contrib_abs_error", "contrib_sq_error"}:
            continue
        scored[col] = [m[col] for m in row_metrics] if row_metrics else pd.Series(dtype=float)

    return scored


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    pair = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(pair) < 2:
        return float("nan")
    if pair["a"].nunique(dropna=True) < 2 or pair["b"].nunique(dropna=True) < 2:
        return float("nan")
    corr = pair["a"].corr(pair["b"])
    return float(corr) if corr is not None else float("nan")


def _mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(pd.to_numeric(series, errors="coerce").mean())


def _group_metrics(group: pd.DataFrame) -> Dict[str, float]:
    parse_rate = _mean(group["predicted_contribution_parsed_bool"].astype(float))
    contrib_mae = _mean(group["contrib_abs_error"])
    rmse_base = _mean(group["contrib_sq_error"])
    contrib_rmse = float(math.sqrt(rmse_base)) if not math.isnan(rmse_base) else float("nan")
    contrib_bias = _mean(group["contrib_error"])
    contrib_corr = _safe_corr(group["predicted_contribution"], group["actual_contribution"])

    metrics = {
        "n_rows": int(len(group)),
        "predicted_contribution_parsed": parse_rate,
        "contrib_mae": contrib_mae,
        "contrib_rmse": contrib_rmse,
        "contrib_bias": contrib_bias,
        "contrib_corr": contrib_corr,
        "action_exact_match": _mean(group["action_exact_match"]),
        "target_set_exact": _mean(group["target_set_exact"]),
        "target_precision": _mean(group["target_precision"]),
        "target_recall": _mean(group["target_recall"]),
        "target_f1": _mean(group["target_f1"]),
        "target_jaccard": _mean(group["target_jaccard"]),
        "target_hit_any": _mean(group["target_hit_any"]),
        "unit_mae_on_overlap": _mean(group["unit_mae_on_overlap"]),
        "unit_exact_on_overlap_rate": _mean(group["unit_exact_on_overlap_rate"]),
        "actual_any_action_rate": _mean(group["actual_any_action"]),
        "pred_any_action_rate": _mean(group["pred_any_action"]),
        "actual_any_punish_rate": _mean(group["actual_any_punish"]),
        "pred_any_punish_rate": _mean(group["pred_any_punish"]),
        "actual_any_reward_rate": _mean(group["actual_any_reward"]),
        "pred_any_reward_rate": _mean(group["pred_any_reward"]),
    }
    return metrics


def aggregate_scores(scored_df: pd.DataFrame, group_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if group_cols is None:
        if scored_df.empty:
            return pd.DataFrame(columns=AGG_METRIC_COLUMNS)
        return pd.DataFrame([_group_metrics(scored_df)])

    group_cols = list(group_cols)
    all_columns = group_cols + AGG_METRIC_COLUMNS
    if scored_df.empty:
        return pd.DataFrame(columns=all_columns)

    rows: List[Dict[str, float]] = []
    grouped = scored_df.groupby(group_cols, dropna=False, sort=True)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        rec = {col: val for col, val in zip(group_cols, key)}
        rec.update(_group_metrics(group))
        rows.append(rec)
    return pd.DataFrame(rows, columns=all_columns)


def _to_bool_or_nan(value: object) -> object:
    if isinstance(value, bool):
        return value
    if value is None:
        return float("nan")
    try:
        if pd.isna(value):
            return float("nan")
    except Exception:
        pass
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return float("nan")


def _binary_scores(actual_pos: pd.Series, pred_pos: pd.Series) -> Dict[str, float]:
    if len(actual_pos) == 0:
        return {
            "contrib_binary_n": 0,
            "contrib_binary_accuracy": float("nan"),
            "contrib_binary_precision": float("nan"),
            "contrib_binary_recall": float("nan"),
            "contrib_binary_f1": float("nan"),
        }

    actual = actual_pos.astype(bool)
    pred = pred_pos.astype(bool)
    tp = int((actual & pred).sum())
    tn = int((~actual & ~pred).sum())
    fp = int((~actual & pred).sum())
    fn = int((actual & ~pred).sum())
    n = int(len(actual))

    accuracy = float((tp + tn) / n) if n > 0 else float("nan")
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))
    return {
        "contrib_binary_n": n,
        "contrib_binary_accuracy": accuracy,
        "contrib_binary_precision": precision,
        "contrib_binary_recall": recall,
        "contrib_binary_f1": f1,
    }


def _contrib_regime_group_metrics(
    group: pd.DataFrame,
    regime_value: object,
    threshold: float,
) -> Dict[str, float]:
    is_all_or_nothing = bool(regime_value)
    parse_rate = _mean(group["predicted_contribution_parsed_bool"].astype(float))
    contrib_mae = _mean(group["contrib_abs_error"])
    contrib_mae_norm20 = _mean(group["contrib_abs_error_norm20"])
    rmse_base = _mean(group["contrib_sq_error"])
    contrib_rmse = float(math.sqrt(rmse_base)) if not math.isnan(rmse_base) else float("nan")
    contrib_bias = _mean(group["contrib_error"])
    contrib_corr = _safe_corr(group["predicted_contribution"], group["actual_contribution"])

    out: Dict[str, float] = {
        "n_rows": int(len(group)),
        "predicted_contribution_parsed": parse_rate,
        "contrib_mae": contrib_mae,
        "contrib_mae_norm20": contrib_mae_norm20,
        "contrib_rmse": contrib_rmse,
        "contrib_bias": contrib_bias,
        "contrib_corr": contrib_corr,
    }

    if is_all_or_nothing:
        pair = group[["actual_contribution", "predicted_contribution"]].copy()
        pair["actual_contribution"] = pd.to_numeric(pair["actual_contribution"], errors="coerce")
        pair["predicted_contribution"] = pd.to_numeric(pair["predicted_contribution"], errors="coerce")
        pair = pair.dropna(subset=["actual_contribution", "predicted_contribution"])
        actual_pos = pair["actual_contribution"] >= float(threshold)
        pred_pos = pair["predicted_contribution"] >= float(threshold)
        out.update(_binary_scores(actual_pos, pred_pos))
    else:
        out.update(
            {
                "contrib_binary_n": 0,
                "contrib_binary_accuracy": float("nan"),
                "contrib_binary_precision": float("nan"),
                "contrib_binary_recall": float("nan"),
                "contrib_binary_f1": float("nan"),
            }
        )
    return out


def aggregate_contribution_by_regime(
    scored_df: pd.DataFrame,
    group_cols: Optional[Sequence[str]] = None,
    regime_col: str = "CONFIG_allOrNothing",
    contrib_max: float = 20.0,
) -> pd.DataFrame:
    base_cols = list(group_cols or [])
    all_cols = [*base_cols, regime_col, *CONTRIB_REGIME_METRIC_COLUMNS]
    if scored_df.empty or regime_col not in scored_df.columns:
        return pd.DataFrame(columns=all_cols)

    threshold = float(contrib_max) / 2.0
    work = scored_df.copy()
    work[regime_col] = work[regime_col].map(_to_bool_or_nan)
    work = work[work[regime_col].isin([True, False])].copy()
    if work.empty:
        return pd.DataFrame(columns=all_cols)

    rows: List[Dict[str, float]] = []
    grouped = work.groupby([*base_cols, regime_col], dropna=False, sort=True)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        key_map = {col: val for col, val in zip([*base_cols, regime_col], key)}
        rec = dict(key_map)
        rec.update(_contrib_regime_group_metrics(group, key_map.get(regime_col), threshold))
        rows.append(rec)
    return pd.DataFrame(rows, columns=all_cols)
