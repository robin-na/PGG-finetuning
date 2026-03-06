from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


METRICS = ["r2", "r2_oos_train_mean", "rmse"]


def _metric_from_mse(mse: np.ndarray, den_test: np.ndarray, den_train: np.ndarray) -> dict[str, np.ndarray]:
    r2 = np.full_like(mse, np.nan, dtype=float)
    r2_oos = np.full_like(mse, np.nan, dtype=float)
    valid_test = den_test > 0
    valid_train = den_train > 0
    r2[valid_test] = 1.0 - mse[valid_test] / den_test[valid_test]
    r2_oos[valid_train] = 1.0 - mse[valid_train] / den_train[valid_train]
    rmse = np.sqrt(np.maximum(mse, 0.0))
    return {"r2": r2, "r2_oos_train_mean": r2_oos, "rmse": rmse}


def _point_metrics(y: np.ndarray, pred: np.ndarray, train_mean: float) -> dict[str, float]:
    mse = float(np.mean((y - pred) ** 2))
    den_test = float(np.mean((y - float(np.mean(y))) ** 2))
    den_train = float(np.mean((y - float(train_mean)) ** 2))
    r2 = float("nan") if den_test <= 0 else float(1.0 - mse / den_test)
    r2_oos = float("nan") if den_train <= 0 else float(1.0 - mse / den_train)
    return {"r2": r2, "r2_oos_train_mean": r2_oos, "rmse": float(np.sqrt(max(mse, 0.0)))}


def _point_ceiling(y: np.ndarray, sampling_terms: np.ndarray, train_mean: float) -> dict[str, float]:
    mse = float(np.mean(sampling_terms))
    den_test = float(np.mean((y - float(np.mean(y))) ** 2))
    den_train = float(np.mean((y - float(train_mean)) ** 2))
    r2 = float("nan") if den_test <= 0 else float(1.0 - mse / den_test)
    r2_oos = float("nan") if den_train <= 0 else float(1.0 - mse / den_train)
    return {"r2": r2, "r2_oos_train_mean": r2_oos, "rmse": float(np.sqrt(max(mse, 0.0)))}


def bootstrap_methods_by_groups(
    *,
    y_true: np.ndarray,
    pred_by_method: dict[str, np.ndarray],
    train_mean_target: float,
    sampling_terms: np.ndarray | None = None,
    n_boot: int = 5000,
    seed: int = 42,
    ci_alpha: float = 0.05,
    comparison_pairs: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, dict[str, float]]]:
    y = np.asarray(y_true, dtype=float)
    n = len(y)
    if n == 0:
        raise ValueError("Empty y_true for bootstrap.")

    methods = list(pred_by_method.keys())
    preds = {m: np.asarray(v, dtype=float) for m, v in pred_by_method.items()}
    for m, p in preds.items():
        if len(p) != n:
            raise ValueError(f"Length mismatch for method={m}: len(pred)={len(p)}, len(y)={n}")

    if sampling_terms is not None:
        sampling_terms = np.asarray(sampling_terms, dtype=float)
        if len(sampling_terms) != n:
            raise ValueError(
                f"Length mismatch for sampling terms: len(terms)={len(sampling_terms)}, len(y)={n}"
            )
        methods = methods + ["noise_ceiling"]

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    yb = y[idx]
    yb_mean = yb.mean(axis=1, keepdims=True)
    den_test = np.mean((yb - yb_mean) ** 2, axis=1)
    den_train = np.mean((yb - float(train_mean_target)) ** 2, axis=1)

    boot_arrays: dict[str, dict[str, np.ndarray]] = {}
    point_metrics: dict[str, dict[str, float]] = {}

    for m in pred_by_method:
        pb = preds[m][idx]
        mse = np.mean((yb - pb) ** 2, axis=1)
        boot_arrays[m] = _metric_from_mse(mse, den_test, den_train)
        point_metrics[m] = _point_metrics(y, preds[m], train_mean_target)

    if sampling_terms is not None:
        mse_floor = np.mean(sampling_terms[idx], axis=1)
        boot_arrays["noise_ceiling"] = _metric_from_mse(mse_floor, den_test, den_train)
        point_metrics["noise_ceiling"] = _point_ceiling(y, sampling_terms, train_mean_target)

    q_low = 100.0 * (ci_alpha / 2.0)
    q_high = 100.0 * (1.0 - ci_alpha / 2.0)

    method_rows: list[dict[str, Any]] = []
    for m in methods:
        for metric in METRICS:
            arr = boot_arrays[m][metric]
            method_rows.append(
                {
                    "method": m,
                    "metric": metric,
                    "point": float(point_metrics[m][metric]),
                    "bootstrap_mean": float(np.nanmean(arr)),
                    "ci_low": float(np.nanpercentile(arr, q_low)),
                    "ci_high": float(np.nanpercentile(arr, q_high)),
                    "n_boot": int(n_boot),
                    "n_groups": int(n),
                }
            )
    method_ci_df = pd.DataFrame(method_rows)

    if comparison_pairs is None:
        base_methods = [m for m in pred_by_method.keys()]
        comparison_pairs = list(combinations(base_methods, 2))

    delta_rows: list[dict[str, Any]] = []
    for a, b in comparison_pairs:
        if a not in boot_arrays or b not in boot_arrays:
            continue
        for metric in METRICS:
            if metric == "rmse":
                # Positive means model a has lower RMSE than model b.
                delta = boot_arrays[b][metric] - boot_arrays[a][metric]
                point = point_metrics[b][metric] - point_metrics[a][metric]
                direction = "positive_is_better_for_a"
            else:
                # Positive means model a has higher R2 than model b.
                delta = boot_arrays[a][metric] - boot_arrays[b][metric]
                point = point_metrics[a][metric] - point_metrics[b][metric]
                direction = "positive_is_better_for_a"

            ci_low = float(np.nanpercentile(delta, q_low))
            ci_high = float(np.nanpercentile(delta, q_high))
            p_le_zero = float(np.nanmean(delta <= 0.0))
            p_ge_zero = float(np.nanmean(delta >= 0.0))
            delta_rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "metric": metric,
                    "delta_point": float(point),
                    "delta_bootstrap_mean": float(np.nanmean(delta)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_le_zero": p_le_zero,
                    "p_ge_zero": p_ge_zero,
                    "ci_excludes_zero": bool((ci_low > 0.0) or (ci_high < 0.0)),
                    "direction": direction,
                    "n_boot": int(n_boot),
                    "n_groups": int(n),
                }
            )
    delta_df = pd.DataFrame(delta_rows)

    return method_ci_df, delta_df, boot_arrays, point_metrics
