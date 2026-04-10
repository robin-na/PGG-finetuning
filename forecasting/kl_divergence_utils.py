from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def kl_divergence_from_probs(human_probs: np.ndarray, model_probs: np.ndarray) -> float:
    human = np.asarray(human_probs, dtype=float)
    model = np.asarray(model_probs, dtype=float)
    return float(np.sum(human * np.log(human / model)))


def smoothed_probs_from_counts(counts: np.ndarray, *, alpha: float) -> np.ndarray:
    counts_arr = np.asarray(counts, dtype=float)
    if counts_arr.ndim != 1:
        raise ValueError("counts must be a 1D array")
    k = counts_arr.size
    if k == 0:
        raise ValueError("counts must be non-empty")
    smoothed = counts_arr + float(alpha)
    return smoothed / float(smoothed.sum())


def categorical_kl_divergence(
    human_values: Iterable[object],
    model_values: Iterable[object],
    *,
    support: list[object],
    alpha: float = 1.0,
) -> float:
    human = pd.Series(list(human_values)).dropna()
    model = pd.Series(list(model_values)).dropna()
    if human.empty or model.empty:
        return float("nan")
    human_counts = np.array([int((human == value).sum()) for value in support], dtype=float)
    model_counts = np.array([int((model == value).sum()) for value in support], dtype=float)
    human_probs = smoothed_probs_from_counts(human_counts, alpha=alpha)
    model_probs = smoothed_probs_from_counts(model_counts, alpha=alpha)
    return kl_divergence_from_probs(human_probs, model_probs)


def integer_kl_divergence(
    human_values: Iterable[object],
    model_values: Iterable[object],
    *,
    min_value: int,
    max_value: int,
    alpha: float = 1.0,
) -> float:
    support = list(range(int(min_value), int(max_value) + 1))
    human = pd.Series(list(human_values)).dropna().astype(int)
    model = pd.Series(list(model_values)).dropna().astype(int)
    return categorical_kl_divergence(human, model, support=support, alpha=alpha)


def histogram_kl_divergence(
    human_values: Iterable[object],
    model_values: Iterable[object],
    *,
    bin_edges: np.ndarray,
    alpha: float = 1.0,
) -> float:
    human = pd.Series(list(human_values)).dropna().astype(float)
    model = pd.Series(list(model_values)).dropna().astype(float)
    if human.empty or model.empty:
        return float("nan")
    human_counts, _ = np.histogram(human.to_numpy(), bins=bin_edges)
    model_counts, _ = np.histogram(model.to_numpy(), bins=bin_edges)
    human_probs = smoothed_probs_from_counts(human_counts, alpha=alpha)
    model_probs = smoothed_probs_from_counts(model_counts, alpha=alpha)
    return kl_divergence_from_probs(human_probs, model_probs)


def mean_and_stderr(values: pd.Series) -> tuple[float, float]:
    clean = pd.Series(values).dropna().astype(float)
    if clean.empty:
        return float("nan"), float("nan")
    mean = float(clean.mean())
    if clean.shape[0] <= 1:
        return mean, float("nan")
    return mean, float(clean.std(ddof=1) / np.sqrt(clean.shape[0]))


def bootstrap_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return {
            "bootstrap_mean": float("nan"),
            "bootstrap_p05": float("nan"),
            "bootstrap_p95": float("nan"),
        }
    return {
        "bootstrap_mean": float(arr.mean()),
        "bootstrap_p05": float(np.quantile(arr, 0.05)),
        "bootstrap_p95": float(np.quantile(arr, 0.95)),
    }

