from __future__ import annotations

import csv
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple

from .aggregation import SummaryRow
@dataclass
class AlignmentRow:
    config: str
    metric: str
    sim_mean: float
    human_mean: float
    human_std: float


@dataclass
class MetricSummary:
    metric: str
    rmse: float
    mae: float
    r2: float
    n_pairs: int


@dataclass
class ConfigMetricSummary:
    config: str
    metric: str
    rmse: float
    noise_ceiling: float
    n_samples: int


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def _finite(values: Iterable[float]) -> List[float]:
    return [float(value) for value in values if value is not None and isfinite(value)]


def _mae(errors: Iterable[float]) -> float:
    errors = list(errors)
    return _mean(abs(e) for e in errors) if errors else 0.0


def _rmse(errors: Iterable[float]) -> float:
    errors = list(errors)
    return sqrt(_mean(e * e for e in errors)) if errors else 0.0


def _r2(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return 0.0
    mean_true = _mean(y_true)
    ss_tot = sum((y - mean_true) ** 2 for y in y_true)
    ss_res = sum((y - y_hat) ** 2 for y, y_hat in zip(y_true, y_pred))
    return 1 - ss_res / ss_tot if ss_tot else 0.0


def _to_config_map(summaries: Dict[str, List[SummaryRow]], metric: str) -> Dict[str, List[float]]:
    mapping: Dict[str, List[float]] = {}
    for config_name, rows in summaries.items():
        values = [row.metrics.get(metric) for row in rows if metric in row.metrics]
        mapping[config_name] = _finite(values)
    return mapping


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return pstdev(values)


def compare_configs(
    sim_summaries: Dict[str, List[SummaryRow]],
    human_summaries: Dict[str, List[SummaryRow]],
    metrics: Iterable[str],
) -> Tuple[List[AlignmentRow], List[MetricSummary]]:
    alignment_rows: List[AlignmentRow] = []
    metric_summaries: List[MetricSummary] = []

    for metric in metrics:
        sim_map = _to_config_map(sim_summaries, metric)
        human_map = _to_config_map(human_summaries, metric)

        errors: List[float] = []
        y_true: List[float] = []
        y_pred: List[float] = []
        n_pairs = 0

        for sim_config, sim_values in sim_map.items():
            human_values = human_map.get(sim_config)
            if not human_values:
                continue
            sim_mean = _mean(sim_values)
            human_mean = _mean(human_values)
            human_std = _std(human_values)
            alignment_rows.append(
                AlignmentRow(
                    config=sim_config,
                    metric=metric,
                    sim_mean=sim_mean,
                    human_mean=human_mean,
                    human_std=human_std,
                )
            )
            error = sim_mean - human_mean
            errors.append(error)
            y_true.append(human_mean)
            y_pred.append(sim_mean)
            n_pairs += 1

        metric_summaries.append(
            MetricSummary(
                metric=metric,
                rmse=_rmse(errors),
                mae=_mae(errors),
                r2=_r2(y_true, y_pred),
                n_pairs=n_pairs,
            )
        )

    return alignment_rows, metric_summaries


def write_alignment(path: Path, rows: List[AlignmentRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "metric", "sim_mean", "human_mean", "human_std"])
        for row in rows:
            writer.writerow(
                [row.config, row.metric, row.sim_mean, row.human_mean, row.human_std]
            )


def write_metric_summary(path: Path, rows: List[MetricSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "rmse", "mae", "r2", "n_pairs"])
        for row in rows:
            writer.writerow([row.metric, row.rmse, row.mae, row.r2, row.n_pairs])


def compare_by_config(
    sim_summaries: Dict[str, List[SummaryRow]],
    human_summaries: Dict[str, List[SummaryRow]],
    metrics: Iterable[str],
) -> List[ConfigMetricSummary]:
    sim_metric_means: Dict[str, Dict[str, float]] = {}
    human_metric_means: Dict[str, Dict[str, float]] = {}
    human_metric_std: Dict[str, Dict[str, float]] = {}
    human_metric_counts: Dict[str, Dict[str, int]] = {}

    for metric in metrics:
        sim_map = _to_config_map(sim_summaries, metric)
        human_map = _to_config_map(human_summaries, metric)
        for config_name, values in sim_map.items():
            sim_metric_means.setdefault(config_name, {})[metric] = _mean(values)
        for config_name, values in human_map.items():
            human_metric_means.setdefault(config_name, {})[metric] = _mean(values)
            std = pstdev(values) if len(values) > 1 else 0.0
            human_metric_std.setdefault(config_name, {})[metric] = std
            human_metric_counts.setdefault(config_name, {})[metric] = len(values)

    summaries: List[ConfigMetricSummary] = []
    for config_name, sim_metrics in sim_metric_means.items():
        human_metrics = human_metric_means.get(config_name)
        if not human_metrics:
            continue
        for metric in metrics:
            if metric not in sim_metrics or metric not in human_metrics:
                continue
            sim_value = sim_metrics[metric]
            human_value = human_metrics[metric]
            error = sim_value - human_value
            noise_ceiling = human_metric_std.get(config_name, {}).get(metric, 0.0)
            n_samples = human_metric_counts.get(config_name, {}).get(metric, 0)
            summaries.append(
                ConfigMetricSummary(
                    config=config_name,
                    metric=metric,
                    rmse=_rmse([error]),
                    noise_ceiling=noise_ceiling,
                    n_samples=n_samples,
                )
            )
    return summaries


def write_config_metric_summary(path: Path, rows: List[ConfigMetricSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "metric", "rmse", "noise_ceiling", "n_samples"])
        for row in rows:
            writer.writerow(
                [row.config, row.metric, row.rmse, row.noise_ceiling, row.n_samples]
            )
