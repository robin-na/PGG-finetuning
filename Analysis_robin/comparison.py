from __future__ import annotations

import csv
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple

from .aggregation import SummaryRow
from .config import build_pair


@dataclass
class AlignmentRow:
    config_pair: str
    metric: str
    sim_mean: float
    human_mean: float
    human_noise: float


@dataclass
class MetricSummary:
    metric: str
    rmse: float
    mae: float
    r2: float
    n_pairs: int


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


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
        mapping[config_name] = [float(v) for v in values]
    return mapping


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
            pair = build_pair(sim_config)
            if not pair:
                continue
            human_values = human_map.get(pair.human_config)
            if not human_values:
                continue
            sim_mean = _mean(sim_values)
            human_mean = _mean(human_values)
            human_noise = pstdev(human_values) if len(human_values) > 1 else 0.0
            alignment_rows.append(
                AlignmentRow(
                    config_pair=f"{pair.sim_config} vs {pair.human_config}",
                    metric=metric,
                    sim_mean=sim_mean,
                    human_mean=human_mean,
                    human_noise=human_noise,
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
        writer.writerow(["config_pair", "metric", "sim_mean", "human_mean", "human_noise"])
        for row in rows:
            writer.writerow(
                [row.config_pair, row.metric, row.sim_mean, row.human_mean, row.human_noise]
            )


def write_metric_summary(path: Path, rows: List[MetricSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "rmse", "mae", "r2", "n_pairs"])
        for row in rows:
            writer.writerow([row.metric, row.rmse, row.mae, row.r2, row.n_pairs])
