"""Uncertainty checks for chip-bargaining persona-transfer behavior skew."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    "final_surplus",
    "final_welfare",
    "proposer_mean_net_surplus",
    "proposer_acceptance_rate",
    "proposer_mean_trade_ratio",
    "response_acceptance_rate",
    "response_mean_net_surplus_if_accepted",
    "received_trade_rate",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_ci(values: list[float], rng: random.Random, iterations: int) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    n = len(values)
    estimates = []
    for _ in range(iterations):
        estimates.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return (_percentile(estimates, 0.025), _percentile(estimates, 0.975))


def _cluster_bootstrap_ci(
    rows: list[dict[str, Any]],
    cluster_key: str,
    rng: random.Random,
    iterations: int,
) -> tuple[float, float]:
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(cluster_key)
        if value is None:
            continue
        by_cluster[str(value)].append(float(row["diff"]))
    clusters = list(by_cluster)
    if not clusters:
        return (float("nan"), float("nan"))
    estimates = []
    for _ in range(iterations):
        sampled_values = []
        for _ in clusters:
            sampled_values.extend(by_cluster[clusters[rng.randrange(len(clusters))]])
        estimates.append(_mean(sampled_values))
    return (_percentile(estimates, 0.025), _percentile(estimates, 0.975))


def _paired_sign_p(values: list[float]) -> float:
    positives = sum(value > 0 for value in values)
    negatives = sum(value < 0 for value in values)
    n = positives + negatives
    if n == 0:
        return float("nan")
    observed = min(positives, negatives)
    probability = sum(math.comb(n, k) for k in range(observed + 1)) / (2**n)
    return min(1.0, 2 * probability)


def _candidate_baselines(candidate_rows: list[dict[str, str]], metrics: list[str]) -> dict[tuple[str, str], float]:
    baseline: dict[tuple[str, str], float] = {}
    record_ids = sorted({row["record_id"] for row in candidate_rows})
    for record_id in record_ids:
        rows = [row for row in candidate_rows if row["record_id"] == record_id]
        for metric in metrics:
            values = []
            weights = []
            for row in rows:
                value = float(row[metric])
                if math.isnan(value):
                    continue
                values.append(value)
                weights.append(float(row["candidate_uniform_weight"]))
            denominator = sum(weights)
            if denominator == 0:
                raise ValueError(f"Zero candidate baseline weight for record {record_id}")
            baseline[(record_id, metric)] = sum(value * weight for value, weight in zip(values, weights)) / denominator
    return baseline


def run(args: argparse.Namespace) -> None:
    metadata_dir = args.metadata_dir.expanduser().resolve()
    matched_rows = _read_csv(metadata_dir / "chip_matched_player_behavior_long.csv")
    candidate_rows = _read_csv(metadata_dir / "chip_candidate_uniform_behavior_long.csv")
    metrics = args.metrics or DEFAULT_METRICS
    baseline_by_record_metric = _candidate_baselines(candidate_rows, metrics)

    request_ids = sorted({row["custom_id"] for row in matched_rows})
    rng = random.Random(args.seed)
    request_diff_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for metric in metrics:
        metric_rows = []
        diffs = []
        for custom_id in request_ids:
            rows = [row for row in matched_rows if row["custom_id"] == custom_id]
            if not rows:
                continue
            record_id = rows[0]["record_id"]
            persona_pid = rows[0]["persona_pid"]
            matched_value = 0.0
            denominator = 0.0
            for row in rows:
                value = float(row[metric])
                if math.isnan(value):
                    continue
                probability = float(row["match_probability"])
                matched_value += probability * value
                denominator += probability
            if denominator == 0:
                continue
            matched_value /= denominator
            baseline_value = baseline_by_record_metric[(record_id, metric)]
            diff = matched_value - baseline_value
            diff_row = {
                "custom_id": custom_id,
                "persona_pid": persona_pid,
                "record_id": record_id,
                "metric": metric,
                "matched_value": matched_value,
                "candidate_uniform_value": baseline_value,
                "diff": diff,
            }
            diffs.append(diff)
            metric_rows.append(diff_row)
            request_diff_rows.append(diff_row)

        iid_low, iid_high = _bootstrap_ci(diffs, rng, args.iterations)
        record_low, record_high = _cluster_bootstrap_ci(metric_rows, "record_id", rng, args.iterations)
        persona_low, persona_high = _cluster_bootstrap_ci(metric_rows, "persona_pid", rng, args.iterations)
        summary_rows.append(
            {
                "metric": metric,
                "n_requests": len(diffs),
                "mean_diff": _mean(diffs),
                "iid_low": iid_low,
                "iid_high": iid_high,
                "record_low": record_low,
                "record_high": record_high,
                "persona_low": persona_low,
                "persona_high": persona_high,
                "paired_sign_p": _paired_sign_p(diffs),
            }
        )

    _write_csv(metadata_dir / "chip_significance_checks.csv", summary_rows)
    _write_csv(metadata_dir / "chip_request_level_behavior_differences.csv", request_diff_rows)
    print(json.dumps({"metadata_dir": str(metadata_dir), "metrics": len(summary_rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--metrics", nargs="*", default=None)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
