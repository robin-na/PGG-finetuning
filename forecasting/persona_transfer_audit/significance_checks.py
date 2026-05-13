"""Request-level uncertainty checks for persona-transfer behavior skew."""

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
    "mean_contribution_rate",
    "full_contribution_rate",
    "zero_contribution_rate",
    "contribution_sd",
    "messages_per_round",
    "reward_given_round_rate",
    "punish_given_round_rate",
    "punish_received_round_rate",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _se(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


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
    means = []
    n = len(values)
    for _ in range(iterations):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return (_percentile(means, 0.025), _percentile(means, 0.975))


def _cluster_bootstrap_ci(
    rows: list[dict[str, Any]],
    cluster_key: str,
    rng: random.Random,
    iterations: int,
) -> tuple[float, float]:
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_cluster[str(row[cluster_key])].append(float(row["diff"]))
    clusters = list(by_cluster)
    if not clusters:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(iterations):
        sampled_values = []
        for _ in clusters:
            sampled_values.extend(by_cluster[clusters[rng.randrange(len(clusters))]])
        means.append(_mean(sampled_values))
    return (_percentile(means, 0.025), _percentile(means, 0.975))


def _paired_sign_p(values: list[float]) -> float:
    positives = sum(value > 0 for value in values)
    negatives = sum(value < 0 for value in values)
    n = positives + negatives
    if n == 0:
        return float("nan")
    observed = min(positives, negatives)
    probability = sum(math.comb(n, k) for k in range(observed + 1)) / (2**n)
    return min(1.0, 2 * probability)


def run(args: argparse.Namespace) -> None:
    metadata_dir = args.metadata_dir.expanduser().resolve()
    matched_rows = _read_csv(metadata_dir / "matched_player_behavior_long.csv")
    candidate_rows = _read_csv(metadata_dir / "candidate_uniform_behavior_long.csv")

    metrics = args.metrics or DEFAULT_METRICS

    baseline_by_game_metric: dict[tuple[str, str], float] = {}
    for game_id in {row["game_id"] for row in candidate_rows}:
        game_rows = [row for row in candidate_rows if row["game_id"] == game_id]
        for metric in metrics:
            total_weight = sum(float(row["candidate_uniform_weight"]) for row in game_rows)
            if total_weight == 0:
                raise ValueError(f"Zero candidate baseline weight for game {game_id}")
            baseline_by_game_metric[(game_id, metric)] = (
                sum(float(row["candidate_uniform_weight"]) * float(row[metric]) for row in game_rows)
                / total_weight
            )

    request_keys = sorted({row["custom_id"] for row in matched_rows})
    manifest_by_request = {}
    for row in _read_csv(metadata_dir / "request_manifest.csv"):
        manifest_by_request[row["custom_id"]] = row

    rng = random.Random(args.seed)
    summary_rows: list[dict[str, Any]] = []
    request_diff_rows: list[dict[str, Any]] = []

    for metric in metrics:
        diffs = []
        metric_rows = []
        for custom_id in request_keys:
            rows = [row for row in matched_rows if row["custom_id"] == custom_id]
            if not rows:
                continue
            game_id = rows[0]["game_id"]
            persona_pid = rows[0]["persona_pid"]
            treatment_name = rows[0]["treatment_name"]
            matched_value = sum(float(row["match_probability"]) * float(row[metric]) for row in rows)
            baseline_value = baseline_by_game_metric[(game_id, metric)]
            diff = matched_value - baseline_value
            diff_row = {
                "custom_id": custom_id,
                "persona_pid": persona_pid,
                "game_id": game_id,
                "treatment_name": treatment_name,
                "metric": metric,
                "matched_value": matched_value,
                "candidate_uniform_value": baseline_value,
                "diff": diff,
            }
            diffs.append(diff)
            metric_rows.append(diff_row)
            request_diff_rows.append(diff_row)

        ci_low, ci_high = _bootstrap_ci(diffs, rng, args.iterations)
        game_low, game_high = _cluster_bootstrap_ci(metric_rows, "game_id", rng, args.iterations)
        persona_low, persona_high = _cluster_bootstrap_ci(metric_rows, "persona_pid", rng, args.iterations)
        summary_rows.append(
            {
                "metric": metric,
                "n": len(diffs),
                "mean_diff": _mean(diffs),
                "request_se": _se(diffs),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "paired_sign_p": _paired_sign_p(diffs),
                "game_id_cluster_ci_low": game_low,
                "game_id_cluster_ci_high": game_high,
                "persona_pid_cluster_ci_low": persona_low,
                "persona_pid_cluster_ci_high": persona_high,
            }
        )

    _write_csv(metadata_dir / "comprehensive_significance_checks.csv", summary_rows)
    _write_csv(metadata_dir / "request_level_behavior_differences.csv", request_diff_rows)
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
