"""Clustered uncertainty checks for PGG demographic skew."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_METADATA_DIR = Path(
    "forecasting/persona_transfer_audit/metadata/"
    "twin_direct_summary_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2"
)

CATEGORICAL_SPECS = {
    "gender_self_report": ["Man", "Woman"],
    "prolific_sex": ["Man", "Woman"],
    "education_self_report": ["High school", "Bachelor", "Master", "Other"],
    "country_of_residence": ["United States", "United Kingdom"],
    "nationality": ["United States", "United Kingdom"],
    "employment_status": [
        "Full-Time",
        "Part-Time",
        "Not in paid work (e.g. homemaker', 'retired or disabled)",
    ],
}

NUMERIC_FIELDS = ["age_self_report", "age_prolific", "prolific_total_approvals"]


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


def _values_by_distribution_request(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    by_dist_req: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        distribution = row["distribution"]
        custom_id = row["custom_id"]
        if custom_id and distribution in {"candidate_uniform", "matched_probability", "matched_top1"}:
            by_dist_req[distribution][custom_id].append(row)
    return by_dist_req


def _numeric_value(rows: list[dict[str, str]], field: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(field, "")
        if value not in {"", None}:
            values.append((float(value), float(row["weight"])))
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in values) / total_weight


def _category_share(rows: list[dict[str, str]], field: str, category: str) -> float:
    total_weight = sum(float(row["weight"]) for row in rows)
    if total_weight <= 0:
        return 0.0
    return sum(float(row["weight"]) for row in rows if row[field] == category) / total_weight


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
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _ci(values: list[float]) -> tuple[float, float]:
    return (_percentile(values, 0.025), _percentile(values, 0.975))


def _cluster_ci(rows: list[dict[str, Any]], key: str, iterations: int, rng: random.Random) -> tuple[float, float]:
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_cluster[str(row[key])].append(float(row["diff"]))
    clusters = list(by_cluster)
    if not clusters:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(iterations):
        values = []
        for _ in clusters:
            values.extend(by_cluster[clusters[rng.randrange(len(clusters))]])
        means.append(_mean(values))
    return _ci(means)


def _crossed_cluster_ci(
    rows: list[dict[str, Any]],
    iterations: int,
    rng: random.Random,
) -> tuple[float, float]:
    games = sorted({str(row["game_id"]) for row in rows})
    personas = sorted({str(row["persona_pid"]) for row in rows})
    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        by_cell[(str(row["game_id"]), str(row["persona_pid"]))].append(float(row["diff"]))
    if not games or not personas:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(iterations):
        sampled_games = [games[rng.randrange(len(games))] for _ in games]
        sampled_personas = [personas[rng.randrange(len(personas))] for _ in personas]
        values = []
        for game_id in sampled_games:
            for persona_pid in sampled_personas:
                values.extend(by_cell.get((game_id, persona_pid), []))
        means.append(_mean(values))
    return _ci(means)


def _request_diff_rows(weighted_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_dist_req = _values_by_distribution_request(weighted_rows)
    request_ids = sorted(
        set(by_dist_req["candidate_uniform"])
        & set(by_dist_req["matched_probability"])
        & set(by_dist_req["matched_top1"])
    )

    output: list[dict[str, Any]] = []
    for custom_id in request_ids:
        candidate_rows = by_dist_req["candidate_uniform"][custom_id]
        metadata = candidate_rows[0]
        for distribution in ["matched_probability", "matched_top1"]:
            matched_rows = by_dist_req[distribution][custom_id]
            for field in NUMERIC_FIELDS:
                candidate_value = _numeric_value(candidate_rows, field)
                matched_value = _numeric_value(matched_rows, field)
                if candidate_value is None or matched_value is None:
                    continue
                output.append(
                    {
                        "custom_id": custom_id,
                        "game_id": metadata["game_id"],
                        "persona_pid": metadata["persona_pid"],
                        "distribution": distribution,
                        "metric_type": "numeric",
                        "field": field,
                        "category": "",
                        "diff": matched_value - candidate_value,
                    }
                )
            for field, categories in CATEGORICAL_SPECS.items():
                for category in categories:
                    output.append(
                        {
                            "custom_id": custom_id,
                            "game_id": metadata["game_id"],
                            "persona_pid": metadata["persona_pid"],
                            "distribution": distribution,
                            "metric_type": "categorical",
                            "field": field,
                            "category": category,
                            "diff": _category_share(matched_rows, field, category)
                            - _category_share(candidate_rows, field, category),
                        }
                    )
    return output


def run(args: argparse.Namespace) -> None:
    metadata_dir = args.metadata_dir.expanduser().resolve()
    request_rows = _request_diff_rows(_read_csv(metadata_dir / "pgg_demographic_weighted_rows.csv"))
    rng = random.Random(args.seed)

    output = []
    specs = sorted(
        {
            (row["distribution"], row["metric_type"], row["field"], row["category"])
            for row in request_rows
        }
    )
    for distribution, metric_type, field, category in specs:
        rows = [
            row
            for row in request_rows
            if row["distribution"] == distribution
            and row["metric_type"] == metric_type
            and row["field"] == field
            and row["category"] == category
        ]
        values = [float(row["diff"]) for row in rows]
        game_low, game_high = _cluster_ci(rows, "game_id", args.iterations, rng)
        persona_low, persona_high = _cluster_ci(rows, "persona_pid", args.iterations, rng)
        crossed_low, crossed_high = _crossed_cluster_ci(rows, args.crossed_iterations, rng)
        output.append(
            {
                "distribution": distribution,
                "metric_type": metric_type,
                "field": field,
                "category": category,
                "n": len(values),
                "mean_diff": _mean(values),
                "game_cluster_ci_low": game_low,
                "game_cluster_ci_high": game_high,
                "persona_cluster_ci_low": persona_low,
                "persona_cluster_ci_high": persona_high,
                "crossed_cluster_ci_low": crossed_low,
                "crossed_cluster_ci_high": crossed_high,
            }
        )

    _write_csv(metadata_dir / "pgg_demographic_cluster_significance.csv", output)
    print(metadata_dir / "pgg_demographic_cluster_significance.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--crossed-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
