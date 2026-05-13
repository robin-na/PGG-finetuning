"""Request-conditional global identity-collapse test.

For each successful persona-game request, the null draws one top-1 player
uniformly from the players shown in that request. This preserves the exact
request structure, number of personas per game, and number of players per game.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _identity_prefix(row: dict[str, str]) -> str:
    game_id = row.get("game_id") or row.get("record_id")
    if not game_id:
        raise ValueError(f"Manifest row missing game_id/record_id: {row}")
    return str(game_id)


def _load_candidates(metadata_dir: Path, successful_request_ids: set[str]) -> tuple[list[list[str]], set[str]]:
    manifest_rows = _read_csv(metadata_dir / "request_manifest.csv")
    candidates: list[list[str]] = []
    all_identities: set[str] = set()
    for row in manifest_rows:
        custom_id = row["custom_id"]
        if custom_id not in successful_request_ids:
            continue
        players = [str(player) for player in ast.literal_eval(row["players"])]
        prefix = _identity_prefix(row)
        candidate_keys = [f"{prefix}::{player}" for player in players]
        candidates.append(candidate_keys)
        all_identities.update(candidate_keys)
    return candidates, all_identities


def _load_observed_top1(metadata_dir: Path) -> tuple[Counter[str], set[str]]:
    rows = _read_csv(metadata_dir / "parsed_matches_long.csv")
    counts: Counter[str] = Counter()
    successful_request_ids: set[str] = set()
    for row in rows:
        custom_id = row["custom_id"]
        successful_request_ids.add(custom_id)
        is_top1 = str(row.get("is_top1", "")).lower() in {"true", "1"}
        is_rank1 = str(row.get("match_rank") or row.get("rank") or "") == "1"
        if not (is_top1 or is_rank1):
            continue
        prefix = row.get("game_id") or row.get("record_id")
        if not prefix:
            raise ValueError(f"Parsed row missing game_id/record_id: {row}")
        counts[f"{prefix}::{row['player']}"] += 1
    return counts, successful_request_ids


def _entropy_effective_n(counts: Counter[str], total: int) -> float:
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return math.exp(entropy)


def _simpson_effective_n(counts: Counter[str], total: int) -> float:
    if total <= 0:
        return 0.0
    hhi = sum((count / total) ** 2 for count in counts.values())
    return 1.0 / hhi if hhi > 0 else 0.0


def _gini(counts: Counter[str], all_identities: set[str]) -> float:
    values = sorted(float(counts.get(identity, 0)) for identity in all_identities)
    n = len(values)
    total = sum(values)
    if n == 0 or total <= 0:
        return 0.0
    weighted_sum = sum((index + 1) * value for index, value in enumerate(values))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def _top_fraction_share(counts: Counter[str], all_identities: set[str], fraction: float) -> float:
    total = sum(counts.values())
    if total <= 0 or not all_identities:
        return 0.0
    n = max(1, math.ceil(len(all_identities) * fraction))
    values = sorted((counts.get(identity, 0) for identity in all_identities), reverse=True)
    return sum(values[:n]) / total


def _metrics(counts: Counter[str], all_identities: set[str]) -> dict[str, float]:
    total = sum(counts.values())
    selected = sum(1 for identity in all_identities if counts.get(identity, 0) > 0)
    hhi = sum((counts.get(identity, 0) / total) ** 2 for identity in all_identities) if total else 0.0
    return {
        "total_top1": float(total),
        "candidate_identities": float(len(all_identities)),
        "selected_identities": float(selected),
        "selected_identity_share": selected / len(all_identities) if all_identities else 0.0,
        "never_selected_identities": float(len(all_identities) - selected),
        "never_selected_identity_share": (len(all_identities) - selected) / len(all_identities)
        if all_identities
        else 0.0,
        "entropy_effective_n": _entropy_effective_n(counts, total),
        "entropy_effective_n_share": _entropy_effective_n(counts, total) / len(all_identities)
        if all_identities
        else 0.0,
        "simpson_effective_n": _simpson_effective_n(counts, total),
        "simpson_effective_n_share": _simpson_effective_n(counts, total) / len(all_identities)
        if all_identities
        else 0.0,
        "hhi": hhi,
        "gini": _gini(counts, all_identities),
        "top_1pct_share": _top_fraction_share(counts, all_identities, 0.01),
        "top_5pct_share": _top_fraction_share(counts, all_identities, 0.05),
        "top_10pct_share": _top_fraction_share(counts, all_identities, 0.10),
    }


def _percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    index = (len(values) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def _tail_direction(metric: str) -> str:
    if metric in {
        "selected_identities",
        "selected_identity_share",
        "entropy_effective_n",
        "entropy_effective_n_share",
        "simpson_effective_n",
        "simpson_effective_n_share",
    }:
        return "lower_is_more_collapsed"
    return "higher_is_more_collapsed"


def run(args: argparse.Namespace) -> None:
    metadata_dir = args.metadata_dir.expanduser().resolve()
    rng = random.Random(args.seed)

    observed_counts, successful_request_ids = _load_observed_top1(metadata_dir)
    request_candidates, all_identities = _load_candidates(metadata_dir, successful_request_ids)
    observed = _metrics(observed_counts, all_identities)

    simulation_rows: list[dict[str, Any]] = []
    null_values: dict[str, list[float]] = {metric: [] for metric in observed}
    for iteration in range(args.iterations):
        counts: Counter[str] = Counter()
        for candidates in request_candidates:
            counts[rng.choice(candidates)] += 1
        metrics = _metrics(counts, all_identities)
        simulation_rows.append({"iteration": iteration, **metrics})
        for metric, value in metrics.items():
            null_values[metric].append(value)

    summary_rows: list[dict[str, Any]] = []
    for metric, observed_value in observed.items():
        values = null_values[metric]
        direction = _tail_direction(metric)
        if direction == "lower_is_more_collapsed":
            p_value = (sum(value <= observed_value for value in values) + 1) / (len(values) + 1)
        else:
            p_value = (sum(value >= observed_value for value in values) + 1) / (len(values) + 1)
        null_mean = sum(values) / len(values)
        null_sd = math.sqrt(sum((value - null_mean) ** 2 for value in values) / len(values))
        summary_rows.append(
            {
                "metric": metric,
                "direction": direction,
                "observed": observed_value,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "null_ci_low": _percentile(values, 0.025),
                "null_ci_high": _percentile(values, 0.975),
                "observed_minus_null_mean": observed_value - null_mean,
                "z_score": (observed_value - null_mean) / null_sd if null_sd > 0 else float("nan"),
                "collapse_tail_p": p_value,
            }
        )

    prefix = args.output_prefix or "global_identity_collapse"
    _write_csv(metadata_dir / f"{prefix}_summary.csv", summary_rows)
    if args.save_simulations:
        _write_csv(metadata_dir / f"{prefix}_simulations.csv", simulation_rows)
    _write_json(
        metadata_dir / f"{prefix}_summary.json",
        {
            "metadata_dir": str(metadata_dir),
            "iterations": args.iterations,
            "seed": args.seed,
            "successful_requests": len(request_candidates),
            "candidate_identities": len(all_identities),
            "observed": observed,
            "summary": {row["metric"]: row for row in summary_rows},
        },
    )

    key_metrics = [
        "selected_identity_share",
        "entropy_effective_n_share",
        "simpson_effective_n_share",
        "never_selected_identity_share",
        "top_5pct_share",
        "gini",
    ]
    print(f"Run: {metadata_dir.name}")
    print(f"Successful requests: {len(request_candidates)}")
    print(f"Candidate identities: {len(all_identities)}")
    for metric in key_metrics:
        row = next(item for item in summary_rows if item["metric"] == metric)
        print(
            f"{metric}: observed={row['observed']:.4f}, "
            f"null_mean={row['null_mean']:.4f}, "
            f"95% null=[{row['null_ci_low']:.4f}, {row['null_ci_high']:.4f}], "
            f"p={row['collapse_tail_p']:.5f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run request-conditional global identity-collapse test.")
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-prefix", default="global_identity_collapse")
    parser.add_argument("--save-simulations", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
