"""Evaluate persona-match outputs against observed chip-bargaining behavior."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


METRIC_COLUMNS = [
    "final_surplus",
    "final_welfare",
    "proposer_mean_net_surplus",
    "proposer_acceptance_rate",
    "proposer_mean_trade_ratio",
    "response_acceptance_rate",
    "response_mean_net_surplus_if_accepted",
    "received_trade_rate",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _entropy(weights: dict[str, float]) -> float:
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for weight in weights.values():
        if weight <= 0:
            continue
        p = weight / total
        value -= p * math.log(p)
    return value


def _effective_n(weights: dict[str, float]) -> float:
    return math.exp(_entropy(weights))


def _top_fraction_share(weights: dict[str, float], fraction: float) -> float:
    if not weights:
        return 0.0
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    n = max(1, math.ceil(len(weights) * fraction))
    return sum(sorted(weights.values(), reverse=True)[:n]) / total


def _value_delta(quantity_map: dict[str, int], values: dict[str, float]) -> float:
    return sum(int(quantity) * float(values[color]) for color, quantity in quantity_map.items())


def _first_player_states(target: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return target["rounds"][0]["turns"][0]["player_states_before"]


def _last_player_states(target: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return target["rounds"][-1]["turns"][-1]["player_states_after"]


def _player_behavior_rows(gold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gold_row in gold_rows:
        target = gold_row["gold_target"]
        record_id = str(gold_row["record_id"])
        treatment_name = str(gold_row["treatment_name"])
        players = [str(player_id) for player_id in target["players"]]
        values = {
            str(player_id): {str(color): float(value) for color, value in player_values.items()}
            for player_id, player_values in target["participant_chip_values"].items()
        }
        initial_states = _first_player_states(target)
        final_states = _last_player_states(target)
        accum = {
            player_id: {
                "proposer_net_surplus": [],
                "proposer_trade_ratio": [],
                "proposer_accepted": [],
                "response_accepted": [],
                "response_net_surplus_if_accepted": [],
                "selected_as_recipient": [],
            }
            for player_id in players
        }

        for round_data in target["rounds"]:
            for turn in round_data["turns"]:
                sender = str(turn["sender_id"])
                buy = {str(color): int(quantity) for color, quantity in turn["buy"].items()}
                sell = {str(color): int(quantity) for color, quantity in turn["sell"].items()}
                buy_quantity = sum(buy.values())
                sell_quantity = sum(sell.values())
                proposer_gain = _value_delta(buy, values[sender]) - _value_delta(sell, values[sender])
                accum[sender]["proposer_net_surplus"].append(proposer_gain)
                accum[sender]["proposer_trade_ratio"].append(
                    sell_quantity / buy_quantity if buy_quantity else float("nan")
                )
                accum[sender]["proposer_accepted"].append(1.0 if str(turn["status"]).upper() == "ACCEPTED" else 0.0)

                for player_id, response in turn["responses"].items():
                    player_id = str(player_id)
                    response_gain = _value_delta(sell, values[player_id]) - _value_delta(buy, values[player_id])
                    accepted = bool(response["accepted"])
                    selected = bool(response["selected_as_recipient"])
                    accum[player_id]["response_accepted"].append(1.0 if accepted else 0.0)
                    if accepted:
                        accum[player_id]["response_net_surplus_if_accepted"].append(response_gain)
                    accum[player_id]["selected_as_recipient"].append(1.0 if selected else 0.0)

        for player_id in players:
            data = accum[player_id]
            initial_welfare = float(initial_states[player_id]["payout"])
            final_welfare = float(final_states[player_id]["payout"])
            rows.append(
                {
                    "record_id": record_id,
                    "treatment_name": treatment_name,
                    "player": player_id,
                    "player_key": f"{record_id}::{player_id}",
                    "initial_welfare": initial_welfare,
                    "final_welfare": final_welfare,
                    "final_surplus": final_welfare - initial_welfare,
                    "proposer_turn_count": len(data["proposer_net_surplus"]),
                    "proposer_mean_net_surplus": _mean(data["proposer_net_surplus"]),
                    "proposer_acceptance_rate": _mean(data["proposer_accepted"]),
                    "proposer_mean_trade_ratio": _mean(data["proposer_trade_ratio"]),
                    "response_turn_count": len(data["response_accepted"]),
                    "response_acceptance_rate": _mean(data["response_accepted"]),
                    "response_mean_net_surplus_if_accepted": _mean(data["response_net_surplus_if_accepted"]),
                    "received_trade_rate": _mean(data["selected_as_recipient"]),
                }
            )
    return rows


def _weighted_mean(rows: list[dict[str, Any]], weight_column: str) -> dict[str, float]:
    total_weight = sum(float(row.get(weight_column, 0.0)) for row in rows)
    result = {"total_weight": total_weight}
    for metric in METRIC_COLUMNS:
        if total_weight <= 0:
            result[metric] = float("nan")
        else:
            values = []
            weights = []
            for row in rows:
                value = float(row.get(metric, float("nan")))
                weight = float(row.get(weight_column, 0.0))
                if not math.isnan(value):
                    values.append(value)
                    weights.append(weight)
            denom = sum(weights)
            result[metric] = sum(value * weight for value, weight in zip(values, weights)) / denom if denom else float("nan")
    return result


def _unweighted_mean(rows: list[dict[str, Any]]) -> dict[str, float]:
    result = {"n_players": len(rows)}
    for metric in METRIC_COLUMNS:
        values = [float(row.get(metric, float("nan"))) for row in rows]
        values = [value for value in values if not math.isnan(value)]
        result[metric] = _mean(values)
    return result


def evaluate(args: argparse.Namespace) -> None:
    metadata_dir = args.metadata_dir.expanduser().resolve()
    manifest = {row["custom_id"]: row for row in _read_csv(metadata_dir / "request_manifest.csv")}
    gold_rows = _read_jsonl(metadata_dir / "gold_targets.jsonl")
    parsed_long = _read_csv(metadata_dir / "parsed_matches_long.csv")

    behavior_rows = list({row["player_key"]: row for row in _player_behavior_rows(gold_rows)}.values())
    behavior_lookup = {(row["record_id"], row["player"]): row for row in behavior_rows}

    matched_rows = []
    candidate_rows = []
    probability_by_player_key: dict[str, float] = defaultdict(float)
    top1_by_player_key: Counter[str] = Counter()
    probability_by_record_player: dict[tuple[str, str], float] = defaultdict(float)
    top1_by_record_player: Counter[tuple[str, str]] = Counter()
    probability_by_persona_player_key: dict[tuple[str, str], float] = defaultdict(float)
    top1_by_persona_player_key: Counter[tuple[str, str]] = Counter()
    for match in parsed_long:
        custom_id = match["custom_id"]
        manifest_row = manifest[custom_id]
        record_id = str(manifest_row["record_id"])
        player = str(match["player"])
        behavior = behavior_lookup[(record_id, player)]
        player_key = str(behavior["player_key"])
        probability = float(match["probability"])
        rank = int(match.get("rank") or match.get("match_rank") or 0)
        persona_pid = str(manifest_row["persona_pid"])
        probability_by_player_key[player_key] += probability
        probability_by_record_player[(record_id, player)] += probability
        probability_by_persona_player_key[(persona_pid, player_key)] += probability
        if rank == 1:
            top1_by_player_key[player_key] += 1
            top1_by_record_player[(record_id, player)] += 1
            top1_by_persona_player_key[(persona_pid, player_key)] += 1
        matched_rows.append(
            {
                **match,
                "record_id": record_id,
                "player_key": player_key,
                "match_probability": probability,
                **{metric: behavior[metric] for metric in METRIC_COLUMNS},
            }
        )

    parsed_request_ids = sorted({row["custom_id"] for row in parsed_long})
    for custom_id in parsed_request_ids:
        manifest_row = manifest[custom_id]
        record_id = str(manifest_row["record_id"])
        players = ast.literal_eval(manifest_row["players"]) if manifest_row.get("players") else []
        weight = 1.0 / len(players) if players else 0.0
        for player in players:
            behavior = behavior_lookup[(record_id, str(player))]
            candidate_rows.append(
                {
                    "custom_id": custom_id,
                    "candidate_uniform_weight": weight,
                    "record_id": record_id,
                    "treatment_name": str(manifest_row["treatment_name"]),
                    "player": str(player),
                    "player_key": behavior["player_key"],
                    **{metric: behavior[metric] for metric in METRIC_COLUMNS},
                }
            )

    matched_mean = _weighted_mean(matched_rows, "match_probability")
    candidate_mean = _weighted_mean(candidate_rows, "candidate_uniform_weight")
    unique_mean = _unweighted_mean(behavior_rows)
    metric_rows = []
    for metric in METRIC_COLUMNS:
        metric_rows.append(
            {
                "baseline": "candidate_uniform",
                "metric": metric,
                "matched_value": matched_mean[metric],
                "baseline_value": candidate_mean[metric],
                "difference": matched_mean[metric] - candidate_mean[metric],
            }
        )
        metric_rows.append(
            {
                "baseline": "unique_sampled_players",
                "metric": metric,
                "matched_value": matched_mean[metric],
                "baseline_value": unique_mean[metric],
                "difference": matched_mean[metric] - unique_mean[metric],
            }
        )

    n_candidate_players = len(behavior_rows)
    top1_weights = {key: float(value) for key, value in top1_by_player_key.items()}
    probability_weights = dict(probability_by_player_key)
    coverage_rows = [
        {
            "metric": "top1_any_match_player_coverage",
            "value": len(top1_by_player_key),
            "share_of_candidate_players": len(top1_by_player_key) / n_candidate_players
            if n_candidate_players
            else 0.0,
            "description": "Observed player identities selected as top-1 at least once.",
        },
        {
            "metric": "topk_any_probability_player_coverage",
            "value": len(probability_by_player_key),
            "share_of_candidate_players": len(probability_by_player_key) / n_candidate_players
            if n_candidate_players
            else 0.0,
            "description": "Observed player identities assigned positive top-k probability at least once.",
        },
        {
            "metric": "top1_effective_n_player_identities",
            "value": _effective_n(top1_weights),
            "share_of_candidate_players": _effective_n(top1_weights) / n_candidate_players
            if n_candidate_players
            else 0.0,
            "description": "Entropy effective number of top-1 matched observed player identities.",
        },
        {
            "metric": "probability_effective_n_player_identities",
            "value": _effective_n(probability_weights),
            "share_of_candidate_players": _effective_n(probability_weights) / n_candidate_players
            if n_candidate_players
            else 0.0,
            "description": "Entropy effective number of probability-weighted observed player identities.",
        },
        {
            "metric": "top1_mass_in_top_5pct_players",
            "value": _top_fraction_share(top1_weights, 0.05),
            "share_of_candidate_players": "",
            "description": "Share of top-1 matches captured by the most selected 5% of matched player identities.",
        },
        {
            "metric": "probability_mass_in_top_5pct_players",
            "value": _top_fraction_share(probability_weights, 0.05),
            "share_of_candidate_players": "",
            "description": "Share of probability mass captured by the most selected 5% of matched player identities.",
        },
    ]
    record_ids = sorted({str(row["record_id"]) for row in manifest.values()}, key=str)
    game_rows = []
    for record_id in record_ids:
        requests_for_record = [row for row in manifest.values() if str(row["record_id"]) == record_id]
        if not requests_for_record:
            continue
        players = ast.literal_eval(requests_for_record[0]["players"]) if requests_for_record[0].get("players") else []
        game_probability = {
            player: mass
            for (candidate_record_id, player), mass in probability_by_record_player.items()
            if candidate_record_id == record_id
        }
        game_top1 = {
            player: float(count)
            for (candidate_record_id, player), count in top1_by_record_player.items()
            if candidate_record_id == record_id
        }
        n_players = len(players)
        top1_total = sum(game_top1.values())
        top1_effective_n = _effective_n(game_top1)
        game_rows.append(
            {
                "record_id": record_id,
                "treatment_name": str(requests_for_record[0]["treatment_name"]),
                "n_requests": len(requests_for_record),
                "n_players": n_players,
                "top1_total": top1_total,
                "top1_unique_players": len(game_top1),
                "top1_effective_n": top1_effective_n,
                "top1_effective_n_share_of_players": top1_effective_n / n_players if n_players else 0.0,
                "top1_top_player": max(game_top1, key=game_top1.get) if game_top1 else "",
                "top1_top_player_count": max(game_top1.values()) if game_top1 else 0.0,
                "top1_top_player_share": max(game_top1.values()) / top1_total
                if top1_total
                else 0.0,
                "top1_uniform_expected_share": 1.0 / n_players if n_players else 0.0,
                "probability_unique_players": len(game_probability),
                "probability_effective_n": _effective_n(game_probability),
                "top_probability_player": max(game_probability, key=game_probability.get)
                if game_probability
                else "",
                "top_probability_share": max(game_probability.values()) / sum(game_probability.values())
                if game_probability
                else 0.0,
            }
        )
    top1_top_shares = [
        float(row["top1_top_player_share"])
        for row in game_rows
        if float(row["top1_total"]) > 0
    ]
    top1_effective_shares = [
        float(row["top1_effective_n_share_of_players"])
        for row in game_rows
        if float(row["top1_total"]) > 0
    ]
    if top1_top_shares:
        sorted_top_shares = sorted(top1_top_shares)
        sorted_effective_shares = sorted(top1_effective_shares)
        midpoint = len(sorted_top_shares) // 2
        if len(sorted_top_shares) % 2:
            median_top_share = sorted_top_shares[midpoint]
            median_effective_share = sorted_effective_shares[midpoint]
        else:
            median_top_share = (sorted_top_shares[midpoint - 1] + sorted_top_shares[midpoint]) / 2
            median_effective_share = (
                sorted_effective_shares[midpoint - 1] + sorted_effective_shares[midpoint]
            ) / 2
    else:
        median_top_share = 0.0
        median_effective_share = 0.0
    persona_rows = []
    persona_ids = sorted({str(row["persona_pid"]) for row in manifest.values()}, key=str)
    for persona_pid in persona_ids:
        persona_probability = {
            player_key: mass
            for (candidate_pid, player_key), mass in probability_by_persona_player_key.items()
            if candidate_pid == persona_pid
        }
        persona_top1 = {
            player_key: float(count)
            for (candidate_pid, player_key), count in top1_by_persona_player_key.items()
            if candidate_pid == persona_pid
        }
        persona_rows.append(
            {
                "persona_pid": persona_pid,
                "n_requests": sum(1 for row in manifest.values() if str(row["persona_pid"]) == persona_pid),
                "top1_unique_player_identities": len(persona_top1),
                "top1_effective_n": _effective_n(persona_top1),
                "probability_unique_player_identities": len(persona_probability),
                "probability_effective_n": _effective_n(persona_probability),
                "top_probability_player_identity": max(persona_probability, key=persona_probability.get)
                if persona_probability
                else "",
                "top_probability_share": max(persona_probability.values()) / sum(persona_probability.values())
                if persona_probability
                else 0.0,
            }
        )

    _write_csv(metadata_dir / "chip_player_behavior_summary.csv", behavior_rows)
    _write_csv(metadata_dir / "chip_matched_player_behavior_long.csv", matched_rows)
    _write_csv(metadata_dir / "chip_candidate_uniform_behavior_long.csv", candidate_rows)
    _write_csv(metadata_dir / "chip_matched_vs_baseline_behavior_metrics.csv", metric_rows)
    _write_csv(metadata_dir / "chip_coverage_collapse.csv", coverage_rows)
    _write_csv(metadata_dir / "chip_game_concentration.csv", game_rows)
    _write_csv(metadata_dir / "chip_persona_concentration.csv", persona_rows)
    _write_json(
        metadata_dir / "chip_match_behavior_eval_summary.json",
        {
            "n_matched_rows": len(matched_rows),
            "n_candidate_rows": len(candidate_rows),
            "n_behavior_players": len(behavior_rows),
            "top1_unique_player_identities": len(top1_by_player_key),
            "probability_unique_player_identities": len(probability_by_player_key),
            "top1_effective_n_player_identity": _effective_n(top1_weights),
            "probability_effective_n_player_identity": _effective_n(probability_weights),
            "coverage_collapse": {row["metric"]: row for row in coverage_rows},
            "within_game_top1_uniformity": {
                "median_effective_n_share_of_players": median_effective_share,
                "median_top_player_share": median_top_share,
            },
            "candidate_uniform_metrics": {
                row["metric"]: row
                for row in metric_rows
                if row["baseline"] == "candidate_uniform"
            },
        },
    )
    print(json.dumps({"matched_rows": len(matched_rows), "candidate_rows": len(candidate_rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
