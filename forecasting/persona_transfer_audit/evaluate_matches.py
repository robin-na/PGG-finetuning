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
    "mean_contribution",
    "mean_contribution_rate",
    "first_contribution",
    "first_contribution_rate",
    "final_contribution",
    "final_contribution_rate",
    "contribution_sd",
    "zero_contribution_rate",
    "full_contribution_rate",
    "contribution_slope",
    "message_count",
    "messages_per_round",
    "rounds_with_message_rate",
    "reward_given_units",
    "reward_given_events",
    "reward_given_round_rate",
    "reward_received_units",
    "reward_received_events",
    "reward_received_round_rate",
    "punish_given_units",
    "punish_given_events",
    "punish_given_round_rate",
    "punish_received_units",
    "punish_received_events",
    "punish_received_round_rate",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def _load_request_manifest(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            row["players"] = ast.literal_eval(row["players"]) if row.get("players") else []
            rows[row["custom_id"]] = row
        return rows


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _normalized_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    x_values = [index / (n - 1) for index in range(n)]
    x_mean = sum(x_values) / n
    y_mean = sum(values) / n
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    if denominator == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values)) / denominator


def _interaction_mode(gold_text: str) -> str:
    if "### PUNISHMENT/REWARD:" in gold_text:
        return "punishment_reward"
    if "### PUNISHMENT:" in gold_text:
        return "punishment"
    if "### REWARD:" in gold_text:
        return "reward"
    return "none"


def _empty_player_accumulator(player: str) -> dict[str, Any]:
    return {
        "player": player,
        "contributions": [],
        "message_rounds": set(),
        "message_count": 0,
        "reward_given_units": 0.0,
        "reward_given_events": 0,
        "reward_given_rounds": set(),
        "reward_received_units": 0.0,
        "reward_received_events": 0,
        "reward_received_rounds": set(),
        "punish_given_units": 0.0,
        "punish_given_events": 0,
        "punish_given_rounds": set(),
        "punish_received_units": 0.0,
        "punish_received_events": 0,
        "punish_received_rounds": set(),
    }


def _player_behavior_rows(gold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game in gold_rows:
        players = list(game["players"])
        num_rounds = len(game.get("gold_rounds", []))
        if num_rounds == 0:
            continue
        mode = _interaction_mode(str(game.get("gold_continuation_text", "")))
        player_data = {player: _empty_player_accumulator(player) for player in players}
        max_contribution = 0.0

        for round_row in game.get("gold_rounds", []):
            round_number = int(round_row["round_number"])
            contributions = list(round_row.get("contributions", []))
            if contributions:
                max_contribution = max(max_contribution, max(float(value) for value in contributions))
            for player, contribution in zip(players, contributions):
                player_data[player]["contributions"].append(float(contribution))

            for message in round_row.get("messages", []):
                speaker = str(message.get("speaker", ""))
                if speaker in player_data:
                    player_data[speaker]["message_count"] += 1
                    player_data[speaker]["message_rounds"].add(round_number)

            for source, target, raw_unit in round_row.get("interactions", []):
                if source not in player_data or target not in player_data:
                    continue
                unit = float(raw_unit)
                if mode == "reward" or (mode == "punishment_reward" and unit > 0):
                    amount = abs(unit)
                    player_data[source]["reward_given_units"] += amount
                    player_data[source]["reward_given_events"] += 1
                    player_data[source]["reward_given_rounds"].add(round_number)
                    player_data[target]["reward_received_units"] += amount
                    player_data[target]["reward_received_events"] += 1
                    player_data[target]["reward_received_rounds"].add(round_number)
                elif mode == "punishment" or (mode == "punishment_reward" and unit < 0):
                    amount = abs(unit)
                    player_data[source]["punish_given_units"] += amount
                    player_data[source]["punish_given_events"] += 1
                    player_data[source]["punish_given_rounds"].add(round_number)
                    player_data[target]["punish_received_units"] += amount
                    player_data[target]["punish_received_events"] += 1
                    player_data[target]["punish_received_rounds"].add(round_number)

        denominator = max_contribution if max_contribution > 0 else 1.0
        for player in players:
            data = player_data[player]
            contributions = data["contributions"]
            contribution_rates = [value / denominator for value in contributions]
            rows.append(
                {
                    "game_id": game["game_id"],
                    "treatment_name": game["treatment_name"],
                    "config_id": game.get("config_id", ""),
                    "player": player,
                    "player_key": f"{game['game_id']}::{player}",
                    "num_rounds": num_rounds,
                    "interaction_mode": mode,
                    "max_observed_contribution": denominator,
                    "mean_contribution": sum(contributions) / len(contributions),
                    "mean_contribution_rate": sum(contribution_rates) / len(contribution_rates),
                    "first_contribution": contributions[0],
                    "first_contribution_rate": contribution_rates[0],
                    "final_contribution": contributions[-1],
                    "final_contribution_rate": contribution_rates[-1],
                    "contribution_sd": _sample_sd(contributions),
                    "zero_contribution_rate": sum(value == 0 for value in contributions) / len(contributions),
                    "full_contribution_rate": sum(value == denominator for value in contributions)
                    / len(contributions),
                    "contribution_slope": _normalized_slope(contribution_rates),
                    "message_count": data["message_count"],
                    "messages_per_round": data["message_count"] / num_rounds,
                    "rounds_with_message_rate": len(data["message_rounds"]) / num_rounds,
                    "reward_given_units": data["reward_given_units"],
                    "reward_given_events": data["reward_given_events"],
                    "reward_given_round_rate": len(data["reward_given_rounds"]) / num_rounds,
                    "reward_received_units": data["reward_received_units"],
                    "reward_received_events": data["reward_received_events"],
                    "reward_received_round_rate": len(data["reward_received_rounds"]) / num_rounds,
                    "punish_given_units": data["punish_given_units"],
                    "punish_given_events": data["punish_given_events"],
                    "punish_given_round_rate": len(data["punish_given_rounds"]) / num_rounds,
                    "punish_received_units": data["punish_received_units"],
                    "punish_received_events": data["punish_received_events"],
                    "punish_received_round_rate": len(data["punish_received_rounds"]) / num_rounds,
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
            result[metric] = (
                sum(float(row.get(metric, 0.0)) * float(row.get(weight_column, 0.0)) for row in rows)
                / total_weight
            )
    return result


def _unweighted_mean(rows: list[dict[str, Any]]) -> dict[str, float]:
    result = {"n_players": len(rows)}
    for metric in METRIC_COLUMNS:
        result[metric] = (
            sum(float(row.get(metric, 0.0)) for row in rows) / len(rows) if rows else float("nan")
        )
    return result


def _merge_metric_differences(
    matched: dict[str, float], baseline: dict[str, float], baseline_name: str
) -> list[dict[str, Any]]:
    rows = []
    for metric in METRIC_COLUMNS:
        matched_value = matched.get(metric, float("nan"))
        baseline_value = baseline.get(metric, float("nan"))
        rows.append(
            {
                "baseline": baseline_name,
                "metric": metric,
                "matched_value": matched_value,
                "baseline_value": baseline_value,
                "difference": matched_value - baseline_value,
            }
        )
    return rows


def _entropy_from_weights(weights: dict[str, float]) -> float:
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for weight in weights.values():
        if weight <= 0:
            continue
        p = weight / total
        entropy -= p * math.log(p)
    return entropy


def evaluate(args: argparse.Namespace) -> None:
    metadata_dir = args.metadata_dir.expanduser().resolve()
    manifest = _read_json(metadata_dir / "manifest.json")
    request_manifest = _load_request_manifest(metadata_dir / "request_manifest.csv")
    parsed_matches = _read_jsonl(metadata_dir / "parsed_matches.jsonl")
    source_gold = _read_jsonl(Path(manifest["source_gold"]))
    selected_game_ids = {row["game_id"] for row in request_manifest.values()}
    selected_gold = [row for row in source_gold if row["game_id"] in selected_game_ids]

    behavior_rows = _player_behavior_rows(selected_gold)
    behavior_by_key = {row["player_key"]: row for row in behavior_rows}

    long_rows: list[dict[str, Any]] = []
    probability_errors = []
    duplicate_errors = []
    for match_row in parsed_matches:
        custom_id = match_row["custom_id"]
        manifest_row = request_manifest[custom_id]
        game_id = manifest_row["game_id"]
        top_matches = match_row.get("top_matches", [])
        if float(match_row.get("top_match_probability_sum_error", 0.0)) > args.probability_tolerance:
            probability_errors.append(custom_id)
        if match_row.get("top_match_has_duplicate_player"):
            duplicate_errors.append(custom_id)
        for rank, item in enumerate(top_matches, start=1):
            player = str(item["player"])
            player_key = f"{game_id}::{player}"
            behavior = behavior_by_key.get(player_key)
            if behavior is None:
                continue
            long_rows.append(
                {
                    "custom_id": custom_id,
                    "persona_pid": manifest_row["persona_pid"],
                    "game_id": game_id,
                    "treatment_name": manifest_row["treatment_name"],
                    "player": player,
                    "player_key": player_key,
                    "match_rank": rank,
                    "match_probability": float(item["probability"]),
                    "is_top1": rank == 1,
                    **{metric: behavior[metric] for metric in METRIC_COLUMNS},
                }
            )

    uniform_rows: list[dict[str, Any]] = []
    for manifest_row in request_manifest.values():
        players = list(manifest_row["players"])
        if not players:
            continue
        weight = 1.0 / len(players)
        for player in players:
            behavior = behavior_by_key.get(f"{manifest_row['game_id']}::{player}")
            if behavior is None:
                continue
            uniform_rows.append({"candidate_uniform_weight": weight, **behavior})

    matched_summary = _weighted_mean(long_rows, "match_probability")
    candidate_uniform_summary = _weighted_mean(uniform_rows, "candidate_uniform_weight")
    unique_player_summary = _unweighted_mean(behavior_rows)

    metric_comparison = []
    metric_comparison.extend(
        _merge_metric_differences(matched_summary, candidate_uniform_summary, "candidate_uniform")
    )
    metric_comparison.extend(
        _merge_metric_differences(matched_summary, unique_player_summary, "unique_sampled_players")
    )

    top1_counts = Counter(row["player_key"] for row in long_rows if row["is_top1"])
    probability_mass = defaultdict(float)
    for row in long_rows:
        probability_mass[row["player_key"]] += float(row["match_probability"])
    concentration = {
        "n_parsed_matches": len(parsed_matches),
        "n_long_match_rows": len(long_rows),
        "n_candidate_players": len(behavior_rows),
        "probability_error_count": len(probability_errors),
        "duplicate_top_match_count": len(duplicate_errors),
        "top1_unique_player_count": len(top1_counts),
        "top1_effective_n": math.exp(_entropy_from_weights(dict(top1_counts))),
        "probability_mass_unique_player_count": len(probability_mass),
        "probability_mass_effective_n": math.exp(_entropy_from_weights(dict(probability_mass))),
        "top1_most_common": top1_counts.most_common(10),
        "probability_mass_top10": sorted(
            probability_mass.items(), key=lambda item: item[1], reverse=True
        )[:10],
    }

    _write_csv(metadata_dir / "player_behavior_summary.csv", behavior_rows)
    _write_csv(metadata_dir / "matched_player_behavior_long.csv", long_rows)
    _write_csv(metadata_dir / "candidate_uniform_behavior_long.csv", uniform_rows)
    _write_csv(metadata_dir / "matched_vs_baseline_behavior_metrics.csv", metric_comparison)
    _write_json(
        metadata_dir / "match_behavior_eval_summary.json",
        {
            "matched_behavior": matched_summary,
            "candidate_uniform_behavior": candidate_uniform_summary,
            "unique_sampled_player_behavior": unique_player_summary,
            "concentration": concentration,
            "probability_error_custom_ids": probability_errors,
            "duplicate_top_match_custom_ids": duplicate_errors,
        },
    )
    print(json.dumps(concentration, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate persona-transfer matches against observed PGG player behavior."
    )
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--probability-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
