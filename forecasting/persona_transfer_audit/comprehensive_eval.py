from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from forecasting.persona_transfer_audit.evaluate_matches import (
    METRIC_COLUMNS,
    _player_behavior_rows,
    _read_json,
    _read_jsonl,
    _write_csv,
    _write_json,
)


def _load_request_manifest(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            row["players"] = ast.literal_eval(row["players"]) if row.get("players") else []
            rows[row["custom_id"]] = row
        return rows


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


def _weighted_stats(rows: list[dict[str, Any]], metric: str, weight_col: str) -> dict[str, float]:
    total_weight = sum(float(row.get(weight_col, 0.0)) for row in rows)
    if total_weight <= 0:
        return {"mean": float("nan"), "sd": float("nan"), "total_weight": 0.0}
    mean = sum(float(row[metric]) * float(row.get(weight_col, 0.0)) for row in rows) / total_weight
    variance = (
        sum(float(row.get(weight_col, 0.0)) * (float(row[metric]) - mean) ** 2 for row in rows)
        / total_weight
    )
    return {"mean": mean, "sd": math.sqrt(max(variance, 0.0)), "total_weight": total_weight}


def _cohens_like_diff(matched_mean: float, baseline_mean: float, baseline_sd: float) -> float:
    if baseline_sd == 0 or math.isnan(baseline_sd):
        return float("nan")
    return (matched_mean - baseline_mean) / baseline_sd


def _top_rows(rows: list[dict[str, Any]], key: str, n: int = 12) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row.get(key, 0.0)), reverse=True)[:n]


def _top_fraction_share(weights: dict[str, float], fraction: float) -> float:
    if not weights:
        return 0.0
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    n = max(1, math.ceil(len(weights) * fraction))
    return sum(sorted(weights.values(), reverse=True)[:n]) / total


def comprehensive_eval(args: argparse.Namespace) -> None:
    metadata_dir = args.metadata_dir.expanduser().resolve()
    manifest = _read_json(metadata_dir / "manifest.json")
    request_manifest = _load_request_manifest(metadata_dir / "request_manifest.csv")
    parsed_matches = _read_jsonl(metadata_dir / "parsed_matches.jsonl")
    source_gold = _read_jsonl(Path(manifest["source_gold"]))
    selected_game_ids = {row["game_id"] for row in request_manifest.values()}
    selected_gold = [row for row in source_gold if row["game_id"] in selected_game_ids]

    behavior_rows = _player_behavior_rows(selected_gold)
    behavior_by_key = {row["player_key"]: row for row in behavior_rows}

    probability_by_player_key: dict[str, float] = defaultdict(float)
    top1_by_player_key: Counter[str] = Counter()
    probability_by_avatar: dict[str, float] = defaultdict(float)
    top1_by_avatar: Counter[str] = Counter()
    probability_by_game_player: dict[tuple[str, str], float] = defaultdict(float)
    top1_by_game_player: Counter[tuple[str, str]] = Counter()
    probability_by_persona_player_key: dict[tuple[str, str], float] = defaultdict(float)
    top1_by_persona_player_key: Counter[tuple[str, str]] = Counter()

    long_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for manifest_row in request_manifest.values():
        players = list(manifest_row["players"])
        if not players:
            continue
        uniform_weight = 1.0 / len(players)
        for player in players:
            behavior = behavior_by_key.get(f"{manifest_row['game_id']}::{player}")
            if behavior is None:
                continue
            candidate_rows.append(
                {
                    "custom_id": manifest_row["custom_id"],
                    "persona_pid": manifest_row["persona_pid"],
                    "candidate_uniform_weight": uniform_weight,
                    **behavior,
                }
            )

    for match in parsed_matches:
        custom_id = match["custom_id"]
        manifest_row = request_manifest[custom_id]
        game_id = manifest_row["game_id"]
        persona_pid = manifest_row["persona_pid"]
        for rank, item in enumerate(match.get("top_matches", []), start=1):
            player = str(item["player"])
            probability = float(item["probability"])
            player_key = f"{game_id}::{player}"
            behavior = behavior_by_key.get(player_key)
            if behavior is None:
                continue
            probability_by_player_key[player_key] += probability
            probability_by_avatar[player] += probability
            probability_by_game_player[(game_id, player)] += probability
            probability_by_persona_player_key[(persona_pid, player_key)] += probability
            if rank == 1:
                top1_by_player_key[player_key] += 1
                top1_by_avatar[player] += 1
                top1_by_game_player[(game_id, player)] += 1
                top1_by_persona_player_key[(persona_pid, player_key)] += 1
            long_rows.append(
                {
                    "custom_id": custom_id,
                    "persona_pid": persona_pid,
                    "game_id": game_id,
                    "treatment_name": manifest_row["treatment_name"],
                    "player": player,
                    "player_key": player_key,
                    "rank": rank,
                    "probability": probability,
                    "is_top1": rank == 1,
                    **{metric: behavior[metric] for metric in METRIC_COLUMNS},
                }
            )

    player_distribution_rows = []
    total_probability = sum(probability_by_player_key.values())
    total_top1 = sum(top1_by_player_key.values())
    for behavior in behavior_rows:
        key = behavior["player_key"]
        player_distribution_rows.append(
            {
                **behavior,
                "top1_count": top1_by_player_key.get(key, 0),
                "top1_share": top1_by_player_key.get(key, 0) / total_top1 if total_top1 else 0.0,
                "probability_mass": probability_by_player_key.get(key, 0.0),
                "probability_share": probability_by_player_key.get(key, 0.0) / total_probability
                if total_probability
                else 0.0,
            }
        )

    avatar_rows = []
    for avatar in sorted(set(top1_by_avatar) | set(probability_by_avatar)):
        avatar_rows.append(
            {
                "avatar": avatar,
                "top1_count": top1_by_avatar.get(avatar, 0),
                "top1_share": top1_by_avatar.get(avatar, 0) / total_top1 if total_top1 else 0.0,
                "probability_mass": probability_by_avatar.get(avatar, 0.0),
                "probability_share": probability_by_avatar.get(avatar, 0.0) / total_probability
                if total_probability
                else 0.0,
            }
        )

    game_rows = []
    for game_id in sorted(selected_game_ids):
        game_probability = {
            player: mass
            for (candidate_game_id, player), mass in probability_by_game_player.items()
            if candidate_game_id == game_id
        }
        game_top1 = {
            player: float(count)
            for (candidate_game_id, player), count in top1_by_game_player.items()
            if candidate_game_id == game_id
        }
        requests_for_game = [
            row for row in request_manifest.values() if row["game_id"] == game_id
        ]
        treatment_name = requests_for_game[0]["treatment_name"] if requests_for_game else ""
        n_players = len(requests_for_game[0]["players"]) if requests_for_game else 0
        top1_total = sum(game_top1.values())
        top1_effective_n = _effective_n(game_top1)
        game_rows.append(
            {
                "game_id": game_id,
                "treatment_name": treatment_name,
                "n_requests": len(requests_for_game),
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

    persona_rows = []
    persona_ids = sorted({row["persona_pid"] for row in request_manifest.values()}, key=str)
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
                "n_requests": sum(1 for row in request_manifest.values() if row["persona_pid"] == persona_pid),
                "top1_unique_player_identities": len(persona_top1),
                "top1_effective_n": _effective_n(persona_top1),
                "probability_unique_player_identities": len(persona_probability),
                "probability_effective_n": _effective_n(persona_probability),
                "top_probability_player_identity": max(persona_probability, key=persona_probability.get)
                if persona_probability
                else "",
                "top_probability_share": max(persona_probability.values())
                / sum(persona_probability.values())
                if persona_probability
                else 0.0,
            }
        )

    top1_weights = {key: float(value) for key, value in top1_by_player_key.items()}
    probability_weights = dict(probability_by_player_key)
    n_candidate_players = len(behavior_rows)
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
            "metric": "top1_mass_in_top_1pct_players",
            "value": _top_fraction_share(top1_weights, 0.01),
            "share_of_candidate_players": "",
            "description": "Share of top-1 matches captured by the most selected 1% of matched player identities.",
        },
        {
            "metric": "probability_mass_in_top_1pct_players",
            "value": _top_fraction_share(probability_weights, 0.01),
            "share_of_candidate_players": "",
            "description": "Share of probability mass captured by the most selected 1% of matched player identities.",
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
    game_top1_effective_shares = [
        float(row["top1_effective_n_share_of_players"])
        for row in game_rows
        if float(row["top1_total"]) > 0
    ]
    game_top1_top_shares = [
        float(row["top1_top_player_share"])
        for row in game_rows
        if float(row["top1_total"]) > 0
    ]
    if game_top1_effective_shares:
        sorted_effective_shares = sorted(game_top1_effective_shares)
        sorted_top_shares = sorted(game_top1_top_shares)
        midpoint = len(sorted_effective_shares) // 2
        if len(sorted_effective_shares) % 2:
            median_effective_share = sorted_effective_shares[midpoint]
            median_top_share = sorted_top_shares[midpoint]
        else:
            median_effective_share = (
                sorted_effective_shares[midpoint - 1] + sorted_effective_shares[midpoint]
            ) / 2
            median_top_share = (sorted_top_shares[midpoint - 1] + sorted_top_shares[midpoint]) / 2
    else:
        median_effective_share = 0.0
        median_top_share = 0.0

    metric_rows = []
    for metric in METRIC_COLUMNS:
        matched = _weighted_stats(long_rows, metric, "probability")
        uniform = _weighted_stats(candidate_rows, metric, "candidate_uniform_weight")
        unique_mean = (
            sum(float(row[metric]) for row in behavior_rows) / len(behavior_rows)
            if behavior_rows
            else float("nan")
        )
        unique_sd = math.sqrt(
            sum((float(row[metric]) - unique_mean) ** 2 for row in behavior_rows) / len(behavior_rows)
        ) if behavior_rows else float("nan")
        metric_rows.append(
            {
                "metric": metric,
                "matched_mean": matched["mean"],
                "candidate_uniform_mean": uniform["mean"],
                "candidate_uniform_sd": uniform["sd"],
                "matched_minus_candidate_uniform": matched["mean"] - uniform["mean"],
                "candidate_uniform_standardized_difference": _cohens_like_diff(
                    matched["mean"], uniform["mean"], uniform["sd"]
                ),
                "unique_player_mean": unique_mean,
                "unique_player_sd": unique_sd,
                "matched_minus_unique_player": matched["mean"] - unique_mean,
                "unique_player_standardized_difference": _cohens_like_diff(
                    matched["mean"], unique_mean, unique_sd
                ),
            }
        )

    top_player_rows = _top_rows(player_distribution_rows, "probability_mass", n=20)
    key_metrics = [
        "mean_contribution_rate",
        "full_contribution_rate",
        "zero_contribution_rate",
        "contribution_sd",
        "messages_per_round",
        "reward_given_round_rate",
        "punish_given_round_rate",
        "punish_received_round_rate",
    ]
    metric_lookup = {row["metric"]: row for row in metric_rows}

    report_lines = [
        "# Persona Transfer Audit: Comprehensive Pilot Evaluation",
        "",
        f"Run: `{metadata_dir.name}`",
        f"Parsed requests: {len(parsed_matches)}",
        f"Candidate observed players: {len(behavior_rows)}",
        f"Returned top-k rows: {len(long_rows)}",
        "",
        "## Concentration",
        "",
        f"Top-1 player coverage: {len(top1_by_player_key)} / {n_candidate_players} ({len(top1_by_player_key) / n_candidate_players:.1%})",
        f"Top-k probability player coverage: {len(probability_by_player_key)} / {n_candidate_players} ({len(probability_by_player_key) / n_candidate_players:.1%})",
        f"Top-1 effective number of observed player identities: {_effective_n(top1_weights):.2f}",
        f"Probability-mass effective number of observed player identities: {_effective_n(probability_weights):.2f}",
        f"Top 5% of matched identities capture {_top_fraction_share(probability_weights, 0.05):.1%} of probability mass.",
        f"Median within-game top-1 effective N / players: {median_effective_share:.1%}",
        f"Median within-game top player share across personas: {median_top_share:.1%}",
        "",
        "## Avatar-Label Diagnostic",
        "",
        "Avatar labels are reused across games and are not stable person identities. This table is only a label/order diagnostic.",
        "",
        "| Avatar | Top-1 share | Probability share |",
        "|---|---:|---:|",
    ]
    for row in sorted(avatar_rows, key=lambda item: item["probability_share"], reverse=True):
        report_lines.append(
            f"| {row['avatar']} | {row['top1_share']:.3f} | {row['probability_share']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## Matched Behavior Skew",
            "",
            "| Metric | Matched | Candidate-uniform | Difference | Std. diff |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric in key_metrics:
        row = metric_lookup[metric]
        report_lines.append(
            f"| {metric} | {row['matched_mean']:.3f} | {row['candidate_uniform_mean']:.3f} | "
            f"{row['matched_minus_candidate_uniform']:.3f} | {row['candidate_uniform_standardized_difference']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## Most-Matched Observed Players",
            "",
            "| Player key | Treatment | Probability mass | Top-1 count | Mean contribution rate | Full contribution rate |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in top_player_rows[:10]:
        report_lines.append(
            f"| {row['player_key']} | {row['treatment_name']} | {row['probability_mass']:.2f} | "
            f"{row['top1_count']} | {row['mean_contribution_rate']:.3f} | {row['full_contribution_rate']:.3f} |"
        )
    report_lines.append("")

    _write_csv(metadata_dir / "comprehensive_player_match_distribution.csv", player_distribution_rows)
    _write_csv(metadata_dir / "comprehensive_avatar_match_distribution.csv", avatar_rows)
    _write_csv(metadata_dir / "comprehensive_coverage_collapse.csv", coverage_rows)
    _write_csv(metadata_dir / "comprehensive_game_concentration.csv", game_rows)
    _write_csv(metadata_dir / "comprehensive_persona_concentration.csv", persona_rows)
    _write_csv(metadata_dir / "comprehensive_behavior_metric_comparison.csv", metric_rows)
    _write_json(
        metadata_dir / "comprehensive_eval_summary.json",
        {
            "n_parsed_matches": len(parsed_matches),
            "n_candidate_players": len(behavior_rows),
            "n_long_rows": len(long_rows),
            "top1_unique_player_identities": len(top1_by_player_key),
            "probability_unique_player_identities": len(probability_by_player_key),
            "top1_effective_n_player_identity": _effective_n(top1_weights),
            "probability_effective_n_player_identity": _effective_n(probability_weights),
            "coverage_collapse": {row["metric"]: row for row in coverage_rows},
            "within_game_top1_uniformity": {
                "median_effective_n_share_of_players": median_effective_share,
                "median_top_player_share": median_top_share,
            },
            "avatar_distribution": sorted(
                avatar_rows, key=lambda row: row["probability_share"], reverse=True
            ),
            "key_metric_comparison": {metric: metric_lookup[metric] for metric in key_metrics},
        },
    )
    (metadata_dir / "comprehensive_eval_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    print("\n".join(report_lines[:35]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation for persona transfer audit.")
    parser.add_argument("--metadata-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    comprehensive_eval(parse_args())


if __name__ == "__main__":
    main()
