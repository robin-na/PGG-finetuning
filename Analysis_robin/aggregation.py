from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

from .io_utils import RowRecord
from .metrics import RowMetrics, compute_row_metrics


@dataclass
class SummaryRow:
    key: Tuple
    metrics: Dict[str, float]


def _config_value(config: Dict, key: str, default: float = 0.0) -> float:
    value = config.get(key, default)
    return float(value)


def summarize_by_game(rows: Iterable[RowRecord]) -> List[SummaryRow]:
    metrics_rows = compute_row_metrics(rows)
    by_game: Dict[str, List[RowMetrics]] = {}
    for item in metrics_rows:
        by_game.setdefault(item.row.game_id, []).append(item)
    summaries: List[SummaryRow] = []
    for game_id, items in by_game.items():
        env = items[0].row.config.environment
        endowment = _config_value(env, "CONFIG_endowment")
        num_rounds = _config_value(env, "CONFIG_numRounds")
        player_count = _config_value(env, "CONFIG_playerCount")
        multiplier = _config_value(env, "CONFIG_multiplier")
        p_full_coop = endowment * num_rounds * player_count * multiplier
        p_full_defect = endowment * num_rounds * player_count
        total_payoff = sum(item.payoff for item in items)
        normalized_efficiency = (
            (total_payoff - p_full_defect) / (p_full_coop - p_full_defect)
            if p_full_coop != p_full_defect
            else 0.0
        )
        summaries.append(
            SummaryRow(
                key=(game_id,),
                metrics={
                    "mean_contribution_rate": mean(item.contribution_rate for item in items),
                    "mean_payoff": mean(item.payoff for item in items),
                    "punishment_rate": mean(item.punished_flag for item in items),
                    "reward_rate": mean(item.rewarded_flag for item in items),
                    "normalized_efficiency": normalized_efficiency,
                    "total_payoff": total_payoff,
                },
            )
        )
    return summaries


def summarize_by_player(rows: Iterable[RowRecord]) -> List[SummaryRow]:
    metrics_rows = compute_row_metrics(rows)
    by_player: Dict[Tuple[str, str], List[RowMetrics]] = {}
    for item in metrics_rows:
        by_player.setdefault((item.row.game_id, item.row.player_id), []).append(item)
    summaries: List[SummaryRow] = []
    for key, items in by_player.items():
        summaries.append(
            SummaryRow(
                key=key,
                metrics={
                    "mean_contribution_rate": mean(item.contribution_rate for item in items),
                    "mean_payoff": mean(item.payoff for item in items),
                    "punishment_rate": mean(item.punished_flag for item in items),
                    "reward_rate": mean(item.rewarded_flag for item in items),
                },
            )
        )
    return summaries


def summarize_by_round(rows: Iterable[RowRecord]) -> List[SummaryRow]:
    metrics_rows = compute_row_metrics(rows)
    by_round: Dict[Tuple[str, int], List[RowMetrics]] = {}
    for item in metrics_rows:
        by_round.setdefault((item.row.game_id, item.row.round_index), []).append(item)
    summaries: List[SummaryRow] = []
    for key, items in by_round.items():
        summaries.append(
            SummaryRow(
                key=key,
                metrics={
                    "mean_contribution_rate": mean(item.contribution_rate for item in items),
                    "mean_payoff": mean(item.payoff for item in items),
                    "punishment_rate": mean(item.punished_flag for item in items),
                    "reward_rate": mean(item.rewarded_flag for item in items),
                },
            )
        )
    return summaries


def write_summary(path: Path, headers: List[str], rows: List[SummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(list(row.key) + [row.metrics.get(h, "") for h in headers[len(row.key) :]])


def write_game_summary(path: Path, summary: List[SummaryRow]) -> None:
    headers = [
        "game_id",
        "mean_contribution_rate",
        "mean_payoff",
        "punishment_rate",
        "reward_rate",
        "normalized_efficiency",
        "total_payoff",
    ]
    write_summary(path, headers, summary)


def write_player_summary(path: Path, summary: List[SummaryRow]) -> None:
    headers = [
        "game_id",
        "player_id",
        "mean_contribution_rate",
        "mean_payoff",
        "punishment_rate",
        "reward_rate",
    ]
    write_summary(path, headers, summary)


def write_round_summary(path: Path, summary: List[SummaryRow]) -> None:
    headers = [
        "game_id",
        "round_index",
        "mean_contribution_rate",
        "mean_payoff",
        "punishment_rate",
        "reward_rate",
    ]
    write_summary(path, headers, summary)
