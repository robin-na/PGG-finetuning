from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .io_utils import RowRecord


@dataclass
class RowMetrics:
    row: RowRecord
    contribution_rate: float
    payoff: float
    punished_flag: int
    rewarded_flag: int


def _config_value(config: Dict, key: str, default: float = 0.0) -> float:
    value = config.get(key, default)
    return float(value)


def compute_row_metrics(rows: Iterable[RowRecord]) -> List[RowMetrics]:
    rows_list = list(rows)
    by_game_round: Dict[tuple, List[RowRecord]] = {}
    for row in rows_list:
        by_game_round.setdefault((row.game_id, row.round_index), []).append(row)

    metrics: List[RowMetrics] = []
    for (game_id, round_index), round_rows in by_game_round.items():
        punish_received: Dict[str, float] = {}
        reward_received: Dict[str, float] = {}
        for round_row in round_rows:
            for target, units in round_row.punished.items():
                punish_received[target] = punish_received.get(target, 0.0) + units
            for target, units in round_row.rewarded.items():
                reward_received[target] = reward_received.get(target, 0.0) + units
        for round_row in round_rows:
            env = round_row.config.environment
            endowment = _config_value(env, "CONFIG_endowment")
            multiplier = _config_value(env, "CONFIG_multiplier")
            punishment_cost = _config_value(env, "CONFIG_punishmentCost")
            punishment_magnitude = _config_value(env, "CONFIG_punishmentMagnitude")
            reward_cost = _config_value(env, "CONFIG_rewardCost")
            reward_magnitude = _config_value(env, "CONFIG_rewardMagnitude")
            punishment_exists = bool(env.get("CONFIG_punishmentExists"))
            reward_exists = bool(env.get("CONFIG_rewardExists"))

            base_payoff = (round_row.contribution * multiplier) + (
                endowment - round_row.contribution
            )

            punishment_given_units = sum(round_row.punished.values()) if punishment_exists else 0.0
            reward_given_units = sum(round_row.rewarded.values()) if reward_exists else 0.0

            punishment_received_units = (
                punish_received.get(round_row.player_id, 0.0) if punishment_exists else 0.0
            )
            reward_received_units = (
                reward_received.get(round_row.player_id, 0.0) if reward_exists else 0.0
            )

            payoff = base_payoff
            if punishment_exists:
                payoff -= punishment_given_units * punishment_cost
                payoff -= punishment_received_units * punishment_magnitude
            if reward_exists:
                payoff -= reward_given_units * reward_cost
                payoff += reward_received_units * reward_magnitude

            contribution_rate = round_row.contribution / endowment if endowment else 0.0
            metrics.append(
                RowMetrics(
                    row=round_row,
                    contribution_rate=contribution_rate,
                    payoff=payoff,
                    punished_flag=int(bool(round_row.punished)) if punishment_exists else 0,
                    rewarded_flag=int(bool(round_row.rewarded)) if reward_exists else 0,
                )
            )
    return metrics
