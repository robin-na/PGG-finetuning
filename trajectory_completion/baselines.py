from __future__ import annotations

from typing import Iterable

import numpy as np

from .data import GameTrajectory, RoundRecord


def _round_and_clip(value: float, endowment: int) -> int:
    clipped = min(max(value, 0.0), float(endowment))
    return int(round(clipped))


def _normalize_contribution(value: float, game: GameTrajectory) -> int:
    endowment = game.config.endowment
    if game.config.all_or_nothing:
        return endowment if value >= (endowment / 2.0) else 0
    return _round_and_clip(value, endowment)


def _average(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(sum(values_list) / len(values_list))


def _action_units(action_map: dict[str, int]) -> int:
    return int(sum(action_map.values()))


def _history_action_units(history: list[RoundRecord], player_id: str, kind: str) -> list[int]:
    attr = "punished" if kind == "punish" else "rewarded"
    return [_action_units(getattr(round_record, attr)[player_id]) for round_record in history]


def _history_target_counts(history: list[RoundRecord], player_id: str, kind: str) -> list[int]:
    attr = "punished" if kind == "punish" else "rewarded"
    return [len(getattr(round_record, attr)[player_id]) for round_record in history]


def _rank_targets(last_round: RoundRecord, actor_id: str, kind: str) -> list[str]:
    contributions = last_round.contributions
    group_mean = _average(contributions.values())
    scored: list[tuple[float, str]] = []
    for player_id, contribution in contributions.items():
        if player_id == actor_id:
            continue
        if kind == "punish":
            score = group_mean - contribution
        else:
            score = contribution - group_mean
        scored.append((float(score), player_id))

    positive = [item for item in scored if item[0] > 0]
    ranked = positive if positive else scored
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [player_id for _, player_id in ranked]


def _predict_ranked_action(
    history: list[RoundRecord],
    actor_id: str,
    kind: str,
    enabled: bool,
    units_alpha: float = 0.7,
    targets_alpha: float = 0.7,
) -> dict[str, int]:
    if not enabled or not history:
        return {}

    units_history = _history_action_units(history, actor_id, kind)
    last_units = units_history[-1]
    mean_units = _average(units_history)
    predicted_units = int(round((units_alpha * last_units) + ((1.0 - units_alpha) * mean_units)))
    if predicted_units <= 0:
        return {}

    target_count_history = _history_target_counts(history, actor_id, kind)
    last_target_count = target_count_history[-1]
    mean_target_count = _average(target_count_history)
    predicted_target_count = int(round((targets_alpha * last_target_count) + ((1.0 - targets_alpha) * mean_target_count)))
    predicted_target_count = max(1, predicted_target_count)

    ranked_targets = _rank_targets(history[-1], actor_id, kind)
    if not ranked_targets:
        return {}

    num_targets = min(predicted_target_count, len(ranked_targets))
    selected_targets = ranked_targets[:num_targets]
    action_map = {target_id: 0 for target_id in selected_targets}
    for unit_index in range(predicted_units):
        target_id = selected_targets[unit_index % num_targets]
        action_map[target_id] += 1
    return {target_id: units for target_id, units in action_map.items() if units > 0}


def simulate_round(
    game: GameTrajectory,
    round_index: int,
    contributions: dict[str, int],
    punished: dict[str, dict[str, int]],
    rewarded: dict[str, dict[str, int]],
) -> RoundRecord:
    total_contribution = sum(contributions.values())
    base_return = game.config.mpcr * total_contribution
    round_payoffs: dict[str, float] = {}

    for player_id in game.players:
        payoff = game.config.endowment - contributions[player_id] + base_return

        if game.config.punishment_exists:
            outgoing_punish = sum(punished[player_id].values())
            payoff -= game.config.punishment_cost * outgoing_punish
            incoming_punish = sum(
                actor_actions.get(player_id, 0) for actor_actions in punished.values()
            )
            payoff -= game.config.punishment_magnitude * incoming_punish

        if game.config.reward_exists:
            outgoing_reward = sum(rewarded[player_id].values())
            payoff -= game.config.reward_cost * outgoing_reward
            incoming_reward = sum(
                actor_actions.get(player_id, 0) for actor_actions in rewarded.values()
            )
            payoff += game.config.reward_magnitude * incoming_reward

        round_payoffs[player_id] = float(np.rint(payoff))

    return RoundRecord(
        index=round_index,
        contributions=contributions,
        punished=punished,
        rewarded=rewarded,
        round_payoffs=round_payoffs,
    )


class BaselineRollout:
    def __init__(self, name: str, game: GameTrajectory, observed_rounds: list[RoundRecord]):
        self.name = name
        self.game = game
        self.observed_rounds = observed_rounds

    def predict_next_round(self, history: list[RoundRecord]) -> RoundRecord:
        raise NotImplementedError


class PersistenceRollout(BaselineRollout):
    def __init__(self, game: GameTrajectory, observed_rounds: list[RoundRecord]):
        super().__init__(name="persistence", game=game, observed_rounds=observed_rounds)

    def predict_next_round(self, history: list[RoundRecord]) -> RoundRecord:
        last_round = history[-1]
        contributions = dict(last_round.contributions)
        punished = {player_id: dict(action_map) for player_id, action_map in last_round.punished.items()}
        rewarded = {player_id: dict(action_map) for player_id, action_map in last_round.rewarded.items()}
        return simulate_round(self.game, len(history), contributions, punished, rewarded)


class EWMARollout(BaselineRollout):
    def __init__(self, game: GameTrajectory, observed_rounds: list[RoundRecord], contribution_alpha: float = 0.7):
        super().__init__(name="ewma", game=game, observed_rounds=observed_rounds)
        self.contribution_alpha = contribution_alpha

    def _predict_contribution(self, history: list[RoundRecord], player_id: str) -> int:
        contributions = [round_record.contributions[player_id] for round_record in history]
        last_value = contributions[-1]
        mean_value = _average(contributions)
        forecast = (self.contribution_alpha * last_value) + ((1.0 - self.contribution_alpha) * mean_value)
        return _normalize_contribution(forecast, self.game)

    def predict_next_round(self, history: list[RoundRecord]) -> RoundRecord:
        contributions = {
            player_id: self._predict_contribution(history, player_id)
            for player_id in self.game.players
        }
        punished = {
            player_id: _predict_ranked_action(
                history,
                player_id,
                "punish",
                enabled=self.game.config.punishment_exists,
            )
            for player_id in self.game.players
        }
        rewarded = {
            player_id: _predict_ranked_action(
                history,
                player_id,
                "reward",
                enabled=self.game.config.reward_exists,
            )
            for player_id in self.game.players
        }
        return simulate_round(self.game, len(history), contributions, punished, rewarded)


class WithinGameARRollout(BaselineRollout):
    def __init__(self, game: GameTrajectory, observed_rounds: list[RoundRecord], ridge: float = 1.0):
        super().__init__(name="within_game_ar", game=game, observed_rounds=observed_rounds)
        self.ridge = ridge
        self.coefficients = self._fit_coefficients()
        self.ewma_fallback = EWMARollout(game, observed_rounds)

    def _build_training_rows(self, player_id: str) -> tuple[np.ndarray, np.ndarray]:
        features: list[list[float]] = []
        targets: list[float] = []
        total_rounds = max(self.game.num_rounds - 1, 1)

        for current_index in range(1, len(self.observed_rounds)):
            previous_round = self.observed_rounds[current_index - 1]
            current_round = self.observed_rounds[current_index]
            features.append(
                [
                    1.0,
                    float(previous_round.contributions[player_id]),
                    _average(previous_round.contributions.values()),
                    float(previous_round.round_payoffs[player_id]),
                    float(current_index / total_rounds),
                ]
            )
            targets.append(float(current_round.contributions[player_id]))

        if not features:
            return np.empty((0, 5), dtype=float), np.empty((0,), dtype=float)
        return np.asarray(features, dtype=float), np.asarray(targets, dtype=float)

    def _fit_coefficients(self) -> dict[str, np.ndarray]:
        coefficients: dict[str, np.ndarray] = {}
        penalty = np.eye(5, dtype=float)
        penalty[0, 0] = 0.0

        for player_id in self.game.players:
            x, y = self._build_training_rows(player_id)
            if len(x) < 2:
                continue
            xtx = x.T @ x
            xty = x.T @ y
            beta = np.linalg.pinv(xtx + (self.ridge * penalty)) @ xty
            coefficients[player_id] = beta
        return coefficients

    def _predict_contribution(self, history: list[RoundRecord], player_id: str) -> int:
        coefficients = self.coefficients.get(player_id)
        if coefficients is None:
            return self.ewma_fallback._predict_contribution(history, player_id)

        previous_round = history[-1]
        total_rounds = max(self.game.num_rounds - 1, 1)
        feature_vector = np.asarray(
            [
                1.0,
                float(previous_round.contributions[player_id]),
                _average(previous_round.contributions.values()),
                float(previous_round.round_payoffs[player_id]),
                float(len(history) / total_rounds),
            ],
            dtype=float,
        )
        forecast = float(feature_vector @ coefficients)
        return _normalize_contribution(forecast, self.game)

    def predict_next_round(self, history: list[RoundRecord]) -> RoundRecord:
        contributions = {
            player_id: self._predict_contribution(history, player_id)
            for player_id in self.game.players
        }
        punished = {
            player_id: _predict_ranked_action(
                history,
                player_id,
                "punish",
                enabled=self.game.config.punishment_exists,
            )
            for player_id in self.game.players
        }
        rewarded = {
            player_id: _predict_ranked_action(
                history,
                player_id,
                "reward",
                enabled=self.game.config.reward_exists,
            )
            for player_id in self.game.players
        }
        return simulate_round(self.game, len(history), contributions, punished, rewarded)


def make_rollouts(game: GameTrajectory, observed_rounds: list[RoundRecord]) -> list[BaselineRollout]:
    return [
        PersistenceRollout(game=game, observed_rounds=observed_rounds),
        EWMARollout(game=game, observed_rounds=observed_rounds),
        WithinGameARRollout(game=game, observed_rounds=observed_rounds),
    ]
