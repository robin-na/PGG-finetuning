from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trajectory_completion.baselines import simulate_round
from trajectory_completion.data import GameConfig, GameTrajectory

from .analyze_vs_human_treatments import _relative_efficiency, _summarize_game_metrics


GAME_METADATA_COLUMNS = [
    "game_id",
    "chat_log",
    "config_num_rounds",
    "endowment",
    "all_or_nothing",
    "chat_enabled",
    "punishment_enabled",
    "punishment_cost",
    "punishment_magnitude",
    "reward_enabled",
    "reward_cost",
    "reward_magnitude",
    "treatment_name",
    "mpcr",
    "valid_start",
]


def select_shared_game_skeletons(
    human_game_df: pd.DataFrame,
    human_round_df: pd.DataFrame,
    human_actor_df: pd.DataFrame,
    shared_counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count_map = {
        str(row["treatment_name"]): int(row["shared_generated_count"])
        for row in shared_counts.to_dict(orient="records")
    }
    kept_game_frames: list[pd.DataFrame] = []
    for treatment_name, group in human_game_df.groupby("treatment_name", sort=True):
        shared_count = count_map.get(str(treatment_name), 0)
        if shared_count <= 0:
            continue
        kept_game_frames.append(group.sort_values("game_id").head(shared_count).copy())

    selected_game_df = pd.concat(kept_game_frames, ignore_index=True)
    selected_game_ids = set(selected_game_df["game_id"].astype(str))
    selected_round_df = human_round_df[human_round_df["game_id"].astype(str).isin(selected_game_ids)].copy()
    selected_actor_df = human_actor_df[human_actor_df["game_id"].astype(str).isin(selected_game_ids)].copy()
    return selected_game_df, selected_round_df, selected_actor_df


def _sample_uniform_contributions(
    players: list[str],
    *,
    endowment: int,
    all_or_nothing: bool,
    rng: np.random.Generator,
) -> dict[str, int]:
    if all_or_nothing:
        return {
            player_id: int(endowment if rng.integers(0, 2) else 0)
            for player_id in players
        }
    return {
        player_id: int(rng.integers(0, endowment + 1))
        for player_id in players
    }


def _empty_action_maps(players: list[str]) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    punished = {player_id: {} for player_id in players}
    rewarded = {player_id: {} for player_id in players}
    return punished, rewarded


def build_uniform_random_rollout_tables(
    selected_game_df: pd.DataFrame,
    selected_round_df: pd.DataFrame,
    selected_actor_df: pd.DataFrame,
    *,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    actor_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []

    game_metadata_by_id = {
        str(row["game_id"]): row
        for row in selected_game_df[GAME_METADATA_COLUMNS].to_dict(orient="records")
    }

    for game_id, game_round_df in selected_round_df.groupby("game_id", sort=True):
        game_id_str = str(game_id)
        game_meta = game_metadata_by_id[game_id_str]
        game_actor_df = selected_actor_df[selected_actor_df["game_id"].astype(str) == game_id_str]

        for round_number, round_actor_df in game_actor_df.groupby("round_number", sort=True):
            players = sorted(round_actor_df["player_id"].astype(str).unique().tolist())
            if not players:
                continue

            endowment = int(game_meta["endowment"])
            contributions = _sample_uniform_contributions(
                players,
                endowment=endowment,
                all_or_nothing=bool(game_meta["all_or_nothing"]),
                rng=rng,
            )
            punished, rewarded = _empty_action_maps(players)
            game = GameTrajectory(
                game_id=game_id_str,
                players=players,
                config=GameConfig(
                    num_rounds=int(game_meta["config_num_rounds"]),
                    endowment=endowment,
                    all_or_nothing=bool(game_meta["all_or_nothing"]),
                    punishment_exists=bool(game_meta["punishment_enabled"]),
                    reward_exists=bool(game_meta["reward_enabled"]),
                    punishment_cost=float(game_meta["punishment_cost"]),
                    punishment_magnitude=float(game_meta["punishment_magnitude"]),
                    reward_cost=float(game_meta["reward_cost"]),
                    reward_magnitude=float(game_meta["reward_magnitude"]),
                    mpcr=float(game_meta["mpcr"]),
                ),
                rounds=[],
            )
            round_record = simulate_round(
                game,
                round_index=int(round_number) - 1,
                contributions=contributions,
                punished=punished,
                rewarded=rewarded,
            )

            num_active_players = len(players)
            total_contribution = int(sum(round_record.contributions.values()))
            total_round_payoff = float(sum(round_record.round_payoffs.values()))
            contribution_rates = np.asarray(
                [value / float(endowment) for value in round_record.contributions.values()],
                dtype=float,
            )
            defect_round_coin_gen = float(num_active_players * endowment)
            max_round_coin_gen = float(game.config.mpcr * (num_active_players**2) * endowment)
            round_rows.append(
                {
                    "game_id": game_id_str,
                    "treatment_name": str(game_meta["treatment_name"]),
                    "round_number": int(round_number),
                    "num_active_players": num_active_players,
                    "total_contribution": total_contribution,
                    "total_contribution_rate": total_contribution / float(num_active_players * endowment),
                    "total_round_payoff": total_round_payoff,
                    "round_normalized_efficiency": _relative_efficiency(
                        total_round_payoff,
                        defect_round_coin_gen,
                        max_round_coin_gen,
                    ),
                    "within_round_contribution_rate_var": float(contribution_rates.var(ddof=0)),
                    "message_count": 0,
                    "has_chat": 0,
                }
            )

            defect_player_payoff = float(endowment)
            max_player_payoff = float(game.config.mpcr * num_active_players * endowment)
            for player_id in players:
                actor_rows.append(
                    {
                        "game_id": game_id_str,
                        "treatment_name": str(game_meta["treatment_name"]),
                        "round_number": int(round_number),
                        "player_id": player_id,
                        "contribution_rate": round_record.contributions[player_id] / float(endowment),
                        "round_payoff": float(round_record.round_payoffs[player_id]),
                        "round_normalized_payoff": _relative_efficiency(
                            float(round_record.round_payoffs[player_id]),
                            defect_player_payoff,
                            max_player_payoff,
                        ),
                        "punish_target_count": 0,
                        "reward_target_count": 0,
                        "has_punish": 0,
                        "has_reward": 0,
                    }
                )

    actor_df = pd.DataFrame(actor_rows)
    round_df = pd.DataFrame(round_rows)
    game_df = _summarize_game_metrics(
        actor_df,
        round_df,
        selected_game_df[GAME_METADATA_COLUMNS].copy(),
        entity_id_col="game_id",
    )
    return game_df, actor_df, round_df


def summarize_random_baseline_draws(
    draw_scores_df: pd.DataFrame,
    *,
    score_cols: list[str],
) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for score_key, metric, metric_label in (
        draw_scores_df[score_cols].drop_duplicates().sort_values(score_cols).itertuples(index=False, name=None)
    ):
        group = draw_scores_df[
            (draw_scores_df[score_cols[0]] == score_key)
            & (draw_scores_df[score_cols[1]] == metric)
            & (draw_scores_df[score_cols[2]] == metric_label)
        ]["score"].dropna().astype(float)
        if group.empty:
            score_mean = float("nan")
            score_stderr = float("nan")
        else:
            score_mean = float(group.mean())
            score_stderr = float(group.std(ddof=1) / np.sqrt(group.shape[0])) if group.shape[0] > 1 else float("nan")
        summary_rows.append(
            {
                score_cols[0]: score_key,
                score_cols[1]: metric,
                score_cols[2]: metric_label,
                "uniform_random_mean": score_mean,
                "uniform_random_stderr": score_stderr,
                "uniform_random_iters": int(group.shape[0]),
            }
        )
    return pd.DataFrame(summary_rows)
