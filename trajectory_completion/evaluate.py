from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .baselines import make_rollouts, simulate_round
from .data import GameTrajectory, RoundRecord, load_learning_wave_games


def _typed_action_map(round_record: RoundRecord, player_id: str) -> dict[tuple[str, str], int]:
    action_map: dict[tuple[str, str], int] = {}
    for target_id, units in round_record.punished[player_id].items():
        if units > 0:
            action_map[("P", target_id)] = units
    for target_id, units in round_record.rewarded[player_id].items():
        if units > 0:
            action_map[("R", target_id)] = units
    return action_map


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _precision_recall_f1(match_count: int, predicted_count: int, actual_count: int) -> tuple[float, float, float]:
    if predicted_count == 0 and actual_count == 0:
        return 1.0, 1.0, 1.0
    precision = 0.0 if predicted_count == 0 else match_count / predicted_count
    recall = 0.0 if actual_count == 0 else match_count / actual_count
    if precision == 0.0 and recall == 0.0:
        return precision, recall, 0.0
    return precision, recall, (2.0 * precision * recall) / (precision + recall)


def _relative_efficiency(total_coin_gen: float, defect_coin_gen: float, max_coin_gen: float) -> float:
    denominator = max_coin_gen - defect_coin_gen
    if denominator == 0:
        return float("nan")
    return float((total_coin_gen - defect_coin_gen) / denominator)


def _round_coin_gen_bounds(game: GameTrajectory) -> tuple[float, float]:
    empty_actions = {player_id: {} for player_id in game.players}
    defect_contributions = {player_id: 0 for player_id in game.players}
    max_contributions = {player_id: game.config.endowment for player_id in game.players}
    defect_round = simulate_round(game, round_index=0, contributions=defect_contributions, punished=empty_actions, rewarded=empty_actions)
    max_round = simulate_round(game, round_index=0, contributions=max_contributions, punished=empty_actions, rewarded=empty_actions)
    defect_coin_gen = float(sum(defect_round.round_payoffs.values()))
    max_coin_gen = float(sum(max_round.round_payoffs.values()))
    return defect_coin_gen, max_coin_gen


def _evaluate_game_rollout(
    game: GameTrajectory,
    baseline_name: str,
    k: int,
    predicted_rounds: list[RoundRecord],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    actor_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []
    actual_future_rounds = game.rounds[k:]
    defect_round_coin_gen, max_round_coin_gen = _round_coin_gen_bounds(game)
    max_player_contribution = float(game.config.endowment)
    max_total_contribution = float(game.num_players * game.config.endowment)

    for actual_round, predicted_round in zip(actual_future_rounds, predicted_rounds):
        actual_total_contribution = sum(actual_round.contributions.values())
        predicted_total_contribution = sum(predicted_round.contributions.values())
        actual_total_contribution_rate = actual_total_contribution / max_total_contribution
        predicted_total_contribution_rate = predicted_total_contribution / max_total_contribution
        actual_round_coin_gen = float(sum(actual_round.round_payoffs.values()))
        predicted_round_coin_gen = float(sum(predicted_round.round_payoffs.values()))
        actual_round_relative_efficiency = _relative_efficiency(
            actual_round_coin_gen,
            defect_round_coin_gen,
            max_round_coin_gen,
        )
        predicted_round_relative_efficiency = _relative_efficiency(
            predicted_round_coin_gen,
            defect_round_coin_gen,
            max_round_coin_gen,
        )
        round_rows.append(
            {
                "baseline": baseline_name,
                "k": k,
                "game_id": game.game_id,
                "round_index": actual_round.index,
                "actual_total_contribution": actual_total_contribution,
                "predicted_total_contribution": predicted_total_contribution,
                "total_contribution_abs_error": abs(predicted_total_contribution - actual_total_contribution),
                "actual_total_contribution_rate": actual_total_contribution_rate,
                "predicted_total_contribution_rate": predicted_total_contribution_rate,
                "total_contribution_rate_abs_error": abs(
                    predicted_total_contribution_rate - actual_total_contribution_rate
                ),
                "defect_round_coin_gen": defect_round_coin_gen,
                "max_round_coin_gen": max_round_coin_gen,
                "actual_round_coin_gen": actual_round_coin_gen,
                "predicted_round_coin_gen": predicted_round_coin_gen,
                "round_coin_gen_abs_error": abs(predicted_round_coin_gen - actual_round_coin_gen),
                "actual_round_relative_efficiency": actual_round_relative_efficiency,
                "predicted_round_relative_efficiency": predicted_round_relative_efficiency,
                "round_relative_efficiency_abs_error": abs(
                    predicted_round_relative_efficiency - actual_round_relative_efficiency
                ),
                "actual_round_normalized_efficiency": actual_round_relative_efficiency,
                "predicted_round_normalized_efficiency": predicted_round_relative_efficiency,
                "round_normalized_efficiency_abs_error": abs(
                    predicted_round_relative_efficiency - actual_round_relative_efficiency
                ),
            }
        )

        for player_id in game.players:
            actual_contribution_rate = actual_round.contributions[player_id] / max_player_contribution
            predicted_contribution_rate = predicted_round.contributions[player_id] / max_player_contribution
            actual_typed = _typed_action_map(actual_round, player_id)
            predicted_typed = _typed_action_map(predicted_round, player_id)
            actual_targets = set(actual_typed)
            predicted_targets = set(predicted_typed)
            matched_targets = actual_targets & predicted_targets

            actual_punish_targets = set(actual_round.punished[player_id])
            predicted_punish_targets = set(predicted_round.punished[player_id])
            matched_punish_targets = actual_punish_targets & predicted_punish_targets

            actual_reward_targets = set(actual_round.rewarded[player_id])
            predicted_reward_targets = set(predicted_round.rewarded[player_id])
            matched_reward_targets = actual_reward_targets & predicted_reward_targets

            overlap_unit_abs_error_sum = sum(
                abs(predicted_typed[target] - actual_typed[target]) for target in matched_targets
            )
            punish_overlap_unit_abs_error_sum = sum(
                abs(predicted_round.punished[player_id][target] - actual_round.punished[player_id][target])
                for target in matched_punish_targets
            )
            reward_overlap_unit_abs_error_sum = sum(
                abs(predicted_round.rewarded[player_id][target] - actual_round.rewarded[player_id][target])
                for target in matched_reward_targets
            )

            actor_rows.append(
                {
                    "baseline": baseline_name,
                    "k": k,
                    "game_id": game.game_id,
                    "num_rounds": game.num_rounds,
                    "num_players": game.num_players,
                    "round_index": actual_round.index,
                    "player_id": player_id,
                    "actual_contribution": actual_round.contributions[player_id],
                    "predicted_contribution": predicted_round.contributions[player_id],
                    "contribution_abs_error": abs(
                        predicted_round.contributions[player_id] - actual_round.contributions[player_id]
                    ),
                    "actual_contribution_rate": actual_contribution_rate,
                    "predicted_contribution_rate": predicted_contribution_rate,
                    "contribution_rate_abs_error": abs(predicted_contribution_rate - actual_contribution_rate),
                    "contribution_exact": int(
                        predicted_round.contributions[player_id] == actual_round.contributions[player_id]
                    ),
                    "actual_round_payoff": actual_round.round_payoffs[player_id],
                    "predicted_round_payoff": predicted_round.round_payoffs[player_id],
                    "round_payoff_abs_error": abs(
                        predicted_round.round_payoffs[player_id] - actual_round.round_payoffs[player_id]
                    ),
                    "actual_target_count": len(actual_targets),
                    "predicted_target_count": len(predicted_targets),
                    "matched_target_count": len(matched_targets),
                    "target_set_exact": int(actual_targets == predicted_targets),
                    "action_exact_match": int(actual_typed == predicted_typed),
                    "overlap_target_count": len(matched_targets),
                    "overlap_unit_abs_error_sum": overlap_unit_abs_error_sum,
                    "actual_punish_target_count": len(actual_punish_targets),
                    "predicted_punish_target_count": len(predicted_punish_targets),
                    "matched_punish_target_count": len(matched_punish_targets),
                    "punish_overlap_target_count": len(matched_punish_targets),
                    "punish_overlap_unit_abs_error_sum": punish_overlap_unit_abs_error_sum,
                    "actual_reward_target_count": len(actual_reward_targets),
                    "predicted_reward_target_count": len(predicted_reward_targets),
                    "matched_reward_target_count": len(matched_reward_targets),
                    "reward_overlap_target_count": len(matched_reward_targets),
                    "reward_overlap_unit_abs_error_sum": reward_overlap_unit_abs_error_sum,
                    "actual_has_punish": int(bool(actual_punish_targets)),
                    "predicted_has_punish": int(bool(predicted_punish_targets)),
                    "actual_has_reward": int(bool(actual_reward_targets)),
                    "predicted_has_reward": int(bool(predicted_reward_targets)),
                    "actual_punished": json.dumps(actual_round.punished[player_id], sort_keys=True),
                    "predicted_punished": json.dumps(predicted_round.punished[player_id], sort_keys=True),
                    "actual_rewarded": json.dumps(actual_round.rewarded[player_id], sort_keys=True),
                    "predicted_rewarded": json.dumps(predicted_round.rewarded[player_id], sort_keys=True),
                }
            )

    return actor_rows, round_rows


def _summarize_game(game_actor_rows: pd.DataFrame, game_round_rows: pd.DataFrame) -> dict[str, object]:
    match_count = int(game_actor_rows["matched_target_count"].sum())
    predicted_count = int(game_actor_rows["predicted_target_count"].sum())
    actual_count = int(game_actor_rows["actual_target_count"].sum())
    target_precision, target_recall, target_f1 = _precision_recall_f1(match_count, predicted_count, actual_count)

    punish_match_count = int(game_actor_rows["matched_punish_target_count"].sum())
    predicted_punish_count = int(game_actor_rows["predicted_punish_target_count"].sum())
    actual_punish_count = int(game_actor_rows["actual_punish_target_count"].sum())
    punish_precision, punish_recall, punish_f1 = _precision_recall_f1(
        punish_match_count,
        predicted_punish_count,
        actual_punish_count,
    )

    reward_match_count = int(game_actor_rows["matched_reward_target_count"].sum())
    predicted_reward_count = int(game_actor_rows["predicted_reward_target_count"].sum())
    actual_reward_count = int(game_actor_rows["actual_reward_target_count"].sum())
    reward_precision, reward_recall, reward_f1 = _precision_recall_f1(
        reward_match_count,
        predicted_reward_count,
        actual_reward_count,
    )

    cumulative_payoffs = (
        game_actor_rows.groupby(["baseline", "k", "game_id", "player_id"], as_index=False)[
            ["actual_round_payoff", "predicted_round_payoff"]
        ]
        .sum()
        .rename(
            columns={
                "actual_round_payoff": "actual_future_payoff",
                "predicted_round_payoff": "predicted_future_payoff",
            }
        )
    )
    cumulative_payoff_mae = float(
        (cumulative_payoffs["predicted_future_payoff"] - cumulative_payoffs["actual_future_payoff"]).abs().mean()
    )
    future_rounds = int(game_round_rows["round_index"].nunique())
    defect_round_coin_gen = float(game_round_rows["defect_round_coin_gen"].iloc[0])
    max_round_coin_gen = float(game_round_rows["max_round_coin_gen"].iloc[0])
    future_defect_coin_gen = defect_round_coin_gen * future_rounds
    future_max_coin_gen = max_round_coin_gen * future_rounds
    actual_future_coin_gen = float(game_round_rows["actual_round_coin_gen"].sum())
    predicted_future_coin_gen = float(game_round_rows["predicted_round_coin_gen"].sum())
    actual_future_relative_efficiency = _relative_efficiency(
        actual_future_coin_gen,
        future_defect_coin_gen,
        future_max_coin_gen,
    )
    predicted_future_relative_efficiency = _relative_efficiency(
        predicted_future_coin_gen,
        future_defect_coin_gen,
        future_max_coin_gen,
    )

    return {
        "baseline": game_actor_rows["baseline"].iloc[0],
        "k": int(game_actor_rows["k"].iloc[0]),
        "game_id": game_actor_rows["game_id"].iloc[0],
        "num_rounds": int(game_actor_rows["num_rounds"].iloc[0]),
        "num_players": int(game_actor_rows["num_players"].iloc[0]),
        "future_rounds": future_rounds,
        "actor_rows": int(len(game_actor_rows)),
        "contribution_mae": float(game_actor_rows["contribution_abs_error"].mean()),
        "contribution_rate_mae": float(game_actor_rows["contribution_rate_abs_error"].mean()),
        "contribution_exact_rate": float(game_actor_rows["contribution_exact"].mean()),
        "round_payoff_mae": float(game_actor_rows["round_payoff_abs_error"].mean()),
        "round_coin_gen_mae": float(game_round_rows["round_coin_gen_abs_error"].mean()),
        "round_relative_efficiency_mae": float(game_round_rows["round_relative_efficiency_abs_error"].mean()),
        "round_normalized_efficiency_mae": float(game_round_rows["round_normalized_efficiency_abs_error"].mean()),
        "total_contribution_mae": float(game_round_rows["total_contribution_abs_error"].mean()),
        "total_contribution_rate_mae": float(game_round_rows["total_contribution_rate_abs_error"].mean()),
        "actual_future_relative_efficiency": actual_future_relative_efficiency,
        "predicted_future_relative_efficiency": predicted_future_relative_efficiency,
        "future_relative_efficiency_abs_error": abs(
            predicted_future_relative_efficiency - actual_future_relative_efficiency
        ),
        "actual_future_normalized_efficiency": actual_future_relative_efficiency,
        "predicted_future_normalized_efficiency": predicted_future_relative_efficiency,
        "future_normalized_efficiency_abs_error": abs(
            predicted_future_relative_efficiency - actual_future_relative_efficiency
        ),
        "target_precision": target_precision,
        "target_recall": target_recall,
        "target_f1": target_f1,
        "target_set_exact_rate": float(game_actor_rows["target_set_exact"].mean()),
        "action_exact_match_rate": float(game_actor_rows["action_exact_match"].mean()),
        "punish_target_precision": punish_precision,
        "punish_target_recall": punish_recall,
        "punish_target_f1": punish_f1,
        "reward_target_precision": reward_precision,
        "reward_target_recall": reward_recall,
        "reward_target_f1": reward_f1,
        "punish_any_accuracy": float(
            (game_actor_rows["actual_has_punish"] == game_actor_rows["predicted_has_punish"]).mean()
        ),
        "reward_any_accuracy": float(
            (game_actor_rows["actual_has_reward"] == game_actor_rows["predicted_has_reward"]).mean()
        ),
        "unit_mae_on_overlap": _safe_ratio(
            float(game_actor_rows["overlap_unit_abs_error_sum"].sum()),
            float(game_actor_rows["overlap_target_count"].sum()),
        ),
        "punish_unit_mae_on_overlap": _safe_ratio(
            float(game_actor_rows["punish_overlap_unit_abs_error_sum"].sum()),
            float(game_actor_rows["punish_overlap_target_count"].sum()),
        ),
        "reward_unit_mae_on_overlap": _safe_ratio(
            float(game_actor_rows["reward_overlap_unit_abs_error_sum"].sum()),
            float(game_actor_rows["reward_overlap_target_count"].sum()),
        ),
        "cumulative_future_payoff_mae": cumulative_payoff_mae,
    }


def _summarize_overall(actor_rows: pd.DataFrame, round_rows: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    for (baseline, k), group in actor_rows.groupby(["baseline", "k"], sort=True):
        round_group = round_rows[(round_rows["baseline"] == baseline) & (round_rows["k"] == k)]
        match_count = int(group["matched_target_count"].sum())
        predicted_count = int(group["predicted_target_count"].sum())
        actual_count = int(group["actual_target_count"].sum())
        target_precision, target_recall, target_f1 = _precision_recall_f1(match_count, predicted_count, actual_count)

        punish_match_count = int(group["matched_punish_target_count"].sum())
        predicted_punish_count = int(group["predicted_punish_target_count"].sum())
        actual_punish_count = int(group["actual_punish_target_count"].sum())
        punish_precision, punish_recall, punish_f1 = _precision_recall_f1(
            punish_match_count,
            predicted_punish_count,
            actual_punish_count,
        )

        reward_match_count = int(group["matched_reward_target_count"].sum())
        predicted_reward_count = int(group["predicted_reward_target_count"].sum())
        actual_reward_count = int(group["actual_reward_target_count"].sum())
        reward_precision, reward_recall, reward_f1 = _precision_recall_f1(
            reward_match_count,
            predicted_reward_count,
            actual_reward_count,
        )

        cumulative_payoffs = (
            group.groupby(["game_id", "player_id"], as_index=False)[["actual_round_payoff", "predicted_round_payoff"]]
            .sum()
            .rename(
                columns={
                    "actual_round_payoff": "actual_future_payoff",
                    "predicted_round_payoff": "predicted_future_payoff",
                }
            )
        )
        summaries.append(
            {
                "baseline": baseline,
                "k": int(k),
                "num_games": int(group["game_id"].nunique()),
                "future_rounds": int(len(round_group)),
                "actor_rows": int(len(group)),
                "contribution_mae": float(group["contribution_abs_error"].mean()),
                "contribution_rate_mae": float(group["contribution_rate_abs_error"].mean()),
                "contribution_exact_rate": float(group["contribution_exact"].mean()),
                "round_payoff_mae": float(group["round_payoff_abs_error"].mean()),
                "round_coin_gen_mae": float(round_group["round_coin_gen_abs_error"].mean()),
                "round_relative_efficiency_mae": float(round_group["round_relative_efficiency_abs_error"].mean()),
                "round_normalized_efficiency_mae": float(round_group["round_normalized_efficiency_abs_error"].mean()),
                "total_contribution_mae": float(round_group["total_contribution_abs_error"].mean()),
                "total_contribution_rate_mae": float(round_group["total_contribution_rate_abs_error"].mean()),
                "target_precision": target_precision,
                "target_recall": target_recall,
                "target_f1": target_f1,
                "target_set_exact_rate": float(group["target_set_exact"].mean()),
                "action_exact_match_rate": float(group["action_exact_match"].mean()),
                "punish_target_precision": punish_precision,
                "punish_target_recall": punish_recall,
                "punish_target_f1": punish_f1,
                "reward_target_precision": reward_precision,
                "reward_target_recall": reward_recall,
                "reward_target_f1": reward_f1,
                "punish_any_accuracy": float((group["actual_has_punish"] == group["predicted_has_punish"]).mean()),
                "reward_any_accuracy": float((group["actual_has_reward"] == group["predicted_has_reward"]).mean()),
                "unit_mae_on_overlap": _safe_ratio(
                    float(group["overlap_unit_abs_error_sum"].sum()),
                    float(group["overlap_target_count"].sum()),
                ),
                "punish_unit_mae_on_overlap": _safe_ratio(
                    float(group["punish_overlap_unit_abs_error_sum"].sum()),
                    float(group["punish_overlap_target_count"].sum()),
                ),
                "reward_unit_mae_on_overlap": _safe_ratio(
                    float(group["reward_overlap_unit_abs_error_sum"].sum()),
                    float(group["reward_overlap_target_count"].sum()),
                ),
                "cumulative_future_payoff_mae": float(
                    (cumulative_payoffs["predicted_future_payoff"] - cumulative_payoffs["actual_future_payoff"])
                    .abs()
                    .mean()
                ),
            }
        )
    return pd.DataFrame(summaries).sort_values(["k", "baseline"]).reset_index(drop=True)


def _parse_k_values(raw_value: str) -> list[int]:
    values = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("At least one k value is required.")
    return values


def _rollout_game(game: GameTrajectory, k: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    observed_rounds = list(game.rounds[:k])
    actor_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []

    for rollout in make_rollouts(game, observed_rounds):
        history = list(observed_rounds)
        predicted_rounds: list[RoundRecord] = []
        for _ in range(k, game.num_rounds):
            predicted_round = rollout.predict_next_round(history)
            history.append(predicted_round)
            predicted_rounds.append(predicted_round)
        game_actor_rows, game_round_rows = _evaluate_game_rollout(game, rollout.name, k, predicted_rounds)
        actor_rows.extend(game_actor_rows)
        round_rows.extend(game_round_rows)

    return actor_rows, round_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate within-game trajectory completion baselines.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--k-values", type=str, default="1,3,5,8")
    parser.add_argument("--min-num-rounds-exclusive", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "learning_wave_complete_gt10_k1358",
    )
    parser.add_argument("--limit-games", type=int, default=0)
    args = parser.parse_args()

    k_values = _parse_k_values(args.k_values)
    games = load_learning_wave_games(args.repo_root, min_num_rounds_exclusive=args.min_num_rounds_exclusive)
    if args.limit_games > 0:
        games = games[: args.limit_games]

    actor_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []

    for k in k_values:
        eligible_games = [game for game in games if game.num_rounds > k]
        for game in eligible_games:
            game_actor_rows, game_round_rows = _rollout_game(game, k)
            actor_rows.extend(game_actor_rows)
            round_rows.extend(game_round_rows)

    actor_df = pd.DataFrame(actor_rows)
    round_df = pd.DataFrame(round_rows)
    if actor_df.empty or round_df.empty:
        raise RuntimeError("No evaluation rows were produced.")

    game_summary_rows = []
    for (baseline, k, game_id), group in actor_df.groupby(["baseline", "k", "game_id"], sort=True):
        game_round_group = round_df[
            (round_df["baseline"] == baseline) & (round_df["k"] == k) & (round_df["game_id"] == game_id)
        ]
        game_summary_rows.append(_summarize_game(group, game_round_group))

    game_summary_df = pd.DataFrame(game_summary_rows).sort_values(["k", "baseline", "game_id"]).reset_index(drop=True)
    overall_df = _summarize_overall(actor_df, round_df)
    future_efficiency_df = (
        game_summary_df.groupby(["baseline", "k"], as_index=False)[
            [
                "actual_future_relative_efficiency",
                "predicted_future_relative_efficiency",
                "future_relative_efficiency_abs_error",
                "actual_future_normalized_efficiency",
                "predicted_future_normalized_efficiency",
                "future_normalized_efficiency_abs_error",
            ]
        ]
        .mean()
    )
    overall_df = overall_df.merge(future_efficiency_df, on=["baseline", "k"], how="left")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    actor_df.to_csv(args.output_dir / "actor_level_predictions.csv", index=False)
    round_df.to_csv(args.output_dir / "round_level_predictions.csv", index=False)
    game_summary_df.to_csv(args.output_dir / "game_summary.csv", index=False)
    overall_df.to_csv(args.output_dir / "overall_summary.csv", index=False)

    manifest = {
        "repo_root": str(args.repo_root),
        "k_values": k_values,
        "min_num_rounds_exclusive": args.min_num_rounds_exclusive,
        "games_loaded": len(games),
        "games_by_k": {
            str(k): int(sum(game.num_rounds > k for game in games))
            for k in k_values
        },
        "baselines": ["persistence", "ewma", "within_game_ar"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote outputs to {args.output_dir}")
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
