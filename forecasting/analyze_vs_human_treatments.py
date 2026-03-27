from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trajectory_completion.data import _parse_action_dict, load_wave_games
from trajectory_completion.evaluate import _relative_efficiency

from .build_batch_inputs import _index_chat_log, _load_avatar_map, _load_game_rows
from .evaluate_outputs import (
    _build_avatar_mapping,
    _load_request_manifest,
    _parsed_rounds_to_round_records,
)
from .parse_outputs import _read_jsonl


CORE_METRICS = [
    "mean_total_contribution_rate",
    "final_total_contribution_rate",
    "total_contribution_rate_sd",
    "mean_round_normalized_efficiency",
    "final_round_normalized_efficiency",
    "total_contribution_rate_rmse_to_treatment_mean",
    "round_normalized_efficiency_rmse_to_treatment_mean",
    "punish_actor_round_rate",
    "reward_actor_round_rate",
    "messages_per_round",
    "rounds_with_chat_rate",
]


def _safe_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def _load_validation_metadata(repo_root: Path, treatment_names: set[str]) -> pd.DataFrame:
    cols = [
        "gameId",
        "CONFIG_treatmentName",
        "CONFIG_numRounds",
        "CONFIG_endowment",
        "CONFIG_MPCR",
        "CONFIG_chat",
        "CONFIG_punishmentExists",
        "CONFIG_rewardExists",
        "valid_number_of_starting_players",
        "chat_log",
    ]
    metadata = (
        pd.read_csv(repo_root / "data/processed_data/df_analysis_val.csv", usecols=cols)
        .rename(
            columns={
                "gameId": "game_id",
                "CONFIG_treatmentName": "treatment_name",
                "CONFIG_numRounds": "config_num_rounds",
                "CONFIG_endowment": "endowment",
                "CONFIG_MPCR": "mpcr",
                "CONFIG_chat": "chat_enabled",
                "CONFIG_punishmentExists": "punishment_enabled",
                "CONFIG_rewardExists": "reward_enabled",
                "valid_number_of_starting_players": "valid_start",
            }
        )
        .drop_duplicates("game_id")
    )
    metadata = metadata[metadata["valid_start"].astype(bool)].copy()
    metadata = metadata[metadata["treatment_name"].isin(treatment_names)].copy()
    metadata["chat_enabled"] = metadata["chat_enabled"].astype(bool)
    metadata["punishment_enabled"] = metadata["punishment_enabled"].astype(bool)
    metadata["reward_enabled"] = metadata["reward_enabled"].astype(bool)
    return metadata.reset_index(drop=True)


def _round_summary_from_round_records(
    *,
    entity_id: str,
    treatment_name: str,
    rounds: list[Any],
    message_count_by_round: dict[int, int],
    endowment: int,
    mpcr: float,
    num_players: int,
    entity_kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actor_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []

    defect_round_coin_gen = float(num_players * endowment)
    max_round_coin_gen = float(mpcr * (num_players**2) * endowment)

    for round_record in rounds:
        round_number = int(round_record.index) + 1
        total_contribution = int(sum(round_record.contributions.values()))
        total_contribution_rate = total_contribution / float(num_players * endowment)
        total_round_payoff = float(sum(round_record.round_payoffs.values()))
        round_normalized_efficiency = _relative_efficiency(
            total_round_payoff,
            defect_round_coin_gen,
            max_round_coin_gen,
        )
        round_rows.append(
            {
                "entity_kind": entity_kind,
                "entity_id": entity_id,
                "treatment_name": treatment_name,
                "round_number": round_number,
                "num_active_players": num_players,
                "total_contribution": total_contribution,
                "total_contribution_rate": total_contribution_rate,
                "total_round_payoff": total_round_payoff,
                "round_normalized_efficiency": round_normalized_efficiency,
                "message_count": int(message_count_by_round.get(round_number, 0)),
                "has_chat": int(message_count_by_round.get(round_number, 0) > 0),
            }
        )
        for player_id in round_record.contributions:
            punish_target_count = len(round_record.punished[player_id])
            reward_target_count = len(round_record.rewarded[player_id])
            actor_rows.append(
                {
                    "entity_kind": entity_kind,
                    "entity_id": entity_id,
                    "treatment_name": treatment_name,
                    "round_number": round_number,
                    "player_id": player_id,
                    "contribution_rate": round_record.contributions[player_id] / float(endowment),
                    "punish_target_count": punish_target_count,
                    "reward_target_count": reward_target_count,
                    "has_punish": int(punish_target_count > 0),
                    "has_reward": int(reward_target_count > 0),
                }
            )

    return pd.DataFrame(actor_rows), pd.DataFrame(round_rows)


def _message_count_by_round_from_parsed_rounds(parsed_rounds: list[dict[str, Any]]) -> dict[int, int]:
    return {
        int(round_payload["round_number"]): len(round_payload.get("messages", []))
        for round_payload in parsed_rounds
    }


def _summarize_game_metrics(
    actor_df: pd.DataFrame,
    round_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    *,
    entity_id_col: str,
) -> pd.DataFrame:
    actor_summary = (
        actor_df.groupby(entity_id_col, as_index=False)
        .agg(
            mean_contribution_rate=("contribution_rate", "mean"),
            punish_actor_round_rate=("has_punish", "mean"),
            reward_actor_round_rate=("has_reward", "mean"),
            punish_targets_per_actor_round=("punish_target_count", "mean"),
            reward_targets_per_actor_round=("reward_target_count", "mean"),
            actor_rounds=("player_id", "count"),
        )
    )
    round_summary = (
        round_df.groupby(entity_id_col, as_index=False)
        .agg(
            mean_total_contribution_rate=("total_contribution_rate", "mean"),
            total_contribution_rate_sd=("total_contribution_rate", lambda s: float(s.std(ddof=0))),
            final_total_contribution_rate=("total_contribution_rate", lambda s: float(s.iloc[-1])),
            mean_round_normalized_efficiency=("round_normalized_efficiency", "mean"),
            final_round_normalized_efficiency=("round_normalized_efficiency", lambda s: float(s.iloc[-1])),
            messages_per_round=("message_count", "mean"),
            rounds_with_chat_rate=("has_chat", "mean"),
            observed_rounds=("round_number", "count"),
        )
    )
    return metadata_df.merge(actor_summary, on=entity_id_col, how="left").merge(
        round_summary,
        on=entity_id_col,
        how="left",
    )


def _percentile_rank(human_values: pd.Series, generated_value: float) -> float:
    values = human_values.dropna().astype(float).to_numpy()
    if values.size == 0 or math.isnan(generated_value):
        return float("nan")
    less = float(np.sum(values < generated_value))
    equal = float(np.sum(values == generated_value))
    return (less + (0.5 * equal)) / float(values.size)


def _z_score(generated_value: float, human_mean: float, human_std: float) -> float:
    if math.isnan(generated_value) or math.isnan(human_mean) or math.isnan(human_std):
        return float("nan")
    if human_std == 0:
        return 0.0 if generated_value == human_mean else float("nan")
    return (generated_value - human_mean) / human_std


def _build_human_round_metrics(repo_root: Path, metadata_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    round_index_df = pd.read_csv(
        repo_root / "data/raw_data/validation_wave/rounds.csv",
        usecols=["_id", "index"],
    ).rename(columns={"_id": "roundId"})
    player_rounds = pd.read_csv(
        repo_root / "data/raw_data/validation_wave/player-rounds.csv",
        usecols=[
            "playerId",
            "roundId",
            "gameId",
            "data.punished",
            "data.rewarded",
            "data.contribution",
            "data.roundPayoff",
        ],
    ).rename(
        columns={
            "gameId": "game_id",
            "data.punished": "punished_raw",
            "data.rewarded": "rewarded_raw",
            "data.contribution": "contribution",
            "data.roundPayoff": "round_payoff",
        }
    )
    player_rounds = player_rounds.merge(round_index_df, on="roundId", how="left", validate="many_to_one")
    player_rounds = player_rounds[player_rounds["game_id"].isin(set(metadata_df["game_id"]))].copy()
    player_rounds = player_rounds[player_rounds["contribution"].notna()].copy()
    player_rounds["punish_target_count"] = player_rounds["punished_raw"].apply(lambda value: len(_parse_action_dict(value)))
    player_rounds["reward_target_count"] = player_rounds["rewarded_raw"].apply(lambda value: len(_parse_action_dict(value)))
    player_rounds["has_punish"] = (player_rounds["punish_target_count"] > 0).astype(int)
    player_rounds["has_reward"] = (player_rounds["reward_target_count"] > 0).astype(int)
    player_rounds = player_rounds.merge(
        metadata_df[
            [
                "game_id",
                "treatment_name",
                "endowment",
                "mpcr",
                "chat_enabled",
                "punishment_enabled",
                "reward_enabled",
                "config_num_rounds",
                "chat_log",
            ]
        ],
        on="game_id",
        how="left",
    )
    player_rounds["round_number"] = player_rounds["index"].astype(int) + 1
    player_rounds["contribution_rate"] = player_rounds["contribution"].astype(float) / player_rounds["endowment"].astype(float)

    chat_rows: list[dict[str, Any]] = []
    for row in metadata_df.itertuples(index=False):
        if not bool(row.chat_enabled):
            continue
        chat_index = _index_chat_log("" if pd.isna(row.chat_log) else str(row.chat_log))
        for round_number, phase_map in chat_index.items():
            chat_rows.append(
                {
                    "game_id": row.game_id,
                    "round_number": int(round_number),
                    "message_count": int(sum(len(messages) for messages in phase_map.values())),
                }
            )
    chat_df = pd.DataFrame(chat_rows)

    human_round_df = (
        player_rounds.groupby(["game_id", "treatment_name", "round_number"], as_index=False)
        .agg(
            endowment=("endowment", "first"),
            mpcr=("mpcr", "first"),
            num_active_players=("playerId", "count"),
            total_contribution=("contribution", "sum"),
            total_round_payoff=("round_payoff", "sum"),
        )
        .sort_values(["game_id", "round_number"])
        .reset_index(drop=True)
    )
    human_round_df["total_contribution_rate"] = (
        human_round_df["total_contribution"] / (human_round_df["num_active_players"] * human_round_df["endowment"])
    )
    defect_round_coin_gen = human_round_df["num_active_players"] * human_round_df["endowment"]
    max_round_coin_gen = (
        human_round_df["mpcr"] * (human_round_df["num_active_players"] ** 2) * human_round_df["endowment"]
    )
    human_round_df["round_normalized_efficiency"] = (
        human_round_df["total_round_payoff"] - defect_round_coin_gen
    ) / (max_round_coin_gen - defect_round_coin_gen)
    if not chat_df.empty:
        human_round_df = human_round_df.merge(
            chat_df,
            on=["game_id", "round_number"],
            how="left",
        )
    else:
        human_round_df["message_count"] = 0
    human_round_df["message_count"] = human_round_df["message_count"].fillna(0).astype(int)
    human_round_df["has_chat"] = (human_round_df["message_count"] > 0).astype(int)

    human_actor_df = player_rounds[
        [
            "game_id",
            "treatment_name",
            "round_number",
            "playerId",
            "contribution_rate",
            "punish_target_count",
            "reward_target_count",
            "has_punish",
            "has_reward",
        ]
    ].rename(columns={"playerId": "player_id"})

    return human_actor_df, human_round_df


def _add_trajectory_rmse(
    human_game_df: pd.DataFrame,
    human_round_df: pd.DataFrame,
    generated_game_df: pd.DataFrame,
    generated_round_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    treatment_round_stats = (
        human_round_df.groupby(["treatment_name", "round_number"], as_index=False)
        .agg(
            human_total_contribution_rate_sum=("total_contribution_rate", "sum"),
            human_round_normalized_efficiency_sum=("round_normalized_efficiency", "sum"),
            human_round_count=("game_id", "count"),
        )
    )
    human_round_with_loo = human_round_df.merge(
        treatment_round_stats,
        on=["treatment_name", "round_number"],
        how="left",
    )
    denom = human_round_with_loo["human_round_count"] - 1
    human_round_with_loo["loo_total_contribution_rate_mean"] = np.where(
        denom > 0,
        (human_round_with_loo["human_total_contribution_rate_sum"] - human_round_with_loo["total_contribution_rate"]) / denom,
        np.nan,
    )
    human_round_with_loo["loo_round_normalized_efficiency_mean"] = np.where(
        denom > 0,
        (
            human_round_with_loo["human_round_normalized_efficiency_sum"]
            - human_round_with_loo["round_normalized_efficiency"]
        )
        / denom,
        np.nan,
    )

    human_round_with_loo["total_contribution_rate_sq_error_to_loo"] = (
        human_round_with_loo["total_contribution_rate"] - human_round_with_loo["loo_total_contribution_rate_mean"]
    ) ** 2
    human_round_with_loo["round_normalized_efficiency_sq_error_to_loo"] = (
        human_round_with_loo["round_normalized_efficiency"]
        - human_round_with_loo["loo_round_normalized_efficiency_mean"]
    ) ** 2

    human_rmse = (
        human_round_with_loo.groupby("game_id", as_index=False)
        .agg(
            total_contribution_rate_rmse_to_treatment_mean=(
                "total_contribution_rate_sq_error_to_loo",
                lambda s: float(np.sqrt(np.nanmean(s))),
            ),
            round_normalized_efficiency_rmse_to_treatment_mean=(
                "round_normalized_efficiency_sq_error_to_loo",
                lambda s: float(np.sqrt(np.nanmean(s))),
            ),
        )
    )
    human_game_df = human_game_df.merge(human_rmse, on="game_id", how="left")

    generated_round_targets = (
        treatment_round_stats.assign(
            human_total_contribution_rate_mean=lambda df: df["human_total_contribution_rate_sum"] / df["human_round_count"],
            human_round_normalized_efficiency_mean=lambda df: (
                df["human_round_normalized_efficiency_sum"] / df["human_round_count"]
            ),
        )[
            [
                "treatment_name",
                "round_number",
                "human_total_contribution_rate_mean",
                "human_round_normalized_efficiency_mean",
            ]
        ]
    )
    generated_round_eval = generated_round_df.merge(
        generated_round_targets,
        on=["treatment_name", "round_number"],
        how="left",
    )
    generated_round_eval["total_contribution_rate_sq_error_to_human_mean"] = (
        generated_round_eval["total_contribution_rate"] - generated_round_eval["human_total_contribution_rate_mean"]
    ) ** 2
    generated_round_eval["round_normalized_efficiency_sq_error_to_human_mean"] = (
        generated_round_eval["round_normalized_efficiency"]
        - generated_round_eval["human_round_normalized_efficiency_mean"]
    ) ** 2

    generated_rmse = (
        generated_round_eval.groupby("custom_id", as_index=False)
        .agg(
            total_contribution_rate_rmse_to_treatment_mean=(
                "total_contribution_rate_sq_error_to_human_mean",
                lambda s: float(np.sqrt(np.nanmean(s))),
            ),
            round_normalized_efficiency_rmse_to_treatment_mean=(
                "round_normalized_efficiency_sq_error_to_human_mean",
                lambda s: float(np.sqrt(np.nanmean(s))),
            ),
        )
    )
    generated_game_df = generated_game_df.merge(generated_rmse, on="custom_id", how="left")
    return human_game_df, generated_game_df


def _build_metric_comparison(
    generated_game_df: pd.DataFrame,
    human_game_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for generated_row in generated_game_df.to_dict(orient="records"):
        treatment_name = str(generated_row["treatment_name"])
        human_subset = human_game_df[human_game_df["treatment_name"] == treatment_name]
        for metric in CORE_METRICS:
            human_values = human_subset[metric].dropna()
            generated_value = _safe_float(generated_row.get(metric))
            human_mean = float(human_values.mean()) if not human_values.empty else float("nan")
            human_std = float(human_values.std(ddof=0)) if not human_values.empty else float("nan")
            human_median = float(human_values.median()) if not human_values.empty else float("nan")
            q25 = float(human_values.quantile(0.25)) if not human_values.empty else float("nan")
            q75 = float(human_values.quantile(0.75)) if not human_values.empty else float("nan")
            human_min = float(human_values.min()) if not human_values.empty else float("nan")
            human_max = float(human_values.max()) if not human_values.empty else float("nan")
            rows.append(
                {
                    "custom_id": generated_row["custom_id"],
                    "treatment_name": treatment_name,
                    "metric": metric,
                    "generated_value": generated_value,
                    "human_mean": human_mean,
                    "human_std": human_std,
                    "human_median": human_median,
                    "human_q25": q25,
                    "human_q75": q75,
                    "human_min": human_min,
                    "human_max": human_max,
                    "human_game_count": int(human_values.shape[0]),
                    "abs_diff_from_human_mean": abs(generated_value - human_mean)
                    if not math.isnan(generated_value) and not math.isnan(human_mean)
                    else float("nan"),
                    "z_score": _z_score(generated_value, human_mean, human_std),
                    "abs_z_score": abs(_z_score(generated_value, human_mean, human_std))
                    if not math.isnan(_z_score(generated_value, human_mean, human_std))
                    else float("nan"),
                    "percentile_within_humans": _percentile_rank(human_values, generated_value),
                    "within_human_range": int(
                        not human_values.empty and not math.isnan(generated_value) and human_min <= generated_value <= human_max
                    ),
                    "within_human_iqr": int(
                        not human_values.empty and not math.isnan(generated_value) and q25 <= generated_value <= q75
                    ),
                    "chat_enabled": bool(generated_row["chat_enabled"]),
                    "punishment_enabled": bool(generated_row["punishment_enabled"]),
                    "reward_enabled": bool(generated_row["reward_enabled"]),
                }
            )
    comparison_df = pd.DataFrame(rows)
    comparison_df["metric_scope"] = "all"
    chat_mask = comparison_df["metric"].isin(["messages_per_round", "rounds_with_chat_rate"])
    punish_mask = comparison_df["metric"] == "punish_actor_round_rate"
    reward_mask = comparison_df["metric"] == "reward_actor_round_rate"
    comparison_df.loc[chat_mask, "metric_scope"] = np.where(
        comparison_df.loc[chat_mask, "chat_enabled"],
        "enabled",
        "disabled",
    )
    comparison_df.loc[punish_mask, "metric_scope"] = np.where(
        comparison_df.loc[punish_mask, "punishment_enabled"],
        "enabled",
        "disabled",
    )
    comparison_df.loc[reward_mask, "metric_scope"] = np.where(
        comparison_df.loc[reward_mask, "reward_enabled"],
        "enabled",
        "disabled",
    )
    return comparison_df


def _summarize_metric_comparison(comparison_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    for metric, metric_df in comparison_df.groupby("metric", sort=True):
        if metric in {"messages_per_round", "rounds_with_chat_rate", "punish_actor_round_rate", "reward_actor_round_rate"}:
            metric_df = metric_df[metric_df["metric_scope"] == "enabled"].copy()
        if metric_df.empty:
            continue
        summary_rows.append(
            {
                "metric": metric,
                "num_treatments": int(metric_df["treatment_name"].nunique()),
                "mean_abs_diff_from_human_mean": float(metric_df["abs_diff_from_human_mean"].mean()),
                "mean_abs_z_score": float(metric_df["abs_z_score"].dropna().mean())
                if metric_df["abs_z_score"].notna().any()
                else float("nan"),
                "median_abs_z_score": float(metric_df["abs_z_score"].dropna().median())
                if metric_df["abs_z_score"].notna().any()
                else float("nan"),
                "mean_percentile_within_humans": float(metric_df["percentile_within_humans"].mean()),
                "median_percentile_within_humans": float(metric_df["percentile_within_humans"].median()),
                "within_human_range_rate": float(metric_df["within_human_range"].mean()),
                "within_human_iqr_rate": float(metric_df["within_human_iqr"].mean()),
            }
        )
    return pd.DataFrame(summary_rows).sort_values("metric").reset_index(drop=True)


def _summarize_treatment_typicality(comparison_df: pd.DataFrame) -> pd.DataFrame:
    scored = comparison_df.copy()
    scored = scored[
        (
            ~scored["metric"].isin(["messages_per_round", "rounds_with_chat_rate", "punish_actor_round_rate", "reward_actor_round_rate"])
        )
        | (scored["metric_scope"] == "enabled")
    ].copy()
    summary = (
        scored.groupby("treatment_name", as_index=False)
        .agg(
            metrics_used=("metric", "count"),
            mean_abs_z_score=("abs_z_score", "mean"),
            median_percentile=("percentile_within_humans", "median"),
        )
        .sort_values("mean_abs_z_score", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def _plot_generated_vs_human_means(
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> None:
    metric_specs = [
        ("mean_total_contribution_rate", "Mean Contribution Rate"),
        ("mean_round_normalized_efficiency", "Mean Normalized Efficiency"),
        ("punish_actor_round_rate", "Punish Actor-Rate"),
        ("messages_per_round", "Messages / Round"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (metric, label) in zip(axes.flat, metric_specs):
        plot_df = comparison_df[comparison_df["metric"] == metric].copy()
        if metric in {"messages_per_round", "punish_actor_round_rate"}:
            plot_df = plot_df[plot_df["metric_scope"] == "enabled"].copy()
        ax.scatter(
            plot_df["human_mean"],
            plot_df["generated_value"],
            alpha=0.8,
            s=28,
            color="#1f77b4",
        )
        finite_values = plot_df[["human_mean", "generated_value"]].replace([np.inf, -np.inf], np.nan).dropna()
        if not finite_values.empty:
            lower = float(min(finite_values.min()))
            upper = float(max(finite_values.max()))
            ax.plot([lower, upper], [lower, upper], linestyle="--", color="0.45", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("Human Treatment Mean")
        ax.set_ylabel("GPT-5.1 Generated Value")
    fig.suptitle("GPT-5.1 k=0 Rollouts vs Human Validation Treatments", fontsize=14)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare forecasting rollouts against the human validation-wave treatment distribution."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--parsed-output-jsonl", type=Path, default=None)
    parser.add_argument("--request-manifest-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-label", type=str, default="gpt_5_1_k0")
    args = parser.parse_args()

    if args.run_name:
        root = args.forecasting_root
        args.parsed_output_jsonl = args.parsed_output_jsonl or (
            root / "metadata" / args.run_name / "parsed_output.jsonl"
        )
        args.request_manifest_csv = args.request_manifest_csv or (
            root / "metadata" / args.run_name / "request_manifest.csv"
        )
        args.output_dir = args.output_dir or (root / "results" / f"{args.run_name}__vs_human_treatments")

    if args.parsed_output_jsonl is None or args.request_manifest_csv is None or args.output_dir is None:
        raise ValueError(
            "Provide either --run-name or all of --parsed-output-jsonl, --request-manifest-csv, and --output-dir."
        )

    parsed_rows = _read_jsonl(args.parsed_output_jsonl)
    request_manifest = _load_request_manifest(args.request_manifest_csv)
    manifest_df = pd.read_csv(args.request_manifest_csv)
    treatment_names = set(manifest_df["treatment_name"].astype(str))

    validation_games = load_wave_games(
        repo_root=args.repo_root,
        wave_name="validation_wave",
        processed_suffix="val",
        min_num_rounds_exclusive=0,
    )
    validation_games_by_id = {game.game_id: game for game in validation_games}
    avatar_map = _load_avatar_map(args.repo_root / "data/raw_data/validation_wave/players.csv")
    game_rows = _load_game_rows(args.repo_root / "data/raw_data/validation_wave/games.csv")

    generated_actor_frames: list[pd.DataFrame] = []
    generated_round_frames: list[pd.DataFrame] = []
    generated_metadata_rows: list[dict[str, Any]] = []

    for parsed_row in parsed_rows:
        if not bool(parsed_row.get("parse_success")):
            continue
        custom_id = str(parsed_row["custom_id"])
        manifest_row = request_manifest[custom_id]
        game_id = str(manifest_row["game_id"])
        game = validation_games_by_id[game_id]
        raw_player_order, avatar_to_player = _build_avatar_mapping(
            game_id=game_id,
            expected_avatars=list(manifest_row["avatars"]),
            game_rows=game_rows,
            avatar_map=avatar_map,
        )
        predicted_rounds = _parsed_rounds_to_round_records(
            game=game,
            k=int(manifest_row["k"]),
            raw_player_order=raw_player_order,
            avatar_to_player=avatar_to_player,
            punishment_enabled=bool(manifest_row.get("punishment_enabled")),
            reward_enabled=bool(manifest_row.get("reward_enabled")),
            parsed_rounds=list(parsed_row.get("predicted_rounds", [])),
        )
        actor_df, round_df = _round_summary_from_round_records(
            entity_id=custom_id,
            treatment_name=str(manifest_row["treatment_name"]),
            rounds=predicted_rounds,
            message_count_by_round=_message_count_by_round_from_parsed_rounds(
                list(parsed_row.get("predicted_rounds", []))
            ),
            endowment=int(game.config.endowment),
            mpcr=float(game.config.mpcr),
            num_players=int(game.num_players),
            entity_kind="generated",
        )
        actor_df.insert(0, "custom_id", custom_id)
        round_df.insert(0, "custom_id", custom_id)
        generated_actor_frames.append(actor_df)
        generated_round_frames.append(round_df)
        generated_metadata_rows.append(
            {
                "custom_id": custom_id,
                "game_id": game_id,
                "treatment_name": str(manifest_row["treatment_name"]),
                "chat_enabled": bool(manifest_row["chat_enabled"]),
                "punishment_enabled": bool(manifest_row["punishment_enabled"]),
                "reward_enabled": bool(manifest_row["reward_enabled"]),
                "num_players": int(manifest_row["num_players"]),
                "num_rounds": int(manifest_row["num_rounds"]),
                "model_label": args.model_label,
            }
        )

    generated_actor_df = pd.concat(generated_actor_frames, ignore_index=True)
    generated_round_df = pd.concat(generated_round_frames, ignore_index=True)
    generated_metadata_df = pd.DataFrame(generated_metadata_rows)
    generated_game_df = _summarize_game_metrics(
        generated_actor_df,
        generated_round_df,
        generated_metadata_df,
        entity_id_col="custom_id",
    )

    human_metadata_df = _load_validation_metadata(args.repo_root, treatment_names)
    human_actor_df, human_round_df = _build_human_round_metrics(args.repo_root, human_metadata_df)
    human_game_df = _summarize_game_metrics(
        human_actor_df.rename(columns={"game_id": "entity_id"}),
        human_round_df.rename(columns={"game_id": "entity_id"}),
        human_metadata_df.rename(columns={"game_id": "entity_id"}),
        entity_id_col="entity_id",
    ).rename(columns={"entity_id": "game_id"})

    human_game_df, generated_game_df = _add_trajectory_rmse(
        human_game_df,
        human_round_df,
        generated_game_df,
        generated_round_df,
    )

    comparison_df = _build_metric_comparison(generated_game_df, human_game_df)
    overall_df = _summarize_metric_comparison(comparison_df)
    treatment_typicality_df = _summarize_treatment_typicality(comparison_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_game_df.sort_values("treatment_name").to_csv(args.output_dir / "generated_game_summary.csv", index=False)
    generated_round_df.sort_values(["treatment_name", "round_number"]).to_csv(
        args.output_dir / "generated_round_summary.csv",
        index=False,
    )
    human_game_df.sort_values(["treatment_name", "game_id"]).to_csv(
        args.output_dir / "human_game_summary.csv",
        index=False,
    )
    human_round_df.sort_values(["treatment_name", "game_id", "round_number"]).to_csv(
        args.output_dir / "human_round_summary.csv",
        index=False,
    )
    comparison_df.sort_values(["metric", "treatment_name"]).to_csv(
        args.output_dir / "treatment_metric_comparison.csv",
        index=False,
    )
    overall_df.to_csv(args.output_dir / "overall_metric_summary.csv", index=False)
    treatment_typicality_df.to_csv(args.output_dir / "treatment_typicality.csv", index=False)
    _plot_generated_vs_human_means(comparison_df, args.output_dir / "generated_vs_human_treatment_means.png")

    manifest = {
        "parsed_output_jsonl": str(args.parsed_output_jsonl),
        "request_manifest_csv": str(args.request_manifest_csv),
        "model_label": args.model_label,
        "num_generated_games": int(generated_game_df["custom_id"].nunique()),
        "num_human_games": int(human_game_df["game_id"].nunique()),
        "human_selection": "validation-wave games with valid_number_of_starting_players == True, including incomplete games",
        "human_treatment_counts": (
            human_game_df.groupby("treatment_name")["game_id"].nunique().sort_index().astype(int).to_dict()
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote outputs to {args.output_dir}")
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
