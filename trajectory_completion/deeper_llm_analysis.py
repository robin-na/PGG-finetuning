from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_common_games(path: Path) -> dict[int, set[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): set(v) for k, v in manifest["common_games_by_k"].items()}


def _filter_to_common_subset(df: pd.DataFrame, common_games_by_k: dict[int, set[str]]) -> pd.DataFrame:
    parts = []
    for k, game_ids in common_games_by_k.items():
        parts.append(df[(df["k"] == k) & (df["game_id"].astype(str).isin(game_ids))])
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def _load_config_metadata(path: Path) -> pd.DataFrame:
    cols = [
        "gameId",
        "CONFIG_playerCount",
        "CONFIG_numRounds",
        "CONFIG_showNRounds",
        "CONFIG_allOrNothing",
        "CONFIG_chat",
        "CONFIG_punishmentExists",
        "CONFIG_rewardExists",
        "CONFIG_showOtherSummaries",
        "CONFIG_showPunishmentId",
        "CONFIG_showRewardId",
        "CONFIG_MPCR",
        "CONFIG_multiplier",
        "CONFIG_punishmentCost",
        "CONFIG_punishmentMagnitude",
        "CONFIG_rewardCost",
        "CONFIG_rewardMagnitude",
        "CONFIG_treatmentName",
    ]
    df = pd.read_csv(path, usecols=cols).rename(columns={"gameId": "game_id"})
    df["show_interaction_id"] = df["CONFIG_showPunishmentId"].astype(bool)
    df["interaction_combo"] = np.select(
        [
            df["CONFIG_punishmentExists"].astype(bool) & df["CONFIG_rewardExists"].astype(bool),
            df["CONFIG_punishmentExists"].astype(bool) & ~df["CONFIG_rewardExists"].astype(bool),
            ~df["CONFIG_punishmentExists"].astype(bool) & df["CONFIG_rewardExists"].astype(bool),
        ],
        [
            "both",
            "punishment_only",
            "reward_only",
        ],
        default="none",
    )
    return df


def _inter_model_similarity(actor_a: pd.DataFrame, actor_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    join_keys = ["k", "game_id", "round_index", "player_id"]
    left = actor_a[join_keys + ["predicted_contribution", "predicted_has_punish", "predicted_has_reward", "predicted_punished", "predicted_rewarded"]].rename(
        columns={
            "predicted_contribution": "predicted_contribution_a",
            "predicted_has_punish": "predicted_has_punish_a",
            "predicted_has_reward": "predicted_has_reward_a",
            "predicted_punished": "predicted_punished_a",
            "predicted_rewarded": "predicted_rewarded_a",
        }
    )
    right = actor_b[join_keys + ["predicted_contribution", "predicted_has_punish", "predicted_has_reward", "predicted_punished", "predicted_rewarded"]].rename(
        columns={
            "predicted_contribution": "predicted_contribution_b",
            "predicted_has_punish": "predicted_has_punish_b",
            "predicted_has_reward": "predicted_has_reward_b",
            "predicted_punished": "predicted_punished_b",
            "predicted_rewarded": "predicted_rewarded_b",
        }
    )
    merged = left.merge(right, on=join_keys, how="inner")
    merged["same_contribution"] = (merged["predicted_contribution_a"] == merged["predicted_contribution_b"]).astype(int)
    merged["contribution_abs_diff"] = (
        merged["predicted_contribution_a"] - merged["predicted_contribution_b"]
    ).abs()
    merged["same_punish_presence"] = (
        merged["predicted_has_punish_a"] == merged["predicted_has_punish_b"]
    ).astype(int)
    merged["same_reward_presence"] = (
        merged["predicted_has_reward_a"] == merged["predicted_has_reward_b"]
    ).astype(int)
    merged["same_punished_dict"] = (
        merged["predicted_punished_a"] == merged["predicted_punished_b"]
    ).astype(int)
    merged["same_rewarded_dict"] = (
        merged["predicted_rewarded_a"] == merged["predicted_rewarded_b"]
    ).astype(int)

    actor_summary = (
        merged.groupby("k", as_index=False)[
            [
                "same_contribution",
                "contribution_abs_diff",
                "same_punish_presence",
                "same_reward_presence",
                "same_punished_dict",
                "same_rewarded_dict",
            ]
        ]
        .mean()
        .rename(
            columns={
                "same_contribution": "actor_round_contribution_match_rate",
                "contribution_abs_diff": "actor_round_mean_contribution_abs_diff",
                "same_punish_presence": "actor_round_punish_presence_match_rate",
                "same_reward_presence": "actor_round_reward_presence_match_rate",
                "same_punished_dict": "actor_round_punished_dict_match_rate",
                "same_rewarded_dict": "actor_round_rewarded_dict_match_rate",
            }
        )
    )
    actor_counts = (
        merged.groupby("k", as_index=False)
        .agg(num_actor_rows=("player_id", "count"), num_games=("game_id", "nunique"))
    )
    actor_summary = actor_counts.merge(actor_summary, on="k", how="left")

    trajectory = (
        merged.groupby(["k", "game_id", "player_id"], as_index=False)[
            ["same_contribution", "same_punished_dict", "same_rewarded_dict"]
        ]
        .mean()
        .rename(
            columns={
                "same_contribution": "trajectory_contribution_match_rate",
                "same_punished_dict": "trajectory_punished_dict_match_rate",
                "same_rewarded_dict": "trajectory_rewarded_dict_match_rate",
            }
        )
    )
    trajectory["full_contribution_trajectory_match"] = (
        trajectory["trajectory_contribution_match_rate"] == 1.0
    ).astype(int)
    trajectory_summary = (
        trajectory.groupby("k", as_index=False)[
            [
                "full_contribution_trajectory_match",
                "trajectory_punished_dict_match_rate",
                "trajectory_rewarded_dict_match_rate",
            ]
        ]
        .mean()
    )
    trajectory_counts = (
        trajectory.groupby("k", as_index=False)
        .agg(num_player_trajectories=("player_id", "count"), num_games=("game_id", "nunique"))
    )
    trajectory_summary = trajectory_counts.merge(trajectory_summary, on="k", how="left")
    return actor_summary, trajectory_summary


def _group_means(game_summary: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    return (
        game_summary.groupby(group_cols, as_index=False)
        .agg({**{"game_id": "nunique"}, **{metric: "mean" for metric in metrics}})
        .rename(columns={"game_id": "num_games"})
    )


def _zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / std


def _fit_standardized_ols(game_summary: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, float]:
    design = pd.DataFrame(index=game_summary.index)
    design["intercept"] = 1.0
    design["chat_enabled"] = game_summary["CONFIG_chat"].astype(int)
    design["all_or_nothing"] = game_summary["CONFIG_allOrNothing"].astype(int)
    design["punishment_exists"] = game_summary["CONFIG_punishmentExists"].astype(int)
    design["reward_exists"] = game_summary["CONFIG_rewardExists"].astype(int)
    design["show_n_rounds"] = game_summary["CONFIG_showNRounds"].astype(int)
    design["show_other_summaries"] = game_summary["CONFIG_showOtherSummaries"].astype(int)
    design["show_interaction_id"] = game_summary["show_interaction_id"].astype(int)
    design["player_count_z"] = _zscore(game_summary["CONFIG_playerCount"].astype(float))
    design["num_rounds_z"] = _zscore(game_summary["CONFIG_numRounds"].astype(float))
    design["mpcr_z"] = _zscore(game_summary["CONFIG_MPCR"].astype(float))
    for k_value in sorted(game_summary["k"].unique()):
        if k_value == min(game_summary["k"].unique()):
            continue
        design[f"k_{k_value}"] = (game_summary["k"] == k_value).astype(int)

    y = _zscore(game_summary[metric].astype(float))
    x = design.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y.to_numpy(dtype=float), rcond=None)
    y_hat = x @ beta
    ss_res = float(((y.to_numpy(dtype=float) - y_hat) ** 2).sum())
    ss_tot = float(((y.to_numpy(dtype=float) - y.mean()) ** 2).sum())
    r2 = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    coef_df = pd.DataFrame({"feature": design.columns, "standardized_coef": beta})
    coef_df = coef_df[coef_df["feature"] != "intercept"].sort_values(
        "standardized_coef",
        key=lambda s: s.abs(),
        ascending=False,
    )
    return coef_df.reset_index(drop=True), r2


def main() -> None:
    parser = argparse.ArgumentParser(description="Deeper LLM-vs-LLM similarity and config analysis.")
    parser.add_argument("--common-manifest-json", type=Path, required=True)
    parser.add_argument("--df-analysis-csv", type=Path, required=True)
    parser.add_argument("--gpt4-actor-csv", type=Path, required=True)
    parser.add_argument("--gpt4-game-summary-csv", type=Path, required=True)
    parser.add_argument("--gpt5-actor-csv", type=Path, required=True)
    parser.add_argument("--gpt5-game-summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    common_games_by_k = _load_common_games(args.common_manifest_json)
    config_df = _load_config_metadata(args.df_analysis_csv)
    metrics = [
        "contribution_rate_mae",
        "future_normalized_efficiency_abs_error",
        "punish_target_f1",
        "reward_target_f1",
    ]

    gpt4_actor = _filter_to_common_subset(pd.read_csv(args.gpt4_actor_csv), common_games_by_k)
    gpt5_actor = _filter_to_common_subset(pd.read_csv(args.gpt5_actor_csv), common_games_by_k)
    gpt4_game = _filter_to_common_subset(pd.read_csv(args.gpt4_game_summary_csv), common_games_by_k).merge(
        config_df,
        on="game_id",
        how="left",
    )
    gpt5_game = _filter_to_common_subset(pd.read_csv(args.gpt5_game_summary_csv), common_games_by_k).merge(
        config_df,
        on="game_id",
        how="left",
    )

    inter_actor, inter_traj = _inter_model_similarity(gpt4_actor, gpt5_actor)
    inter_actor.insert(0, "model_pair", "gpt_4_1_mini_vs_gpt_5_1")
    inter_traj.insert(0, "model_pair", "gpt_4_1_mini_vs_gpt_5_1")

    grouped_rows = []
    for model_name, df in [("gpt_4_1_mini", gpt4_game), ("gpt_5_1", gpt5_game)]:
        for group_col in ["interaction_combo", "CONFIG_showNRounds", "show_interaction_id", "CONFIG_playerCount", "CONFIG_numRounds"]:
            grouped = _group_means(df, ["k", group_col], metrics)
            grouped.insert(0, "group_name", group_col)
            grouped.insert(0, "model", model_name)
            grouped_rows.append(grouped.rename(columns={group_col: "group_value"}))

    regression_rows = []
    regression_manifest_rows = []
    for model_name, df in [("gpt_4_1_mini", gpt4_game), ("gpt_5_1", gpt5_game)]:
        for metric in metrics:
            coef_df, r2 = _fit_standardized_ols(df, metric)
            coef_df.insert(0, "metric", metric)
            coef_df.insert(0, "model", model_name)
            regression_rows.append(coef_df)
            regression_manifest_rows.append({"model": model_name, "metric": metric, "r2": r2, "n_rows": len(df)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inter_actor.to_csv(args.output_dir / "inter_model_actor_similarity.csv", index=False)
    inter_traj.to_csv(args.output_dir / "inter_model_trajectory_similarity.csv", index=False)
    pd.concat(grouped_rows, ignore_index=True).to_csv(args.output_dir / "grouped_config_means.csv", index=False)
    pd.concat(regression_rows, ignore_index=True).to_csv(args.output_dir / "config_regression_coefficients.csv", index=False)
    pd.DataFrame(regression_manifest_rows).to_csv(args.output_dir / "config_regression_manifest.csv", index=False)

    manifest = {
        "common_manifest_json": str(args.common_manifest_json),
        "df_analysis_csv": str(args.df_analysis_csv),
        "gpt4_actor_csv": str(args.gpt4_actor_csv),
        "gpt4_game_summary_csv": str(args.gpt4_game_summary_csv),
        "gpt5_actor_csv": str(args.gpt5_actor_csv),
        "gpt5_game_summary_csv": str(args.gpt5_game_summary_csv),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
