from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from trajectory_completion.data import _parse_action_dict

from .analyze_vs_human_treatments import _normalized_round_slope, _spearman_rank_correlation
from .build_batch_inputs import _index_chat_log


TARGET_METRICS = [
    ("mean_total_contribution_rate", "Mean contrib"),
    ("mean_round_normalized_efficiency", "Mean eff"),
    ("final_total_contribution_rate", "Final contrib"),
    ("final_round_normalized_efficiency", "Final eff"),
    ("total_contribution_rate_sd", "Contrib SD"),
    ("total_contribution_rate_decay_slope", "Contrib slope"),
    ("round_normalized_efficiency_decay_slope", "Eff slope"),
    ("mean_within_round_contribution_rate_var", "Within-round var"),
    ("punish_actor_round_rate", "Punish rate"),
    ("reward_actor_round_rate", "Reward rate"),
    ("messages_per_round", "Msgs/round"),
    ("rounds_with_chat_rate", "Chat-round rate"),
]

FEATURE_COLUMNS = [
    "CONFIG_playerCount",
    "CONFIG_numRounds",
    "CONFIG_showNRounds",
    "CONFIG_endowment",
    "CONFIG_multiplier",
    "CONFIG_allOrNothing",
    "CONFIG_chat",
    "CONFIG_defaultContribProp",
    "CONFIG_punishmentExists",
    "CONFIG_punishmentCost",
    "CONFIG_punishmentMagnitude",
    "CONFIG_rewardExists",
    "CONFIG_rewardCost",
    "CONFIG_rewardMagnitude",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
    "CONFIG_MPCR",
    "CONFIG_punishmentTech",
    "CONFIG_rewardTech",
]

BOOL_FEATURES = [
    "CONFIG_showNRounds",
    "CONFIG_allOrNothing",
    "CONFIG_chat",
    "CONFIG_defaultContribProp",
    "CONFIG_punishmentExists",
    "CONFIG_rewardExists",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
]

CATEGORICAL_FEATURES = [
    "CONFIG_punishmentTech",
    "CONFIG_rewardTech",
]


def _relative_efficiency(
    total_round_payoff: pd.Series | float,
    defect_round_coin_gen: pd.Series | float,
    max_round_coin_gen: pd.Series | float,
) -> pd.Series | float:
    return (total_round_payoff - defect_round_coin_gen) / (max_round_coin_gen - defect_round_coin_gen)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _load_wave_metadata(repo_root: Path, split: str) -> pd.DataFrame:
    processed_path = repo_root / "data" / "processed_data" / f"df_analysis_{split}.csv"
    usecols = ["gameId", "CONFIG_treatmentName", "valid_number_of_starting_players", "chat_log", *FEATURE_COLUMNS]
    metadata = pd.read_csv(processed_path, usecols=usecols).drop_duplicates("gameId")
    metadata = metadata.rename(
        columns={
            "gameId": "game_id",
            "CONFIG_treatmentName": "treatment_name",
            "valid_number_of_starting_players": "valid_start",
        }
    )
    metadata = metadata[metadata["valid_start"].apply(_as_bool)].copy()
    for col in BOOL_FEATURES:
        metadata[col] = metadata[col].apply(_as_bool)
    for col in CATEGORICAL_FEATURES:
        metadata[col] = metadata[col].fillna("missing").astype(str)
    return metadata.reset_index(drop=True)


def _build_human_round_metrics(
    repo_root: Path,
    wave_name: str,
    metadata_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    round_index_df = pd.read_csv(
        repo_root / "data" / "raw_data" / wave_name / "rounds.csv",
        usecols=["_id", "index"],
    ).rename(columns={"_id": "roundId"})
    player_rounds = pd.read_csv(
        repo_root / "data" / "raw_data" / wave_name / "player-rounds.csv",
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
    player_rounds["punish_target_count"] = player_rounds["punished_raw"].apply(
        lambda value: len(_parse_action_dict(value))
    )
    player_rounds["reward_target_count"] = player_rounds["rewarded_raw"].apply(
        lambda value: len(_parse_action_dict(value))
    )
    player_rounds["has_punish"] = (player_rounds["punish_target_count"] > 0).astype(int)
    player_rounds["has_reward"] = (player_rounds["reward_target_count"] > 0).astype(int)
    player_rounds = player_rounds.merge(
        metadata_df[
            [
                "game_id",
                "treatment_name",
                "CONFIG_endowment",
                "CONFIG_MPCR",
                "CONFIG_chat",
                "chat_log",
            ]
        ],
        on="game_id",
        how="left",
    )
    player_rounds["round_number"] = player_rounds["index"].astype(int) + 1
    player_rounds["contribution_rate"] = (
        player_rounds["contribution"].astype(float) / player_rounds["CONFIG_endowment"].astype(float)
    )

    chat_rows: list[dict[str, Any]] = []
    for row in metadata_df.itertuples(index=False):
        if not bool(row.CONFIG_chat):
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
            endowment=("CONFIG_endowment", "first"),
            mpcr=("CONFIG_MPCR", "first"),
            num_active_players=("playerId", "count"),
            total_contribution=("contribution", "sum"),
            total_round_payoff=("round_payoff", "sum"),
            within_round_contribution_rate_var=("contribution_rate", lambda s: float(s.var(ddof=0))),
        )
        .sort_values(["game_id", "round_number"])
        .reset_index(drop=True)
    )
    human_round_df["total_contribution_rate"] = (
        human_round_df["total_contribution"]
        / (human_round_df["num_active_players"] * human_round_df["endowment"])
    )
    defect_round_coin_gen = human_round_df["num_active_players"] * human_round_df["endowment"]
    max_round_coin_gen = (
        human_round_df["mpcr"] * (human_round_df["num_active_players"] ** 2) * human_round_df["endowment"]
    )
    human_round_df["round_normalized_efficiency"] = _relative_efficiency(
        human_round_df["total_round_payoff"],
        defect_round_coin_gen,
        max_round_coin_gen,
    )
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


def _summarize_game_metrics(
    actor_df: pd.DataFrame,
    round_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    actor_summary = (
        actor_df.groupby("entity_id", as_index=False)
        .agg(
            punish_actor_round_rate=("has_punish", "mean"),
            reward_actor_round_rate=("has_reward", "mean"),
        )
    )
    round_summary = (
        round_df.groupby("entity_id", as_index=False)
        .agg(
            mean_total_contribution_rate=("total_contribution_rate", "mean"),
            total_contribution_rate_sd=("total_contribution_rate", lambda s: float(s.std(ddof=0))),
            final_total_contribution_rate=("total_contribution_rate", lambda s: float(s.iloc[-1])),
            mean_round_normalized_efficiency=("round_normalized_efficiency", "mean"),
            final_round_normalized_efficiency=("round_normalized_efficiency", lambda s: float(s.iloc[-1])),
            mean_within_round_contribution_rate_var=("within_round_contribution_rate_var", "mean"),
            messages_per_round=("message_count", "mean"),
            rounds_with_chat_rate=("has_chat", "mean"),
        )
    )
    dynamic_rows: list[dict[str, Any]] = []
    for entity_id, group in round_df.groupby("entity_id", sort=False):
        ordered = group.sort_values("round_number")
        dynamic_rows.append(
            {
                "entity_id": entity_id,
                "total_contribution_rate_decay_slope": _normalized_round_slope(
                    ordered["round_number"],
                    ordered["total_contribution_rate"],
                ),
                "round_normalized_efficiency_decay_slope": _normalized_round_slope(
                    ordered["round_number"],
                    ordered["round_normalized_efficiency"],
                ),
            }
        )
    dynamic_summary = pd.DataFrame(dynamic_rows)
    return (
        metadata_df.merge(actor_summary, on="entity_id", how="left")
        .merge(round_summary, on="entity_id", how="left")
        .merge(dynamic_summary, on="entity_id", how="left")
    )


def _build_wave_game_summary(repo_root: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    wave_name = "learning_wave" if split == "learn" else "validation_wave"
    metadata_df = _load_wave_metadata(repo_root, split)
    actor_df, round_df = _build_human_round_metrics(repo_root, wave_name, metadata_df)
    game_summary_df = _summarize_game_metrics(
        actor_df.rename(columns={"game_id": "entity_id"}),
        round_df.rename(columns={"game_id": "entity_id"}),
        metadata_df.rename(columns={"game_id": "entity_id"}),
    ).rename(columns={"entity_id": "game_id"})
    return game_summary_df, round_df


def _build_treatment_summary(game_summary_df: pd.DataFrame) -> pd.DataFrame:
    feature_check = game_summary_df.groupby("treatment_name")[FEATURE_COLUMNS].nunique(dropna=False)
    inconsistent = feature_check[(feature_check > 1).any(axis=1)]
    if not inconsistent.empty:
        raise ValueError(f"CONFIG features vary within treatment for {len(inconsistent)} treatments.")

    feature_df = game_summary_df.groupby("treatment_name", as_index=False)[FEATURE_COLUMNS].first()
    target_df = game_summary_df.groupby("treatment_name", as_index=False)[
        [metric for metric, _ in TARGET_METRICS]
    ].mean()
    count_df = (
        game_summary_df.groupby("treatment_name", as_index=False)["game_id"]
        .nunique()
        .rename(columns={"game_id": "n_games"})
    )
    return feature_df.merge(target_df, on="treatment_name", how="left").merge(
        count_df,
        on="treatment_name",
        how="left",
    )


def _build_design_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_design = train_df[FEATURE_COLUMNS].copy()
    val_design = val_df[FEATURE_COLUMNS].copy()
    for col in BOOL_FEATURES:
        train_design[col] = train_design[col].astype(int)
        val_design[col] = val_design[col].astype(int)
    for col in CATEGORICAL_FEATURES:
        train_design[col] = train_design[col].fillna("missing").astype(str)
        val_design[col] = val_design[col].fillna("missing").astype(str)

    combined = pd.concat(
        [
            train_design.assign(_split="train"),
            val_design.assign(_split="val"),
        ],
        ignore_index=True,
    )
    combined = pd.get_dummies(combined, columns=CATEGORICAL_FEATURES, dummy_na=False)
    train_encoded = combined[combined["_split"] == "train"].drop(columns="_split").reset_index(drop=True)
    val_encoded = combined[combined["_split"] == "val"].drop(columns="_split").reset_index(drop=True)

    numeric_cols = train_encoded.columns.tolist()
    medians = train_encoded[numeric_cols].median(numeric_only=True)
    train_encoded = train_encoded.fillna(medians)
    val_encoded = val_encoded.fillna(medians)
    return train_encoded, val_encoded


def _oracle_noise_floor_rmse(validation_game_summary_df: pd.DataFrame, metric: str) -> float:
    per_treatment = (
        validation_game_summary_df.groupby("treatment_name")[metric]
        .agg(["count", lambda s: float(s.var(ddof=0))])
        .reset_index()
        .rename(columns={"count": "n_games", "<lambda_0>": "variance"})
    )
    if per_treatment.empty:
        return float("nan")
    mse_floor = np.mean(per_treatment["variance"] / per_treatment["n_games"])
    return float(np.sqrt(mse_floor))


def _fit_and_predict(
    train_design: pd.DataFrame,
    train_targets: pd.Series,
    val_design: pd.DataFrame,
) -> np.ndarray:
    model = LinearRegression()
    model.fit(train_design.to_numpy(dtype=float), train_targets.to_numpy(dtype=float))
    return model.predict(val_design.to_numpy(dtype=float))


def _plot_core_scatter(prediction_df: pd.DataFrame, output_path: Path) -> None:
    metric_labels = dict(TARGET_METRICS)
    selected_metrics = [
        "mean_total_contribution_rate",
        "mean_round_normalized_efficiency",
        "final_total_contribution_rate",
        "final_round_normalized_efficiency",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, metric in zip(axes.flat, selected_metrics):
        subset = prediction_df[prediction_df["metric"] == metric].copy()
        ax.scatter(
            subset["human_treatment_mean"],
            subset["predicted_treatment_mean"],
            s=28,
            alpha=0.85,
            color="#1f77b4",
        )
        finite_values = subset[["human_treatment_mean", "predicted_treatment_mean"]].replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()
        if not finite_values.empty:
            lower = float(min(finite_values.min()))
            upper = float(max(finite_values.max()))
            ax.plot([lower, upper], [lower, upper], linestyle="--", color="0.45", linewidth=1)
        ax.set_title(metric_labels[metric])
        ax.set_xlabel("Validation Human Treatment Mean")
        ax.set_ylabel("Linear Regression Prediction")
    fig.suptitle("CONFIG-Only Linear Regression Vs Validation Treatment Means", fontsize=14)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a CONFIG-only linear regression baseline on learning-wave treatment means."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.forecasting_root / "results" / "config_linear_regression__macro_eval"

    learning_game_summary_df, _ = _build_wave_game_summary(args.repo_root, "learn")
    validation_game_summary_df, _ = _build_wave_game_summary(args.repo_root, "val")
    learning_treatment_df = _build_treatment_summary(learning_game_summary_df)
    validation_treatment_df = _build_treatment_summary(validation_game_summary_df)
    train_design, val_design = _build_design_matrices(learning_treatment_df, validation_treatment_df)

    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for metric, _ in TARGET_METRICS:
        train_targets = learning_treatment_df[metric].astype(float)
        val_targets = validation_treatment_df[metric].astype(float)
        valid_train = train_targets.notna()
        valid_val = val_targets.notna()
        if not valid_train.any() or not valid_val.any():
            continue
        predictions = _fit_and_predict(
            train_design.loc[valid_train].reset_index(drop=True),
            train_targets.loc[valid_train].reset_index(drop=True),
            val_design.loc[valid_val].reset_index(drop=True),
        )
        prediction_df = pd.DataFrame(
            {
                "treatment_name": validation_treatment_df.loc[valid_val, "treatment_name"].to_numpy(),
                "metric": metric,
                "predicted_treatment_mean": predictions,
                "human_treatment_mean": val_targets.loc[valid_val].to_numpy(dtype=float),
            }
        )
        prediction_df["abs_error"] = (
            prediction_df["predicted_treatment_mean"] - prediction_df["human_treatment_mean"]
        ).abs()
        prediction_rows.extend(prediction_df.to_dict(orient="records"))

        rmse = float(
            np.sqrt(
                np.mean(
                    (prediction_df["predicted_treatment_mean"] - prediction_df["human_treatment_mean"]) ** 2
                )
            )
        )
        summary_rows.append(
            {
                "metric": metric,
                "num_learning_treatments": int(learning_treatment_df["treatment_name"].nunique()),
                "num_validation_treatments": int(prediction_df["treatment_name"].nunique()),
                "mae": float(prediction_df["abs_error"].mean()),
                "rmse": rmse,
                "spearman": _spearman_rank_correlation(
                    prediction_df["predicted_treatment_mean"],
                    prediction_df["human_treatment_mean"],
                ),
                "oracle_noise_floor_rmse": _oracle_noise_floor_rmse(validation_game_summary_df, metric),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df["ratio_to_oracle_noise_floor"] = (
        summary_df["rmse"] / summary_df["oracle_noise_floor_rmse"]
    )
    prediction_df = pd.DataFrame(prediction_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    learning_game_summary_df.sort_values(["treatment_name", "game_id"]).to_csv(
        args.output_dir / "learning_game_summary.csv",
        index=False,
    )
    validation_game_summary_df.sort_values(["treatment_name", "game_id"]).to_csv(
        args.output_dir / "validation_game_summary.csv",
        index=False,
    )
    learning_treatment_df.sort_values("treatment_name").to_csv(
        args.output_dir / "learning_treatment_means.csv",
        index=False,
    )
    validation_treatment_df.sort_values("treatment_name").to_csv(
        args.output_dir / "validation_treatment_means.csv",
        index=False,
    )
    prediction_df.sort_values(["metric", "treatment_name"]).to_csv(
        args.output_dir / "linear_regression_validation_predictions.csv",
        index=False,
    )
    summary_df.sort_values("metric").to_csv(
        args.output_dir / "linear_regression_macro_summary.csv",
        index=False,
    )
    _plot_core_scatter(prediction_df, args.output_dir / "linear_regression_vs_validation_means.png")

    manifest = {
        "train_split": "learn",
        "test_split": "val",
        "valid_start_only": True,
        "feature_columns": FEATURE_COLUMNS,
        "target_metrics": [metric for metric, _ in TARGET_METRICS],
        "num_learning_games": int(learning_game_summary_df["game_id"].nunique()),
        "num_validation_games": int(validation_game_summary_df["game_id"].nunique()),
        "num_learning_treatments": int(learning_treatment_df["treatment_name"].nunique()),
        "num_validation_treatments": int(validation_treatment_df["treatment_name"].nunique()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote outputs to {args.output_dir}")
    print(summary_df.sort_values("metric").to_string(index=False))


if __name__ == "__main__":
    main()
