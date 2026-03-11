from __future__ import annotations

from pathlib import Path

import pandas as pd

from simulation_statistical.archetype_distribution_embedding.models.env_distribution_dirichlet import DirichletEnvRegressor
from simulation_statistical.archetype_distribution_embedding.utils.constants import REQUIRED_CONFIG_COLUMNS


def aggregate_player_weights_to_games(
    player_weights_df: pd.DataFrame,
    player_game_df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    prob_columns = [column for column in player_weights_df.columns if column.startswith("cluster_") and column.endswith("_prob")]
    join_columns = ["row_id", "wave", "game_id", *REQUIRED_CONFIG_COLUMNS]
    merged = player_weights_df.merge(
        player_game_df[join_columns].drop_duplicates("row_id"),
        on=["row_id", "wave", "game_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(player_weights_df):
        raise ValueError("Game aggregation merge lost player rows")

    config_frame = merged.groupby(["wave", "game_id"], as_index=False)[REQUIRED_CONFIG_COLUMNS].first()
    dist_frame = merged.groupby(["wave", "game_id"], as_index=False)[prob_columns].mean()
    out = config_frame.merge(dist_frame, on=["wave", "game_id"], how="inner", validate="one_to_one")
    if output_path is not None:
        out.to_parquet(output_path, index=False)
    return out


def fit_env_distribution_model(
    learn_game_df: pd.DataFrame,
    val_game_df: pd.DataFrame,
    model_path: str | Path | None = None,
    learn_output_path: str | Path | None = None,
    val_output_path: str | Path | None = None,
) -> tuple[DirichletEnvRegressor, pd.DataFrame, pd.DataFrame]:
    prob_columns = [column for column in learn_game_df.columns if column.startswith("cluster_") and column.endswith("_prob")]
    if not prob_columns:
        raise ValueError("No cluster probability columns found for environment model training")

    model = DirichletEnvRegressor(feature_columns=list(REQUIRED_CONFIG_COLUMNS))
    model.fit(learn_game_df[REQUIRED_CONFIG_COLUMNS], learn_game_df[prob_columns].to_numpy())

    learn_pred = learn_game_df[["wave", "game_id", *REQUIRED_CONFIG_COLUMNS]].copy()
    val_pred = val_game_df[["wave", "game_id", *REQUIRED_CONFIG_COLUMNS]].copy()
    learn_pred[prob_columns] = model.predict(learn_game_df[REQUIRED_CONFIG_COLUMNS])
    val_pred[prob_columns] = model.predict(val_game_df[REQUIRED_CONFIG_COLUMNS])

    if model_path is not None:
        model.save(model_path)
    if learn_output_path is not None:
        learn_pred.to_parquet(learn_output_path, index=False)
    if val_output_path is not None:
        val_pred.to_parquet(val_output_path, index=False)
    return model, learn_pred, val_pred
