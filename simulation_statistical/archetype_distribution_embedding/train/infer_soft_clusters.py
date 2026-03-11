from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from simulation_statistical.archetype_distribution_embedding.models.soft_cluster_gmm import (
    SoftClusterGMM,
    cluster_probability_columns,
)


def infer_player_cluster_weights(
    model: SoftClusterGMM,
    embedding_df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    feature_columns = [column for column in embedding_df.columns if column.startswith("embed_")]
    base_columns = ["row_id", "wave", "game_id", "player_id"]
    X = embedding_df[feature_columns].to_numpy()
    probs = model.predict_proba(X)
    prob_columns = cluster_probability_columns(probs.shape[1])
    out = embedding_df[base_columns].copy()
    out[prob_columns] = probs
    out["assignment_entropy"] = -(probs * np.log(probs + 1e-8)).sum(axis=1)
    if output_path is not None:
        out.to_parquet(output_path, index=False)
    return out
