from __future__ import annotations

from pathlib import Path

import pandas as pd

from simulation_statistical.archetype_distribution_embedding.models.soft_cluster_gmm import SoftClusterGMM
from simulation_statistical.archetype_distribution_embedding.utils.constants import GMM_CLUSTER_GRID_DEFAULT


def fit_soft_cluster_grid(
    learn_embedding_df: pd.DataFrame,
    cluster_grid: list[int] | None = None,
    model_path: str | Path | None = None,
    diagnostics_path: str | Path | None = None,
    random_state: int = 0,
) -> tuple[SoftClusterGMM, pd.DataFrame]:
    feature_columns = [column for column in learn_embedding_df.columns if column.startswith("embed_")]
    if not feature_columns:
        raise ValueError("No reduced embedding feature columns found for GMM training")

    X = learn_embedding_df[feature_columns].to_numpy()
    candidates = cluster_grid or list(GMM_CLUSTER_GRID_DEFAULT)
    diagnostics_rows = []
    best_model: SoftClusterGMM | None = None
    best_bic = float("inf")

    for n_clusters in candidates:
        if n_clusters >= len(X):
            continue
        model = SoftClusterGMM(n_components=n_clusters, random_state=random_state)
        model.fit(X)
        metrics = model.diagnostics(X)
        diagnostics_rows.append(metrics)
        if metrics["bic"] < best_bic:
            best_bic = metrics["bic"]
            best_model = model

    if best_model is None:
        raise ValueError("No valid GMM candidate could be fit. Check cluster grid against training row count.")

    diagnostics_df = pd.DataFrame(diagnostics_rows).sort_values("n_clusters").reset_index(drop=True)
    diagnostics_df["selected_model"] = diagnostics_df["n_clusters"] == best_model.n_components

    if model_path is not None:
        best_model.save(model_path)
    if diagnostics_path is not None:
        diagnostics_df.to_csv(diagnostics_path, index=False)
    return best_model, diagnostics_df
