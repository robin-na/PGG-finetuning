from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from simulation_statistical.archetype_distribution_embedding.models.soft_cluster_gmm import SoftClusterGMM
from simulation_statistical.archetype_distribution_embedding.utils.io_utils import write_text


def evaluate_clustering(
    model: SoftClusterGMM,
    diagnostics_df: pd.DataFrame,
    learn_embedding_df: pd.DataFrame,
    learn_player_df: pd.DataFrame,
    player_weights_df: pd.DataFrame,
    summary_path: str | Path,
    report_path: str | Path | None = None,
) -> pd.DataFrame:
    feature_columns = [column for column in learn_embedding_df.columns if column.startswith("embed_")]
    prob_columns = [column for column in player_weights_df.columns if column.startswith("cluster_") and column.endswith("_prob")]
    X = learn_embedding_df[feature_columns].to_numpy()
    probs = player_weights_df[prob_columns].to_numpy()
    cluster_mass = probs.mean(axis=0)
    entropy = -(probs * np.log(probs + 1e-8)).sum(axis=1)

    summary = pd.DataFrame(
        [
            {
                "selected_n_clusters": int(model.n_components),
                "avg_assignment_entropy": float(entropy.mean()),
                "cluster_mass_min": float(cluster_mass.min()),
                "cluster_mass_max": float(cluster_mass.max()),
                "cluster_mass_std": float(cluster_mass.std()),
                "candidate_grid_size": int(len(diagnostics_df)),
            }
        ]
    )
    summary.to_csv(summary_path, index=False)

    if report_path is not None:
        merged = learn_embedding_df.merge(
            learn_player_df[["row_id", "archetype_text_clean"]],
            on="row_id",
            how="inner",
            validate="one_to_one",
        ).merge(
            player_weights_df[["row_id", *prob_columns]],
            on="row_id",
            how="inner",
            validate="one_to_one",
        )
        lines = ["# Clustering report", ""]
        for cluster_index, center in enumerate(model.means_, start=1):
            distances = np.linalg.norm(X - center, axis=1)
            nearest_idx = np.argsort(distances)[:3]
            lines.append(f"## Cluster {cluster_index}")
            lines.append("")
            for idx in nearest_idx:
                row = merged.iloc[idx]
                snippet = str(row["archetype_text_clean"]).replace("\n", " ")[:240]
                lines.append(
                    f"- {row['row_id']} | prob={row[f'cluster_{cluster_index}_prob']:.4f} | {snippet}"
                )
            lines.append("")
        write_text(report_path, "\n".join(lines).strip() + "\n")
    return summary
