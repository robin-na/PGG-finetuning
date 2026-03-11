from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + 1e-12) / (m + 1e-12)), axis=1)
    kl_qm = np.sum(q * np.log((q + 1e-12) / (m + 1e-12)), axis=1)
    return 0.5 * (kl_pm + kl_qm)


def evaluate_env_predictions(
    observed_game_df: pd.DataFrame,
    predicted_game_df: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    prob_columns = [column for column in observed_game_df.columns if column.startswith("cluster_") and column.endswith("_prob")]
    merged = observed_game_df[["wave", "game_id", *prob_columns]].merge(
        predicted_game_df[["wave", "game_id", *prob_columns]],
        on=["wave", "game_id"],
        how="inner",
        suffixes=("_obs", "_pred"),
        validate="one_to_one",
    )

    rows = []
    for wave, group in merged.groupby("wave", sort=False):
        obs = group[[f"{column}_obs" for column in prob_columns]].to_numpy()
        pred = group[[f"{column}_pred" for column in prob_columns]].to_numpy()
        mae = np.abs(obs - pred).mean(axis=0)
        correlations = []
        for index in range(obs.shape[1]):
            if np.std(obs[:, index]) == 0 or np.std(pred[:, index]) == 0:
                correlations.append(np.nan)
            else:
                correlations.append(float(np.corrcoef(obs[:, index], pred[:, index])[0, 1]))

        row = {
            "wave": wave,
            "n_games": int(len(group)),
            "mean_cluster_mae": float(mae.mean()),
            "avg_l1_distance": float(np.abs(obs - pred).sum(axis=1).mean()),
            "mean_js_divergence": float(_js_divergence(obs, pred).mean()),
            "mean_cluster_correlation": float(np.nanmean(correlations)),
        }
        for index, column in enumerate(prob_columns, start=1):
            row[f"mae_cluster_{index}"] = float(mae[index - 1])
            row[f"corr_cluster_{index}"] = correlations[index - 1]
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out
