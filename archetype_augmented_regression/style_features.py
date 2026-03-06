from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from archetype_augmented_regression.modeling import build_preprocessor


def cluster_distribution_by_game(df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    distribution = (
        df.groupby(["experiment", "cluster"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(n_clusters), fill_value=0)
    )
    distribution = distribution.div(distribution.sum(axis=1), axis=0).fillna(0.0)
    distribution.columns = [f"style_cluster_{idx}_share" for idx in distribution.columns]
    return distribution


def normalize_share_matrix(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = np.clip(arr, 0.0, None)
    row_sums = arr.sum(axis=1, keepdims=True)
    n_clusters = arr.shape[1]
    zero_mask = row_sums[:, 0] <= 1e-12
    if np.any(zero_mask):
        arr[zero_mask] = 1.0 / max(1, n_clusters)
        row_sums = arr.sum(axis=1, keepdims=True)
    return arr / row_sums


def fit_predict_synthetic_style(
    x_learn_cfg: pd.DataFrame,
    x_val_cfg: pd.DataFrame,
    learn_style_true: pd.DataFrame,
    style_cols: list[str],
    style_ridge_alpha: float,
    style_oof_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_style_learn = learn_style_true[style_cols].to_numpy(dtype=float)
    n_rows = len(x_learn_cfg)

    n_splits = min(int(style_oof_folds), n_rows)
    if n_splits < 2:
        n_splits = 2
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_oof = np.zeros_like(y_style_learn, dtype=float)
    for train_idx, hold_idx in splitter.split(np.arange(n_rows)):
        fold_pre, _, _ = build_preprocessor(x_learn_cfg.iloc[train_idx], list(x_learn_cfg.columns))
        fold_model = Pipeline(
            steps=[("pre", fold_pre), ("model", Ridge(alpha=style_ridge_alpha, random_state=42))]
        )
        fold_model.fit(x_learn_cfg.iloc[train_idx], y_style_learn[train_idx])
        y_oof[hold_idx] = fold_model.predict(x_learn_cfg.iloc[hold_idx])

    full_pre, _, _ = build_preprocessor(x_learn_cfg, list(x_learn_cfg.columns))
    full_model = Pipeline(
        steps=[("pre", full_pre), ("model", Ridge(alpha=style_ridge_alpha, random_state=42))]
    )
    full_model.fit(x_learn_cfg, y_style_learn)
    y_val_pred = full_model.predict(x_val_cfg)

    return normalize_share_matrix(y_oof), normalize_share_matrix(y_val_pred)

