from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from simulation_statistical.archetype_distribution_embedding.utils.io_utils import read_jsonl, save_pickle


def load_embedding_output(path: str | Path) -> pd.DataFrame:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Embedding output is empty: {path}")
    df = pd.DataFrame(rows)
    missing = {"row_id", "embedding"} - set(df.columns)
    if missing:
        raise ValueError(f"Embedding output missing required fields {sorted(missing)}: {path}")
    df = df.copy()
    df["embedding"] = df["embedding"].apply(lambda row: np.asarray(row, dtype=float))
    return df


def _embedding_table_to_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.vstack(df["embedding"].to_numpy())


def _feature_columns(prefix: str, width: int) -> list[str]:
    return [f"{prefix}_{index:03d}" for index in range(width)]


def _prepare_embedding_join_frame(
    player_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    if int(embedding_df.duplicated("row_id").sum()):
        raise ValueError(f"{label} embedding output contains duplicate row_id values")

    base = player_df[["row_id", "wave", "game_id", "player_id"]].copy()
    merged = base.merge(
        embedding_df,
        on="row_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_embed"),
    )
    if len(merged) != len(base):
        raise ValueError(
            f"{label} embedding join lost rows: expected {len(base)}, got {len(merged)}"
        )

    for column in ("wave", "game_id", "player_id"):
        embed_column = f"{column}_embed"
        if embed_column not in merged.columns:
            continue
        mismatch = merged[embed_column].notna() & (merged[column].astype(str) != merged[embed_column].astype(str))
        if bool(mismatch.any()):
            raise ValueError(
                f"{label} embedding metadata mismatch for column {column}: "
                f"{int(mismatch.sum())} rows disagree with canonical player table"
            )
        merged = merged.drop(columns=[embed_column])

    drop_columns = [column for column in ("text",) if column in merged.columns]
    if drop_columns:
        merged = merged.drop(columns=drop_columns)
    return merged


def join_and_reduce_embeddings(
    learn_player_df: pd.DataFrame,
    val_player_df: pd.DataFrame,
    learn_embedding_df: pd.DataFrame,
    val_embedding_df: pd.DataFrame,
    n_components: int = 50,
    standardize: bool = True,
    pca_model_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base_columns = ["row_id", "wave", "game_id", "player_id"]
    learn_joined = _prepare_embedding_join_frame(
        player_df=learn_player_df,
        embedding_df=learn_embedding_df,
        label="Learn",
    )
    val_joined = _prepare_embedding_join_frame(
        player_df=val_player_df,
        embedding_df=val_embedding_df,
        label="Validation",
    )

    X_learn = _embedding_table_to_matrix(learn_joined)
    X_val = _embedding_table_to_matrix(val_joined)
    if X_learn.shape[1] != X_val.shape[1]:
        raise ValueError(
            f"Embedding width mismatch: learn has {X_learn.shape[1]} dims, val has {X_val.shape[1]} dims"
        )

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_learn = scaler.fit_transform(X_learn)
        X_val = scaler.transform(X_val)

    pca = None
    actual_components = min(int(n_components), X_learn.shape[0], X_learn.shape[1])
    if actual_components > 0 and actual_components < X_learn.shape[1]:
        pca = PCA(n_components=actual_components, random_state=0)
        X_learn = pca.fit_transform(X_learn)
        X_val = pca.transform(X_val)
    else:
        actual_components = X_learn.shape[1]

    feature_cols = _feature_columns("embed", actual_components)
    learn_out = learn_joined[base_columns].copy()
    val_out = val_joined[base_columns].copy()
    learn_out[feature_cols] = X_learn
    val_out[feature_cols] = X_val

    transform_bundle = {
        "standardize": standardize,
        "scaler": scaler,
        "pca": pca,
        "input_dim": int(_embedding_table_to_matrix(learn_joined).shape[1]),
        "output_dim": int(actual_components),
        "feature_columns": feature_cols,
    }
    if pca_model_path is not None:
        save_pickle(transform_bundle, pca_model_path)
    return learn_out, val_out, transform_bundle
