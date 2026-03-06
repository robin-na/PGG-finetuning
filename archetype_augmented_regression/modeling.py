from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    frame: pd.DataFrame, columns: list[str]
) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]
    categorical_cols = [c for c in columns if c not in numeric_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def r2_oos_train_mean(y_true: np.ndarray, y_pred: np.ndarray, train_mean: float) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_base = float(np.sum((y_true - train_mean) ** 2))
    if ss_base <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_base
