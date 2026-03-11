from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammaln
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from simulation_statistical.archetype_distribution_embedding.utils.constants import (
    ENV_BOOLEAN_COLUMNS,
    ENV_CATEGORICAL_COLUMNS,
    ENV_NUMERIC_COLUMNS,
    EPSILON,
    PUNISHMENT_FEATURE_COLUMNS,
    REQUIRED_CONFIG_COLUMNS,
    REWARD_FEATURE_COLUMNS,
)


def _to_bool_int(series: pd.Series) -> pd.Series:
    return series.map(lambda value: int(str(value).strip().lower() in {"1", "true", "t", "yes"})).astype(float)


def _safe_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    positive_mask = x >= 0
    out = np.empty_like(x, dtype=float)
    out[positive_mask] = 1.0 / (1.0 + np.exp(-x[positive_mask]))
    exp_x = np.exp(x[~positive_mask])
    out[~positive_mask] = exp_x / (1.0 + exp_x)
    return out


class DirichletEnvRegressor:
    def __init__(
        self,
        feature_columns: list[str] | None = None,
        l2_penalty: float = 1e-3,
        max_iter: int = 500,
        epsilon: float = EPSILON,
    ) -> None:
        self.feature_columns = feature_columns or list(REQUIRED_CONFIG_COLUMNS)
        self.l2_penalty = l2_penalty
        self.max_iter = max_iter
        self.epsilon = epsilon

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing environment feature columns: {sorted(missing)}")

        out = df[self.feature_columns].copy()
        for column in ENV_BOOLEAN_COLUMNS:
            out[column] = _to_bool_int(out[column])

        for column in ENV_NUMERIC_COLUMNS:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

        out["CONFIG_punishmentCost"] = np.where(
            out["CONFIG_punishmentExists"] > 0.5,
            out["CONFIG_punishmentCost"],
            0.0,
        )
        out["CONFIG_rewardCost"] = np.where(
            out["CONFIG_rewardExists"] > 0.5,
            out["CONFIG_rewardCost"],
            0.0,
        )

        for column in ENV_CATEGORICAL_COLUMNS:
            out[column] = out[column].astype(str)

        out["CONFIG_punishmentTech"] = np.where(
            out["CONFIG_punishmentExists"] > 0.5,
            out["CONFIG_punishmentTech"],
            "inactive",
        )
        out["CONFIG_rewardTech"] = np.where(
            out["CONFIG_rewardExists"] > 0.5,
            out["CONFIG_rewardTech"],
            "inactive",
        )
        return out

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_columns = [column for column in ENV_NUMERIC_COLUMNS + ENV_BOOLEAN_COLUMNS if column in self.feature_columns]
        categorical_columns = [column for column in ENV_CATEGORICAL_COLUMNS if column in self.feature_columns]
        return ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), numeric_columns),
                ("categorical", _safe_one_hot_encoder(), categorical_columns),
            ],
            remainder="drop",
        )

    def _transform_features(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        prepared = self._prepare_frame(df)
        if fit:
            self.preprocessor_ = self._build_preprocessor()
            transformed = self.preprocessor_.fit_transform(prepared)
            self.feature_names_out_ = ["intercept", *self.preprocessor_.get_feature_names_out().tolist()]
        else:
            transformed = self.preprocessor_.transform(prepared)
        transformed = np.asarray(transformed, dtype=float)
        intercept = np.ones((len(transformed), 1), dtype=float)
        return np.hstack([intercept, transformed])

    def _loss_and_grad(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        coef = params.reshape(X.shape[1], y.shape[1])
        eta = X @ coef
        alpha = _softplus(eta) + self.epsilon
        alpha0 = alpha.sum(axis=1, keepdims=True)
        log_y = np.log(y)
        log_likelihood = (
            gammaln(alpha0).sum()
            - gammaln(alpha).sum()
            + ((alpha - 1.0) * log_y).sum()
        )

        grad_alpha = digamma(alpha0) - digamma(alpha) + log_y
        grad_eta = _sigmoid(eta) * grad_alpha
        grad_coef = X.T @ grad_eta

        loss = -log_likelihood / len(X)
        grad = -grad_coef / len(X)
        if self.l2_penalty:
            loss += 0.5 * self.l2_penalty * float(np.sum(coef[1:] ** 2))
            grad[1:] += self.l2_penalty * coef[1:]
        return loss, grad.ravel()

    def fit(self, X_df: pd.DataFrame, y: np.ndarray) -> "DirichletEnvRegressor":
        y = np.asarray(y, dtype=float)
        y = np.clip(y, self.epsilon, None)
        y = y / y.sum(axis=1, keepdims=True)

        X = self._transform_features(X_df, fit=True)
        initial = np.zeros((X.shape[1], y.shape[1]), dtype=float)
        result = minimize(
            fun=lambda params: self._loss_and_grad(params, X, y),
            x0=initial.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        if not result.success:
            raise RuntimeError(f"Dirichlet regression did not converge: {result.message}")

        self.coef_ = result.x.reshape(X.shape[1], y.shape[1])
        self.n_targets_ = y.shape[1]
        self.n_iter_ = int(result.nit)
        self.final_loss_ = float(result.fun)
        return self

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        X = self._transform_features(X_df, fit=False)
        eta = X @ self.coef_
        alpha = _softplus(eta) + self.epsilon
        return alpha / alpha.sum(axis=1, keepdims=True)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "DirichletEnvRegressor":
        return joblib.load(Path(path))
