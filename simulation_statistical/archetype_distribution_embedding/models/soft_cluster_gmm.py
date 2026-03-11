from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

from simulation_statistical.archetype_distribution_embedding.utils.constants import EPSILON


def cluster_probability_columns(n_clusters: int) -> list[str]:
    return [f"cluster_{index + 1}_prob" for index in range(n_clusters)]


class SoftClusterGMM:
    def __init__(
        self,
        n_components: int,
        covariance_type: str = "full",
        random_state: int = 0,
        reg_covar: float = 1e-6,
        max_iter: int = 500,
        n_init: int = 5,
    ) -> None:
        self.n_components = n_components
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
        )

    def fit(self, X: np.ndarray) -> "SoftClusterGMM":
        self.model.fit(X)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def diagnostics(self, X: np.ndarray) -> dict[str, float]:
        probs = self.predict_proba(X)
        cluster_mass = probs.mean(axis=0)
        entropy = -np.sum(probs * np.log(probs + EPSILON), axis=1)
        return {
            "n_clusters": int(self.n_components),
            "train_log_likelihood": float(self.model.score(X)),
            "bic": float(self.model.bic(X)),
            "aic": float(self.model.aic(X)),
            "avg_assignment_entropy": float(entropy.mean()),
            "cluster_mass_min": float(cluster_mass.min()),
            "cluster_mass_max": float(cluster_mass.max()),
            "cluster_mass_std": float(cluster_mass.std()),
        }

    @property
    def means_(self) -> np.ndarray:
        return self.model.means_

    def save(self, path: str | Path) -> None:
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "SoftClusterGMM":
        return joblib.load(Path(path))
