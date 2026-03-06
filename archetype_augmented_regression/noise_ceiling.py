from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from archetype_augmented_regression.modeling import metrics, r2_oos_train_mean


@dataclass
class NoiseCeilingResult:
    n_train_rows: int
    n_test_rows: int
    n_train_unique_configs: int
    n_test_unique_configs: int
    n_overlap_configs: int
    unseen_test_share: float
    train_mean_target: float
    oracle_test_config_mean: dict[str, float]
    train_to_test_config_mean: dict[str, float]

    def as_dict(self) -> dict[str, float | int | dict[str, float]]:
        return {
            "n_train_rows": self.n_train_rows,
            "n_test_rows": self.n_test_rows,
            "n_train_unique_configs": self.n_train_unique_configs,
            "n_test_unique_configs": self.n_test_unique_configs,
            "n_overlap_configs": self.n_overlap_configs,
            "unseen_test_share": self.unseen_test_share,
            "train_mean_target": self.train_mean_target,
            "oracle_test_config_mean": self.oracle_test_config_mean,
            "train_to_test_config_mean": self.train_to_test_config_mean,
        }


def _config_key(frame: pd.DataFrame) -> pd.Series:
    # Use a stable string key across mixed types.
    txt = frame.fillna("NA").astype(str)
    return txt.agg("|".join, axis=1)


def _metrics_with_oos(y: np.ndarray, yhat: np.ndarray, train_mean: float) -> dict[str, float]:
    out = metrics(y, yhat)
    out["r2_oos_train_mean"] = float(r2_oos_train_mean(y, yhat, train_mean))
    return out


def compute_noise_ceiling(
    x_train_cfg: pd.DataFrame,
    y_train: np.ndarray,
    x_test_cfg: pd.DataFrame,
    y_test: np.ndarray,
) -> NoiseCeilingResult:
    key_train = _config_key(x_train_cfg)
    key_test = _config_key(x_test_cfg)
    train_mean = float(np.mean(y_train))

    train_group_mean = pd.DataFrame({"key": key_train, "y": y_train}).groupby("key")["y"].mean()
    test_group_mean = pd.DataFrame({"key": key_test, "y": y_test}).groupby("key")["y"].mean()

    yhat_train_to_test = key_test.map(train_group_mean).fillna(train_mean).to_numpy(dtype=float)
    yhat_oracle_test = key_test.map(test_group_mean).to_numpy(dtype=float)

    overlap = int(len(set(train_group_mean.index) & set(test_group_mean.index)))
    unseen_share = float((~key_test.isin(train_group_mean.index)).mean())

    return NoiseCeilingResult(
        n_train_rows=int(len(y_train)),
        n_test_rows=int(len(y_test)),
        n_train_unique_configs=int(train_group_mean.shape[0]),
        n_test_unique_configs=int(test_group_mean.shape[0]),
        n_overlap_configs=overlap,
        unseen_test_share=unseen_share,
        train_mean_target=train_mean,
        oracle_test_config_mean=_metrics_with_oos(y_test, yhat_oracle_test, train_mean),
        train_to_test_config_mean=_metrics_with_oos(y_test, yhat_train_to_test, train_mean),
    )


@dataclass
class SamplingNoiseCeilingResult:
    group_col: str
    n_rows: int
    n_groups: int
    n_groups_with_replicates: int
    mean_group_size: float
    median_group_size: float
    mse_floor: float
    rmse_floor: float
    r2_ceiling_test_mean: float
    r2_ceiling_train_mean: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "group_col": self.group_col,
            "n_rows": self.n_rows,
            "n_groups": self.n_groups,
            "n_groups_with_replicates": self.n_groups_with_replicates,
            "mean_group_size": self.mean_group_size,
            "median_group_size": self.median_group_size,
            "mse_floor": self.mse_floor,
            "rmse_floor": self.rmse_floor,
            "r2_ceiling_test_mean": self.r2_ceiling_test_mean,
            "r2_ceiling_train_mean": self.r2_ceiling_train_mean,
        }


def compute_sampling_noise_ceiling(
    *,
    y_obs: np.ndarray,
    group_ids: pd.Series,
    train_mean_target: float,
    group_col: str,
) -> SamplingNoiseCeilingResult:
    """Estimate sampling-noise floor for predicting group means.

    For groups x with observed replicates y_{x,1..n_x}:
      mse_floor = mean_x( s_x^2 / n_x )
      rmse_floor = sqrt(mse_floor)
    where s_x^2 is the sample variance within group x.

    We also report implied R2 ceilings against observed group means:
      r2_ceiling_test_mean = 1 - mse_floor / mean((ybar_x - mean(ybar))^2)
      r2_ceiling_train_mean = 1 - mse_floor / mean((ybar_x - train_mean)^2)
    """
    work = pd.DataFrame({"group": group_ids.astype(str), "y": np.asarray(y_obs, dtype=float)})
    grouped = work.groupby("group")["y"]
    n = grouped.size().astype(float)
    ybar = grouped.mean().astype(float)
    # ddof=1 sample variance; singleton groups become NaN, treated as 0 contribution.
    s2 = grouped.var(ddof=1).fillna(0.0).astype(float)

    mse_floor = float(np.mean((s2 / n).to_numpy(dtype=float))) if len(n) else float("nan")
    rmse_floor = float(np.sqrt(mse_floor)) if np.isfinite(mse_floor) and mse_floor >= 0 else float("nan")

    ybar_arr = ybar.to_numpy(dtype=float)
    if len(ybar_arr):
        den_test = float(np.mean((ybar_arr - float(np.mean(ybar_arr))) ** 2))
        den_train = float(np.mean((ybar_arr - float(train_mean_target)) ** 2))
        r2_test = float("nan") if den_test <= 0 else float(1.0 - mse_floor / den_test)
        r2_train = float("nan") if den_train <= 0 else float(1.0 - mse_floor / den_train)
    else:
        r2_test = float("nan")
        r2_train = float("nan")

    return SamplingNoiseCeilingResult(
        group_col=group_col,
        n_rows=int(work.shape[0]),
        n_groups=int(ybar.shape[0]),
        n_groups_with_replicates=int((n > 1).sum()),
        mean_group_size=float(n.mean()) if len(n) else float("nan"),
        median_group_size=float(n.median()) if len(n) else float("nan"),
        mse_floor=mse_floor,
        rmse_floor=rmse_floor,
        r2_ceiling_test_mean=r2_test,
        r2_ceiling_train_mean=r2_train,
    )
