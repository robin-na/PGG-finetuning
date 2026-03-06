from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from archetype_augmented_regression.io_utils import (
    coerce_config_col,
    load_jsonl,
    resolve_target,
)
from archetype_augmented_regression.modeling import build_preprocessor
from archetype_augmented_regression.style_features import (
    cluster_distribution_by_game,
    fit_predict_synthetic_style,
)


def _aggregate_with_group(
    x_df: pd.DataFrame,
    y_arr: np.ndarray,
    group_ids: pd.Series,
) -> pd.DataFrame:
    tmp = x_df.copy()
    tmp["__target__"] = np.asarray(y_arr, dtype=float)
    tmp["__group__"] = group_ids.astype(str).to_numpy()

    numeric_cols = [col for col in x_df.columns if pd.api.types.is_numeric_dtype(x_df[col])]
    categorical_cols = [col for col in x_df.columns if col not in numeric_cols]
    agg_map: dict[str, str] = {col: "mean" for col in numeric_cols}
    agg_map.update({col: "first" for col in categorical_cols})
    agg_map["__target__"] = "mean"

    grouped = tmp.groupby("__group__", as_index=False).agg(agg_map)
    return grouped


def _align_grouped(
    grouped: pd.DataFrame,
    group_order: np.ndarray,
    x_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    sub = grouped.set_index("__group__").reindex(group_order)
    x = sub[x_cols].copy()
    y = sub["__target__"].to_numpy(dtype=float)
    return x, y


def _estimator(model: str, ridge_alpha: float):
    if model == "ridge":
        return Ridge(alpha=ridge_alpha, random_state=42)
    if model == "linear":
        return LinearRegression()
    raise ValueError(f"Unsupported model: {model}")


def _choose_best_k(results_df: pd.DataFrame, *, style_source: str, model: str) -> int:
    sub = results_df[
        (results_df["eval_granularity"] == "config_treatment")
        & (results_df["style_source"] == style_source)
        & (results_df["model"] == model)
    ].copy()
    if sub.empty:
        raise ValueError(
            f"No rows for style_source={style_source}, model={model}, eval_granularity=config_treatment"
        )
    row = sub.sort_values(["delta_rmse", "delta_mae"], ascending=[True, True]).iloc[0]
    return int(row["k_clusters"])


def infer_best_k_from_results(results_csv: Path, model: str) -> dict[str, int]:
    results_df = pd.read_csv(results_csv)
    return {
        "oracle": _choose_best_k(results_df, style_source="oracle", model=model),
        "synthetic": _choose_best_k(results_df, style_source="synthetic", model=model),
    }


def fit_config_treatment_predictions(
    *,
    learn_analysis_csv: Path,
    val_analysis_csv: Path,
    learn_persona_jsonl: Path,
    val_persona_jsonl: Path,
    target_requested: str,
    model: str,
    ridge_alpha: float,
    style_ridge_alpha: float,
    style_oof_folds: int,
    best_k_oracle: int,
    best_k_synthetic: int,
    group_col: str = "CONFIG_treatmentName",
    exclude_config_cols: list[str] | None = None,
    max_features: int = 5000,
    min_df: int = 2,
    ngram_max: int = 2,
    valid_col: str = "valid_number_of_starting_players",
) -> dict[str, Any]:
    if exclude_config_cols is None:
        exclude_config_cols = ["CONFIG_configId", "CONFIG_treatmentName"]

    learn_df = pd.read_csv(learn_analysis_csv)
    val_df = pd.read_csv(val_analysis_csv)
    target = resolve_target(target_requested, learn_df, val_df)

    if group_col not in learn_df.columns or group_col not in val_df.columns:
        raise ValueError(f"Missing grouping column: {group_col}")

    if valid_col in learn_df.columns:
        learn_df = learn_df[learn_df[valid_col].fillna(False).astype(bool)].copy()
    if valid_col in val_df.columns:
        val_df = val_df[val_df[valid_col].fillna(False).astype(bool)].copy()

    learn_df["gameId"] = learn_df["gameId"].astype(str)
    val_df["gameId"] = val_df["gameId"].astype(str)

    config_cols = [
        col
        for col in learn_df.columns
        if col.startswith("CONFIG_")
        and col in val_df.columns
        and col not in set(exclude_config_cols)
    ]
    if not config_cols:
        raise ValueError("No CONFIG feature columns found.")

    x_learn_cfg = learn_df[config_cols].copy()
    x_val_cfg = val_df[config_cols].copy()
    for col in config_cols:
        x_learn_cfg[col] = coerce_config_col(x_learn_cfg[col])
        x_val_cfg[col] = coerce_config_col(x_val_cfg[col])

    y_learn = learn_df[target].to_numpy(dtype=float)
    y_val = val_df[target].to_numpy(dtype=float)

    grouped_learn_base = _aggregate_with_group(x_learn_cfg, y_learn, learn_df[group_col])
    grouped_val_base = _aggregate_with_group(x_val_cfg, y_val, val_df[group_col])
    learn_groups = grouped_learn_base["__group__"].to_numpy()
    val_groups = grouped_val_base["__group__"].to_numpy()
    x_learn_base, y_learn_group = _align_grouped(grouped_learn_base, learn_groups, config_cols)
    x_val_base, y_val_group = _align_grouped(grouped_val_base, val_groups, config_cols)
    train_mean_target = float(np.mean(y_learn_group))

    est = _estimator(model, ridge_alpha)
    pre, _, _ = build_preprocessor(x_learn_base, list(x_learn_base.columns))
    baseline_pipe = Pipeline(steps=[("pre", pre), ("model", est)])
    baseline_pipe.fit(x_learn_base, y_learn_group)
    pred_config = baseline_pipe.predict(x_val_base)

    learn_games = set(learn_df["gameId"].astype(str))
    val_games = set(val_df["gameId"].astype(str))
    persona_learn = load_jsonl(learn_persona_jsonl)
    persona_val = load_jsonl(val_persona_jsonl)
    persona_learn = persona_learn[persona_learn["experiment"].isin(learn_games)].copy()
    persona_val = persona_val[persona_val["experiment"].isin(val_games)].copy()

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, ngram_max),
    )
    learn_text = vectorizer.fit_transform(persona_learn["text"].fillna(""))
    val_text = vectorizer.transform(persona_val["text"].fillna(""))

    def _fit_augmented_with_k(k: int, style_source: str) -> np.ndarray:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        learn_clusters = persona_learn.copy()
        val_clusters = persona_val.copy()
        learn_clusters["cluster"] = kmeans.fit_predict(learn_text)
        val_clusters["cluster"] = kmeans.predict(val_text)

        learn_dist = cluster_distribution_by_game(learn_clusters, k)
        val_dist = cluster_distribution_by_game(val_clusters, k)
        learn_oracle = learn_dist.reindex(learn_df["gameId"]).reset_index(drop=True)
        val_oracle = val_dist.reindex(val_df["gameId"]).reset_index(drop=True)
        style_cols = list(learn_oracle.columns)
        for col in style_cols:
            learn_oracle[col] = pd.to_numeric(learn_oracle[col], errors="coerce").fillna(0.0)
            val_oracle[col] = pd.to_numeric(val_oracle[col], errors="coerce").fillna(0.0)

        if style_source == "oracle":
            learn_style = learn_oracle
            val_style = val_oracle
        elif style_source == "synthetic":
            syn_learn_arr, syn_val_arr = fit_predict_synthetic_style(
                x_learn_cfg=x_learn_cfg,
                x_val_cfg=x_val_cfg,
                learn_style_true=learn_oracle,
                style_cols=style_cols,
                style_ridge_alpha=style_ridge_alpha,
                style_oof_folds=style_oof_folds,
            )
            learn_style = pd.DataFrame(syn_learn_arr, columns=style_cols)
            val_style = pd.DataFrame(syn_val_arr, columns=style_cols)
        else:
            raise ValueError(f"Unknown style source: {style_source}")

        x_learn_aug = pd.concat([x_learn_cfg.reset_index(drop=True), learn_style], axis=1)
        x_val_aug = pd.concat([x_val_cfg.reset_index(drop=True), val_style], axis=1)

        grouped_learn = _aggregate_with_group(x_learn_aug, y_learn, learn_df[group_col])
        grouped_val = _aggregate_with_group(x_val_aug, y_val, val_df[group_col])
        aug_cols = list(x_learn_aug.columns)
        x_learn_eval, y_learn_eval = _align_grouped(grouped_learn, learn_groups, aug_cols)
        x_val_eval, _ = _align_grouped(grouped_val, val_groups, aug_cols)

        pre_aug, _, _ = build_preprocessor(x_learn_eval, aug_cols)
        aug_pipe = Pipeline(steps=[("pre", pre_aug), ("model", _estimator(model, ridge_alpha))])
        aug_pipe.fit(x_learn_eval, y_learn_eval)
        return aug_pipe.predict(x_val_eval)

    pred_oracle = _fit_augmented_with_k(best_k_oracle, "oracle")
    pred_synthetic = _fit_augmented_with_k(best_k_synthetic, "synthetic")

    val_stats = (
        val_df[[group_col, target]]
        .assign(**{group_col: val_df[group_col].astype(str)})
        .groupby(group_col)[target]
        .agg(ybar="mean", s2=lambda s: float(np.var(s.to_numpy(dtype=float), ddof=1)) if len(s) > 1 else np.nan, n="count")
    )
    val_stats["s2"] = val_stats["s2"].fillna(0.0).astype(float)
    val_stats["sampling_term"] = val_stats["s2"] / val_stats["n"].astype(float)
    sampling_terms = val_stats.reindex(val_groups)["sampling_term"].fillna(0.0).to_numpy(dtype=float)

    return {
        "target": target,
        "group_col": group_col,
        "groups": val_groups,
        "y_true": y_val_group,
        "train_mean_target": train_mean_target,
        "pred_by_method": {
            "config_only": pred_config,
            "synthetic": pred_synthetic,
            "oracle": pred_oracle,
        },
        "sampling_terms": sampling_terms,
        "best_k": {"oracle": best_k_oracle, "synthetic": best_k_synthetic},
    }
