from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Macro_simulation_eval.analysis.run_analysis import (
    _extract_analysis_csv_from_config,
    _load_run_config,
    _resolve_existing_path,
    compute_sim_game_metrics,
    load_human_game_table,
    parse_binary_series,
    parse_dict_field,
)


LINEAR_CONFIG_FEATURES: List[str] = [
    "CONFIG_playerCount",
    "CONFIG_numRounds",
    "CONFIG_showNRounds",
    "CONFIG_allOrNothing",
    "CONFIG_chat",
    "CONFIG_defaultContribProp",
    "CONFIG_punishmentExists",
    "CONFIG_punishmentCost",
    "CONFIG_punishmentTech",
    "CONFIG_rewardExists",
    "CONFIG_rewardCost",
    "CONFIG_rewardTech",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
    "CONFIG_MPCR",
]

CLUSTER_DISTRIBUTION_FEATURES: List[str] = [f"cluster_{index}_prob" for index in range(1, 7)]

REGRESSION_TARGET_SPECS: List[Tuple[str, str, str]] = [
    ("human_mean_contribution_rate", "mean_contribution_rate", "contribution_rate"),
    ("human_punishment_rate", "punishment_rate", "punishment_rate"),
    ("human_reward_rate", "reward_rate", "reward_rate"),
    ("human_normalized_efficiency", "normalized_efficiency", "normalized_efficiency"),
    ("human_var_players_contrib_rate", "var_players_contrib_rate", "var_players_contrib_rate"),
    ("human_var_players_punishment_rate", "var_players_punishment_rate", "var_players_punishment_rate"),
    ("human_var_players_reward_rate", "var_players_reward_rate", "var_players_reward_rate"),
]

DIRECT_BASELINE_SPECS: List[Dict[str, str]] = [
    {
        "feature_source": "config",
        "model_kind": "linear",
        "run_id": "linear_config_baseline",
        "label": "linear_config",
    },
    {
        "feature_source": "config",
        "model_kind": "ridge",
        "run_id": "ridge_config_baseline",
        "label": "ridge_config",
    },
    {
        "feature_source": "cluster_pred",
        "model_kind": "linear",
        "run_id": "linear_cluster_pred_baseline",
        "label": "linear_cluster_pred",
    },
    {
        "feature_source": "cluster_pred",
        "model_kind": "ridge",
        "run_id": "ridge_cluster_pred_baseline",
        "label": "ridge_cluster_pred",
    },
    {
        "feature_source": "cluster_pred_plus_config",
        "model_kind": "linear",
        "run_id": "linear_cluster_pred_plus_config_baseline",
        "label": "linear_cluster_pred_plus_config",
    },
    {
        "feature_source": "cluster_pred_plus_config",
        "model_kind": "ridge",
        "run_id": "ridge_cluster_pred_plus_config_baseline",
        "label": "ridge_cluster_pred_plus_config",
    },
]


def extract_rounds_csv_from_config(run_id: Optional[str], eval_root: str) -> Optional[Path]:
    if not run_id:
        return None
    run_cfg = _load_run_config(run_id, eval_root)
    run_dir = (Path(eval_root).resolve() / run_id).resolve()
    inputs = run_cfg.get("inputs") if isinstance(run_cfg.get("inputs"), dict) else {}
    args = run_cfg.get("args") if isinstance(run_cfg.get("args"), dict) else {}
    for raw in [inputs.get("rounds_csv"), args.get("rounds_csv")]:
        resolved = _resolve_existing_path(raw, base_dir=run_dir)
        if resolved is not None:
            return resolved
    return None


def resolve_human_analysis_csv(run_id: Optional[str], eval_root: str, human_analysis_csv: Optional[str]) -> Optional[Path]:
    if human_analysis_csv:
        candidate = Path(human_analysis_csv).resolve()
        if candidate.exists():
            return candidate
    if run_id:
        run_cfg = _load_run_config(run_id, eval_root)
        return _extract_analysis_csv_from_config(run_cfg, eval_root=eval_root, run_id=run_id)
    return None


def resolve_full_analysis_csv(analysis_csv: Path) -> Path:
    analysis_csv = analysis_csv.resolve()
    if analysis_csv.name.endswith("_averaged.csv"):
        candidate = analysis_csv.with_name(analysis_csv.name.replace("_averaged.csv", ".csv"))
        if candidate.exists():
            return candidate
    return analysis_csv


def _benchmark_series(df: pd.DataFrame) -> pd.Series:
    if "CONFIG_treatmentName" in df.columns:
        return df["CONFIG_treatmentName"].astype(str)
    if "name" in df.columns:
        return df["name"].astype(str)
    return df["gameId"].astype(str)


def _coerce_numeric_feature(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    if np.issubdtype(series.dtype, np.number):
        return pd.to_numeric(series, errors="coerce")
    normalized = (
        series.astype(str)
        .str.strip()
        .replace(
            {
                "True": "1",
                "False": "0",
                "true": "1",
                "false": "0",
                "nan": np.nan,
                "None": np.nan,
                "": np.nan,
            }
        )
    )
    return pd.to_numeric(normalized, errors="coerce")


def _prepare_feature_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    missing = [column for column in feature_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required CONFIG columns for linear baseline: {missing}")
    out = pd.DataFrame(index=df.index)
    for column in feature_cols:
        out[column] = _coerce_numeric_feature(df[column])
    return out


def _load_config_slice(analysis_csv: Path) -> pd.DataFrame:
    cfg = pd.read_csv(analysis_csv).drop_duplicates(subset=["gameId"], keep="first").copy()
    cfg["gameId"] = cfg["gameId"].astype(str)
    cfg["benchmark_id"] = _benchmark_series(cfg)
    for column, default in (
        ("CONFIG_endowment", 20.0),
        ("CONFIG_punishmentExists", False),
        ("CONFIG_rewardExists", False),
    ):
        if column not in cfg.columns:
            cfg[column] = default
    keep = [
        "gameId",
        "benchmark_id",
        "CONFIG_endowment",
        "CONFIG_punishmentExists",
        "CONFIG_rewardExists",
    ]
    return cfg[keep].copy()


def _cluster_wave_suffix_from_analysis_csv(analysis_csv: Path) -> str:
    name = resolve_full_analysis_csv(analysis_csv).name
    if "_learn" in name:
        return "learn"
    if "_val" in name:
        return "val"
    raise ValueError(f"Could not infer wave from analysis CSV name: {analysis_csv}")


def _cluster_distribution_artifact_path(analysis_csv: Path, *, predicted: bool) -> Path:
    wave = _cluster_wave_suffix_from_analysis_csv(analysis_csv)
    stem = "predicted_game_cluster_distribution" if predicted else "game_cluster_distribution"
    return (
        Path(__file__).resolve().parents[2]
        / "archetype_distribution_embedding"
        / "artifacts"
        / "outputs"
        / f"{stem}_{wave}.parquet"
    ).resolve()


def build_benchmark_cluster_feature_table(
    analysis_csv: Path,
    *,
    predicted: bool = True,
) -> pd.DataFrame:
    parquet_path = _cluster_distribution_artifact_path(analysis_csv, predicted=predicted)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Cluster distribution artifact not found: {parquet_path}")
    cluster_df = pd.read_parquet(parquet_path).copy()
    if "game_id" in cluster_df.columns:
        cluster_df = cluster_df.rename(columns={"game_id": "gameId"})
    cluster_df["gameId"] = cluster_df["gameId"].astype(str)

    analysis = pd.read_csv(resolve_full_analysis_csv(analysis_csv), usecols=["gameId", "CONFIG_treatmentName"]).copy()
    analysis["gameId"] = analysis["gameId"].astype(str)
    analysis["benchmark_id"] = analysis["CONFIG_treatmentName"].astype(str)
    analysis = analysis[["gameId", "benchmark_id"]].drop_duplicates(subset=["gameId"], keep="first")

    merged = cluster_df.merge(analysis, on="gameId", how="inner", validate="one_to_one")
    return (
        merged.groupby("benchmark_id", as_index=False)[CLUSTER_DISTRIBUTION_FEATURES]
        .mean()
        .copy()
    )


def _augment_benchmark_table_with_feature_source(
    benchmark_table: pd.DataFrame,
    analysis_csv: Path,
    *,
    feature_source: str,
) -> Tuple[pd.DataFrame, List[str]]:
    out = benchmark_table.copy()
    if feature_source == "config":
        return out, list(LINEAR_CONFIG_FEATURES)
    if feature_source in {"cluster_pred", "cluster_pred_plus_config", "cluster_oracle", "cluster_oracle_plus_config"}:
        predicted = "oracle" not in feature_source
        cluster_features = build_benchmark_cluster_feature_table(analysis_csv, predicted=predicted)
        out = out.merge(cluster_features, on="benchmark_id", how="inner", validate="one_to_one")
        feature_cols = list(CLUSTER_DISTRIBUTION_FEATURES)
        if feature_source.endswith("_plus_config"):
            feature_cols.extend(LINEAR_CONFIG_FEATURES)
        return out, feature_cols
    raise ValueError(f"Unsupported feature_source: {feature_source}")


def _build_regression_pipeline(model_kind: str) -> Pipeline:
    if model_kind == "linear":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("linear_regression", LinearRegression()),
            ]
        )
    if model_kind == "ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 25))),
            ]
        )
    raise ValueError(f"Unsupported model_kind: {model_kind}")


def _build_rate_rows(
    rows_csv: Path,
    analysis_csv: Path,
    *,
    contribution_col: str,
    punished_col: str,
    rewarded_col: str,
) -> pd.DataFrame:
    cfg = _load_config_slice(analysis_csv)
    rows = pd.read_csv(rows_csv).copy()
    rows["gameId"] = rows["gameId"].astype(str)
    rows["playerId"] = rows["playerId"].astype(str)
    rows["contribution"] = pd.to_numeric(rows.get(contribution_col), errors="coerce").fillna(0.0)
    rows["punished_dict"] = rows[punished_col].map(parse_dict_field) if punished_col in rows.columns else [{} for _ in range(len(rows))]
    rows["rewarded_dict"] = rows[rewarded_col].map(parse_dict_field) if rewarded_col in rows.columns else [{} for _ in range(len(rows))]

    merged = rows.merge(cfg, on="gameId", how="left")
    if merged["benchmark_id"].isna().any():
        merged = merged[merged["benchmark_id"].notna()].copy()
    if merged.empty:
        raise ValueError(f"No mapped games remained after joining {rows_csv} to {analysis_csv}.")
    endowment = pd.to_numeric(merged["CONFIG_endowment"], errors="coerce").fillna(20.0)
    punish_exists = parse_binary_series(merged["CONFIG_punishmentExists"])
    reward_exists = parse_binary_series(merged["CONFIG_rewardExists"])
    merged["contrib_rate"] = np.where(endowment != 0, merged["contribution"] / endowment, 0.0)
    merged["punishment_rate"] = np.where(
        punish_exists, merged["punished_dict"].map(lambda value: float(bool(value))), 0.0
    )
    merged["reward_rate"] = np.where(
        reward_exists, merged["rewarded_dict"].map(lambda value: float(bool(value))), 0.0
    )
    return merged[["gameId", "benchmark_id", "playerId", "contrib_rate", "punishment_rate", "reward_rate"]].copy()


def build_human_benchmark_rate_targets(rows_csv: Path, analysis_csv: Path) -> pd.DataFrame:
    rate_rows = _build_rate_rows(
        rows_csv=rows_csv,
        analysis_csv=analysis_csv,
        contribution_col="data.contribution",
        punished_col="data.punished",
        rewarded_col="data.rewarded",
    )
    return (
        rate_rows.groupby("benchmark_id", as_index=False)
        .agg(
            human_mean_contribution_rate=("contrib_rate", "mean"),
            human_punishment_rate=("punishment_rate", "mean"),
            human_reward_rate=("reward_rate", "mean"),
            human_games_in_benchmark=("gameId", "nunique"),
        )
        .copy()
    )


def _build_sim_rate_rows(rows_csv: Path, analysis_csv: Path) -> pd.DataFrame:
    head = pd.read_csv(rows_csv, nrows=1)
    contribution_col = "data.contribution_clamped" if "data.contribution_clamped" in head.columns else "data.contribution"
    return _build_rate_rows(
        rows_csv=rows_csv,
        analysis_csv=analysis_csv,
        contribution_col=contribution_col,
        punished_col="data.punished",
        rewarded_col="data.rewarded",
    )


def _player_variance_summary(rate_rows: pd.DataFrame, prefix: str) -> Tuple[Dict[str, float], pd.DataFrame]:
    player_means = (
        rate_rows.groupby(["benchmark_id", "gameId", "playerId"], as_index=False)
        .agg(
            contrib_rate=("contrib_rate", "mean"),
            punishment_rate=("punishment_rate", "mean"),
            reward_rate=("reward_rate", "mean"),
        )
        .copy()
    )
    game_variances = (
        player_means.groupby(["benchmark_id", "gameId"], as_index=False)
        .agg(
            var_players_contrib_rate=("contrib_rate", lambda s: float(pd.Series(s).var(ddof=0))),
            var_players_punishment_rate=("punishment_rate", lambda s: float(pd.Series(s).var(ddof=0))),
            var_players_reward_rate=("reward_rate", lambda s: float(pd.Series(s).var(ddof=0))),
        )
        .copy()
    )
    benchmark_variances = (
        game_variances.groupby("benchmark_id", as_index=False)
        .agg(
            var_players_contrib_rate=("var_players_contrib_rate", "mean"),
            var_players_punishment_rate=("var_players_punishment_rate", "mean"),
            var_players_reward_rate=("var_players_reward_rate", "mean"),
            benchmark_game_count=("gameId", "nunique"),
        )
        .copy()
    )
    summary = {
        f"mean_var_players_contrib_rate_{prefix}": float(benchmark_variances["var_players_contrib_rate"].mean()),
        f"mean_var_players_punishment_rate_{prefix}": float(benchmark_variances["var_players_punishment_rate"].mean()),
        f"mean_var_players_reward_rate_{prefix}": float(benchmark_variances["var_players_reward_rate"].mean()),
    }
    return summary, benchmark_variances


def build_benchmark_target_table(analysis_csv: Path, rows_csv: Path) -> pd.DataFrame:
    analysis = pd.read_csv(analysis_csv).copy()
    if "gameId" not in analysis.columns:
        raise ValueError(f"{analysis_csv} is missing gameId.")
    analysis["gameId"] = analysis["gameId"].astype(str)
    analysis["benchmark_id"] = _benchmark_series(analysis)

    config_cols = [column for column in LINEAR_CONFIG_FEATURES if column in analysis.columns]
    config_table = (
        analysis.sort_values(["benchmark_id", "gameId"])
        .groupby("benchmark_id", as_index=False)
        .agg(
            gameId=("gameId", "first"),
            benchmark_game_count=("gameId", "nunique"),
            human_normalized_efficiency=("itt_relative_efficiency", "mean"),
            **{column: (column, "first") for column in config_cols},
        )
        .copy()
    )

    benchmark_rates = build_human_benchmark_rate_targets(rows_csv, analysis_csv).copy()
    _, benchmark_player_variances = _player_variance_summary(
        _build_rate_rows(
            rows_csv=rows_csv,
            analysis_csv=analysis_csv,
            contribution_col="data.contribution",
            punished_col="data.punished",
            rewarded_col="data.rewarded",
        ),
        prefix="human",
    )
    benchmark_player_variances = benchmark_player_variances.rename(
        columns={
            "var_players_contrib_rate": "human_var_players_contrib_rate",
            "var_players_punishment_rate": "human_var_players_punishment_rate",
            "var_players_reward_rate": "human_var_players_reward_rate",
            "benchmark_game_count": "benchmark_game_count_from_variance",
        }
    )

    out = config_table.merge(benchmark_rates, on="benchmark_id", how="left")
    out = out.merge(benchmark_player_variances, on="benchmark_id", how="left")
    if "benchmark_game_count_from_variance" in out.columns:
        out["benchmark_game_count"] = (
            pd.to_numeric(out["benchmark_game_count"], errors="coerce")
            .fillna(pd.to_numeric(out["benchmark_game_count_from_variance"], errors="coerce"))
            .astype(int)
        )
        out = out.drop(columns=["benchmark_game_count_from_variance"])
    return out


def build_regression_baseline_predictions(
    *,
    learn_benchmark_table: pd.DataFrame,
    val_benchmark_table: pd.DataFrame,
    feature_cols: Sequence[str],
    model_kind: str,
    prediction_prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_x = _prepare_feature_frame(learn_benchmark_table, feature_cols)
    val_x = _prepare_feature_frame(val_benchmark_table, feature_cols)

    predictions = val_benchmark_table[["benchmark_id"]].copy()
    summary_rows: List[Dict[str, Any]] = []

    for target_col, metric_suffix, metric_name in REGRESSION_TARGET_SPECS:
        if target_col not in learn_benchmark_table.columns or target_col not in val_benchmark_table.columns:
            continue
        train_y = pd.to_numeric(learn_benchmark_table[target_col], errors="coerce")
        val_y = pd.to_numeric(val_benchmark_table[target_col], errors="coerce")
        train_mask = train_y.notna()
        val_mask = val_y.notna()
        if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0:
            continue

        pred_col = f"{prediction_prefix}_{metric_suffix}"
        model = _build_regression_pipeline(model_kind)
        model.fit(train_x.loc[train_mask], train_y.loc[train_mask])
        pred_all = pd.Series(model.predict(val_x), index=val_benchmark_table.index, dtype=float)
        if metric_name in {"contribution_rate", "punishment_rate", "reward_rate"}:
            pred_all = pred_all.clip(lower=0.0, upper=1.0)
        if metric_name.startswith("var_players_"):
            pred_all = pred_all.clip(lower=0.0)
        predictions[pred_col] = pred_all

        pair = pd.DataFrame(
            {
                "actual": val_y.loc[val_mask],
                "predicted": pred_all.loc[val_mask],
            }
        ).dropna()
        summary_rows.append(
            {
                "metric": metric_name,
                "target_col": target_col,
                "prediction_col": pred_col,
                "model_kind": model_kind,
                "prediction_prefix": prediction_prefix,
                "feature_cols": ",".join(feature_cols),
                "n_train": int(train_mask.sum()),
                "n_eval": int(len(pair)),
                "rmse": _rmse(pair["predicted"], pair["actual"]),
                "mae": _mae(pair["predicted"], pair["actual"]),
                "corr": _corr(pair["predicted"], pair["actual"]),
            }
        )

    return predictions, pd.DataFrame(summary_rows)


def build_linear_baseline_predictions(
    *,
    learn_benchmark_table: pd.DataFrame,
    val_benchmark_table: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return build_regression_baseline_predictions(
        learn_benchmark_table=learn_benchmark_table,
        val_benchmark_table=val_benchmark_table,
        feature_cols=feature_cols,
        model_kind="linear",
        prediction_prefix="linear",
    )


def _build_direct_baseline_predictions_from_sources(
    *,
    learn_analysis_csv: Path,
    val_analysis_csv: Path,
    learn_rows_csv: Path,
    val_rows_csv: Path,
    feature_source: str,
    model_kind: str,
    prediction_prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    learn_benchmark_table = build_benchmark_target_table(learn_analysis_csv, learn_rows_csv)
    val_benchmark_table = build_benchmark_target_table(val_analysis_csv, val_rows_csv)
    learn_feature_table, feature_cols = _augment_benchmark_table_with_feature_source(
        learn_benchmark_table,
        learn_analysis_csv,
        feature_source=feature_source,
    )
    val_feature_table, _ = _augment_benchmark_table_with_feature_source(
        val_benchmark_table,
        val_analysis_csv,
        feature_source=feature_source,
    )
    return build_regression_baseline_predictions(
        learn_benchmark_table=learn_feature_table,
        val_benchmark_table=val_feature_table,
        feature_cols=feature_cols,
        model_kind=model_kind,
        prediction_prefix=prediction_prefix,
    )


def build_direct_baseline_game_table(
    *,
    human_analysis_csv: Path,
    human_rows_csv: Path,
    binary_factors: Sequence[str],
    median_factors: Sequence[str],
    feature_source: str,
    model_kind: str,
    run_id: str,
    label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    human_analysis_csv = human_analysis_csv.resolve()
    human_rows_csv = human_rows_csv.resolve()
    human_full_analysis_csv = resolve_full_analysis_csv(human_analysis_csv)
    learn_analysis_csv = human_full_analysis_csv.parent / "df_analysis_learn.csv"
    learn_rows_csv = human_rows_csv.parents[1] / "learning_wave" / "player-rounds.csv"
    if not learn_analysis_csv.exists() or not learn_rows_csv.exists():
        return pd.DataFrame(), pd.DataFrame()

    prediction_prefix = run_id.replace("_baseline", "")
    predictions, summary = _build_direct_baseline_predictions_from_sources(
        learn_analysis_csv=learn_analysis_csv,
        val_analysis_csv=human_full_analysis_csv,
        learn_rows_csv=learn_rows_csv,
        val_rows_csv=human_rows_csv,
        feature_source=feature_source,
        model_kind=model_kind,
        prediction_prefix=prediction_prefix,
    )
    if predictions.empty:
        return pd.DataFrame(), summary

    human_game_table = load_human_game_table(
        human_analysis_csv,
        binary_factors=binary_factors,
        median_factors=median_factors,
    ).copy()
    benchmark_meta = pd.read_csv(human_analysis_csv).copy()
    benchmark_meta["gameId"] = benchmark_meta["gameId"].astype(str)
    benchmark_meta["benchmark_id"] = _benchmark_series(benchmark_meta)
    benchmark_meta = benchmark_meta[["gameId", "benchmark_id"]].drop_duplicates(subset=["gameId"], keep="first")

    baseline_game_table = human_game_table.merge(benchmark_meta, on="gameId", how="left", validate="one_to_one")
    baseline_game_table = baseline_game_table.merge(predictions, on="benchmark_id", how="left", validate="many_to_one")
    baseline_game_table["sim_mean_contribution_rate"] = pd.to_numeric(
        baseline_game_table.get(f"{prediction_prefix}_mean_contribution_rate"),
        errors="coerce",
    )
    baseline_game_table["sim_punishment_rate"] = pd.to_numeric(
        baseline_game_table.get(f"{prediction_prefix}_punishment_rate"),
        errors="coerce",
    )
    baseline_game_table["sim_reward_rate"] = pd.to_numeric(
        baseline_game_table.get(f"{prediction_prefix}_reward_rate"),
        errors="coerce",
    )
    baseline_game_table["sim_normalized_efficiency"] = pd.to_numeric(
        baseline_game_table.get(f"{prediction_prefix}_normalized_efficiency"),
        errors="coerce",
    )
    baseline_game_table["run_id"] = run_id
    baseline_game_table["label"] = label
    if not summary.empty:
        summary = summary.copy()
        summary["run_id"] = run_id
        summary["label"] = label
        summary["feature_source"] = feature_source
    return baseline_game_table, summary


def build_linear_config_baseline_game_table(
    *,
    human_analysis_csv: Path,
    human_rows_csv: Path,
    binary_factors: Sequence[str],
    median_factors: Sequence[str],
    run_id: str = "linear_config_baseline",
    label: str = "linear_config",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return build_direct_baseline_game_table(
        human_analysis_csv=human_analysis_csv,
        human_rows_csv=human_rows_csv,
        binary_factors=binary_factors,
        median_factors=median_factors,
        feature_source="config",
        model_kind="linear",
        run_id=run_id,
        label=label,
    )


def build_direct_baseline_benchmark_table(
    *,
    human_analysis_csv: Path,
    human_rows_csv: Path,
    feature_source: str,
    model_kind: str,
    run_id: str,
    label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    human_analysis_csv = human_analysis_csv.resolve()
    human_rows_csv = human_rows_csv.resolve()
    human_full_analysis_csv = resolve_full_analysis_csv(human_analysis_csv)
    learn_analysis_csv = human_full_analysis_csv.parent / "df_analysis_learn.csv"
    learn_rows_csv = human_rows_csv.parents[1] / "learning_wave" / "player-rounds.csv"
    if not learn_analysis_csv.exists() or not learn_rows_csv.exists():
        return pd.DataFrame(), pd.DataFrame()

    prediction_prefix = run_id.replace("_baseline", "")
    predictions, summary = _build_direct_baseline_predictions_from_sources(
        learn_analysis_csv=learn_analysis_csv,
        val_analysis_csv=human_full_analysis_csv,
        learn_rows_csv=learn_rows_csv,
        val_rows_csv=human_rows_csv,
        feature_source=feature_source,
        model_kind=model_kind,
        prediction_prefix=prediction_prefix,
    )
    if predictions.empty:
        return pd.DataFrame(), summary

    baseline_benchmark_table = build_benchmark_target_table(human_full_analysis_csv, human_rows_csv).merge(
        predictions,
        on="benchmark_id",
        how="left",
        validate="one_to_one",
    )
    baseline_benchmark_table["sim_mean_contribution_rate"] = pd.to_numeric(
        baseline_benchmark_table.get(f"{prediction_prefix}_mean_contribution_rate"),
        errors="coerce",
    )
    baseline_benchmark_table["sim_punishment_rate"] = pd.to_numeric(
        baseline_benchmark_table.get(f"{prediction_prefix}_punishment_rate"),
        errors="coerce",
    )
    baseline_benchmark_table["sim_reward_rate"] = pd.to_numeric(
        baseline_benchmark_table.get(f"{prediction_prefix}_reward_rate"),
        errors="coerce",
    )
    baseline_benchmark_table["sim_normalized_efficiency"] = pd.to_numeric(
        baseline_benchmark_table.get(f"{prediction_prefix}_normalized_efficiency"),
        errors="coerce",
    )
    baseline_benchmark_table["run_id"] = run_id
    baseline_benchmark_table["label"] = label
    baseline_benchmark_table["sim_games_in_benchmark"] = pd.to_numeric(
        baseline_benchmark_table.get("benchmark_game_count"),
        errors="coerce",
    )
    if not summary.empty:
        summary = summary.copy()
        summary["run_id"] = run_id
        summary["label"] = label
        summary["feature_source"] = feature_source
    return baseline_benchmark_table, summary


def build_sim_benchmark_game_table(
    *,
    sim_rows_csv: Path,
    human_analysis_csv: Path,
    human_rows_csv: Path,
    binary_factors: Sequence[str],
    median_factors: Sequence[str],
    run_id: str,
    label: str,
) -> pd.DataFrame:
    human_full_analysis_csv = resolve_full_analysis_csv(human_analysis_csv)
    human_game_table = load_human_game_table(
        human_full_analysis_csv,
        binary_factors=binary_factors,
        median_factors=median_factors,
    )
    sim_game_table = compute_sim_game_metrics(sim_rows_csv, human_cfg=human_game_table).copy()

    benchmark_meta = pd.read_csv(human_full_analysis_csv).copy()
    benchmark_meta["gameId"] = benchmark_meta["gameId"].astype(str)
    benchmark_meta["benchmark_id"] = _benchmark_series(benchmark_meta)
    benchmark_meta = benchmark_meta[["gameId", "benchmark_id"]].drop_duplicates(subset=["gameId"], keep="first")

    sim_game_table = sim_game_table.merge(benchmark_meta, on="gameId", how="left", validate="one_to_one")
    sim_game_table = sim_game_table[sim_game_table["benchmark_id"].notna()].copy()
    if sim_game_table.empty:
        return pd.DataFrame()

    sim_benchmark_table = (
        sim_game_table.groupby("benchmark_id", as_index=False)
        .agg(
            sim_mean_contribution_rate=("sim_mean_contribution_rate", "mean"),
            sim_punishment_rate=("sim_punishment_rate", "mean"),
            sim_reward_rate=("sim_reward_rate", "mean"),
            sim_normalized_efficiency=("sim_normalized_efficiency", "mean"),
            sim_games_in_benchmark=("gameId", "nunique"),
        )
        .copy()
    )

    benchmark_targets = build_benchmark_target_table(human_full_analysis_csv, human_rows_csv).copy()
    merged = benchmark_targets.merge(sim_benchmark_table, on="benchmark_id", how="inner", validate="one_to_one")
    merged["run_id"] = run_id
    merged["label"] = label
    return merged


def build_linear_config_baseline_benchmark_table(
    *,
    human_analysis_csv: Path,
    human_rows_csv: Path,
    run_id: str = "linear_config_baseline",
    label: str = "linear_config",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return build_direct_baseline_benchmark_table(
        human_analysis_csv=human_analysis_csv,
        human_rows_csv=human_rows_csv,
        feature_source="config",
        model_kind="linear",
        run_id=run_id,
        label=label,
    )


def build_human_game_macro_output_table(analysis_csv: Path, rows_csv: Path) -> pd.DataFrame:
    analysis_csv = resolve_full_analysis_csv(analysis_csv)
    analysis = pd.read_csv(analysis_csv).copy()
    if "gameId" not in analysis.columns:
        raise ValueError(f"{analysis_csv} is missing gameId.")
    analysis["gameId"] = analysis["gameId"].astype(str)
    analysis["benchmark_id"] = _benchmark_series(analysis)
    game_efficiency = (
        analysis[["benchmark_id", "gameId", "itt_relative_efficiency"]]
        .drop_duplicates(subset=["gameId"], keep="first")
        .rename(columns={"itt_relative_efficiency": "normalized_efficiency"})
        .copy()
    )

    rate_rows = _build_rate_rows(
        rows_csv=rows_csv,
        analysis_csv=analysis_csv,
        contribution_col="data.contribution",
        punished_col="data.punished",
        rewarded_col="data.rewarded",
    )
    game_rates = (
        rate_rows.groupby(["benchmark_id", "gameId"], as_index=False)
        .agg(
            mean_contribution_rate=("contrib_rate", "mean"),
            punishment_rate=("punishment_rate", "mean"),
            reward_rate=("reward_rate", "mean"),
        )
        .copy()
    )
    player_means = (
        rate_rows.groupby(["benchmark_id", "gameId", "playerId"], as_index=False)
        .agg(
            contrib_rate=("contrib_rate", "mean"),
            punishment_rate=("punishment_rate", "mean"),
            reward_rate=("reward_rate", "mean"),
        )
        .copy()
    )
    game_variances = (
        player_means.groupby(["benchmark_id", "gameId"], as_index=False)
        .agg(
            var_players_contrib_rate=("contrib_rate", lambda s: float(pd.Series(s).var(ddof=0))),
            var_players_punishment_rate=("punishment_rate", lambda s: float(pd.Series(s).var(ddof=0))),
            var_players_reward_rate=("reward_rate", lambda s: float(pd.Series(s).var(ddof=0))),
        )
        .copy()
    )

    return (
        game_efficiency.merge(game_rates, on=["benchmark_id", "gameId"], how="inner", validate="one_to_one")
        .merge(game_variances, on=["benchmark_id", "gameId"], how="inner", validate="one_to_one")
        .copy()
    )


def build_noise_ceiling_tables(
    *,
    human_analysis_csv: Path,
    human_rows_csv: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    game_table = build_human_game_macro_output_table(human_analysis_csv, human_rows_csv)
    metric_specs = [
        ("mean_contribution_rate", "Mean Contribution Rate"),
        ("punishment_rate", "Punishment Rate"),
        ("reward_rate", "Reward Rate"),
        ("normalized_efficiency", "Normalized Efficiency"),
        ("var_players_contrib_rate", "Player Contribution Variance"),
        ("var_players_punishment_rate", "Player Punishment Variance"),
        ("var_players_reward_rate", "Player Reward Variance"),
    ]

    detail_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    for metric_col, metric_label in metric_specs:
        grouped = (
            game_table.groupby("benchmark_id")[metric_col]
            .agg(
                n_games="count",
                benchmark_mean="mean",
                sample_variance=lambda s: float(pd.Series(s).var(ddof=1)) if len(s) > 1 else np.nan,
            )
            .reset_index()
        )
        grouped["metric"] = metric_col
        grouped["metric_label"] = metric_label
        grouped["mse_floor_component"] = grouped["sample_variance"] / grouped["n_games"]
        detail_frames.append(grouped)

        valid = grouped[grouped["mse_floor_component"].notna()].copy()
        summary_rows.append(
            {
                "metric": metric_col,
                "metric_label": metric_label,
                "n_benchmarks_total": int(len(grouped)),
                "n_benchmarks_used": int(len(valid)),
                "mean_n_games_per_benchmark": float(valid["n_games"].mean()) if len(valid) else None,
                "mean_within_benchmark_sample_variance": float(valid["sample_variance"].mean()) if len(valid) else None,
                "mse_floor_plugin": float(valid["mse_floor_component"].mean()) if len(valid) else None,
                "rmse_floor_plugin": float(np.sqrt(valid["mse_floor_component"].mean())) if len(valid) else None,
                "formula": "sqrt(mean(s2_x / n_x))",
            }
        )

    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, detail_df


def _rmse(series_a: pd.Series, series_b: pd.Series) -> float:
    pair = pd.DataFrame({"a": pd.to_numeric(series_a, errors="coerce"), "b": pd.to_numeric(series_b, errors="coerce")}).dropna()
    if pair.empty:
        return float("nan")
    return float(np.sqrt(((pair["a"] - pair["b"]) ** 2).mean()))


def _mae(series_a: pd.Series, series_b: pd.Series) -> float:
    pair = pd.DataFrame({"a": pd.to_numeric(series_a, errors="coerce"), "b": pd.to_numeric(series_b, errors="coerce")}).dropna()
    if pair.empty:
        return float("nan")
    return float((pair["a"] - pair["b"]).abs().mean())


def _corr(series_a: pd.Series, series_b: pd.Series) -> float:
    pair = pd.DataFrame({"a": pd.to_numeric(series_a, errors="coerce"), "b": pd.to_numeric(series_b, errors="coerce")}).dropna()
    if len(pair) < 2 or pair["a"].nunique() < 2 or pair["b"].nunique() < 2:
        return float("nan")
    return float(pair["a"].corr(pair["b"]))


def _signed(value: float, eps: float = 1e-12) -> int:
    if pd.isna(value):
        return 0
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def build_macro_metric_bundle(
    *,
    sim_rows_csv: Path,
    human_rows_csv: Path,
    human_analysis_csv: Path,
    game_level_df: pd.DataFrame,
) -> Dict[str, Any]:
    human_full_analysis_csv = resolve_full_analysis_csv(human_analysis_csv)
    learn_analysis_csv = human_full_analysis_csv.parent / "df_analysis_learn.csv"
    learn_rows_csv = human_rows_csv.parents[1] / "learning_wave" / "player-rounds.csv"

    sim_rate_rows = _build_sim_rate_rows(sim_rows_csv, human_analysis_csv)
    human_rate_rows = _build_rate_rows(
        rows_csv=human_rows_csv,
        analysis_csv=human_full_analysis_csv,
        contribution_col="data.contribution",
        punished_col="data.punished",
        rewarded_col="data.rewarded",
    )

    benchmark_meta = pd.read_csv(human_analysis_csv).copy()
    benchmark_meta["gameId"] = benchmark_meta["gameId"].astype(str)
    benchmark_meta["benchmark_id"] = _benchmark_series(benchmark_meta)
    keep_meta = ["gameId", "benchmark_id"]
    if "benchmark_game_count" in benchmark_meta.columns:
        keep_meta.append("benchmark_game_count")
    benchmark_meta = benchmark_meta[keep_meta].drop_duplicates(subset=["gameId"], keep="first")
    if "benchmark_id" in game_level_df.columns:
        enriched_game_level = game_level_df.copy()
        if "benchmark_game_count" not in enriched_game_level.columns and "benchmark_game_count" in benchmark_meta.columns:
            benchmark_counts = (
                benchmark_meta[["benchmark_id", "benchmark_game_count"]]
                .drop_duplicates(subset=["benchmark_id"], keep="first")
                .copy()
            )
            enriched_game_level = enriched_game_level.merge(benchmark_counts, on="benchmark_id", how="left")
    else:
        enriched_game_level = game_level_df.merge(benchmark_meta, on="gameId", how="left")
    if "benchmark_id" not in enriched_game_level.columns or enriched_game_level["benchmark_id"].isna().any():
        raise ValueError("Could not attach treatment benchmark IDs to the game-level macro analysis table.")

    shared_benchmarks = set(enriched_game_level["benchmark_id"].astype(str))
    sim_rate_rows = sim_rate_rows[sim_rate_rows["benchmark_id"].isin(shared_benchmarks)].copy()
    human_rate_rows = human_rate_rows[human_rate_rows["benchmark_id"].isin(shared_benchmarks)].copy()

    human_benchmark_rates = build_human_benchmark_rate_targets(human_rows_csv, human_full_analysis_csv)
    human_benchmark_rates = human_benchmark_rates[
        human_benchmark_rates["benchmark_id"].isin(shared_benchmarks)
    ].copy()
    enriched_game_level = enriched_game_level.merge(human_benchmark_rates, on="benchmark_id", how="left")

    linear_predictions = pd.DataFrame()
    linear_summary = pd.DataFrame()
    if learn_analysis_csv.exists() and learn_rows_csv.exists():
        learn_benchmark_table = build_benchmark_target_table(learn_analysis_csv, learn_rows_csv)
        val_benchmark_table = build_benchmark_target_table(human_full_analysis_csv, human_rows_csv)
        val_benchmark_table = val_benchmark_table[val_benchmark_table["benchmark_id"].isin(shared_benchmarks)].copy()
        linear_predictions, linear_summary = build_linear_baseline_predictions(
            learn_benchmark_table=learn_benchmark_table,
            val_benchmark_table=val_benchmark_table,
            feature_cols=LINEAR_CONFIG_FEATURES,
        )
        if not linear_predictions.empty:
            enriched_game_level = enriched_game_level.merge(linear_predictions, on="benchmark_id", how="left")

    rmse_summary = pd.DataFrame(
        [
            {
                "metric": "contribution_rate",
                "sim_rmse": _rmse(enriched_game_level["sim_mean_contribution_rate"], enriched_game_level["human_mean_contribution_rate"]),
                "sim_mae": _mae(enriched_game_level["sim_mean_contribution_rate"], enriched_game_level["human_mean_contribution_rate"]),
                "sim_corr": _corr(enriched_game_level["sim_mean_contribution_rate"], enriched_game_level["human_mean_contribution_rate"]),
                "linear_rmse": _rmse(enriched_game_level.get("linear_mean_contribution_rate"), enriched_game_level["human_mean_contribution_rate"]),
                "linear_mae": _mae(enriched_game_level.get("linear_mean_contribution_rate"), enriched_game_level["human_mean_contribution_rate"]),
                "linear_corr": _corr(enriched_game_level.get("linear_mean_contribution_rate"), enriched_game_level["human_mean_contribution_rate"]),
            },
            {
                "metric": "punishment_rate",
                "sim_rmse": _rmse(enriched_game_level["sim_punishment_rate"], enriched_game_level["human_punishment_rate"]),
                "sim_mae": _mae(enriched_game_level["sim_punishment_rate"], enriched_game_level["human_punishment_rate"]),
                "sim_corr": _corr(enriched_game_level["sim_punishment_rate"], enriched_game_level["human_punishment_rate"]),
                "linear_rmse": _rmse(enriched_game_level.get("linear_punishment_rate"), enriched_game_level["human_punishment_rate"]),
                "linear_mae": _mae(enriched_game_level.get("linear_punishment_rate"), enriched_game_level["human_punishment_rate"]),
                "linear_corr": _corr(enriched_game_level.get("linear_punishment_rate"), enriched_game_level["human_punishment_rate"]),
            },
            {
                "metric": "reward_rate",
                "sim_rmse": _rmse(enriched_game_level["sim_reward_rate"], enriched_game_level["human_reward_rate"]),
                "sim_mae": _mae(enriched_game_level["sim_reward_rate"], enriched_game_level["human_reward_rate"]),
                "sim_corr": _corr(enriched_game_level["sim_reward_rate"], enriched_game_level["human_reward_rate"]),
                "linear_rmse": _rmse(enriched_game_level.get("linear_reward_rate"), enriched_game_level["human_reward_rate"]),
                "linear_mae": _mae(enriched_game_level.get("linear_reward_rate"), enriched_game_level["human_reward_rate"]),
                "linear_corr": _corr(enriched_game_level.get("linear_reward_rate"), enriched_game_level["human_reward_rate"]),
            },
            {
                "metric": "normalized_efficiency",
                "sim_rmse": _rmse(enriched_game_level["sim_normalized_efficiency"], enriched_game_level["human_normalized_efficiency"]),
                "sim_mae": _mae(enriched_game_level["sim_normalized_efficiency"], enriched_game_level["human_normalized_efficiency"]),
                "sim_corr": _corr(enriched_game_level["sim_normalized_efficiency"], enriched_game_level["human_normalized_efficiency"]),
                "linear_rmse": _rmse(enriched_game_level.get("linear_normalized_efficiency"), enriched_game_level["human_normalized_efficiency"]),
                "linear_mae": _mae(enriched_game_level.get("linear_normalized_efficiency"), enriched_game_level["human_normalized_efficiency"]),
                "linear_corr": _corr(enriched_game_level.get("linear_normalized_efficiency"), enriched_game_level["human_normalized_efficiency"]),
            },
        ]
    )
    rmse_summary["rmse"] = rmse_summary["sim_rmse"]
    rmse_summary["mae"] = rmse_summary["sim_mae"]
    rmse_summary["corr"] = rmse_summary["sim_corr"]

    sim_player_summary, sim_player_variances = _player_variance_summary(sim_rate_rows, prefix="sim")
    human_player_summary, human_player_variances = _player_variance_summary(human_rate_rows, prefix="human")
    variance_summary = pd.DataFrame(
        [
            {
                **sim_player_summary,
                **human_player_summary,
                "mean_var_players_contrib_rate_linear": float(pd.to_numeric(enriched_game_level.get("linear_var_players_contrib_rate"), errors="coerce").mean()),
                "mean_var_players_punishment_rate_linear": float(pd.to_numeric(enriched_game_level.get("linear_var_players_punishment_rate"), errors="coerce").mean()),
                "mean_var_players_reward_rate_linear": float(pd.to_numeric(enriched_game_level.get("linear_var_players_reward_rate"), errors="coerce").mean()),
                "var_across_games_contrib_rate_sim": float(pd.to_numeric(enriched_game_level["sim_mean_contribution_rate"], errors="coerce").var(ddof=0)),
                "var_across_games_contrib_rate_human": float(pd.to_numeric(enriched_game_level["human_mean_contribution_rate"], errors="coerce").var(ddof=0)),
                "var_across_games_contrib_rate_linear": float(pd.to_numeric(enriched_game_level.get("linear_mean_contribution_rate"), errors="coerce").var(ddof=0)),
                "var_across_games_punishment_rate_sim": float(pd.to_numeric(enriched_game_level["sim_punishment_rate"], errors="coerce").var(ddof=0)),
                "var_across_games_punishment_rate_human": float(pd.to_numeric(enriched_game_level["human_punishment_rate"], errors="coerce").var(ddof=0)),
                "var_across_games_punishment_rate_linear": float(pd.to_numeric(enriched_game_level.get("linear_punishment_rate"), errors="coerce").var(ddof=0)),
                "var_across_games_reward_rate_sim": float(pd.to_numeric(enriched_game_level["sim_reward_rate"], errors="coerce").var(ddof=0)),
                "var_across_games_reward_rate_human": float(pd.to_numeric(enriched_game_level["human_reward_rate"], errors="coerce").var(ddof=0)),
                "var_across_games_reward_rate_linear": float(pd.to_numeric(enriched_game_level.get("linear_reward_rate"), errors="coerce").var(ddof=0)),
                "var_across_games_normalized_efficiency_sim": float(pd.to_numeric(enriched_game_level["sim_normalized_efficiency"], errors="coerce").var(ddof=0)),
                "var_across_games_normalized_efficiency_human": float(pd.to_numeric(enriched_game_level["human_normalized_efficiency"], errors="coerce").var(ddof=0)),
                "var_across_games_normalized_efficiency_linear": float(pd.to_numeric(enriched_game_level.get("linear_normalized_efficiency"), errors="coerce").var(ddof=0)),
                "n_games": int(enriched_game_level["gameId"].nunique()),
                "n_benchmarks": int(enriched_game_level["benchmark_id"].nunique()),
            }
        ]
    )
    player_variance_by_game = sim_player_variances.merge(
        human_player_variances,
        on="benchmark_id",
        how="outer",
        suffixes=("_sim", "_human"),
    )
    if not linear_predictions.empty:
        linear_var_cols = [
            "benchmark_id",
            "linear_var_players_contrib_rate",
            "linear_var_players_punishment_rate",
            "linear_var_players_reward_rate",
        ]
        linear_var_rows = enriched_game_level[linear_var_cols].drop_duplicates(subset=["benchmark_id"], keep="first")
        player_variance_by_game = player_variance_by_game.merge(linear_var_rows, on="benchmark_id", how="left")
    noise_ceiling_summary, noise_ceiling_detail = build_noise_ceiling_tables(
        human_analysis_csv=human_full_analysis_csv,
        human_rows_csv=human_rows_csv,
    )
    return {
        "enriched_game_level": enriched_game_level,
        "rmse_summary": rmse_summary,
        "variance_summary": variance_summary,
        "player_variance_by_game": player_variance_by_game,
        "linear_summary": linear_summary,
        "noise_ceiling_summary": noise_ceiling_summary,
        "noise_ceiling_detail": noise_ceiling_detail,
        "linear_config_features": list(LINEAR_CONFIG_FEATURES),
        "learn_analysis_csv": str(learn_analysis_csv),
        "learn_rows_csv": str(learn_rows_csv),
    }


def build_directional_rows_with_baseline(
    *,
    merged: pd.DataFrame,
    factors: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    binary_factors = {
        "CONFIG_chat",
        "CONFIG_allOrNothing",
        "CONFIG_defaultContribProp",
        "CONFIG_rewardExists",
        "CONFIG_showNRounds",
        "CONFIG_showPunishmentId",
        "CONFIG_showOtherSummaries",
        "CONFIG_punishmentExists",
    }

    for factor in factors:
        if factor not in merged.columns:
            continue
        sub = merged[
            [
                factor,
                "human_normalized_efficiency",
                "sim_normalized_efficiency",
                "linear_normalized_efficiency",
            ]
        ].dropna()
        if sub.empty:
            continue

        if factor in binary_factors:
            mask_high = parse_binary_series(sub[factor])
            high = sub[mask_high]
            low = sub[~mask_high]
            mode = "binary"
            threshold = None
        else:
            numeric = pd.to_numeric(sub[factor], errors="coerce")
            threshold = float(numeric.median())
            high = sub[numeric > threshold]
            low = sub[numeric < threshold]
            mode = "median"

        if high.empty or low.empty:
            continue

        human_delta = float(high["human_normalized_efficiency"].mean() - low["human_normalized_efficiency"].mean())
        sim_delta = float(high["sim_normalized_efficiency"].mean() - low["sim_normalized_efficiency"].mean())
        linear_delta = float(high["linear_normalized_efficiency"].mean() - low["linear_normalized_efficiency"].mean())
        human_sign = _signed(human_delta)
        sim_sign = _signed(sim_delta)
        linear_sign = _signed(linear_delta)

        rows.append(
            {
                "factor": factor,
                "mode": mode,
                "threshold": threshold,
                "n_total": int(len(sub)),
                "n_high": int(len(high)),
                "n_low": int(len(low)),
                "human_delta": human_delta,
                "sim_delta": sim_delta,
                "linear_delta": linear_delta,
                "human_sign": human_sign,
                "sim_sign": sim_sign,
                "linear_sign": linear_sign,
                "sign_match_sim": bool(human_sign == sim_sign),
                "sign_match_linear": bool(human_sign == linear_sign),
                "sign_match_nonzero_human_sim": bool(human_sign != 0 and human_sign == sim_sign),
                "sign_match_nonzero_human_linear": bool(human_sign != 0 and human_sign == linear_sign),
            }
        )
    return pd.DataFrame(rows)


def build_directional_sign_summary(directional_df: pd.DataFrame) -> pd.DataFrame:
    if directional_df.empty:
        return pd.DataFrame()
    nonzero = directional_df["human_sign"] != 0
    return pd.DataFrame(
        [
            {
                "n_factors_evaluated": int(len(directional_df)),
                "sim_sign_match_rate_all": float(directional_df["sign_match_sim"].mean()),
                "sim_sign_match_rate_nonzero_human": float(directional_df.loc[nonzero, "sign_match_sim"].mean())
                if nonzero.any()
                else None,
                "linear_sign_match_rate_all": float(directional_df["sign_match_linear"].mean()),
                "linear_sign_match_rate_nonzero_human": float(
                    directional_df.loc[nonzero, "sign_match_linear"].mean()
                )
                if nonzero.any()
                else None,
            }
        ]
    )


def _ensure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_rmse_summary(rmse_summary: pd.DataFrame, out_path: Path, dpi: int) -> Optional[str]:
    if rmse_summary.empty:
        return None
    plt = _ensure_matplotlib()
    x = np.arange(len(rmse_summary))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, rmse_summary["sim_rmse"], width=width, label="simulation")
    ax.bar(x + width / 2, rmse_summary["linear_rmse"], width=width, label="linear_config")
    ax.set_title("Macro RMSE by Target")
    ax.set_ylabel("RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(rmse_summary["metric"], rotation=20)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_aggregate_efficiency_comparison(
    aggregate_df: pd.DataFrame,
    out_path: Path,
    *,
    dpi: int,
) -> Optional[str]:
    if aggregate_df.empty:
        return None
    plot_df = aggregate_df.copy()
    plot_df["label"] = plot_df["label"].astype(str)
    metrics = [
        ("mae_normalized_efficiency", "MAE", False),
        ("rmse_normalized_efficiency", "RMSE", False),
        ("corr_normalized_efficiency", "Correlation", True),
    ]
    plt = _ensure_matplotlib()
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4.2))
    if len(metrics) == 1:
        axes = [axes]
    for ax, (column, title, higher_is_better) in zip(axes, metrics):
        if column not in plot_df.columns:
            ax.set_visible(False)
            continue
        values = pd.to_numeric(plot_df[column], errors="coerce")
        subset = plot_df.loc[values.notna(), ["label", column]].copy()
        if subset.empty:
            ax.set_visible(False)
            continue
        y = np.arange(len(subset))
        bars = ax.barh(y, subset[column].astype(float), color="#4C78A8")
        ax.set_yticks(y)
        ax.set_yticklabels(subset["label"])
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
        if higher_is_better:
            ax.set_xlabel("Higher is better")
        else:
            ax.set_xlabel("Lower is better")
        for bar, value in zip(bars, subset[column].astype(float)):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2.0,
                f" {value:.3f}",
                va="center",
                ha="left",
                fontsize=9,
            )
    fig.suptitle("Macro Normalized-Efficiency Comparison")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_variance_bars(
    variance_summary: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    metric_specs: Sequence[Tuple[str, str, str, str]],
    dpi: int,
) -> Optional[str]:
    if variance_summary.empty:
        return None
    row = variance_summary.iloc[0]
    labels = [label for _, _, _, label in metric_specs]
    sim_vals = [float(row[sim_col]) for sim_col, _, _, _ in metric_specs]
    linear_vals = [float(row[linear_col]) for _, linear_col, _, _ in metric_specs]
    human_vals = [float(row[human_col]) for _, _, human_col, _ in metric_specs]
    plt = _ensure_matplotlib()
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width, sim_vals, width=width, label="simulation")
    ax.bar(x, linear_vals, width=width, label="linear_config")
    ax.bar(x + width, human_vals, width=width, label="human")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Variance")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_game_level_scatter(game_level_df: pd.DataFrame, out_path: Path, dpi: int) -> Optional[str]:
    if game_level_df.empty:
        return None
    plt = _ensure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    specs = [
        (
            "human_mean_contribution_rate",
            "sim_mean_contribution_rate",
            "linear_mean_contribution_rate",
            "Contribution Rate",
        ),
        ("human_punishment_rate", "sim_punishment_rate", "linear_punishment_rate", "Punishment Rate"),
        ("human_reward_rate", "sim_reward_rate", "linear_reward_rate", "Reward Rate"),
        (
            "human_normalized_efficiency",
            "sim_normalized_efficiency",
            "linear_normalized_efficiency",
            "Normalized Efficiency",
        ),
    ]
    for ax, (human_col, sim_col, linear_col, title) in zip(axes.flat, specs):
        pair = game_level_df[[human_col, sim_col, linear_col]].dropna()
        if pair.empty:
            ax.set_visible(False)
            continue
        ax.scatter(pair[human_col], pair[sim_col], alpha=0.7, s=22, label="simulation")
        ax.scatter(pair[human_col], pair[linear_col], alpha=0.7, s=22, marker="s", label="linear_config")
        mn = min(pair[human_col].min(), pair[sim_col].min(), pair[linear_col].min())
        mx = max(pair[human_col].max(), pair[sim_col].max(), pair[linear_col].max())
        ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.0, color="black")
        ax.set_title(title)
        ax.set_xlabel("Human")
        ax.set_ylabel("Prediction")
        ax.grid(alpha=0.3)
        ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_directional_effects_with_baseline(
    directional_df: pd.DataFrame,
    out_path: Path,
    *,
    label: str,
    dpi: int,
) -> Optional[str]:
    if directional_df.empty:
        return None
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(12, max(3.2, 0.5 * len(directional_df))))
    x = np.arange(len(directional_df))
    width = 0.25
    ax.barh(x - width, directional_df["human_delta"], height=width, label="human")
    ax.barh(x, directional_df["sim_delta"], height=width, label="simulation")
    ax.barh(x + width, directional_df["linear_delta"], height=width, label="linear_config")
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(x)
    ax.set_yticklabels(directional_df["factor"])
    ax.set_title(f"{label}: normalized-efficiency directional deltas")
    ax.set_xlabel("high-or-True mean minus low-or-False mean")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def update_manifest(manifest_path: Path, supplemental_payload: Dict[str, Any]) -> None:
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    reports = list(payload.get("supplemental_reports", []))
    reports.append(supplemental_payload)
    payload["supplemental_reports"] = reports
    plot_paths = list(payload.get("plots", []))
    plot_paths.extend(supplemental_payload.get("generated_files", []))
    payload["plots"] = list(dict.fromkeys(plot_paths))
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
