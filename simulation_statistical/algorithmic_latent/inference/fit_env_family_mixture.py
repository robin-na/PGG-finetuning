from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = str(SCRIPT_PATH.parent)
ALGORITHMIC_LATENT_ROOT = SCRIPT_PATH.parents[1]
SIMULATION_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[3]
for path in (SCRIPT_DIR, str(SIMULATION_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from simulation_statistical.algorithmic_latent.inference.infer_player_posteriors import (  # noqa: E402
    DEFAULT_PLAYER_OUTPUT_PATH,
)
from simulation_statistical.archetype_distribution_embedding.models.env_distribution_dirichlet import (  # noqa: E402
    DirichletEnvRegressor,
)
from simulation_statistical.archetype_distribution_embedding.utils.constants import (  # noqa: E402
    REQUIRED_CONFIG_COLUMNS,
    EPSILON,
)
from simulation_statistical.paths import BENCHMARK_DATA_ROOT  # noqa: E402


DEFAULT_MODEL_OUTPUT_PATH = (
    ALGORITHMIC_LATENT_ROOT / "artifacts" / "models" / "env_family_mixture_model.pkl"
)
DEFAULT_LEARN_PREDICTIONS_PATH = (
    ALGORITHMIC_LATENT_ROOT / "artifacts" / "outputs" / "learning_wave_env_family_predictions.csv"
)
DEFAULT_VAL_PREDICTIONS_PATH = (
    ALGORITHMIC_LATENT_ROOT / "artifacts" / "outputs" / "validation_wave_env_family_predictions.csv"
)
DEFAULT_EVAL_SUMMARY_PATH = (
    ALGORITHMIC_LATENT_ROOT / "artifacts" / "outputs" / "env_family_mixture_eval_summary.csv"
)
DEFAULT_METADATA_PATH = (
    ALGORITHMIC_LATENT_ROOT / "artifacts" / "outputs" / "env_family_mixture_eval_summary.json"
)


def _analysis_csv_for_wave(wave: str) -> Path:
    if str(wave) == "learning_wave":
        return Path(REPO_ROOT) / BENCHMARK_DATA_ROOT / "processed_data" / "df_analysis_learn.csv"
    if str(wave) == "validation_wave":
        return Path(REPO_ROOT) / BENCHMARK_DATA_ROOT / "processed_data" / "df_analysis_val.csv"
    raise ValueError(f"Unsupported wave '{wave}'.")


def _load_player_posteriors(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_parquet(path)
    family_prob_columns = sorted(
        [column for column in frame.columns if str(column).endswith("__posterior_prob")]
    )
    if not family_prob_columns:
        raise ValueError(f"No family posterior probability columns found in {path}.")
    return frame, family_prob_columns


def _load_game_configs(analysis_csv: Path) -> pd.DataFrame:
    analysis = pd.read_csv(analysis_csv)
    analysis["gameId"] = analysis["gameId"].astype(str)
    keep_columns = ["gameId", "CONFIG_treatmentName", *REQUIRED_CONFIG_COLUMNS]
    keep_columns = [column for column in keep_columns if column in analysis.columns]
    analysis = analysis.loc[:, keep_columns].drop_duplicates(subset=["gameId"], keep="first").reset_index(drop=True)
    return analysis


def _game_level_family_mixture(
    player_posteriors: pd.DataFrame,
    game_configs: pd.DataFrame,
    family_prob_columns: Sequence[str],
) -> pd.DataFrame:
    grouped = (
        player_posteriors.groupby(["gameId", "CONFIG_treatmentName"], dropna=False)
        .agg(
            n_players=("playerId", "nunique"),
            posterior_entropy_mean=("posterior_entropy", "mean"),
            **{
                str(column): (str(column), "mean")
                for column in family_prob_columns
            },
        )
        .reset_index()
    )
    grouped["gameId"] = grouped["gameId"].astype(str)
    out = pd.merge(game_configs, grouped, on=["gameId", "CONFIG_treatmentName"], how="inner")
    if out.empty:
        raise ValueError("Game-level family mixture table is empty after joining configs.")
    return out


def _js_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=1)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=1)
    return 0.5 * (kl_pm + kl_qm)


def _top_family_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual_top = np.argmax(actual, axis=1)
    pred_top = np.argmax(predicted, axis=1)
    return float(np.mean(actual_top == pred_top))


def _eval_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    family_mae = np.mean(np.abs(actual - predicted), axis=0)
    l1 = np.sum(np.abs(actual - predicted), axis=1)
    js = _js_divergence(actual, predicted)
    return {
        "mean_family_mae": float(np.mean(family_mae)),
        "max_family_mae": float(np.max(family_mae)),
        "avg_l1_distance": float(np.mean(l1)),
        "avg_js_divergence": float(np.mean(js)),
        "top_family_accuracy": float(_top_family_accuracy(actual, predicted)),
    }


def _mean_baseline(train_target: np.ndarray, n_rows: int) -> np.ndarray:
    baseline = np.mean(train_target, axis=0, keepdims=True)
    baseline = baseline / np.clip(baseline.sum(axis=1, keepdims=True), EPSILON, None)
    return np.repeat(baseline, repeats=n_rows, axis=0)


def _treatment_aggregate_predictions(
    frame: pd.DataFrame,
    family_prob_columns: Sequence[str],
    predicted_columns: Sequence[str],
) -> pd.DataFrame:
    grouped = (
        frame.groupby("CONFIG_treatmentName", dropna=False)
        .agg(
            n_games=("gameId", "nunique"),
            **{
                str(column): (str(column), "mean")
                for column in [*family_prob_columns, *predicted_columns]
            },
        )
        .reset_index()
    )
    return grouped


def fit_env_family_mixture(
    *,
    learn_player_posteriors_path: Path = DEFAULT_PLAYER_OUTPUT_PATH,
    val_player_posteriors_path: Path = (
        ALGORITHMIC_LATENT_ROOT / "artifacts" / "outputs" / "validation_wave_player_family_posteriors.parquet"
    ),
    learn_analysis_csv: Path | None = None,
    val_analysis_csv: Path | None = None,
    model_output_path: Path = DEFAULT_MODEL_OUTPUT_PATH,
    learn_predictions_output_path: Path = DEFAULT_LEARN_PREDICTIONS_PATH,
    val_predictions_output_path: Path = DEFAULT_VAL_PREDICTIONS_PATH,
    eval_summary_output_path: Path = DEFAULT_EVAL_SUMMARY_PATH,
    metadata_output_path: Path = DEFAULT_METADATA_PATH,
) -> Dict[str, Any]:
    learn_analysis_csv = learn_analysis_csv or _analysis_csv_for_wave("learning_wave")
    val_analysis_csv = val_analysis_csv or _analysis_csv_for_wave("validation_wave")

    learn_posteriors, family_prob_columns = _load_player_posteriors(learn_player_posteriors_path)
    val_posteriors, val_family_prob_columns = _load_player_posteriors(val_player_posteriors_path)
    if list(family_prob_columns) != list(val_family_prob_columns):
        raise ValueError("Learning and validation posterior tables do not share the same family probability columns.")

    learn_game_configs = _load_game_configs(learn_analysis_csv)
    val_game_configs = _load_game_configs(val_analysis_csv)

    learn_game_mixture = _game_level_family_mixture(learn_posteriors, learn_game_configs, family_prob_columns)
    val_game_mixture = _game_level_family_mixture(val_posteriors, val_game_configs, family_prob_columns)

    target_columns = [str(column) for column in family_prob_columns]
    X_learn = learn_game_mixture.loc[:, list(REQUIRED_CONFIG_COLUMNS)].copy()
    X_val = val_game_mixture.loc[:, list(REQUIRED_CONFIG_COLUMNS)].copy()
    y_learn = learn_game_mixture.loc[:, target_columns].to_numpy(dtype=float)
    y_val = val_game_mixture.loc[:, target_columns].to_numpy(dtype=float)

    model = DirichletEnvRegressor(feature_columns=list(REQUIRED_CONFIG_COLUMNS))
    model.fit(X_learn, y_learn)
    learn_pred = model.predict(X_learn)
    val_pred = model.predict(X_val)

    predicted_columns = [column.replace("__posterior_prob", "__pred_prob") for column in target_columns]
    learn_predictions = learn_game_mixture.copy()
    val_predictions = val_game_mixture.copy()
    learn_predictions.loc[:, predicted_columns] = learn_pred
    val_predictions.loc[:, predicted_columns] = val_pred

    mean_baseline_val = _mean_baseline(y_learn, len(val_pred))
    learn_metrics = _eval_metrics(y_learn, learn_pred)
    val_metrics = _eval_metrics(y_val, val_pred)
    baseline_val_metrics = _eval_metrics(y_val, mean_baseline_val)

    learn_treatment = _treatment_aggregate_predictions(learn_predictions, target_columns, predicted_columns)
    val_treatment = _treatment_aggregate_predictions(val_predictions, target_columns, predicted_columns)
    learn_treatment_actual = learn_treatment.loc[:, target_columns].to_numpy(dtype=float)
    learn_treatment_pred = learn_treatment.loc[:, predicted_columns].to_numpy(dtype=float)
    val_treatment_actual = val_treatment.loc[:, target_columns].to_numpy(dtype=float)
    val_treatment_pred = val_treatment.loc[:, predicted_columns].to_numpy(dtype=float)
    learn_treatment_metrics = _eval_metrics(learn_treatment_actual, learn_treatment_pred)
    val_treatment_metrics = _eval_metrics(val_treatment_actual, val_treatment_pred)
    baseline_val_treatment_metrics = _eval_metrics(
        val_treatment_actual,
        _mean_baseline(y_learn, len(val_treatment_actual)),
    )

    summary_rows = [
        {"split": "learning_game", "model": "dirichlet_env_family", **learn_metrics},
        {"split": "validation_game", "model": "dirichlet_env_family", **val_metrics},
        {"split": "validation_game", "model": "mean_family_baseline", **baseline_val_metrics},
        {"split": "learning_treatment", "model": "dirichlet_env_family", **learn_treatment_metrics},
        {"split": "validation_treatment", "model": "dirichlet_env_family", **val_treatment_metrics},
        {"split": "validation_treatment", "model": "mean_family_baseline", **baseline_val_treatment_metrics},
    ]

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    learn_predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
    val_predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_output_path)
    learn_predictions.to_csv(learn_predictions_output_path, index=False)
    val_predictions.to_csv(val_predictions_output_path, index=False)
    pd.DataFrame(summary_rows).to_csv(eval_summary_output_path, index=False)

    metadata = {
        "version": 1,
        "family_probability_columns": target_columns,
        "learn_player_posteriors_path": str(learn_player_posteriors_path),
        "val_player_posteriors_path": str(val_player_posteriors_path),
        "learn_analysis_csv": str(learn_analysis_csv),
        "val_analysis_csv": str(val_analysis_csv),
        "model_output_path": str(model_output_path),
        "learn_predictions_output_path": str(learn_predictions_output_path),
        "val_predictions_output_path": str(val_predictions_output_path),
        "eval_summary_output_path": str(eval_summary_output_path),
        "n_learning_games": int(len(learn_game_mixture)),
        "n_validation_games": int(len(val_game_mixture)),
        "n_learning_treatments": int(learn_treatment["CONFIG_treatmentName"].nunique(dropna=True)),
        "n_validation_treatments": int(val_treatment["CONFIG_treatmentName"].nunique(dropna=True)),
        "model_n_iter": int(getattr(model, "n_iter_", 0)),
        "model_final_loss": float(getattr(model, "final_loss_", float("nan"))),
        "summary_rows": summary_rows,
    }
    with open(metadata_output_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit a CONFIG -> family-mixture environment model from player posterior tables."
    )
    parser.add_argument("--learn_player_posteriors_path", type=str, default=str(DEFAULT_PLAYER_OUTPUT_PATH))
    parser.add_argument(
        "--val_player_posteriors_path",
        type=str,
        default=str(ALGORITHMIC_LATENT_ROOT / "artifacts" / "outputs" / "validation_wave_player_family_posteriors.parquet"),
    )
    parser.add_argument("--learn_analysis_csv", type=str, default=None)
    parser.add_argument("--val_analysis_csv", type=str, default=None)
    parser.add_argument("--model_output_path", type=str, default=str(DEFAULT_MODEL_OUTPUT_PATH))
    parser.add_argument("--learn_predictions_output_path", type=str, default=str(DEFAULT_LEARN_PREDICTIONS_PATH))
    parser.add_argument("--val_predictions_output_path", type=str, default=str(DEFAULT_VAL_PREDICTIONS_PATH))
    parser.add_argument("--eval_summary_output_path", type=str, default=str(DEFAULT_EVAL_SUMMARY_PATH))
    parser.add_argument("--metadata_output_path", type=str, default=str(DEFAULT_METADATA_PATH))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    metadata = fit_env_family_mixture(
        learn_player_posteriors_path=Path(args.learn_player_posteriors_path).resolve(),
        val_player_posteriors_path=Path(args.val_player_posteriors_path).resolve(),
        learn_analysis_csv=Path(args.learn_analysis_csv).resolve() if args.learn_analysis_csv else None,
        val_analysis_csv=Path(args.val_analysis_csv).resolve() if args.val_analysis_csv else None,
        model_output_path=Path(args.model_output_path).resolve(),
        learn_predictions_output_path=Path(args.learn_predictions_output_path).resolve(),
        val_predictions_output_path=Path(args.val_predictions_output_path).resolve(),
        eval_summary_output_path=Path(args.eval_summary_output_path).resolve(),
        metadata_output_path=Path(args.metadata_output_path).resolve(),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
