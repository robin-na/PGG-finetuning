from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .analyze_vs_human_treatments import _spearman_rank_correlation
from .evaluate_config_linear_regression import (
    BOOL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    TARGET_METRICS,
    _build_treatment_summary,
    _build_wave_game_summary,
    _oracle_noise_floor_rmse,
)


MODEL_SPECS = {
    "linear": {
        "estimator": LinearRegression(),
        "param_grid": {},
        "scale_numeric": True,
    },
    "ridge": {
        "estimator": Ridge(),
        "param_grid": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "scale_numeric": True,
    },
    "elastic_net": {
        "estimator": ElasticNet(max_iter=20000, random_state=7),
        "param_grid": {
            "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "model__l1_ratio": [0.1, 0.5, 0.9],
        },
        "scale_numeric": True,
    },
    "random_forest": {
        "estimator": RandomForestRegressor(
            n_estimators=300,
            random_state=7,
            n_jobs=1,
        ),
        "param_grid": {
            "model__max_depth": [None, 8],
            "model__min_samples_leaf": [1, 3],
            "model__max_features": ["sqrt", 0.7],
        },
        "scale_numeric": False,
    },
}

MODEL_LABELS = {
    "linear": "Linear",
    "ridge": "Ridge",
    "elastic_net": "Elastic Net",
    "random_forest": "Random Forest",
}

CORE_PLOT_METRICS = [
    "mean_total_contribution_rate",
    "mean_round_normalized_efficiency",
    "final_total_contribution_rate",
    "final_round_normalized_efficiency",
]


def _build_feature_frames(
    learning_treatment_df: pd.DataFrame,
    validation_treatment_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_features = learning_treatment_df[FEATURE_COLUMNS].copy()
    val_features = validation_treatment_df[FEATURE_COLUMNS].copy()
    for col in BOOL_FEATURES:
        train_features[col] = train_features[col].astype(int)
        val_features[col] = val_features[col].astype(int)
    for col in CATEGORICAL_FEATURES:
        train_features[col] = train_features[col].fillna("missing").astype(str)
        val_features[col] = val_features[col].fillna("missing").astype(str)
    return train_features, val_features


def _build_pipeline(*, estimator: Any, scale_numeric: bool) -> Pipeline:
    numeric_features = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_FEATURES]
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_features),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def _fit_model_for_metric(
    *,
    model_name: str,
    train_features: pd.DataFrame,
    train_target: pd.Series,
) -> tuple[Any, float, dict[str, Any]]:
    spec = MODEL_SPECS[model_name]
    pipeline = _build_pipeline(
        estimator=spec["estimator"],
        scale_numeric=bool(spec["scale_numeric"]),
    )
    param_grid = dict(spec["param_grid"])
    if not param_grid:
        fitted = pipeline.fit(train_features, train_target)
        return fitted, float("nan"), {}

    cv = KFold(n_splits=5, shuffle=True, random_state=7)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(train_features, train_target)
    return search.best_estimator_, float(-search.best_score_), dict(search.best_params_)


def _plot_rmse_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    metric_labels = dict(TARGET_METRICS)
    plot_df = summary_df[summary_df["metric"].isin(CORE_PLOT_METRICS)].copy()
    metric_order = CORE_PLOT_METRICS
    model_order = ["linear", "ridge", "elastic_net", "random_forest"]
    x = np.arange(len(metric_order), dtype=float)
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for idx, model_name in enumerate(model_order):
        subset = (
            plot_df[plot_df["model_name"] == model_name]
            .set_index("metric")
            .reindex(metric_order)
        )
        ax.bar(
            x + ((idx - 1.5) * bar_width),
            subset["rmse"].to_numpy(dtype=float),
            width=bar_width,
            label=MODEL_LABELS[model_name],
            alpha=0.9,
        )

    oracle = (
        plot_df.drop_duplicates("metric")
        .set_index("metric")
        .reindex(metric_order)["oracle_noise_floor_rmse"]
        .to_numpy(dtype=float)
    )
    ax.plot(x, oracle, linestyle="--", color="0.25", linewidth=1.5, label="Oracle floor")
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[metric] for metric in metric_order], rotation=20, ha="right")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("CONFIG-Only Supervised Baselines On Validation Treatment Means")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CONFIG-only supervised baselines on validation treatment means."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.forecasting_root / "results" / "config_supervised_models__macro_eval"

    learning_game_summary_df, _ = _build_wave_game_summary(args.repo_root, "learn")
    validation_game_summary_df, _ = _build_wave_game_summary(args.repo_root, "val")
    learning_treatment_df = _build_treatment_summary(learning_game_summary_df)
    validation_treatment_df = _build_treatment_summary(validation_game_summary_df)
    train_features, val_features = _build_feature_frames(learning_treatment_df, validation_treatment_df)

    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for metric, _ in TARGET_METRICS:
        train_target = learning_treatment_df[metric].astype(float)
        val_target = validation_treatment_df[metric].astype(float)
        valid_train = train_target.notna()
        valid_val = val_target.notna()
        if not valid_train.any() or not valid_val.any():
            continue

        oracle_rmse = _oracle_noise_floor_rmse(validation_game_summary_df, metric)
        for model_name in MODEL_SPECS:
            fitted_model, cv_rmse, best_params = _fit_model_for_metric(
                model_name=model_name,
                train_features=train_features.loc[valid_train].reset_index(drop=True),
                train_target=train_target.loc[valid_train].reset_index(drop=True),
            )
            predictions = fitted_model.predict(val_features.loc[valid_val].reset_index(drop=True))
            prediction_df = pd.DataFrame(
                {
                    "model_name": model_name,
                    "treatment_name": validation_treatment_df.loc[valid_val, "treatment_name"].to_numpy(),
                    "metric": metric,
                    "predicted_treatment_mean": predictions,
                    "human_treatment_mean": val_target.loc[valid_val].to_numpy(dtype=float),
                }
            )
            prediction_df["abs_error"] = (
                prediction_df["predicted_treatment_mean"] - prediction_df["human_treatment_mean"]
            ).abs()
            prediction_rows.extend(prediction_df.to_dict(orient="records"))

            rmse = float(
                np.sqrt(
                    np.mean(
                        (prediction_df["predicted_treatment_mean"] - prediction_df["human_treatment_mean"]) ** 2
                    )
                )
            )
            summary_rows.append(
                {
                    "model_name": model_name,
                    "metric": metric,
                    "num_learning_treatments": int(learning_treatment_df["treatment_name"].nunique()),
                    "num_validation_treatments": int(prediction_df["treatment_name"].nunique()),
                    "mae": float(prediction_df["abs_error"].mean()),
                    "rmse": rmse,
                    "spearman": _spearman_rank_correlation(
                        prediction_df["predicted_treatment_mean"],
                        prediction_df["human_treatment_mean"],
                    ),
                    "oracle_noise_floor_rmse": oracle_rmse,
                    "ratio_to_oracle_noise_floor": rmse / oracle_rmse if oracle_rmse > 0 else float("nan"),
                    "cv_rmse": cv_rmse,
                    "best_params": json.dumps(best_params, sort_keys=True),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    best_model_df = (
        summary_df.sort_values(["metric", "rmse"])
        .groupby("metric", as_index=False)
        .first()
        .sort_values("rmse")
        .reset_index(drop=True)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.sort_values(["metric", "rmse", "model_name"]).to_csv(
        args.output_dir / "supervised_model_macro_summary.csv",
        index=False,
    )
    prediction_df.sort_values(["metric", "model_name", "treatment_name"]).to_csv(
        args.output_dir / "supervised_model_validation_predictions.csv",
        index=False,
    )
    best_model_df.to_csv(args.output_dir / "best_model_by_metric.csv", index=False)
    _plot_rmse_comparison(summary_df, args.output_dir / "supervised_model_rmse_comparison.png")

    manifest = {
        "train_split": "learn",
        "test_split": "val",
        "valid_start_only": True,
        "feature_columns": FEATURE_COLUMNS,
        "target_metrics": [metric for metric, _ in TARGET_METRICS],
        "models": list(MODEL_SPECS.keys()),
        "num_learning_treatments": int(learning_treatment_df["treatment_name"].nunique()),
        "num_validation_treatments": int(validation_treatment_df["treatment_name"].nunique()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote outputs to {args.output_dir}")
    print(summary_df.sort_values(["metric", "rmse", "model_name"]).to_string(index=False))


if __name__ == "__main__":
    main()
