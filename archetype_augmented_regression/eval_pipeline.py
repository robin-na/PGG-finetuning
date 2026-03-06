from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from archetype_augmented_regression.io_utils import (
    aggregate_xy,
    coerce_config_col,
    load_jsonl,
    resolve_target,
)
from archetype_augmented_regression.modeling import build_preprocessor, metrics, r2_oos_train_mean
from archetype_augmented_regression.noise_ceiling import (
    compute_noise_ceiling,
    compute_sampling_noise_ceiling,
)
from archetype_augmented_regression.style_features import (
    cluster_distribution_by_game,
    fit_predict_synthetic_style,
)

DEFAULT_LEARN_ANALYSIS = Path("benchmark/data/processed_data/df_analysis_learn.csv")
DEFAULT_VAL_ANALYSIS = Path("benchmark/data/processed_data/df_analysis_val.csv")
DEFAULT_LEARN_PERSONA = Path("Persona/archetype_oracle_gpt51_learn.jsonl")
DEFAULT_VAL_PERSONA = Path("Persona/archetype_oracle_gpt51_val.jsonl")
DEFAULT_OUTPUT_CSV = Path("reports/archetype_augmented_regression/results.csv")
DEFAULT_OUTPUT_JSON = Path("reports/archetype_augmented_regression/summary.json")


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate style-cluster augmentation for validation-wave efficiency prediction."
    )
    parser.add_argument("--learn-analysis-csv", type=Path, default=DEFAULT_LEARN_ANALYSIS)
    parser.add_argument("--val-analysis-csv", type=Path, default=DEFAULT_VAL_ANALYSIS)
    parser.add_argument("--learn-persona-jsonl", type=Path, default=DEFAULT_LEARN_PERSONA)
    parser.add_argument("--val-persona-jsonl", type=Path, default=DEFAULT_VAL_PERSONA)
    parser.add_argument(
        "--target",
        type=str,
        default="itt_efficiency",
        help=(
            "Target column in df_analysis files. If 'itt_normalized_efficiency' is requested but missing, "
            "the script falls back to 'itt_efficiency'."
        ),
    )
    parser.add_argument("--cluster-counts", type=int, nargs="+", default=[4, 6, 8, 10, 12, 16])
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--style-source",
        choices=["oracle", "synthetic", "both"],
        default="oracle",
        help=(
            "oracle: use observed validation style shares (upper bound). "
            "synthetic: predict style shares from CONFIG only. "
            "both: evaluate both."
        ),
    )
    parser.add_argument(
        "--style-ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge alpha for CONFIG->style mapping in synthetic mode.",
    )
    parser.add_argument(
        "--style-oof-folds",
        type=int,
        default=5,
        help="OOF folds used to generate learning-wave synthetic style shares.",
    )
    parser.add_argument(
        "--eval-granularity",
        choices=["game", "config_treatment", "both"],
        default="game",
        help=(
            "game: evaluate on individual games; "
            "config_treatment: average rows by group column before fit/eval; "
            "both: run both."
        ),
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default="CONFIG_treatmentName",
        help="Grouping column used when --eval-granularity includes config_treatment.",
    )
    parser.add_argument(
        "--exclude-config-cols",
        nargs="+",
        default=["CONFIG_configId", "CONFIG_treatmentName"],
        help="CONFIG columns to exclude from the feature matrix.",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def _best_rows(results_df: pd.DataFrame) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    best_rows: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for style_source in sorted(results_df["style_source"].unique()):
        best_rows[style_source] = {}
        sub = results_df[results_df["style_source"] == style_source]
        for eval_mode in sorted(sub["eval_granularity"].unique()):
            best_rows[style_source][eval_mode] = {}
            sub_mode = sub[sub["eval_granularity"] == eval_mode]
            for model_name in sorted(sub_mode["model"].unique()):
                best_row = sub_mode[sub_mode["model"] == model_name].sort_values(
                    ["delta_rmse", "delta_mae"], ascending=[True, True]
                ).iloc[0]
                best_rows[style_source][eval_mode][model_name] = {
                    k: v.item() if hasattr(v, "item") else v for k, v in best_row.items()
                }
    return best_rows


def run_eval(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    learn_df = pd.read_csv(args.learn_analysis_csv)
    val_df = pd.read_csv(args.val_analysis_csv)
    learn_df["gameId"] = learn_df["gameId"].astype(str)
    val_df["gameId"] = val_df["gameId"].astype(str)

    target = resolve_target(args.target, learn_df, val_df)
    eval_modes = ["game"] if args.eval_granularity == "game" else (
        ["config_treatment"] if args.eval_granularity == "config_treatment" else ["game", "config_treatment"]
    )
    if "config_treatment" in eval_modes:
        if args.group_col not in learn_df.columns or args.group_col not in val_df.columns:
            raise ValueError(
                f"Grouping column '{args.group_col}' missing from analysis tables. "
                f"learn_has={args.group_col in learn_df.columns}, val_has={args.group_col in val_df.columns}"
            )

    config_cols = [
        col
        for col in learn_df.columns
        if col.startswith("CONFIG_")
        and col in val_df.columns
        and col not in set(args.exclude_config_cols)
    ]
    if not config_cols:
        raise ValueError("No overlapping CONFIG columns found.")

    x_learn_cfg = learn_df[config_cols].copy()
    x_val_cfg = val_df[config_cols].copy()
    for col in config_cols:
        x_learn_cfg[col] = coerce_config_col(x_learn_cfg[col])
        x_val_cfg[col] = coerce_config_col(x_val_cfg[col])

    y_learn = learn_df[target].to_numpy()
    y_val = val_df[target].to_numpy()

    valid_col = "valid_number_of_starting_players"
    if valid_col in learn_df.columns:
        learn_valid_mask = learn_df[valid_col].fillna(False).astype(bool)
    else:
        learn_valid_mask = pd.Series(True, index=learn_df.index)
    if valid_col in val_df.columns:
        val_valid_mask = val_df[valid_col].fillna(False).astype(bool)
    else:
        val_valid_mask = pd.Series(True, index=val_df.index)

    learn_noise_df = learn_df.loc[learn_valid_mask].copy()
    val_noise_df = val_df.loc[val_valid_mask].copy()

    learn_games = set(learn_df["gameId"].astype(str))
    val_games = set(val_df["gameId"].astype(str))
    persona_learn = load_jsonl(args.learn_persona_jsonl)
    persona_val = load_jsonl(args.val_persona_jsonl)
    persona_learn = persona_learn[persona_learn["experiment"].isin(learn_games)].copy()
    persona_val = persona_val[persona_val["experiment"].isin(val_games)].copy()

    print(
        f"Persona coverage: learn games={persona_learn['experiment'].nunique()}/{len(learn_games)}, "
        f"val games={persona_val['experiment'].nunique()}/{len(val_games)}"
    )

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_range=(1, args.ngram_max),
    )
    learn_text_matrix = vectorizer.fit_transform(persona_learn["text"].fillna(""))
    val_text_matrix = vectorizer.transform(persona_val["text"].fillna(""))

    baseline_models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=args.ridge_alpha, random_state=42),
    }
    baseline_scores: dict[tuple[str, str], dict[str, float]] = {}
    baseline_counts: dict[str, dict[str, int]] = {}
    noise_ceiling: dict[str, dict[str, Any]] = {}
    for eval_mode in eval_modes:
        if eval_mode == "game":
            x_learn_eval = x_learn_cfg
            y_learn_eval = y_learn
            x_val_eval = x_val_cfg
            y_val_eval = y_val
        else:
            x_learn_eval, y_learn_eval = aggregate_xy(
                x_df=x_learn_cfg,
                y_arr=y_learn,
                group_ids=learn_df[args.group_col],
            )
            x_val_eval, y_val_eval = aggregate_xy(
                x_df=x_val_cfg,
                y_arr=y_val,
                group_ids=val_df[args.group_col],
            )

        train_mean_target = float(np.mean(y_learn_eval))
        baseline_counts[eval_mode] = {
            "n_train_eval_rows": int(len(x_learn_eval)),
            "n_val_eval_rows": int(len(x_val_eval)),
            "train_mean_target": train_mean_target,
        }
        noise_ceiling[eval_mode] = compute_noise_ceiling(
            x_train_cfg=x_learn_eval,
            y_train=y_learn_eval,
            x_test_cfg=x_val_eval,
            y_test=y_val_eval,
        ).as_dict()

        baseline_preprocessor, _, _ = build_preprocessor(x_learn_eval, list(x_learn_eval.columns))
        for model_name, estimator in baseline_models.items():
            baseline_pipe = Pipeline(steps=[("pre", baseline_preprocessor), ("model", estimator)])
            baseline_pipe.fit(x_learn_eval, y_learn_eval)
            preds = baseline_pipe.predict(x_val_eval)
            m = metrics(y_val_eval, preds)
            m["r2_oos_train_mean"] = float(r2_oos_train_mean(y_val_eval, preds, train_mean_target))
            baseline_scores[(eval_mode, model_name)] = m

    rows: list[dict[str, Any]] = []
    for n_clusters in args.cluster_counts:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        learn_clusters = persona_learn.copy()
        val_clusters = persona_val.copy()
        learn_clusters["cluster"] = kmeans.fit_predict(learn_text_matrix)
        val_clusters["cluster"] = kmeans.predict(val_text_matrix)

        learn_dist = cluster_distribution_by_game(learn_clusters, n_clusters)
        val_dist = cluster_distribution_by_game(val_clusters, n_clusters)
        learn_oracle = learn_dist.reindex(learn_df["gameId"]).reset_index(drop=True)
        val_oracle = val_dist.reindex(val_df["gameId"]).reset_index(drop=True)

        style_cols = list(learn_oracle.columns)
        for col in style_cols:
            learn_oracle[col] = pd.to_numeric(learn_oracle[col], errors="coerce").fillna(0.0)
            val_oracle[col] = pd.to_numeric(val_oracle[col], errors="coerce").fillna(0.0)

        aug_sets: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]] = {}
        if args.style_source in {"oracle", "both"}:
            aug_sets["oracle"] = (
                learn_oracle.copy(),
                val_oracle.copy(),
                {"style_map_val_rmse": 0.0, "style_map_val_mae": 0.0},
            )

        if args.style_source in {"synthetic", "both"}:
            syn_learn_arr, syn_val_arr = fit_predict_synthetic_style(
                x_learn_cfg=x_learn_cfg,
                x_val_cfg=x_val_cfg,
                learn_style_true=learn_oracle,
                style_cols=style_cols,
                style_ridge_alpha=args.style_ridge_alpha,
                style_oof_folds=args.style_oof_folds,
            )
            syn_learn = pd.DataFrame(syn_learn_arr, columns=style_cols)
            syn_val = pd.DataFrame(syn_val_arr, columns=style_cols)
            style_map_val_rmse = float(
                np.sqrt(
                    mean_squared_error(
                        val_oracle[style_cols].to_numpy(dtype=float),
                        syn_val[style_cols].to_numpy(dtype=float),
                    )
                )
            )
            style_map_val_mae = float(
                mean_absolute_error(
                    val_oracle[style_cols].to_numpy(dtype=float),
                    syn_val[style_cols].to_numpy(dtype=float),
                )
            )
            aug_sets["synthetic"] = (
                syn_learn,
                syn_val,
                {
                    "style_map_val_rmse": style_map_val_rmse,
                    "style_map_val_mae": style_map_val_mae,
                },
            )

        for style_source, (learn_style, val_style, style_meta) in aug_sets.items():
            x_learn_aug = pd.concat([x_learn_cfg.reset_index(drop=True), learn_style], axis=1)
            x_val_aug = pd.concat([x_val_cfg.reset_index(drop=True), val_style], axis=1)
            for eval_mode in eval_modes:
                if eval_mode == "game":
                    x_learn_eval = x_learn_aug
                    y_learn_eval = y_learn
                    x_val_eval = x_val_aug
                    y_val_eval = y_val
                else:
                    x_learn_eval, y_learn_eval = aggregate_xy(
                        x_df=x_learn_aug,
                        y_arr=y_learn,
                        group_ids=learn_df[args.group_col],
                    )
                    x_val_eval, y_val_eval = aggregate_xy(
                        x_df=x_val_aug,
                        y_arr=y_val,
                        group_ids=val_df[args.group_col],
                    )

                aug_preprocessor, _, _ = build_preprocessor(x_learn_eval, list(x_learn_eval.columns))
                for model_name in ("linear", "ridge"):
                    if model_name == "linear":
                        aug_estimator = LinearRegression()
                    else:
                        aug_estimator = Ridge(alpha=args.ridge_alpha, random_state=42)
                    aug_pipe = Pipeline(steps=[("pre", aug_preprocessor), ("model", aug_estimator)])
                    aug_pipe.fit(x_learn_eval, y_learn_eval)
                    aug_pred = aug_pipe.predict(x_val_eval)
                    aug_scores = metrics(y_val_eval, aug_pred)
                    aug_scores["r2_oos_train_mean"] = float(
                        r2_oos_train_mean(
                            y_val_eval,
                            aug_pred,
                            baseline_counts[eval_mode]["train_mean_target"],
                        )
                    )
                    base_scores = baseline_scores[(eval_mode, model_name)]
                    rows.append(
                        {
                            "target": target,
                            "style_source": style_source,
                            "eval_granularity": eval_mode,
                            "group_col": args.group_col if eval_mode == "config_treatment" else "",
                            "model": model_name,
                            "k_clusters": int(n_clusters),
                            "baseline_r2": base_scores["r2"],
                            "baseline_rmse": base_scores["rmse"],
                            "baseline_mae": base_scores["mae"],
                            "augmented_r2": aug_scores["r2"],
                            "augmented_rmse": aug_scores["rmse"],
                            "augmented_mae": aug_scores["mae"],
                            "delta_r2": aug_scores["r2"] - base_scores["r2"],
                            "baseline_r2_oos_train_mean": base_scores["r2_oos_train_mean"],
                            "augmented_r2_oos_train_mean": aug_scores["r2_oos_train_mean"],
                            "delta_r2_oos_train_mean": (
                                aug_scores["r2_oos_train_mean"] - base_scores["r2_oos_train_mean"]
                            ),
                            "delta_rmse": aug_scores["rmse"] - base_scores["rmse"],
                            "delta_mae": aug_scores["mae"] - base_scores["mae"],
                            "missing_style_games_learn": int(
                                x_learn_aug[style_cols].sum(axis=1).eq(0).sum()
                            ),
                            "missing_style_games_val": int(
                                x_val_aug[style_cols].sum(axis=1).eq(0).sum()
                            ),
                            "n_train_eval_rows": baseline_counts[eval_mode]["n_train_eval_rows"],
                            "n_val_eval_rows": baseline_counts[eval_mode]["n_val_eval_rows"],
                            **style_meta,
                        }
                    )

    results_df = pd.DataFrame(rows).sort_values(
        ["style_source", "eval_granularity", "model", "delta_rmse", "k_clusters"]
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)

    best_rows = _best_rows(results_df)
    summary = {
        "target": target,
        "learn_analysis_csv": str(args.learn_analysis_csv),
        "val_analysis_csv": str(args.val_analysis_csv),
        "learn_persona_jsonl": str(args.learn_persona_jsonl),
        "val_persona_jsonl": str(args.val_persona_jsonl),
        "cluster_counts": args.cluster_counts,
        "style_source": args.style_source,
        "eval_granularity": args.eval_granularity,
        "group_col": args.group_col,
        "ridge_alpha": args.ridge_alpha,
        "style_ridge_alpha": args.style_ridge_alpha,
        "style_oof_folds": args.style_oof_folds,
        "noise_ceiling_by_eval_granularity": noise_ceiling,
        "noise_ceiling_sampling_config_treatment": (
            compute_sampling_noise_ceiling(
                y_obs=val_noise_df[target].to_numpy(),
                group_ids=val_noise_df[args.group_col],
                train_mean_target=float(learn_noise_df.groupby(args.group_col)[target].mean().mean()),
                group_col=args.group_col,
            ).as_dict()
            if args.group_col in learn_noise_df.columns and args.group_col in val_noise_df.columns
            else {}
        ),
        "valid_player_rows": {
            "learn_total_rows": int(len(learn_df)),
            "learn_valid_rows": int(learn_valid_mask.sum()),
            "val_total_rows": int(len(val_df)),
            "val_valid_rows": int(val_valid_mask.sum()),
        },
        "persona_coverage": {
            "learn_games_total": int(len(learn_games)),
            "learn_games_with_persona": int(persona_learn["experiment"].nunique()),
            "val_games_total": int(len(val_games)),
            "val_games_with_persona": int(persona_val["experiment"].nunique()),
        },
        "best_by_model": best_rows,
        "output_csv": str(args.output_csv),
    }
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(results_df.to_string(index=False))
    print(f"\nSaved results CSV: {args.output_csv}")
    print(f"Saved summary JSON: {args.output_json}")
    return results_df, summary


def main() -> None:
    args = parse_eval_args()
    run_eval(args)


if __name__ == "__main__":
    main()
