#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_EXTREME_FLAGS = Path("reports/learning_extreme_contributors/player_game_extreme_strategy_flags.csv")
DEFAULT_NONEXTREME_LABELS = Path("reports/learning_nonextreme_patterns/valid_only_player_game_pattern_labels.csv")
DEFAULT_TAG_BASE = Path("Persona/archetype_retrieval/learning_wave")
DEFAULT_OUTPUT_DIR = Path("reports/learning_noncontribution_persona_ood")

DEFAULT_BINARY_CONFIGS = [
    "CONFIG_chat",
    "CONFIG_punishmentExists",
    "CONFIG_showRewardId",
    "CONFIG_showNRounds",
    "CONFIG_showPunishmentId",
    "CONFIG_rewardExists",
    "CONFIG_showOtherSummaries",
    "CONFIG_allOrNothing",
    "CONFIG_defaultContribProp",
]

DEFAULT_TAGS = [
    "COMMUNICATION",
    "PUNISHMENT",
    "REWARD",
    "RESPONSE_TO_END_GAME",
    "RESPONSE_TO_OTHERS_OUTCOME",
    "RESPONSE_TO_PUNISHER",
    "RESPONSE_TO_REWARDER",
]

OUTCOME_ORDER = ["always_full", "always_zero", "binary_0_20", "other"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate how well non-contribution persona tags transfer across unseen config values "
            "for extreme/binary contribution behavior."
        )
    )
    parser.add_argument("--extreme-flags", type=Path, default=DEFAULT_EXTREME_FLAGS)
    parser.add_argument("--nonextreme-labels", type=Path, default=DEFAULT_NONEXTREME_LABELS)
    parser.add_argument("--tag-base", type=Path, default=DEFAULT_TAG_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_BINARY_CONFIGS))
    parser.add_argument("--tags", nargs="+", default=list(DEFAULT_TAGS))
    return parser.parse_args()


def normalize_config_value(value: object) -> object:
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return int(numeric)
    return numeric


def value_sort_key(value: object) -> tuple[int, object]:
    if pd.isna(value):
        return (3, "")
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float, np.number)):
        return (1, float(value))
    return (2, str(value))


def build_behavior_labels(
    *,
    extreme_flags_path: Path,
    nonextreme_labels_path: Path,
    config_cols: list[str],
) -> pd.DataFrame:
    flags = pd.read_csv(extreme_flags_path)
    flags = flags[
        flags["valid_number_of_starting_players"].astype(bool)
        & flags["complete_round_coverage"].astype(bool)
    ].copy()

    flags["pattern_7class"] = pd.Series(pd.NA, index=flags.index, dtype="object")
    flags.loc[flags["always_full"].astype(bool), "pattern_7class"] = "always_full"
    flags.loc[flags["always_zero"].astype(bool), "pattern_7class"] = "always_zero"

    nonextreme = pd.read_csv(nonextreme_labels_path).rename(columns={"pattern": "pattern_nonextreme"})
    labels = flags[["gameId", "playerId", "pattern_7class"] + config_cols].merge(
        nonextreme[["gameId", "playerId", "pattern_nonextreme"]],
        on=["gameId", "playerId"],
        how="left",
    )
    labels["pattern_7class"] = labels["pattern_7class"].fillna(labels["pattern_nonextreme"])
    labels = labels.drop(columns=["pattern_nonextreme"])
    labels["outcome_4class"] = labels["pattern_7class"].where(
        labels["pattern_7class"].isin({"always_full", "always_zero", "binary_0_20"}),
        other="other",
    )
    labels["gameId"] = labels["gameId"].astype(str)
    labels["playerId"] = labels["playerId"].astype(str)
    for col in config_cols:
        labels[col] = labels[col].map(normalize_config_value)
    return labels.dropna(subset=["pattern_7class", "outcome_4class"]).copy()


def load_tag_rows(tag_base: Path, tag: str) -> pd.DataFrame:
    matches = sorted((tag_base / tag).glob("*_clustered.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No cluster file found for tag {tag}")
    df = pd.read_json(matches[0], lines=True)
    df["gameId"] = df["gameId"].astype(str)
    df["playerId"] = df["playerId"].astype(str)
    return df[["gameId", "playerId", "cluster_id", "cluster_title"]].copy()


def make_model(feature_cols: Iterable[str]) -> Pipeline:
    feature_cols = list(feature_cols)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                feature_cols,
            )
        ]
    )
    classifier = LogisticRegression(max_iter=2000, solver="lbfgs")
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def fit_and_score(
    *,
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, object]:
    y_train = train_df["outcome_4class"].astype(str)
    y_test = test_df["outcome_4class"].astype(str)
    class_order = [label for label in OUTCOME_ORDER if label in set(y_train) | set(y_test)]

    if model_name == "dummy_prior":
        estimator = DummyClassifier(strategy="prior")
        X_train = pd.DataFrame(index=train_df.index)
        X_test = pd.DataFrame(index=test_df.index)
    else:
        estimator = make_model(feature_cols)
        X_train = train_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()
        for col in feature_cols:
            X_train[col] = X_train[col].astype("string")
            X_test[col] = X_test[col].astype("string")

    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)

    proba = np.zeros((len(test_df), len(class_order)), dtype=float)
    learned_classes = list(estimator.classes_)
    learned_proba = estimator.predict_proba(X_test)
    for idx, learned_class in enumerate(learned_classes):
        proba[:, class_order.index(str(learned_class))] = learned_proba[:, idx]
    proba = np.clip(proba, 1e-15, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, labels=class_order, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_test, proba, labels=class_order)),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_train_games": int(train_df["gameId"].nunique()),
        "n_test_games": int(test_df["gameId"].nunique()),
        "train_class_counts": json.dumps(y_train.value_counts().to_dict(), sort_keys=True),
        "test_class_counts": json.dumps(y_test.value_counts().to_dict(), sort_keys=True),
    }


def evaluate_tag_ood(tag: str, df: pd.DataFrame, config_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for heldout_config in config_cols:
        available = df.dropna(subset=[heldout_config]).copy()
        values = sorted(available[heldout_config].unique(), key=value_sort_key)
        if len(values) != 2:
            continue

        other_configs = [col for col in config_cols if col != heldout_config]

        for heldout_value in values:
            train_df = available.loc[available[heldout_config] != heldout_value].copy()
            test_df = available.loc[available[heldout_config] == heldout_value].copy()

            if train_df.empty or test_df.empty:
                continue
            if train_df["outcome_4class"].nunique() < 2:
                continue

            feature_sets = {
                "dummy_prior": [],
                "config_only": other_configs,
                "tag_only": ["cluster_id"],
                "config_plus_tag": other_configs + ["cluster_id"],
            }

            for model_name, feature_cols in feature_sets.items():
                score = fit_and_score(
                    model_name=model_name,
                    train_df=train_df,
                    test_df=test_df,
                    feature_cols=feature_cols,
                )
                rows.append(
                    {
                        "tag": tag,
                        "heldout_config": heldout_config,
                        "heldout_value": str(heldout_value),
                        "model_name": model_name,
                        **score,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_cols = list(dict.fromkeys(args.configs))
    labels = build_behavior_labels(
        extreme_flags_path=args.extreme_flags,
        nonextreme_labels_path=args.nonextreme_labels,
        config_cols=config_cols,
    )

    metric_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []

    for tag in args.tags:
        tag_rows = load_tag_rows(args.tag_base, tag)
        merged = labels.merge(tag_rows, on=["gameId", "playerId"], how="inner")
        if merged.empty:
            continue

        coverage_rows.append(
            {
                "tag": tag,
                "n_player_game_rows": int(len(merged)),
                "n_games": int(merged["gameId"].nunique()),
                "n_clusters": int(merged["cluster_id"].nunique()),
            }
        )
        metric_frames.append(evaluate_tag_ood(tag, merged, config_cols=config_cols))

    metrics_df = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    coverage_df = pd.DataFrame(coverage_rows).sort_values("n_player_game_rows", ascending=False)

    if not metrics_df.empty:
        summary_df = (
            metrics_df.groupby(["tag", "model_name"], as_index=False)
            .agg(
                mean_accuracy=("accuracy", "mean"),
                mean_macro_f1=("macro_f1", "mean"),
                mean_log_loss=("log_loss", "mean"),
                n_splits=("heldout_config", "size"),
            )
            .sort_values(["tag", "mean_macro_f1"], ascending=[True, False])
        )

        wide_df = (
            summary_df.pivot(index="tag", columns="model_name", values=["mean_accuracy", "mean_macro_f1", "mean_log_loss"])
            .sort_index()
        )
        wide_df.columns = [f"{metric}_{model}" for metric, model in wide_df.columns]
        wide_df = wide_df.reset_index()
        if {"mean_macro_f1_config_plus_tag", "mean_macro_f1_config_only"}.issubset(wide_df.columns):
            wide_df["mean_macro_f1_gain_vs_config"] = (
                wide_df["mean_macro_f1_config_plus_tag"] - wide_df["mean_macro_f1_config_only"]
            )
        if {"mean_log_loss_config_plus_tag", "mean_log_loss_config_only"}.issubset(wide_df.columns):
            wide_df["mean_log_loss_improvement_vs_config"] = (
                wide_df["mean_log_loss_config_only"] - wide_df["mean_log_loss_config_plus_tag"]
            )
        wide_df = wide_df.sort_values("mean_macro_f1_gain_vs_config", ascending=False)

        config_summary_df = (
            metrics_df.groupby(["tag", "heldout_config", "model_name"], as_index=False)
            .agg(
                mean_accuracy=("accuracy", "mean"),
                mean_macro_f1=("macro_f1", "mean"),
                mean_log_loss=("log_loss", "mean"),
                n_directional_splits=("heldout_value", "size"),
            )
            .sort_values(["tag", "heldout_config", "mean_macro_f1"], ascending=[True, True, False])
        )
    else:
        summary_df = pd.DataFrame()
        wide_df = pd.DataFrame()
        config_summary_df = pd.DataFrame()

    labels.to_csv(args.output_dir / "valid_only_complete_behavior_labels_4class.csv", index=False)
    coverage_df.to_csv(args.output_dir / "tag_coverage_summary.csv", index=False)
    metrics_df.to_csv(args.output_dir / "ood_metrics_by_split.csv", index=False)
    summary_df.to_csv(args.output_dir / "ood_metrics_summary.csv", index=False)
    wide_df.to_csv(args.output_dir / "ood_metrics_wide.csv", index=False)
    config_summary_df.to_csv(args.output_dir / "ood_metrics_by_config.csv", index=False)

    manifest = {
        "extreme_flags": str(args.extreme_flags),
        "nonextreme_labels": str(args.nonextreme_labels),
        "tag_base": str(args.tag_base),
        "output_dir": str(args.output_dir),
        "binary_configs": config_cols,
        "tags": args.tags,
        "notes": [
            "This is the closest feasible proxy to a seed-games-only benchmark in this dataset.",
            "No participant repeats across games, so true same-person seed-to-target transfer is impossible.",
            "For each tag and held-out config value, models are trained on the opposite config value only.",
            "config_only and config_plus_tag exclude the held-out config from features, so this is genuine value-level extrapolation on that dimension.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
