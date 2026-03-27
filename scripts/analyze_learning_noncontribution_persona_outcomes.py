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

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:  # pragma: no cover
    from sklearn.model_selection import GroupKFold

    HAS_STRATIFIED_GROUP_KFOLD = False


DEFAULT_EXTREME_FLAGS = Path("reports/learning_extreme_contributors/player_game_extreme_strategy_flags.csv")
DEFAULT_NONEXTREME_LABELS = Path("reports/learning_nonextreme_patterns/valid_only_player_game_pattern_labels.csv")
DEFAULT_TAG_BASE = Path("Persona/archetype_retrieval/learning_wave")
DEFAULT_CLUSTER_CATALOG_BASE = Path("Persona/cluster/tag_section_clusters_openai_learn")
DEFAULT_OUTPUT_DIR = Path("reports/learning_noncontribution_persona_outcomes")

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
            "Use non-contribution persona clusters to explain and predict whether players "
            "become always-full, always-zero, binary switchers, or other."
        )
    )
    parser.add_argument("--extreme-flags", type=Path, default=DEFAULT_EXTREME_FLAGS)
    parser.add_argument("--nonextreme-labels", type=Path, default=DEFAULT_NONEXTREME_LABELS)
    parser.add_argument("--tag-base", type=Path, default=DEFAULT_TAG_BASE)
    parser.add_argument("--cluster-catalog-base", type=Path, default=DEFAULT_CLUSTER_CATALOG_BASE)
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


def load_cluster_titles(path: Path) -> dict[int, str]:
    with path.open() as handle:
        payload = json.load(handle)
    return {int(row["cluster_id"]): str(row["cluster_title"]) for row in payload}


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

    merged = flags[["gameId", "playerId", "pattern_7class"] + config_cols].merge(
        nonextreme[["gameId", "playerId", "pattern_nonextreme"]],
        on=["gameId", "playerId"],
        how="left",
    )
    merged["pattern_7class"] = merged["pattern_7class"].fillna(merged["pattern_nonextreme"])
    merged = merged.drop(columns=["pattern_nonextreme"])

    merged["outcome_4class"] = merged["pattern_7class"].where(
        merged["pattern_7class"].isin({"always_full", "always_zero", "binary_0_20"}),
        other="other",
    )
    merged["gameId"] = merged["gameId"].astype(str)
    merged["playerId"] = merged["playerId"].astype(str)

    for config_col in config_cols:
        merged[config_col] = merged[config_col].map(normalize_config_value)

    return merged.dropna(subset=["pattern_7class", "outcome_4class"]).copy()


def load_tag_rows(
    *,
    tag_base: Path,
    cluster_catalog_base: Path,
    tag: str,
) -> pd.DataFrame:
    cluster_path_candidates = sorted((tag_base / tag).glob("*_clustered.jsonl"))
    if not cluster_path_candidates:
        raise FileNotFoundError(f"No clustered jsonl found for tag {tag}")
    cluster_path = cluster_path_candidates[0]
    df = pd.read_json(cluster_path, lines=True)
    df["gameId"] = df["gameId"].astype(str)
    df["playerId"] = df["playerId"].astype(str)

    catalog_path = cluster_catalog_base / tag / "cluster_catalog_polished.json"
    if catalog_path.exists():
        cluster_titles = load_cluster_titles(catalog_path)
        df["cluster_title"] = df["cluster_id"].map(cluster_titles).fillna(df["cluster_title"])

    return df[["gameId", "playerId", "cluster_id", "cluster_title"]].copy()


def summarize_coverage(tag: str, df: pd.DataFrame) -> dict[str, object]:
    class_counts = df["outcome_4class"].value_counts()
    return {
        "tag": tag,
        "n_player_game_rows": int(len(df)),
        "n_games": int(df["gameId"].nunique()),
        "n_clusters": int(df["cluster_id"].nunique()),
        "share_always_full": float((df["outcome_4class"] == "always_full").mean()),
        "share_always_zero": float((df["outcome_4class"] == "always_zero").mean()),
        "share_binary_0_20": float((df["outcome_4class"] == "binary_0_20").mean()),
        "share_other": float((df["outcome_4class"] == "other").mean()),
        "min_class_count": int(class_counts.min()),
    }


def summarize_cluster_purity(tag: str, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dist_rows: list[dict[str, object]] = []
    purity_rows: list[dict[str, object]] = []

    for (cluster_id, cluster_title), group in df.groupby(["cluster_id", "cluster_title"], sort=True):
        shares = group["outcome_4class"].value_counts(normalize=True)
        for outcome, count in group["outcome_4class"].value_counts().items():
            dist_rows.append(
                {
                    "tag": tag,
                    "cluster_id": int(cluster_id),
                    "cluster_title": str(cluster_title),
                    "outcome_4class": str(outcome),
                    "n_player_game_rows": int(count),
                    "share_within_cluster": float(count / len(group)),
                }
            )
        purity_rows.append(
            {
                "tag": tag,
                "cluster_id": int(cluster_id),
                "cluster_title": str(cluster_title),
                "n_player_game_rows": int(len(group)),
                "n_outcomes": int(group["outcome_4class"].nunique()),
                "dominant_outcome": str(shares.index[0]),
                "dominant_share": float(shares.iloc[0]),
                "entropy_bits": float(-(shares * np.log2(shares)).sum()),
            }
        )

    return (
        pd.DataFrame(dist_rows).sort_values(["tag", "cluster_id", "share_within_cluster"], ascending=[True, True, False]),
        pd.DataFrame(purity_rows).sort_values(["tag", "cluster_id"]).reset_index(drop=True),
    )


def summarize_cluster_config_shifts(tag: str, df: pd.DataFrame, config_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (cluster_id, cluster_title), cluster_group in df.groupby(["cluster_id", "cluster_title"], sort=True):
        for config_col in config_cols:
            sub = cluster_group.dropna(subset=[config_col]).copy()
            if sub[config_col].nunique() != 2:
                continue

            values = sorted(sub[config_col].unique(), key=value_sort_key)
            value_details: dict[object, dict[str, object]] = {}

            for value, value_group in sub.groupby(config_col):
                shares = value_group["outcome_4class"].value_counts(normalize=True)
                value_details[value] = {
                    "n_rows": int(len(value_group)),
                    "dominant_outcome": str(shares.index[0]),
                    "dominant_share": float(shares.iloc[0]),
                    "shares": {str(idx): float(share) for idx, share in shares.items()},
                }

            value0, value1 = values
            share0 = value_details[value0]["shares"]
            share1 = value_details[value1]["shares"]
            outcomes = sorted(set(share0) | set(share1))
            tvd = 0.5 * sum(
                abs(float(share0.get(outcome, 0.0)) - float(share1.get(outcome, 0.0)))
                for outcome in outcomes
            )

            rows.append(
                {
                    "tag": tag,
                    "cluster_id": int(cluster_id),
                    "cluster_title": str(cluster_title),
                    "config_feature": config_col,
                    "config_value_0": str(value0),
                    "config_value_1": str(value1),
                    "n_rows_0": int(value_details[value0]["n_rows"]),
                    "n_rows_1": int(value_details[value1]["n_rows"]),
                    "dominant_outcome_0": str(value_details[value0]["dominant_outcome"]),
                    "dominant_outcome_1": str(value_details[value1]["dominant_outcome"]),
                    "dominant_share_0": float(value_details[value0]["dominant_share"]),
                    "dominant_share_1": float(value_details[value1]["dominant_share"]),
                    "dominant_outcome_changes": bool(
                        value_details[value0]["dominant_outcome"] != value_details[value1]["dominant_outcome"]
                    ),
                    "total_variation_distance": float(tvd),
                    "always_full_gap": float(share1.get("always_full", 0.0) - share0.get("always_full", 0.0)),
                    "always_zero_gap": float(share1.get("always_zero", 0.0) - share0.get("always_zero", 0.0)),
                    "binary_0_20_gap": float(share1.get("binary_0_20", 0.0) - share0.get("binary_0_20", 0.0)),
                    "other_gap": float(share1.get("other", 0.0) - share0.get("other", 0.0)),
                }
            )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["tag", "total_variation_distance", "dominant_outcome_changes", "cluster_id"],
            ascending=[True, False, False, True],
        )
        .reset_index(drop=True)
    )


def make_model(feature_cols: Iterable[str], balanced: bool = False) -> Pipeline:
    feature_cols = list(feature_cols)
    categorical = feature_cols
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
                categorical,
            )
        ]
    )

    classifier = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced" if balanced else None,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def make_splitter(n_splits: int):
    if HAS_STRATIFIED_GROUP_KFOLD:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    return GroupKFold(n_splits=n_splits)


def predict_probabilities(
    *,
    model_name: str,
    feature_cols: list[str],
    df: pd.DataFrame,
    class_order: list[str],
) -> dict[str, object]:
    X = df[feature_cols].copy()
    y = df["outcome_4class"].astype(str)
    groups = df["gameId"].astype(str)

    class_counts = y.value_counts()
    n_splits = int(min(5, groups.nunique(), class_counts.min()))
    if n_splits < 2:
        raise ValueError("Not enough class support for grouped CV.")

    splitter = make_splitter(n_splits)

    if model_name == "dummy_prior":
        estimator = DummyClassifier(strategy="prior")
    else:
        estimator = make_model(feature_cols=feature_cols, balanced=False)

    proba = np.zeros((len(df), len(class_order)), dtype=float)
    pred = np.empty(len(df), dtype=object)
    fold_rows: list[dict[str, object]] = []

    split_iter = (
        splitter.split(X, y, groups)
        if HAS_STRATIFIED_GROUP_KFOLD
        else splitter.split(X, y, groups)
    )

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        estimator.fit(X_train, y_train)
        fold_pred = estimator.predict(X_test)
        pred[test_idx] = fold_pred

        fold_proba = np.zeros((len(test_idx), len(class_order)), dtype=float)
        learned_classes = list(estimator.classes_)
        learned_proba = estimator.predict_proba(X_test)
        for class_idx, learned_class in enumerate(learned_classes):
            fold_proba[:, class_order.index(str(learned_class))] = learned_proba[:, class_idx]
        fold_proba = np.clip(fold_proba, 1e-15, 1.0)
        fold_proba = fold_proba / fold_proba.sum(axis=1, keepdims=True)
        proba[test_idx] = fold_proba

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_test_rows": int(len(test_idx)),
                "n_test_games": int(groups.iloc[test_idx].nunique()),
                "accuracy": float(accuracy_score(y_test, fold_pred)),
                "macro_f1": float(f1_score(y_test, fold_pred, labels=class_order, average="macro", zero_division=0)),
                "log_loss": float(log_loss(y_test, fold_proba, labels=class_order)),
            }
        )

    return {
        "proba": proba,
        "pred": pred,
        "fold_metrics": pd.DataFrame(fold_rows),
        "n_splits": n_splits,
    }


def evaluate_predictive_models(tag: str, df: pd.DataFrame, config_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_sets = {
        "dummy_prior": [],
        "config_only": config_cols,
        "tag_only": ["cluster_id"],
        "config_plus_tag": config_cols + ["cluster_id"],
    }

    class_order = [label for label in OUTCOME_ORDER if label in set(df["outcome_4class"])]
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[pd.DataFrame] = []

    for model_name, feature_cols in feature_sets.items():
        eval_df = df.copy()
        if model_name == "dummy_prior":
            eval_df = eval_df[["gameId", "outcome_4class"]].copy()
        else:
            eval_df = eval_df[["gameId", "outcome_4class"] + feature_cols].copy()
            for col in feature_cols:
                eval_df[col] = eval_df[col].astype("string")

        result = predict_probabilities(
            model_name=model_name,
            feature_cols=feature_cols,
            df=eval_df,
            class_order=class_order,
        )
        proba = result["proba"]
        pred = result["pred"]
        y_true = eval_df["outcome_4class"].astype(str).to_numpy()

        summary_rows.append(
            {
                "tag": tag,
                "model_name": model_name,
                "n_player_game_rows": int(len(eval_df)),
                "n_games": int(eval_df["gameId"].nunique()),
                "n_splits": int(result["n_splits"]),
                "accuracy": float(accuracy_score(y_true, pred)),
                "macro_f1": float(f1_score(y_true, pred, labels=class_order, average="macro", zero_division=0)),
                "log_loss": float(log_loss(y_true, proba, labels=class_order)),
            }
        )

        fold_metric_df = result["fold_metrics"].copy()
        fold_metric_df["tag"] = tag
        fold_metric_df["model_name"] = model_name
        fold_rows.append(fold_metric_df)

    return pd.DataFrame(summary_rows), pd.concat(fold_rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_cols = list(dict.fromkeys(args.configs))
    behavior_labels = build_behavior_labels(
        extreme_flags_path=args.extreme_flags,
        nonextreme_labels_path=args.nonextreme_labels,
        config_cols=config_cols,
    )

    coverage_rows: list[dict[str, object]] = []
    purity_frames: list[pd.DataFrame] = []
    dist_frames: list[pd.DataFrame] = []
    shift_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    metric_fold_frames: list[pd.DataFrame] = []

    for tag in args.tags:
        tag_rows = load_tag_rows(
            tag_base=args.tag_base,
            cluster_catalog_base=args.cluster_catalog_base,
            tag=tag,
        )
        merged = behavior_labels.merge(tag_rows, on=["gameId", "playerId"], how="inner")
        if merged.empty:
            continue

        coverage_rows.append(summarize_coverage(tag, merged))
        dist_df, purity_df = summarize_cluster_purity(tag, merged)
        shift_df = summarize_cluster_config_shifts(tag, merged, config_cols=config_cols)
        metric_df, metric_fold_df = evaluate_predictive_models(tag, merged, config_cols=config_cols)

        dist_frames.append(dist_df)
        purity_frames.append(purity_df)
        if not shift_df.empty:
            shift_frames.append(shift_df)
        metric_frames.append(metric_df)
        metric_fold_frames.append(metric_fold_df)

    coverage_df = pd.DataFrame(coverage_rows).sort_values("n_player_game_rows", ascending=False)
    purity_df = pd.concat(purity_frames, ignore_index=True) if purity_frames else pd.DataFrame()
    dist_df = pd.concat(dist_frames, ignore_index=True) if dist_frames else pd.DataFrame()
    shift_df = pd.concat(shift_frames, ignore_index=True) if shift_frames else pd.DataFrame()
    metric_df = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    metric_fold_df = pd.concat(metric_fold_frames, ignore_index=True) if metric_fold_frames else pd.DataFrame()

    if not purity_df.empty:
        tag_stability_df = (
            purity_df.groupby("tag", as_index=False)
            .agg(
                n_clusters=("cluster_id", "nunique"),
                mean_cluster_dominant_share=("dominant_share", "mean"),
                median_cluster_dominant_share=("dominant_share", "median"),
                max_cluster_dominant_share=("dominant_share", "max"),
                mean_cluster_entropy_bits=("entropy_bits", "mean"),
            )
            .sort_values("mean_cluster_dominant_share", ascending=False)
        )
    else:
        tag_stability_df = pd.DataFrame()

    if not shift_df.empty:
        tag_shift_df = (
            shift_df.groupby("tag", as_index=False)
            .agg(
                n_cluster_config_pairs=("cluster_id", "size"),
                share_dominant_outcome_changes=("dominant_outcome_changes", "mean"),
                max_total_variation_distance=("total_variation_distance", "max"),
                mean_total_variation_distance=("total_variation_distance", "mean"),
            )
            .sort_values(
                ["share_dominant_outcome_changes", "max_total_variation_distance"],
                ascending=[False, False],
            )
        )
    else:
        tag_shift_df = pd.DataFrame()

    if not metric_df.empty:
        metric_pivot_df = (
            metric_df.pivot(index="tag", columns="model_name", values=["accuracy", "macro_f1", "log_loss"])
            .sort_index()
        )
        metric_pivot_df.columns = [f"{metric}_{model}" for metric, model in metric_pivot_df.columns]
        metric_pivot_df = metric_pivot_df.reset_index()
        if {"macro_f1_config_plus_tag", "macro_f1_config_only"}.issubset(metric_pivot_df.columns):
            metric_pivot_df["macro_f1_gain_vs_config"] = (
                metric_pivot_df["macro_f1_config_plus_tag"] - metric_pivot_df["macro_f1_config_only"]
            )
        if {"log_loss_config_plus_tag", "log_loss_config_only"}.issubset(metric_pivot_df.columns):
            metric_pivot_df["log_loss_improvement_vs_config"] = (
                metric_pivot_df["log_loss_config_only"] - metric_pivot_df["log_loss_config_plus_tag"]
            )
        metric_pivot_df = metric_pivot_df.sort_values("macro_f1_gain_vs_config", ascending=False)
    else:
        metric_pivot_df = pd.DataFrame()

    behavior_labels.to_csv(args.output_dir / "valid_only_complete_behavior_labels_4class.csv", index=False)
    coverage_df.to_csv(args.output_dir / "tag_coverage_summary.csv", index=False)
    tag_stability_df.to_csv(args.output_dir / "tag_cluster_stability_summary.csv", index=False)
    tag_shift_df.to_csv(args.output_dir / "tag_cluster_config_shift_overview.csv", index=False)
    dist_df.to_csv(args.output_dir / "cluster_outcome_distribution.csv", index=False)
    purity_df.to_csv(args.output_dir / "cluster_outcome_purity.csv", index=False)
    shift_df.to_csv(args.output_dir / "cluster_outcome_config_shifts.csv", index=False)
    metric_df.to_csv(args.output_dir / "predictive_metrics_summary.csv", index=False)
    metric_fold_df.to_csv(args.output_dir / "predictive_metrics_by_fold.csv", index=False)
    metric_pivot_df.to_csv(args.output_dir / "predictive_metrics_wide.csv", index=False)

    manifest = {
        "extreme_flags": str(args.extreme_flags),
        "nonextreme_labels": str(args.nonextreme_labels),
        "tag_base": str(args.tag_base),
        "cluster_catalog_base": str(args.cluster_catalog_base),
        "output_dir": str(args.output_dir),
        "binary_configs": config_cols,
        "tags": args.tags,
        "outcome_4class": OUTCOME_ORDER,
        "notes": [
            "Outcome labels are always_full, always_zero, binary_0_20, and other (all remaining patterns).",
            "Only valid_only players with complete round coverage are included.",
            "Each playerId appears in exactly one game in the learning wave, so this is type-level rather than person-level analysis.",
            "Predictive evaluation uses grouped cross-validation by gameId to avoid train/test leakage within the same game.",
            "Non-contribution tag coverage is conditional on the availability of that section in the persona artifacts.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
