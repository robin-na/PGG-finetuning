from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_statistical.archetype_distribution_embedding.models.soft_cluster_gmm import (  # noqa: E402
    SoftClusterGMM,
    cluster_probability_columns,
)
from simulation_statistical.archetype_distribution_embedding.train.fit_env_model import (  # noqa: E402
    aggregate_player_weights_to_games,
    fit_env_distribution_model,
)
from simulation_statistical.archetype_distribution_embedding.utils.constants import (  # noqa: E402
    REQUIRED_CONFIG_COLUMNS,
)
from simulation_statistical.macro.analysis.supplemental import (  # noqa: E402
    LINEAR_CONFIG_FEATURES,
    build_regression_baseline_predictions,
)
from simulation_statistical.raw_behavior_cluster.run_raw_behavior_cluster_ablation import (  # noqa: E402
    build_raw_behavior_feature_table,
)


TEXT_CLUSTER_K = 6
RAW_CLUSTER_K = 4


@dataclass(frozen=True)
class SplitSpec:
    slug: str
    column: str
    kind: str  # "numeric" or "boolean"


SPLIT_SPECS: list[SplitSpec] = [
    SplitSpec(slug="player_count", column="CONFIG_playerCount", kind="numeric"),
    SplitSpec(slug="num_rounds", column="CONFIG_numRounds", kind="numeric"),
    SplitSpec(slug="all_or_nothing", column="CONFIG_allOrNothing", kind="boolean"),
    SplitSpec(slug="default_contrib_prop", column="CONFIG_defaultContribProp", kind="boolean"),
    SplitSpec(slug="reward_exists", column="CONFIG_rewardExists", kind="boolean"),
    SplitSpec(slug="show_n_rounds", column="CONFIG_showNRounds", kind="boolean"),
    SplitSpec(slug="show_punishment_id", column="CONFIG_showPunishmentId", kind="boolean"),
    SplitSpec(slug="show_other_summaries", column="CONFIG_showOtherSummaries", kind="boolean"),
    SplitSpec(slug="mpcr", column="CONFIG_MPCR", kind="numeric"),
]


def _normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "t", "yes"})
    )


def _load_analysis(root_data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    learn = pd.read_csv(root_data / "processed_data" / "df_analysis_learn.csv")
    val = pd.read_csv(root_data / "processed_data" / "df_analysis_val.csv")
    for df, wave in [(learn, "learn"), (val, "val")]:
        df["gameId"] = df["gameId"].astype(str)
        df["benchmark_id"] = df["CONFIG_treatmentName"].astype(str)
        df["wave"] = wave
    return learn, val


def _load_rows(root_data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    learn = pd.read_csv(root_data / "raw_data" / "learning_wave" / "player-rounds.csv")
    val = pd.read_csv(root_data / "raw_data" / "validation_wave" / "player-rounds.csv")
    for df in (learn, val):
        df["gameId"] = df["gameId"].astype(str)
        df["playerId"] = df["playerId"].astype(str)
    return learn, val


def _make_split_masks(
    learn_df: pd.DataFrame,
    val_df: pd.DataFrame,
    spec: SplitSpec,
) -> dict[str, tuple[pd.Series, pd.Series, dict[str, Any]]]:
    if spec.kind == "numeric":
        learn_values = pd.to_numeric(learn_df[spec.column], errors="coerce")
        val_values = pd.to_numeric(val_df[spec.column], errors="coerce")
        learn_median = float(learn_values.median())
        val_median = float(val_values.median())
        return {
            "low_to_high": (
                learn_values <= learn_median,
                val_values > val_median,
                {"learning_median": learn_median, "validation_median": val_median},
            ),
            "high_to_low": (
                learn_values > learn_median,
                val_values <= val_median,
                {"learning_median": learn_median, "validation_median": val_median},
            ),
        }
    if spec.kind == "boolean":
        learn_values = _normalize_bool(learn_df[spec.column])
        val_values = _normalize_bool(val_df[spec.column])
        return {
            "false_to_true": (
                ~learn_values,
                val_values,
                {},
            ),
            "true_to_false": (
                learn_values,
                ~val_values,
                {},
            ),
        }
    raise ValueError(f"Unsupported split kind: {spec.kind}")


def _build_benchmark_target_table(
    analysis_df: pd.DataFrame,
    rows_df: pd.DataFrame,
) -> pd.DataFrame:
    config_cols = [column for column in LINEAR_CONFIG_FEATURES if column in analysis_df.columns]
    config_table = (
        analysis_df.sort_values(["benchmark_id", "gameId"])
        .groupby("benchmark_id", as_index=False)
        .agg(
            gameId=("gameId", "first"),
            benchmark_game_count=("gameId", "nunique"),
            human_normalized_efficiency=("itt_relative_efficiency", "mean"),
            **{column: (column, "first") for column in config_cols},
        )
        .copy()
    )

    cfg = analysis_df[["gameId", "benchmark_id", "CONFIG_endowment"]].drop_duplicates(subset=["gameId"], keep="first").copy()
    joined = rows_df.merge(cfg, on="gameId", how="left", validate="many_to_one")
    joined["CONFIG_endowment"] = pd.to_numeric(joined["CONFIG_endowment"], errors="coerce").fillna(20.0)
    joined["contrib_rate"] = pd.to_numeric(joined["data.contribution"], errors="coerce").fillna(0.0) / joined["CONFIG_endowment"]
    rate_table = (
        joined.groupby("benchmark_id", as_index=False)
        .agg(human_mean_contribution_rate=("contrib_rate", "mean"))
        .copy()
    )
    return config_table.merge(rate_table, on="benchmark_id", how="left", validate="one_to_one")


def _benchmark_cluster_feature_table(
    cluster_game_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    cluster_cols: list[str],
) -> pd.DataFrame:
    tmp = cluster_game_df.copy()
    if "game_id" in tmp.columns:
        tmp = tmp.rename(columns={"game_id": "gameId"})
    tmp["gameId"] = tmp["gameId"].astype(str)
    lookup = analysis_df[["gameId", "benchmark_id"]].drop_duplicates(subset=["gameId"], keep="first")
    merged = tmp.merge(lookup, on="gameId", how="inner", validate="one_to_one")
    return merged.groupby("benchmark_id", as_index=False)[cluster_cols].mean()


def _fit_text_cluster_split(
    *,
    learn_game_ids: set[str],
    val_game_ids: set[str],
) -> dict[str, Any]:
    learn_player_game = pd.read_parquet(
        ROOT / "simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/player_game_table_learn_clean.parquet"
    )
    val_player_game = pd.read_parquet(
        ROOT / "simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/player_game_table_val_clean.parquet"
    )
    learn_embedding = pd.read_parquet(
        ROOT / "simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/embedding_matrix_learn.parquet"
    )
    val_embedding = pd.read_parquet(
        ROOT / "simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/embedding_matrix_val.parquet"
    )

    learn_player_game = learn_player_game[learn_player_game["game_id"].astype(str).isin(learn_game_ids)].copy()
    val_player_game = val_player_game[val_player_game["game_id"].astype(str).isin(val_game_ids)].copy()
    learn_embedding = learn_embedding[learn_embedding["game_id"].astype(str).isin(learn_game_ids)].copy()
    val_embedding = val_embedding[val_embedding["game_id"].astype(str).isin(val_game_ids)].copy()

    feature_cols = [column for column in learn_embedding.columns if column.startswith("embed_")]
    model = SoftClusterGMM(n_components=TEXT_CLUSTER_K, random_state=0)
    model.fit(learn_embedding[feature_cols].to_numpy())

    cluster_cols = cluster_probability_columns(TEXT_CLUSTER_K)
    learn_probs = model.predict_proba(learn_embedding[feature_cols].to_numpy())
    val_probs = model.predict_proba(val_embedding[feature_cols].to_numpy())

    learn_weights = learn_embedding[["row_id", "wave", "game_id", "player_id"]].copy()
    val_weights = val_embedding[["row_id", "wave", "game_id", "player_id"]].copy()
    for idx, column in enumerate(cluster_cols):
        learn_weights[column] = learn_probs[:, idx]
        val_weights[column] = val_probs[:, idx]

    learn_game = aggregate_player_weights_to_games(learn_weights, learn_player_game)
    val_game = aggregate_player_weights_to_games(val_weights, val_player_game)
    _, learn_pred, val_pred = fit_env_distribution_model(learn_game, val_game)

    env_mae = float(np.abs(val_pred[cluster_cols].to_numpy() - val_game[cluster_cols].to_numpy()).mean())
    env_l1 = float(np.abs(val_pred[cluster_cols].to_numpy() - val_game[cluster_cols].to_numpy()).sum(axis=1).mean())
    return {
        "cluster_cols": cluster_cols,
        "learn_game_actual": learn_game,
        "val_game_actual": val_game,
        "learn_game_pred": learn_pred,
        "val_game_pred": val_pred,
        "env_val_mean_cluster_mae": env_mae,
        "env_val_avg_l1": env_l1,
    }


def _fit_raw_cluster_split(
    *,
    learn_feature_df: pd.DataFrame,
    val_feature_df: pd.DataFrame,
    learn_game_ids: set[str],
    val_game_ids: set[str],
) -> dict[str, Any]:
    learn_df = learn_feature_df[learn_feature_df["game_id"].astype(str).isin(learn_game_ids)].copy()
    val_df = val_feature_df[val_feature_df["game_id"].astype(str).isin(val_game_ids)].copy()

    feature_cols = [
        "mean_contribution_rate",
        "zero_contribution_rate",
        "full_contribution_rate",
        "contribution_std_rate",
        "endgame_delta_rate",
        "mean_round_payoff_rate",
        "punish_row_rate",
        "reward_row_rate",
        "punish_units_per_round",
        "reward_units_per_round",
        "punish_targets_per_round",
        "reward_targets_per_round",
        "punish_received_row_rate",
        "reward_received_row_rate",
        "punish_received_units_per_round",
        "reward_received_units_per_round",
        "chat_active",
        "message_count_per_round",
        "word_count_per_round",
        "contribution_phase_messages_per_round",
        "outcome_phase_messages_per_round",
        "summary_phase_messages_per_round",
        "known_horizon",
    ]

    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    scaler = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    learn_x = scaler.fit_transform(learn_df[feature_cols])
    val_x = scaler.transform(val_df[feature_cols])

    model = SoftClusterGMM(n_components=RAW_CLUSTER_K, random_state=0)
    model.fit(learn_x)

    cluster_cols = cluster_probability_columns(RAW_CLUSTER_K)
    learn_probs = model.predict_proba(learn_x)
    val_probs = model.predict_proba(val_x)

    learn_weights = learn_df[["row_id", "wave", "game_id", "player_id"]].copy()
    val_weights = val_df[["row_id", "wave", "game_id", "player_id"]].copy()
    for idx, column in enumerate(cluster_cols):
        learn_weights[column] = learn_probs[:, idx]
        val_weights[column] = val_probs[:, idx]

    learn_player_game = pd.read_parquet(
        ROOT / "simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/player_game_table_learn_clean.parquet"
    )
    val_player_game = pd.read_parquet(
        ROOT / "simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/player_game_table_val_clean.parquet"
    )
    learn_player_game = learn_player_game[learn_player_game["game_id"].astype(str).isin(learn_game_ids)].copy()
    val_player_game = val_player_game[val_player_game["game_id"].astype(str).isin(val_game_ids)].copy()

    learn_game = aggregate_player_weights_to_games(learn_weights, learn_player_game)
    val_game = aggregate_player_weights_to_games(val_weights, val_player_game)
    _, learn_pred, val_pred = fit_env_distribution_model(learn_game, val_game)

    env_mae = float(np.abs(val_pred[cluster_cols].to_numpy() - val_game[cluster_cols].to_numpy()).mean())
    env_l1 = float(np.abs(val_pred[cluster_cols].to_numpy() - val_game[cluster_cols].to_numpy()).sum(axis=1).mean())
    return {
        "cluster_cols": cluster_cols,
        "learn_game_actual": learn_game,
        "val_game_actual": val_game,
        "learn_game_pred": learn_pred,
        "val_game_pred": val_pred,
        "env_val_mean_cluster_mae": env_mae,
        "env_val_avg_l1": env_l1,
    }


def _evaluate_ridge(
    *,
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    feature_cols: list[str],
    prediction_prefix: str,
) -> pd.DataFrame:
    _, summary = build_regression_baseline_predictions(
        learn_benchmark_table=train_table,
        val_benchmark_table=test_table,
        feature_cols=feature_cols,
        model_kind="ridge",
        prediction_prefix=prediction_prefix,
    )
    keep = summary[summary["metric"].isin({"contribution_rate", "normalized_efficiency"})].copy()
    return keep[["metric", "mae", "rmse", "corr"]]


def run_ood_direct_benchmarks() -> None:
    data_root = ROOT / "benchmark_statistical/data"
    out_dir = ROOT / "simulation_statistical/ood/artifacts/outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis_learn, analysis_val = _load_analysis(data_root)
    rows_learn, rows_val = _load_rows(data_root)

    raw_learn_features = build_raw_behavior_feature_table(
        player_rounds_csv=data_root / "raw_data" / "learning_wave" / "player-rounds.csv",
        analysis_csv=data_root / "processed_data" / "df_analysis_learn.csv",
        player_game_table_path=ROOT / "simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/player_game_table_learn_clean.parquet",
    )
    raw_val_features = build_raw_behavior_feature_table(
        player_rounds_csv=data_root / "raw_data" / "validation_wave" / "player-rounds.csv",
        analysis_csv=data_root / "processed_data" / "df_analysis_val.csv",
        player_game_table_path=ROOT / "simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/player_game_table_val_clean.parquet",
    )

    result_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for spec in SPLIT_SPECS:
        split_defs = _make_split_masks(analysis_learn, analysis_val, spec)
        for direction, (learn_mask, val_mask, extras) in split_defs.items():
            train_analysis = analysis_learn.loc[learn_mask].copy()
            test_analysis = analysis_val.loc[val_mask].copy()
            train_game_ids = set(train_analysis["gameId"].astype(str))
            test_game_ids = set(test_analysis["gameId"].astype(str))
            train_rows = rows_learn[rows_learn["gameId"].isin(train_game_ids)].copy()
            test_rows = rows_val[rows_val["gameId"].isin(test_game_ids)].copy()

            split_rows.append(
                {
                    "split_slug": spec.slug,
                    "direction": direction,
                    "column": spec.column,
                    "kind": spec.kind,
                    "n_train_games": len(train_game_ids),
                    "n_test_games": len(test_game_ids),
                    "n_train_benchmarks": int(train_analysis["benchmark_id"].nunique()),
                    "n_test_benchmarks": int(test_analysis["benchmark_id"].nunique()),
                    **extras,
                }
            )

            if len(train_game_ids) == 0 or len(test_game_ids) == 0:
                continue

            train_targets = _build_benchmark_target_table(train_analysis, train_rows)
            test_targets = _build_benchmark_target_table(test_analysis, test_rows)

            if len(train_targets) < 5 or len(test_targets) < 3:
                continue

            config_summary = _evaluate_ridge(
                train_table=train_targets,
                test_table=test_targets,
                feature_cols=[column for column in LINEAR_CONFIG_FEATURES if column in train_targets.columns],
                prediction_prefix=f"{spec.slug}_{direction}_ridge_config",
            )
            for _, row in config_summary.iterrows():
                result_rows.append(
                    {
                        "split_slug": spec.slug,
                        "direction": direction,
                        "target": row["metric"],
                        "model_label": "ridge_config",
                        "feature_source": "config",
                        "k": np.nan,
                        "oracle": False,
                        "env_val_mean_cluster_mae": np.nan,
                        "env_val_avg_l1": np.nan,
                        "mae": row["mae"],
                        "rmse": row["rmse"],
                        "corr": row["corr"],
                    }
                )

            text_fit = _fit_text_cluster_split(learn_game_ids=train_game_ids, val_game_ids=test_game_ids)
            for oracle, prefix, learn_game, test_game in [
                (False, "text_cluster_pred", text_fit["learn_game_pred"], text_fit["val_game_pred"]),
                (True, "text_cluster_oracle", text_fit["learn_game_actual"], text_fit["val_game_actual"]),
            ]:
                learn_cluster = _benchmark_cluster_feature_table(learn_game, train_analysis, text_fit["cluster_cols"])
                test_cluster = _benchmark_cluster_feature_table(test_game, test_analysis, text_fit["cluster_cols"])
                train_table = train_targets.merge(learn_cluster, on="benchmark_id", how="inner", validate="one_to_one")
                test_table = test_targets.merge(test_cluster, on="benchmark_id", how="inner", validate="one_to_one")
                for feature_name, feature_cols in [
                    ("cluster_only", list(text_fit["cluster_cols"])),
                    ("cluster_plus_config", [*text_fit["cluster_cols"], *[c for c in LINEAR_CONFIG_FEATURES if c in train_table.columns]]),
                ]:
                    summary = _evaluate_ridge(
                        train_table=train_table,
                        test_table=test_table,
                        feature_cols=feature_cols,
                        prediction_prefix=f"{spec.slug}_{direction}_{prefix}_{feature_name}",
                    )
                    for _, row in summary.iterrows():
                        result_rows.append(
                            {
                                "split_slug": spec.slug,
                                "direction": direction,
                                "target": row["metric"],
                                "model_label": f"ridge_{prefix}_{feature_name}",
                                "feature_source": feature_name,
                                "k": TEXT_CLUSTER_K,
                                "oracle": oracle,
                                "env_val_mean_cluster_mae": np.nan if oracle else text_fit["env_val_mean_cluster_mae"],
                                "env_val_avg_l1": np.nan if oracle else text_fit["env_val_avg_l1"],
                                "mae": row["mae"],
                                "rmse": row["rmse"],
                                "corr": row["corr"],
                            }
                        )

            raw_fit = _fit_raw_cluster_split(
                learn_feature_df=raw_learn_features,
                val_feature_df=raw_val_features,
                learn_game_ids=train_game_ids,
                val_game_ids=test_game_ids,
            )
            for oracle, prefix, learn_game, test_game in [
                (False, "raw_cluster_pred", raw_fit["learn_game_pred"], raw_fit["val_game_pred"]),
                (True, "raw_cluster_oracle", raw_fit["learn_game_actual"], raw_fit["val_game_actual"]),
            ]:
                learn_cluster = _benchmark_cluster_feature_table(learn_game, train_analysis, raw_fit["cluster_cols"])
                test_cluster = _benchmark_cluster_feature_table(test_game, test_analysis, raw_fit["cluster_cols"])
                train_table = train_targets.merge(learn_cluster, on="benchmark_id", how="inner", validate="one_to_one")
                test_table = test_targets.merge(test_cluster, on="benchmark_id", how="inner", validate="one_to_one")
                for feature_name, feature_cols in [
                    ("cluster_only", list(raw_fit["cluster_cols"])),
                    ("cluster_plus_config", [*raw_fit["cluster_cols"], *[c for c in LINEAR_CONFIG_FEATURES if c in train_table.columns]]),
                ]:
                    summary = _evaluate_ridge(
                        train_table=train_table,
                        test_table=test_table,
                        feature_cols=feature_cols,
                        prediction_prefix=f"{spec.slug}_{direction}_{prefix}_{feature_name}",
                    )
                    for _, row in summary.iterrows():
                        result_rows.append(
                            {
                                "split_slug": spec.slug,
                                "direction": direction,
                                "target": row["metric"],
                                "model_label": f"ridge_{prefix}_{feature_name}",
                                "feature_source": feature_name,
                                "k": RAW_CLUSTER_K,
                                "oracle": oracle,
                                "env_val_mean_cluster_mae": np.nan if oracle else raw_fit["env_val_mean_cluster_mae"],
                                "env_val_avg_l1": np.nan if oracle else raw_fit["env_val_avg_l1"],
                                "mae": row["mae"],
                                "rmse": row["rmse"],
                                "corr": row["corr"],
                            }
                        )

    split_df = pd.DataFrame(split_rows).sort_values(["split_slug", "direction"]).reset_index(drop=True)
    result_df = pd.DataFrame(result_rows).sort_values(["split_slug", "direction", "target", "model_label"]).reset_index(drop=True)
    split_df.to_csv(out_dir / "wave_anchored_ood_split_summary.csv", index=False)
    result_df.to_csv(out_dir / "wave_anchored_ood_results.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for target in sorted(result_df["target"].unique()):
        subset = result_df[(result_df["target"] == target) & (~result_df["oracle"])].copy()
        for model_label in sorted(subset["model_label"].unique()):
            model_df = subset[subset["model_label"] == model_label]
            summary_rows.append(
                {
                    "target": target,
                    "model_label": model_label,
                    "n_splits": int(len(model_df)),
                    "mean_mae": float(model_df["mae"].mean()),
                    "median_mae": float(model_df["mae"].median()),
                    "mean_rmse": float(model_df["rmse"].mean()),
                    "mean_corr": float(model_df["corr"].mean()),
                }
            )
    summary_df = pd.DataFrame(summary_rows).sort_values(["target", "mean_mae"]).reset_index(drop=True)
    summary_df.to_csv(out_dir / "wave_anchored_ood_summary.csv", index=False)

    lines = [
        "# Wave-Anchored OOD Direct Benchmark Results",
        "",
        "This report evaluates direct macro predictors under true wave-anchored one-factor OOD splits built from `benchmark_statistical/data`.",
        "",
        "Setup:",
        f"- text-cluster model uses `K={TEXT_CLUSTER_K}` and refits the GMM + Dirichlet env model on each split train set",
        f"- raw-behavior model uses `K={RAW_CLUSTER_K}` and refits the GMM + Dirichlet env model on each split train set",
        "- all reported main results below are deployable ridge models, not oracle mixtures",
        "",
        "## Mean across OOD splits",
        "",
    ]
    for target in ["contribution_rate", "normalized_efficiency"]:
        lines.append(f"### `{target}`")
        lines.append("")
        sub = summary_df[summary_df["target"] == target]
        for _, row in sub.iterrows():
            lines.append(
                f"- `{row['model_label']}`: mean MAE `{row['mean_mae']:.4f}`, mean RMSE `{row['mean_rmse']:.4f}`, mean corr `{row['mean_corr']:.4f}` across `{int(row['n_splits'])}` splits"
            )
        lines.append("")

    (out_dir / "wave_anchored_ood_report.md").write_text("\n".join(lines).strip() + "\n")


if __name__ == "__main__":
    run_ood_direct_benchmarks()
