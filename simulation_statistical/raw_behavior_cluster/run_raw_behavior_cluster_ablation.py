from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_statistical.archetype_distribution_embedding.models.soft_cluster_gmm import (
    SoftClusterGMM,
    cluster_probability_columns,
)
from simulation_statistical.archetype_distribution_embedding.train.fit_env_model import (
    aggregate_player_weights_to_games,
    fit_env_distribution_model,
)
from simulation_statistical.macro.analysis.supplemental import (
    LINEAR_CONFIG_FEATURES,
    build_benchmark_target_table,
    build_regression_baseline_predictions,
)


RAW_CLUSTER_GRID = [4, 6, 8, 10]
TARGETS_TO_KEEP = {"contribution_rate", "normalized_efficiency"}


def _parse_action_dict(value: object) -> dict[str, float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    text = str(value).strip()
    if not text or text == "{}":
        return {}
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, float] = {}
    for key, raw in parsed.items():
        try:
            out[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


def _action_units(value: object) -> float:
    return float(sum(v for v in _parse_action_dict(value).values() if v > 0))


def _action_targets(value: object) -> float:
    return float(sum(1 for v in _parse_action_dict(value).values() if v > 0))


def _parse_chat_blob(value: object) -> list[dict[str, Any]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if not text.startswith("["):
        text = "[" + text + "]"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _build_chat_feature_table(analysis_csv: Path) -> pd.DataFrame:
    analysis = pd.read_csv(analysis_csv, usecols=["gameId", "chat_log"]).copy()
    rows: list[dict[str, Any]] = []
    for record in analysis.itertuples(index=False):
        for msg in _parse_chat_blob(record.chat_log):
            player_id = msg.get("playerId")
            if not player_id:
                continue
            text = str(msg.get("text", "")).strip()
            phase = str(msg.get("gamePhase", "")).lower()
            rows.append(
                {
                    "gameId": str(record.gameId),
                    "playerId": str(player_id),
                    "message_count": 1,
                    "word_count": len(text.split()),
                    "contribution_phase_messages": int("contribution" in phase),
                    "outcome_phase_messages": int("outcome" in phase),
                    "summary_phase_messages": int("summary" in phase),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "gameId",
                "playerId",
                "message_count",
                "word_count",
                "contribution_phase_messages",
                "outcome_phase_messages",
                "summary_phase_messages",
            ]
        )
    return (
        pd.DataFrame(rows)
        .groupby(["gameId", "playerId"], as_index=False)
        .sum(numeric_only=True)
    )


def build_raw_behavior_feature_table(
    *,
    player_rounds_csv: Path,
    analysis_csv: Path,
    player_game_table_path: Path,
) -> pd.DataFrame:
    rounds = pd.read_csv(player_rounds_csv).copy()
    rounds["gameId"] = rounds["gameId"].astype(str)
    rounds["playerId"] = rounds["playerId"].astype(str)

    cfg = pd.read_csv(
        analysis_csv,
        usecols=[
            "gameId",
            "CONFIG_endowment",
            "CONFIG_numRounds",
            "CONFIG_chat",
        ],
    ).drop_duplicates(subset=["gameId"], keep="first")
    cfg["gameId"] = cfg["gameId"].astype(str)
    rounds = rounds.merge(cfg, on="gameId", how="left", validate="many_to_one")
    rounds["CONFIG_endowment"] = pd.to_numeric(rounds["CONFIG_endowment"], errors="coerce").fillna(20.0)
    rounds["CONFIG_numRounds"] = pd.to_numeric(rounds["CONFIG_numRounds"], errors="coerce").fillna(0)
    rounds["CONFIG_chat"] = (
        rounds["CONFIG_chat"]
        .astype(str)
        .str.lower()
        .isin({"1", "true", "t", "yes"})
        .astype(float)
    )

    rounds = rounds.sort_values(["gameId", "playerId", "createdAt", "roundId"]).reset_index(drop=True)
    rounds["round_index"] = rounds.groupby(["gameId", "playerId"]).cumcount() + 1

    rounds["punish_units_given"] = rounds["data.punished"].map(_action_units)
    rounds["reward_units_given"] = rounds["data.rewarded"].map(_action_units)
    rounds["punish_targets_given"] = rounds["data.punished"].map(_action_targets)
    rounds["reward_targets_given"] = rounds["data.rewarded"].map(_action_targets)
    rounds["punish_units_received"] = rounds["data.punishedBy"].map(_action_units)
    rounds["reward_units_received"] = rounds["data.rewardedBy"].map(_action_units)

    rounds["contribution"] = pd.to_numeric(rounds["data.contribution"], errors="coerce").fillna(0.0)
    rounds["round_payoff"] = pd.to_numeric(rounds["data.roundPayoff"], errors="coerce").fillna(0.0)
    rounds["contribution_rate"] = rounds["contribution"] / rounds["CONFIG_endowment"]
    rounds["round_payoff_rate"] = rounds["round_payoff"] / rounds["CONFIG_endowment"]
    rounds["zero_contribution"] = (rounds["contribution"] <= 0).astype(float)
    rounds["full_contribution"] = (rounds["contribution"] >= rounds["CONFIG_endowment"] - 1e-9).astype(float)
    rounds["punish_any"] = (rounds["punish_units_given"] > 0).astype(float)
    rounds["reward_any"] = (rounds["reward_units_given"] > 0).astype(float)
    rounds["punish_received_any"] = (rounds["punish_units_received"] > 0).astype(float)
    rounds["reward_received_any"] = (rounds["reward_units_received"] > 0).astype(float)

    def _aggregate(group: pd.DataFrame) -> pd.Series:
        contrib = group["contribution_rate"]
        rounds_n = len(group)
        final_known = group["CONFIG_numRounds"].iloc[0] > 0
        return pd.Series(
            {
                "mean_contribution_rate": float(contrib.mean()),
                "zero_contribution_rate": float(group["zero_contribution"].mean()),
                "full_contribution_rate": float(group["full_contribution"].mean()),
                "contribution_std_rate": float(contrib.std(ddof=0)),
                "endgame_delta_rate": float(contrib.iloc[-1] - contrib.iloc[0]),
                "mean_round_payoff_rate": float(group["round_payoff_rate"].mean()),
                "punish_row_rate": float(group["punish_any"].mean()),
                "reward_row_rate": float(group["reward_any"].mean()),
                "punish_units_per_round": float(group["punish_units_given"].mean()),
                "reward_units_per_round": float(group["reward_units_given"].mean()),
                "punish_targets_per_round": float(group["punish_targets_given"].mean()),
                "reward_targets_per_round": float(group["reward_targets_given"].mean()),
                "punish_received_row_rate": float(group["punish_received_any"].mean()),
                "reward_received_row_rate": float(group["reward_received_any"].mean()),
                "punish_received_units_per_round": float(group["punish_units_received"].mean()),
                "reward_received_units_per_round": float(group["reward_units_received"].mean()),
                "observed_rounds": float(rounds_n),
                "known_horizon": float(final_known),
            }
        )

    behavior = (
        rounds.groupby(["gameId", "playerId"], as_index=False)
        .apply(_aggregate, include_groups=False)
        .reset_index()
        .rename(columns={"gameId": "game_id", "playerId": "player_id"})
    )
    behavior = behavior.drop(columns=[c for c in ["level_2", "index"] if c in behavior.columns])

    chat = _build_chat_feature_table(analysis_csv)
    if not chat.empty:
        chat = chat.rename(columns={"gameId": "game_id", "playerId": "player_id"})
        behavior = behavior.merge(chat, on=["game_id", "player_id"], how="left")
    else:
        for column in [
            "message_count",
            "word_count",
            "contribution_phase_messages",
            "outcome_phase_messages",
            "summary_phase_messages",
        ]:
            behavior[column] = 0.0

    for column in [
        "message_count",
        "word_count",
        "contribution_phase_messages",
        "outcome_phase_messages",
        "summary_phase_messages",
    ]:
        behavior[column] = pd.to_numeric(behavior[column], errors="coerce").fillna(0.0)
        behavior[f"{column}_per_round"] = behavior[column] / behavior["observed_rounds"].clip(lower=1.0)

    behavior["chat_active"] = (behavior["message_count"] > 0).astype(float)

    player_game = pd.read_parquet(
        player_game_table_path,
        columns=["row_id", "wave", "game_id", "player_id", *LINEAR_CONFIG_FEATURES],
    )
    behavior = behavior.merge(
        player_game,
        on=["game_id", "player_id"],
        how="inner",
        validate="one_to_one",
    )
    return behavior


def _prepare_feature_matrix(
    learn_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    learn_x = pd.DataFrame(
        pipeline.fit_transform(learn_df[feature_cols]),
        index=learn_df.index,
        columns=feature_cols,
    )
    val_x = pd.DataFrame(
        pipeline.transform(val_df[feature_cols]),
        index=val_df.index,
        columns=feature_cols,
    )
    return learn_x, val_x


def _benchmark_cluster_feature_table(
    mixture_df: pd.DataFrame,
    analysis_csv: Path,
    cluster_cols: list[str],
) -> pd.DataFrame:
    analysis = pd.read_csv(analysis_csv, usecols=["gameId", "CONFIG_treatmentName"]).copy()
    analysis["gameId"] = analysis["gameId"].astype(str)
    analysis["benchmark_id"] = analysis["CONFIG_treatmentName"].astype(str)
    analysis = analysis[["gameId", "benchmark_id"]].drop_duplicates(subset=["gameId"], keep="first")
    tmp = mixture_df.copy()
    if "game_id" in tmp.columns:
        tmp = tmp.rename(columns={"game_id": "gameId"})
    tmp["gameId"] = tmp["gameId"].astype(str)
    merged = tmp.merge(analysis, on="gameId", how="inner", validate="one_to_one")
    return merged.groupby("benchmark_id", as_index=False)[cluster_cols].mean()


def _run_direct_regression_eval(
    *,
    learn_benchmark_table: pd.DataFrame,
    val_benchmark_table: pd.DataFrame,
    cluster_cols: list[str],
    k: int,
    env_val_mean_cluster_mae: float,
    env_val_avg_l1: float,
    oracle: bool,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for feature_set, feature_cols in {
        "cluster_only": cluster_cols,
        "cluster_plus_config": [*cluster_cols, *LINEAR_CONFIG_FEATURES],
    }.items():
        source_name = f"{'oracle' if oracle else 'pred'}_k{k}_{feature_set}"
        for model_kind in ["linear", "ridge"]:
            preds, summary = build_regression_baseline_predictions(
                learn_benchmark_table=learn_benchmark_table,
                val_benchmark_table=val_benchmark_table,
                feature_cols=feature_cols,
                model_kind=model_kind,
                prediction_prefix=source_name,
            )
            if preds.empty or summary.empty:
                continue
            keep = summary[summary["metric"].isin(TARGETS_TO_KEEP)].copy()
            keep["k"] = k
            keep["feature_set"] = feature_set if not oracle else f"oracle_{feature_set}"
            keep["env_val_mean_cluster_mae"] = np.nan if oracle else env_val_mean_cluster_mae
            keep["env_val_avg_l1"] = np.nan if oracle else env_val_avg_l1
            rows.append(keep)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[["k", "model_kind", "feature_set", "metric", "env_val_mean_cluster_mae", "env_val_avg_l1", "mae", "rmse", "corr"]]


def run_behavioral_cluster_ablation() -> None:
    root = ROOT
    out_dir = root / "simulation_statistical/raw_behavior_cluster/artifacts/outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    learn_features = build_raw_behavior_feature_table(
        player_rounds_csv=root / "benchmark_statistical/data/raw_data/learning_wave/player-rounds.csv",
        analysis_csv=root / "benchmark_statistical/data/processed_data/df_analysis_learn.csv",
        player_game_table_path=root / "simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/player_game_table_learn_clean.parquet",
    )
    val_features = build_raw_behavior_feature_table(
        player_rounds_csv=root / "benchmark_statistical/data/raw_data/validation_wave/player-rounds.csv",
        analysis_csv=root / "benchmark_statistical/data/processed_data/df_analysis_val.csv",
        player_game_table_path=root / "simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/player_game_table_val_clean.parquet",
    )
    learn_features.to_parquet(out_dir / "learn_behavior_features.parquet", index=False)
    val_features.to_parquet(out_dir / "val_behavior_features.parquet", index=False)

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

    learn_x, val_x = _prepare_feature_matrix(learn_features, val_features, feature_cols)
    learn_x.to_parquet(out_dir / "learn_behavior_features_scaled.parquet", index=False)
    val_x.to_parquet(out_dir / "val_behavior_features_scaled.parquet", index=False)

    learn_benchmark_table = build_benchmark_target_table(
        root / "benchmark_statistical/data/processed_data/df_analysis_learn.csv",
        root / "benchmark_statistical/data/raw_data/learning_wave/player-rounds.csv",
    )
    val_benchmark_table = build_benchmark_target_table(
        root / "benchmark_statistical/data/processed_data/df_analysis_val.csv",
        root / "benchmark_statistical/data/raw_data/validation_wave/player-rounds.csv",
    )

    diagnostics_rows: list[dict[str, Any]] = []
    direct_rows: list[pd.DataFrame] = []

    learn_player_game = pd.read_parquet(
        root / "simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/player_game_table_learn_clean.parquet"
    )
    val_player_game = pd.read_parquet(
        root / "simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/player_game_table_val_clean.parquet"
    )

    for k in RAW_CLUSTER_GRID:
        model = SoftClusterGMM(n_components=k, random_state=0)
        model.fit(learn_x.to_numpy())
        learn_probs = model.predict_proba(learn_x.to_numpy())
        val_probs = model.predict_proba(val_x.to_numpy())
        diag = model.diagnostics(learn_x.to_numpy())
        diagnostics_rows.append(diag)

        cluster_cols = cluster_probability_columns(k)
        learn_weights = learn_features[["row_id", "wave", "game_id", "player_id"]].copy()
        val_weights = val_features[["row_id", "wave", "game_id", "player_id"]].copy()
        for idx, col in enumerate(cluster_cols):
            learn_weights[col] = learn_probs[:, idx]
            val_weights[col] = val_probs[:, idx]

        learn_game = aggregate_player_weights_to_games(learn_weights, learn_player_game)
        val_game = aggregate_player_weights_to_games(val_weights, val_player_game)
        learn_game.to_parquet(out_dir / f"learn_game_cluster_distribution_k{k}.parquet", index=False)
        val_game.to_parquet(out_dir / f"val_game_cluster_distribution_k{k}.parquet", index=False)

        _, _, val_pred = fit_env_distribution_model(learn_game, val_game)
        val_pred.to_parquet(out_dir / f"predicted_val_game_cluster_distribution_k{k}.parquet", index=False)

        env_val_mean_cluster_mae = float(np.abs(val_pred[cluster_cols].to_numpy() - val_game[cluster_cols].to_numpy()).mean())
        env_val_avg_l1 = float(np.abs(val_pred[cluster_cols].to_numpy() - val_game[cluster_cols].to_numpy()).sum(axis=1).mean())

        learn_benchmark_actual = _benchmark_cluster_feature_table(
            learn_game.rename(columns={"game_id": "gameId"}),
            root / "benchmark_statistical/data/processed_data/df_analysis_learn.csv",
            cluster_cols,
        )
        val_benchmark_actual = _benchmark_cluster_feature_table(
            val_game.rename(columns={"game_id": "gameId"}),
            root / "benchmark_statistical/data/processed_data/df_analysis_val.csv",
            cluster_cols,
        )
        val_benchmark_pred = _benchmark_cluster_feature_table(
            val_pred.rename(columns={"game_id": "gameId"}),
            root / "benchmark_statistical/data/processed_data/df_analysis_val.csv",
            cluster_cols,
        )

        learn_eval_table = learn_benchmark_table.merge(learn_benchmark_actual, on="benchmark_id", how="inner", validate="one_to_one")
        val_eval_pred_table = val_benchmark_table.merge(val_benchmark_pred, on="benchmark_id", how="inner", validate="one_to_one")
        val_eval_oracle_table = val_benchmark_table.merge(val_benchmark_actual, on="benchmark_id", how="inner", validate="one_to_one")

        pred_summary = _run_direct_regression_eval(
            learn_benchmark_table=learn_eval_table,
            val_benchmark_table=val_eval_pred_table,
            cluster_cols=cluster_cols,
            k=k,
            env_val_mean_cluster_mae=env_val_mean_cluster_mae,
            env_val_avg_l1=env_val_avg_l1,
            oracle=False,
        )
        oracle_summary = _run_direct_regression_eval(
            learn_benchmark_table=learn_eval_table,
            val_benchmark_table=val_eval_oracle_table,
            cluster_cols=cluster_cols,
            k=k,
            env_val_mean_cluster_mae=env_val_mean_cluster_mae,
            env_val_avg_l1=env_val_avg_l1,
            oracle=True,
        )
        direct_rows.extend([pred_summary, oracle_summary])

    diagnostics_df = pd.DataFrame(diagnostics_rows).sort_values("n_clusters").reset_index(drop=True)
    diagnostics_df.to_csv(out_dir / "raw_behavior_cluster_diagnostics.csv", index=False)

    direct_df = pd.concat([df for df in direct_rows if not df.empty], ignore_index=True)
    direct_df = direct_df.rename(columns={"metric": "target"})
    direct_df.to_csv(out_dir / "raw_behavior_cluster_direct_eval.csv", index=False)

    text_pred = pd.read_csv(
        root / "simulation_statistical/archetype_distribution_embedding/artifacts/outputs/cluster_count_direct_prediction_sweep.csv"
    )
    text_oracle = pd.read_csv(
        root / "simulation_statistical/archetype_distribution_embedding/artifacts/outputs/cluster_count_oracle_direct_prediction_sweep.csv"
    )
    text_k6 = pd.concat(
        [
            text_pred.assign(feature_set=lambda df: "text_" + df["feature_set"].astype(str)),
            text_oracle.assign(feature_set=lambda df: "text_" + df["feature_set"].astype(str)),
        ],
        ignore_index=True,
    )
    text_k6 = text_k6[text_k6["k"] == 6].copy()

    report_lines = [
        "# Raw behavior cluster ablation",
        "",
        "This ablation clusters players directly from observed raw behavior rather than LLM summary embeddings.",
        "",
        "## Behavior feature vector",
        "",
        "- contribution level, zero/full rates, volatility, and endgame delta",
        "- punishment/reward giving and receiving rates and units",
        "- round payoff rate",
        "- chat activity, words per round, and phase-specific message rates when game chat exists",
        "",
        "## Downstream evaluation",
        "",
        "- same direct benchmark-level prediction target table as the text-cluster analysis",
        "- same linear and ridge regressions",
        "- same deployable (`CONFIG -> predicted cluster mixture`) and oracle (`actual validation mixture`) settings",
        "",
        "## Best raw-behavior results",
        "",
    ]

    best_rows = []
    for feature_set in ["cluster_only", "cluster_plus_config", "oracle_cluster_only", "oracle_cluster_plus_config"]:
        subset = direct_df[(direct_df["target"] == "normalized_efficiency") & (direct_df["feature_set"] == feature_set) & (direct_df["model_kind"] == "ridge")].copy()
        if subset.empty:
            continue
        best = subset.sort_values(["rmse", "mae"]).iloc[0]
        best_rows.append(best)
        report_lines.append(
            f"- `{feature_set}` best at `K={int(best['k'])}`: MAE `{best['mae']:.4f}`, RMSE `{best['rmse']:.4f}`, corr `{best['corr']:.4f}`"
        )
    report_lines.append("")
    report_lines.append("## Best raw-behavior results on mean contribution rate")
    report_lines.append("")
    for feature_set in ["cluster_only", "cluster_plus_config", "oracle_cluster_only", "oracle_cluster_plus_config"]:
        subset = direct_df[(direct_df["target"] == "contribution_rate") & (direct_df["feature_set"] == feature_set) & (direct_df["model_kind"] == "ridge")].copy()
        if subset.empty:
            continue
        best = subset.sort_values(["rmse", "mae"]).iloc[0]
        report_lines.append(
            f"- `{feature_set}` best at `K={int(best['k'])}`: MAE `{best['mae']:.4f}`, RMSE `{best['rmse']:.4f}`, corr `{best['corr']:.4f}`"
        )
    report_lines.append("")
    report_lines.append("## Text-cluster reference at K=6")
    report_lines.append("")
    for target in ["mean_contribution_rate", "normalized_efficiency"]:
        report_lines.append(f"### `{target}`")
        report_lines.append("")
        for feature_set in ["cluster_only", "cluster_plus_config", "cluster_oracle_only", "cluster_oracle_plus_config"]:
            subset = text_k6[(text_k6["target"] == target) & (text_k6["feature_set"] == f"text_{feature_set}") & (text_k6["model_kind"] == "ridge")].copy()
            if subset.empty:
                continue
            row = subset.iloc[0]
            report_lines.append(
                f"- `{feature_set}` at text `K=6`: MAE `{row['mae']:.4f}`, RMSE `{row['rmse']:.4f}`, corr `{row['corr']:.4f}`"
            )
        report_lines.append("")
    report_lines.append("")
    report_lines.append("## Files")
    report_lines.append("")
    report_lines.append("- `learn_behavior_features.parquet` / `val_behavior_features.parquet`")
    report_lines.append("- `raw_behavior_cluster_diagnostics.csv`")
    report_lines.append("- `raw_behavior_cluster_direct_eval.csv`")

    (out_dir / "raw_behavior_cluster_report.md").write_text("\n".join(report_lines).strip() + "\n")


if __name__ == "__main__":
    run_behavioral_cluster_ablation()
