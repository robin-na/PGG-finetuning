from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _parse_model_spec(value: str) -> tuple[str, Path, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "Model spec must have the form NAME:/abs/path/to/actor.csv:/abs/path/to/game_summary.csv"
        )
    name, actor_csv, game_csv = parts
    return name, Path(actor_csv), Path(game_csv)


def _load_common_games(path: Path) -> dict[int, set[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): set(v) for k, v in manifest["common_games_by_k"].items()}


def _filter_to_common_subset(df: pd.DataFrame, common_games_by_k: dict[int, set[str]]) -> pd.DataFrame:
    parts = []
    for k, game_ids in common_games_by_k.items():
        parts.append(df[(df["k"] == k) & (df["game_id"].astype(str).isin(game_ids))])
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def _summarize_similarity_to_persistence(
    model_name: str,
    actor_df: pd.DataFrame,
    persistence_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    join_keys = ["k", "game_id", "round_index", "player_id"]
    persistence_small = persistence_df[
        join_keys
        + [
            "predicted_contribution",
            "predicted_has_punish",
            "predicted_has_reward",
            "predicted_punished",
            "predicted_rewarded",
        ]
    ].rename(
        columns={
            "predicted_contribution": "persist_contribution",
            "predicted_has_punish": "persist_has_punish",
            "predicted_has_reward": "persist_has_reward",
            "predicted_punished": "persist_punished",
            "predicted_rewarded": "persist_rewarded",
        }
    )
    merged = actor_df.merge(persistence_small, on=join_keys, how="inner")
    merged["same_contribution_as_persistence"] = (
        merged["predicted_contribution"] == merged["persist_contribution"]
    ).astype(int)
    merged["contribution_abs_diff_to_persistence"] = (
        merged["predicted_contribution"] - merged["persist_contribution"]
    ).abs()
    merged["same_punish_presence_as_persistence"] = (
        merged["predicted_has_punish"] == merged["persist_has_punish"]
    ).astype(int)
    merged["same_reward_presence_as_persistence"] = (
        merged["predicted_has_reward"] == merged["persist_has_reward"]
    ).astype(int)
    merged["same_punished_dict_as_persistence"] = (
        merged["predicted_punished"] == merged["persist_punished"]
    ).astype(int)
    merged["same_rewarded_dict_as_persistence"] = (
        merged["predicted_rewarded"] == merged["persist_rewarded"]
    ).astype(int)

    actor_summary = (
        merged.groupby("k", as_index=False)[
            [
                "same_contribution_as_persistence",
                "contribution_abs_diff_to_persistence",
                "same_punish_presence_as_persistence",
                "same_reward_presence_as_persistence",
                "same_punished_dict_as_persistence",
                "same_rewarded_dict_as_persistence",
            ]
        ]
        .mean()
        .rename(
            columns={
                "same_contribution_as_persistence": "actor_round_contribution_match_rate_to_persistence",
                "contribution_abs_diff_to_persistence": "actor_round_mean_contribution_abs_diff_to_persistence",
                "same_punish_presence_as_persistence": "actor_round_punish_presence_match_rate_to_persistence",
                "same_reward_presence_as_persistence": "actor_round_reward_presence_match_rate_to_persistence",
                "same_punished_dict_as_persistence": "actor_round_punished_dict_match_rate_to_persistence",
                "same_rewarded_dict_as_persistence": "actor_round_rewarded_dict_match_rate_to_persistence",
            }
        )
    )
    actor_counts = (
        merged.groupby("k", as_index=False)
        .agg(num_actor_rows=("player_id", "count"), num_games=("game_id", "nunique"))
    )
    actor_summary = actor_counts.merge(actor_summary, on="k", how="left")
    actor_summary.insert(0, "model", model_name)

    trajectory = (
        merged.groupby(["k", "game_id", "player_id"], as_index=False)[
            [
                "same_contribution_as_persistence",
                "same_punished_dict_as_persistence",
                "same_rewarded_dict_as_persistence",
            ]
        ]
        .mean()
        .rename(
            columns={
                "same_contribution_as_persistence": "trajectory_contribution_match_rate_to_persistence",
                "same_punished_dict_as_persistence": "trajectory_punished_dict_match_rate_to_persistence",
                "same_rewarded_dict_as_persistence": "trajectory_rewarded_dict_match_rate_to_persistence",
            }
        )
    )
    trajectory["full_contribution_trajectory_matches_persistence"] = (
        trajectory["trajectory_contribution_match_rate_to_persistence"] == 1.0
    ).astype(int)
    trajectory_summary = (
        trajectory.groupby("k", as_index=False)[
            [
                "full_contribution_trajectory_matches_persistence",
                "trajectory_punished_dict_match_rate_to_persistence",
                "trajectory_rewarded_dict_match_rate_to_persistence",
            ]
        ]
        .mean()
    )
    trajectory_counts = (
        trajectory.groupby("k", as_index=False)
        .agg(num_player_trajectories=("player_id", "count"), num_games=("game_id", "nunique"))
    )
    trajectory_summary = trajectory_counts.merge(trajectory_summary, on="k", how="left")
    trajectory_summary.insert(0, "model", model_name)
    return actor_summary, trajectory_summary


def _summarize_by_chat(model_name: str, game_summary_df: pd.DataFrame) -> pd.DataFrame:
    base = (
        game_summary_df.groupby(["k", "chat_enabled"], as_index=False)
        .agg(
            num_games=("game_id", "nunique"),
            contribution_rate_mae=("contribution_rate_mae", "mean"),
            future_normalized_efficiency_abs_error=("future_normalized_efficiency_abs_error", "mean"),
        )
    )
    punish = (
        game_summary_df[game_summary_df["punishment_enabled"]]
        .groupby(["k", "chat_enabled"], as_index=False)
        .agg(
            punishment_enabled_games=("game_id", "nunique"),
            punish_target_f1_punishment_enabled_only=("punish_target_f1", "mean"),
        )
    )
    reward = (
        game_summary_df[game_summary_df["reward_enabled"]]
        .groupby(["k", "chat_enabled"], as_index=False)
        .agg(
            reward_enabled_games=("game_id", "nunique"),
            reward_target_f1_reward_enabled_only=("reward_target_f1", "mean"),
        )
    )
    result = base.merge(punish, on=["k", "chat_enabled"], how="left").merge(
        reward,
        on=["k", "chat_enabled"],
        how="left",
    )
    result.insert(0, "model", model_name)
    return result


def _summarize_by_flag(model_name: str, game_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for flag in ["chat_enabled", "all_or_nothing", "punishment_enabled", "reward_enabled"]:
        grouped = (
            game_summary_df.groupby(flag, as_index=False)
            .agg(
                num_games=("game_id", "count"),
                contribution_rate_mae=("contribution_rate_mae", "mean"),
                future_normalized_efficiency_abs_error=("future_normalized_efficiency_abs_error", "mean"),
            )
            .rename(columns={flag: "flag_value"})
        )
        grouped.insert(0, "flag_name", flag)
        grouped.insert(0, "model", model_name)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze persistence-likeness and config heterogeneity of LLM trajectory-completion runs."
    )
    parser.add_argument("--common-manifest-json", type=Path, required=True)
    parser.add_argument("--request-manifest-csv", type=Path, required=True)
    parser.add_argument("--persistence-actor-csv", type=Path, required=True)
    parser.add_argument("--model-spec", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.model_spec:
        raise ValueError("At least one --model-spec is required.")

    common_games_by_k = _load_common_games(args.common_manifest_json)
    metadata_df = pd.read_csv(args.request_manifest_csv)[
        [
            "game_id",
            "k",
            "chat_enabled",
            "punishment_enabled",
            "reward_enabled",
            "all_or_nothing",
            "treatment_name",
        ]
    ].drop_duplicates()

    persistence_df = pd.read_csv(args.persistence_actor_csv)
    persistence_df = persistence_df[persistence_df["baseline"] == "persistence"].copy()
    persistence_df = _filter_to_common_subset(persistence_df, common_games_by_k)

    actor_similarity_rows = []
    trajectory_similarity_rows = []
    heterogeneity_by_chat_rows = []
    heterogeneity_by_flag_rows = []

    for raw_model_spec in args.model_spec:
        model_name, actor_csv, game_csv = _parse_model_spec(raw_model_spec)
        actor_df = _filter_to_common_subset(pd.read_csv(actor_csv), common_games_by_k)
        game_summary_df = _filter_to_common_subset(pd.read_csv(game_csv), common_games_by_k)
        game_summary_df = game_summary_df.merge(metadata_df, on=["game_id", "k"], how="left")

        actor_similarity, trajectory_similarity = _summarize_similarity_to_persistence(
            model_name=model_name,
            actor_df=actor_df,
            persistence_df=persistence_df,
        )
        actor_similarity_rows.append(actor_similarity)
        trajectory_similarity_rows.append(trajectory_similarity)
        heterogeneity_by_chat_rows.append(_summarize_by_chat(model_name, game_summary_df))
        heterogeneity_by_flag_rows.append(_summarize_by_flag(model_name, game_summary_df))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(actor_similarity_rows, ignore_index=True).to_csv(
        args.output_dir / "actor_similarity_to_persistence.csv",
        index=False,
    )
    pd.concat(trajectory_similarity_rows, ignore_index=True).to_csv(
        args.output_dir / "trajectory_similarity_to_persistence.csv",
        index=False,
    )
    pd.concat(heterogeneity_by_chat_rows, ignore_index=True).to_csv(
        args.output_dir / "heterogeneity_by_chat.csv",
        index=False,
    )
    pd.concat(heterogeneity_by_flag_rows, ignore_index=True).to_csv(
        args.output_dir / "heterogeneity_by_flag.csv",
        index=False,
    )

    manifest = {
        "common_manifest_json": str(args.common_manifest_json),
        "request_manifest_csv": str(args.request_manifest_csv),
        "persistence_actor_csv": str(args.persistence_actor_csv),
        "model_specs": args.model_spec,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
