from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


CLUSTER_LABELS = {
    "cluster_1_prob": "strong_mildly_conditional_cooperator",
    "cluster_2_prob": "opportunistic_free_rider",
    "cluster_3_prob": "fragile_high_cooperator",
    "cluster_4_prob": "near_unconditional_full_cooperator",
    "cluster_5_prob": "unknown_or_sparse_info",
    "cluster_6_prob": "moderate_payoff_aware_conditional_cooperator",
}


TAG_ORDER = [
    "CONTRIBUTION",
    "COMMUNICATION",
    "PUNISHMENT",
    "REWARD",
    "RESPONSE_TO_END_GAME",
    "RESPONSE_TO_OTHERS_OUTCOME",
    "RESPONSE_TO_PUNISHER",
    "RESPONSE_TO_REWARDER",
]


BEHAVIOR_METRICS = [
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
]


def _parse_action_dict(value: object) -> dict[str, float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
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


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if not mask.any():
        return float("nan")
    w = weights[mask].astype(float)
    v = values[mask].astype(float)
    denom = w.sum()
    if denom <= 0:
        return float("nan")
    return float(np.dot(v, w) / denom)


def _first_sentence(text: str, limit: int = 220) -> str:
    cleaned = " ".join(str(text).split())
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    first = parts[0] if parts else cleaned
    if len(first) <= limit:
        return first
    return first[: limit - 3].rstrip() + "..."


def build_cluster_posthoc_profile(
    weights_path: str | Path,
    player_rounds_path: str | Path,
    game_table_path: str | Path,
    analysis_path: str | Path,
    tag_blocks_path: str | Path,
    behavior_output_path: str | Path,
    tag_output_path: str | Path,
    report_output_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights_df = pd.read_parquet(weights_path)
    cluster_cols = [c for c in weights_df.columns if c.startswith("cluster_") and c.endswith("_prob")]

    rounds_df = pd.read_csv(player_rounds_path)
    game_table_df = pd.read_parquet(game_table_path, columns=["row_id", "game_id", "player_id", "text"])
    tag_df = pd.read_parquet(tag_blocks_path)

    analysis_df = pd.read_csv(analysis_path, usecols=["gameId", "CONFIG_endowment"]).drop_duplicates()
    rounds_df = rounds_df.merge(analysis_df, on="gameId", how="left", validate="many_to_one")
    rounds_df["CONFIG_endowment"] = rounds_df["CONFIG_endowment"].fillna(20.0)
    rounds_df = rounds_df.sort_values(["gameId", "playerId", "createdAt", "roundId"]).reset_index(drop=True)
    rounds_df["round_index"] = rounds_df.groupby(["gameId", "playerId"]).cumcount() + 1

    rounds_df["punish_units_given"] = rounds_df["data.punished"].map(_action_units)
    rounds_df["reward_units_given"] = rounds_df["data.rewarded"].map(_action_units)
    rounds_df["punish_targets_given"] = rounds_df["data.punished"].map(_action_targets)
    rounds_df["reward_targets_given"] = rounds_df["data.rewarded"].map(_action_targets)
    rounds_df["punish_units_received"] = rounds_df["data.punishedBy"].map(_action_units)
    rounds_df["reward_units_received"] = rounds_df["data.rewardedBy"].map(_action_units)

    rounds_df["contribution_rate"] = rounds_df["data.contribution"] / rounds_df["CONFIG_endowment"]
    rounds_df["round_payoff_rate"] = rounds_df["data.roundPayoff"] / rounds_df["CONFIG_endowment"]
    rounds_df["zero_contribution"] = (rounds_df["data.contribution"] <= 0).astype(float)
    rounds_df["full_contribution"] = (
        rounds_df["data.contribution"] >= rounds_df["CONFIG_endowment"] - 1e-9
    ).astype(float)
    rounds_df["punish_any"] = (rounds_df["punish_units_given"] > 0).astype(float)
    rounds_df["reward_any"] = (rounds_df["reward_units_given"] > 0).astype(float)
    rounds_df["punish_received_any"] = (rounds_df["punish_units_received"] > 0).astype(float)
    rounds_df["reward_received_any"] = (rounds_df["reward_units_received"] > 0).astype(float)

    def _player_game_metrics(group: pd.DataFrame) -> pd.Series:
        contrib = group["contribution_rate"]
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
            }
        )

    player_game_metrics_df = (
        rounds_df.groupby(["gameId", "playerId"], as_index=False)
        .apply(_player_game_metrics, include_groups=False)
        .reset_index()
        .rename(columns={"gameId": "game_id", "playerId": "player_id"})
    )
    if "level_2" in player_game_metrics_df.columns:
        player_game_metrics_df = player_game_metrics_df.drop(columns=["level_2"])

    behavior_df = weights_df.merge(
        player_game_metrics_df,
        on=["game_id", "player_id"],
        how="left",
        validate="one_to_one",
    )

    behavior_rows: list[dict[str, object]] = []
    top_assign = behavior_df[cluster_cols].idxmax(axis=1)
    for cluster in cluster_cols:
        row: dict[str, object] = {
            "cluster": cluster,
            "cluster_label": CLUSTER_LABELS.get(cluster, cluster),
            "mean_mass": float(behavior_df[cluster].mean()),
            "top_assignment_share": float((top_assign == cluster).mean()),
        }
        weights = behavior_df[cluster]
        for metric in BEHAVIOR_METRICS:
            row[metric] = _weighted_mean(behavior_df[metric], weights)
        behavior_rows.append(row)
    behavior_profile_df = pd.DataFrame(behavior_rows)
    behavior_profile_df.to_csv(behavior_output_path, index=False)

    base_rows = weights_df[["row_id", *cluster_cols]].copy()
    tag_presence_df = (
        tag_df.assign(tag_present=1)
        .pivot_table(index="row_id", columns="tag", values="tag_present", aggfunc="max", fill_value=0)
        .reset_index()
    )
    tag_presence_df.columns.name = None
    base_with_tags_df = base_rows.merge(tag_presence_df, on="row_id", how="left").fillna(0)
    base_with_tags_df = base_with_tags_df.merge(game_table_df[["row_id", "text"]], on="row_id", how="left")

    tag_rows: list[dict[str, object]] = []
    snippet_rows: list[dict[str, object]] = []
    for tag in TAG_ORDER:
        present_col = tag if tag in base_with_tags_df.columns else None
        for cluster in cluster_cols:
            weights = base_with_tags_df[cluster]
            presence = base_with_tags_df[present_col] if present_col else pd.Series(0, index=base_with_tags_df.index)
            tag_rows.append(
                {
                    "cluster": cluster,
                    "cluster_label": CLUSTER_LABELS.get(cluster, cluster),
                    "tag": tag,
                    "presence_rate": _weighted_mean(presence, weights),
                    "avg_tag_length_chars": float("nan"),
                    "top_terms": "",
                }
            )

            subset = tag_df[tag_df["tag"] == tag].merge(
                weights_df[["row_id", cluster]],
                on="row_id",
                how="inner",
                validate="many_to_one",
            )
            if subset.empty:
                continue
            tag_rows[-1]["avg_tag_length_chars"] = _weighted_mean(
                subset["tag_text"].fillna("").map(len),
                subset[cluster],
            )
            exemplar = subset.sort_values(cluster, ascending=False).iloc[0]
            snippet_rows.append(
                {
                    "cluster": cluster,
                    "tag": tag,
                    "snippet": _first_sentence(exemplar["tag_text"]),
                }
            )

    tag_profile_df = pd.DataFrame(tag_rows)

    keyword_rows: list[dict[str, object]] = []
    for tag in ["COMMUNICATION", "PUNISHMENT", "REWARD"]:
        tag_subset = tag_df[tag_df["tag"] == tag].copy()
        if tag_subset.empty:
            continue
        cleaned_text = tag_subset["tag_text"].fillna("").astype(str)
        informative_mask = ~cleaned_text.str.lower().str.startswith("unknown")
        tag_subset = tag_subset[informative_mask].copy()
        cleaned_text = cleaned_text[informative_mask]
        if tag_subset.empty:
            continue
        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=5,
            max_features=3000,
        )
        X = vectorizer.fit_transform(cleaned_text)
        vocab = np.array(vectorizer.get_feature_names_out())
        merged = tag_subset.merge(weights_df[["row_id", *cluster_cols]], on="row_id", how="inner", validate="many_to_one")
        X_dense = X.astype(float)
        for cluster in cluster_cols:
            weights = merged[cluster].to_numpy()
            denom = weights.sum()
            if denom <= 0:
                continue
            scores = np.asarray(X_dense.T.dot(weights) / denom).ravel()
            top_idx = np.argsort(scores)[::-1][:8]
            terms = [vocab[idx] for idx in top_idx if scores[idx] > 0]
            keyword_rows.append(
                {
                    "cluster": cluster,
                    "cluster_label": CLUSTER_LABELS.get(cluster, cluster),
                    "tag": tag,
                    "top_terms": ", ".join(terms),
                }
            )

    keyword_df = pd.DataFrame(keyword_rows)
    if not keyword_df.empty:
        tag_profile_df = tag_profile_df.merge(
            keyword_df,
            on=["cluster", "cluster_label", "tag"],
            how="left",
            suffixes=("", "_kw"),
        )
        tag_profile_df["top_terms"] = tag_profile_df["top_terms_kw"].fillna(tag_profile_df["top_terms"])
        tag_profile_df = tag_profile_df.drop(columns=["top_terms_kw"])
    tag_profile_df.to_csv(tag_output_path, index=False)

    snippet_df = pd.DataFrame(snippet_rows)

    lines = ["# Post-hoc cluster profile", ""]
    lines.append(
        "This report characterizes the current six clusters using learning-wave actual player-round behavior plus the tagged LLM archetype summaries. Behavioral averages are soft-cluster weighted, not hard-cluster averages."
    )
    lines.append("")
    for _, cluster_row in behavior_profile_df.sort_values("cluster").iterrows():
        cluster = cluster_row["cluster"]
        label = cluster_row["cluster_label"]
        lines.append(f"## {cluster.replace('_prob', '')} / `{label}`")
        lines.append("")
        lines.append(
            f"- mass: mean `{cluster_row['mean_mass']:.3f}`, top-assignment share `{cluster_row['top_assignment_share']:.3f}`"
        )
        lines.append(
            f"- contribution: mean rate `{cluster_row['mean_contribution_rate']:.3f}`, zero rate `{cluster_row['zero_contribution_rate']:.3f}`, full rate `{cluster_row['full_contribution_rate']:.3f}`, volatility `{cluster_row['contribution_std_rate']:.3f}`, endgame delta `{cluster_row['endgame_delta_rate']:+.3f}`"
        )
        lines.append(
            f"- actions given: punish row rate `{cluster_row['punish_row_rate']:.3f}`, reward row rate `{cluster_row['reward_row_rate']:.3f}`, punish units/round `{cluster_row['punish_units_per_round']:.3f}`, reward units/round `{cluster_row['reward_units_per_round']:.3f}`"
        )
        lines.append(
            f"- actions received: punished row rate `{cluster_row['punish_received_row_rate']:.3f}`, rewarded row rate `{cluster_row['reward_received_row_rate']:.3f}`, payoff/endowment `{cluster_row['mean_round_payoff_rate']:.3f}`"
        )
        lines.append("")
        lines.append("Tag profile:")
        cluster_tags = tag_profile_df[tag_profile_df["cluster"] == cluster].set_index("tag")
        for tag in ["COMMUNICATION", "PUNISHMENT", "REWARD", "RESPONSE_TO_END_GAME"]:
            if tag not in cluster_tags.index:
                continue
            row = cluster_tags.loc[tag]
            lines.append(
                f"- {tag.lower()}: presence `{row['presence_rate']:.3f}`, avg chars `{row['avg_tag_length_chars']:.1f}`"
                + (f", keywords `{row['top_terms']}`" if isinstance(row["top_terms"], str) and row["top_terms"] else "")
            )
            snippet_match = snippet_df[(snippet_df["cluster"] == cluster) & (snippet_df["tag"] == tag)]
            if not snippet_match.empty:
                lines.append(f"  snippet: {snippet_match.iloc[0]['snippet']}")
        lines.append("")

    Path(report_output_path).write_text("\n".join(lines).strip() + "\n")
    return behavior_profile_df, tag_profile_df


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    build_cluster_posthoc_profile(
        weights_path=root / "simulation_statistical/archetype_distribution_embedding/artifacts/outputs/player_cluster_weights_learn.parquet",
        player_rounds_path=root / "benchmark_statistical/data/raw_data/learning_wave/player-rounds.csv",
        game_table_path=root / "simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/player_game_table_learn_clean.parquet",
        analysis_path=root / "benchmark_statistical/data/processed_data/df_analysis_learn.csv",
        tag_blocks_path=root / "simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/tag_blocks_learn.parquet",
        behavior_output_path=root / "simulation_statistical/archetype_distribution_embedding/artifacts/outputs/cluster_posthoc_behavior_profile.csv",
        tag_output_path=root / "simulation_statistical/archetype_distribution_embedding/artifacts/outputs/cluster_posthoc_tag_profile.csv",
        report_output_path=root / "simulation_statistical/archetype_distribution_embedding/artifacts/outputs/cluster_posthoc_profile.md",
    )
