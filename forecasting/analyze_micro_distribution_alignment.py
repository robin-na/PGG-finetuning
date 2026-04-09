from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .analyze_vs_human_treatments import _wasserstein_distance_1d
from .random_action_baseline import (
    build_uniform_random_rollout_tables,
    select_shared_game_skeletons,
    summarize_random_baseline_draws,
)


RUN_NAME_TO_LABEL = {
    "baseline_gpt_5_1": "gpt-5.1 baseline",
    "twin_sampled_seed_0_gpt_5_1": "gpt-5.1 twin",
    "twin_sampled_unadjusted_seed_0_gpt_5_1": "gpt-5.1 twin unadj",
    "demographic_only_row_resampled_seed_0_gpt_5_1": "gpt-5.1 demo-only",
    "baseline_gpt_5_mini": "gpt-5-mini baseline",
    "twin_sampled_seed_0_gpt_5_mini": "gpt-5-mini twin",
    "twin_sampled_unadjusted_seed_0_gpt_5_mini": "gpt-5-mini twin unadj",
    "demographic_only_row_resampled_seed_0_gpt_5_mini": "gpt-5-mini demo-only",
}

MODEL_STYLE = {
    "gpt-5.1 baseline": {"color": "#1f77b4"},
    "gpt-5.1 twin": {"color": "#6baed6"},
    "gpt-5.1 twin unadj": {"color": "#9ecae1"},
    "gpt-5.1 demo-only": {"color": "#c6dbef"},
    "gpt-5-mini baseline": {"color": "#ff7f0e"},
    "gpt-5-mini twin": {"color": "#fdae6b"},
    "gpt-5-mini twin unadj": {"color": "#fdd0a2"},
    "gpt-5-mini demo-only": {"color": "#fee6ce"},
    "noise_ceiling": {"color": "#2ca02c"},
}

MODEL_PLOT_ORDER = [
    "gpt-5.1 baseline",
    "gpt-5.1 demo-only",
    "gpt-5.1 twin unadj",
    "gpt-5.1 twin",
    "gpt-5-mini baseline",
    "gpt-5-mini demo-only",
    "gpt-5-mini twin unadj",
    "gpt-5-mini twin",
    "noise_ceiling",
]

PLAYER_METRIC_SPECS = [
    ("player_mean_contribution_rate", "Player mean contrib"),
    ("player_mean_normalized_payoff", "Player mean payoff"),
]

ROUND_METRIC_SPECS = [
    ("round_total_contribution_rate", "Round contrib"),
    ("round_normalized_efficiency", "Round eff"),
]


def _load_run_tables(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    generated_game_df = pd.read_csv(result_dir / "generated_game_summary.csv")
    generated_actor_df = pd.read_csv(result_dir / "generated_actor_summary.csv")
    generated_round_df = pd.read_csv(result_dir / "generated_round_summary.csv")
    human_actor_df = pd.read_csv(result_dir / "human_actor_summary.csv")
    human_round_df = pd.read_csv(result_dir / "human_round_summary.csv")
    return generated_game_df, generated_actor_df, generated_round_df, human_actor_df, human_round_df


def _subset_generated_entity_tables(
    generated_game_df: pd.DataFrame,
    generated_actor_df: pd.DataFrame,
    generated_round_df: pd.DataFrame,
    shared_counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count_map = {
        str(row["treatment_name"]): int(row["shared_generated_count"])
        for row in shared_counts.to_dict(orient="records")
    }
    kept_game_frames: list[pd.DataFrame] = []
    kept_ids: list[str] = []
    for treatment_name, group in generated_game_df.groupby("treatment_name", sort=True):
        shared_count = count_map.get(str(treatment_name), 0)
        if shared_count <= 0:
            continue
        kept = group.sort_values("custom_id").head(shared_count).copy()
        kept_game_frames.append(kept)
        kept_ids.extend(kept["custom_id"].astype(str).tolist())
    kept_game_df = pd.concat(kept_game_frames, ignore_index=True)
    kept_id_set = set(kept_ids)
    kept_actor_df = generated_actor_df[generated_actor_df["custom_id"].astype(str).isin(kept_id_set)].copy()
    kept_round_df = generated_round_df[generated_round_df["custom_id"].astype(str).isin(kept_id_set)].copy()
    return kept_game_df, kept_actor_df, kept_round_df


def _build_player_game_summary(
    actor_df: pd.DataFrame,
    *,
    entity_col: str,
) -> pd.DataFrame:
    return (
        actor_df.groupby(["treatment_name", entity_col, "player_id"], as_index=False)
        .agg(
            player_mean_contribution_rate=("contribution_rate", "mean"),
            player_mean_normalized_payoff=("round_normalized_payoff", "mean"),
        )
        .sort_values(["treatment_name", entity_col, "player_id"])
        .reset_index(drop=True)
    )


def _build_round_delta_summary(
    round_df: pd.DataFrame,
    *,
    entity_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (treatment_name, entity_id), group in round_df.groupby(["treatment_name", entity_col], sort=False):
        ordered = group.sort_values("round_number")
        previous_contrib = ordered["total_contribution_rate"].shift(1)
        previous_eff = ordered["round_normalized_efficiency"].shift(1)
        current = ordered.copy()
        current["delta_total_contribution_rate"] = current["total_contribution_rate"] - previous_contrib
        current["delta_round_normalized_efficiency"] = current["round_normalized_efficiency"] - previous_eff
        current = current[current["round_number"] > current["round_number"].min()].copy()
        if current.empty:
            continue
        rows.extend(
            current[
                [
                    "treatment_name",
                    entity_col,
                    "round_number",
                    "delta_total_contribution_rate",
                    "delta_round_normalized_efficiency",
                ]
            ].rename(columns={entity_col: "entity_id"}).to_dict(orient="records")
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "treatment_name",
                "entity_id",
                "round_number",
                "delta_total_contribution_rate",
                "delta_round_normalized_efficiency",
            ]
        )
    return pd.DataFrame(rows)


def _per_treatment_player_wd(
    generated_player_df: pd.DataFrame,
    human_player_df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    treatment_names = sorted(set(generated_player_df["treatment_name"]) & set(human_player_df["treatment_name"]))
    rows: list[dict[str, Any]] = []
    for treatment_name in treatment_names:
        generated_values = generated_player_df.loc[
            generated_player_df["treatment_name"] == treatment_name,
            metric,
        ]
        human_values = human_player_df.loc[
            human_player_df["treatment_name"] == treatment_name,
            metric,
        ]
        rows.append(
            {
                "treatment_name": treatment_name,
                "score": _wasserstein_distance_1d(generated_values, human_values),
            }
        )
    return pd.DataFrame(rows)


def _per_treatment_round_matched_wd(
    generated_round_df: pd.DataFrame,
    human_round_df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    treatment_names = sorted(set(generated_round_df["treatment_name"]) & set(human_round_df["treatment_name"]))
    rows: list[dict[str, Any]] = []
    for treatment_name in treatment_names:
        generated_subset = generated_round_df[generated_round_df["treatment_name"] == treatment_name]
        human_subset = human_round_df[human_round_df["treatment_name"] == treatment_name]
        round_numbers = sorted(set(generated_subset["round_number"]) & set(human_subset["round_number"]))
        round_scores: list[float] = []
        for round_number in round_numbers:
            generated_values = generated_subset.loc[generated_subset["round_number"] == round_number, metric]
            human_values = human_subset.loc[human_subset["round_number"] == round_number, metric]
            round_scores.append(_wasserstein_distance_1d(generated_values, human_values))
        rows.append(
            {
                "treatment_name": treatment_name,
                "score": float(np.mean(round_scores)) if round_scores else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _mean_stderr(scores_df: pd.DataFrame) -> tuple[float, float]:
    clean = scores_df["score"].dropna().astype(float)
    if clean.empty:
        return float("nan"), float("nan")
    mean = float(clean.mean())
    if clean.shape[0] < 2:
        return mean, float("nan")
    stderr = float(clean.std(ddof=1) / np.sqrt(clean.shape[0]))
    return mean, stderr


def _build_uniform_random_micro_score_table(
    human_game_df: pd.DataFrame,
    human_round_df: pd.DataFrame,
    human_actor_df: pd.DataFrame,
    shared_counts: pd.DataFrame,
    *,
    random_iters: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_game_df, selected_round_df, selected_actor_df = select_shared_game_skeletons(
        human_game_df,
        human_round_df,
        human_actor_df,
        shared_counts,
    )
    human_player_df = _build_player_game_summary(human_actor_df, entity_col="game_id")

    rng = np.random.default_rng(random_seed)
    draw_rows: list[dict[str, Any]] = []
    for draw_index in range(random_iters):
        _, random_actor_df, random_round_df = build_uniform_random_rollout_tables(
            selected_game_df,
            selected_round_df,
            selected_actor_df,
            rng=rng,
        )
        random_player_df = _build_player_game_summary(random_actor_df, entity_col="game_id")
        for metric, metric_label in PLAYER_METRIC_SPECS:
            score_df = _per_treatment_player_wd(random_player_df, human_player_df, metric)
            mean, _ = _mean_stderr(score_df)
            draw_rows.append(
                {
                    "metric_family": "player_within_config_wd",
                    "metric": metric,
                    "metric_label": metric_label,
                    "draw_index": draw_index,
                    "score": mean,
                }
            )
        round_specs = [
            ("round_total_contribution_rate", "Round contrib", "total_contribution_rate"),
            ("round_normalized_efficiency", "Round eff", "round_normalized_efficiency"),
        ]
        for metric_key, metric_label, value_col in round_specs:
            score_df = _per_treatment_round_matched_wd(random_round_df, human_round_df, value_col)
            mean, _ = _mean_stderr(score_df)
            draw_rows.append(
                {
                    "metric_family": "round_within_config_wd",
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "draw_index": draw_index,
                    "score": mean,
                }
            )

    draws_df = pd.DataFrame(draw_rows)
    summary_df = summarize_random_baseline_draws(
        draws_df,
        score_cols=["metric_family", "metric", "metric_label"],
    )
    return draws_df, summary_df


def _resample_entity_tables(
    actor_df: pd.DataFrame,
    round_df: pd.DataFrame,
    *,
    counts_by_treatment: dict[str, int],
    entity_col: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actor_frames: list[pd.DataFrame] = []
    round_frames: list[pd.DataFrame] = []
    for treatment_name, count in counts_by_treatment.items():
        if count <= 0:
            continue
        actor_subset = actor_df[actor_df["treatment_name"] == treatment_name]
        round_subset = round_df[round_df["treatment_name"] == treatment_name]
        entity_ids = sorted(set(round_subset[entity_col].astype(str)))
        if not entity_ids:
            continue
        sampled_ids = rng.choice(entity_ids, size=count, replace=True)
        for sample_index, source_id in enumerate(sampled_ids):
            sampled_entity_id = f"{treatment_name}__sample_{sample_index}"
            actor_block = actor_subset[actor_subset[entity_col].astype(str) == str(source_id)].copy()
            round_block = round_subset[round_subset[entity_col].astype(str) == str(source_id)].copy()
            actor_block[entity_col] = sampled_entity_id
            round_block[entity_col] = sampled_entity_id
            actor_frames.append(actor_block)
            round_frames.append(round_block)
    sampled_actor_df = pd.concat(actor_frames, ignore_index=True) if actor_frames else actor_df.iloc[0:0].copy()
    sampled_round_df = pd.concat(round_frames, ignore_index=True) if round_frames else round_df.iloc[0:0].copy()
    return sampled_actor_df, sampled_round_df


def _bootstrap_noise_ceiling(
    human_actor_df: pd.DataFrame,
    human_round_df: pd.DataFrame,
    shared_counts: pd.DataFrame,
    *,
    bootstrap_iters: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    generated_counts = {
        str(row["treatment_name"]): int(row["shared_generated_count"])
        for row in shared_counts.to_dict(orient="records")
    }
    human_counts = {
        str(row["treatment_name"]): int(row["human_count"])
        for row in shared_counts.to_dict(orient="records")
    }

    bootstrap_rows: list[dict[str, Any]] = []
    global_bootstrap_rows: list[dict[str, Any]] = []
    for bootstrap_index in range(bootstrap_iters):
        sampled_generated_actor_df, sampled_generated_round_df = _resample_entity_tables(
            human_actor_df,
            human_round_df,
            counts_by_treatment=generated_counts,
            entity_col="game_id",
            rng=rng,
        )
        sampled_human_actor_df, sampled_human_round_df = _resample_entity_tables(
            human_actor_df,
            human_round_df,
            counts_by_treatment=human_counts,
            entity_col="game_id",
            rng=rng,
        )

        sampled_generated_player_df = _build_player_game_summary(sampled_generated_actor_df, entity_col="game_id")
        sampled_human_player_df = _build_player_game_summary(sampled_human_actor_df, entity_col="game_id")
        sampled_generated_delta_df = _build_round_delta_summary(sampled_generated_round_df, entity_col="game_id")
        sampled_human_delta_df = _build_round_delta_summary(sampled_human_round_df, entity_col="game_id")

        for metric, metric_label in PLAYER_METRIC_SPECS:
            score_df = _per_treatment_player_wd(sampled_generated_player_df, sampled_human_player_df, metric)
            for row in score_df.to_dict(orient="records"):
                bootstrap_rows.append(
                    {
                        "bootstrap_index": bootstrap_index,
                        "metric_family": "player_within_config_wd",
                        "metric": metric,
                        "metric_label": metric_label,
                        **row,
                    }
                )
            global_bootstrap_rows.append(
                {
                    "bootstrap_index": bootstrap_index,
                    "metric": metric,
                    "metric_label": metric_label,
                    "score": _wasserstein_distance_1d(
                        sampled_generated_player_df[metric],
                        sampled_human_player_df[metric],
                    ),
                }
            )

        round_specs = [
            ("round_total_contribution_rate", "Round contrib", sampled_generated_round_df, sampled_human_round_df, "total_contribution_rate"),
            ("round_normalized_efficiency", "Round eff", sampled_generated_round_df, sampled_human_round_df, "round_normalized_efficiency"),
            (
                "delta_total_contribution_rate",
                "Round-to-round contrib change",
                sampled_generated_delta_df,
                sampled_human_delta_df,
                "delta_total_contribution_rate",
            ),
            (
                "delta_round_normalized_efficiency",
                "Round-to-round eff change",
                sampled_generated_delta_df,
                sampled_human_delta_df,
                "delta_round_normalized_efficiency",
            ),
        ]
        for metric_key, metric_label, generated_df, human_df, value_col in round_specs:
            score_df = _per_treatment_round_matched_wd(generated_df, human_df, value_col)
            for row in score_df.to_dict(orient="records"):
                bootstrap_rows.append(
                    {
                        "bootstrap_index": bootstrap_index,
                        "metric_family": "round_within_config_wd",
                        "metric": metric_key,
                        "metric_label": metric_label,
                        **row,
                    }
                )

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    global_bootstrap_df = pd.DataFrame(global_bootstrap_rows)

    summary_rows: list[dict[str, Any]] = []
    if not bootstrap_df.empty:
        treatment_means = (
            bootstrap_df.groupby(["metric_family", "metric", "metric_label", "treatment_name"], as_index=False)["score"]
            .mean()
        )
        for (metric_family, metric, metric_label), group in treatment_means.groupby(
            ["metric_family", "metric", "metric_label"],
            sort=True,
        ):
            mean, stderr = _mean_stderr(group)
            summary_rows.append(
                {
                    "metric_family": metric_family,
                    "metric": metric,
                    "metric_label": metric_label,
                    "noise_ceiling_mean": mean,
                    "noise_ceiling_stderr": stderr,
                    "noise_ceiling_num_treatments": int(group["treatment_name"].nunique()),
                }
            )
    return bootstrap_df, pd.DataFrame(summary_rows), global_bootstrap_df


def _draw_micro_panel(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    *,
    metric_family: str,
    metric_specs: list[tuple[str, str]],
    title: str,
) -> None:
    available_models = [model_name for model_name in MODEL_PLOT_ORDER if model_name in set(summary_df["model_name"])]
    metric_order = [metric for metric, _ in metric_specs]
    metric_labels = {metric: label for metric, label in metric_specs}
    x = np.arange(len(metric_order), dtype=float)
    bar_width = min(0.11, 0.84 / len(available_models))
    center_offset = (len(available_models) - 1) / 2.0
    family_df = summary_df[summary_df["metric_family"] == metric_family].copy()

    for index, model_name in enumerate(available_models):
        model_df = family_df[family_df["model_name"] == model_name].set_index("metric").reindex(metric_order)
        positions = x + ((index - center_offset) * bar_width)
        heights = model_df["score"].to_numpy(dtype=float)
        yerr = model_df["stderr"].to_numpy(dtype=float)
        ax.bar(
            positions,
            heights,
            width=bar_width,
            color=MODEL_STYLE[model_name]["color"],
            alpha=0.8 if model_name == "noise_ceiling" else 0.9,
            yerr=yerr,
            capsize=3,
            label="Noise ceiling" if model_name == "noise_ceiling" else model_name,
        )
    line_df = family_df.drop_duplicates("metric").set_index("metric").reindex(metric_order)
    cluster_half_width = ((len(available_models) - 1) * bar_width) / 2.0
    for metric_index, metric in enumerate(metric_order):
        line_y = line_df.loc[metric, "uniform_random_mean"]
        if pd.isna(line_y):
            continue
        ax.hlines(
            y=float(line_y),
            xmin=x[metric_index] - cluster_half_width - (0.18 * bar_width),
            xmax=x[metric_index] + cluster_half_width + (0.18 * bar_width),
            colors="#4d4d4d",
            linestyles=(0, (5, 3)),
            linewidth=1.8,
        )
    ax.set_title(title, pad=10)
    ax.set_ylabel("Lower is better")
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[metric] for metric in metric_order])


def _plot_micro_combined_figure(
    summary_df: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(19.5, 5.8), constrained_layout=False)
    fig.subplots_adjust(top=0.82, bottom=0.17, wspace=0.22)
    _draw_micro_panel(
        axes[0],
        summary_df,
        metric_family="player_within_config_wd",
        metric_specs=PLAYER_METRIC_SPECS,
        title="Between Players: Within-Config WD",
    )
    _draw_micro_panel(
        axes[1],
        summary_df,
        metric_family="round_within_config_wd",
        metric_specs=ROUND_METRIC_SPECS,
        title="Between Rounds: Round-Matched WD",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            color="#4d4d4d",
            linestyle=(0, (5, 3)),
            linewidth=1.8,
            label="Uniform random contrib baseline",
        )
    )
    labels.append("Uniform random contrib baseline")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(5, len(labels)),
        frameon=False,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _empty_noise_ceiling_bootstrap_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "bootstrap_index",
            "metric_family",
            "metric",
            "metric_label",
            "treatment_name",
            "score",
        ]
    )


def _empty_global_noise_ceiling_bootstrap_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "bootstrap_index",
            "metric",
            "metric_label",
            "score",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze micro-level distributional alignment for forecasting runs."
    )
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--run-names",
        nargs="+",
        default=list(RUN_NAME_TO_LABEL.keys()),
    )
    parser.add_argument("--bootstrap-iters", type=int, default=0)
    parser.add_argument("--random-baseline-iters", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.forecasting_root / "results" / "micro_distribution_alignment__llms"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_dirs = {
        run_name: args.forecasting_root / "results" / f"{run_name}__vs_human_treatments"
        for run_name in args.run_names
    }

    generated_game_by_model: dict[str, pd.DataFrame] = {}
    raw_tables_by_model: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    human_actor_df: pd.DataFrame | None = None
    human_round_df: pd.DataFrame | None = None
    for run_name, result_dir in result_dirs.items():
        if not result_dir.exists():
            raise FileNotFoundError(f"Missing result directory: {result_dir}")
        generated_game_df, generated_actor_df, generated_round_df, run_human_actor_df, run_human_round_df = _load_run_tables(
            result_dir
        )
        label = RUN_NAME_TO_LABEL.get(run_name, run_name)
        generated_game_by_model[label] = generated_game_df
        raw_tables_by_model[label] = (generated_actor_df, generated_round_df)
        if human_actor_df is None:
            human_actor_df = run_human_actor_df
            human_round_df = run_human_round_df

    assert human_actor_df is not None and human_round_df is not None
    result_human_game_df = next(iter(result_dirs.values())) / "human_game_summary.csv"
    human_game_df = pd.read_csv(result_human_game_df)
    shared_counts = (
        human_round_df.groupby("treatment_name")["game_id"]
        .nunique()
        .rename("human_count")
        .to_frame()
    )
    for model_name, generated_game_df in generated_game_by_model.items():
        shared_counts[model_name] = generated_game_df.groupby("treatment_name")["custom_id"].nunique()
    shared_counts = shared_counts.fillna(0).astype(int)
    shared_counts["shared_generated_count"] = shared_counts[[*generated_game_by_model.keys()]].min(axis=1)
    shared_counts = shared_counts.reset_index()

    summary_rows: list[dict[str, Any]] = []
    per_treatment_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    for model_name, generated_game_df in generated_game_by_model.items():
        generated_actor_df, generated_round_df = raw_tables_by_model[model_name]
        _, generated_actor_df, generated_round_df = _subset_generated_entity_tables(
            generated_game_df,
            generated_actor_df,
            generated_round_df,
            shared_counts,
        )
        generated_player_df = _build_player_game_summary(generated_actor_df, entity_col="custom_id")
        human_player_df = _build_player_game_summary(human_actor_df, entity_col="game_id")
        for metric, metric_label in PLAYER_METRIC_SPECS:
            score_df = _per_treatment_player_wd(generated_player_df, human_player_df, metric)
            score_df["model_name"] = model_name
            score_df["metric_family"] = "player_within_config_wd"
            score_df["metric"] = metric
            score_df["metric_label"] = metric_label
            per_treatment_rows.extend(score_df.to_dict(orient="records"))
            mean, stderr = _mean_stderr(score_df)
            summary_rows.append(
                {
                    "model_name": model_name,
                    "metric_family": "player_within_config_wd",
                    "metric": metric,
                    "metric_label": metric_label,
                    "score": mean,
                    "stderr": stderr,
                    "num_treatments": int(score_df["treatment_name"].nunique()),
                }
            )
            global_rows.append(
                {
                    "model_name": model_name,
                    "metric": metric,
                    "metric_label": metric_label,
                    "global_wd": _wasserstein_distance_1d(
                        generated_player_df[metric],
                        human_player_df[metric],
                    ),
                }
            )

        round_specs = [
            (
                "round_total_contribution_rate",
                "Round contrib",
                "total_contribution_rate",
            ),
            (
                "round_normalized_efficiency",
                "Round eff",
                "round_normalized_efficiency",
            ),
        ]
        for metric_key, metric_label, value_col in round_specs:
            score_df = _per_treatment_round_matched_wd(generated_round_df, human_round_df, value_col)
            score_df["model_name"] = model_name
            score_df["metric_family"] = "round_within_config_wd"
            score_df["metric"] = metric_key
            score_df["metric_label"] = metric_label
            per_treatment_rows.extend(score_df.to_dict(orient="records"))
            mean, stderr = _mean_stderr(score_df)
            summary_rows.append(
                {
                    "model_name": model_name,
                    "metric_family": "round_within_config_wd",
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "score": mean,
                    "stderr": stderr,
                    "num_treatments": int(score_df["treatment_name"].nunique()),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    uniform_random_draws_df, uniform_random_df = _build_uniform_random_micro_score_table(
        human_game_df,
        human_round_df,
        human_actor_df,
        shared_counts,
        random_iters=args.random_baseline_iters,
        random_seed=args.random_seed + 2000,
    )
    summary_df = summary_df.merge(
        uniform_random_df,
        on=["metric_family", "metric", "metric_label"],
        how="left",
    )
    plot_df = summary_df.copy()
    bootstrap_df = _empty_noise_ceiling_bootstrap_df()
    global_bootstrap_df = _empty_global_noise_ceiling_bootstrap_df()
    if args.bootstrap_iters > 0:
        bootstrap_df, noise_ceiling_df, global_bootstrap_df = _bootstrap_noise_ceiling(
            human_actor_df,
            human_round_df,
            shared_counts,
            bootstrap_iters=args.bootstrap_iters,
            random_seed=args.random_seed,
        )
        summary_df = summary_df.merge(
            noise_ceiling_df,
            on=["metric_family", "metric", "metric_label"],
            how="left",
        )
        summary_df["gap_to_noise_ceiling"] = summary_df["score"] - summary_df["noise_ceiling_mean"]
        summary_df["ratio_to_noise_ceiling"] = summary_df["score"] / summary_df["noise_ceiling_mean"]

        noise_rows = noise_ceiling_df.copy()
        noise_rows["model_name"] = "noise_ceiling"
        noise_rows["score"] = noise_rows["noise_ceiling_mean"]
        noise_rows["stderr"] = noise_rows["noise_ceiling_stderr"]
        noise_rows["gap_to_noise_ceiling"] = 0.0
        noise_rows["ratio_to_noise_ceiling"] = 1.0
        plot_df = pd.concat([summary_df, noise_rows.reindex(columns=summary_df.columns)], ignore_index=True)
    else:
        summary_df["noise_ceiling_mean"] = np.nan
        summary_df["noise_ceiling_stderr"] = np.nan
        summary_df["noise_ceiling_num_treatments"] = np.nan
        summary_df["gap_to_noise_ceiling"] = np.nan
        summary_df["ratio_to_noise_ceiling"] = np.nan

    global_df = pd.DataFrame(global_rows)
    if not global_bootstrap_df.empty:
        global_summary = (
            global_bootstrap_df.groupby(["metric", "metric_label"], as_index=False)
            .agg(
                noise_ceiling_mean=("score", "mean"),
                noise_ceiling_stderr=("score", lambda s: float(s.std(ddof=1) / np.sqrt(s.shape[0])) if s.shape[0] > 1 else float("nan")),
            )
        )
        global_df = global_df.merge(global_summary, on=["metric", "metric_label"], how="left")
        global_df["gap_to_noise_ceiling"] = global_df["global_wd"] - global_df["noise_ceiling_mean"]
        global_df["ratio_to_noise_ceiling"] = global_df["global_wd"] / global_df["noise_ceiling_mean"]
    else:
        global_df["noise_ceiling_mean"] = np.nan
        global_df["noise_ceiling_stderr"] = np.nan
        global_df["gap_to_noise_ceiling"] = np.nan
        global_df["ratio_to_noise_ceiling"] = np.nan

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_treatment_rows).sort_values(["metric_family", "metric", "model_name", "treatment_name"]).to_csv(
        args.output_dir / "micro_within_config_by_treatment.csv",
        index=False,
    )
    summary_df.sort_values(["metric_family", "metric", "model_name"]).to_csv(
        args.output_dir / "micro_within_config_summary.csv",
        index=False,
    )
    uniform_random_draws_df.sort_values(["metric_family", "metric", "draw_index"]).to_csv(
        args.output_dir / "micro_uniform_random_baseline_draws.csv",
        index=False,
    )
    uniform_random_df.sort_values(["metric_family", "metric"]).to_csv(
        args.output_dir / "micro_uniform_random_baseline.csv",
        index=False,
    )
    global_df.sort_values(["metric", "model_name"]).to_csv(
        args.output_dir / "micro_global_player_wd.csv",
        index=False,
    )
    bootstrap_df.sort_values(["metric_family", "metric", "treatment_name", "bootstrap_index"]).to_csv(
        args.output_dir / "micro_noise_ceiling_bootstrap.csv",
        index=False,
    )
    global_bootstrap_df.sort_values(["metric", "bootstrap_index"]).to_csv(
        args.output_dir / "micro_global_noise_ceiling_bootstrap.csv",
        index=False,
    )
    _plot_micro_combined_figure(
        plot_df,
        output_path=args.output_dir / "micro_distribution_alignment.png",
    )

    manifest = {
        "run_names": args.run_names,
        "bootstrap_iters": args.bootstrap_iters,
        "random_seed": args.random_seed,
        "player_metrics": [metric for metric, _ in PLAYER_METRIC_SPECS],
        "round_metrics": [metric for metric, _ in ROUND_METRIC_SPECS],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
