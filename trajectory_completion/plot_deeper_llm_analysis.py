from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


MODEL_LABELS = {
    "gpt_4_1_mini": "GPT-4.1-mini",
    "gpt_5_1": "GPT-5.1",
}

MODEL_COLORS = {
    "gpt_4_1_mini": "#1f77b4",
    "gpt_5_1": "#17becf",
}

FLAG_LABELS = {
    "chat_enabled": "Chat",
    "all_or_nothing": "All-Or-Nothing",
    "show_n_rounds": "Show N Rounds",
    "show_interaction_id": "Show Interaction ID",
}

FLAG_LINESTYLES = {
    False: "-",
    True: "--",
}


def _load_common_games(path: Path) -> dict[int, set[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): set(v) for k, v in manifest["common_games_by_k"].items()}


def _filter_to_common_subset(df: pd.DataFrame, common_games_by_k: dict[int, set[str]]) -> pd.DataFrame:
    parts = []
    for k, game_ids in common_games_by_k.items():
        parts.append(df[(df["k"] == k) & (df["game_id"].astype(str).isin(game_ids))])
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def _load_config_metadata(path: Path) -> pd.DataFrame:
    cols = [
        "gameId",
        "CONFIG_showNRounds",
        "CONFIG_allOrNothing",
        "CONFIG_chat",
        "CONFIG_showPunishmentId",
        "CONFIG_showRewardId",
    ]
    df = pd.read_csv(path, usecols=cols).rename(columns={"gameId": "game_id"})
    # In this selected validation subset these two move together, so one combined visibility flag is enough.
    df["show_interaction_id"] = df["CONFIG_showPunishmentId"].astype(bool)
    df["chat_enabled"] = df["CONFIG_chat"].astype(bool)
    df["all_or_nothing"] = df["CONFIG_allOrNothing"].astype(bool)
    df["show_n_rounds"] = df["CONFIG_showNRounds"].astype(bool)
    return df[["game_id", "chat_enabled", "all_or_nothing", "show_n_rounds", "show_interaction_id"]]


def _build_similarity_plot_data(actor_similarity_csv: Path, trajectory_similarity_csv: Path) -> pd.DataFrame:
    actor = pd.read_csv(actor_similarity_csv)
    traj = pd.read_csv(trajectory_similarity_csv)
    merged = actor.merge(
        traj[["k", "full_contribution_trajectory_match"]],
        on="k",
        how="left",
    )
    return merged.sort_values("k").reset_index(drop=True)


def _plot_similarity(similarity_df: pd.DataFrame, output_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.flatten()
    metrics = [
        ("actor_round_contribution_match_rate", "Actor-Round Contribution Match Rate", True),
        ("actor_round_mean_contribution_abs_diff", "Actor-Round Contribution Abs Diff", False),
        ("full_contribution_trajectory_match", "Full Contribution Trajectory Match Rate", True),
        ("actor_round_punished_dict_match_rate", "Punish Dict Match Rate", True),
    ]

    x = similarity_df["k"]
    for ax, (metric, title, higher_is_better) in zip(axes, metrics):
        ax.plot(
            x,
            similarity_df[metric],
            color="#2b2b2b",
            marker="o",
            linewidth=2.5,
        )
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.set_xticks(sorted(x.unique()))
        ax.grid(True, alpha=0.25)
        if higher_is_better:
            ax.set_ylim(0, 1.02)
        else:
            ax.set_ylim(bottom=0)

    fig.suptitle("GPT-4.1-mini vs GPT-5.1 Prediction Similarity (Common Subset)", fontsize=15, y=1.02)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _build_config_plot_data(
    *,
    common_manifest_json: Path,
    df_analysis_csv: Path,
    model_game_summaries: dict[str, Path],
) -> pd.DataFrame:
    common_games_by_k = _load_common_games(common_manifest_json)
    config_df = _load_config_metadata(df_analysis_csv)
    rows = []
    for model_name, game_summary_csv in model_game_summaries.items():
        df = _filter_to_common_subset(pd.read_csv(game_summary_csv), common_games_by_k)
        df = df.merge(config_df, on="game_id", how="left")
        for flag in ["chat_enabled", "all_or_nothing", "show_n_rounds", "show_interaction_id"]:
            grouped = (
                df.groupby(["k", flag], as_index=False)[
                    ["contribution_rate_mae", "future_normalized_efficiency_abs_error"]
                ]
                .mean()
                .rename(columns={flag: "flag_value"})
            )
            grouped["flag_name"] = flag
            grouped["model"] = model_name
            rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def _plot_config_effects(plot_df: pd.DataFrame, output_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
        }
    )
    flags = ["chat_enabled", "all_or_nothing", "show_n_rounds", "show_interaction_id"]
    metrics = [
        ("contribution_rate_mae", "Contribution Rate MAE"),
        ("future_normalized_efficiency_abs_error", "Future Normalized Efficiency Error"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

    for row_idx, (metric, row_title) in enumerate(metrics):
        for col_idx, flag in enumerate(flags):
            ax = axes[row_idx, col_idx]
            panel_df = plot_df[plot_df["flag_name"] == flag].copy()
            for model_name in ["gpt_4_1_mini", "gpt_5_1"]:
                for flag_value in [False, True]:
                    series_df = panel_df[
                        (panel_df["model"] == model_name) & (panel_df["flag_value"].astype(bool) == flag_value)
                    ].sort_values("k")
                    if series_df.empty:
                        continue
                    ax.plot(
                        series_df["k"],
                        series_df[metric],
                        color=MODEL_COLORS[model_name],
                        linestyle=FLAG_LINESTYLES[flag_value],
                        marker="o",
                        linewidth=2,
                    )
            if row_idx == 0:
                ax.set_title(FLAG_LABELS[flag])
            if col_idx == 0:
                ax.set_ylabel(row_title)
            ax.set_xlabel("k")
            ax.set_xticks(sorted(plot_df["k"].unique()))
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.25)

    model_handles = [
        Line2D([0], [0], color=MODEL_COLORS[name], linewidth=2.5, label=MODEL_LABELS[name])
        for name in ["gpt_4_1_mini", "gpt_5_1"]
    ]
    flag_handles = [
        Line2D([0], [0], color="#444444", linestyle=FLAG_LINESTYLES[value], linewidth=2.5, label=label)
        for value, label in [(False, "Flag=False"), (True, "Flag=True")]
    ]
    fig.legend(
        handles=model_handles + flag_handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        framealpha=1.0,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("Config Effects on LLM Forecasting (Common Subset)", fontsize=15, y=1.02)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot deeper LLM similarity and config analyses.")
    parser.add_argument("--deeper-analysis-dir", type=Path, required=True)
    parser.add_argument("--common-manifest-json", type=Path, required=True)
    parser.add_argument("--df-analysis-csv", type=Path, required=True)
    parser.add_argument("--gpt4-game-summary-csv", type=Path, required=True)
    parser.add_argument("--gpt5-game-summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    similarity_df = _build_similarity_plot_data(
        actor_similarity_csv=args.deeper_analysis_dir / "inter_model_actor_similarity.csv",
        trajectory_similarity_csv=args.deeper_analysis_dir / "inter_model_trajectory_similarity.csv",
    )
    config_plot_df = _build_config_plot_data(
        common_manifest_json=args.common_manifest_json,
        df_analysis_csv=args.df_analysis_csv,
        model_game_summaries={
            "gpt_4_1_mini": args.gpt4_game_summary_csv,
            "gpt_5_1": args.gpt5_game_summary_csv,
        },
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    similarity_df.to_csv(args.output_dir / "similarity_plot_data.csv", index=False)
    config_plot_df.to_csv(args.output_dir / "config_effect_plot_data.csv", index=False)

    _plot_similarity(similarity_df, args.output_dir / "gpt4mini_vs_gpt5_similarity.png")
    _plot_config_effects(config_plot_df, args.output_dir / "config_effects.png")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
