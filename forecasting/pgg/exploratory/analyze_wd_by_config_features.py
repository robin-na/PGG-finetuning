from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_NAME_TO_LABEL = {
    "baseline_gpt_5_1": "gpt-5.1 baseline",
    "twin_sampled_seed_0_gpt_5_1": "gpt-5.1 twin",
    "baseline_gpt_5_mini": "gpt-5-mini baseline",
    "twin_sampled_seed_0_gpt_5_mini": "gpt-5-mini twin",
}

MODEL_ORDER = [
    "gpt-5.1 baseline",
    "gpt-5.1 twin",
    "gpt-5-mini baseline",
    "gpt-5-mini twin",
]

FEATURE_SPECS = [
    ("chat_enabled", "Chat enabled"),
    ("punishment_enabled", "Punishment enabled"),
    ("reward_enabled", "Reward enabled"),
    ("all_or_nothing", "All-or-nothing"),
    ("opt_out_default", "Opt-out default"),
    ("show_n_rounds", "Show # rounds"),
    ("show_other_summaries", "Show outcome summaries"),
    ("show_actor_id", "Show actor IDs"),
]

MACRO_METRIC_LABELS = {
    "mean_total_contribution_rate": "Mean contrib",
    "first_round_total_contribution_rate": "First-round contrib",
    "final_total_contribution_rate": "Final-round contrib",
    "mean_round_normalized_efficiency": "Mean eff",
    "first_round_normalized_efficiency": "First-round eff",
    "final_round_normalized_efficiency": "Final-round eff",
}

MICRO_METRIC_ORDER = [
    "Player mean contrib",
    "Player mean payoff",
    "Round contrib",
    "Round eff",
    "Round-to-round contrib change",
    "Round-to-round eff change",
]


def _load_validation_feature_metadata(repo_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(repo_root / "data" / "exp_config_files" / "validation.csv")
    frame = frame.rename(
        columns={
            "treatmentName": "treatment_name",
            "chat": "chat_enabled",
            "punishmentExists": "punishment_enabled",
            "rewardExists": "reward_enabled",
            "numRounds": "config_num_rounds",
            "allOrNothing": "all_or_nothing",
            "playerCount": "player_count",
            "showNRounds": "show_n_rounds",
            "showOtherSummaries": "show_other_summaries",
            "showPunishmentId": "show_punishment_id",
            "showRewardId": "show_reward_id",
            "defaultContribProp": "opt_out_default",
            "multiplier": "multiplier",
        }
    ).copy()
    frame["mpcr"] = frame["multiplier"] / frame["player_count"]
    frame["show_actor_id"] = frame["show_punishment_id"] | frame["show_reward_id"]
    for col in [
        "chat_enabled",
        "punishment_enabled",
        "reward_enabled",
        "all_or_nothing",
        "show_n_rounds",
        "show_other_summaries",
        "show_punishment_id",
        "show_reward_id",
        "show_actor_id",
        "opt_out_default",
    ]:
        frame[col] = frame[col].astype(bool)
    return frame[
        [
            "treatment_name",
            "chat_enabled",
            "punishment_enabled",
            "reward_enabled",
            "all_or_nothing",
            "opt_out_default",
            "show_n_rounds",
            "show_other_summaries",
            "show_actor_id",
            "config_num_rounds",
            "player_count",
            "mpcr",
        ]
    ].drop_duplicates("treatment_name")


def _mean_stderr(values: pd.Series) -> tuple[float, float, int]:
    clean = values.dropna().astype(float)
    count = int(clean.shape[0])
    if count == 0:
        return float("nan"), float("nan"), 0
    mean = float(clean.mean())
    if count == 1:
        return mean, float("nan"), count
    stderr = float(clean.std(ddof=1) / np.sqrt(count))
    return mean, stderr, count


def _build_macro_treatment_df(forecasting_root: Path, metadata_df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_name, model_name in RUN_NAME_TO_LABEL.items():
        result_dir = forecasting_root / "results" / f"{run_name}__vs_human_treatments"
        frame = pd.read_csv(result_dir / "treatment_wasserstein_distance.csv")
        frame = frame[frame["metric"].isin(MACRO_METRIC_LABELS)].copy()
        frame["metric_label"] = frame["metric"].map(MACRO_METRIC_LABELS)
        frame["model_name"] = model_name
        frames.append(frame)
    macro_df = pd.concat(frames, ignore_index=True)
    macro_df = macro_df.rename(columns={"wasserstein_distance": "score"})
    macro_df["domain"] = "macro"
    return macro_df.merge(metadata_df, on="treatment_name", how="left")


def _build_micro_treatment_df(forecasting_root: Path, metadata_df: pd.DataFrame) -> pd.DataFrame:
    micro_df = pd.read_csv(
        forecasting_root / "results" / "micro_distribution_alignment__llms" / "micro_within_config_by_treatment.csv"
    ).copy()
    micro_df["domain"] = "micro"
    return micro_df.merge(metadata_df, on="treatment_name", how="left")


def _summarize_by_feature(score_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    level_rows: list[dict[str, object]] = []
    diff_rows: list[dict[str, object]] = []

    for domain, domain_df in score_df.groupby("domain", sort=True):
        for feature_name, feature_label in FEATURE_SPECS:
            for (metric_label, model_name), group in domain_df.groupby(["metric_label", "model_name"], sort=True):
                feature_group = group.dropna(subset=[feature_name]).copy()
                low = feature_group[feature_group[feature_name] == False]
                high = feature_group[feature_group[feature_name] == True]

                low_mean, low_stderr, low_n = _mean_stderr(low["score"])
                high_mean, high_stderr, high_n = _mean_stderr(high["score"])

                level_rows.extend(
                    [
                        {
                            "domain": domain,
                            "feature_name": feature_name,
                            "feature_label": feature_label,
                            "feature_value": False,
                            "metric_label": metric_label,
                            "model_name": model_name,
                            "mean_wd": low_mean,
                            "stderr": low_stderr,
                            "num_treatments": low_n,
                        },
                        {
                            "domain": domain,
                            "feature_name": feature_name,
                            "feature_label": feature_label,
                            "feature_value": True,
                            "metric_label": metric_label,
                            "model_name": model_name,
                            "mean_wd": high_mean,
                            "stderr": high_stderr,
                            "num_treatments": high_n,
                        },
                    ]
                )

                diff = high_mean - low_mean
                diff_stderr = float(
                    np.sqrt(
                        (0.0 if np.isnan(high_stderr) else high_stderr**2)
                        + (0.0 if np.isnan(low_stderr) else low_stderr**2)
                    )
                )
                diff_rows.append(
                    {
                        "domain": domain,
                        "feature_name": feature_name,
                        "feature_label": feature_label,
                        "metric_label": metric_label,
                        "model_name": model_name,
                        "wd_diff_true_minus_false": diff,
                        "diff_stderr": diff_stderr,
                        "true_num_treatments": high_n,
                        "false_num_treatments": low_n,
                    }
                )

    return pd.DataFrame(level_rows), pd.DataFrame(diff_rows)


def _metric_order_for_domain(domain: str) -> list[str]:
    if domain == "macro":
        return list(MACRO_METRIC_LABELS.values())
    return MICRO_METRIC_ORDER


def _plot_feature_heatmap(diff_df: pd.DataFrame, *, domain: str, output_path: Path) -> None:
    domain_df = diff_df[diff_df["domain"] == domain].copy()
    metric_order = [metric for metric in _metric_order_for_domain(domain) if metric in set(domain_df["metric_label"])]
    features = FEATURE_SPECS
    ncols = 4
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(20, max(6.8, 0.7 * len(metric_order) * nrows + 1.5)),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)
    flat_axes = list(axes_array.ravel())

    max_abs = domain_df["wd_diff_true_minus_false"].abs().max()
    color_limit = float(max(max_abs, 1e-9))

    im = None
    for ax, (feature_name, feature_label) in zip(flat_axes, features, strict=True):
        feature_df = domain_df[domain_df["feature_name"] == feature_name].copy()
        pivot = (
            feature_df.pivot_table(
                index="metric_label",
                columns="model_name",
                values="wd_diff_true_minus_false",
                aggfunc="first",
            )
            .reindex(index=metric_order, columns=MODEL_ORDER)
        )
        matrix = pivot.to_numpy(dtype=float)
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-color_limit, vmax=color_limit, aspect="auto")
        ax.set_title(feature_label)
        ax.set_xticks(np.arange(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_ORDER, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(metric_order)))
        ax.set_yticklabels(metric_order)
        for row_index, metric_label in enumerate(metric_order):
            for col_index, model_name in enumerate(MODEL_ORDER):
                value = matrix[row_index, col_index]
                if np.isnan(value):
                    text = "NA"
                else:
                    text = f"{value:+.03f}"
                ax.text(
                    col_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    for ax in flat_axes[len(features):]:
        ax.axis("off")

    colorbar = fig.colorbar(im, ax=flat_axes, shrink=0.9, location="bottom", pad=0.08)
    colorbar.set_label("Enabled minus disabled mean WD (negative means better when enabled)")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_twin_reduction_summary(diff_df: pd.DataFrame) -> pd.DataFrame:
    family_map = {
        "gpt-5.1 baseline": ("gpt-5.1", "baseline"),
        "gpt-5.1 twin": ("gpt-5.1", "twin"),
        "gpt-5-mini baseline": ("gpt-5-mini", "baseline"),
        "gpt-5-mini twin": ("gpt-5-mini", "twin"),
    }
    current = diff_df.copy()
    current["model_family"] = current["model_name"].map(lambda value: family_map[value][0])
    current["variant"] = current["model_name"].map(lambda value: family_map[value][1])
    current["abs_diff"] = current["wd_diff_true_minus_false"].abs()

    by_feature = (
        current.groupby(["domain", "feature_name", "feature_label", "model_family", "variant"], as_index=False)["abs_diff"]
        .mean()
        .pivot_table(
            index=["domain", "feature_name", "feature_label", "model_family"],
            columns="variant",
            values="abs_diff",
        )
        .reset_index()
    )
    by_feature["twin_minus_baseline_abs_diff"] = by_feature["twin"] - by_feature["baseline"]
    by_feature["twin_reduces_dependence"] = by_feature["twin_minus_baseline_abs_diff"] < 0

    overall = (
        current.groupby(["domain", "model_family", "variant"], as_index=False)["abs_diff"]
        .mean()
        .pivot_table(index=["domain", "model_family"], columns="variant", values="abs_diff")
        .reset_index()
    )
    overall["twin_minus_baseline_abs_diff"] = overall["twin"] - overall["baseline"]
    overall["twin_reduces_dependence"] = overall["twin_minus_baseline_abs_diff"] < 0
    overall["feature_name"] = "ALL"
    overall["feature_label"] = "All features"

    return pd.concat([by_feature, overall], ignore_index=True, sort=False)


def _plot_twin_dependence_trend(summary_df: pd.DataFrame, output_path: Path) -> None:
    domain_order = ["macro", "micro"]
    domain_titles = {
        "macro": "Macro WD Feature Dependence",
        "micro": "Micro WD Feature Dependence",
    }
    feature_order = [
        "All features",
        "Chat enabled",
        "Punishment enabled",
        "Reward enabled",
        "All-or-nothing",
        "Opt-out default",
        "Show # rounds",
        "Show outcome summaries",
        "Show actor IDs",
    ]
    family_colors = {
        "gpt-5.1": "#1f77b4",
        "gpt-5-mini": "#ff7f0e",
    }
    variant_markers = {
        "baseline": "o",
        "twin": "D",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.2), constrained_layout=True, sharex=False, sharey=True)

    for ax, domain in zip(axes, domain_order, strict=True):
        domain_df = summary_df[summary_df["domain"] == domain].copy()
        feature_to_y = {label: index for index, label in enumerate(feature_order)}
        for model_family, offset in [("gpt-5.1", -0.12), ("gpt-5-mini", 0.12)]:
            family_df = domain_df[domain_df["model_family"] == model_family].copy()
            for feature_label in feature_order:
                row = family_df[family_df["feature_label"] == feature_label]
                if row.empty:
                    continue
                row = row.iloc[0]
                y = feature_to_y[feature_label] + offset
                baseline = float(row["baseline"])
                twin = float(row["twin"])
                ax.annotate(
                    "",
                    xy=(twin, y),
                    xytext=(baseline, y),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "lw": 2.2,
                        "color": family_colors[model_family],
                        "alpha": 0.7,
                        "shrinkA": 3,
                        "shrinkB": 3,
                    },
                    zorder=2,
                )
                ax.scatter(
                    [baseline],
                    [y],
                    color=family_colors[model_family],
                    marker=variant_markers["baseline"],
                    s=55,
                    zorder=3,
                    label=f"{model_family} baseline" if feature_label == feature_order[0] else None,
                )
                ax.scatter(
                    [twin],
                    [y],
                    color=family_colors[model_family],
                    marker=variant_markers["twin"],
                    s=60,
                    zorder=3,
                    label=f"{model_family} twin" if feature_label == feature_order[0] else None,
                )

        ax.set_title(domain_titles[domain], pad=10)
        ax.set_xlabel("Absolute enabled-vs-disabled WD gap\n(lower means less config dependence)")
        ax.set_yticks(np.arange(len(feature_order), dtype=float))
        ax.set_yticklabels(feature_order)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze WD dependence on validation config features.")
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.forecasting_root / "results" / "wd_by_config_features"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = _load_validation_feature_metadata(args.repo_root)
    macro_df = _build_macro_treatment_df(args.forecasting_root, metadata_df)
    micro_df = _build_micro_treatment_df(args.forecasting_root, metadata_df)
    combined_df = pd.concat([macro_df, micro_df], ignore_index=True, sort=False)

    level_df, diff_df = _summarize_by_feature(combined_df)
    twin_reduction_df = _build_twin_reduction_summary(diff_df)
    level_df.sort_values(["domain", "feature_name", "metric_label", "model_name", "feature_value"]).to_csv(
        args.output_dir / "wd_by_config_feature_levels.csv",
        index=False,
    )
    diff_df.sort_values(["domain", "feature_name", "metric_label", "model_name"]).to_csv(
        args.output_dir / "wd_by_config_feature_differences.csv",
        index=False,
    )
    twin_reduction_df.sort_values(["domain", "feature_name", "model_family"]).to_csv(
        args.output_dir / "wd_feature_dependence_twin_reduction.csv",
        index=False,
    )

    _plot_feature_heatmap(
        diff_df,
        domain="macro",
        output_path=args.output_dir / "macro_wd_by_config_feature.png",
    )
    _plot_feature_heatmap(
        diff_df,
        domain="micro",
        output_path=args.output_dir / "micro_wd_by_config_feature.png",
    )
    _plot_twin_dependence_trend(
        twin_reduction_df,
        args.output_dir / "wd_feature_dependence_twin_trend.png",
    )

    manifest = {
        "features": [feature_name for feature_name, _ in FEATURE_SPECS],
        "macro_metrics": list(MACRO_METRIC_LABELS.keys()),
        "micro_metrics": MICRO_METRIC_ORDER,
        "models": MODEL_ORDER,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
