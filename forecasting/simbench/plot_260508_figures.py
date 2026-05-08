from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
PREDICTION_RUNS_DIR = SCRIPT_DIR / "transfer_prediction" / "runs"
OUTPUT_DIR = RESULTS_DIR / "260508"
SIMBENCH_PAPER_DATASET_ORDER = [
    "DICES",
    "OpinionQA",
    "MoralMachineClassic",
    "WisdomOfCrowds",
    "OSPsychMGKT",
    "Afrobarometer",
    "ChaosNLI",
    "GlobalOpinionQA",
    "OSPsychBig5",
    "ISSP",
    "ESS",
    "OSPsychRWAS",
    "TISP",
    "ConspiracyCorr",
    "NumberGame",
    "LatinoBarometro",
    "Choices13k",
    "Jester",
    "OSPsychMACH",
    "MoralMachine",
]


def _load_mini_task_scores() -> tuple[pd.DataFrame, dict[str, float]]:
    dataset_path = (
        RESULTS_DIR
        / "simbenchpop__baseline_vs_persona_summary_overlap__gpt_5_mini"
        / "dataset_overlap_simbench_score_summary.csv"
    )
    overall_path = (
        RESULTS_DIR / "simbenchpop__baseline_vs_persona_summary_overlap__gpt_5_mini" / "overall_overlap_summary.json"
    )

    df = pd.read_csv(dataset_path).rename(
        columns={
            "baseline_mean_simbench_score": "baseline_score",
            "persona_summary_mean_simbench_score": "augmented_score",
            "delta_simbench_score_persona_summary_minus_baseline": "delta_score",
        }
    )
    overall = json.loads(overall_path.read_text(encoding="utf-8"))
    summary = {
        "baseline_mean_simbench_score": float(overall["baseline_mean_simbench_score"]),
        "augmented_mean_simbench_score": float(overall["persona_summary_mean_simbench_score"]),
        "delta_mean_simbench_score": float(
            overall["persona_summary_mean_simbench_score"] - overall["baseline_mean_simbench_score"]
        ),
        "n_overlap_rows": int(overall["rows_in_overlap"]),
    }
    return df[["dataset_name", "baseline_score", "augmented_score", "delta_score"]], summary


def _load_nano_task_scores() -> tuple[pd.DataFrame, dict[str, float]]:
    baseline_path = (
        RESULTS_DIR / "simbenchpop__baseline_group_batched_explained__gpt_5_nano__us_only__gold_eval"
        / "row_level_evaluation.csv"
    )
    augmented_path = (
        RESULTS_DIR
        / "simbenchpop__twin_persona_summary_batched_seed_0__n64__gpt_5_nano__us_only__gold_eval"
        / "row_level_evaluation.csv"
    )
    overall_path = (
        RESULTS_DIR
        / "simbenchpop__baseline_vs_persona_summary_overlap__gpt_5_nano"
        / "overall_overlap_simbench_score_summary.json"
    )

    baseline = pd.read_csv(baseline_path)
    augmented = pd.read_csv(augmented_path)
    baseline = baseline[baseline["evaluated"].fillna(False).astype(bool)][
        ["simbench_row_id", "dataset_name", "simbench_score"]
    ].rename(columns={"simbench_score": "baseline_score"})
    augmented = augmented[augmented["evaluated"].fillna(False).astype(bool)][
        ["simbench_row_id", "dataset_name", "simbench_score"]
    ].rename(columns={"simbench_score": "augmented_score"})
    merged = baseline.merge(augmented, on=["simbench_row_id", "dataset_name"], how="inner")
    dataset = (
        merged.groupby("dataset_name", as_index=False)[["baseline_score", "augmented_score"]]
        .mean()
        .assign(delta_score=lambda df: df["augmented_score"] - df["baseline_score"])
    )
    overall = json.loads(overall_path.read_text(encoding="utf-8"))
    summary = {
        "baseline_mean_simbench_score": float(overall["baseline_mean_simbench_score"]),
        "augmented_mean_simbench_score": float(overall["persona_summary_mean_simbench_score"]),
        "delta_mean_simbench_score": float(overall["delta_persona_minus_baseline"]),
        "n_overlap_rows": int(overall["n_rows"]),
    }
    return dataset, summary


def _load_predictor_order() -> list[str]:
    path = PREDICTION_RUNS_DIR / "gpt_5_1_global_ranking_response_normalized.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(name) for name in payload["ranking_most_helpful_to_most_harmful"]]


def _load_baseline_heatmap_frame() -> pd.DataFrame:
    mini_path = (
        RESULTS_DIR
        / "simbenchpop__baseline_vs_persona_summary_overlap__gpt_5_mini"
        / "dataset_overlap_simbench_score_summary.csv"
    )
    nano_path = (
        RESULTS_DIR
        / "simbenchpop__persona_summary_ablation_compare__gpt_5_nano__us_only"
        / "dataset_summary_common_rows.csv"
    )

    mini = pd.read_csv(mini_path)[["dataset_name", "baseline_mean_simbench_score"]].rename(
        columns={"baseline_mean_simbench_score": "gpt-5-mini"}
    )
    nano = (
        pd.read_csv(nano_path)
        .query("label == 'baseline'")
        [["dataset_name", "mean_simbench_score"]]
        .rename(columns={"mean_simbench_score": "gpt-5-nano"})
    )

    merged = pd.DataFrame({"dataset_name": SIMBENCH_PAPER_DATASET_ORDER}).merge(mini, on="dataset_name", how="left")
    merged = merged.merge(nano, on="dataset_name", how="left")
    return merged


def _plot_baseline_heatmap(output_dir: Path) -> pd.DataFrame:
    frame = _load_baseline_heatmap_frame()
    frame.to_csv(output_dir / "baseline_heatmap_data.csv", index=False)

    rows = ["gpt-5-mini", "gpt-5-nano"]
    values = np.array([[frame[row].iloc[col] for col in range(len(frame))] for row in rows], dtype=float)
    masked = np.ma.masked_invalid(values)

    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#e5e7eb")

    finite = values[np.isfinite(values)]
    vmax = max(100.0, float(np.nanmax(finite))) if finite.size else 100.0
    vmin = min(-180.0, float(np.nanmin(finite))) if finite.size else -180.0

    fig, ax = plt.subplots(figsize=(16.5, 3.4))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=11)
    ax.set_xticks(np.arange(len(frame)))
    ax.set_xticklabels(frame["dataset_name"], rotation=45, ha="right", fontsize=9)

    for i in range(len(rows)):
        for j in range(len(frame)):
            value = values[i, j]
            if np.isnan(value):
                text = "N/A"
                color = "#374151"
            else:
                text = f"{value:.1f}"
                norm_value = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                color = "white" if norm_value < 0.2 or norm_value > 0.8 else "#111827"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(frame), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("SimBench Score", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "baseline_simbench_score_heatmap__mini_vs_nano.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "baseline_simbench_score_heatmap__mini_vs_nano.pdf", bbox_inches="tight")
    plt.close(fig)
    return frame


def _load_significance_table(model: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"simbenchpop__baseline_vs_persona_summary_overlap__{model}" / "simbench_score_significance_summary.csv"
    df = pd.read_csv(path)
    keep = [
        "dataset_name",
        "delta_persona_minus_baseline",
        "p_value_bh",
        "significant_bh_0_05",
        "direction",
    ]
    return df[keep].copy()


def _plot_raw_score_heterogeneity(
    mini_df: pd.DataFrame,
    nano_df: pd.DataFrame,
    mini_summary: dict[str, float],
    nano_summary: dict[str, float],
    output_dir: Path,
) -> pd.DataFrame:
    combined = mini_df.merge(
        nano_df,
        on="dataset_name",
        how="inner",
        suffixes=("_mini", "_nano"),
    )
    combined["avg_delta"] = (combined["delta_score_mini"] + combined["delta_score_nano"]) / 2.0
    order = combined.sort_values("avg_delta", ascending=False)["dataset_name"].tolist()

    plot_df = pd.concat(
        [
            mini_df.assign(model="gpt-5-mini"),
            nano_df.assign(model="gpt-5-nano"),
        ],
        ignore_index=True,
    )
    plot_df["dataset_name"] = pd.Categorical(plot_df["dataset_name"], categories=order, ordered=True)
    plot_df = plot_df.sort_values(["model", "dataset_name"]).reset_index(drop=True)
    plot_df.to_csv(output_dir / "heterogeneity_raw_simbench_score_data.csv", index=False)

    x = np.arange(len(order))
    width = 0.38
    baseline_color = "#6e7781"
    augmented_color = "#0f766e"

    all_scores = pd.concat(
        [
            mini_df[["baseline_score", "augmented_score"]],
            nano_df[["baseline_score", "augmented_score"]],
        ],
        ignore_index=True,
    ).to_numpy()
    ymin = float(np.nanmin(all_scores))
    ymax = float(np.nanmax(all_scores))
    pad = 0.08 * (ymax - ymin)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), sharey=True)
    for ax, df, model_label, summary in [
        (axes[0], mini_df.set_index("dataset_name").loc[order].reset_index(), "gpt-5-mini", mini_summary),
        (axes[1], nano_df.set_index("dataset_name").loc[order].reset_index(), "gpt-5-nano", nano_summary),
    ]:
        ax.bar(x - width / 2, df["baseline_score"], width=width, color=baseline_color, alpha=0.88, label="Baseline")
        ax.bar(
            x + width / 2,
            df["augmented_score"],
            width=width,
            color=augmented_color,
            alpha=0.88,
            label="Twin persona_summary",
        )
        ax.axhline(0, color="#444444", linewidth=1.0, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=55, ha="right", fontsize=8)
        ax.set_title(model_label, fontsize=13, pad=10)
        ax.grid(axis="y", color="#d0d7de", alpha=0.5, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.text(
            0.01,
            0.98,
            f"Δ mean score = {summary['delta_mean_simbench_score']:+.2f}\n"
            f"baseline = {summary['baseline_mean_simbench_score']:.2f}, "
            f"augmented = {summary['augmented_mean_simbench_score']:.2f}\n"
            f"overlap rows = {summary['n_overlap_rows']}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.9},
        )

    axes[0].set_ylabel("Mean SimBench Score", fontsize=11)
    axes[1].legend(loc="lower right", frameon=False, fontsize=9)
    axes[0].set_ylim(ymin - pad, ymax + pad)
    fig.tight_layout()
    fig.savefig(output_dir / "heterogeneity_raw_simbench_score__mini_vs_nano.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "heterogeneity_raw_simbench_score__mini_vs_nano.pdf", bbox_inches="tight")
    plt.close(fig)
    return plot_df


def _plot_delta_score_heterogeneity(
    mini_df: pd.DataFrame,
    nano_df: pd.DataFrame,
    mini_summary: dict[str, float],
    nano_summary: dict[str, float],
    output_dir: Path,
) -> pd.DataFrame:
    combined = mini_df.merge(
        nano_df,
        on="dataset_name",
        how="inner",
        suffixes=("_mini", "_nano"),
    )
    combined["avg_delta"] = (combined["delta_score_mini"] + combined["delta_score_nano"]) / 2.0
    order = combined.sort_values("avg_delta", ascending=False)["dataset_name"].tolist()

    plot_df = pd.concat(
        [
            mini_df.assign(model="gpt-5-mini"),
            nano_df.assign(model="gpt-5-nano"),
        ],
        ignore_index=True,
    )
    plot_df["dataset_name"] = pd.Categorical(plot_df["dataset_name"], categories=order, ordered=True)
    plot_df = plot_df.sort_values(["model", "dataset_name"]).reset_index(drop=True)
    plot_df.to_csv(output_dir / "heterogeneity_delta_simbench_score_data.csv", index=False)

    x = np.arange(len(order))
    delta_color = "#0f766e"
    delta_negative_color = "#b42318"

    all_deltas = pd.concat([mini_df["delta_score"], nano_df["delta_score"]], ignore_index=True).to_numpy()
    ymin = float(np.nanmin(all_deltas))
    ymax = float(np.nanmax(all_deltas))
    pad = 0.08 * (ymax - ymin if ymax > ymin else 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), sharey=True)
    for ax, df, model_label, summary in [
        (axes[0], mini_df.set_index("dataset_name").loc[order].reset_index(), "gpt-5-mini", mini_summary),
        (axes[1], nano_df.set_index("dataset_name").loc[order].reset_index(), "gpt-5-nano", nano_summary),
    ]:
        colors = [delta_color if value >= 0 else delta_negative_color for value in df["delta_score"]]
        ax.bar(x, df["delta_score"], color=colors, alpha=0.9)
        ax.axhline(0, color="#444444", linewidth=1.0, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=55, ha="right", fontsize=8)
        ax.set_title(model_label, fontsize=13, pad=10)
        ax.grid(axis="y", color="#d0d7de", alpha=0.5, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.text(
            0.01,
            0.98,
            f"Δ mean score = {summary['delta_mean_simbench_score']:+.2f}\n"
            f"baseline = {summary['baseline_mean_simbench_score']:.2f}, "
            f"augmented = {summary['augmented_mean_simbench_score']:.2f}\n"
            f"overlap rows = {summary['n_overlap_rows']}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.9},
        )

    axes[0].set_ylabel("Δ Mean SimBench Score\n(persona_summary − baseline)", fontsize=11)
    axes[0].set_ylim(ymin - pad, ymax + pad)
    fig.tight_layout()
    fig.savefig(output_dir / "heterogeneity_delta_simbench_score__mini_vs_nano.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "heterogeneity_delta_simbench_score__mini_vs_nano.pdf", bbox_inches="tight")
    plt.close(fig)
    return plot_df


def _plot_delta_score_grouped_by_model(
    mini_df: pd.DataFrame,
    nano_df: pd.DataFrame,
    mini_summary: dict[str, float],
    nano_summary: dict[str, float],
    output_dir: Path,
) -> pd.DataFrame:
    combined = mini_df.merge(
        nano_df,
        on="dataset_name",
        how="inner",
        suffixes=("_mini", "_nano"),
    )
    combined["avg_delta"] = (combined["delta_score_mini"] + combined["delta_score_nano"]) / 2.0
    combined = combined.sort_values("avg_delta", ascending=False).reset_index(drop=True)
    combined.to_csv(output_dir / "heterogeneity_delta_grouped_by_model_data.csv", index=False)

    order = combined["dataset_name"].tolist()
    x = np.arange(len(order))
    width = 0.38
    mini_color = "#2563eb"
    nano_color = "#0f766e"

    all_deltas = pd.concat([combined["delta_score_mini"], combined["delta_score_nano"]], ignore_index=True).to_numpy()
    ymin = float(np.nanmin(all_deltas))
    ymax = float(np.nanmax(all_deltas))
    pad = 0.08 * (ymax - ymin if ymax > ymin else 1.0)

    fig, ax = plt.subplots(figsize=(14.5, 6.8))
    ax.bar(x - width / 2, combined["delta_score_mini"], width=width, color=mini_color, alpha=0.9, label="gpt-5-mini")
    ax.bar(x + width / 2, combined["delta_score_nano"], width=width, color=nano_color, alpha=0.9, label="gpt-5-nano")
    ax.axhline(0, color="#444444", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Δ Mean SimBench Score\n(persona_summary − baseline)", fontsize=11)
    ax.grid(axis="y", color="#d0d7de", alpha=0.5, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.text(
        0.01,
        0.98,
        f"Mean Δ: mini = {mini_summary['delta_mean_simbench_score']:+.2f}, "
        f"nano = {nano_summary['delta_mean_simbench_score']:+.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(output_dir / "heterogeneity_delta_simbench_score__grouped_mini_vs_nano.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "heterogeneity_delta_simbench_score__grouped_mini_vs_nano.pdf", bbox_inches="tight")
    plt.close(fig)
    return combined


def _plot_delta_score_significance_side_by_side(output_dir: Path) -> pd.DataFrame:
    mini_sig = _load_significance_table("gpt_5_mini").assign(model="gpt-5-mini")
    nano_sig = _load_significance_table("gpt_5_nano").assign(model="gpt-5-nano")
    combined = pd.concat([mini_sig, nano_sig], ignore_index=True)
    combined.to_csv(output_dir / "heterogeneity_delta_significance_side_by_side_data.csv", index=False)

    colors = {
        "persona_summary_better_sig": "#2f855a",
        "baseline_better_sig": "#c05656",
        "not_significant": "#9aa1a9",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 9.5), sharex=True)
    legend_handles = None
    mini_order = (
        mini_sig.sort_values("delta_persona_minus_baseline", ascending=False)["dataset_name"].astype(str).tolist()
    )

    for ax, model_label in zip(axes, ["gpt-5-mini", "gpt-5-nano"]):
        df = combined[combined["model"] == model_label].copy()
        df["dataset_name"] = pd.Categorical(df["dataset_name"], categories=mini_order, ordered=True)
        df = df.sort_values("dataset_name").reset_index(drop=True)
        df["color_key"] = np.where(
            df["significant_bh_0_05"].fillna(False).astype(bool)
            & (df["direction"] == "persona_summary_better"),
            "persona_summary_better_sig",
            np.where(
                df["significant_bh_0_05"].fillna(False).astype(bool)
                & (df["direction"] == "baseline_better"),
                "baseline_better_sig",
                "not_significant",
            ),
        )

        y = np.arange(len(df))
        bars = ax.barh(
            y,
            df["delta_persona_minus_baseline"],
            color=[colors[key] for key in df["color_key"]],
            alpha=0.9,
        )
        ax.axvline(0, color="#333333", linewidth=1.0, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(df["dataset_name"], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(model_label, fontsize=13, pad=10)
        ax.grid(axis="x", color="#d0d7de", alpha=0.5, linewidth=0.8)
        ax.set_axisbelow(True)

        for idx, row in df.iterrows():
            value = float(row["delta_persona_minus_baseline"])
            label = f"{value:+.1f}"
            if bool(row["significant_bh_0_05"]):
                label += " *"
            if value >= 40:
                ax.text(value - 1.5, idx, label, va="center", ha="right", fontsize=9, color="white")
            elif value <= -40:
                ax.text(value + 1.5, idx, label, va="center", ha="left", fontsize=9, color="white")
            elif value >= 0:
                ax.text(value + 0.8, idx, label, va="center", ha="left", fontsize=9)
            else:
                ax.text(value - 0.8, idx, label, va="center", ha="right", fontsize=9)

        if legend_handles is None:
            legend_handles = [
                plt.Rectangle((0, 0), 1, 1, color=colors["persona_summary_better_sig"], alpha=0.9),
                plt.Rectangle((0, 0), 1, 1, color=colors["baseline_better_sig"], alpha=0.9),
                plt.Rectangle((0, 0), 1, 1, color=colors["not_significant"], alpha=0.9),
            ]

    axes[0].set_ylabel("Task", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Δ SimBench score (persona_summary - baseline)", fontsize=10)

    all_delta = combined["delta_persona_minus_baseline"].astype(float)
    xmin = float(all_delta.min()) - 4.0
    xmax = float(all_delta.max()) + 4.0
    for ax in axes:
        ax.set_xlim(xmin, xmax)

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            ["Persona summary better", "Baseline better", "Not significant"],
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_dir / "heterogeneity_delta_significance_side_by_side__mini_vs_nano.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "heterogeneity_delta_significance_side_by_side__mini_vs_nano.pdf", bbox_inches="tight")
    plt.close(fig)
    return combined


def _plot_predictor_order_vs_actual_delta(
    mini_df: pd.DataFrame,
    nano_df: pd.DataFrame,
    predictor_order: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    helpfulness = {dataset: len(predictor_order) - idx for idx, dataset in enumerate(predictor_order)}
    combined = pd.concat(
        [
            mini_df.assign(model="gpt-5-mini"),
            nano_df.assign(model="gpt-5-nano"),
        ],
        ignore_index=True,
    )
    combined["predicted_helpfulness"] = combined["dataset_name"].map(helpfulness)
    combined["predictor_order_index"] = combined["dataset_name"].map({dataset: idx for idx, dataset in enumerate(predictor_order)})
    combined["dataset_name"] = pd.Categorical(combined["dataset_name"], categories=predictor_order, ordered=True)
    combined = combined.sort_values(["model", "dataset_name"]).reset_index(drop=True)
    combined.to_csv(output_dir / "predictor_vs_actual_delta_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), sharey=False)
    bar_positive = "#2563eb"
    bar_negative = "#dc2626"
    x = np.arange(len(predictor_order))

    for ax, model_label in zip(axes, ["gpt-5-mini", "gpt-5-nano"]):
        df = combined[combined["model"] == model_label].set_index("dataset_name").loc[predictor_order].reset_index()
        rho, pvalue = stats.spearmanr(df["predicted_helpfulness"], df["delta_score"])
        colors = [bar_positive if value >= 0 else bar_negative for value in df["delta_score"]]
        ax.bar(x, df["delta_score"], color=colors, alpha=0.9)
        ax.axhline(0, color="#444444", linewidth=1.0, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(predictor_order, rotation=55, ha="right", fontsize=8)
        ax.set_title(model_label, fontsize=13, pad=10)
        ax.set_xlabel("gpt-5.1 predicted rank: most helpful → most harmful", fontsize=10)
        ax.grid(axis="y", color="#d0d7de", alpha=0.5, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.text(
            0.01,
            0.98,
            f"Spearman ρ = {rho:+.2f}\n"
            f"p = {pvalue:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.9},
        )

    axes[0].set_ylabel("Actual Δ SimBench Score\n(persona_summary − baseline)", fontsize=11)
    fig.tight_layout()
    fig.savefig(
        output_dir / "prediction_rank_vs_actual_delta__gpt_5_1_predictor__mini_vs_nano.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "prediction_rank_vs_actual_delta__gpt_5_1_predictor__mini_vs_nano.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    return combined


def _plot_predictor_rank_vs_actual_rank_scatter(
    mini_df: pd.DataFrame,
    nano_df: pd.DataFrame,
    predictor_order: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    predicted_rank = {dataset: idx + 1 for idx, dataset in enumerate(predictor_order)}
    combined = pd.concat(
        [
            mini_df.assign(model="gpt-5-mini"),
            nano_df.assign(model="gpt-5-nano"),
        ],
        ignore_index=True,
    )
    combined["predicted_rank"] = combined["dataset_name"].map(predicted_rank)
    combined["actual_rank"] = combined.groupby("model")["delta_score"].rank(method="average", ascending=False)
    combined = combined.sort_values(["model", "predicted_rank"]).reset_index(drop=True)
    combined.to_csv(output_dir / "predictor_rank_vs_actual_rank_scatter_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.6), sharex=True, sharey=True)
    point_color = "#2563eb"
    fit_color = "#b42318"
    rank_min = 0.5
    rank_max = len(predictor_order) + 0.5
    xx = np.linspace(1, len(predictor_order), 100)

    for ax, model_label in zip(axes, ["gpt-5-mini", "gpt-5-nano"]):
        df = combined[combined["model"] == model_label].copy()
        rho, pvalue = stats.spearmanr(df["predicted_rank"], df["actual_rank"])
        fit = np.polyfit(df["actual_rank"], df["predicted_rank"], deg=1)
        yy = fit[0] * xx + fit[1]

        ax.scatter(df["actual_rank"], df["predicted_rank"], color=point_color, s=44, alpha=0.9)
        ax.plot(xx, yy, color=fit_color, linewidth=1.8, alpha=0.9)
        ax.plot([1, len(predictor_order)], [1, len(predictor_order)], color="#6e7781", linewidth=1.2, linestyle="--", alpha=0.8)

        for row in df.itertuples(index=False):
            ax.annotate(
                row.dataset_name,
                (row.actual_rank, row.predicted_rank),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7.5,
                alpha=0.9,
            )

        ax.set_title(model_label, fontsize=13, pad=10)
        ax.set_xlim(rank_min, rank_max)
        ax.set_ylim(rank_min, rank_max)
        ax.set_xticks(np.arange(1, len(predictor_order) + 1, 1))
        ax.set_yticks(np.arange(1, len(predictor_order) + 1, 1))
        ax.grid(color="#d0d7de", alpha=0.5, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.text(
            0.02,
            0.98,
            f"Spearman ρ = {rho:+.2f}\n"
            f"p = {pvalue:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.9},
        )

    axes[0].set_ylabel("Predicted rank\n(1 = most helpful)", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Actual rank of Δ SimBench score\n(1 = most helpful)", fontsize=10)

    fig.tight_layout()
    fig.savefig(
        output_dir / "prediction_rank_vs_actual_rank_scatter__gpt_5_1_predictor__mini_vs_nano.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "prediction_rank_vs_actual_rank_scatter__gpt_5_1_predictor__mini_vs_nano.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    return combined


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mini_df, mini_summary = _load_mini_task_scores()
    nano_df, nano_summary = _load_nano_task_scores()
    predictor_order = _load_predictor_order()

    raw_df = _plot_raw_score_heterogeneity(mini_df, nano_df, mini_summary, nano_summary, OUTPUT_DIR)
    delta_df = _plot_delta_score_heterogeneity(mini_df, nano_df, mini_summary, nano_summary, OUTPUT_DIR)
    grouped_df = _plot_delta_score_grouped_by_model(mini_df, nano_df, mini_summary, nano_summary, OUTPUT_DIR)
    sig_side_df = _plot_delta_score_significance_side_by_side(OUTPUT_DIR)
    baseline_heatmap_df = _plot_baseline_heatmap(OUTPUT_DIR)
    predictor_df = _plot_predictor_order_vs_actual_delta(mini_df, nano_df, predictor_order, OUTPUT_DIR)
    predictor_rank_df = _plot_predictor_rank_vs_actual_rank_scatter(mini_df, nano_df, predictor_order, OUTPUT_DIR)

    summary = {
        "heterogeneity_plot": {
            "mini_delta_mean_simbench_score": mini_summary["delta_mean_simbench_score"],
            "nano_delta_mean_simbench_score": nano_summary["delta_mean_simbench_score"],
            "delta_plot_rows": int(len(delta_df)),
            "grouped_plot_rows": int(len(grouped_df)),
            "significance_side_by_side_rows": int(len(sig_side_df)),
            "baseline_heatmap_columns": int(len(baseline_heatmap_df)),
        },
        "prediction_plot": {},
    }
    for model_label, df in predictor_df.groupby("model"):
        rho, pvalue = stats.spearmanr(df["predicted_helpfulness"], df["delta_score"])
        summary["prediction_plot"][model_label] = {
            "spearman_rho_predicted_helpfulness_vs_actual_delta": float(rho),
            "p_value": float(pvalue),
        }
    for model_label, df in predictor_rank_df.groupby("model"):
        rho, pvalue = stats.spearmanr(df["predicted_rank"], df["actual_rank"])
        summary["prediction_plot"][f"{model_label}_rank_scatter"] = {
            "spearman_rho_predicted_rank_vs_actual_rank": float(rho),
            "p_value": float(pvalue),
        }

    (OUTPUT_DIR / "figure_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
