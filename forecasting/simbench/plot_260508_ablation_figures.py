from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
ABLATION_DIR = RESULTS_DIR / "simbenchpop__persona_summary_ablation_compare__gpt_5_nano__us_only"
OUTPUT_DIR = RESULTS_DIR / "260508"

VARIANT_ORDER = [
    "full_card",
    "background_only",
    "direct_social_only",
    "self_report_social_only",
    "non_social_econ_only",
    "cognitive_only",
    "misc_heuristics_pricing_text_only",
]

VARIANT_LABELS = {
    "full_card": "Full profile",
    "background_only": "Background",
    "direct_social_only": "Direct social",
    "self_report_social_only": "Self-report social",
    "non_social_econ_only": "Non-social econ",
    "cognitive_only": "Cognitive",
    "misc_heuristics_pricing_text_only": "Misc/pricing/text",
}

VARIANT_COLORS = {
    "full_card": "#1f77b4",
    "background_only": "#2ca02c",
    "direct_social_only": "#9467bd",
    "self_report_social_only": "#e377c2",
    "non_social_econ_only": "#ff7f0e",
    "cognitive_only": "#8c564b",
    "misc_heuristics_pricing_text_only": "#17becf",
}


def _load_overall_delta() -> pd.DataFrame:
    path = ABLATION_DIR / "pairwise_vs_baseline.csv"
    df = pd.read_csv(path)
    df = df[df["scope"] == "common_rows"].copy()
    df = df[df["label"].isin(VARIANT_ORDER)].copy()
    df["variant_label"] = df["label"].map(VARIANT_LABELS)
    df = df.sort_values("delta_mean_simbench_score", ascending=False).reset_index(drop=True)
    return df[
        [
            "label",
            "variant_label",
            "row_count",
            "delta_mean_simbench_score",
            "pvalue_paired_t_simbench_score",
            "delta_mean_tvd",
            "delta_mean_jsd",
            "delta_modal_match_rate",
        ]
    ]


def _load_task_delta() -> pd.DataFrame:
    path = ABLATION_DIR / "dataset_summary_common_rows.csv"
    df = pd.read_csv(path)
    df = df[df["label"].isin(["baseline", *VARIANT_ORDER])].copy()

    baseline = (
        df[df["label"] == "baseline"][["dataset_name", "mean_simbench_score"]]
        .rename(columns={"mean_simbench_score": "baseline_mean_simbench_score"})
        .reset_index(drop=True)
    )
    variants = df[df["label"] != "baseline"].copy()
    merged = variants.merge(baseline, on="dataset_name", how="left")
    merged["delta_mean_simbench_score"] = (
        merged["mean_simbench_score"] - merged["baseline_mean_simbench_score"]
    )
    merged["variant_label"] = merged["label"].map(VARIANT_LABELS)
    merged["variant_order"] = merged["label"].map({label: idx for idx, label in enumerate(VARIANT_ORDER)})
    merged = merged.sort_values(["dataset_name", "variant_order"]).reset_index(drop=True)
    return merged[
        [
            "dataset_name",
            "label",
            "variant_label",
            "variant_order",
            "row_count",
            "baseline_mean_simbench_score",
            "mean_simbench_score",
            "delta_mean_simbench_score",
        ]
    ]


def _plot_overall_ablation_heterogeneity(output_dir: Path) -> pd.DataFrame:
    df = _load_overall_delta()
    df.to_csv(output_dir / "ablation_variant_delta_overall__gpt_5_nano.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    y = np.arange(len(df))
    colors = ["#2f855a" if value >= 0 else "#c05656" for value in df["delta_mean_simbench_score"]]
    bars = ax.barh(y, df["delta_mean_simbench_score"], color=colors, alpha=0.92)
    ax.axvline(0, color="#333333", linewidth=1.0, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["variant_label"], fontsize=11)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#d0d7de", alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Δ SimBench score (variant - baseline)", fontsize=11)

    max_abs = float(np.abs(df["delta_mean_simbench_score"]).max())
    xpad = max(0.8, 0.08 * max_abs)
    ax.set_xlim(float(df["delta_mean_simbench_score"].min()) - xpad, float(df["delta_mean_simbench_score"].max()) + xpad)

    for idx, row in df.iterrows():
        value = float(row["delta_mean_simbench_score"])
        label = f"{value:+.2f}"
        if value >= 0:
            ax.text(value + 0.18, idx, label, va="center", ha="left", fontsize=10)
        else:
            ax.text(value - 0.18, idx, label, va="center", ha="right", fontsize=10)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "ablation_variant_delta_overall__gpt_5_nano.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "ablation_variant_delta_overall__gpt_5_nano.pdf", bbox_inches="tight")
    plt.close(fig)
    return df


def _plot_task_grouped_ablation_delta(output_dir: Path) -> pd.DataFrame:
    df = _load_task_delta()
    full_card_order = (
        df[df["label"] == "full_card"]
        .sort_values("delta_mean_simbench_score", ascending=False)["dataset_name"]
        .astype(str)
        .tolist()
    )
    df["dataset_name"] = pd.Categorical(df["dataset_name"], categories=full_card_order, ordered=True)
    df = df.sort_values(["dataset_name", "variant_order"]).reset_index(drop=True)
    df.to_csv(output_dir / "ablation_variant_delta_by_task__gpt_5_nano.csv", index=False)

    tasks = full_card_order
    x = np.arange(len(tasks))
    width = 0.11
    offsets = np.linspace(-3 * width, 3 * width, len(VARIANT_ORDER))

    fig, ax = plt.subplots(figsize=(19, 7.8))
    for offset, variant in zip(offsets, VARIANT_ORDER):
        subset = df[df["label"] == variant].set_index("dataset_name").loc[tasks].reset_index()
        ax.bar(
            x + offset,
            subset["delta_mean_simbench_score"],
            width=width,
            color=VARIANT_COLORS[variant],
            alpha=0.9,
            label=VARIANT_LABELS[variant],
        )

    ax.axhline(0, color="#333333", linewidth=1.0, alpha=0.8)
    ax.grid(axis="y", color="#d0d7de", alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("Δ SimBench score (variant - baseline)", fontsize=11)
    ax.set_xlabel("Task", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=10)

    ymax = float(df["delta_mean_simbench_score"].max())
    ymin = float(df["delta_mean_simbench_score"].min())
    ypad = max(1.5, 0.1 * max(abs(ymin), abs(ymax)))
    ax.set_ylim(ymin - ypad, ymax + ypad)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_variant_delta_by_task__gpt_5_nano.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "ablation_variant_delta_by_task__gpt_5_nano.pdf", bbox_inches="tight")
    plt.close(fig)
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_overall_ablation_heterogeneity(OUTPUT_DIR)
    _plot_task_grouped_ablation_delta(OUTPUT_DIR)


if __name__ == "__main__":
    main()
