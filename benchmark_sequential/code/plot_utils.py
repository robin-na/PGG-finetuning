from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def acquisition_display_name(acquisition: str) -> str:
    if acquisition == "ei":
        return "GP-EI"
    if acquisition == "max_variance":
        return "GP-MaxVar"
    return acquisition


def make_plot(
    random_agg: Optional[pd.DataFrame],
    method_curves: dict[str, pd.DataFrame],
    full_anchor_rmse: float,
    full_anchor_r2: float,
    bo_acquisition: str,
    out_path: Path,
    zoom: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    rmse_values_for_zoom: list[float] = [full_anchor_rmse]
    r2_values_for_zoom: list[float] = [full_anchor_r2]

    if random_agg is not None and not random_agg.empty:
        x = random_agg["n_pairs"].to_numpy()
        axes[0].plot(x, random_agg["rmse_mean"], marker="o", label="Random (mean)")
        axes[0].fill_between(x, random_agg["rmse_p10"], random_agg["rmse_p90"], alpha=0.2)
        axes[1].plot(x, random_agg["r2_mean"], marker="o", label="Random (mean)")
        axes[1].fill_between(x, random_agg["r2_p10"], random_agg["r2_p90"], alpha=0.2)
        rmse_values_for_zoom.extend(random_agg["rmse_p90"].tolist())
        r2_values_for_zoom.extend(random_agg["r2_p10"].tolist())
        r2_values_for_zoom.extend(random_agg["r2_p90"].tolist())

    if "bo" in method_curves:
        bo_df = method_curves["bo"]
        axes[0].plot(
            bo_df["n_pairs"],
            bo_df["rmse"],
            marker="s",
            label=f"Adaptive BO ({acquisition_display_name(bo_acquisition)})",
        )
        axes[1].plot(
            bo_df["n_pairs"],
            bo_df["r2_custom"],
            marker="s",
            label=f"Adaptive BO ({acquisition_display_name(bo_acquisition)})",
        )
        rmse_values_for_zoom.extend(bo_df["rmse"].tolist())
        r2_values_for_zoom.extend(bo_df["r2_custom"].tolist())

    if "llm" in method_curves:
        llm_df = method_curves["llm"]
        axes[0].plot(
            llm_df["n_pairs"],
            llm_df["rmse"],
            marker="^",
            label="Adaptive LLM Rerank (GP-EI shortlist)",
        )
        axes[1].plot(
            llm_df["n_pairs"],
            llm_df["r2_custom"],
            marker="^",
            label="Adaptive LLM Rerank (GP-EI shortlist)",
        )
        rmse_values_for_zoom.extend(llm_df["rmse"].tolist())
        r2_values_for_zoom.extend(llm_df["r2_custom"].tolist())

    axes[0].axhline(
        y=full_anchor_rmse,
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="Full-data anchor (n=150)",
    )
    axes[1].axhline(
        y=full_anchor_r2,
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="Full-data anchor (n=150)",
    )

    axes[0].set_xlabel("Number of paired experiments")
    axes[0].set_ylabel("RMSE (OOS on val)")
    axes[0].set_title("RMSE Learning Curve")
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Number of paired experiments")
    axes[1].set_ylabel("OOS R^2 (custom)")
    axes[1].set_title("R^2 Learning Curve")
    axes[1].grid(alpha=0.3)

    if zoom:
        rmse_upper = float(np.quantile(np.array(rmse_values_for_zoom), 0.95))
        axes[0].set_ylim(bottom=0.0, top=rmse_upper * 1.1)

        r2_low = float(np.quantile(np.array(r2_values_for_zoom), 0.05))
        r2_high = float(np.quantile(np.array(r2_values_for_zoom), 0.95))
        axes[1].set_ylim(bottom=max(-1.0, r2_low), top=min(1.0, r2_high + 0.05))

    for ax in axes:
        ax.legend()

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def collect_seed_plot_data(
    n_values: list[int],
    random_agg: Optional[pd.DataFrame],
    bo_df: Optional[pd.DataFrame],
    llm_df: Optional[pd.DataFrame],
    full_anchor_rmse: float,
    full_anchor_r2: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if random_agg is not None and not random_agg.empty:
        r = random_agg[["n_pairs", "rmse_mean", "r2_mean"]].copy()
        r = r.rename(columns={"rmse_mean": "rmse", "r2_mean": "r2_custom"})
        r["method"] = "random_addition_mean"
        frames.append(r[["method", "n_pairs", "rmse", "r2_custom"]])

    if bo_df is not None and not bo_df.empty:
        b = bo_df[["n_pairs", "rmse", "r2_custom"]].copy()
        b["method"] = "adaptive_bo_gp_ei"
        frames.append(b[["method", "n_pairs", "rmse", "r2_custom"]])

    if llm_df is not None and not llm_df.empty:
        l = llm_df[["n_pairs", "rmse", "r2_custom"]].copy()
        l["method"] = "adaptive_llm_rerank_gp_ei"
        frames.append(l[["method", "n_pairs", "rmse", "r2_custom"]])

    anchor = pd.DataFrame(
        {
            "method": "full_data_anchor",
            "n_pairs": n_values,
            "rmse": [full_anchor_rmse] * len(n_values),
            "r2_custom": [full_anchor_r2] * len(n_values),
        }
    )
    frames.append(anchor)

    return pd.concat(frames, ignore_index=True)


def make_batch_plot(
    agg_df: pd.DataFrame,
    full_anchor_rmse: float,
    full_anchor_r2: float,
    out_path: Path,
    zoom: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    method_labels = {
        "random_addition_mean": "Random (mean over seeds)",
        "adaptive_bo_gp_ei": "Adaptive BO (GP-EI)",
        "adaptive_llm_rerank_gp_ei": "Adaptive LLM Rerank (GP-EI shortlist)",
    }

    rmse_vals = [full_anchor_rmse]
    r2_vals = [full_anchor_r2]

    for method in ["random_addition_mean", "adaptive_bo_gp_ei", "adaptive_llm_rerank_gp_ei"]:
        sub = agg_df[agg_df["method"] == method].sort_values("n_pairs")
        if sub.empty:
            continue
        axes[0].plot(sub["n_pairs"], sub["rmse_mean"], marker="o", label=method_labels[method])
        axes[0].fill_between(sub["n_pairs"], sub["rmse_p10"], sub["rmse_p90"], alpha=0.2)
        axes[1].plot(sub["n_pairs"], sub["r2_mean"], marker="o", label=method_labels[method])
        axes[1].fill_between(sub["n_pairs"], sub["r2_p10"], sub["r2_p90"], alpha=0.2)
        rmse_vals.extend(sub["rmse_p90"].tolist())
        r2_vals.extend(sub["r2_p10"].tolist())
        r2_vals.extend(sub["r2_p90"].tolist())

    axes[0].axhline(full_anchor_rmse, linestyle="--", color="black", linewidth=1.5, label="Full-data anchor")
    axes[1].axhline(full_anchor_r2, linestyle="--", color="black", linewidth=1.5, label="Full-data anchor")

    axes[0].set_xlabel("Number of paired experiments")
    axes[0].set_ylabel("RMSE (OOS on val)")
    axes[0].set_title("RMSE Learning Curve (Batch Aggregated)")
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Number of paired experiments")
    axes[1].set_ylabel("OOS R^2 (custom)")
    axes[1].set_title("R^2 Learning Curve (Batch Aggregated)")
    axes[1].grid(alpha=0.3)

    if zoom:
        rmse_upper = float(np.quantile(np.array(rmse_vals), 0.95))
        axes[0].set_ylim(bottom=0.0, top=rmse_upper * 1.1)
        r2_low = float(np.quantile(np.array(r2_vals), 0.05))
        r2_high = float(np.quantile(np.array(r2_vals), 0.95))
        axes[1].set_ylim(bottom=max(-1.0, r2_low), top=min(1.0, r2_high + 0.05))

    for ax in axes:
        ax.legend()

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
