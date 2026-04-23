from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FORECASTING_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = FORECASTING_ROOT / "results"

SCHEMA_SUFFIX = "mechanism_v2_calc_ui_v1"
PANELS: list[tuple[str, str]] = [
    ("BL · mechanism_v2", f"baseline_{{model}}_{SCHEMA_SUFFIX}"),
    ("Twin · mechanism_v2", f"twin_sampled_unadjusted_seed_0_{{model}}_{SCHEMA_SUFFIX}"),
]
MODEL_STYLES: list[tuple[str, str, str]] = [
    ("gpt_5_1", "gpt-5.1", "#4a4a4a"),
    ("gpt_5_4", "gpt-5.4", "#E45756"),
]
HUMAN_COLOR = "#4C78A8"
CHIP_FAMILY_FILTER = "chip3"


def _load_turn_csv(run_name: str, kind: str) -> pd.DataFrame:
    path = RESULTS_ROOT / f"{run_name}__vs_human_treatments" / f"{kind}_turn_records.csv"
    df = pd.read_csv(path)
    if "chip_family" in df.columns:
        df = df[df["chip_family"].astype(str) == CHIP_FAMILY_FILTER].copy()
    return df


def _panel_data(metric: str) -> tuple[pd.Series, list[tuple[str, list[tuple[str, str, pd.Series, str]]]]]:
    human_series: pd.Series | None = None
    panels: list[tuple[str, list[tuple[str, str, pd.Series, str]]]] = []
    for panel_title, run_template in PANELS:
        model_entries: list[tuple[str, str, pd.Series, str]] = []
        for model_slug, model_label, color in MODEL_STYLES:
            run_name = run_template.format(model=model_slug)
            gen_df = _load_turn_csv(run_name, "generated")
            values = pd.to_numeric(gen_df[metric], errors="coerce").dropna()
            model_entries.append((model_slug, model_label, values, color))
            if human_series is None:
                human_df = _load_turn_csv(run_name, "human")
                human_series = pd.to_numeric(human_df[metric], errors="coerce").dropna()
        panels.append((panel_title, model_entries))
    assert human_series is not None
    return human_series, panels


def _clip_range(
    human_series: pd.Series,
    panels: list[tuple[str, list[tuple[str, str, pd.Series, str]]]],
    percentile: float,
) -> tuple[float, float]:
    all_values: list[float] = list(human_series.astype(float))
    for _, model_entries in panels:
        for _, _, values, _ in model_entries:
            all_values.extend(values.astype(float).tolist())
    arr = np.asarray(all_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    low = float(np.quantile(arr, 1.0 - percentile / 100.0))
    high = float(np.quantile(arr, percentile / 100.0))
    if low == high:
        high = low + 1.0
    return low, high


def _plot_metric(
    metric: str,
    xlabel: str,
    title: str,
    output_path: Path,
    *,
    bins: int,
    percentile: float,
) -> None:
    human_series, panels = _panel_data(metric)
    low, high = _clip_range(human_series, panels, percentile)
    bin_edges = np.linspace(low, high, bins + 1)
    human_clipped = human_series.clip(lower=low, upper=high)
    human_mean = float(human_series.mean())

    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 4.2), sharex=True, sharey=True)
    if len(panels) == 1:
        axes = np.asarray([axes])

    ymax = 0.0
    for ax, (panel_title, model_entries) in zip(axes, panels):
        counts_human, _, _ = ax.hist(
            human_clipped,
            bins=bin_edges,
            color=HUMAN_COLOR,
            alpha=0.45,
            edgecolor="none",
            label=f"human μ={human_mean:.2f}",
            zorder=2,
        )
        ymax = max(ymax, float(counts_human.max()))
        ax.axvline(human_mean, color=HUMAN_COLOR, linestyle=":", linewidth=1.3, zorder=3)
        for _, model_label, values, color in model_entries:
            model_clipped = values.clip(lower=low, upper=high)
            counts, _, _ = ax.hist(
                model_clipped,
                bins=bin_edges,
                color=color,
                alpha=0.55,
                edgecolor="none",
                label=f"{model_label} μ={float(values.mean()):.2f} (n={len(values)})",
                zorder=3,
            )
            ymax = max(ymax, float(counts.max()))
            ax.axvline(float(values.mean()), color=color, linestyle="--", linewidth=1.2, zorder=4)
        ax.set_title(panel_title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.grid(axis="y", alpha=0.2, linestyle=":")
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

    for ax in axes:
        ax.set_ylim(0, ymax * 1.12)
    axes[0].set_ylabel("turn count", fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Histograms under mechanism_v2 schema: baseline + twin, gpt-5.1 vs gpt-5.4 overlaid against human.")
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--percentile", type=float, default=99.0, help="Upper percentile for clipping (e.g. 99 → clip to [1p,99p])")
    args = parser.parse_args()

    human_n = len(
        _load_turn_csv(
            f"baseline_{MODEL_STYLES[0][0]}_{SCHEMA_SUFFIX}",
            "human",
        )
    )
    shared_subtitle = (
        f"schema=mechanism_v2 (CHIP3 only; clipped to [{100 - args.percentile:.0f}p, {args.percentile:.0f}p])\n"
        f"human CHIP3 n = {human_n}"
    )

    _plot_metric(
        "proposer_net_surplus",
        xlabel="proposer_net_surplus",
        title=f"proposer_net_surplus — A/B: gpt-5.1 vs gpt-5.4 — {shared_subtitle}",
        output_path=RESULTS_ROOT / "hist_mechanism_v2_proposer_net_surplus.png",
        bins=args.bins,
        percentile=args.percentile,
    )
    _plot_metric(
        "trade_ratio",
        xlabel="trade_ratio (sell_quantity / buy_quantity)",
        title=f"trade_ratio — A/B: gpt-5.1 vs gpt-5.4 — {shared_subtitle}",
        output_path=RESULTS_ROOT / "hist_mechanism_v2_trade_ratio.png",
        bins=args.bins,
        percentile=args.percentile,
    )
    print("wrote hist_mechanism_v2_proposer_net_surplus.png and hist_mechanism_v2_trade_ratio.png")


if __name__ == "__main__":
    main()
