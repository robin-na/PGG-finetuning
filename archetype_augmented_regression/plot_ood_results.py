from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


METHOD_ORDER = ["config_only", "synthetic", "oracle", "noise_ceiling"]
METHOD_LABEL = {
    "config_only": "CONFIG only",
    "synthetic": "Synthetic style",
    "oracle": "Oracle style",
    "noise_ceiling": "Noise ceiling",
}
METHOD_COLOR = {
    "config_only": "#6b7280",
    "synthetic": "#2563eb",
    "oracle": "#f59e0b",
    "noise_ceiling": "#dc2626",
}
EVAL_ORDER = ["game", "config_treatment"]
EVAL_LABEL = {"game": "Game", "config_treatment": "CONFIG treatment"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot OOD archetype-augmented regression results."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "outputs/archetype_augmented_regression/ood_wave_anchored/runs/run_full_oos_noise_itt_efficiency"
        ),
    )
    parser.add_argument("--model", type=str, default="ridge", choices=["ridge", "linear"])
    parser.add_argument(
        "--eval-granularity",
        type=str,
        default="both",
        choices=["both", "game", "config_treatment"],
        help="Restrict plots to one evaluation granularity or include both.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to reports/archetype_augmented_regression/figures/<run_id>/<model>",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def build_method_table(
    all_df: pd.DataFrame, noise_df: pd.DataFrame, model: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_df = all_df[all_df["model"] == model].copy()
    if model_df.empty:
        raise ValueError(f"No rows for model='{model}'")

    records: list[dict[str, object]] = []
    group_cols = ["factor", "direction", "split", "eval_granularity", "model"]
    for _, sub in model_df.groupby(group_cols, sort=True):
        first = sub.iloc[0]
        records.append(
            {
                "factor": first["factor"],
                "direction": first["direction"],
                "split": first["split"],
                "eval_granularity": first["eval_granularity"],
                "model": first["model"],
                "method": "config_only",
                "r2": float(first["baseline_r2"]),
                "r2_oos_train_mean": float(first["baseline_r2_oos_train_mean"]),
                "rmse": float(first["baseline_rmse"]),
                "mae": float(first["baseline_mae"]),
            }
        )
        for style in ("synthetic", "oracle"):
            row = sub[sub["style_source"] == style]
            if row.empty:
                continue
            row0 = row.iloc[0]
            records.append(
                {
                    "factor": row0["factor"],
                    "direction": row0["direction"],
                    "split": row0["split"],
                    "eval_granularity": row0["eval_granularity"],
                    "model": row0["model"],
                    "method": style,
                    "r2": float(row0["augmented_r2"]),
                    "r2_oos_train_mean": float(row0["augmented_r2_oos_train_mean"]),
                    "rmse": float(row0["augmented_rmse"]),
                    "mae": float(row0["augmented_mae"]),
                }
            )

    method_df = pd.DataFrame(records)

    ceiling_df = noise_df[
        ["factor", "direction", "split", "eval_granularity", "oracle_test_config_mean_r2", "oracle_test_config_mean_r2_oos_train_mean", "oracle_test_config_mean_rmse", "oracle_test_config_mean_mae"]
    ].copy()
    ceiling_df = ceiling_df.rename(
        columns={
            "oracle_test_config_mean_r2": "r2",
            "oracle_test_config_mean_r2_oos_train_mean": "r2_oos_train_mean",
            "oracle_test_config_mean_rmse": "rmse",
            "oracle_test_config_mean_mae": "mae",
        }
    )
    ceiling_df["method"] = "noise_ceiling"
    ceiling_df["model"] = model

    method_plus_ceiling_df = pd.concat([method_df, ceiling_df], ignore_index=True)
    method_plus_ceiling_df["method"] = pd.Categorical(
        method_plus_ceiling_df["method"], categories=METHOD_ORDER, ordered=True
    )

    base_df = method_df[method_df["method"] == "config_only"][
        ["split", "eval_granularity", "r2", "rmse", "mae", "r2_oos_train_mean"]
    ].rename(
        columns={
            "r2": "base_r2",
            "rmse": "base_rmse",
            "mae": "base_mae",
            "r2_oos_train_mean": "base_r2_oos_train_mean",
        }
    )
    delta_df = method_df[method_df["method"].isin(["oracle", "synthetic"])].merge(
        base_df, on=["split", "eval_granularity"], how="left"
    )
    delta_df["delta_r2_vs_config"] = delta_df["r2"] - delta_df["base_r2"]
    delta_df["rmse_drop_vs_config"] = delta_df["base_rmse"] - delta_df["rmse"]
    delta_df["delta_r2_oos_vs_config"] = delta_df["r2_oos_train_mean"] - delta_df["base_r2_oos_train_mean"]

    gap_df = method_df[method_df["method"].isin(["config_only", "oracle", "synthetic"])].merge(
        ceiling_df[["split", "eval_granularity", "r2", "r2_oos_train_mean", "rmse"]],
        on=["split", "eval_granularity"],
        how="left",
        suffixes=("", "_ceiling"),
    )
    gap_df["r2_gap_to_ceiling"] = gap_df["r2_ceiling"] - gap_df["r2"]
    gap_df["r2_oos_gap_to_ceiling"] = gap_df["r2_oos_train_mean_ceiling"] - gap_df["r2_oos_train_mean"]
    gap_df["rmse_gap_to_ceiling"] = gap_df["rmse"] - gap_df["rmse_ceiling"]
    return method_plus_ceiling_df, delta_df, gap_df


def _metric_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = (
        df.groupby(["eval_granularity", "method"], as_index=False, observed=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    out["ci95"] = 1.96 * out["sem"].fillna(0.0)
    return out


def plot_mean_comparison(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int,
    model: str,
    eval_modes: list[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    configs = [("r2", "R2 (higher is better)"), ("rmse", "RMSE (lower is better)")]

    for ax, (metric, ylabel) in zip(axes, configs):
        summary = _metric_summary(df, metric)
        x = np.arange(len(eval_modes))
        width = 0.18
        for i, method in enumerate(METHOD_ORDER):
            sub = summary[summary["method"] == method].set_index("eval_granularity")
            vals = [sub.loc[g, "mean"] if g in sub.index else np.nan for g in eval_modes]
            errs = [sub.loc[g, "ci95"] if g in sub.index else 0.0 for g in eval_modes]
            ax.bar(
                x + (i - 1.5) * width,
                vals,
                width=width,
                yerr=errs,
                capsize=3,
                color=METHOD_COLOR[method],
                alpha=0.9,
                label=METHOD_LABEL[method],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([EVAL_LABEL[g] for g in eval_modes], rotation=0)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(f"{metric.upper()} mean across OOD splits")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"OOD Method Comparison with Noise Ceiling ({model})", y=1.16)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _heatmap(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    title: str,
    cmap: str,
    vcenter: float = 0.0,
) -> None:
    values = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values))
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1e-6
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=vcenter, vmax=vmax)
    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([METHOD_LABEL[m] for m in pivot.columns], rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)


def plot_split_heatmaps(delta_df: pd.DataFrame, output_path: Path, dpi: int, model: str) -> None:
    eval_modes = [m for m in EVAL_ORDER if m in set(delta_df["eval_granularity"])]
    if len(eval_modes) == 0:
        raise ValueError("No eval granularity rows available for split heatmap.")

    fig_w = 7 * len(eval_modes)
    fig, axes = plt.subplots(2, len(eval_modes), figsize=(fig_w, 10), constrained_layout=True)
    if len(eval_modes) == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # shape: (2,1)

    for col, eval_mode in enumerate(eval_modes):
        sub = delta_df[delta_df["eval_granularity"] == eval_mode].copy()
        sub = sub.sort_values(["factor", "direction"])
        p_r2 = sub.pivot(index="split", columns="method", values="delta_r2_vs_config")
        p_rmse = sub.pivot(index="split", columns="method", values="rmse_drop_vs_config")
        p_r2 = p_r2[[c for c in ["oracle", "synthetic"] if c in p_r2.columns]]
        p_rmse = p_rmse[[c for c in ["oracle", "synthetic"] if c in p_rmse.columns]]

        _heatmap(
            axes[0, col],
            p_r2,
            f"{EVAL_LABEL[eval_mode]}: Delta R2 vs CONFIG",
            cmap="RdBu_r",
        )
        _heatmap(
            axes[1, col],
            p_rmse,
            f"{EVAL_LABEL[eval_mode]}: RMSE Drop vs CONFIG",
            cmap="RdBu_r",
        )

    fig.suptitle(f"Split-Level Improvement Heatmaps ({model})", y=1.02)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _boxplot_grouped(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    methods = ["config_only", "synthetic", "oracle"]
    positions: list[float] = []
    data: list[np.ndarray] = []
    colors: list[str] = []
    labels: list[str] = []
    x = 0.0
    eval_modes = [m for m in EVAL_ORDER if m in set(df["eval_granularity"])]
    for eval_mode in eval_modes:
        sub_eval = df[df["eval_granularity"] == eval_mode]
        for method in methods:
            vals = sub_eval[sub_eval["method"] == method][metric].to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            positions.append(x)
            data.append(vals)
            colors.append(METHOD_COLOR[method])
            labels.append(f"{EVAL_LABEL[eval_mode]}\n{METHOD_LABEL[method]}")
            x += 1.0
        x += 0.7
    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.65, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def plot_gap_to_ceiling(gap_df: pd.DataFrame, output_path: Path, dpi: int, model: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    _boxplot_grouped(
        axes[0],
        gap_df,
        "r2_gap_to_ceiling",
        "Ceiling R2 - Model R2 (lower is better)",
        "R2 gap to noise ceiling",
    )
    _boxplot_grouped(
        axes[1],
        gap_df,
        "rmse_gap_to_ceiling",
        "Model RMSE - Ceiling RMSE (lower is better)",
        "RMSE gap to noise ceiling",
    )
    fig.suptitle(f"Gap to Noise Ceiling Across Splits ({model})", y=1.03)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = args.run_root
    run_id = run_root.name
    output_dir = args.output_dir or Path(
        f"reports/archetype_augmented_regression/figures/{run_id}/{args.model}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_df = pd.read_csv(_require(run_root / "all_splits_best_rows.csv"))
    noise_df = pd.read_csv(_require(run_root / "noise_ceiling_by_split.csv"))

    method_plus_ceiling_df, delta_df, gap_df = build_method_table(all_df, noise_df, args.model)

    if args.eval_granularity == "both":
        eval_modes = EVAL_ORDER.copy()
    else:
        eval_modes = [args.eval_granularity]
    method_plus_ceiling_df = method_plus_ceiling_df[
        method_plus_ceiling_df["eval_granularity"].isin(eval_modes)
    ].copy()
    delta_df = delta_df[delta_df["eval_granularity"].isin(eval_modes)].copy()
    gap_df = gap_df[gap_df["eval_granularity"].isin(eval_modes)].copy()

    eval_suffix = args.eval_granularity
    if method_plus_ceiling_df.empty:
        raise ValueError(f"No rows after eval granularity filter: {args.eval_granularity}")

    method_plus_ceiling_df.to_csv(output_dir / f"{args.model}_methods_with_ceiling_by_split.csv", index=False)
    delta_df.to_csv(output_dir / f"{args.model}_delta_vs_config_by_split.csv", index=False)
    gap_df.to_csv(output_dir / f"{args.model}_gap_to_noise_ceiling_by_split.csv", index=False)

    plot_mean_comparison(
        df=method_plus_ceiling_df,
        output_path=output_dir
        / f"{args.model}_{eval_suffix}_ood_method_comparison_with_noise_ceiling.png",
        dpi=args.dpi,
        model=args.model,
        eval_modes=eval_modes,
    )
    plot_split_heatmaps(
        delta_df=delta_df,
        output_path=output_dir / f"{args.model}_{eval_suffix}_ood_split_delta_heatmaps.png",
        dpi=args.dpi,
        model=args.model,
    )
    plot_gap_to_ceiling(
        gap_df=gap_df,
        output_path=output_dir / f"{args.model}_{eval_suffix}_ood_gap_to_noise_ceiling.png",
        dpi=args.dpi,
        model=args.model,
    )

    print(f"Saved plots and plot tables in: {output_dir}")


if __name__ == "__main__":
    main()
