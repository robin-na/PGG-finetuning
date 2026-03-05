#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_BINARY_FACTORS: Tuple[str, ...] = (
    "CONFIG_chat",
    "CONFIG_allOrNothing",
    "CONFIG_defaultContribProp",
    "CONFIG_rewardExists",
    "CONFIG_showNRounds",
    "CONFIG_showPunishmentId",
    "CONFIG_showOtherSummaries",
    "CONFIG_punishmentExists",
)

DEFAULT_MEDIAN_FACTORS: Tuple[str, ...] = (
    "CONFIG_playerCount",
    "CONFIG_numRounds",
    "CONFIG_MPCR",
)


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if np.issubdtype(series.dtype, np.number):
        return series.fillna(0).astype(float) != 0.0
    true_values = {"1", "true", "t", "yes", "y"}
    return (
        series.astype(str).str.strip().str.lower().map(lambda x: x in true_values).fillna(False)
    )


def signed(value: float, eps: float = 1e-12) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def split_csv_arg(value: str | None) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def train_linear_config_baseline(
    learn_csv: Path,
    val_csv: Path,
    eval_game_ids: Sequence[str],
    target_col: str,
) -> pd.DataFrame:
    learn = pd.read_csv(learn_csv).drop_duplicates(subset=["gameId"], keep="first")
    val = pd.read_csv(val_csv).drop_duplicates(subset=["gameId"], keep="first")
    val = val[val["gameId"].astype(str).isin(set(map(str, eval_game_ids)))].copy()

    config_cols = [c for c in learn.columns if c.startswith("CONFIG_") and c in val.columns]
    if not config_cols:
        raise ValueError("No shared CONFIG_* feature columns found between learn and val CSVs.")
    if target_col not in learn.columns or target_col not in val.columns:
        raise ValueError(f"Target column not found in both learn/val CSVs: {target_col}")

    x_train = learn[config_cols].copy()
    y_train = pd.to_numeric(learn[target_col], errors="coerce")
    x_test = val[config_cols].copy()
    y_test = pd.to_numeric(val[target_col], errors="coerce")

    train_mask = y_train.notna()
    test_mask = y_test.notna()
    x_train = x_train.loc[train_mask].copy()
    y_train = y_train.loc[train_mask].copy()
    x_test = x_test.loc[test_mask].copy()
    y_test = y_test.loc[test_mask].copy()
    val = val.loc[test_mask].copy()

    num_cols = [c for c in x_train.columns if pd.api.types.is_numeric_dtype(x_train[c])]
    cat_cols = [c for c in x_train.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    model = Pipeline([("pre", pre), ("lr", LinearRegression())])
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return pd.DataFrame(
        {
            "gameId": val["gameId"].astype(str).values,
            "human_normalized_efficiency": y_test.values,
            "pred_linear_config": y_pred,
            "abs_error_linear_config": np.abs(y_test.values - y_pred),
            "sq_error_linear_config": (y_test.values - y_pred) ** 2,
        }
    )


def compute_directional_rows(
    table: pd.DataFrame,
    binary_factors: Sequence[str],
    median_factors: Sequence[str],
    model_cols: Dict[str, str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    all_factors = list(binary_factors) + list(median_factors)
    for factor in all_factors:
        if factor not in table.columns:
            continue

        needed = ["gameId", factor, "human_normalized_efficiency"] + list(model_cols.values())
        sub = table[needed].dropna(subset=["human_normalized_efficiency"]).copy()
        if sub.empty:
            continue

        mode = "binary" if factor in set(binary_factors) else "median"
        if mode == "binary":
            mask_high = parse_bool_series(sub[factor])
            threshold = np.nan
        else:
            factor_num = pd.to_numeric(sub[factor], errors="coerce")
            med = float(factor_num.median())
            sub = sub[(factor_num > med) | (factor_num < med)].copy()
            factor_num = pd.to_numeric(sub[factor], errors="coerce")
            mask_high = factor_num > med
            threshold = med

        high = sub[mask_high]
        low = sub[~mask_high]
        if high.empty or low.empty:
            continue

        human_delta = float(
            high["human_normalized_efficiency"].mean()
            - low["human_normalized_efficiency"].mean()
        )
        human_sign = signed(human_delta)

        for model_name, model_col in model_cols.items():
            model_delta = float(high[model_col].mean() - low[model_col].mean())
            model_sign = signed(model_delta)
            rows.append(
                {
                    "factor": factor,
                    "mode": mode,
                    "threshold": threshold,
                    "n_total": int(len(sub)),
                    "n_high": int(len(high)),
                    "n_low": int(len(low)),
                    "human_delta": human_delta,
                    "human_sign": human_sign,
                    "model": model_name,
                    "model_delta": model_delta,
                    "model_sign": model_sign,
                    "sign_match": bool(human_sign == model_sign),
                    "sign_match_nonzero_human": bool(
                        human_sign != 0 and human_sign == model_sign
                    ),
                    "abs_delta_error": float(abs(model_delta - human_delta)),
                }
            )

    return pd.DataFrame(rows)


def summarize_directional(rows_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        rows_df.groupby("model", as_index=False)
        .agg(
            n_factors=("factor", "count"),
            sign_match_rate_all=("sign_match", "mean"),
            sign_match_rate_nonzero_human=("sign_match_nonzero_human", "mean"),
            mean_abs_delta_error=("abs_delta_error", "mean"),
        )
        .sort_values("sign_match_rate_all", ascending=False)
    )

    summary_by_mode = (
        rows_df.groupby(["model", "mode"], as_index=False)
        .agg(
            n_factors=("factor", "count"),
            sign_match_rate=("sign_match", "mean"),
            mean_abs_delta_error=("abs_delta_error", "mean"),
        )
        .sort_values(["mode", "sign_match_rate"], ascending=[True, False])
    )
    return summary, summary_by_mode


def plot_sign_match_rates(summary_df: pd.DataFrame, out_path: Path, dpi: int = 160) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(summary_df))
    ax.bar(x, summary_df["sign_match_rate_all"], color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model"], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Sign-match Rate")
    ax.set_title("CONFIG Directional Effect Sign Match (vs Human)")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(summary_df["sign_match_rate_all"]):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_factor_deltas(rows_df: pd.DataFrame, out_path: Path, dpi: int = 160) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = rows_df.pivot_table(
        index="factor", columns="model", values="model_delta", aggfunc="first"
    )
    human = rows_df.groupby("factor", as_index=True)["human_delta"].first()
    factors = list(pivot.index)
    y = np.arange(len(factors))

    fig_h = max(4.0, 0.45 * len(factors))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axvline(0.0, color="black", linewidth=1.0)

    width = 0.16
    model_order = [c for c in ["linear_config", "oracle_archetype", "retrieved_archetype", "no_archetype"] if c in pivot.columns]
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(model_order) + 1)

    ax.barh(y + offsets[0], human.reindex(factors).values, height=width, label="human", color="#264653")
    palette = ["#2a9d8f", "#e76f51", "#f4a261", "#457b9d"]
    for idx, model in enumerate(model_order, start=1):
        ax.barh(
            y + offsets[idx],
            pivot[model].reindex(factors).values,
            height=width,
            label=model,
            color=palette[(idx - 1) % len(palette)],
        )

    ax.set_yticks(y)
    ax.set_yticklabels(factors)
    ax.set_xlabel("Delta (high/True - low/False) in normalized efficiency")
    ax.set_title("Directional CONFIG Effect Deltas")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare directional CONFIG effects on normalized efficiency across "
            "macro variants plus a CONFIG-only linear baseline."
        )
    )
    parser.add_argument("--analysis_dir", type=Path, required=True)
    parser.add_argument(
        "--learn_analysis_csv",
        type=Path,
        default=Path("benchmark/data/processed_data/df_analysis_learn.csv"),
    )
    parser.add_argument(
        "--val_analysis_csv",
        type=Path,
        default=Path("benchmark/data/processed_data/df_analysis_val.csv"),
    )
    parser.add_argument("--target_col", type=str, default="itt_relative_efficiency")
    parser.add_argument("--no_label", type=str, default="no archetype")
    parser.add_argument("--retrieved_label", type=str, default="retrieved archetype")
    parser.add_argument("--oracle_label", type=str, default="oracle archetype")
    parser.add_argument("--binary_factors", type=str, default=",".join(DEFAULT_BINARY_FACTORS))
    parser.add_argument("--median_factors", type=str, default=",".join(DEFAULT_MEDIAN_FACTORS))
    parser.add_argument("--output_prefix", type=str, default="config_directional_effects_four_models")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    learn_csv = (PROJECT_ROOT / args.learn_analysis_csv).resolve() if not args.learn_analysis_csv.is_absolute() else args.learn_analysis_csv.resolve()
    val_csv = (PROJECT_ROOT / args.val_analysis_csv).resolve() if not args.val_analysis_csv.is_absolute() else args.val_analysis_csv.resolve()

    game_level_path = analysis_dir / "game_level_metrics.csv"
    if not game_level_path.exists():
        raise FileNotFoundError(f"Missing game-level metrics: {game_level_path}")
    if not learn_csv.exists():
        raise FileNotFoundError(f"Missing learning analysis CSV: {learn_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Missing validation analysis CSV: {val_csv}")

    macro = pd.read_csv(game_level_path)
    labels_needed = [args.no_label, args.retrieved_label, args.oracle_label]
    missing_labels = [x for x in labels_needed if x not in set(macro["label"].astype(str))]
    if missing_labels:
        raise ValueError(f"Missing labels in game_level_metrics.csv: {missing_labels}")

    shared_game_ids: set[str] | None = None
    for label in labels_needed:
        ids = set(macro.loc[macro["label"] == label, "gameId"].astype(str).tolist())
        shared_game_ids = ids if shared_game_ids is None else shared_game_ids & ids
    shared_ids = sorted(shared_game_ids or set())
    if not shared_ids:
        raise ValueError("No shared game IDs across the three macro variants.")

    # Base human/config rows (one row per game).
    factor_candidates = split_csv_arg(args.binary_factors) + split_csv_arg(args.median_factors)
    available_factor_cols = [c for c in factor_candidates if c in macro.columns]
    base = (
        macro.loc[macro["label"] == args.no_label, ["gameId", "human_normalized_efficiency"] + available_factor_cols]
        .copy()
    )
    base["gameId"] = base["gameId"].astype(str)
    base = base[base["gameId"].isin(shared_ids)].drop_duplicates(subset=["gameId"], keep="first")

    # Macro model predictions on shared IDs.
    pred_no = (
        macro.loc[macro["label"] == args.no_label, ["gameId", "sim_normalized_efficiency"]]
        .rename(columns={"sim_normalized_efficiency": "pred_no_archetype"})
        .copy()
    )
    pred_retrieved = (
        macro.loc[macro["label"] == args.retrieved_label, ["gameId", "sim_normalized_efficiency"]]
        .rename(columns={"sim_normalized_efficiency": "pred_retrieved_archetype"})
        .copy()
    )
    pred_oracle = (
        macro.loc[macro["label"] == args.oracle_label, ["gameId", "sim_normalized_efficiency"]]
        .rename(columns={"sim_normalized_efficiency": "pred_oracle_archetype"})
        .copy()
    )
    for df in (pred_no, pred_retrieved, pred_oracle):
        df["gameId"] = df["gameId"].astype(str)
        df.drop_duplicates(subset=["gameId"], keep="first", inplace=True)

    # Linear CONFIG-only baseline on shared IDs.
    linear_pred = train_linear_config_baseline(
        learn_csv=learn_csv,
        val_csv=val_csv,
        eval_game_ids=shared_ids,
        target_col=args.target_col,
    )

    merged = (
        base.merge(pred_no, on="gameId", how="inner")
        .merge(pred_retrieved, on="gameId", how="inner")
        .merge(pred_oracle, on="gameId", how="inner")
        .merge(linear_pred[["gameId", "pred_linear_config"]], on="gameId", how="inner")
    )
    if merged.empty:
        raise ValueError("Merged directional-analysis table is empty.")

    model_cols = {
        "no_archetype": "pred_no_archetype",
        "retrieved_archetype": "pred_retrieved_archetype",
        "oracle_archetype": "pred_oracle_archetype",
        "linear_config": "pred_linear_config",
    }
    binary_factors = [f for f in split_csv_arg(args.binary_factors) if f in merged.columns]
    median_factors = [f for f in split_csv_arg(args.median_factors) if f in merged.columns]

    rows_df = compute_directional_rows(
        table=merged,
        binary_factors=binary_factors,
        median_factors=median_factors,
        model_cols=model_cols,
    )
    if rows_df.empty:
        raise ValueError("No directional rows computed. Check factor columns and data availability.")

    summary_df, summary_by_mode_df = summarize_directional(rows_df)
    wide_df = (
        rows_df.pivot_table(
            index=[
                "factor",
                "mode",
                "threshold",
                "n_total",
                "n_high",
                "n_low",
                "human_delta",
                "human_sign",
            ],
            columns="model",
            values=["model_delta", "model_sign", "sign_match", "abs_delta_error"],
            aggfunc="first",
        )
        .sort_index()
    )
    wide_df.columns = [f"{a}__{b}" for a, b in wide_df.columns]
    wide_df = wide_df.reset_index()

    prefix = args.output_prefix
    rows_path = analysis_dir / f"{prefix}_shared23.csv"
    summary_path = analysis_dir / f"{prefix}_summary_shared23.csv"
    summary_mode_path = analysis_dir / f"{prefix}_summary_by_mode_shared23.csv"
    wide_path = analysis_dir / f"{prefix}_wide_shared23.csv"
    linear_pred_path = analysis_dir / "linear_config_baseline_shared23_predictions.csv"

    rows_df.to_csv(rows_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    summary_by_mode_df.to_csv(summary_mode_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    linear_pred.to_csv(linear_pred_path, index=False)

    plot_paths: List[str] = []
    if not args.no_plots:
        figures_dir = analysis_dir / "figures"
        sign_plot = figures_dir / f"{prefix}_sign_match_rate_shared23.png"
        delta_plot = figures_dir / f"{prefix}_delta_by_factor_shared23.png"
        plot_sign_match_rates(summary_df, sign_plot, dpi=args.dpi)
        plot_factor_deltas(rows_df, delta_plot, dpi=args.dpi)
        plot_paths.extend([str(sign_plot), str(delta_plot)])

    manifest = {
        "analysis_dir": str(analysis_dir),
        "learn_analysis_csv": str(learn_csv),
        "val_analysis_csv": str(val_csv),
        "target_col": args.target_col,
        "labels": {
            "no_label": args.no_label,
            "retrieved_label": args.retrieved_label,
            "oracle_label": args.oracle_label,
        },
        "n_shared_games": int(len(merged)),
        "binary_factors": binary_factors,
        "median_factors": median_factors,
        "outputs": {
            "rows_csv": str(rows_path),
            "summary_csv": str(summary_path),
            "summary_by_mode_csv": str(summary_mode_path),
            "wide_csv": str(wide_path),
            "linear_pred_csv": str(linear_pred_path),
            "plots": plot_paths,
        },
    }
    manifest_path = analysis_dir / f"{prefix}_manifest_shared23.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Quick terminal summary
    rmse_linear = math.sqrt(float(linear_pred["sq_error_linear_config"].mean()))
    print(f"Wrote directional outputs under: {analysis_dir}")
    print(f"Shared games: {len(merged)}")
    print(f"Linear CONFIG-only baseline RMSE on shared games: {rmse_linear:.6f}")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
