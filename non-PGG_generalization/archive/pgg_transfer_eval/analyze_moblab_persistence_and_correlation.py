#!/usr/bin/env python3
"""Analyze MobLab first-round persistence and cross-game correlation."""

from __future__ import annotations

import argparse
import ast
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOBLAB_DIR = PROJECT_ROOT / "non-PGG_generalization" / "data" / "MobLab"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "pgg_transfer_eval"
    / "output"
    / "moblab_round1_multiround"
)

MEASURE_LABELS = {
    "dictator": "Dictator Offer Share",
    "trust_investor": "Trust Investor Share",
    "trust_banker": "Trust Banker Return Rate",
    "ultimatum_proposer": "Ultimatum Offer Share",
    "ultimatum_responder": "Ultimatum Acceptance Threshold",
    "pg_contribution": "PGG Contribution Share",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def configure_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "#fbf7ef",
            "axes.facecolor": "#fbf7ef",
            "savefig.facecolor": "#fbf7ef",
            "axes.edgecolor": "#1d2a3a",
            "axes.labelcolor": "#1d2a3a",
            "text.color": "#1d2a3a",
            "xtick.color": "#1d2a3a",
            "ytick.color": "#1d2a3a",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "figure.titlesize": 18,
            "figure.titleweight": "bold",
            "grid.color": "#d8cfc2",
            "grid.alpha": 0.35,
            "axes.grid": True,
            "grid.linestyle": "--",
        }
    )


def robust_literal(value: object) -> Optional[object]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text == "None":
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def safe_float(value: object) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def total_components(value: object) -> Tuple[Optional[float], Optional[float]]:
    parsed = robust_literal(value)
    if isinstance(parsed, (list, tuple)):
        first = safe_float(parsed[0]) if len(parsed) >= 1 else None
        second = safe_float(parsed[1]) if len(parsed) >= 2 else None
        return first, second
    scalar = safe_float(parsed if parsed is not None else value)
    return scalar, None


def sessionize(df: pd.DataFrame, role_col: str = "Role") -> pd.DataFrame:
    df = df.reset_index().rename(columns={"index": "_row_order"}).copy()
    df["session_id"] = None
    for (user_id, role), group in df.groupby(["UserID", role_col], sort=False):
        group = group.sort_values("_row_order")
        session_idx = 0
        prev_round = None
        prev_total = None
        for row in group.itertuples():
            current_round = int(row.Round)
            current_total = int(row.totalRound)
            if prev_round is None or current_round <= prev_round or current_total != prev_total:
                session_idx += 1
            df.loc[row.Index, "session_id"] = f"{user_id}::{role}::{session_idx}"
            prev_round = current_round
            prev_total = current_total
    return df


def spearman_pair(left: pd.Series, right: pd.Series) -> Tuple[float, int]:
    merged = pd.concat([left, right], axis=1, join="inner").dropna()
    if len(merged) < 3:
        return float("nan"), len(merged)
    ranked = merged.rank(method="average")
    value = float(ranked.iloc[:, 0].corr(ranked.iloc[:, 1]))
    return value, len(merged)


def build_dictator_rounds() -> pd.DataFrame:
    df = pd.read_csv(MOBLAB_DIR / "dictator.csv")
    df = df[(df["gameType"] == "dictator") & (df["Role"] == "first") & (df["Total"] == 100)].copy()
    df["move_num"] = pd.to_numeric(df["move"], errors="coerce")
    df = df[df["move_num"].between(0, 100, inclusive="both")]
    df = sessionize(df)
    df = df.sort_values(["UserID", "session_id", "Round", "_row_order"])
    df["measure"] = "dictator"
    df["value"] = df["move_num"] / 100.0
    return df[["UserID", "session_id", "Round", "totalRound", "measure", "value"]]


def build_trust_rounds() -> pd.DataFrame:
    df = pd.read_csv(MOBLAB_DIR / "trust_investment.csv")
    df = df[df["gameType"] == "trust_investment"].copy()
    df["move_num"] = pd.to_numeric(df["move"], errors="coerce")
    df["round_tuple"] = df["roundResult"].apply(robust_literal)
    df["total_tuple"] = df["Total"].apply(total_components)
    df["trust_scale_100"] = df["total_tuple"].apply(lambda x: safe_float(x[0]) == 100.0 if isinstance(x, tuple) else False)
    df = df[df["trust_scale_100"]].copy()
    df = sessionize(df)
    df = df.sort_values(["UserID", "session_id", "Round", "_row_order"])

    rows: List[Dict[str, object]] = []
    for row in df.itertuples():
        if row.Role == "first":
            move = safe_float(row.move_num)
            if move is None or not (0 <= move <= 100):
                continue
            rows.append(
                {
                    "UserID": int(row.UserID),
                    "session_id": row.session_id,
                    "Round": int(row.Round),
                    "totalRound": int(row.totalRound),
                    "measure": "trust_investor",
                    "value": move / 100.0,
                }
            )
        elif row.Role == "second":
            pair = row.round_tuple if isinstance(row.round_tuple, (list, tuple)) else None
            if not pair or len(pair) < 2:
                continue
            invest = safe_float(pair[0])
            ret = safe_float(pair[1])
            if invest is None or ret is None or invest <= 0 or invest > 100 or ret < 0 or ret > invest * 3:
                continue
            rows.append(
                {
                    "UserID": int(row.UserID),
                    "session_id": row.session_id,
                    "Round": int(row.Round),
                    "totalRound": int(row.totalRound),
                    "measure": "trust_banker",
                    "value": ret / (3.0 * invest),
                }
            )
    return pd.DataFrame(rows)


def build_ultimatum_rounds() -> pd.DataFrame:
    df = pd.read_csv(MOBLAB_DIR / "ultimatum_strategy.csv")
    df = df[(df["gameType"] == "ultimatum_strategy") & (df["Role"] == "player") & (df["Total"] == 100)].copy()
    df["move_tuple"] = df["move"].apply(robust_literal)
    df = df[df["move_tuple"].notna()].copy()
    df = sessionize(df)
    df = df.sort_values(["UserID", "session_id", "Round", "_row_order"])

    rows: List[Dict[str, object]] = []
    for row in df.itertuples():
        pair = row.move_tuple if isinstance(row.move_tuple, (list, tuple)) else None
        if not pair or len(pair) < 2:
            continue
        propose = safe_float(pair[0])
        accept = safe_float(pair[1])
        if propose is None or accept is None or not (0 <= propose <= 100) or not (0 <= accept <= 100):
            continue
        rows.append(
            {
                "UserID": int(row.UserID),
                "session_id": row.session_id,
                "Round": int(row.Round),
                "totalRound": int(row.totalRound),
                "measure": "ultimatum_proposer",
                "value": propose / 100.0,
            }
        )
        rows.append(
            {
                "UserID": int(row.UserID),
                "session_id": row.session_id,
                "Round": int(row.Round),
                "totalRound": int(row.totalRound),
                "measure": "ultimatum_responder",
                "value": accept / 100.0,
            }
        )
    return pd.DataFrame(rows)


def build_pg_rounds() -> pd.DataFrame:
    df = pd.read_csv(MOBLAB_DIR / "public_goods_linear_water.csv")
    df = df[(df["gameType"] == "public_goods_linear_water") & (df["Role"] == "contributor") & (df["Total"] == 20)].copy()
    df["move_num"] = pd.to_numeric(df["move"], errors="coerce")
    df = df[df["move_num"].between(0, 20, inclusive="both")]
    df = sessionize(df)
    df = df.sort_values(["UserID", "session_id", "Round", "_row_order"])
    df["measure"] = "pg_contribution"
    df["value"] = df["move_num"] / 20.0
    return df[["UserID", "session_id", "Round", "totalRound", "measure", "value"]]


def build_all_rounds() -> pd.DataFrame:
    parts = [
        build_dictator_rounds(),
        build_trust_rounds(),
        build_ultimatum_rounds(),
        build_pg_rounds(),
    ]
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values(["measure", "UserID", "session_id", "Round"]).reset_index(drop=True)
    return combined


def summarize_sessions(rounds_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (measure, user_id, session_id), group in rounds_df.groupby(["measure", "UserID", "session_id"], sort=False):
        values = group.sort_values("Round")["value"].astype(float).to_numpy()
        if values.size == 0:
            continue
        future_mean = float(np.mean(values[1:])) if values.size >= 2 else float("nan")
        rows.append(
            {
                "measure": measure,
                "UserID": int(user_id),
                "session_id": session_id,
                "session_rounds": int(values.size),
                "first_round_value": float(values[0]),
                "multiround_mean": float(np.mean(values)),
                "future_mean": future_mean,
                "last_round_value": float(values[-1]),
                "persistence_abs_error_future_mean": abs(float(values[0]) - future_mean) if values.size >= 2 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_users(session_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = (
        session_df.groupby(["UserID", "measure"])[value_col]
        .mean()
        .unstack("measure")
        .sort_index()
    )
    return pivot


def persistence_table(session_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for measure, group in session_df.groupby("measure", sort=False):
        valid_future = group["future_mean"].notna()
        rows.append(
            {
                "measure": measure,
                "label": MEASURE_LABELS[measure],
                "n_sessions": int(len(group)),
                "n_users": int(group["UserID"].nunique()),
                "median_session_rounds": float(group["session_rounds"].median()),
                "round1_mean": float(group["first_round_value"].mean()),
                "round1_std": float(group["first_round_value"].std(ddof=0)),
                "multiround_mean": float(group["multiround_mean"].mean()),
                "future_mean": float(group.loc[valid_future, "future_mean"].mean()) if valid_future.any() else float("nan"),
                "k1_persistence_mae_future_mean": float(group.loc[valid_future, "persistence_abs_error_future_mean"].mean()) if valid_future.any() else float("nan"),
            }
        )
    table = pd.DataFrame(rows)
    order = list(MEASURE_LABELS.keys())
    table["measure"] = pd.Categorical(table["measure"], categories=order, ordered=True)
    return table.sort_values("measure").reset_index(drop=True)


def correlation_matrix(user_pivot: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    measures = [col for col in MEASURE_LABELS if col in user_pivot.columns]
    corr = pd.DataFrame(np.nan, index=measures, columns=measures)
    overlap = pd.DataFrame(0, index=measures, columns=measures, dtype=int)
    for left in measures:
        for right in measures:
            value, n = spearman_pair(user_pivot[left], user_pivot[right])
            corr.loc[left, right] = value
            overlap.loc[left, right] = n
    return corr, overlap


def plot_distribution_comparison(session_df: pd.DataFrame, persistence_df: pd.DataFrame, out_path: Path) -> None:
    measures = [m for m in MEASURE_LABELS if m in session_df["measure"].unique()]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.8))
    axes = axes.flatten()
    colors = {"round1": "#1f6f8b", "multiround": "#d17b0f"}
    xlims = {
        "trust_banker": (0.0, 1.05),
    }
    for ax, measure in zip(axes, measures):
        group = session_df[session_df["measure"] == measure]
        round1 = group["first_round_value"].dropna().astype(float)
        multi = group["multiround_mean"].dropna().astype(float)
        bins = np.linspace(0.0, xlims.get(measure, (0.0, 1.0))[1], 26)
        ax.hist(round1, bins=bins, alpha=0.55, density=True, color=colors["round1"], label="Round 1")
        ax.hist(multi, bins=bins, alpha=0.45, density=True, color=colors["multiround"], label="Multiround mean")
        row = persistence_df.loc[persistence_df["measure"] == measure].iloc[0]
        ax.set_title(f"{MEASURE_LABELS[measure]}\nK=1 persistence MAE={row['k1_persistence_mae_future_mean']:.3f}")
        ax.set_xlabel("Normalized behavior")
        ax.set_ylabel("Density")
        ax.set_xlim(*xlims.get(measure, (0.0, 1.0)))
        ax.legend(frameon=False, fontsize=9)
    for ax in axes[len(measures):]:
        ax.axis("off")
    fig.suptitle("Round-1 vs Multiround Distribution by Game Measure", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_corr_heatmap(matrix: pd.DataFrame, title: str, out_path: Path, fmt: str = ".2f", vmin: float = -0.3, vmax: float = 0.3) -> None:
    labels = [MEASURE_LABELS[idx] for idx in matrix.index]
    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    cmap = LinearSegmentedColormap.from_list("corrmap", ["#c44536", "#f7d08a", "#fbf7ef", "#7fb3c8", "#1d3557"])
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            label = "" if np.isnan(value) else format(value, fmt)
            ax.text(j, i, label, ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.82, label="Spearman correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_change_table(first_corr: pd.DataFrame, multi_corr: pd.DataFrame, overlap_first: pd.DataFrame, overlap_multi: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    measures = list(first_corr.index)
    for i, left in enumerate(measures):
        for right in measures[i + 1:]:
            first_value = first_corr.loc[left, right]
            multi_value = multi_corr.loc[left, right]
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "left_label": MEASURE_LABELS[left],
                    "right_label": MEASURE_LABELS[right],
                    "first_round_spearman": first_value,
                    "multiround_spearman": multi_value,
                    "delta_spearman": multi_value - first_value if pd.notna(first_value) and pd.notna(multi_value) else float("nan"),
                    "first_overlap_users": int(overlap_first.loc[left, right]),
                    "multiround_overlap_users": int(overlap_multi.loc[left, right]),
                }
            )
    return pd.DataFrame(rows).sort_values("delta_spearman", ascending=False).reset_index(drop=True)


def build_summary_markdown(persistence_df: pd.DataFrame, change_df: pd.DataFrame) -> str:
    lines = [
        "# MobLab Round-1 Persistence and Cross-Game Correlation",
        "",
        "## Setup",
        "",
        "- Measures are normalized within the standard environment for each game: dictator offer share, trust-investor share, trust-banker return rate, ultimatum proposer offer share, ultimatum responder threshold, and ordinary-PGG contribution share.",
        "- Trust is restricted to the fixed primary scale `Total[0] == 100`.",
        "- `k=1 persistence` is summarized as the absolute error between the round-1 action and the session's future mean action.",
        "",
        "## Persistence Table",
        "",
        "| Measure | Sessions | Users | Median rounds | Round-1 mean | Multiround mean | K=1 persistence MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in persistence_df.itertuples():
        lines.append(
            f"| {row.label} | {row.n_sessions} | {row.n_users} | {row.median_session_rounds:.1f} | "
            f"{row.round1_mean:.3f} | {row.multiround_mean:.3f} | {row.k1_persistence_mae_future_mean:.3f} |"
        )
    increased = change_df.dropna(subset=["delta_spearman"]).head(5)
    decreased = change_df.dropna(subset=["delta_spearman"]).sort_values("delta_spearman").head(5)
    lines.extend(
        [
            "",
            "## Largest Increases in Cross-Game Correlation",
            "",
        ]
    )
    for row in increased.itertuples():
        lines.append(
            f"- {row.left_label} vs {row.right_label}: first-round {row.first_round_spearman:.3f}, "
            f"multiround {row.multiround_spearman:.3f}, delta {row.delta_spearman:+.3f}"
        )
    lines.extend(
        [
            "",
            "## Largest Decreases in Cross-Game Correlation",
            "",
        ]
    )
    for row in decreased.itertuples():
        lines.append(
            f"- {row.left_label} vs {row.right_label}: first-round {row.first_round_spearman:.3f}, "
            f"multiround {row.multiround_spearman:.3f}, delta {row.delta_spearman:+.3f}"
        )
    return "\n".join(lines) + "\n"


def write_analysis_bundle(session_df: pd.DataFrame, output_dir: Path, title_note: Optional[str] = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    persistence_df = persistence_table(session_df)
    first_user = summarize_users(session_df, "first_round_value")
    multi_user = summarize_users(session_df, "multiround_mean")
    first_corr, first_overlap = correlation_matrix(first_user)
    multi_corr, multi_overlap = correlation_matrix(multi_user)
    delta_corr = multi_corr - first_corr
    change_df = build_change_table(first_corr, multi_corr, first_overlap, multi_overlap)

    persistence_df.to_csv(output_dir / "persistence_by_measure.csv", index=False)
    first_user.reset_index().to_csv(output_dir / "user_first_round_measures.csv", index=False)
    multi_user.reset_index().to_csv(output_dir / "user_multiround_measures.csv", index=False)
    first_corr.to_csv(output_dir / "corr_first_round_spearman.csv")
    multi_corr.to_csv(output_dir / "corr_multiround_spearman.csv")
    delta_corr.to_csv(output_dir / "corr_multiround_minus_first.csv")
    change_df.to_csv(output_dir / "corr_change_long.csv", index=False)

    plot_distribution_comparison(session_df, persistence_df, plots_dir / "round1_vs_multiround_distribution.png")
    plot_corr_heatmap(first_corr, "First-Round Cross-Game Spearman Correlation", plots_dir / "corr_first_round_heatmap.png")
    plot_corr_heatmap(multi_corr, "Multiround Cross-Game Spearman Correlation", plots_dir / "corr_multiround_heatmap.png")
    plot_corr_heatmap(delta_corr, "Multiround - First-Round Correlation Change", plots_dir / "corr_delta_heatmap.png", vmin=-0.2, vmax=0.2)

    summary_md = build_summary_markdown(persistence_df, change_df)
    if title_note:
        summary_md += f"\n## Note\n\n- {title_note}\n"
    (output_dir / "summary.md").write_text(summary_md, encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_plot_style()

    output_dir = args.output_dir.resolve()
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rounds_df = build_all_rounds()
    session_df = summarize_sessions(rounds_df)

    rounds_df.to_csv(output_dir / "round_level_measures.csv", index=False)
    session_df.to_csv(output_dir / "session_level_measures.csv", index=False)
    write_analysis_bundle(session_df, output_dir)

    strict_df = session_df[session_df["session_rounds"] >= 2].copy()
    multi_user_counts = strict_df.groupby("measure")["UserID"].nunique()
    keep_measures = multi_user_counts[multi_user_counts >= 100].index.tolist()
    strict_df = strict_df[strict_df["measure"].isin(keep_measures)].copy()
    strict_out = output_dir / "strict_multiround_only"
    write_analysis_bundle(
        strict_df,
        strict_out,
        title_note="Strict multiround-only subset: only sessions with at least 2 observed rounds are retained; measures with fewer than 100 users in that subset are dropped.",
    )

    print(f"Wrote MobLab persistence/correlation analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
