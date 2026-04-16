#!/usr/bin/env python3
"""
Generate charts and English summary report for PGG archetype transfer experiment.
Reads all per_question CSV files from evaluation/ and produces publication-ready figures.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

import argparse

from config import OUTPUT_ROOT, QUESTION_CATALOG_JSON

# ── Display names ─────────────────────────────────────────────────────────────
MODE_LABELS = {
    "archetype":          "PGG Archetype (D-retrieval)",
    "demographics_only":  "Demographics Only",
    "random_archetype":   "Random Archetype",
}
MODE_COLORS = {
    "archetype":          "#2196F3",   # blue
    "demographics_only":  "#9E9E9E",   # grey
    "random_archetype":   "#FF9800",   # orange
}
MODE_ORDER = ["demographics_only", "random_archetype", "archetype"]


# ── Question type helper ──────────────────────────────────────────────────────

def load_question_meta() -> Dict[str, Dict]:
    catalog = json.load(QUESTION_CATALOG_JSON.open())
    out = {}
    for q in catalog:
        if "Economic preferences" not in q.get("BlockName", ""):
            continue
        if q.get("is_descriptive") or q["QuestionType"] not in ("MC", "Matrix"):
            continue
        game_tag = _game_tag(q["QuestionID"], q["QuestionText"])
        for col in q.get("csv_columns", []):
            out[col] = {"qid": q["QuestionID"], "qtype": q["QuestionType"], "game": game_tag}
    return out


def _game_tag(qid: str, text: str) -> str:
    text_lower = text.lower()
    if qid in ("QID224",):
        return "Ultimatum (proposer)"
    if qid in ("QID225", "QID226", "QID227", "QID228", "QID229", "QID230"):
        return "Ultimatum (responder)"
    if qid in ("QID117",):
        return "Trust (sender)"
    if qid in ("QID118", "QID119", "QID120", "QID121", "QID122"):
        return "Trust (receiver)"
    if qid == "QID231":
        return "Dictator"
    if qid in ("QID149", "QID150", "QID151", "QID152"):
        return "Mental accounting"
    if qid in ("QID84", "QID244", "QID245", "QID246", "QID247", "QID248"):
        return "Time preferences"
    if qid in ("QID250", "QID251", "QID252"):
        return "Risk preferences (gain)"
    if qid in ("QID276", "QID277", "QID278", "QID279"):
        return "Risk preferences (loss)"
    return "Other"


# ── Load all evaluation results ───────────────────────────────────────────────

def load_all_results(eval_dir: Path) -> Dict[str, pd.DataFrame]:
    results = {}
    for f in sorted(eval_dir.glob("per_question_*.csv")):
        mode = f.stem.replace("per_question_", "")
        df = pd.read_csv(f)
        results[mode] = df
    return results


def load_summaries(eval_dir: Path) -> Dict[str, Dict]:
    summaries = {}
    for f in sorted(eval_dir.glob("summary_*.json")):
        mode = f.stem.replace("summary_", "")
        summaries[mode] = json.loads(f.read_text())
    return summaries


# ── Chart 1: Overall metrics bar chart ───────────────────────────────────────

def plot_overall_metrics(summaries: Dict[str, Dict], out_dir: Path) -> None:
    modes = [m for m in MODE_ORDER if m in summaries]
    if not modes:
        return

    metrics = ["mean_accuracy", "mean_correlation", "mean_mad"]
    metric_labels = ["Accuracy", "Pearson r", "MAD (↓)"]
    metric_better = ["↑", "↑", "↓"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("Overall Performance on Economic Game Prediction\n(200 participants × 31 questions)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, metric, label, better in zip(axes, metrics, metric_labels, metric_better):
        vals = [summaries[m].get(metric, np.nan) for m in modes]
        colors = [MODE_COLORS.get(m, "#ccc") for m in modes]
        labels = [MODE_LABELS.get(m, m) for m in modes]

        bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.2, width=0.55)
        ax.set_title(f"{label} {better}", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(v for v in vals if not np.isnan(v)) * 1.25 if any(not np.isnan(v) for v in vals) else 1)
        ax.tick_params(axis="x", rotation=20, labelsize=8.5)
        ax.grid(axis="y", alpha=0.35, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "fig1_overall_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig1_overall_metrics.png")


# ── Chart 2: Per-game accuracy ─────────────────────────────────────────────────

def plot_per_game(results: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    qmeta = load_question_meta()
    modes = [m for m in MODE_ORDER if m in results]
    if not modes:
        return

    # Tag each column with its game
    for mode, df in results.items():
        df["game"] = df["column"].map(lambda c: qmeta.get(c, {}).get("game", "Other"))

    # Compute per-game mean accuracy
    game_data = {}
    for mode in modes:
        df = results[mode]
        grp = df.groupby("game")["accuracy"].mean()
        game_data[mode] = grp

    all_games = sorted(set(g for g in game_data[modes[0]].index))
    x = np.arange(len(all_games))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 5.5))
    for i, mode in enumerate(modes):
        vals = [game_data[mode].get(g, np.nan) for g in all_games]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=MODE_LABELS.get(mode, mode),
                      color=MODE_COLORS.get(mode, "#ccc"), edgecolor="white", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([g.replace(" ", "\n") for g in all_games], fontsize=9)
    ax.set_ylabel("Accuracy (exact match)", fontsize=11)
    ax.set_title("Per-Game Accuracy by Prediction Mode", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(out_dir / "fig2_per_game_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig2_per_game_accuracy.png")


# ── Chart 3: Correlation distribution ─────────────────────────────────────────

def plot_correlation_distribution(results: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    modes = [m for m in MODE_ORDER if m in results]
    if not modes:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for mode in modes:
        df = results[mode]
        corrs = df["correlation"].dropna()
        ax.hist(corrs, bins=20, alpha=0.55, label=MODE_LABELS.get(mode, mode),
                color=MODE_COLORS.get(mode, "#ccc"), edgecolor="white")
        ax.axvline(corrs.median(), color=MODE_COLORS.get(mode, "#ccc"),
                   linestyle="--", linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Pearson r (per question)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Per-Question Correlation with Human Responses", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.97, 0.96, "dashed = median", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(out_dir / "fig3_correlation_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig3_correlation_distribution.png")


# ── Chart 4: MAD heatmap by game × mode ──────────────────────────────────────

def plot_mad_heatmap(results: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    qmeta = load_question_meta()
    modes = [m for m in MODE_ORDER if m in results]
    if len(modes) < 2:
        return

    for mode, df in results.items():
        df["game"] = df["column"].map(lambda c: qmeta.get(c, {}).get("game", "Other"))

    all_games = sorted(set(results[modes[0]]["game"].unique()))
    mat = np.zeros((len(modes), len(all_games)))
    for i, mode in enumerate(modes):
        df = results[mode]
        grp = df.groupby("game")["mad"].mean()
        for j, game in enumerate(all_games):
            mat[i, j] = grp.get(game, np.nan)

    fig, ax = plt.subplots(figsize=(13, 3.5))
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto",
                   vmin=np.nanmin(mat) * 0.8, vmax=np.nanmax(mat) * 1.1)
    plt.colorbar(im, ax=ax, shrink=0.8, label="MAD (lower = better)")

    ax.set_xticks(range(len(all_games)))
    ax.set_xticklabels([g.replace(" ", "\n") for g in all_games], fontsize=9)
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels([MODE_LABELS.get(m, m) for m in modes], fontsize=9)
    ax.set_title("Mean Absolute Deviation (MAD) by Game × Mode", fontsize=12, fontweight="bold")

    for i in range(len(modes)):
        for j in range(len(all_games)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black")

    plt.tight_layout()
    plt.savefig(out_dir / "fig4_mad_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig4_mad_heatmap.png")


# ── Chart 5: Archetype vs Demographics lift ──────────────────────────────────

def plot_lift(results: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    if "archetype" not in results or "demographics_only" not in results:
        return

    qmeta = load_question_meta()
    base = results["demographics_only"].set_index("column")
    arch = results["archetype"].set_index("column")
    shared = base.index.intersection(arch.index)

    lift_acc = arch.loc[shared, "accuracy"] - base.loc[shared, "accuracy"]
    lift_corr = arch.loc[shared, "correlation"] - base.loc[shared, "correlation"]

    game_tags = pd.Series({c: qmeta.get(c, {}).get("game", "Other") for c in shared})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, lift, label in zip(axes,
                                [lift_acc, lift_corr],
                                ["Accuracy lift (Archetype − Demographics-only)",
                                 "Correlation lift (Archetype − Demographics-only)"]):
        df_lift = pd.DataFrame({"lift": lift, "game": game_tags})
        grp = df_lift.groupby("game")["lift"].mean().sort_values()
        colors = ["#d32f2f" if v < 0 else "#388e3c" for v in grp.values]
        grp.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(label, fontsize=10)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Archetype Conditioning Lift over Demographics-Only Baseline",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "fig5_archetype_lift.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig5_archetype_lift.png")


# ── English description ───────────────────────────────────────────────────────

def write_description(summaries: Dict[str, Dict], results: Dict[str, pd.DataFrame],
                      out_dir: Path) -> None:
    qmeta = load_question_meta()
    modes_available = [m for m in MODE_ORDER if m in summaries]

    # Compute key numbers
    n_q = summaries.get("demographics_only", summaries.get(modes_available[0], {})).get(
        "n_questions_evaluated", "?")
    n_p = summaries.get("demographics_only", summaries.get(modes_available[0], {})).get(
        "n_participants", "?")

    lines = []
    lines.append("# PGG Archetype Transfer to Economic Game Prediction")
    lines.append("## Experiment Description\n")
    lines.append(textwrap.dedent(f"""\
        We evaluate whether behavioral archetypes derived from a Public Goods Game (PGG)
        can improve predictions of individual behavior in other economic games, when only
        sparse demographic information (age, sex, education) is available about participants.

        **Dataset:** Twin-2K-500 (2,058 US participants, Toubia et al. 2025)
        **Target tasks:** {n_q} question-columns across 9 economic game types
          (Ultimatum Game proposer/responder, Dictator Game, Trust Game sender/receiver,
           Time Preferences, Risk Preferences (gain/loss), Mental Accounting)
        **Evaluation:** Pilot of {n_p} randomly sampled participants (pilot_n=200 from 2,058)
        **Prediction model:** GPT-4o, temperature=0
    """))

    lines.append("## Methods\n")
    lines.append(textwrap.dedent("""\
        Three conditions were tested:

        1. **Demographics Only** (baseline): GPT-4o receives only the participant's age
           bracket, sex, and education level, and predicts the game response.

        2. **Random Archetype** (control): GPT-4o receives demographics plus a randomly
           sampled PGG behavioral archetype from the learning wave. This tests whether any
           archetype—regardless of fit—adds information beyond demographics alone.

        3. **PGG Archetype (D-retrieval)** (main condition): A Ridge regression model maps
           the participant's demographic features to a 3072-dim embedding space. The top-10
           nearest PGG archetypes are retrieved from the learning-wave bank via cosine
           similarity, and provided to GPT-4o alongside the question. The LLM selects the
           most relevant behavioral signals and applies them to the target question.

        PGG archetypes are rich prose descriptions of a player's behavior in a Public Goods
        Game, covering: contribution patterns, punishment/reward behavior, responses to
        others' outcomes, and end-game strategy. These behavioral tendencies are used as
        personality signals to predict economic game responses.
    """))

    lines.append("## Results\n")

    # Build results table
    lines.append("| Mode | Accuracy | Pearson r | MAD |")
    lines.append("|------|----------|-----------|-----|")
    for mode in MODE_ORDER:
        if mode not in summaries:
            continue
        s = summaries[mode]
        acc = s.get("mean_accuracy", float("nan"))
        corr = s.get("mean_correlation", float("nan"))
        mad = s.get("mean_mad", float("nan"))
        lines.append(f"| {MODE_LABELS[mode]} | {acc:.3f} | {corr:.3f} | {mad:.3f} |")
    lines.append("")

    # Narrative summary
    if "archetype" in summaries and "demographics_only" in summaries:
        arch_acc = summaries["archetype"].get("mean_accuracy", 0)
        demo_acc = summaries["demographics_only"].get("mean_accuracy", 0)
        arch_corr = summaries["archetype"].get("mean_correlation", 0)
        demo_corr = summaries["demographics_only"].get("mean_correlation", 0)
        lift_acc = arch_acc - demo_acc
        lift_corr = arch_corr - demo_corr

        sign_acc = "above" if lift_acc > 0 else "below"
        sign_corr = "above" if lift_corr > 0 else "below"

        lines.append(textwrap.dedent(f"""\
            The PGG Archetype condition achieves {arch_acc:.3f} accuracy and {arch_corr:.3f}
            mean Pearson r with human responses, compared to {demo_acc:.3f} accuracy and
            {demo_corr:.3f} correlation for the Demographics-Only baseline. This represents
            a {abs(lift_acc):.3f} {sign_acc} lift in accuracy and {abs(lift_corr):.3f}
            {sign_corr} lift in correlation from archetype conditioning.
        """))

        if "random_archetype" in summaries:
            rand_acc = summaries["random_archetype"].get("mean_accuracy", 0)
            rand_corr = summaries["random_archetype"].get("mean_correlation", 0)
            lines.append(textwrap.dedent(f"""\
                The Random Archetype control achieves {rand_acc:.3f} accuracy ({rand_corr:.3f} r),
                {'above' if rand_acc > demo_acc else 'below'} the Demographics-Only baseline.
                Comparing Random vs. Retrieved archetypes isolates the value of demographic-based
                retrieval over providing any arbitrary PGG behavioral profile.
            """))

    # Per-game breakdown
    if "archetype" in results:
        qmeta_loaded = load_question_meta()
        df = results["archetype"].copy()
        df["game"] = df["column"].map(lambda c: qmeta_loaded.get(c, {}).get("game", "Other"))
        game_summary = df.groupby("game").agg(
            accuracy=("accuracy", "mean"),
            correlation=("correlation", "mean"),
            n=("column", "count"),
        ).sort_values("accuracy", ascending=False)

        lines.append("### Per-Game Results (Archetype mode)\n")
        lines.append("| Game | N cols | Accuracy | Pearson r |")
        lines.append("|------|--------|----------|-----------|")
        for game, row in game_summary.iterrows():
            lines.append(f"| {game} | {int(row['n'])} | {row['accuracy']:.3f} | {row['correlation']:.3f} |")
        lines.append("")

    lines.append("## Figures\n")
    lines.append(textwrap.dedent("""\
        - **fig1_overall_metrics.png** — Bar chart comparing Accuracy, Pearson r, and MAD
          across three prediction modes.
        - **fig2_per_game_accuracy.png** — Grouped bar chart of accuracy by game type and mode.
        - **fig3_correlation_distribution.png** — Histogram of per-question Pearson r values
          for each mode; dashed lines show medians.
        - **fig4_mad_heatmap.png** — Heatmap of MAD by game type × mode (red = worse, green = better).
        - **fig5_archetype_lift.png** — Per-game lift of Archetype over Demographics-Only baseline
          for accuracy and correlation.
    """))

    lines.append("## Notes\n")
    lines.append(textwrap.dedent("""\
        - Economic game questions (Ultimatum, Dictator, Trust, Time/Risk Preferences) are
          only available in wave 1-3 of Twin-2K-500 (not repeated in wave 4), so we
          evaluate against wave 1-3 ground truth.
        - The D-only Ridge model has near-zero hit@1 (expected with only 3 demographic
          dimensions), meaning retrieval is effectively random within demographic strata.
          The main test is therefore whether archetype format/content adds signal over
          pure demographics, not whether retrieval precision matters.
        - All predictions are zero-temperature (deterministic).
        - Text entry questions (reasoning rationales) are excluded from evaluation.
    """))

    desc_path = out_dir / "description.md"
    desc_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved description.md")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report from evaluation results.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root output directory (default: OUTPUT_ROOT from config). "
             "Report reads from <output-root>/evaluation/ and writes to <output-root>/report/",
    )
    args = parser.parse_args()

    eval_dir = args.output_root / "evaluation"
    report_dir = args.output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation results...")
    results = load_all_results(eval_dir)
    summaries = load_summaries(eval_dir)

    if not results:
        print(f"No per_question_*.csv files found in {eval_dir}. Run evaluate.py first.")
        return

    print(f"Found modes: {list(results.keys())}")

    print("\nGenerating figures...")
    plot_overall_metrics(summaries, report_dir)
    plot_per_game(results, report_dir)
    plot_correlation_distribution(results, report_dir)
    plot_mad_heatmap(results, report_dir)
    plot_lift(results, report_dir)

    print("\nWriting description...")
    write_description(summaries, results, report_dir)

    print(f"\nAll outputs saved to: {report_dir}")
    print("\nFiles:")
    for f in sorted(report_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
