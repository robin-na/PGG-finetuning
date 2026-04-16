#!/usr/bin/env python3
"""Analyze Twin profile-card trait differences by age, sex, and education."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILES_JSONL = THIS_DIR / "output" / "twin_extended_profiles" / "twin_extended_profiles.jsonl"
DEFAULT_CARDS_CSV = THIS_DIR / "output" / "twin_extended_profile_cards" / "standard" / "twin_extended_profile_cards.csv"
DEFAULT_OUTPUT_DIR = THIS_DIR / "output" / "twin_extended_profile_cards" / "standard" / "analysis_demographics"

TRAIT_ORDER = [
    "cooperation_orientation_score",
    "conditional_cooperation_score",
    "norm_enforcement_score",
    "generosity_without_return_score",
    "exploitation_caution_score",
    "communication_coordination_score",
    "behavioral_stability_score",
]

TRAIT_LABELS = {
    "cooperation_orientation_score": "Cooperation\norientation",
    "conditional_cooperation_score": "Conditional\ncooperation",
    "norm_enforcement_score": "Norm\nenforcement",
    "generosity_without_return_score": "Generosity\nwithout return",
    "exploitation_caution_score": "Exploitation\ncaution",
    "communication_coordination_score": "Communication /\ncoordination",
    "behavioral_stability_score": "Behavioral\nstability",
}

AGE_ORDER = ["18-29", "30-49", "50-64", "65+"]
SEX_ORDER = ["Male", "Female"]
EDUCATION_RAW_ORDER = [
    "Less than high school",
    "High school graduate",
    "Some college, no degree",
    "Associate's degree",
    "College graduate/some postgrad",
    "Postgraduate",
]
EDUCATION_HARMONIZED_ORDER = ["high school", "college/postsecondary", "postgraduate"]

EFFECT_SIZE_THRESHOLDS = {
    "cohen_d": [(0.2, "negligible"), (0.5, "small"), (0.8, "medium"), (float("inf"), "large")],
    "eta_squared": [(0.01, "negligible"), (0.06, "small"), (0.14, "medium"), (float("inf"), "large")],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-jsonl", type=Path, default=DEFAULT_PROFILES_JSONL)
    parser.add_argument("--cards-csv", type=Path, default=DEFAULT_CARDS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_demographics(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            profile = json.loads(line)
            feats = {item["name"]: item["value"].get("raw") for item in profile["background_context"]["harmonized_features"]}
            rows.append(
                {
                    "pid": profile["participant"]["pid"],
                    "age_bracket": feats.get("age_bracket"),
                    "sex_assigned_at_birth": feats.get("sex_assigned_at_birth"),
                    "education_completed_raw": feats.get("education_completed_raw"),
                    "education_completed_harmonized": feats.get("education_completed_harmonized"),
                }
            )
    return pd.DataFrame(rows)


def bh_adjust(p_values: List[float]) -> List[float]:
    if not p_values:
        return []
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = np.array(p_values)[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = min(prev, ranked[i] * n / rank)
        adjusted[i] = value
        prev = value
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out.tolist()


def effect_label(metric: str, value: float) -> str:
    abs_value = abs(value)
    for threshold, label in EFFECT_SIZE_THRESHOLDS[metric]:
        if abs_value < threshold:
            return label
    return "large"


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    n1, n2 = len(a), len(b)
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt((((n1 - 1) * s1**2) + ((n2 - 1) * s2**2)) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def eta_squared(sub: pd.DataFrame, group_col: str, value_col: str) -> float:
    grand = float(sub[value_col].mean())
    ss_between = sum(len(vals) * (float(vals.mean()) - grand) ** 2 for _, vals in sub.groupby(group_col)[value_col])
    ss_total = float(((sub[value_col] - grand) ** 2).sum())
    return 0.0 if ss_total == 0 else float(ss_between / ss_total)


def build_group_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in ["age_bracket", "sex_assigned_at_birth", "education_completed_raw", "education_completed_harmonized"]:
        counts = df[var].value_counts(dropna=False)
        for group, count in counts.items():
            rows.append({"variable": var, "group": group, "count": int(count)})
    return pd.DataFrame(rows)


def build_mean_table(df: pd.DataFrame, group_col: str, order: List[str]) -> pd.DataFrame:
    table = df.groupby(group_col)[TRAIT_ORDER].mean().reindex(order)
    table.index.name = group_col
    return table.reset_index()


def build_effect_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    # Sex
    for trait in TRAIT_ORDER:
        sub = df[["sex_assigned_at_birth", trait]].dropna()
        male = sub[sub["sex_assigned_at_birth"] == "Male"][trait].astype(float)
        female = sub[sub["sex_assigned_at_birth"] == "Female"][trait].astype(float)
        t_stat, p_value = stats.ttest_ind(male, female, equal_var=False)
        d = cohen_d(male, female)
        rows.append(
            {
                "demographic": "sex_assigned_at_birth",
                "trait": trait,
                "test": "welch_t",
                "effect_metric": "cohen_d",
                "effect_size": round(d, 4),
                "effect_interpretation": effect_label("cohen_d", d),
                "p_value": float(p_value),
                "group_min": "Female" if female.mean() < male.mean() else "Male",
                "group_min_mean": round(min(male.mean(), female.mean()), 2),
                "group_max": "Male" if male.mean() > female.mean() else "Female",
                "group_max_mean": round(max(male.mean(), female.mean()), 2),
            }
        )

    # Age and education
    for group_col in ["age_bracket", "education_completed_raw", "education_completed_harmonized"]:
        for trait in TRAIT_ORDER:
            sub = df[[group_col, trait]].dropna()
            grouped = [group[trait].astype(float).values for _, group in sub.groupby(group_col)]
            f_stat, p_value = stats.f_oneway(*grouped)
            eta2 = eta_squared(sub, group_col, trait)
            means = sub.groupby(group_col)[trait].mean().sort_values()
            rows.append(
                {
                    "demographic": group_col,
                    "trait": trait,
                    "test": "anova",
                    "effect_metric": "eta_squared",
                    "effect_size": round(eta2, 4),
                    "effect_interpretation": effect_label("eta_squared", eta2),
                    "p_value": float(p_value),
                    "group_min": means.index[0],
                    "group_min_mean": round(float(means.iloc[0]), 2),
                    "group_max": means.index[-1],
                    "group_max_mean": round(float(means.iloc[-1]), 2),
                }
            )

    out = pd.DataFrame(rows)
    out["q_value"] = bh_adjust(out["p_value"].tolist())
    out["q_value"] = out["q_value"].round(6)
    return out.sort_values(["demographic", "q_value", "effect_size"], ascending=[True, True, False])


def build_noticeable_summary(effect_df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (effect_df["effect_interpretation"] != "negligible")
        | (effect_df["q_value"] <= 0.05)
    )
    return effect_df[mask].copy()


def plot_heatmap(table: pd.DataFrame, group_col: str, output_path: Path, title: str) -> None:
    plot_df = table.set_index(group_col)
    values = plot_df[TRAIT_ORDER].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, max(2.4, 0.45 * len(plot_df.index) + 1.5)))
    im = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=20, vmax=85)
    ax.set_xticks(range(len(TRAIT_ORDER)))
    ax.set_xticklabels([TRAIT_LABELS[col] for col in TRAIT_ORDER], fontsize=9)
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index.tolist(), fontsize=9)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=8, color="#1f1f1f")
    ax.set_title(title, fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Mean trait score", fontsize=9)
    fig.subplots_adjust(left=0.27, right=0.95, top=0.86, bottom=0.20)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_effect_sizes(effect_df: pd.DataFrame, output_path: Path) -> None:
    summary = effect_df.copy()
    summary["trait_label"] = summary["trait"].map(TRAIT_LABELS)
    summary["panel"] = summary["demographic"].map(
        {
            "sex_assigned_at_birth": "Sex (Cohen's d)",
            "age_bracket": "Age (eta²)",
            "education_completed_raw": "Education Raw (eta²)",
            "education_completed_harmonized": "Education Harmonized (eta²)",
        }
    )
    panels = ["Sex (Cohen's d)", "Age (eta²)", "Education Raw (eta²)", "Education Harmonized (eta²)"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, panel in zip(axes, panels):
        sub = summary[summary["panel"] == panel]
        vals = sub["effect_size"].astype(float).to_numpy()
        labels = [TRAIT_LABELS[t].replace("\n", " ") for t in sub["trait"]]
        ax.barh(range(len(vals)), vals, color="#5a8fb0")
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(panel, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Trait Effect Sizes by Demographic Variable", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.28, right=0.97, top=0.90, bottom=0.08, wspace=0.35, hspace=0.35)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    demo = load_demographics(args.profiles_jsonl)
    cards = pd.read_csv(args.cards_csv, dtype={"pid": str})
    df = cards.merge(demo, on="pid", how="left")

    counts_df = build_group_counts(df)
    age_means = build_mean_table(df, "age_bracket", AGE_ORDER)
    sex_means = build_mean_table(df, "sex_assigned_at_birth", SEX_ORDER)
    edu_raw_means = build_mean_table(df, "education_completed_raw", EDUCATION_RAW_ORDER)
    edu_h_means = build_mean_table(df, "education_completed_harmonized", EDUCATION_HARMONIZED_ORDER)
    effect_df = build_effect_tests(df)
    noticeable_df = build_noticeable_summary(effect_df)

    counts_df.to_csv(args.output_dir / "group_counts.csv", index=False)
    age_means.to_csv(args.output_dir / "trait_means_by_age.csv", index=False)
    sex_means.to_csv(args.output_dir / "trait_means_by_sex.csv", index=False)
    edu_raw_means.to_csv(args.output_dir / "trait_means_by_education_raw.csv", index=False)
    edu_h_means.to_csv(args.output_dir / "trait_means_by_education_harmonized.csv", index=False)
    effect_df.to_csv(args.output_dir / "effect_tests.csv", index=False)
    noticeable_df.to_csv(args.output_dir / "noticeable_differences.csv", index=False)

    plot_heatmap(age_means, "age_bracket", args.output_dir / "trait_means_by_age.png", "Mean Trait Scores by Age")
    plot_heatmap(sex_means, "sex_assigned_at_birth", args.output_dir / "trait_means_by_sex.png", "Mean Trait Scores by Sex")
    plot_heatmap(
        edu_h_means,
        "education_completed_harmonized",
        args.output_dir / "trait_means_by_education_harmonized.png",
        "Mean Trait Scores by Harmonized Education",
    )
    plot_effect_sizes(effect_df, args.output_dir / "trait_effect_sizes.png")

    print(f"Rows analyzed: {len(df)}")
    print(f"Wrote: {args.output_dir / 'effect_tests.csv'}")
    print(f"Wrote: {args.output_dir / 'noticeable_differences.csv'}")
    print(f"Wrote: {args.output_dir / 'trait_means_by_age.png'}")
    print(f"Wrote: {args.output_dir / 'trait_means_by_sex.png'}")
    print(f"Wrote: {args.output_dir / 'trait_means_by_education_harmonized.png'}")
    print(f"Wrote: {args.output_dir / 'trait_effect_sizes.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
