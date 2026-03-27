#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


DEFAULT_PGG_CSV = THIS_DIR / "demographics_numeric_val.csv"
DEFAULT_TWIN_CSV = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "data"
    / "Twin-2k-500"
    / "snapshot"
    / "question_catalog_and_human_response_csv"
    / "wave1_3_response.csv"
)
DEFAULT_OUTPUT_DIR = THIS_DIR / "analysis_pgg_vs_twin"

AGE_ORDER = ["18-29", "30-49", "50-64", "65+"]
GENDER_ORDER = ["man", "woman", "non_binary", "unknown"]
EDUCATION_ORDER = ["high_school", "bachelor", "master", "other", "unknown"]
SEX_HARMONIZED_ORDER = ["male", "female"]
EDUCATION_DEGREE_COMPLETED_ORDER = ["high_school", "college_postsecondary", "postgraduate"]
EDUCATION_BACHELOR_PLUS_ORDER = ["below_bachelor", "bachelor_or_higher"]
EDUCATION_POSTGRAD_PLUS_ORDER = ["below_postgraduate", "postgraduate_or_higher"]
PLOT_LABEL_MAP = {
    "high_school": "high school",
    "college_postsecondary": "college/postsecondary",
}

TWIN_AGE_MAP = {1: "18-29", 2: "30-49", 3: "50-64", 4: "65+"}
TWIN_AGE_MIDPOINT_MAP = {1: 24, 2: 40, 3: 57, 4: 72}
TWIN_EDUCATION_MAP = {
    1: "other",
    2: "high_school",
    3: "other",
    4: "other",
    5: "bachelor",
    6: "master",
}


def load_pgg_demographics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "playerId",
        "age",
        "age_missing",
        "gender_man",
        "gender_woman",
        "gender_non_binary",
        "gender_unknown",
        "education_high_school",
        "education_bachelor",
        "education_master",
        "education_other",
        "education_unknown",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required PGG columns: {sorted(missing)}")
    if df["playerId"].nunique() != len(df):
        raise ValueError("Expected one row per PGG participant in demographics CSV.")
    return df


def load_twin_demographics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["pid", "QID12", "QID13", "QID14"])
    if df["pid"].nunique() != len(df):
        raise ValueError("Expected one row per Twin participant.")
    return df


def make_age_distribution(pgg: pd.DataFrame, twin: pd.DataFrame) -> pd.DataFrame:
    bins = [18, 30, 50, 65, np.inf]
    pgg_age = pgg.loc[pgg["age_missing"] == 0, "age"]
    pgg_counts = (
        pd.cut(pgg_age, bins=bins, labels=AGE_ORDER, right=False)
        .value_counts()
        .reindex(AGE_ORDER, fill_value=0)
    )
    twin_counts = twin["QID13"].map(TWIN_AGE_MAP).value_counts().reindex(AGE_ORDER, fill_value=0)
    return make_distribution_frame("age", AGE_ORDER, pgg_counts, twin_counts)


def make_gender_distribution(pgg: pd.DataFrame, twin: pd.DataFrame) -> pd.DataFrame:
    pgg_counts = pd.Series(
        {
            "man": int(pgg["gender_man"].sum()),
            "woman": int(pgg["gender_woman"].sum()),
            "non_binary": int(pgg["gender_non_binary"].sum()),
            "unknown": int(pgg["gender_unknown"].sum()),
        }
    ).reindex(GENDER_ORDER, fill_value=0)
    twin_counts = pd.Series(
        {
            "man": int((twin["QID12"] == 1).sum()),
            "woman": int((twin["QID12"] == 2).sum()),
            "non_binary": 0,
            "unknown": 0,
        }
    ).reindex(GENDER_ORDER, fill_value=0)
    return make_distribution_frame("gender", GENDER_ORDER, pgg_counts, twin_counts)


def make_education_distribution(pgg: pd.DataFrame, twin: pd.DataFrame) -> pd.DataFrame:
    pgg_counts = pd.Series(
        {
            "high_school": int(pgg["education_high_school"].sum()),
            "bachelor": int(pgg["education_bachelor"].sum()),
            "master": int(pgg["education_master"].sum()),
            "other": int(pgg["education_other"].sum()),
            "unknown": int(pgg["education_unknown"].sum()),
        }
    ).reindex(EDUCATION_ORDER, fill_value=0)
    twin_counts = twin["QID14"].map(TWIN_EDUCATION_MAP).value_counts().reindex(EDUCATION_ORDER[:-1], fill_value=0)
    twin_counts["unknown"] = int(twin["QID14"].isna().sum())
    twin_counts = twin_counts.reindex(EDUCATION_ORDER, fill_value=0)
    return make_distribution_frame("education", EDUCATION_ORDER, pgg_counts, twin_counts)


def make_harmonized_education_distributions(pgg: pd.DataFrame, twin: pd.DataFrame) -> pd.DataFrame:
    pgg_bachelor_plus = int(pgg["education_bachelor"].sum() + pgg["education_master"].sum())
    pgg_postgrad_plus = int(pgg["education_master"].sum())

    twin_qid14 = twin["QID14"]
    twin_bachelor_plus = int(((twin_qid14 == 5) | (twin_qid14 == 6)).sum())
    twin_postgrad_plus = int((twin_qid14 == 6).sum())

    bachelor_plus_df = make_distribution_frame(
        "education_bachelor_or_higher",
        EDUCATION_BACHELOR_PLUS_ORDER,
        pd.Series(
            {
                "below_bachelor": int(len(pgg) - pgg_bachelor_plus),
                "bachelor_or_higher": pgg_bachelor_plus,
            }
        ),
        pd.Series(
            {
                "below_bachelor": int(len(twin) - twin_bachelor_plus),
                "bachelor_or_higher": twin_bachelor_plus,
            }
        ),
    )

    postgrad_plus_df = make_distribution_frame(
        "education_postgraduate_or_higher",
        EDUCATION_POSTGRAD_PLUS_ORDER,
        pd.Series(
            {
                "below_postgraduate": int(len(pgg) - pgg_postgrad_plus),
                "postgraduate_or_higher": pgg_postgrad_plus,
            }
        ),
        pd.Series(
            {
                "below_postgraduate": int(len(twin) - twin_postgrad_plus),
                "postgraduate_or_higher": twin_postgrad_plus,
            }
        ),
    )

    return pd.concat([bachelor_plus_df, postgrad_plus_df], ignore_index=True)


def make_harmonized_sex_distribution(pgg: pd.DataFrame, twin: pd.DataFrame) -> pd.DataFrame:
    pgg_counts = pd.Series(
        {
            "male": int(pgg["gender_man"].sum()),
            "female": int(pgg["gender_woman"].sum()),
        }
    ).reindex(SEX_HARMONIZED_ORDER, fill_value=0)
    twin_counts = pd.Series(
        {
            "male": int((twin["QID12"] == 1).sum()),
            "female": int((twin["QID12"] == 2).sum()),
        }
    ).reindex(SEX_HARMONIZED_ORDER, fill_value=0)
    return make_distribution_frame("sex_male_female", SEX_HARMONIZED_ORDER, pgg_counts, twin_counts)


def make_degree_completed_education_distribution(pgg: pd.DataFrame, twin: pd.DataFrame) -> pd.DataFrame:
    # Practical harmonization for reporting:
    # - Twin some-college-no-degree is grouped with high_school because the highest completed degree is high school.
    # - Twin associate's degree is grouped with college/postsecondary.
    # - Twin less-than-high-school is merged into high_school due very low prevalence and lack of a matching PGG category.
    # - PGG other is grouped with college/postsecondary as the closest non-postgraduate catch-all.
    pgg_counts = pd.Series(
        {
            "high_school": int(pgg["education_high_school"].sum()),
            "college_postsecondary": int(pgg["education_bachelor"].sum() + pgg["education_other"].sum()),
            "postgraduate": int(pgg["education_master"].sum()),
        }
    ).reindex(EDUCATION_DEGREE_COMPLETED_ORDER, fill_value=0)
    twin_qid14 = twin["QID14"]
    twin_counts = pd.Series(
        {
            "high_school": int(((twin_qid14 == 1) | (twin_qid14 == 2) | (twin_qid14 == 3)).sum()),
            "college_postsecondary": int(((twin_qid14 == 4) | (twin_qid14 == 5)).sum()),
            "postgraduate": int((twin_qid14 == 6).sum()),
        }
    ).reindex(EDUCATION_DEGREE_COMPLETED_ORDER, fill_value=0)
    return make_distribution_frame(
        "education_degree_completed_harmonized",
        EDUCATION_DEGREE_COMPLETED_ORDER,
        pgg_counts,
        twin_counts,
    )


def make_distribution_frame(
    dimension: str,
    category_order: list[str],
    pgg_counts: pd.Series,
    twin_counts: pd.Series,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "dimension": dimension,
            "category": category_order,
            "pgg_count": [int(pgg_counts.get(cat, 0)) for cat in category_order],
            "twin_count": [int(twin_counts.get(cat, 0)) for cat in category_order],
        }
    )
    out["pgg_pct"] = (out["pgg_count"] / out["pgg_count"].sum() * 100).round(2)
    out["twin_pct"] = (out["twin_count"] / out["twin_count"].sum() * 100).round(2)
    out["pct_point_diff_pgg_minus_twin"] = (out["pgg_pct"] - out["twin_pct"]).round(2)
    return out


def total_variation_distance(frame: pd.DataFrame) -> float:
    p = frame["pgg_pct"].to_numpy(dtype=float) / 100.0
    q = frame["twin_pct"].to_numpy(dtype=float) / 100.0
    return float(0.5 * np.abs(p - q).sum())


def make_divergence_frame(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for frame in frames:
        dimension = str(frame["dimension"].iloc[0])
        rows.append(
            {
                "dimension": dimension,
                "pgg_n_used": int(frame["pgg_count"].sum()),
                "twin_n_used": int(frame["twin_count"].sum()),
                "total_variation_distance": total_variation_distance(frame),
                "total_variation_distance_pct_points": round(total_variation_distance(frame) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def make_test_frame(
    age_df: pd.DataFrame,
    gender_df: pd.DataFrame,
    education_df: pd.DataFrame,
    harmonized_education_df: pd.DataFrame,
    harmonized_sex_df: pd.DataFrame,
    degree_completed_education_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for dimension, frame in [("age", age_df), ("education", education_df)]:
        contingency = frame[["pgg_count", "twin_count"]].to_numpy()
        chi2, p_value, dof, _ = chi2_contingency(contingency)
        rows.append(
            {
                "dimension": dimension,
                "test": "chi2",
                "statistic": round(float(chi2), 6),
                "degrees_of_freedom": int(dof),
                "p_value": float(p_value),
                "notes": "",
            }
        )

    gender_known = gender_df[gender_df["category"].isin(["man", "woman"])]
    contingency = gender_known[["pgg_count", "twin_count"]].to_numpy()
    chi2, p_value, dof, _ = chi2_contingency(contingency)
    rows.append(
        {
            "dimension": "gender",
            "test": "chi2",
            "statistic": round(float(chi2), 6),
            "degrees_of_freedom": int(dof),
            "p_value": float(p_value),
            "notes": "Computed on man/woman only because Twin has no non-binary or unknown category.",
        }
    )

    for dimension, frame in harmonized_education_df.groupby("dimension", sort=False):
        contingency = frame[["pgg_count", "twin_count"]].to_numpy()
        chi2, p_value, dof, _ = chi2_contingency(contingency)
        rows.append(
            {
                "dimension": dimension,
                "test": "chi2",
                "statistic": round(float(chi2), 6),
                "degrees_of_freedom": int(dof),
                "p_value": float(p_value),
                "notes": (
                    "Coarse harmonization. PGG uses high_school/bachelor/master/other, "
                    "while Twin uses less_than_high_school/high_school/some_college/"
                    "associate/college_graduate/postgraduate."
                ),
            }
        )

    for dimension, frame, notes in [
        (
            "sex_male_female",
            harmonized_sex_df,
            "Harmonized male/female comparison. PGG non-binary and unknown responses are excluded.",
        ),
        (
            "education_degree_completed_harmonized",
            degree_completed_education_df,
            (
                "Practical completed-degree harmonization. Twin some_college_no_degree is grouped with high school, "
                "Twin associate's with college/postsecondary, Twin less_than_high_school merged into high school due rarity, "
                "and PGG other grouped with college/postsecondary."
            ),
        ),
    ]:
        contingency = frame[["pgg_count", "twin_count"]].to_numpy()
        chi2, p_value, dof, _ = chi2_contingency(contingency)
        rows.append(
            {
                "dimension": dimension,
                "test": "chi2",
                "statistic": round(float(chi2), 6),
                "degrees_of_freedom": int(dof),
                "p_value": float(p_value),
                "notes": notes,
            }
        )

    return pd.DataFrame(rows)


def make_summary_frame(pgg: pd.DataFrame, twin: pd.DataFrame) -> pd.DataFrame:
    pgg_age = pgg.loc[pgg["age_missing"] == 0, "age"]
    twin_age = twin["QID13"].map(TWIN_AGE_MIDPOINT_MAP)
    twin_less_than_high_school = int((twin["QID14"] == 1).sum())
    rows = [
        {"metric": "pgg_participants", "value": int(len(pgg)), "notes": ""},
        {"metric": "twin_participants", "value": int(len(twin)), "notes": ""},
        {"metric": "pgg_mean_age", "value": round(float(pgg_age.mean()), 4), "notes": ""},
        {"metric": "pgg_median_age", "value": round(float(pgg_age.median()), 4), "notes": ""},
        {
            "metric": "twin_mean_age_midpoint",
            "value": round(float(twin_age.mean()), 4),
            "notes": "Approximate: Twin age is reported in brackets and mapped to bracket midpoints.",
        },
        {
            "metric": "twin_median_age_midpoint",
            "value": round(float(twin_age.median()), 4),
            "notes": "Approximate: Twin age is reported in brackets and mapped to bracket midpoints.",
        },
        {
            "metric": "twin_less_than_high_school_count",
            "value": twin_less_than_high_school,
            "notes": "",
        },
        {
            "metric": "twin_less_than_high_school_pct",
            "value": round(float(twin_less_than_high_school / len(twin) * 100), 4),
            "notes": "",
        },
    ]
    return pd.DataFrame(rows)


def plot_distribution(frame: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(frame))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, frame["pgg_pct"], width=width, label="PGG val")
    ax.bar(x + width / 2, frame["twin_pct"], width=width, label="Twin-2k")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["category"], rotation=20, ha="right")
    ax.set_ylabel("Percent of participants")
    ax.set_title(f"{frame['dimension'].iloc[0].title()} Distribution")
    ax.legend()
    ax.set_ylim(0, max(frame["pgg_pct"].max(), frame["twin_pct"].max()) * 1.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_combined_distributions(
    age_df: pd.DataFrame,
    sex_df: pd.DataFrame,
    education_df: pd.DataFrame,
    output_path: Path,
) -> None:
    frames = [age_df, sex_df, education_df]
    titles = ["Age", "Male/Female", "Education"]
    ymax = max(
        frame[["pgg_pct", "twin_pct"]].to_numpy().max()
        for frame in frames
    ) * 1.18

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)
    handles = None

    for ax, frame, title in zip(axes, frames, titles):
        x = np.arange(len(frame))
        width = 0.38
        pgg_bar = ax.bar(x - width / 2, frame["pgg_pct"], width=width, label="PGG val")
        twin_bar = ax.bar(x + width / 2, frame["twin_pct"], width=width, label="Twin-2k")
        ax.set_xticks(x)
        display_labels = [PLOT_LABEL_MAP.get(category, category) for category in frame["category"]]
        ax.set_xticklabels(display_labels, rotation=18, ha="right")
        ax.set_title(f"{title}\nTVD = {total_variation_distance(frame) * 100:.1f} pp")
        ax.set_ylim(0, ymax)
        if handles is None:
            handles = [pgg_bar[0], twin_bar[0]]

    axes[0].set_ylabel("Percent of participants")
    fig.suptitle("PGG Validation vs Twin-2k Demographic Distributions", y=0.97, fontsize=14)
    fig.legend(handles, ["PGG val", "Twin-2k"], loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False)
    fig.text(
        0.5,
        0.05,
        (
            "Note: Twin uses sex assigned at birth; PGG uses data.gender. "
            "Male/Female excludes PGG non-binary/unknown. Education is completed-degree harmonized."
        ),
        ha="center",
        va="top",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.22, wspace=0.08)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PGG and Twin demographics.")
    parser.add_argument("--pgg-csv", type=Path, default=DEFAULT_PGG_CSV)
    parser.add_argument("--twin-csv", type=Path, default=DEFAULT_TWIN_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    pgg = load_pgg_demographics(args.pgg_csv)
    twin = load_twin_demographics(args.twin_csv)

    age_df = make_age_distribution(pgg, twin)
    gender_df = make_gender_distribution(pgg, twin)
    education_df = make_education_distribution(pgg, twin)
    harmonized_education_df = make_harmonized_education_distributions(pgg, twin)
    harmonized_sex_df = make_harmonized_sex_distribution(pgg, twin)
    degree_completed_education_df = make_degree_completed_education_distribution(pgg, twin)
    harmonized_distribution_df = pd.concat(
        [age_df, harmonized_sex_df, degree_completed_education_df],
        ignore_index=True,
    )
    divergence_df = make_divergence_frame([age_df, harmonized_sex_df, degree_completed_education_df])
    distribution_df = pd.concat([age_df, gender_df, education_df], ignore_index=True)
    tests_df = make_test_frame(
        age_df,
        gender_df,
        education_df,
        harmonized_education_df,
        harmonized_sex_df,
        degree_completed_education_df,
    )
    summary_df = make_summary_frame(pgg, twin)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    distribution_df.to_csv(args.output_dir / "distribution_comparison.csv", index=False)
    harmonized_distribution_df.to_csv(args.output_dir / "harmonized_distribution_comparison.csv", index=False)
    harmonized_education_df.to_csv(args.output_dir / "education_harmonized_comparison.csv", index=False)
    divergence_df.to_csv(args.output_dir / "divergence_metrics.csv", index=False)
    tests_df.to_csv(args.output_dir / "statistical_tests.csv", index=False)
    summary_df.to_csv(args.output_dir / "summary_metrics.csv", index=False)

    plot_distribution(age_df, args.output_dir / "age_distribution.png")
    plot_distribution(gender_df, args.output_dir / "gender_distribution.png")
    plot_distribution(education_df, args.output_dir / "education_distribution.png")
    plot_combined_distributions(
        age_df,
        harmonized_sex_df,
        degree_completed_education_df,
        args.output_dir / "combined_distribution_comparison.png",
    )
    plot_distribution(
        harmonized_education_df[harmonized_education_df["dimension"] == "education_bachelor_or_higher"],
        args.output_dir / "education_bachelor_or_higher_distribution.png",
    )
    plot_distribution(
        harmonized_education_df[harmonized_education_df["dimension"] == "education_postgraduate_or_higher"],
        args.output_dir / "education_postgraduate_or_higher_distribution.png",
    )

    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
