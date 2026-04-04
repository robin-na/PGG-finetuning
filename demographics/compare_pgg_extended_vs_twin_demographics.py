#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from demographics.generate_demo_table import _normalize_gender


DEFAULT_PGG_EXTENDED_CSV = THIS_DIR / "merged_demographcs_prolific.csv"
DEFAULT_VALIDATION_ANALYSIS_CSV = PROJECT_ROOT / "data" / "processed_data" / "df_analysis_val.csv"
DEFAULT_TWIN_CSV = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "data"
    / "Twin-2k-500"
    / "snapshot"
    / "question_catalog_and_human_response_csv"
    / "wave1_3_response.csv"
)
DEFAULT_OUTPUT_DIR = THIS_DIR / "analysis_pgg_extended_vs_twin"
DEFAULT_VALIDATION_ONLY_OUTPUT_DIR = THIS_DIR / "analysis_pgg_validation_extended_vs_twin"

AGE_ORDER = ["18-29", "30-49", "50-64", "65+"]
SEX_ORDER = ["male", "female"]
EDUCATION_ORDER = ["high school", "college/postsecondary", "postgraduate"]
RACE_ORDER = ["white", "black", "asian", "other/mixed"]
EMPLOYMENT_ORDER = ["full-time", "part-time", "unemployed", "student", "other/nonstandard"]
US_BINARY_ORDER = ["yes", "no"]

INVALID_STRINGS = {"", "nan", "none", "data_expired", "consent_revoked", "not applicable"}

TWIN_AGE_MAP = {1: "18-29", 2: "30-49", 3: "50-64", 4: "65+"}
TWIN_AGE_MIDPOINT_MAP = {1: 24, 2: 40, 3: 57, 4: 72}
TWIN_SEX_MAP = {1: "male", 2: "female"}
TWIN_EDU_MAP = {
    1: "high school",
    2: "high school",
    3: "high school",
    4: "college/postsecondary",
    5: "college/postsecondary",
    6: "postgraduate",
}
TWIN_RACE_MAP = {
    1: "white",
    2: "black",
    3: "asian",
    4: "other/mixed",
    5: "other/mixed",
}
TWIN_EMPLOYMENT_MAP = {
    1: "full-time",
    2: "part-time",
    3: "unemployed",
    4: "other/nonstandard",
    5: "other/nonstandard",
    6: "student",
    7: "other/nonstandard",
}

DIMENSION_LABELS = {
    "age": "Age",
    "sex_male_female": "Male/Female",
    "education_completed": "Education",
    "race_ethnicity_harmonized": "Race/Ethnicity",
    "employment_harmonized": "Employment",
    "us_nationality_proxy": "U.S. Nationality Proxy",
    "us_residence_proxy": "U.S. Residence Proxy",
}

PLOT_LABEL_MAP = {
    "high school": "high school",
    "college/postsecondary": "college/postsecondary",
    "postgraduate": "postgraduate",
    "other/mixed": "other/mixed",
    "other/nonstandard": "other/nonstandard",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Twin demographics to the extended PGG-Prolific demographic file.")
    parser.add_argument("--pgg-extended-csv", type=Path, default=DEFAULT_PGG_EXTENDED_CSV)
    parser.add_argument("--validation-analysis-csv", type=Path, default=DEFAULT_VALIDATION_ANALYSIS_CSV)
    parser.add_argument("--twin-csv", type=Path, default=DEFAULT_TWIN_CSV)
    parser.add_argument("--scope", choices=["extended_all", "validation_only"], default="extended_all")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if args.scope == "validation_only":
        return DEFAULT_VALIDATION_ONLY_OUTPUT_DIR
    return DEFAULT_OUTPUT_DIR


def normalized_string(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in INVALID_STRINGS:
        return None
    return text


def parse_numeric(value: Any) -> float | None:
    text = normalized_string(value)
    if text is None:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def harmonize_age_bracket(age_value: Any) -> str | None:
    parsed = parse_numeric(age_value)
    if parsed is None:
        return None
    if parsed < 30:
        return "18-29"
    if parsed < 50:
        return "30-49"
    if parsed < 65:
        return "50-64"
    return "65+"


def harmonize_prolific_sex(value: Any) -> str | None:
    text = normalized_string(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered == "male":
        return "male"
    if lowered == "female":
        return "female"
    return None


def harmonize_exit_gender(value: Any) -> str | None:
    normalized = _normalize_gender(value)
    return {"man": "male", "woman": "female"}.get(normalized)


def harmonize_pgg_education(value: Any) -> str | None:
    text = normalized_string(value)
    if text is None:
        return None
    mapping = {
        "high-school": "high school",
        "bachelor": "college/postsecondary",
        "other": "college/postsecondary",
        "master": "postgraduate",
    }
    return mapping.get(text.lower())


def harmonize_pgg_race_ethnicity(value: Any) -> str | None:
    text = normalized_string(value)
    if text is None:
        return None
    mapping = {
        "white": "white",
        "black": "black",
        "asian": "asian",
        "mixed": "other/mixed",
        "other": "other/mixed",
        "prefer not to say": None,
    }
    return mapping.get(text.lower())


def harmonize_pgg_employment(employment_value: Any, student_value: Any) -> str | None:
    employment = normalized_string(employment_value)
    student = normalized_string(student_value)
    student_yes = student is not None and student.lower() == "yes"
    if employment is None and not student_yes:
        return None
    if student_yes and employment not in {"Full-Time", "Part-Time"}:
        return "student"
    if employment == "Full-Time":
        return "full-time"
    if employment == "Part-Time":
        return "part-time"
    if employment == "Unemployed (and job seeking)":
        return "unemployed"
    if employment in {
        "Not in paid work (e.g. homemaker', 'retired or disabled)",
        "Other",
        "Due to start a new job within the next month",
    }:
        return "other/nonstandard"
    if student_yes:
        return "student"
    return None


def is_us_country(value: Any) -> str | None:
    text = normalized_string(value)
    if text is None:
        return None
    return "yes" if text == "United States" else "no"


def load_validation_game_ids(path: Path) -> set[str]:
    return set(pd.read_csv(path, usecols=["gameId"])["gameId"].astype(str))


def load_pgg_extended(path: Path, validation_game_ids: set[str] | None = None) -> pd.DataFrame:
    usecols = [
        "PGGEXIT_playerId",
        "PGGEXIT_gameId",
        "PGGEXIT_createdAt",
        "PGGEXIT_data.age",
        "PGGEXIT_data.gender",
        "PGGEXIT_data.education",
        "PROLIFIC_Age",
        "PROLIFIC_Sex",
        "PROLIFIC_Ethnicity simplified",
        "PROLIFIC_Country of birth",
        "PROLIFIC_Country of residence",
        "PROLIFIC_Nationality",
        "PROLIFIC_Student status",
        "PROLIFIC_Employment status",
    ]
    df = pd.read_csv(path, usecols=usecols)
    if validation_game_ids is not None:
        df = df[df["PGGEXIT_gameId"].astype(str).isin(validation_game_ids)].copy()
    if df["PGGEXIT_playerId"].nunique() != len(df):
        raise ValueError("Expected one row per merged PGG participant.")

    prolific_age = df["PROLIFIC_Age"].map(parse_numeric)
    exit_age = df["PGGEXIT_data.age"].map(parse_numeric)
    df["age_numeric_best"] = prolific_age.where(prolific_age.notna(), exit_age)
    df["age_bracket_best"] = df["age_numeric_best"].map(harmonize_age_bracket)

    prolific_sex = df["PROLIFIC_Sex"].map(harmonize_prolific_sex)
    exit_sex = df["PGGEXIT_data.gender"].map(harmonize_exit_gender)
    df["sex_best"] = prolific_sex.where(prolific_sex.notna(), exit_sex)

    df["education_completed"] = df["PGGEXIT_data.education"].map(harmonize_pgg_education)
    df["race_ethnicity_harmonized"] = df["PROLIFIC_Ethnicity simplified"].map(harmonize_pgg_race_ethnicity)
    df["employment_harmonized"] = [
        harmonize_pgg_employment(emp, stu)
        for emp, stu in zip(df["PROLIFIC_Employment status"], df["PROLIFIC_Student status"])
    ]
    df["us_nationality_proxy"] = df["PROLIFIC_Nationality"].map(is_us_country)
    df["us_residence_proxy"] = df["PROLIFIC_Country of residence"].map(is_us_country)
    df["us_birth_proxy"] = df["PROLIFIC_Country of birth"].map(is_us_country)
    return df


def load_twin(path: Path) -> pd.DataFrame:
    usecols = ["pid", "QID11", "QID12", "QID13", "QID14", "QID15", "QID16", "QID24"]
    df = pd.read_csv(path, usecols=usecols)
    if df["pid"].nunique() != len(df):
        raise ValueError("Expected one row per Twin participant.")
    df["age_bracket"] = df["QID13"].map(TWIN_AGE_MAP)
    df["sex_male_female"] = df["QID12"].map(TWIN_SEX_MAP)
    df["education_completed"] = df["QID14"].map(TWIN_EDU_MAP)
    df["race_ethnicity_harmonized"] = df["QID15"].map(TWIN_RACE_MAP)
    df["employment_harmonized"] = df["QID24"].map(TWIN_EMPLOYMENT_MAP)
    df["us_nationality_proxy"] = df["QID16"].map({1: "yes", 2: "no"})
    df["us_residence_proxy"] = np.where(df["QID11"].notna(), "yes", None)
    return df


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
    out["pgg_pct"] = (out["pgg_count"] / out["pgg_count"].sum() * 100).round(4)
    out["twin_pct"] = (out["twin_count"] / out["twin_count"].sum() * 100).round(4)
    out["pct_point_diff_pgg_minus_twin"] = (out["pgg_pct"] - out["twin_pct"]).round(4)
    return out


def frame_for_dimension(
    dimension: str,
    categories: list[str],
    pgg_series: pd.Series,
    twin_series: pd.Series,
) -> pd.DataFrame:
    pgg_counts = pgg_series.value_counts().reindex(categories, fill_value=0)
    twin_counts = twin_series.value_counts().reindex(categories, fill_value=0)
    return make_distribution_frame(dimension, categories, pgg_counts, twin_counts)


def total_variation_distance(frame: pd.DataFrame) -> float:
    p = frame["pgg_pct"].to_numpy(dtype=float) / 100.0
    q = frame["twin_pct"].to_numpy(dtype=float) / 100.0
    return float(0.5 * np.abs(p - q).sum())


def make_divergence_frame(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for frame in frames:
        rows.append(
            {
                "dimension": str(frame["dimension"].iloc[0]),
                "pgg_n_used": int(frame["pgg_count"].sum()),
                "twin_n_used": int(frame["twin_count"].sum()),
                "total_variation_distance": round(total_variation_distance(frame), 8),
                "total_variation_distance_pct_points": round(total_variation_distance(frame) * 100.0, 4),
            }
        )
    return pd.DataFrame(rows)


def chi_square_for_frame(frame: pd.DataFrame, notes: str) -> dict[str, Any]:
    contingency = frame.loc[(frame["pgg_count"] + frame["twin_count"]) > 0, ["pgg_count", "twin_count"]].to_numpy()
    chi2, p_value, dof, _ = chi2_contingency(contingency)
    return {
        "dimension": str(frame["dimension"].iloc[0]),
        "test": "chi2",
        "statistic": round(float(chi2), 6),
        "degrees_of_freedom": int(dof),
        "p_value": float(p_value),
        "notes": notes,
    }


def plot_direct_overlap(
    frames: list[pd.DataFrame],
    output_path: Path,
    pgg_label: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes_flat = axes.flatten()
    ymax = max(frame[["pgg_pct", "twin_pct"]].to_numpy().max() for frame in frames) * 1.18
    handles = None

    for ax, frame in zip(axes_flat, frames):
        x = np.arange(len(frame))
        width = 0.38
        pgg_bar = ax.bar(x - width / 2, frame["pgg_pct"], width=width, label=pgg_label)
        twin_bar = ax.bar(x + width / 2, frame["twin_pct"], width=width, label="Twin-2k")
        ax.set_xticks(x)
        display_labels = [PLOT_LABEL_MAP.get(cat, cat) for cat in frame["category"]]
        ax.set_xticklabels(display_labels, rotation=20, ha="right")
        dim = str(frame["dimension"].iloc[0])
        ax.set_title(f"{DIMENSION_LABELS.get(dim, dim)}\nTVD = {total_variation_distance(frame) * 100:.1f} pp")
        ax.set_ylim(0, ymax)
        if handles is None:
            handles = [pgg_bar[0], twin_bar[0]]

    axes_flat[0].set_ylabel("Percent of participants")
    axes_flat[3].set_ylabel("Percent of participants")
    note_ax = axes_flat[-1]
    note_ax.axis("off")
    note_ax.text(
        0.0,
        0.95,
        "Notes",
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    note_ax.text(
        0.0,
        0.82,
        (
            "Age and sex use the best available PGG value:\n"
            "Prolific profile first, then player-input fallback.\n\n"
            "Race/ethnicity is coarsened to white/black/asian/other-mixed.\n"
            "Twin Hispanic is folded into other/mixed because the Prolific simplified field has no direct Hispanic bucket.\n\n"
            "Employment is harmonized from Twin employment status vs Prolific employment + student status."
        ),
        fontsize=9,
        va="top",
    )

    fig.suptitle(title, y=0.98, fontsize=15)
    fig.legend(handles, [pgg_label, "Twin-2k"], loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.12, wspace=0.22, hspace=0.45)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_extended_comparison(
    frames: list[pd.DataFrame],
    output_path: Path,
    pgg_label: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes_flat = axes.flatten()
    ymax = max(frame[["pgg_pct", "twin_pct"]].to_numpy().max() for frame in frames) * 1.18
    handles = None

    for ax, frame in zip(axes_flat, frames):
        x = np.arange(len(frame))
        width = 0.38
        pgg_bar = ax.bar(x - width / 2, frame["pgg_pct"], width=width, label=pgg_label)
        twin_bar = ax.bar(x + width / 2, frame["twin_pct"], width=width, label="Twin-2k")
        ax.set_xticks(x)
        display_labels = [PLOT_LABEL_MAP.get(cat, cat) for cat in frame["category"]]
        ax.set_xticklabels(display_labels, rotation=20, ha="right")
        dim = str(frame["dimension"].iloc[0])
        ax.set_title(f"{DIMENSION_LABELS.get(dim, dim)}\nTVD = {total_variation_distance(frame) * 100:.1f} pp")
        ax.set_ylim(0, ymax)
        if handles is None:
            handles = [pgg_bar[0], twin_bar[0]]

    axes_flat[0].set_ylabel("Percent of participants")
    axes_flat[4].set_ylabel("Percent of participants")
    note_ax = axes_flat[-1]
    note_ax.axis("off")
    note_ax.text(0.0, 0.95, "Notes", fontsize=12, fontweight="bold", va="top")
    note_ax.text(
        0.0,
        0.82,
        (
            "Age and sex use the best available PGG value:\n"
            "Prolific profile first, then player-input fallback.\n\n"
            "Nationality and residence are proxy comparisons only:\n"
            "Twin has U.S. citizenship and U.S. region, not full country fields.\n\n"
            "Race/ethnicity and employment are coarsened to shared categories."
        ),
        fontsize=9,
        va="top",
    )

    fig.suptitle(title, y=0.98, fontsize=15)
    fig.legend(handles, [pgg_label, "Twin-2k"], loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.12, wspace=0.24, hspace=0.46)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_country_summary(df: pd.DataFrame, column: str, top_n: int = 15) -> pd.DataFrame:
    series = df[column].map(normalized_string).dropna()
    counts = series.value_counts().head(top_n)
    total = int(series.shape[0])
    return pd.DataFrame(
        {
            "field": column,
            "category": counts.index,
            "count": counts.values.astype(int),
            "pct": np.round(counts.values / total * 100.0, 4),
            "n_used": total,
        }
    )


def make_value_distribution(series: pd.Series, field: str, dataset: str) -> pd.DataFrame:
    counts = series.value_counts(dropna=False)
    total = int(len(series))
    rows = []
    for value, count in counts.items():
        if pd.isna(value):
            label = "<missing>"
        else:
            label = str(value)
        rows.append(
            {
                "dataset": dataset,
                "field": field,
                "value": label,
                "count": int(count),
                "pct_of_rows": round(int(count) / total * 100.0, 4),
                "n_rows": total,
            }
        )
    return pd.DataFrame(rows)


def make_summary_frame(pgg: pd.DataFrame, twin: pd.DataFrame) -> pd.DataFrame:
    learn_path = PROJECT_ROOT / "data" / "processed_data" / "df_analysis_learn.csv"
    val_path = PROJECT_ROOT / "data" / "processed_data" / "df_analysis_val.csv"
    pgg_games = set(pgg["PGGEXIT_gameId"].astype(str))

    learn_overlap = 0
    learn_total = 0
    if learn_path.exists():
        learn_games = set(pd.read_csv(learn_path, usecols=["gameId"])["gameId"].astype(str))
        learn_total = len(learn_games)
        learn_overlap = len(pgg_games & learn_games)

    val_games = set(pd.read_csv(val_path, usecols=["gameId"])["gameId"].astype(str))
    val_total = len(val_games)
    val_overlap = len(pgg_games & val_games)

    twin_age_mid = twin["QID13"].map(TWIN_AGE_MIDPOINT_MAP)
    pgg_age_mid = pgg["age_numeric_best"].dropna()
    rows = [
        {"metric": "pgg_extended_rows", "value": int(len(pgg)), "notes": "One row per merged PGG participant/session."},
        {"metric": "twin_rows", "value": int(len(twin)), "notes": ""},
        {"metric": "pgg_extended_unique_games", "value": int(pgg["PGGEXIT_gameId"].nunique()), "notes": ""},
        {
            "metric": "pgg_extended_validation_games_covered",
            "value": int(val_overlap),
            "notes": f"Out of {val_total} validation games in df_analysis_val.csv.",
        },
        {
            "metric": "pgg_extended_learning_games_covered",
            "value": int(learn_overlap),
            "notes": f"Out of {learn_total} learning games in df_analysis_learn.csv.",
        },
        {"metric": "pgg_extended_mean_age_best", "value": round(float(pgg_age_mid.mean()), 4), "notes": "Best available: Prolific profile first, player-input fallback."},
        {"metric": "twin_mean_age_midpoint", "value": round(float(twin_age_mid.mean()), 4), "notes": "Approximate: Twin age is reported in brackets and mapped to bracket midpoints."},
        {
            "metric": "pgg_extended_us_residence_pct",
            "value": round(float((pgg["us_residence_proxy"] == "yes").mean() * 100), 4),
            "notes": "Among rows with non-missing country of residence.",
        },
        {
            "metric": "pgg_extended_us_nationality_pct",
            "value": round(float((pgg["us_nationality_proxy"] == "yes").mean() * 100), 4),
            "notes": "Among rows with non-missing nationality.",
        },
        {
            "metric": "pgg_extended_us_birth_pct",
            "value": round(float((pgg["us_birth_proxy"] == "yes").mean() * 100), 4),
            "notes": "Among rows with non-missing country of birth.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_game_ids = None
    if args.scope == "validation_only":
        validation_game_ids = load_validation_game_ids(args.validation_analysis_csv)

    pgg = load_pgg_extended(args.pgg_extended_csv, validation_game_ids=validation_game_ids)
    twin = load_twin(args.twin_csv)

    age_df = frame_for_dimension(
        "age",
        AGE_ORDER,
        pgg["age_bracket_best"].dropna(),
        twin["age_bracket"].dropna(),
    )
    sex_df = frame_for_dimension(
        "sex_male_female",
        SEX_ORDER,
        pgg["sex_best"].dropna(),
        twin["sex_male_female"].dropna(),
    )
    education_df = frame_for_dimension(
        "education_completed",
        EDUCATION_ORDER,
        pgg["education_completed"].dropna(),
        twin["education_completed"].dropna(),
    )
    race_df = frame_for_dimension(
        "race_ethnicity_harmonized",
        RACE_ORDER,
        pgg["race_ethnicity_harmonized"].dropna(),
        twin["race_ethnicity_harmonized"].dropna(),
    )
    employment_df = frame_for_dimension(
        "employment_harmonized",
        EMPLOYMENT_ORDER,
        pgg["employment_harmonized"].dropna(),
        twin["employment_harmonized"].dropna(),
    )

    direct_frames = [age_df, sex_df, education_df, race_df, employment_df]
    direct_distribution_df = pd.concat(direct_frames, ignore_index=True)
    direct_divergence_df = make_divergence_frame(direct_frames)
    direct_tests_df = pd.DataFrame(
        [
            chi_square_for_frame(age_df, "Age brackets: PGG best available age vs Twin age bracket."),
            chi_square_for_frame(sex_df, "Male/female only. PGG uses Prolific sex first, then player-input gender fallback."),
            chi_square_for_frame(education_df, "Completed-degree harmonization: Twin sub-bachelor levels collapsed to high school vs college/postsecondary."),
            chi_square_for_frame(race_df, "Coarse race/ethnicity harmonization. Twin Hispanic is folded into other/mixed because Prolific simplified ethnicity lacks a Hispanic bucket."),
            chi_square_for_frame(employment_df, "Employment harmonization uses Twin employment status vs Prolific employment plus student status."),
        ]
    )

    proxy_frames = [
        frame_for_dimension(
            "us_nationality_proxy",
            US_BINARY_ORDER,
            pgg["us_nationality_proxy"].dropna(),
            twin["us_nationality_proxy"].dropna(),
        ),
        frame_for_dimension(
            "us_residence_proxy",
            US_BINARY_ORDER,
            pgg["us_residence_proxy"].dropna(),
            twin["us_residence_proxy"].dropna(),
        ),
    ]
    proxy_distribution_df = pd.concat(proxy_frames, ignore_index=True)
    proxy_divergence_df = make_divergence_frame(proxy_frames)
    proxy_tests_df = pd.DataFrame(
        [
            chi_square_for_frame(
                proxy_frames[0],
                "Proxy only: Twin U.S. citizenship (QID16) vs PGG U.S. nationality derived from Prolific nationality.",
            ),
            chi_square_for_frame(
                proxy_frames[1],
                "Proxy only: Twin current U.S. residence implied by region question (QID11) vs PGG country of residence.",
            ),
        ]
    )

    country_summary_df = pd.concat(
        [
            make_country_summary(pgg, "PROLIFIC_Country of birth"),
            make_country_summary(pgg, "PROLIFIC_Country of residence"),
            make_country_summary(pgg, "PROLIFIC_Nationality"),
        ],
        ignore_index=True,
    )
    pgg_value_distributions_df = pd.concat(
        [
            make_value_distribution(pgg["PROLIFIC_Age"], "PROLIFIC_Age", "pgg_extended"),
            make_value_distribution(pgg["PROLIFIC_Sex"], "PROLIFIC_Sex", "pgg_extended"),
            make_value_distribution(
                pgg["PROLIFIC_Ethnicity simplified"],
                "PROLIFIC_Ethnicity simplified",
                "pgg_extended",
            ),
            make_value_distribution(
                pgg["PROLIFIC_Country of birth"],
                "PROLIFIC_Country of birth",
                "pgg_extended",
            ),
            make_value_distribution(
                pgg["PROLIFIC_Country of residence"],
                "PROLIFIC_Country of residence",
                "pgg_extended",
            ),
            make_value_distribution(pgg["PROLIFIC_Nationality"], "PROLIFIC_Nationality", "pgg_extended"),
            make_value_distribution(
                pgg["PROLIFIC_Employment status"],
                "PROLIFIC_Employment status",
                "pgg_extended",
            ),
            make_value_distribution(
                pgg["PROLIFIC_Student status"],
                "PROLIFIC_Student status",
                "pgg_extended",
            ),
        ],
        ignore_index=True,
    )
    twin_value_distributions_df = pd.concat(
        [
            make_value_distribution(
                twin["QID11"].map({1: "Northeast", 2: "Midwest", 3: "South", 4: "West", 5: "Pacific"}),
                "QID11_region",
                "twin",
            ),
            make_value_distribution(twin["QID12"].map(TWIN_SEX_MAP), "QID12_sex_assigned_at_birth", "twin"),
            make_value_distribution(twin["QID13"].map(TWIN_AGE_MAP), "QID13_age", "twin"),
            make_value_distribution(
                twin["QID14"].map(
                    {
                        1: "Less than high school",
                        2: "High school graduate",
                        3: "Some college, no degree",
                        4: "Associate's degree",
                        5: "College graduate/some postgrad",
                        6: "Postgraduate",
                    }
                ),
                "QID14_education",
                "twin",
            ),
            make_value_distribution(
                twin["QID15"].map({1: "White", 2: "Black", 3: "Asian", 4: "Hispanic", 5: "Other"}),
                "QID15_race_or_origin",
                "twin",
            ),
            make_value_distribution(twin["QID16"].map({1: "Yes", 2: "No"}), "QID16_us_citizen", "twin"),
            make_value_distribution(
                twin["QID24"].map(
                    {
                        1: "Full-time employment",
                        2: "Part-time employment",
                        3: "Unemployed",
                        4: "Self-employed",
                        5: "Home-maker",
                        6: "Student",
                        7: "Retired",
                    }
                ),
                "QID24_employment_status",
                "twin",
            ),
        ],
        ignore_index=True,
    )
    summary_df = make_summary_frame(pgg, twin)

    direct_distribution_df.to_csv(output_dir / "direct_overlap_distribution_comparison.csv", index=False)
    direct_divergence_df.to_csv(output_dir / "direct_overlap_divergence_metrics.csv", index=False)
    direct_tests_df.to_csv(output_dir / "direct_overlap_statistical_tests.csv", index=False)
    proxy_distribution_df.to_csv(output_dir / "proxy_us_alignment_comparison.csv", index=False)
    proxy_divergence_df.to_csv(output_dir / "proxy_us_alignment_divergence.csv", index=False)
    proxy_tests_df.to_csv(output_dir / "proxy_us_alignment_tests.csv", index=False)
    country_summary_df.to_csv(output_dir / "pgg_country_field_summary.csv", index=False)
    pgg_value_distributions_df.to_csv(output_dir / "pgg_extended_raw_value_distributions.csv", index=False)
    twin_value_distributions_df.to_csv(output_dir / "twin_raw_value_distributions.csv", index=False)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)

    pgg_label = "PGG validation" if args.scope == "validation_only" else "PGG extended"
    direct_title = (
        "Twin-2k vs PGG Validation Demographic Distributions"
        if args.scope == "validation_only"
        else "Twin-2k vs PGG Extended Demographic Distributions"
    )
    extended_title = (
        "Twin-2k vs PGG Validation Demographic Comparison (Extended)"
        if args.scope == "validation_only"
        else "Twin-2k vs PGG Extended Demographic Comparison (Extended)"
    )

    plot_direct_overlap(
        direct_frames,
        output_dir / "combined_direct_overlap_comparison.png",
        pgg_label=pgg_label,
        title=direct_title,
    )
    plot_extended_comparison(
        direct_frames + proxy_frames,
        output_dir / "combined_extended_comparison.png",
        pgg_label=pgg_label,
        title=extended_title,
    )

    method_notes = {
        "scope": args.scope,
        "pgg_extended_fields": {
            "age": "PROLIFIC_Age first, then PGGEXIT_data.age fallback",
            "sex": "PROLIFIC_Sex first, then PGGEXIT_data.gender fallback (male/female only)",
            "education": "PGGEXIT_data.education",
            "race_ethnicity": "PROLIFIC_Ethnicity simplified",
            "employment": "PROLIFIC_Employment status plus PROLIFIC_Student status",
            "country_of_birth": "PROLIFIC_Country of birth",
            "country_of_residence": "PROLIFIC_Country of residence",
            "nationality": "PROLIFIC_Nationality",
        },
        "twin_fields": {
            "age": "QID13",
            "sex": "QID12",
            "education": "QID14",
            "race_ethnicity": "QID15",
            "employment": "QID24",
            "us_citizenship_proxy": "QID16",
            "us_residence_proxy": "QID11",
        },
        "notes": [
            (
                "The merged PGG file is filtered to validation-wave games only."
                if args.scope == "validation_only"
                else "The merged PGG file covers all validation games and only a subset of learning games."
            ),
            "Country of birth/residence/nationality do not have direct Twin counterparts, so they are summarized separately and only linked via limited U.S. proxy comparisons.",
            "Twin Hispanic is folded into other/mixed in the race/ethnicity comparison because the Prolific simplified ethnicity field has no direct Hispanic category.",
        ],
    }
    (output_dir / "method_notes.json").write_text(
        json.dumps(method_notes, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
