#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DEFAULT_DEMOGRAPHICS_CSV = THIS_DIR / "demographics_numeric_val.csv"
DEFAULT_ANALYSIS_CSV = PROJECT_ROOT / "data" / "processed_data" / "df_analysis_val.csv"
DEFAULT_OUTPUT_DIR = THIS_DIR / "analysis_validation_randomization"

GENDER_CATEGORIES = ["man", "woman", "non_binary", "unknown"]
GENDER_KNOWN = ["man", "woman", "non_binary"]
EDUCATION_CATEGORIES = ["high_school", "bachelor", "master", "other", "unknown"]
EDUCATION_KNOWN = ["high_school", "bachelor", "master", "other"]

EXCLUDED_CONFIG_COLUMNS = {
    "CONFIG_treatmentName",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether validation-wave demographics are randomly distributed "
            "across games and config assignments."
        )
    )
    parser.add_argument("--demographics-csv", type=Path, default=DEFAULT_DEMOGRAPHICS_CSV)
    parser.add_argument("--analysis-csv", type=Path, default=DEFAULT_ANALYSIS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument(
        "--eligible-only",
        action="store_true",
        help="Restrict to games with valid_number_of_starting_players == True.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    arr = pvals.to_numpy(dtype=float)
    n = len(arr)
    order = np.argsort(arr)
    ranked = arr[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adjusted[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return pd.Series(out, index=pvals.index)


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "t", "yes"})
    )


def pick_one_hot_category(df: pd.DataFrame, prefix: str, categories: list[str]) -> pd.Series:
    arr = df[[f"{prefix}_{category}" for category in categories]].to_numpy()
    idx = arr.argmax(axis=1)
    selected = []
    for i, row in enumerate(arr):
        if row.sum() == 0:
            selected.append("missing")
        else:
            selected.append(categories[idx[i]])
    return pd.Series(selected, index=df.index)


def eta_squared(group_codes: np.ndarray, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    group_codes = np.asarray(group_codes, dtype=int)
    grand_mean = values.mean()
    ss_total = float(((values - grand_mean) ** 2).sum())
    if ss_total <= 0:
        return float("nan")

    ss_between = 0.0
    for code in np.unique(group_codes):
        group_values = values[group_codes == code]
        if len(group_values) == 0:
            continue
        ss_between += len(group_values) * float((group_values.mean() - grand_mean) ** 2)
    return ss_between / ss_total


def cramers_v_from_codes(group_codes: np.ndarray, cat_codes: np.ndarray) -> float:
    group_codes = np.asarray(group_codes, dtype=int)
    cat_codes = np.asarray(cat_codes, dtype=int)
    n_groups = int(group_codes.max()) + 1
    n_cats = int(cat_codes.max()) + 1
    table = np.zeros((n_groups, n_cats), dtype=int)
    np.add.at(table, (group_codes, cat_codes), 1)
    chi2, _, _, _ = chi2_contingency(table)
    n = int(table.sum())
    denom = n * (min(table.shape) - 1)
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(chi2 / denom))


def permutation_p_value(observed: float, null_values: Iterable[float]) -> float:
    null_arr = np.asarray(list(null_values), dtype=float)
    return float((1 + (null_arr >= observed).sum()) / (len(null_arr) + 1))


def run_game_level_permutations(
    merged: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    unique_games = pd.Index(sorted(merged["gameId"].astype(str).unique()))
    game_codes_all = pd.Categorical(merged["gameId"].astype(str), categories=unique_games).codes
    rows = []

    age_mask = merged["age_missing"] == 0
    age_values = merged.loc[age_mask, "age"].to_numpy(dtype=float)
    age_game_codes = game_codes_all[age_mask]
    age_observed = eta_squared(age_game_codes, age_values)
    age_null = [
        eta_squared(age_game_codes, rng.permutation(age_values))
        for _ in range(n_permutations)
    ]
    rows.append(
        {
            "dimension": "age",
            "metric": "eta_squared",
            "observed_statistic": age_observed,
            "null_mean": float(np.mean(age_null)),
            "null_q025": float(np.quantile(age_null, 0.025)),
            "null_q975": float(np.quantile(age_null, 0.975)),
            "permutation_p_value": permutation_p_value(age_observed, age_null),
            "n_used": int(age_mask.sum()),
            "n_groups": int(len(unique_games)),
            "n_permutations": int(n_permutations),
        }
    )

    gender_mask = merged["gender_cat"].isin(GENDER_KNOWN)
    gender_codes = pd.Categorical(
        merged.loc[gender_mask, "gender_cat"],
        categories=GENDER_KNOWN,
    ).codes
    gender_game_codes = game_codes_all[gender_mask]
    gender_observed = cramers_v_from_codes(gender_game_codes, gender_codes)
    gender_null = [
        cramers_v_from_codes(gender_game_codes, rng.permutation(gender_codes))
        for _ in range(n_permutations)
    ]
    rows.append(
        {
            "dimension": "gender",
            "metric": "cramers_v",
            "observed_statistic": gender_observed,
            "null_mean": float(np.mean(gender_null)),
            "null_q025": float(np.quantile(gender_null, 0.025)),
            "null_q975": float(np.quantile(gender_null, 0.975)),
            "permutation_p_value": permutation_p_value(gender_observed, gender_null),
            "n_used": int(gender_mask.sum()),
            "n_groups": int(len(unique_games)),
            "n_permutations": int(n_permutations),
        }
    )

    education_mask = merged["education_cat"].isin(EDUCATION_KNOWN)
    education_codes = pd.Categorical(
        merged.loc[education_mask, "education_cat"],
        categories=EDUCATION_KNOWN,
    ).codes
    education_game_codes = game_codes_all[education_mask]
    education_observed = cramers_v_from_codes(education_game_codes, education_codes)
    education_null = [
        cramers_v_from_codes(education_game_codes, rng.permutation(education_codes))
        for _ in range(n_permutations)
    ]
    rows.append(
        {
            "dimension": "education",
            "metric": "cramers_v",
            "observed_statistic": education_observed,
            "null_mean": float(np.mean(education_null)),
            "null_q025": float(np.quantile(education_null, 0.025)),
            "null_q975": float(np.quantile(education_null, 0.975)),
            "permutation_p_value": permutation_p_value(education_observed, education_null),
            "n_used": int(education_mask.sum()),
            "n_groups": int(len(unique_games)),
            "n_permutations": int(n_permutations),
        }
    )

    return pd.DataFrame(rows)


def config_columns(merged: pd.DataFrame) -> list[str]:
    cols = []
    for col in merged.columns:
        if not col.startswith("CONFIG_"):
            continue
        if col in EXCLUDED_CONFIG_COLUMNS:
            continue
        if merged[col].nunique(dropna=False) <= 1:
            continue
        cols.append(col)
    return cols


def run_config_level_tests(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in config_columns(merged):
        age_sub = merged[merged["age_missing"] == 0].copy()
        if age_sub[col].nunique(dropna=True) > 1:
            groups = [grp["age"].to_numpy(dtype=float) for _, grp in age_sub.groupby(col, dropna=True)]
            if len(groups) >= 2:
                statistic, p_value = f_oneway(*groups)
                grand_mean = float(age_sub["age"].mean())
                group_stats = age_sub.groupby(col)["age"].agg(["mean", "count"])
                ss_between = float((group_stats["count"] * (group_stats["mean"] - grand_mean) ** 2).sum())
                ss_total = float(((age_sub["age"] - grand_mean) ** 2).sum())
                rows.append(
                    {
                        "config_column": col,
                        "demographic_dimension": "age",
                        "test": "anova",
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                        "effect_size": ss_between / ss_total if ss_total > 0 else float("nan"),
                        "effect_name": "eta_squared",
                        "n_levels": int(age_sub[col].nunique(dropna=True)),
                        "n_used": int(len(age_sub)),
                    }
                )

        gender_sub = merged[merged["gender_cat"].isin(GENDER_KNOWN)].copy()
        gender_table = pd.crosstab(gender_sub[col], gender_sub["gender_cat"])
        if gender_table.shape[0] > 1 and gender_table.shape[1] > 1:
            chi2, p_value, _, _ = chi2_contingency(gender_table)
            n = int(gender_table.to_numpy().sum())
            denom = n * (min(gender_table.shape) - 1)
            rows.append(
                {
                    "config_column": col,
                    "demographic_dimension": "gender",
                    "test": "chi2",
                    "statistic": float(chi2),
                    "p_value": float(p_value),
                    "effect_size": float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan"),
                    "effect_name": "cramers_v",
                    "n_levels": int(gender_table.shape[0]),
                    "n_used": n,
                }
            )

        education_sub = merged[merged["education_cat"].isin(EDUCATION_KNOWN)].copy()
        education_table = pd.crosstab(education_sub[col], education_sub["education_cat"])
        if education_table.shape[0] > 1 and education_table.shape[1] > 1:
            chi2, p_value, _, _ = chi2_contingency(education_table)
            n = int(education_table.to_numpy().sum())
            denom = n * (min(education_table.shape) - 1)
            rows.append(
                {
                    "config_column": col,
                    "demographic_dimension": "education",
                    "test": "chi2",
                    "statistic": float(chi2),
                    "p_value": float(p_value),
                    "effect_size": float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan"),
                    "effect_name": "cramers_v",
                    "n_levels": int(education_table.shape[0]),
                    "n_used": n,
                }
            )

    out = pd.DataFrame(rows).sort_values(["p_value", "effect_size"], ascending=[True, False]).reset_index(drop=True)
    if not out.empty:
        out["p_fdr_bh"] = benjamini_hochberg(out["p_value"])
        out["significant_raw_0_05"] = out["p_value"] < 0.05
        out["significant_fdr_0_05"] = out["p_fdr_bh"] < 0.05
    return out


def build_overview(merged: pd.DataFrame, game_results: pd.DataFrame, config_results: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "n_participants", "value": int(len(merged))},
        {"metric": "n_games", "value": int(merged["gameId"].nunique())},
        {"metric": "n_config_columns_tested", "value": int(config_results["config_column"].nunique())},
        {"metric": "n_config_tests_total", "value": int(len(config_results))},
        {"metric": "n_config_tests_raw_p_lt_0_05", "value": int(config_results["significant_raw_0_05"].sum())},
        {"metric": "n_config_tests_fdr_p_lt_0_05", "value": int(config_results["significant_fdr_0_05"].sum())},
    ]

    for _, row in game_results.iterrows():
        rows.append(
            {
                "metric": f"game_level_{row['dimension']}_{row['metric']}",
                "value": float(row["observed_statistic"]),
            }
        )
        rows.append(
            {
                "metric": f"game_level_{row['dimension']}_permutation_p_value",
                "value": float(row["permutation_p_value"]),
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()

    demographics = pd.read_csv(args.demographics_csv)
    analysis = pd.read_csv(args.analysis_csv, usecols=lambda c: c == "gameId" or c.startswith("CONFIG_") or c == "valid_number_of_starting_players")
    analysis = analysis.drop_duplicates(subset=["gameId"], keep="first")
    analysis["gameId"] = analysis["gameId"].astype(str)
    demographics["gameId"] = demographics["gameId"].astype(str)

    if args.eligible_only and "valid_number_of_starting_players" in analysis.columns:
        eligible_games = analysis.loc[normalize_bool(analysis["valid_number_of_starting_players"]), "gameId"]
        demographics = demographics[demographics["gameId"].isin(set(eligible_games))].copy()
        analysis = analysis[analysis["gameId"].isin(set(eligible_games))].copy()

    merged = demographics.merge(analysis, on="gameId", how="left", validate="many_to_one")
    if merged["CONFIG_configId"].isna().any():
        raise ValueError("Some participants are missing CONFIG metadata after merge.")

    merged["gender_cat"] = pick_one_hot_category(merged, "gender", GENDER_CATEGORIES)
    merged["education_cat"] = pick_one_hot_category(merged, "education", EDUCATION_CATEGORIES)

    game_results = run_game_level_permutations(merged, args.n_permutations, args.seed)
    config_results = run_config_level_tests(merged)
    overview = build_overview(merged, game_results, config_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_eligible_only" if args.eligible_only else ""
    game_results.to_csv(args.output_dir / f"game_level_permutation_tests{suffix}.csv", index=False)
    config_results.to_csv(args.output_dir / f"config_level_tests{suffix}.csv", index=False)
    overview.to_csv(args.output_dir / f"overview{suffix}.csv", index=False)

    print(f"Wrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
