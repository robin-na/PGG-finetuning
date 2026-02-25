#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway


DEFAULT_CLUSTER_ROOT = Path("Persona/misc/tag_section_clusters_openai")
DEFAULT_DEMOGRAPHICS_CSV = Path("demographics/demographics_numeric_learn.csv")
DEFAULT_OUTPUT_DIR = Path("demographics/analysis_learning_vs_clusters")

GENDER_LABELS = {
    0: "unknown",
    1: "man",
    2: "woman",
    3: "non_binary",
}

EDUCATION_LABELS = {
    0: "unknown",
    1: "high_school",
    2: "bachelor",
    3: "master",
    4: "other",
}


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    """
    Benjamini-Hochberg FDR adjustment.
    """
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


def cramers_v(confusion: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(confusion)
    n = confusion.to_numpy().sum()
    r, k = confusion.shape
    denom = n * (min(r, k) - 1)
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(chi2 / denom))


def load_cluster_rows(cluster_root: Path) -> pd.DataFrame:
    files = sorted(cluster_root.glob("*/*_clustered.jsonl"))
    if not files:
        raise FileNotFoundError(f"No *_clustered.jsonl files found under: {cluster_root}")

    rows: List[Dict] = []
    for path in files:
        tag = path.parent.name
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append(
                    {
                        "tag": tag,
                        "gameId": str(obj.get("experiment")),
                        "playerId": str(obj.get("participant")),
                        "cluster_id": int(obj.get("cluster_id")),
                        "cluster_title": str(obj.get("cluster_title") or f"cluster_{int(obj.get('cluster_id'))}"),
                    }
                )

    df = pd.DataFrame(rows)
    # Defensive: ensure one row per tag/game/player.
    df = df.drop_duplicates(subset=["tag", "gameId", "playerId"], keep="last").reset_index(drop=True)
    return df


def build_coverage_table(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tag, sub in merged.groupby("tag", sort=True):
        n_total = int(len(sub))
        n_matched = int((sub["_merge"] == "both").sum())
        rows.append(
            {
                "tag": tag,
                "n_cluster_rows": n_total,
                "n_matched_learning_demographics": n_matched,
                "n_unmatched": n_total - n_matched,
                "match_rate": n_matched / n_total if n_total else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("tag").reset_index(drop=True)


def run_association_tests(merged: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    for tag, sub in merged.groupby("tag", sort=True):
        sub = sub[sub["_merge"] == "both"].copy()
        if sub.empty:
            continue

        # Age (numeric): one-way ANOVA cluster_id -> age
        age_sub = sub[sub["age"].notna()].copy()
        if age_sub["cluster_id"].nunique() > 1:
            groups = [g["age"].to_numpy(dtype=float) for _, g in age_sub.groupby("cluster_id") if len(g) > 1]
            if len(groups) >= 2:
                stat, pval = f_oneway(*groups)
                grand_mean = float(age_sub["age"].mean())
                cluster_stats = age_sub.groupby("cluster_id")["age"].agg(["mean", "count"])
                ss_between = float((cluster_stats["count"] * (cluster_stats["mean"] - grand_mean) ** 2).sum())
                ss_total = float(((age_sub["age"] - grand_mean) ** 2).sum())
                eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
                rows.append(
                    {
                        "tag": tag,
                        "variable": "age",
                        "test": "anova",
                        "n_used": int(len(age_sub)),
                        "n_clusters": int(age_sub["cluster_id"].nunique()),
                        "n_levels": np.nan,
                        "statistic": float(stat),
                        "p_value": float(pval),
                        "effect_size": float(eta_sq),
                        "effect_name": "eta_squared",
                    }
                )

        # Gender (categorical): chi-square on known values only.
        g_sub = sub[sub["gender_code"] != 0].copy()
        if not g_sub.empty:
            conf = pd.crosstab(g_sub["cluster_id"], g_sub["gender_code"])
            if conf.shape[0] > 1 and conf.shape[1] > 1:
                chi2, pval, _, _ = chi2_contingency(conf)
                rows.append(
                    {
                        "tag": tag,
                        "variable": "gender_code",
                        "test": "chi2",
                        "n_used": int(len(g_sub)),
                        "n_clusters": int(conf.shape[0]),
                        "n_levels": int(conf.shape[1]),
                        "statistic": float(chi2),
                        "p_value": float(pval),
                        "effect_size": cramers_v(conf),
                        "effect_name": "cramers_v",
                    }
                )

        # Education (categorical): chi-square on known values only.
        e_sub = sub[sub["education_code"] != 0].copy()
        if not e_sub.empty:
            conf = pd.crosstab(e_sub["cluster_id"], e_sub["education_code"])
            if conf.shape[0] > 1 and conf.shape[1] > 1:
                chi2, pval, _, _ = chi2_contingency(conf)
                rows.append(
                    {
                        "tag": tag,
                        "variable": "education_code",
                        "test": "chi2",
                        "n_used": int(len(e_sub)),
                        "n_clusters": int(conf.shape[0]),
                        "n_levels": int(conf.shape[1]),
                        "statistic": float(chi2),
                        "p_value": float(pval),
                        "effect_size": cramers_v(conf),
                        "effect_name": "cramers_v",
                    }
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr_bh"] = benjamini_hochberg(out["p_value"])
        out["significant_0_05"] = out["p_value"] < 0.05
        out["significant_fdr_0_05"] = out["p_fdr_bh"] < 0.05
        out = out.sort_values(["p_value", "effect_size"], ascending=[True, False]).reset_index(drop=True)
    return out


def build_cluster_profiles(merged: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    matched = merged[merged["_merge"] == "both"].copy()

    for (tag, cluster_id), sub in matched.groupby(["tag", "cluster_id"], sort=True):
        row = {
            "tag": tag,
            "cluster_id": int(cluster_id),
            "n_players": int(len(sub)),
            "age_mean": float(sub["age"].mean()) if sub["age"].notna().any() else np.nan,
            "age_std": float(sub["age"].std(ddof=1)) if sub["age"].notna().sum() >= 2 else np.nan,
            "age_missing_rate": float(sub["age_missing"].mean()),
        }

        for code, label in GENDER_LABELS.items():
            row[f"share_gender_{label}"] = float((sub["gender_code"] == code).mean())
        for code, label in EDUCATION_LABELS.items():
            row[f"share_education_{label}"] = float((sub["education_code"] == code).mean())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["tag", "cluster_id"]).reset_index(drop=True)


def build_age_distribution_shift(merged: pd.DataFrame, tests_df: pd.DataFrame) -> pd.DataFrame:
    """
    For tags with FDR-significant age association, compare
    oldest quartile vs youngest quartile cluster distributions.
    """
    if tests_df.empty:
        return pd.DataFrame()

    sig_tags = tests_df[
        (tests_df["variable"] == "age") & (tests_df["significant_fdr_0_05"])
    ]["tag"].tolist()
    rows: List[Dict] = []

    for tag in sig_tags:
        sub = merged[
            (merged["tag"] == tag)
            & (merged["_merge"] == "both")
            & (merged["age"].notna())
        ].copy()
        if sub.empty:
            continue

        q1_age = float(sub["age"].quantile(0.25))
        q3_age = float(sub["age"].quantile(0.75))
        young = sub[sub["age"] <= q1_age]
        old = sub[sub["age"] >= q3_age]
        if young.empty or old.empty:
            continue

        young_dist = young["cluster_id"].value_counts(normalize=True)
        old_dist = old["cluster_id"].value_counts(normalize=True)
        all_cluster_ids = sorted(set(young_dist.index).union(set(old_dist.index)))

        for cluster_id in all_cluster_ids:
            cluster_title = sub.loc[sub["cluster_id"] == cluster_id, "cluster_title"].iloc[0]
            share_young = float(young_dist.get(cluster_id, 0.0))
            share_old = float(old_dist.get(cluster_id, 0.0))
            rows.append(
                {
                    "tag": tag,
                    "q1_age": q1_age,
                    "q3_age": q3_age,
                    "cluster_id": int(cluster_id),
                    "cluster_title": cluster_title,
                    "share_young_q1": share_young,
                    "share_old_q4": share_old,
                    "old_minus_young_pp": 100.0 * (share_old - share_young),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["tag", "old_minus_young_pp"], ascending=[True, False]
    ).reset_index(drop=True)


def build_education_distribution_shift(merged: pd.DataFrame, tests_df: pd.DataFrame) -> pd.DataFrame:
    """
    For tags with FDR-significant education association, compute
    per-education cluster share shift vs overall tag distribution.
    """
    if tests_df.empty:
        return pd.DataFrame()

    sig_tags = tests_df[
        (tests_df["variable"] == "education_code") & (tests_df["significant_fdr_0_05"])
    ]["tag"].tolist()
    rows: List[Dict] = []
    level_map = {
        1: "high_school",
        2: "bachelor",
        3: "master",
        4: "other",
    }

    for tag in sig_tags:
        sub = merged[
            (merged["tag"] == tag)
            & (merged["_merge"] == "both")
            & (merged["education_code"] != 0)
        ].copy()
        if sub.empty:
            continue

        base_dist = sub["cluster_id"].value_counts(normalize=True)
        for code, label in level_map.items():
            grp = sub[sub["education_code"] == code]
            if grp.empty:
                continue
            grp_dist = grp["cluster_id"].value_counts(normalize=True)
            all_cluster_ids = sorted(set(base_dist.index).union(set(grp_dist.index)))

            for cluster_id in all_cluster_ids:
                cluster_title = sub.loc[sub["cluster_id"] == cluster_id, "cluster_title"].iloc[0]
                share_level = float(grp_dist.get(cluster_id, 0.0))
                share_base = float(base_dist.get(cluster_id, 0.0))
                rows.append(
                    {
                        "tag": tag,
                        "education_level": label,
                        "n_level": int(len(grp)),
                        "cluster_id": int(cluster_id),
                        "cluster_title": cluster_title,
                        "share_level": share_level,
                        "share_overall_known_education": share_base,
                        "level_minus_overall_pp": 100.0 * (share_level - share_base),
                    }
                )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["tag", "education_level", "level_minus_overall_pp"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze relationship between demographics and tag-section clusters.")
    parser.add_argument("--cluster-root", type=Path, default=DEFAULT_CLUSTER_ROOT)
    parser.add_argument("--demographics-csv", type=Path, default=DEFAULT_DEMOGRAPHICS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    demographic_df = pd.read_csv(args.demographics_csv)
    clusters_df = load_cluster_rows(args.cluster_root)

    merged = clusters_df.merge(demographic_df, how="left", on=["gameId", "playerId"], indicator=True)

    coverage_df = build_coverage_table(merged)
    tests_df = run_association_tests(merged)
    profiles_df = build_cluster_profiles(merged)
    age_shift_df = build_age_distribution_shift(merged, tests_df)
    education_shift_df = build_education_distribution_shift(merged, tests_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = args.output_dir / "coverage_by_tag.csv"
    tests_path = args.output_dir / "associations_by_tag.csv"
    profiles_path = args.output_dir / "cluster_demographic_profiles_by_tag.csv"
    age_shift_path = args.output_dir / "age_distribution_shift_old_vs_young_by_tag.csv"
    education_shift_path = args.output_dir / "education_distribution_shift_by_tag.csv"

    coverage_df.to_csv(coverage_path, index=False)
    tests_df.to_csv(tests_path, index=False)
    profiles_df.to_csv(profiles_path, index=False)
    age_shift_df.to_csv(age_shift_path, index=False)
    education_shift_df.to_csv(education_shift_path, index=False)

    print(f"Saved: {coverage_path}")
    print(f"Saved: {tests_path}")
    print(f"Saved: {profiles_path}")
    print(f"Saved: {age_shift_path}")
    print(f"Saved: {education_shift_path}")


if __name__ == "__main__":
    main()
