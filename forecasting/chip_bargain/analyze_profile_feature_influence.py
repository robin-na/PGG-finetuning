from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "forecasting" / "chip_bargain" / "results"
METADATA_ROOT = ROOT / "forecasting" / "chip_bargain" / "metadata"
TWIN_PROFILES_PATH = (
    ROOT
    / "non-PGG_generalization"
    / "twin_profiles"
    / "output"
    / "twin_extended_profiles"
    / "twin_extended_profiles.jsonl"
)

METRICS = [
    "mean_acceptance_rate",
    "final_surplus_ratio",
    "mean_trade_ratio",
]

RAW_SOCIAL_FEATURES = [
    "trust_send_amount",
    "trust_return_share_mean",
    "ultimatum_offer_to_other",
    "ultimatum_min_acceptable_to_self",
    "ultimatum_rejection_rate",
    "dictator_offer_to_other",
]

DERIVED_DIMENSION_SPECS = [
    ("social_preferences", "trustingness"),
    ("social_preferences", "reciprocity"),
    ("social_preferences", "fairness_enforcement"),
    ("social_preferences", "altruistic_sharing"),
    ("social_preferences", "exploitation_caution"),
    ("self_regulation_and_affect", "cooperation_orientation"),
    ("self_regulation_and_affect", "competition_orientation"),
    ("self_regulation_and_affect", "uncertainty_aversion"),
    ("decision_style", "patience"),
    ("decision_style", "numeracy"),
    ("consumer_style", "willingness_to_search"),
    ("consumer_style", "purchase_inhibition"),
    ("consumer_style", "reference_dependence"),
]

RUN_SPECS_BY_MODEL = {
    "gpt-5.1": [
        {
            "run_name": "twin_sampled_unadjusted_seed_0_gpt_5_1_pgg_aligned_v3",
            "run_label": "Twin Unadjusted (PGG Card)",
        },
        {
            "run_name": "twin_sampled_unadjusted_seed_0_gpt_5_1_bargain_card_v1",
            "run_label": "Twin Bargain Card",
        },
        {
            "run_name": "twin_sampled_unadjusted_seed_0_gpt_5_1_descriptive_card_v1",
            "run_label": "Twin Descriptive",
        },
        {
            "run_name": "twin_sampled_unadjusted_seed_0_gpt_5_1_no_econ_games_v1",
            "run_label": "Twin No Econ Games",
        },
        {
            "run_name": "twin_sampled_unadjusted_seed_0_gpt_5_1_ultimatum_only_v1",
            "run_label": "Twin Ultimatum Only",
        },
    ],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_raw_social_features(profile_row: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for feature_row in profile_row["observed_in_twin"]["social_game_behavior"]["summary_features"]:
        name = str(feature_row["name"])
        value = feature_row.get("value") or {}
        raw_value = value.get("raw")
        if raw_value is not None:
            features[name] = float(raw_value)
    return features


def _extract_derived_dimensions(profile_row: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for group_name, feature_name in DERIVED_DIMENSION_SPECS:
        feature_row = profile_row["derived_dimensions"][group_name][feature_name]
        score = feature_row.get("score_0_to_100")
        if score is not None:
            features[feature_name] = float(score)
    return features


def load_profile_feature_map() -> dict[str, dict[str, float]]:
    feature_map: dict[str, dict[str, float]] = {}
    for row in read_jsonl(TWIN_PROFILES_PATH):
        pid = str(row["participant"]["pid"])
        features: dict[str, float] = {}
        features.update(_extract_raw_social_features(row))
        features.update(_extract_derived_dimensions(row))
        feature_map[pid] = features
    return feature_map


def load_card_cue_map(profile_cards_file: Path) -> dict[str, dict[str, float]]:
    cue_map: dict[str, dict[str, float]] = {}
    for row in read_jsonl(profile_cards_file):
        pid = str((row.get("participant") or {}).get("pid"))
        transfer_relevance = row.get("transfer_relevance") or []
        cues: dict[str, float] = {}
        for cue_row in transfer_relevance:
            cue_name = cue_row.get("cue")
            score = cue_row.get("score_0_to_100")
            if cue_name is None or score is None:
                continue
            cues[str(cue_name)] = float(score)
        if pid and cues:
            cue_map[pid] = cues
    return cue_map


def aggregate_assignment_features(
    assignment_file: Path,
    feature_map: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for assignment_row in read_jsonl(assignment_file):
        assignments = assignment_row.get("assignments") or []
        if not assignments:
            continue
        first_pid = str(assignments[0]["twin_pid"])
        feature_names = list(feature_map[first_pid].keys())
        aggregate_row: dict[str, Any] = {
            "record_id": str(assignment_row["gameId"]),
            "treatment_name": str(assignment_row["treatment_name"]),
        }
        for feature_name in feature_names:
            values = [
                feature_map[str(item["twin_pid"])].get(feature_name)
                for item in assignments
                if str(item["twin_pid"]) in feature_map
            ]
            values = [float(value) for value in values if value is not None]
            aggregate_row[feature_name] = float(np.mean(values)) if values else np.nan
        rows.append(aggregate_row)
    return pd.DataFrame(rows)


def within_treatment_corr(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    subset = frame[["treatment_name", x_col, y_col]].dropna()
    if len(subset) < 8:
        return float("nan")
    x_resid = subset[x_col] - subset.groupby("treatment_name")[x_col].transform("mean")
    y_resid = subset[y_col] - subset.groupby("treatment_name")[y_col].transform("mean")
    if float(x_resid.std(ddof=0)) == 0.0 or float(y_resid.std(ddof=0)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_resid, y_resid)[0, 1])


def load_metric_frame(run_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_dir = RESULTS_ROOT / f"{run_name}__vs_human_treatments"
    generated = pd.read_csv(results_dir / "generated_game_records.csv")
    human = pd.read_csv(results_dir / "human_game_records.csv")
    human_means = (
        human.groupby("treatment_name", sort=True)[METRICS]
        .mean()
        .rename(columns={metric: f"human_{metric}" for metric in METRICS})
        .reset_index()
    )
    merged = generated.merge(human_means, on="treatment_name", how="left")
    for metric in METRICS:
        merged[f"signed_gap_{metric}"] = merged[metric] - merged[f"human_{metric}"]
        merged[f"abs_gap_{metric}"] = merged[f"signed_gap_{metric}"].abs()
    return merged, human


def compute_correlation_rows(
    *,
    run_name: str,
    run_label: str,
    metric_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    feature_family: str,
) -> list[dict[str, Any]]:
    merged = metric_frame.merge(feature_frame, on=["record_id", "treatment_name"], how="inner")
    feature_columns = [col for col in feature_frame.columns if col not in {"record_id", "treatment_name"}]
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        for feature_name in feature_columns:
            rows.append(
                {
                    "run_name": run_name,
                    "run_label": run_label,
                    "feature_family": feature_family,
                    "feature_name": feature_name,
                    "metric": metric,
                    "n_games": int(len(merged)),
                    "feature_mean": float(merged[feature_name].mean(skipna=True)),
                    "feature_std": float(merged[feature_name].std(skipna=True)),
                    "outcome_within_treatment_corr": within_treatment_corr(merged, feature_name, metric),
                    "signed_gap_within_treatment_corr": within_treatment_corr(
                        merged,
                        feature_name,
                        f"signed_gap_{metric}",
                    ),
                    "abs_gap_within_treatment_corr": within_treatment_corr(
                        merged,
                        feature_name,
                        f"abs_gap_{metric}",
                    ),
                }
            )
    return rows


def summarize_top_rows(
    rows: list[dict[str, Any]],
    *,
    value_column: str,
    top_k: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["abs_rank_value"] = frame[value_column].abs()
    return (
        frame.sort_values(
            ["run_label", "metric", "abs_rank_value"],
            ascending=[True, True, False],
        )
        .groupby(["run_label", "metric"], sort=True)
        .head(top_k)
        .drop(columns=["abs_rank_value"])
        .reset_index(drop=True)
    )


def build_outputs(model_name: str) -> None:
    run_specs = RUN_SPECS_BY_MODEL[model_name]
    raw_feature_map = load_profile_feature_map()

    raw_derived_rows: list[dict[str, Any]] = []
    explicit_cue_rows: list[dict[str, Any]] = []

    for run_spec in run_specs:
        run_name = str(run_spec["run_name"])
        run_label = str(run_spec["run_label"])
        manifest = json.loads((METADATA_ROOT / run_name / "manifest.json").read_text(encoding="utf-8"))

        metric_frame, _ = load_metric_frame(run_name)
        assignment_frame = aggregate_assignment_features(Path(manifest["profile_assignment_file"]), raw_feature_map)
        raw_derived_rows.extend(
            compute_correlation_rows(
                run_name=run_name,
                run_label=run_label,
                metric_frame=metric_frame,
                feature_frame=assignment_frame,
                feature_family="raw_and_derived",
            )
        )

        profile_cards_file = manifest.get("profile_cards_file")
        if profile_cards_file:
            cue_map = load_card_cue_map(Path(profile_cards_file))
            if cue_map:
                cue_frame = aggregate_assignment_features(Path(manifest["profile_assignment_file"]), cue_map)
                explicit_cue_rows.extend(
                    compute_correlation_rows(
                        run_name=run_name,
                        run_label=run_label,
                        metric_frame=metric_frame,
                        feature_frame=cue_frame,
                        feature_family="explicit_card_cues",
                    )
                )

    raw_output = RESULTS_ROOT / f"{model_name.replace('-', '_')}_profile_influence_raw_derived_correlations.csv"
    explicit_output = RESULTS_ROOT / f"{model_name.replace('-', '_')}_profile_influence_explicit_cue_correlations.csv"
    raw_top_output = RESULTS_ROOT / f"{model_name.replace('-', '_')}_profile_influence_raw_derived_top_abs_gap.csv"
    explicit_top_output = RESULTS_ROOT / f"{model_name.replace('-', '_')}_profile_influence_explicit_cue_top_abs_gap.csv"

    raw_frame = pd.DataFrame(raw_derived_rows).sort_values(
        ["run_label", "metric", "feature_name"],
        ascending=[True, True, True],
    )
    raw_frame.to_csv(raw_output, index=False)
    summarize_top_rows(raw_derived_rows, value_column="abs_gap_within_treatment_corr", top_k=6).to_csv(
        raw_top_output,
        index=False,
    )

    explicit_frame = pd.DataFrame(explicit_cue_rows).sort_values(
        ["run_label", "metric", "feature_name"],
        ascending=[True, True, True],
    )
    explicit_frame.to_csv(explicit_output, index=False)
    summarize_top_rows(explicit_cue_rows, value_column="abs_gap_within_treatment_corr", top_k=6).to_csv(
        explicit_top_output,
        index=False,
    )

    print(f"Wrote {raw_output}")
    print(f"Wrote {raw_top_output}")
    print(f"Wrote {explicit_output}")
    print(f"Wrote {explicit_top_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze how assigned Twin features and card cues shift chip-bargain behavior and alignment.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.1",
        choices=sorted(RUN_SPECS_BY_MODEL.keys()),
        help="Model family whose completed chip-bargain Twin runs should be analyzed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_outputs(args.model)


if __name__ == "__main__":
    main()
