from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    build_generated_scenarios_df,
    build_human_scenarios_df,
    write_csv,
    write_json,
)


TABLE1_SPECS = [
    {
        "question_id": "Q1",
        "question": "Does interacting with AI decrease payoffs?",
        "variable_code": "WI",
        "variable_label": "payoff index",
        "cond1_label": "AI in TRU",
        "cond2_label": "Human in TRU",
        "left_role": "AI",
        "right_role": "human",
        "left_treatments": ["TRU"],
        "right_treatments": ["TRU"],
        "direction": "less",
    },
    {
        "question_id": "Q2",
        "question": "Is opaque delegation more frequent?",
        "variable_code": "DI",
        "variable_label": "delegation frequency",
        "cond1_label": "Support in ODU/ODP",
        "cond2_label": "Support in TDU/TDP",
        "left_role": "support",
        "right_role": "support",
        "left_treatments": ["ODU", "ODP"],
        "right_treatments": ["TDU", "TDP"],
        "direction": "greater",
    },
    {
        "question_id": "Q3",
        "question": "Does opaque interaction decrease payoffs?",
        "variable_code": "WI",
        "variable_label": "payoff index",
        "cond1_label": "Unknown in ODU",
        "cond2_label": "Human in TRU",
        "left_role": "unknown",
        "right_role": "human",
        "left_treatments": ["ODU"],
        "right_treatments": ["TRU"],
        "direction": "less",
    },
    {
        "question_id": "Q4",
        "question": "Does delegation crowd-out prosociality?",
        "variable_code": "PSI",
        "variable_label": "prosociality index",
        "cond1_label": "AI in TDU",
        "cond2_label": "AI in TRU",
        "left_role": "AI",
        "right_role": "AI",
        "left_treatments": ["TDU"],
        "right_treatments": ["TRU"],
        "direction": "less",
    },
    {
        "question_id": "Q5",
        "question": "Does personalizing the AI restore payoffs?",
        "variable_code": "WI",
        "variable_label": "payoff index",
        "cond1_label": "AI in TRP",
        "cond2_label": "AI in TRU",
        "left_role": "AI",
        "right_role": "AI",
        "left_treatments": ["TRP"],
        "right_treatments": ["TRU"],
        "direction": "greater",
    },
    {
        "question_id": "Q6",
        "question": "Does personalizing the AI increase delegation?",
        "variable_code": "DI",
        "variable_label": "delegation frequency",
        "cond1_label": "Support in TDP/ODP",
        "cond2_label": "Support in TDU/ODU",
        "left_role": "support",
        "right_role": "support",
        "left_treatments": ["TDP", "ODP"],
        "right_treatments": ["TDU", "ODU"],
        "direction": "greater",
    },
    {
        "question_id": "Q7",
        "question": "Does personalizing the AI change payoffs if delegation is opaque?",
        "variable_code": "WI",
        "variable_label": "payoff index",
        "cond1_label": "Unknown in ODP",
        "cond2_label": "Unknown in ODU",
        "left_role": "unknown",
        "right_role": "unknown",
        "left_treatments": ["ODP"],
        "right_treatments": ["ODU"],
        "direction": "greater",
    },
    {
        "question_id": "Q8",
        "question": "Does personalizing the AI change prosocial behavior?",
        "variable_code": "PSI",
        "variable_label": "prosociality index",
        "cond1_label": "AI in TDP",
        "cond2_label": "AI in TDU",
        "left_role": "AI",
        "right_role": "AI",
        "left_treatments": ["TDP"],
        "right_treatments": ["TDU"],
        "direction": "greater",
    },
]

FIGURE1_SPECS = [
    {
        "metric_id": "trust_in_llm",
        "metric_label": "Less trust in LLM",
        "variable_code": "TGSender_raw",
        "cond1_label": "AgainstAI",
        "cond2_label": "AgainstHuman",
        "direction": "less",
    },
    {
        "metric_id": "trustworthy_against_llm",
        "metric_label": "Less trustworthy against LLM",
        "variable_code": "TGReceiver_raw",
        "cond1_label": "AgainstAI",
        "cond2_label": "AgainstHuman",
        "direction": "less",
    },
    {
        "metric_id": "pd_against_llm",
        "metric_label": "Less cooperation in PD",
        "variable_code": "PD_raw",
        "cond1_label": "AgainstAI",
        "cond2_label": "AgainstHuman",
        "direction": "less",
    },
    {
        "metric_id": "sh_against_llm",
        "metric_label": "Less cooperation in SH",
        "variable_code": "SH_raw",
        "cond1_label": "AgainstAI",
        "cond2_label": "AgainstHuman",
        "direction": "less",
    },
    {
        "metric_id": "coordination_against_llm",
        "metric_label": "Less predictability in C",
        "variable_code": "Earth",
        "cond1_label": "AgainstAI",
        "cond2_label": "AgainstHuman",
        "direction": "less",
    },
    {
        "metric_id": "ug_offer_against_llm",
        "metric_label": "Lower offers against LLM",
        "variable_code": "UGProposer_raw",
        "cond1_label": "AgainstAI",
        "cond2_label": "AgainstHuman",
        "direction": "less",
    },
    {
        "metric_id": "ug_threshold_against_llm",
        "metric_label": "Tolerate less inequality",
        "variable_code": "UGResponder_raw",
        "cond1_label": "AgainstAI",
        "cond2_label": "AgainstHuman",
        "direction": "less",
    },
]


def _role_from_row(row: pd.Series) -> str:
    if row["scenario"] == "AISupport":
        return "support"
    if row["case"] == "AgainstAI":
        return "AI"
    if row["case"] == "AgainstHuman":
        return "human"
    return "unknown"


def _binary_choice(series: pd.Series, *, positive: str) -> pd.Series:
    return series.map(lambda value: np.nan if pd.isna(value) else 1.0 if str(value) == positive else 0.0)


def _build_analysis_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["role"] = data.apply(_role_from_row, axis=1)

    data["UG_S"] = pd.to_numeric(data["UGProposer_decision"], errors="coerce") / 5.0
    data["UG_R"] = pd.to_numeric(data["UGResponder_decision"], errors="coerce") / 5.0
    data["rUG_R"] = 1.0 - data["UG_R"]
    data["TG_S"] = _binary_choice(data["TGSender_decision"], positive="YES")
    data["TG_T"] = pd.to_numeric(data["TGReceiver_decision"], errors="coerce") / 6.0
    data["PD"] = _binary_choice(data["PD_decision"], positive="A")
    data["SH"] = _binary_choice(data["SH_decision"], positive="X")
    data["Earth"] = data["C_decision"].map(
        lambda value: np.nan if pd.isna(value) else 1.0 if str(value) == "Earth" else 0.0
    )

    for field_name in [
        "UGProposer_delegated",
        "UGResponder_delegated",
        "TGSender_delegated",
        "TGReceiver_delegated",
        "PD_delegated",
        "SH_delegated",
        "C_delegated",
    ]:
        data[field_name] = pd.to_numeric(data[field_name], errors="coerce")

    data["DI"] = data[
        [
            "UGProposer_delegated",
            "UGResponder_delegated",
            "TGSender_delegated",
            "TGReceiver_delegated",
            "PD_delegated",
            "SH_delegated",
            "C_delegated",
        ]
    ].mean(axis=1, skipna=True)
    data["WI"] = data[["UG_S", "UG_R", "TG_S", "TG_T", "PD", "SH", "Earth"]].mean(axis=1, skipna=True)
    data["PSI"] = data[["UG_S", "rUG_R", "TG_S", "TG_T", "PD", "SH"]].mean(axis=1, skipna=True)
    data["PM"] = data["Earth"]
    data["FM"] = data["UG_S"]
    data["KI"] = data[["TG_S", "PD", "SH"]].mean(axis=1, skipna=True)
    data["II"] = data[["UG_R", "TG_T"]].mean(axis=1, skipna=True)

    data["UGProposer_raw"] = pd.to_numeric(data["UGProposer_decision"], errors="coerce")
    data["UGResponder_raw"] = pd.to_numeric(data["UGResponder_decision"], errors="coerce")
    data["TGSender_raw"] = data["TG_S"]
    data["TGReceiver_raw"] = pd.to_numeric(data["TGReceiver_decision"], errors="coerce")
    data["PD_raw"] = data["PD"]
    data["SH_raw"] = data["SH"]
    return data


def _direction_supported(effect: float, direction: str) -> bool | None:
    if pd.isna(effect):
        return None
    if direction == "less":
        return bool(effect < 0)
    if direction == "greater":
        return bool(effect > 0)
    raise ValueError(f"Unsupported direction: {direction}")


def _safe_stats(series: pd.Series) -> tuple[float, float, int]:
    clean = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if clean.empty:
        return float("nan"), float("nan"), 0
    return float(clean.mean()), float(clean.std(ddof=1)) if len(clean) > 1 else float("nan"), int(len(clean))


def _compute_table1_for_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in TABLE1_SPECS:
        left_values = frame.loc[
            (frame["role"] == spec["left_role"])
            & (frame["treatment_name"].isin(spec["left_treatments"])),
            spec["variable_code"],
        ]
        right_values = frame.loc[
            (frame["role"] == spec["right_role"])
            & (frame["treatment_name"].isin(spec["right_treatments"])),
            spec["variable_code"],
        ]
        left_mean, left_sd, left_n = _safe_stats(left_values)
        right_mean, right_sd, right_n = _safe_stats(right_values)
        effect = left_mean - right_mean if pd.notna(left_mean) and pd.notna(right_mean) else float("nan")
        rows.append(
            {
                **spec,
                "cond1_mean": left_mean,
                "cond1_sd": left_sd,
                "cond1_n": left_n,
                "cond2_mean": right_mean,
                "cond2_sd": right_sd,
                "cond2_n": right_n,
                "effect": effect,
                "supports_direction": _direction_supported(effect, str(spec["direction"])),
            }
        )
    return pd.DataFrame(rows)


def _compute_figure1_for_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    subset = frame[
        (frame["scenario"] == "NoAISupport")
        & (frame["treatment_name"].isin(["TRU", "TRP", "TDU", "TDP"]))
    ].copy()
    for spec in FIGURE1_SPECS:
        ai_values = subset.loc[subset["case"] == "AgainstAI", spec["variable_code"]]
        human_values = subset.loc[subset["case"] == "AgainstHuman", spec["variable_code"]]
        ai_mean, ai_sd, ai_n = _safe_stats(ai_values)
        human_mean, human_sd, human_n = _safe_stats(human_values)
        effect = ai_mean - human_mean if pd.notna(ai_mean) and pd.notna(human_mean) else float("nan")
        rows.append(
            {
                **spec,
                "cond1_mean": ai_mean,
                "cond1_sd": ai_sd,
                "cond1_n": ai_n,
                "cond2_mean": human_mean,
                "cond2_sd": human_sd,
                "cond2_n": human_n,
                "effect": effect,
                "supports_direction": _direction_supported(effect, str(spec["direction"])),
            }
        )
    return pd.DataFrame(rows)


def _merge_model_vs_human(model_df: pd.DataFrame, human_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    merged = human_df.merge(
        model_df,
        on=id_col,
        how="inner",
        suffixes=("_human", "_model"),
    )
    merged["effect_abs_error"] = (merged["effect_model"] - merged["effect_human"]).abs()
    merged["cond1_mean_abs_error"] = (merged["cond1_mean_model"] - merged["cond1_mean_human"]).abs()
    merged["cond2_mean_abs_error"] = (merged["cond2_mean_model"] - merged["cond2_mean_human"]).abs()
    merged["direction_match"] = (
        merged["supports_direction_human"].astype("boolean")
        == merged["supports_direction_model"].astype("boolean")
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the multi-game paper's treatment contrasts from model outputs and compare them to the human benchmark."
    )
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()

    metadata_dir = args.forecasting_root / "metadata" / args.run_name
    output_dir = args.forecasting_root / "results" / f"{args.run_name}__paper_reproduction"

    human_scenarios = build_human_scenarios_df(
        gold_targets_jsonl=metadata_dir / "gold_targets.jsonl",
        request_manifest_csv=metadata_dir / "request_manifest.csv",
    )
    generated_scenarios = build_generated_scenarios_df(
        parsed_output_jsonl=metadata_dir / "parsed_output.jsonl",
        request_manifest_csv=metadata_dir / "request_manifest.csv",
    )

    human_data = _build_analysis_frame(human_scenarios)
    model_data = _build_analysis_frame(generated_scenarios)

    human_table1 = _compute_table1_for_frame(human_data)
    model_table1 = _compute_table1_for_frame(model_data)
    table1_comparison = _merge_model_vs_human(model_table1, human_table1, "question_id")

    human_figure1 = _compute_figure1_for_frame(human_data)
    model_figure1 = _compute_figure1_for_frame(model_data)
    figure1_comparison = _merge_model_vs_human(model_figure1, human_figure1, "metric_id")

    summary = {
        "run_name": args.run_name,
        "table1_mean_abs_effect_error": float(table1_comparison["effect_abs_error"].mean()),
        "table1_direction_match_rate": float(table1_comparison["direction_match"].mean()),
        "figure1_mean_abs_effect_error": float(figure1_comparison["effect_abs_error"].mean()),
        "figure1_direction_match_rate": float(figure1_comparison["direction_match"].mean()),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "human_table1_reproduction.csv", human_table1)
    write_csv(output_dir / "model_table1_reproduction.csv", model_table1)
    write_csv(output_dir / "table1_reproduction_comparison.csv", table1_comparison)
    write_csv(output_dir / "human_figure1_reproduction.csv", human_figure1)
    write_csv(output_dir / "model_figure1_reproduction.csv", model_figure1)
    write_csv(output_dir / "figure1_reproduction_comparison.csv", figure1_comparison)
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
