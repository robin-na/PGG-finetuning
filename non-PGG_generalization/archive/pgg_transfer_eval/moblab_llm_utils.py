#!/usr/bin/env python3
"""Shared utilities for MobLab LLM-based inference tasks."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analyze_moblab_persistence_and_correlation import build_all_rounds, summarize_sessions, summarize_users


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "pgg_transfer_eval"
    / "output"
    / "moblab_llm"
)
DEFAULT_TASK_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "tasks"
DEFAULT_STAT_BASELINE_DIR = DEFAULT_OUTPUT_ROOT.parent / "moblab_statistical_baselines"

MEASURE_LABELS = {
    "dictator": "Dictator offer share",
    "trust_investor": "Trust investor share",
    "trust_banker": "Trust banker return share",
    "ultimatum_proposer": "Ultimatum proposer offer share",
    "ultimatum_responder": "Ultimatum responder minimum-acceptable share",
    "pg_contribution": "Public-goods contribution share",
}

MEASURE_MAX_AMOUNT = {
    "dictator": 100.0,
    "trust_investor": 100.0,
    "trust_banker": 100.0,
    "ultimatum_proposer": 100.0,
    "ultimatum_responder": 100.0,
    "pg_contribution": 20.0,
}

TASK2_SUPPORTED_MEASURES = [
    "dictator",
    "trust_investor",
    "trust_banker",
    "pg_contribution",
]

# These templates are adapted from the prompts in yutxie/ChatGPT-Behavioral.
GAME_RULE_TEMPLATES = {
    "dictator": (
        "You are paired with another player. Your role is to decide how to divide $100 "
        "and the other player simply receives your choice."
    ),
    "trust_investor": (
        "This is a two-player game. You are an Investor and the other player is a Banker. "
        "You have $100 to invest and the amount you invest grows by 3x with the Banker. "
        "The Banker then decides how much of the resulting amount to return to you."
    ),
    "trust_banker": (
        "This is a two-player trust game. You are a Banker and the other player is an Investor. "
        "The Investor can invest up to $100. The invested amount becomes 3x with you, and you decide "
        "how much of that amount to return to the Investor."
    ),
    "ultimatum_proposer": (
        "This is a two-player ultimatum game. You are the Proposer, and the other player is the Responder. "
        "You propose how to divide $100. If the Responder accepts, payoffs follow the proposal; if the "
        "Responder rejects, both players get $0."
    ),
    "ultimatum_responder": (
        "This is a two-player ultimatum game. You are the Responder, and the other player is the Proposer. "
        "The Proposer proposes how to divide $100. If you accept, payoffs follow the proposal; if you reject, "
        "both players get $0."
    ),
    "pg_contribution": (
        "In this public-goods game, you and 3 other players each receive $20 in a round and choose how much "
        "to contribute to a common project. The project has a 50% return rate, so each player's payoff equals "
        "the amount not contributed plus 50% of the total group contribution."
    ),
}

TASK1_OUTPUT_SPECS = {
    "dictator": "Predict the participant's typical first-round Dictator offer as a share percent from 0 to 100.",
    "trust_investor": "Predict the participant's typical first-round Trust investment as a share percent from 0 to 100.",
    "trust_banker": "Predict the participant's typical first-round Banker return share as a percent of the available amount from 0 to 100.",
    "ultimatum_proposer": "Predict the participant's typical Ultimatum proposal as a share percent from 0 to 100.",
    "ultimatum_responder": "Predict the participant's typical minimum acceptable Ultimatum share as a percent from 0 to 100.",
    "pg_contribution": "Predict the participant's typical first-round Public Goods contribution as a share percent from 0 to 100.",
}

TASK2_OUTPUT_SPECS = {
    "future_mean": {
        "dictator": "Predict the participant's average share percent over the remaining rounds of this session (rounds 2..T).",
        "trust_investor": "Predict the participant's average future Trust-investor share percent over the remaining rounds of this session (rounds 2..T).",
        "trust_banker": "Predict the participant's average future Banker return share percent over the remaining rounds of this session (rounds 2..T).",
        "pg_contribution": "Predict the participant's average future Public Goods contribution share percent over the remaining rounds of this session (rounds 2..T).",
    },
    "trajectory": {
        "dictator": "Predict the participant's share percent in every remaining round of this Dictator session, in order from round 2 through round T.",
        "trust_investor": "Predict the participant's Trust-investor share percent in every remaining round of this session, in order from round 2 through round T.",
        "trust_banker": "Predict the participant's Banker return share percent in every remaining round of this session, in order from round 2 through round T.",
        "pg_contribution": "Predict the participant's Public Goods contribution share percent in every remaining round of this session, in order from round 2 through round T.",
    },
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def truncate_text(text: str, limit: int) -> str:
    text = normalize_whitespace(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def share_to_percent(value: float) -> float:
    return float(value) * 100.0


def share_to_amount(measure: str, value: float) -> float:
    return float(value) * MEASURE_MAX_AMOUNT[measure]


def round_percent(value: float) -> float:
    return round(float(value), 2)


def measure_family(measure: str) -> str:
    if measure.startswith("trust_"):
        return "trust"
    if measure.startswith("ultimatum_"):
        return "ultimatum"
    if measure == "pg_contribution":
        return "pg"
    return measure


def canonical_source_text(measure: str, share_value: float) -> str:
    percent = round_percent(share_to_percent(share_value))
    amount = round_percent(share_to_amount(measure, share_value))
    if measure == "dictator":
        return f"As a Dictator, this participant typically gives about {percent}% of the endowment (${amount} out of $100) in the first round."
    if measure == "trust_investor":
        return f"As a Trust Investor, this participant typically invests about {percent}% of the available amount (${amount} out of $100) in the first round."
    if measure == "trust_banker":
        return f"As a Trust Banker, this participant typically returns about {percent}% of the amount under their control. On a $100 basis, that is roughly ${amount} returned per $100 available."
    if measure == "ultimatum_proposer":
        return f"As an Ultimatum Proposer, this participant typically offers about {percent}% of the pie (${amount} out of $100)."
    if measure == "ultimatum_responder":
        return f"As an Ultimatum Responder, this participant's typical minimum acceptable offer is about {percent}% of the pie (${amount} out of $100)."
    if measure == "pg_contribution":
        return f"In Public Goods, this participant typically contributes about {percent}% of the endowment (${amount} out of $20) in round 1."
    raise KeyError(f"Unknown measure: {measure}")


def task2_round1_text(measure: str, share_value: float) -> str:
    percent = round_percent(share_to_percent(share_value))
    amount = round_percent(share_to_amount(measure, share_value))
    if measure == "dictator":
        return f"In round 1 of this Dictator session, the participant gave {percent}% of the endowment (${amount} out of $100)."
    if measure == "trust_investor":
        return f"In round 1 of this Trust-Investor session, the participant invested {percent}% of the available amount (${amount} out of $100)."
    if measure == "trust_banker":
        return f"In round 1 of this Trust-Banker session, the participant returned {percent}% of the amount under their control."
    if measure == "pg_contribution":
        return f"In round 1 of this Public Goods session, the participant contributed {percent}% of the endowment (${amount} out of $20)."
    raise KeyError(f"Unsupported task-2 measure: {measure}")


def output_schema_for_prediction(task_type: str, prediction_mode: str) -> Dict[str, Any]:
    prediction_block: Dict[str, Any]
    if task_type == "task2" and prediction_mode == "trajectory":
        prediction_block = {
            "future_round_share_percents": [0.0, 0.0],
            "confidence": "low|medium|high",
        }
    else:
        prediction_block = {
            "share_percent": 0.0,
            "confidence": "low|medium|high",
        }
    return {
        "persona_summary": "1-3 sentence behavioral summary",
        "policy_reflection": "1-3 sentence explanation of how this participant tends to decide in the target game",
        "prediction": prediction_block,
    }


def output_schema_for_persona() -> Dict[str, Any]:
    return {
        "persona_summary": "2-4 sentence summary of the participant",
        "decision_style": "2-4 sentence summary of how this participant typically decides in incentive games",
        "latent_traits": {
            "generosity": "low|medium|high",
            "reciprocity": "low|medium|high",
            "fairness_sensitivity": "low|medium|high",
            "caution": "low|medium|high",
            "stability": "low|medium|high",
        },
    }


def sample_stratified(df: pd.DataFrame, strata_col: str, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0 or len(df) <= sample_size:
        return df.copy()
    groups = {
        str(key): group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        for key, group in df.groupby(strata_col, sort=True)
    }
    picks: List[pd.Series] = []
    cursor = {key: 0 for key in groups}
    ordered_keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(ordered_keys)
    while len(picks) < sample_size:
        advanced = False
        for key in ordered_keys:
            idx = cursor[key]
            group = groups[key]
            if idx >= len(group):
                continue
            picks.append(group.iloc[idx])
            cursor[key] += 1
            advanced = True
            if len(picks) >= sample_size:
                break
        if not advanced:
            break
    sampled = pd.DataFrame(picks).reset_index(drop=True)
    return sampled


def build_task1_instances() -> pd.DataFrame:
    rounds_df = build_all_rounds()
    session_df = summarize_sessions(rounds_df)
    user_first = summarize_users(session_df, "first_round_value").reset_index()
    measures = [measure for measure in MEASURE_LABELS if measure in user_first.columns]

    rows: List[Dict[str, Any]] = []
    for row in user_first.itertuples(index=False):
        user_id = int(getattr(row, "UserID"))
        values = {measure: getattr(row, measure, np.nan) for measure in measures}
        for target_measure in measures:
            target_value = values.get(target_measure)
            if pd.isna(target_value):
                continue
            source_evidence = []
            source_values: Dict[str, float] = {}
            for source_measure in measures:
                if source_measure == target_measure:
                    continue
                source_value = values.get(source_measure)
                if pd.isna(source_value):
                    continue
                source_values[source_measure] = float(source_value)
                source_evidence.append(canonical_source_text(source_measure, float(source_value)))
            if not source_evidence:
                continue
            rows.append(
                {
                    "instance_id": f"task1::{user_id}::{target_measure}",
                    "task_type": "task1",
                    "prediction_mode": "scalar",
                    "user_id": user_id,
                    "target_measure": target_measure,
                    "target_family": measure_family(target_measure),
                    "game_rule_text": GAME_RULE_TEMPLATES[target_measure],
                    "target_instruction": TASK1_OUTPUT_SPECS[target_measure],
                    "target_label": MEASURE_LABELS[target_measure],
                    "source_values": source_values,
                    "source_evidence_text": "\n".join(f"- {line}" for line in source_evidence),
                    "gold_share_percent": round_percent(share_to_percent(float(target_value))),
                    "gold_share_value": float(target_value),
                    "available_source_measures": sorted(source_values),
                }
            )
    return pd.DataFrame(rows).sort_values(["target_measure", "user_id"]).reset_index(drop=True)


def build_task2_instances(prediction_mode: str = "future_mean") -> pd.DataFrame:
    if prediction_mode not in TASK2_OUTPUT_SPECS:
        raise ValueError(f"Unsupported task2 prediction mode: {prediction_mode}")
    rounds_df = build_all_rounds()
    session_df = summarize_sessions(rounds_df)
    strict = session_df[(session_df["session_rounds"] >= 2) & (session_df["measure"].isin(TASK2_SUPPORTED_MEASURES))].copy()
    future_values_by_session: Dict[Tuple[str, int, str], List[float]] = {}
    for (measure, user_id, session_id), group in rounds_df.groupby(["measure", "UserID", "session_id"], sort=False):
        if measure not in TASK2_SUPPORTED_MEASURES or len(group) < 2:
            continue
        future_values_by_session[(str(measure), int(user_id), str(session_id))] = (
            group.sort_values("Round")["value"].astype(float).tolist()[1:]
        )

    rows: List[Dict[str, Any]] = []
    for row in strict.itertuples(index=False):
        key = (str(row.measure), int(row.UserID), str(row.session_id))
        future_rounds = future_values_by_session.get(key)
        if not future_rounds:
            continue
        rows.append(
            {
                "instance_id": f"task2::{prediction_mode}::{row.measure}::{int(row.UserID)}::{row.session_id}",
                "task_type": "task2",
                "prediction_mode": prediction_mode,
                "user_id": int(row.UserID),
                "session_id": row.session_id,
                "target_measure": row.measure,
                "target_family": measure_family(row.measure),
                "game_rule_text": GAME_RULE_TEMPLATES[row.measure],
                "target_instruction": TASK2_OUTPUT_SPECS[prediction_mode][row.measure],
                "target_label": MEASURE_LABELS[row.measure],
                "round1_share_percent": round_percent(share_to_percent(float(row.first_round_value))),
                "round1_share_value": float(row.first_round_value),
                "round1_evidence_text": task2_round1_text(row.measure, float(row.first_round_value)),
                "gold_future_share_percent": round_percent(share_to_percent(float(row.future_mean))),
                "gold_future_share_value": float(row.future_mean),
                "gold_future_rounds_share_percent": [round_percent(share_to_percent(value)) for value in future_rounds],
                "future_round_numbers": list(range(2, len(future_rounds) + 2)),
                "n_future_rounds": len(future_rounds),
                "session_rounds": int(row.session_rounds),
                "persistence_share_percent": round_percent(share_to_percent(float(row.first_round_value))),
                "persistence_future_rounds_share_percent": [
                    round_percent(share_to_percent(float(row.first_round_value))) for _ in future_rounds
                ],
            }
        )
    return pd.DataFrame(rows).sort_values(["target_measure", "user_id", "session_id"]).reset_index(drop=True)


def write_jsonl(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def task_dataframe(task_type: str, task2_prediction_mode: str = "future_mean") -> pd.DataFrame:
    if task_type == "task1":
        return build_task1_instances()
    if task_type == "task2":
        return build_task2_instances(prediction_mode=task2_prediction_mode)
    raise ValueError(f"Unsupported task type: {task_type}")


def stat_baseline_reference(task_type: str) -> pd.DataFrame:
    if task_type == "task1":
        return pd.read_csv(DEFAULT_STAT_BASELINE_DIR / "all_other_first_round_prediction.csv")
    if task_type == "task2":
        return pd.read_csv(DEFAULT_STAT_BASELINE_DIR / "k1_persistence_baseline.csv")
    raise ValueError(f"Unsupported task type: {task_type}")


def prediction_system_prompt() -> str:
    return (
        "You simulate human economic-game behavior from limited prior evidence. "
        "Infer a compact persona from the evidence, then predict the target behavior. "
        "Stay within the legal action range. Return JSON only."
    )


def persona_system_prompt() -> str:
    return (
        "You infer compact behavioral personas from observed economic-game evidence. "
        "Summaries should be short, concrete, and grounded in the observed choices only. "
        "Return JSON only."
    )


def retrieval_query_system_prompt() -> str:
    return (
        "You write retrieval queries for a library of repeated-public-goods-game persona cards. "
        "Translate the observed MobLab behavior into transferable behavioral cues such as generosity, "
        "reciprocity, fairness sensitivity, caution, conditional cooperation, stability, and response to incentives. "
        "Return JSON only."
    )


def build_prediction_messages(
    *,
    instance: Dict[str, Any],
    baseline: str,
    persona_text: Optional[str] = None,
    retrieved_cards: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    prediction_mode = str(instance.get("prediction_mode", "scalar"))
    lines = [
        f"# Task Type\n{instance['task_type']}",
        "",
        f"# Target Behavior\n{instance['target_label']}",
        "",
        "# Target Game Rules",
        instance["game_rule_text"],
        "",
    ]
    if instance["task_type"] == "task1":
        lines.extend(
            [
                "# Observed Behavior From Other Games",
                instance["source_evidence_text"],
                "",
            ]
        )
    else:
        lines.extend(
            [
                "# Observed Early Behavior In This Session",
                instance["round1_evidence_text"],
                f"The session lasts {instance['session_rounds']} rounds, so there are {instance['n_future_rounds']} future rounds to predict.",
                "",
            ]
        )

    if baseline in {"persona", "meta_persona"} and persona_text:
        lines.extend(
            [
                "# Precomputed Persona",
                persona_text,
                "",
            ]
        )

    if baseline == "retrieval" and retrieved_cards:
        lines.extend(["# Retrieved PGG Persona Cards"])
        for idx, card in enumerate(retrieved_cards, start=1):
            lines.extend(
                [
                    f"## Candidate {idx}",
                    truncate_text(card.get("document_text", "") or "", 3200),
                    "",
                ]
            )

    lines.extend(
        [
            "# Required Output",
            instance["target_instruction"],
            (
                "The prediction should be a share percent between 0 and 100."
                if not (instance["task_type"] == "task2" and prediction_mode == "trajectory")
                else f"Return exactly {instance['n_future_rounds']} share percents between 0 and 100, corresponding to rounds {instance['future_round_numbers'][0]}..{instance['future_round_numbers'][-1]}."
            ),
            json.dumps(output_schema_for_prediction(instance["task_type"], prediction_mode), ensure_ascii=False, indent=2),
        ]
    )

    if baseline == "direct":
        lines.append("Use the observed evidence directly. Do not invent latent traits that are not supported by the evidence.")
    elif baseline == "persona":
        lines.append("Use the precomputed persona as the main summary of the participant.")
    elif baseline == "meta_persona":
        lines.append("First interpret the precomputed persona as a decision policy, then forecast the target behavior.")
    elif baseline == "retrieval":
        lines.append("Use the retrieved PGG persona cards as analogical evidence, but do not assume an exact identity match.")
    else:
        raise ValueError(f"Unsupported baseline: {baseline}")

    return [
        {"role": "system", "content": prediction_system_prompt()},
        {"role": "user", "content": "\n".join(lines)},
    ]


def build_persona_messages(instance: Dict[str, Any]) -> List[Dict[str, str]]:
    lines = [
        "# Observed Evidence",
    ]
    if instance["task_type"] == "task1":
        lines.append(instance["source_evidence_text"])
    else:
        lines.append(instance["round1_evidence_text"])
        lines.append(f"The session has {instance['session_rounds']} rounds in total.")
        if str(instance.get("prediction_mode", "future_mean")) == "trajectory":
            lines.append(f"The eventual forecasting task asks for the remaining {instance['n_future_rounds']} rounds in sequence.")
    lines.extend(
        [
            "",
            "# Required Output",
            json.dumps(output_schema_for_persona(), ensure_ascii=False, indent=2),
            "Infer the persona only from the observed evidence. Do not reference the gold answer or any unavailable later-round information.",
        ]
    )
    return [
        {"role": "system", "content": persona_system_prompt()},
        {"role": "user", "content": "\n".join(lines)},
    ]


def build_retrieval_query_messages(instance: Dict[str, Any], persona_text: Optional[str] = None) -> List[Dict[str, str]]:
    lines = [
        "# Retrieval Objective",
        "Find repeated-PGG persona cards that are behaviorally analogous to the current MobLab participant or session.",
        "",
        f"Target measure: {instance['target_label']}",
        f"Target task: {instance['task_type']}",
        "",
        "# Observed Evidence",
    ]
    if instance["task_type"] == "task1":
        lines.append(instance["source_evidence_text"])
    else:
        lines.append(instance["round1_evidence_text"])
        lines.append(f"There are {instance['n_future_rounds']} unobserved future rounds in this session.")
    if persona_text:
        lines.extend(
            [
                "",
                "# Precomputed Persona",
                persona_text,
            ]
        )
    lines.extend(
        [
            "",
            "# Output Format",
            json.dumps(
                {
                    "search_query": "one compact paragraph",
                    "match_cues": ["cue 1", "cue 2", "cue 3"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "Focus on transferable behavioral cues rather than exact game mechanics.",
        ]
    )
    return [
        {"role": "system", "content": retrieval_query_system_prompt()},
        {"role": "user", "content": "\n".join(lines)},
    ]
