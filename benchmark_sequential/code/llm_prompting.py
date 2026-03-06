from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

EFFICIENCY_CONTEXT = (
    "Efficiency measures how close a group's total payoff is to a fully cooperative benchmark "
    "(everyone contributes full endowment each round). "
    "100% efficiency equals the fully cooperative benchmark payoff."
)

COLUMN_MEANINGS: dict[str, str] = {
    "selection_order": "Order in which configs were selected in the current sequential run.",
    "config_id": "Unique paired configuration identifier (punishment OFF vs ON variant pair).",
    "CONFIG_playerCount": "Number of players in the game.",
    "CONFIG_numRounds": "Number of rounds in the game.",
    "CONFIG_showNRounds": "1 if players know the number of rounds in advance, else 0.",
    "CONFIG_allOrNothing": "1 if contribution is binary (all-or-nothing), else 0 for continuous contribution.",
    "CONFIG_chat": "1 if player chat is enabled, else 0.",
    "CONFIG_defaultContribProp": "1 if endowment starts in public fund by default (opt-out), else 0.",
    "CONFIG_punishmentCost": "Coins spent to impose one unit of punishment.",
    "CONFIG_punishmentTech": "Coins deducted from punished player per coin spent punishing.",
    "CONFIG_rewardExists": "1 if reward mechanism exists, else 0.",
    "CONFIG_rewardCost": "Coins spent to grant one unit of reward.",
    "CONFIG_rewardTech": "Coins granted to rewarded player per coin spent rewarding.",
    "CONFIG_showOtherSummaries": "1 if peer outcomes are shown to players, else 0.",
    "CONFIG_showPunishmentId": "1 if identity of punisher/rewarder is shown, else 0.",
    "CONFIG_showRewardId": "1 if identity of rewarder is shown, else 0.",
    "CONFIG_MPCR": "Marginal per-capita return.",
    "control_itt_efficiency": "Observed efficiency when punishment is absent (same paired config).",
    "treatment_itt_efficiency": "Observed efficiency when punishment is present (same paired config).",
    "observed_treatment_effect": "treatment_itt_efficiency - control_itt_efficiency on selected history rows.",
    "gp_mu": "GP posterior mean for treatment_itt_efficiency on candidate.",
    "gp_std": "GP posterior standard deviation on candidate.",
    "bo_score_ei": "Expected Improvement acquisition score used by BO to build shortlist.",
    "ei_rank": "Rank by bo_score_ei descending (1 is highest).",
}


@dataclass(frozen=True)
class LLMSelectionResult:
    final_selected_config_ids: list[str]
    selected_config_ids_raw: list[str]
    confidence: float
    reasoning: str
    fallback_reason: Optional[str]
    raw_response: str
    parsed_response: Optional[dict[str, Any]]


def _table_to_csv_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "<empty>\n"
    return df.to_csv(index=False, float_format="%.6g")


def build_column_glossary_text(columns: list[str]) -> str:
    lines = []
    for col in columns:
        meaning = COLUMN_MEANINGS.get(col, "No description available.")
        lines.append(f"- {col}: {meaning}")
    return "\n".join(lines)


def build_llm_user_prompt(
    n_pairs: int,
    n_unselected: int,
    k_select: int,
    features: list[str],
    target: str,
    history_df: pd.DataFrame,
    shortlist_df: pd.DataFrame,
) -> str:
    shortlist_cols = ["ei_rank", "config_id"] + features + ["gp_mu", "gp_std", "bo_score_ei"]
    history_cols = ["selection_order", "config_id"] + features + [target, "observed_treatment_effect"]

    shortlist_text = _table_to_csv_text(shortlist_df[shortlist_cols])
    history_text = _table_to_csv_text(history_df[history_cols])
    glossary_text = build_column_glossary_text(list(dict.fromkeys(history_cols + shortlist_cols)))

    schema = {
        "reasoning": "string",
        "selected_config_ids": ["string"],
        "confidence": 0.0,
    }

    return (
        "Task: You are assisting adaptive experiment design in repeated public goods games (PGG).\n"
        "Each config_id defines a game design via CONFIG parameters. For each config, outcomes are paired by punishment OFF/ON.\n"
        "BO shortlist construction used before your decision:\n"
        "1) Fit a Gaussian Process surrogate on selected history (features -> treatment_itt_efficiency).\n"
        "2) Predict gp_mu and gp_std for unselected candidates.\n"
        "3) Compute bo_score_ei (Expected Improvement).\n"
        "4) Keep top-K candidates by bo_score_ei as shortlist.\n"
        "Your job: rerank this BO shortlist and choose the next batch of experiments to run.\n\n"
        "PGG context:\n"
        f"- {EFFICIENCY_CONTEXT}\n\n"
        "Column glossary sourced from benchmark_sequential/code_ref/build_positive_case_batch_input.py semantics:\n"
        f"{glossary_text}\n\n"
        "Domain-reasoning requirement:\n"
        "Before choosing IDs, reason through what you know about PGG behavior and mechanism design.\n"
        "Explicitly consider likely directional effects (and uncertainty) for these parameter families:\n"
        "- Game structure: player count, number of rounds, end-of-game information, all-or-nothing constraint.\n"
        "- Communication/information: chat, peer summaries, punisher/rewarder identity visibility.\n"
        "- Incentive strength: punishment cost/tech, reward existence/cost/tech, MPCR.\n"
        "- Baseline state: control_itt_efficiency as context for remaining improvement headroom.\n"
        "Also reason about interactions (e.g., chat x punishment strength, MPCR x sanctioning).\n\n"
        "Rules:\n"
        "1) Choose ONLY from shortlist config_id values.\n"
        f"2) Return exactly {k_select} unique config IDs in selection order.\n"
        "3) Prioritize expected generalization gain across future unseen games, not immediate treatment magnitude.\n"
        "4) Think through your reasoning first, then output final IDs.\n"
        "5) Return strict JSON only, no markdown.\n\n"
        "Current step summary:\n"
        f"- selected_pairs={n_pairs}\n"
        f"- unselected_pairs={n_unselected}\n"
        f"- requested_batch_size={k_select}\n\n"
        "Observed history table (all selected so far):\n"
        f"{history_text}\n"
        "Candidate shortlist table (top GP-EI candidates):\n"
        f"{shortlist_text}\n"
        "Return JSON with keys in this order: reasoning, selected_config_ids, confidence.\n"
        "Return JSON with this schema:\n"
        f"{json.dumps(schema)}\n"
    )


def build_prompt_with_overflow(
    n_pairs: int,
    n_unselected: int,
    k_select: int,
    features: list[str],
    target: str,
    history_df: pd.DataFrame,
    shortlist_df: pd.DataFrame,
    max_chars: int,
    overflow_policy: str,
) -> tuple[Optional[str], int, int, Optional[str]]:
    dropped = 0
    hist = history_df.copy()

    while True:
        prompt = build_llm_user_prompt(
            n_pairs=n_pairs,
            n_unselected=n_unselected,
            k_select=k_select,
            features=features,
            target=target,
            history_df=hist,
            shortlist_df=shortlist_df,
        )
        if len(prompt) <= max_chars:
            return prompt, int(len(hist)), dropped, None

        if overflow_policy != "trim_oldest":
            return None, int(len(hist)), dropped, f"unsupported_overflow_policy:{overflow_policy}"

        if len(hist) == 0:
            return None, 0, dropped, "prompt_too_long_after_trim"

        hist = hist.iloc[1:].reset_index(drop=True)
        dropped += 1


def extract_json_dict(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty_response")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    for block in fenced:
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        parsed = json.loads(text[first : last + 1])
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("json_parse_failed")


def parse_confidence(value: Any) -> float:
    try:
        val = float(value)
    except Exception:
        return 0.0
    return float(np.clip(val, 0.0, 1.0))


def sanitize_reasoning(text: Any) -> str:
    return "" if text is None else str(text).strip()


def finalize_llm_selection(
    parsed: Optional[dict[str, Any]],
    shortlist_ids: list[str],
    top_bo_score_ids: list[str],
    k_select: int,
    raw_response: str,
    fallback_reason: Optional[str],
    normalize_config_id: Callable[[Any], str],
) -> LLMSelectionResult:
    selected_raw: list[str] = []
    selected_final = list(top_bo_score_ids[:k_select])
    confidence = 0.0
    reasoning = ""

    if parsed is not None:
        raw_ids = parsed.get("selected_config_ids")
        if raw_ids is None and parsed.get("selected_config_id") is not None:
            raw_ids = [parsed.get("selected_config_id")]

        if isinstance(raw_ids, list):
            for item in raw_ids:
                cid = normalize_config_id(item)
                if cid and cid not in selected_raw:
                    selected_raw.append(cid)
        elif raw_ids is not None:
            cid = normalize_config_id(raw_ids)
            if cid:
                selected_raw.append(cid)

        confidence = parse_confidence(parsed.get("confidence", 0.0))
        reasoning = sanitize_reasoning(parsed.get("reasoning", parsed.get("rationale_short", "")))

    if parsed is None:
        fallback_reason = fallback_reason or "llm_parse_error"
    elif len(selected_raw) != k_select:
        fallback_reason = fallback_reason or "invalid_selected_count"
    elif any(cid not in shortlist_ids for cid in selected_raw):
        fallback_reason = fallback_reason or "selected_not_in_shortlist"
    else:
        selected_final = selected_raw

    if fallback_reason:
        selected_final = list(top_bo_score_ids[:k_select])

    return LLMSelectionResult(
        final_selected_config_ids=selected_final,
        selected_config_ids_raw=selected_raw,
        confidence=confidence,
        reasoning=reasoning,
        fallback_reason=fallback_reason,
        raw_response=raw_response,
        parsed_response=parsed,
    )
