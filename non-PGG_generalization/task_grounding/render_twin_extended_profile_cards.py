#!/usr/bin/env python3
"""Render deterministic Twin extended profiles into readable profile cards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from jsonschema import Draft202012Validator


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILES_JSONL = THIS_DIR / "output" / "twin_extended_profiles" / "twin_extended_profiles.jsonl"
DEFAULT_SCHEMA_JSON = THIS_DIR / "twin_extended_profile_card_schema.json"
DEFAULT_OUTPUT_DIR = THIS_DIR / "output" / "twin_extended_profile_cards"
DETAIL_MODES = ("compact", "standard", "audit", "pgg_prompt", "pgg_prompt_min")

MODE_SETTINGS = {
    "compact": {
        "background_max": 3,
        "signature_max": 3,
        "anchors_max": 4,
        "limits_max": 3,
    },
    "standard": {
        "background_max": 4,
        "signature_max": 4,
        "anchors_max": 7,
        "limits_max": 5,
    },
    "audit": {
        "background_max": 6,
        "signature_max": 6,
        "anchors_max": 7,
        "limits_max": 6,
    },
    "pgg_prompt": {
        "background_max": 4,
        "signature_max": 5,
        "anchors_max": 5,
        "limits_max": 2,
    },
    "pgg_prompt_min": {
        "background_max": 3,
        "signature_max": 4,
        "anchors_max": 4,
        "limits_max": 2,
    },
}

CUE_ORDER = [
    "cooperation_orientation",
    "conditional_cooperation",
    "norm_enforcement",
    "generosity_without_return",
    "exploitation_caution",
    "communication_coordination",
    "behavioral_stability",
]

CUE_DISPLAY_NAMES = {
    "cooperation_orientation": "Cooperation orientation",
    "conditional_cooperation": "Conditional cooperation",
    "norm_enforcement": "Norm enforcement",
    "generosity_without_return": "Generosity without return",
    "exploitation_caution": "Exploitation caution",
    "communication_coordination": "Communication/coordination",
    "behavioral_stability": "Behavioral stability",
}

BACKGROUND_ORDER = [
    ("age_bracket", "Age"),
    ("sex_assigned_at_birth", "Sex assigned at birth"),
    ("education_completed_raw", "Completed education"),
    ("employment_status", "Employment"),
    ("region", "Region"),
    ("relationship_status", "Relationship"),
]

SOCIAL_STYLE_CONSTRUCTION = [
    "Prosocial and competitive self-report: agreeableness/helpfulness items (QID25), cooperation/competition items (QID233), and social values (QID29).",
    "Direct one-shot social-allocation behavior: trust-game sending/returning (QID117-QID122), ultimatum offers and acceptance thresholds (QID224-QID230), and dictator giving (QID231).",
    "Social-emotional sensitivity: empathy items (QID232) and social-sensitivity / self-monitoring items (QID236).",
]

DECISION_STYLE_CONSTRUCTION = [
    "Intertemporal and risk choices: time-preference matrices (QID84, QID244-QID248) plus gain/loss lottery choices (QID250-QID252, QID276-QID279).",
    "Cognitive performance: financial-literacy, numeracy, CRT-style, and logic items from the cognitive-test blocks.",
    "Heuristics and consumer tasks: anchoring, framing, sale-search, WTA/WTP, and pricing-purchase judgments.",
]

PGG_PROMPT_SHARED_NOTE = [
    "These profiles summarize prior survey and behavioral-task evidence about each participant.",
    "Treat the cues as relative tendencies, not deterministic predictions for any single public-goods-game decision.",
    "Unless a player-specific limit is listed, shared methodological caveats apply to all players.",
]

PGG_PROMPT_SHARED_CAVEATS = [
    "Communication/coordination is indirect: the source tasks do not directly observe repeated group discussion.",
    "Norm-enforcement cues mainly reflect unfair-split resistance and revenge/forgiveness items, not direct repeated-game sanction use.",
    "Pricing, anchoring, and framing signals are secondary and relatively coarse compared with direct social-allocation behavior.",
]

PGG_PROMPT_CUE_GLOSSARY = {
    "cooperation_orientation": [
        "Meaning: blend of one-shot sharing behavior and cooperation/prosocial self-report.",
        "Built from: trust-game sending, dictator giving, cooperation/competition items, agreeableness/helpfulness items, and prosocial values.",
    ],
    "conditional_cooperation": [
        "Meaning: reciprocity and fairness-threshold sensitivity rather than a repeated-game reaction function.",
        "Built from: trust-game return behavior and ultimatum acceptance-threshold signals.",
    ],
    "norm_enforcement": [
        "Meaning: resistance to unfair splits and revenge/low-forgiveness cues in ultimatum-like contexts.",
        "Built from: ultimatum minimum acceptable amounts plus revenge/forgiveness self-report.",
    ],
    "generosity_without_return": [
        "Meaning: willingness to give when repayment incentives are weak or absent.",
        "Built from: dictator giving, trust-game sending, and prosocial/helpfulness cues.",
    ],
    "exploitation_caution": [
        "Meaning: guardedness against being taken advantage of.",
        "Built from: lower trustingness, stricter acceptance thresholds, uncertainty aversion, self-reliance, and revenge tendency.",
    ],
    "communication_coordination": [
        "Meaning: indirect cue for likely social expressiveness and coordination readiness.",
        "Built from: empathy, social-sensitivity/self-monitoring, and extraversion-related self-report.",
    ],
    "behavioral_stability": [
        "Meaning: rule-like internal consistency across self-regulation items.",
        "Built from: conscientiousness-related items, self-concept clarity, and lower volatility-related personality items.",
    ],
}

CUE_MEANINGS = {
    "cooperation_orientation": "Here, cooperation orientation means a blend of one-shot sharing behavior and cooperation/prosocial self-report. It is not a direct measure of repeated-group contribution.",
    "conditional_cooperation": "Here, conditional cooperation means how contingent the participant looks on reciprocity and fairness-threshold signals in the Twin tasks, rather than a round-by-round public-goods reaction function.",
    "norm_enforcement": "Here, norm enforcement means resistance to unfair splits and endorsement of revenge/low-forgiveness cues in ultimatum-like contexts. It is narrower than a generic punishment tendency.",
    "generosity_without_return": "Here, generosity without return means willingness to give in settings with weak or absent repayment incentives, mainly from dictator and trust-game sending behavior.",
    "exploitation_caution": "Here, exploitation caution means guardedness against being taken advantage of, reflected in lower trustingness, stricter acceptance thresholds, uncertainty aversion, and self-reliance cues.",
    "communication_coordination": "Here, communication/coordination is an indirect cue from extraversion, empathy, and social-sensitivity self-report. Twin does not directly observe repeated strategic group discussion.",
    "behavioral_stability": "Here, behavioral stability means how rule-like and internally consistent the participant looks across self-regulation items, not invariance across every future environment.",
}

CUE_COMPONENTS = {
    "cooperation_orientation": [
        "55% from cooperation-oriented self-report: cooperation/collectivism items (QID233), agreeableness/helpfulness items (QID25), and prosocial values (QID29).",
        "45% from direct sharing behavior: trust-game sending (QID117) and dictator giving (QID231).",
    ],
    "conditional_cooperation": [
        "65% from reciprocity signals: trust-game return behavior (QID118-QID122) and related prosocial-response cues.",
        "35% from fairness-threshold signals: ultimatum minimum acceptable amounts / rejection thresholds (QID226-QID230).",
    ],
    "norm_enforcement": [
        "75% from the fairness-enforcement dimension: mainly ultimatum minimum acceptable amounts, plus revenge and lower-forgiveness cues.",
        "25% from revenge-tendency self-report (QID27 with related agreeableness/forgiveness items in QID25).",
    ],
    "generosity_without_return": [
        "70% from the altruistic-sharing dimension: dictator giving (QID231), trust-game sending (QID117), and prosocial/helpfulness values.",
        "30% from the trustingness dimension: trust-game sending plus agreeableness/prosocial-value cues.",
    ],
    "exploitation_caution": [
        "100% from the lower-level exploitation-caution dimension.",
        "That lower-level dimension blends inverse trust-game sending, ultimatum minimum acceptable amounts, uncertainty-aversion items (QID238), self-reliance items (QID233), and revenge tendency.",
    ],
    "communication_coordination": [
        "40% from social-sensitivity / self-monitoring items (QID236).",
        "30% from empathy items (QID232).",
        "30% from extraversion-related items in the BFI-style personality block (QID25).",
    ],
    "behavioral_stability": [
        "40% from conscientiousness-related personality items (QID25).",
        "35% from self-concept clarity items (QID237).",
        "25% from lower neurotic volatility in the personality block (QID25).",
    ],
}


def mode_setting(detail_mode: str, key: str) -> int:
    return int(MODE_SETTINGS[detail_mode][key])


def social_style_construction(detail_mode: str) -> List[str]:
    if detail_mode in {"pgg_prompt", "pgg_prompt_min"}:
        return []
    if detail_mode == "compact":
        return [
            "Prosocial/competitive self-report from the personality and values blocks.",
            "Direct trust/ultimatum/dictator behavior plus empathy/social-sensitivity items.",
        ]
    if detail_mode == "audit":
        return SOCIAL_STYLE_CONSTRUCTION + [
            "This section is descriptive aggregation, not a single validated psychometric scale with fixed internal weighting.",
        ]
    return SOCIAL_STYLE_CONSTRUCTION


def decision_style_construction(detail_mode: str) -> List[str]:
    if detail_mode in {"pgg_prompt", "pgg_prompt_min"}:
        return []
    if detail_mode == "compact":
        return [
            "Time/risk choice tasks plus cognitive-test performance.",
            "Heuristics and consumer-choice tasks.",
        ]
    if detail_mode == "audit":
        return DECISION_STYLE_CONSTRUCTION + [
            "Different subcomponents can move in opposite directions; this section is a structured summary over heterogeneous task families.",
        ]
    return DECISION_STYLE_CONSTRUCTION


def cue_meaning_text(cue: str, payload: Dict[str, Any], detail_mode: str) -> str:
    base = f"{CUE_DISPLAY_NAMES[cue]} is {label_pretty(payload['label'])} ({payload['score_0_to_100']}/100)."
    if detail_mode == "pgg_prompt_min":
        return ""
    if detail_mode == "pgg_prompt":
        prompt_meanings = {
            "cooperation_orientation": " Blend of one-shot sharing behavior and cooperation/prosocial self-report.",
            "conditional_cooperation": " Reciprocity and fairness-threshold sensitivity rather than a repeated-game strategy.",
            "norm_enforcement": " Mainly about unfair-split resistance and revenge/forgiveness cues.",
            "generosity_without_return": " Mainly about giving when repayment is weak or absent.",
            "exploitation_caution": " Mainly about guardedness against exploitation.",
            "communication_coordination": " Indirect cue from self-report; the source tasks do not observe repeated group discussion.",
            "behavioral_stability": " Rule-like consistency across self-regulation items.",
        }
        return base + prompt_meanings[cue]
    if detail_mode == "compact":
        compact_meanings = {
            "cooperation_orientation": " Blend of one-shot sharing behavior and cooperation/prosocial self-report.",
            "conditional_cooperation": " Blend of reciprocity and fairness-threshold signals rather than a repeated-game strategy.",
            "norm_enforcement": " Mainly about unfair-split resistance in ultimatum-like contexts, not generic punishment.",
            "generosity_without_return": " Mainly about giving when repayment is weak or absent.",
            "exploitation_caution": " Mainly about guardedness against being taken advantage of.",
            "communication_coordination": " Indirect cue from self-report; the source tasks do not observe repeated group communication.",
            "behavioral_stability": " Rule-like consistency across self-regulation items, not invariance across all future settings.",
        }
        return base + compact_meanings[cue]
    return base + " " + CUE_MEANINGS[cue]


def cue_components(cue: str, detail_mode: str) -> List[str]:
    if detail_mode in {"pgg_prompt", "pgg_prompt_min"}:
        return []
    if detail_mode == "compact":
        return CUE_COMPONENTS[cue][:2]
    if detail_mode == "audit":
        return CUE_COMPONENTS[cue] + [f"Builder rationale: {cue.replace('_', ' ')} is a transfer cue, not a direct PGG forecast."]
    return CUE_COMPONENTS[cue]

HEADLINE_SPECS = [
    {
        "path": ("derived_dimensions", "decision_style", "patience"),
        "family": "time_horizon",
        "phrases": {
            "very_high": "patient",
            "high": "patient",
            "low": "short-horizon",
            "very_low": "short-horizon",
            "medium": "moderately patient",
        },
    },
    {
        "path": ("pgg_relevant_cues", "behavioral_stability"),
        "family": "stability",
        "phrases": {
            "very_high": "behaviorally stable",
            "high": "behaviorally stable",
            "low": "more behaviorally variable",
            "very_low": "more behaviorally variable",
            "medium": "moderately stable",
        },
    },
    {
        "path": ("derived_dimensions", "self_regulation_and_affect", "uncertainty_aversion"),
        "family": "uncertainty",
        "phrases": {
            "very_high": "structure-seeking",
            "high": "structure-seeking",
            "low": "comfortable with uncertainty",
            "very_low": "comfortable with uncertainty",
            "medium": "moderately structure-seeking",
        },
    },
    {
        "path": ("derived_dimensions", "self_regulation_and_affect", "empathy"),
        "family": "empathy",
        "phrases": {
            "very_high": "empathic",
            "high": "empathic",
            "low": "lower-empathy",
            "very_low": "lower-empathy",
            "medium": "moderately empathic",
        },
    },
    {
        "path": ("derived_dimensions", "self_regulation_and_affect", "cooperation_orientation"),
        "family": "cooperation",
        "phrases": {
            "very_high": "cooperation-oriented",
            "high": "cooperation-oriented",
            "low": "less cooperation-oriented",
            "very_low": "less cooperation-oriented",
            "medium": "moderately cooperative",
        },
    },
    {
        "path": ("pgg_relevant_cues", "communication_coordination"),
        "family": "communication",
        "phrases": {
            "very_high": "socially adaptive",
            "high": "socially adaptive",
            "low": "less socially expressive",
            "very_low": "less socially expressive",
            "medium": "moderately socially adaptive",
        },
    },
    {
        "path": ("derived_dimensions", "decision_style", "cognitive_reflection"),
        "family": "cognition",
        "phrases": {
            "very_high": "reflective",
            "high": "reflective",
            "low": "more intuitive",
            "very_low": "more intuitive",
            "medium": "moderately reflective",
        },
    },
    {
        "path": ("derived_dimensions", "decision_style", "logical_reasoning"),
        "family": "cognition",
        "phrases": {
            "very_high": "strong logical reasoner",
            "high": "strong logical reasoner",
            "low": "weaker logical reasoner",
            "very_low": "weaker logical reasoner",
            "medium": "moderate logical reasoner",
        },
    },
    {
        "path": ("pgg_relevant_cues", "generosity_without_return"),
        "family": "generosity",
        "phrases": {
            "very_high": "generosity-forward",
            "high": "generosity-forward",
            "low": "resource-protective",
            "very_low": "resource-protective",
            "medium": "moderately generous",
        },
    },
    {
        "path": ("pgg_relevant_cues", "norm_enforcement"),
        "family": "fairness",
        "phrases": {
            "very_high": "fairness-enforcing",
            "high": "fairness-enforcing",
            "low": "inequality-tolerant",
            "very_low": "inequality-tolerant",
            "medium": "moderately fairness-sensitive",
        },
    },
    {
        "path": ("pgg_relevant_cues", "exploitation_caution"),
        "family": "guardedness",
        "phrases": {
            "very_high": "guarded against exploitation",
            "high": "guarded against exploitation",
            "low": "less guarded",
            "very_low": "less guarded",
            "medium": "moderately guarded",
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-jsonl", type=Path, default=DEFAULT_PROFILES_JSONL)
    parser.add_argument("--schema-json", type=Path, default=DEFAULT_SCHEMA_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--detail-mode", choices=DETAIL_MODES, default="standard")
    parser.add_argument("--all-modes", action="store_true")
    parser.add_argument("--pid", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--markdown-limit", type=int, default=12)
    return parser.parse_args()


def load_profiles(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def get_nested(obj: Dict[str, Any], path: Sequence[str]) -> Optional[Dict[str, Any]]:
    cur: Any = obj
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if not isinstance(cur, dict):
        return None
    return cur


def feature_map(block: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {item["name"]: item for item in block.get("summary_features", [])}


def harmonized_feature_map(profile: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in profile["background_context"]["harmonized_features"]:
        out[item["name"]] = item["value"].get("raw")
    return out


def confidence_rank(value: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(value, 0)


def label_pretty(value: str) -> str:
    return value.replace("_", " ")


def score_distance(payload: Dict[str, Any]) -> int:
    if payload.get("label") == "unknown":
        return 0
    return abs(int(payload.get("score_0_to_100", 50)) - 50)


def pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{int(round(value * 100))}%"


def money(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    if float(value).is_integer():
        return f"${int(value)}"
    return f"${value:.1f}"


def bounded_score(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{int(round(value))}/100"


def pick_headline_descriptors(profile: Dict[str, Any]) -> List[str]:
    candidates: List[Tuple[int, int, int, str, str]] = []
    for spec in HEADLINE_SPECS:
        payload = get_nested(profile, spec["path"])
        if payload is None:
            continue
        label = payload.get("label", "unknown")
        phrase = spec["phrases"].get(label)
        if not phrase:
            continue
        strength = 1 if label in {"very_high", "high", "low", "very_low"} else 0
        candidates.append(
            (
                strength,
                score_distance(payload),
                confidence_rank(payload.get("confidence", "low")),
                spec["family"],
                phrase,
            )
        )
    candidates.sort(reverse=True)
    chosen: List[str] = []
    seen_families: set[str] = set()
    for _, _, _, family, phrase in candidates:
        if family in seen_families:
            continue
        if phrase not in chosen:
            chosen.append(phrase)
            seen_families.add(family)
        if len(chosen) == 3:
            break
    if not chosen:
        return ["mixed", "evidence-rich", "behavior profile"]
    while len(chosen) < 3:
        chosen.append("mixed")
    return chosen[:3]


def build_headline(profile: Dict[str, Any]) -> str:
    descriptors = pick_headline_descriptors(profile)
    return ", ".join([descriptors[0].capitalize(), descriptors[1], descriptors[2]])


def build_background(profile: Dict[str, Any], detail_mode: str) -> Dict[str, Any]:
    feats = harmonized_feature_map(profile)
    selected_context: List[Dict[str, str]] = []
    for raw_name, label in BACKGROUND_ORDER:
        value = feats.get(raw_name)
        if value in (None, "", "unknown"):
            continue
        selected_context.append({"name": label, "value": str(value)})
    selected_context = selected_context[: mode_setting(detail_mode, "background_max")]
    if len(selected_context) < 3:
        for raw_name, value in feats.items():
            if value in (None, "", "unknown"):
                continue
            label = raw_name.replace("_", " ").capitalize()
            item = {"name": label, "value": str(value)}
            if item not in selected_context:
                selected_context.append(item)
            if len(selected_context) >= 3:
                break
    summary = "; ".join(f"{item['name']}: {item['value']}" for item in selected_context) + "."
    return {
        "summary": summary,
        "selected_context": selected_context[: mode_setting(detail_mode, "background_max")],
    }


def select_distinctive_dimensions(
    dimension_payloads: Dict[str, Dict[str, Any]],
    *,
    max_items: int = 2,
) -> List[Tuple[str, Dict[str, Any]]]:
    scored = [
        (name, payload)
        for name, payload in dimension_payloads.items()
        if payload.get("label") != "unknown"
    ]
    scored.sort(
        key=lambda item: (
            score_distance(item[1]),
            confidence_rank(item[1].get("confidence", "low")),
        ),
        reverse=True,
    )
    return scored[:max_items]


def dimension_phrase(name: str, payload: Dict[str, Any]) -> str:
    return f"{name.replace('_', ' ')} {label_pretty(payload['label'])}"


def build_social_style_section(profile: Dict[str, Any], detail_mode: str) -> Dict[str, Any]:
    social = profile["derived_dimensions"]["social_preferences"]
    affect = profile["derived_dimensions"]["self_regulation_and_affect"]
    dims = {
        **social,
        "cooperation_orientation": affect["cooperation_orientation"],
        "competition_orientation": affect["competition_orientation"],
        "empathy": affect["empathy"],
    }
    top = select_distinctive_dimensions(dims, max_items=2)
    top_text = ", ".join(dimension_phrase(name, payload) for name, payload in top) if top else "mixed evidence"

    cooperation_score = affect["cooperation_orientation"]["score_0_to_100"]
    generosity_score = profile["pgg_relevant_cues"]["generosity_without_return"]["score_0_to_100"]
    cautions = [
        "This section mixes questionnaire evidence with one-shot trust/ultimatum/dictator tasks; it is not a single pure personality scale.",
    ]
    if abs(cooperation_score - generosity_score) >= 20:
        cautions.append("Cooperation-related self-report and direct giving behavior do not fully line up, so the social picture is not one-dimensional.")
    else:
        cautions.append("Self-report prosociality and direct allocation behavior point in a similar broad direction, but they still come from different task formats.")
    if profile["pgg_relevant_cues"]["communication_coordination"]["confidence"] == "low":
        cautions.append("Communication-related readings are especially indirect because the source tasks do not observe repeated strategic group communication.")

    if detail_mode in {"pgg_prompt", "pgg_prompt_min"}:
        cautions = []

    return {
        "summary": (
            "Social style summarizes direct social-allocation behavior together with prosocial/competitive and empathy-related self-report. "
            f"The most distinctive supported social signals are {top_text}."
        ),
        "constructed_from": social_style_construction(detail_mode),
        "cautions": cautions[: (2 if detail_mode == "compact" else 3)],
    }


def build_decision_style_section(profile: Dict[str, Any], detail_mode: str) -> Dict[str, Any]:
    decision = profile["derived_dimensions"]["decision_style"]
    consumer = profile["derived_dimensions"]["consumer_style"]
    dims = {
        **decision,
        **consumer,
    }
    top = select_distinctive_dimensions(dims, max_items=2)
    top_text = ", ".join(dimension_phrase(name, payload) for name, payload in top) if top else "mixed evidence"

    cautions = [
        "This section combines several different task families: time/risk choices, cognitive tests, heuristics tasks, and consumer-choice judgments.",
        "A strong result in one decision-style component does not imply the same direction in the others.",
    ]
    cautions.append("Anchoring/framing and pricing-related inferences are based on relatively sparse task sets and should be read as coarse tendencies.")

    if detail_mode in {"pgg_prompt", "pgg_prompt_min"}:
        cautions = []

    return {
        "summary": (
            "Decision style summarizes intertemporal/risk choices, cognitive-test performance, heuristics/biases, and consumer-choice behavior. "
            f"The most distinctive supported decision signals are {top_text}."
        ),
        "constructed_from": decision_style_construction(detail_mode),
        "cautions": cautions[: (2 if detail_mode == "compact" else 3)],
    }


def build_summary(profile: Dict[str, Any]) -> str:
    descriptors = pick_headline_descriptors(profile)
    if len(descriptors) >= 3:
        descriptor_text = f"{descriptors[0]}, {descriptors[1]}, and {descriptors[2]}"
    elif len(descriptors) == 2:
        descriptor_text = f"{descriptors[0]} and {descriptors[1]}"
    else:
        descriptor_text = descriptors[0]
    coop = profile["derived_dimensions"]["self_regulation_and_affect"]["cooperation_orientation"]["score_0_to_100"]
    generosity = profile["pgg_relevant_cues"]["generosity_without_return"]["score_0_to_100"]
    norm = profile["pgg_relevant_cues"]["norm_enforcement"]["score_0_to_100"]
    if coop >= 65 and generosity >= 60:
        social_line = "Cooperative self-report and direct allocation behavior point in the same general direction."
    elif coop >= 65 and generosity <= 40:
        social_line = "Cooperative self-report is stronger than direct allocation behavior."
    elif coop <= 35 and generosity <= 40:
        social_line = "Both self-report and direct allocation behavior lean resource-protective rather than generous."
    else:
        social_line = "Social evidence is mixed between self-report and direct allocation tasks."
    if norm <= 35:
        social_line += " Fairness-enforcement signals are relatively weak."
    elif norm >= 65:
        social_line += " Fairness-enforcement signals are relatively strong."
    return (
        f"Strongest supported tendencies are {descriptor_text}. "
        f"{social_line}"
    )


def anchor_from_personality(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fmap = feature_map(block)
    wanted = ["empathy", "cooperation_orientation", "uncertainty_aversion"]
    parts: List[str] = []
    refs: List[str] = []
    for name in wanted:
        item = fmap.get(name)
        if not item:
            continue
        value = item["value"]
        if "label" in value:
            parts.append(f"{name.replace('_', ' ')} {label_pretty(value['label'])}")
            refs.extend(item["evidence_refs"])
    if not parts:
        return None
    return {
        "title": "Self-report pattern",
        "detail": "Self-report block emphasizes " + ", ".join(parts[:3]) + ".",
        "evidence_refs": list(dict.fromkeys(refs)),
    }


def anchor_from_social_game(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fmap = feature_map(block)
    send = fmap.get("trust_send_amount", {}).get("value", {}).get("raw")
    ret = fmap.get("trust_return_share_mean", {}).get("value", {}).get("raw")
    ult_offer = fmap.get("ultimatum_offer_to_other", {}).get("value", {}).get("raw")
    ult_min = fmap.get("ultimatum_min_acceptable_to_self", {}).get("value", {}).get("raw")
    dictator = fmap.get("dictator_offer_to_other", {}).get("value", {}).get("raw")
    if all(value is None for value in [send, ret, ult_offer, ult_min, dictator]):
        return None
    refs: List[str] = []
    for key in [
        "trust_send_amount",
        "trust_return_share_mean",
        "ultimatum_offer_to_other",
        "ultimatum_min_acceptable_to_self",
        "dictator_offer_to_other",
    ]:
        if key in fmap:
            refs.extend(fmap[key]["evidence_refs"])
    detail = (
        f"Direct social-allocation tasks: trust send {money(send)}/$5, "
        f"mean trust return {pct(ret)}, ultimatum offer {money(ult_offer)}/$5, "
        f"minimum acceptable {money(ult_min)}/$5, dictator give {money(dictator)}/$5."
    )
    return {
        "title": "Direct allocation tasks",
        "detail": detail,
        "evidence_refs": list(dict.fromkeys(refs)),
    }


def anchor_from_econ(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fmap = feature_map(block)
    later = fmap.get("later_choice_rate", {}).get("value", {}).get("raw")
    gains = fmap.get("lottery_choice_rate_gains", {}).get("value", {}).get("raw")
    losses = fmap.get("lottery_choice_rate_losses", {}).get("value", {}).get("raw")
    mental = fmap.get("mental_accounting_endorsement_rate", {}).get("value", {}).get("raw")
    if all(value is None for value in [later, gains, losses, mental]):
        return None
    refs: List[str] = []
    for key in [
        "later_choice_rate",
        "lottery_choice_rate_gains",
        "lottery_choice_rate_losses",
        "mental_accounting_endorsement_rate",
    ]:
        if key in fmap:
            refs.extend(fmap[key]["evidence_refs"])
    detail = (
        f"Time/risk tasks: later-choice rate {pct(later)}, gain-lottery choice {pct(gains)}, "
        f"loss-lottery choice {pct(losses)}, mental-accounting endorsement {pct(mental)}."
    )
    return {
        "title": "Time and risk profile",
        "detail": detail,
        "evidence_refs": list(dict.fromkeys(refs)),
    }


def anchor_from_cognition(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fmap = feature_map(block)
    fin = fmap.get("financial_literacy_accuracy", {}).get("value", {}).get("raw")
    num = fmap.get("numeracy_accuracy", {}).get("value", {}).get("raw")
    crt = fmap.get("cognitive_reflection_accuracy", {}).get("value", {}).get("raw")
    logic = fmap.get("logical_reasoning_accuracy", {}).get("value", {}).get("raw")
    if all(value is None for value in [fin, num, crt, logic]):
        return None
    refs: List[str] = []
    for key in [
        "financial_literacy_accuracy",
        "numeracy_accuracy",
        "cognitive_reflection_accuracy",
        "logical_reasoning_accuracy",
    ]:
        if key in fmap:
            refs.extend(fmap[key]["evidence_refs"])
    detail = (
        f"Cognitive tasks: financial literacy {pct(fin)}, numeracy {pct(num)}, "
        f"cognitive reflection {pct(crt)}, logical reasoning {pct(logic)}."
    )
    return {
        "title": "Cognitive performance",
        "detail": detail,
        "evidence_refs": list(dict.fromkeys(refs)),
    }


def anchor_from_heuristics(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fmap = feature_map(block)
    search = fmap.get("search_willingness", {}).get("value", {}).get("raw")
    anchor = fmap.get("anchor_pull_average", {}).get("value", {}).get("score_0_to_100")
    framing = fmap.get("framing_reference_sensitivity", {}).get("value", {}).get("score_0_to_100")
    ratio_bias = fmap.get("ratio_bias_small_tray_choice", {}).get("value", {}).get("raw")
    if all(value is None for value in [search, anchor, framing, ratio_bias]):
        return None
    refs: List[str] = []
    for key in [
        "search_willingness",
        "anchor_pull_average",
        "framing_reference_sensitivity",
        "ratio_bias_small_tray_choice",
    ]:
        if key in fmap:
            refs.extend(fmap[key]["evidence_refs"])
    detail = (
        f"Heuristics/biases block: search willingness {pct(search)}, anchor pull {bounded_score(anchor)}, "
        f"framing/reference sensitivity {bounded_score(framing)}, ratio-bias small-tray choice {int(ratio_bias) if ratio_bias is not None else 'NA'}."
    )
    return {
        "title": "Heuristics and biases",
        "detail": detail,
        "evidence_refs": list(dict.fromkeys(refs)),
    }


def anchor_from_pricing(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fmap = feature_map(block)
    yes_rate = fmap.get("purchase_yes_rate", {}).get("value", {}).get("raw")
    if yes_rate is None:
        return None
    refs = fmap.get("purchase_yes_rate", {}).get("evidence_refs", [])
    return {
        "title": "Pricing and consumer choices",
        "detail": f"Across the product-pricing block, the purchase-yes rate is {pct(yes_rate)}.",
        "evidence_refs": list(dict.fromkeys(refs)),
    }


def anchor_from_open_text(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fmap = feature_map(block)
    actual = bool(fmap.get("actual_text_present", {}).get("value", {}).get("raw"))
    aspire = bool(fmap.get("aspire_text_present", {}).get("value", {}).get("raw"))
    ought = bool(fmap.get("ought_text_present", {}).get("value", {}).get("raw"))
    word_count = fmap.get("forward_flow_word_count", {}).get("value", {}).get("raw")
    if not any([actual, aspire, ought]) and word_count is None:
        return None
    refs: List[str] = []
    for key in ["actual_text_present", "aspire_text_present", "ought_text_present", "forward_flow_word_count"]:
        if key in fmap:
            refs.extend(fmap[key]["evidence_refs"])
    present_bits = []
    if actual:
        present_bits.append("actual-self text")
    if aspire:
        present_bits.append("aspired-self text")
    if ought:
        present_bits.append("ought-self text")
    details = ", ".join(present_bits) if present_bits else "open-text material"
    if word_count is not None:
        details += f"; forward-flow response length {int(word_count)} words"
    return {
        "title": "Open-text material",
        "detail": f"Qualitative material is available via {details}.",
        "evidence_refs": list(dict.fromkeys(refs)),
    }


def build_observed_anchors(profile: Dict[str, Any], detail_mode: str) -> List[Dict[str, Any]]:
    observed = profile["observed_in_twin"]
    anchors = [
        anchor_from_social_game(observed["social_game_behavior"]),
        anchor_from_personality(observed["personality_and_self_report"]),
        anchor_from_econ(observed["economic_preferences_non_social"]),
        anchor_from_cognition(observed["cognitive_performance"]),
        anchor_from_heuristics(observed["heuristics_and_biases"]),
        anchor_from_pricing(observed["pricing_and_consumer_choice"]),
        anchor_from_open_text(observed["open_text_responses"]),
    ]
    filtered = [anchor for anchor in anchors if anchor is not None]
    return filtered[: mode_setting(detail_mode, "anchors_max")]


def build_transfer_relevance(profile: Dict[str, Any], detail_mode: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cue in CUE_ORDER:
        payload = profile["pgg_relevant_cues"][cue]
        out.append(
            {
                "cue": cue,
                "label": payload["label"],
                "score_0_to_100": payload["score_0_to_100"],
                "confidence": payload["confidence"],
                "meaning_here": cue_meaning_text(cue, payload, detail_mode),
                "constructed_from": cue_components(cue, detail_mode),
                "scope_note": "" if detail_mode in {"pgg_prompt", "pgg_prompt_min"} else payload["scope_note"],
                "evidence_refs": payload["evidence_refs"],
            }
        )
    return out


def build_limits(profile: Dict[str, Any], detail_mode: str) -> List[Dict[str, Any]]:
    if detail_mode not in {"pgg_prompt", "pgg_prompt_min"}:
        return profile["uncertainties"][: mode_setting(detail_mode, "limits_max")]

    limits: List[Dict[str, Any]] = []
    cooperation_score = profile["derived_dimensions"]["self_regulation_and_affect"]["cooperation_orientation"]["score_0_to_100"]
    generosity_score = profile["pgg_relevant_cues"]["generosity_without_return"]["score_0_to_100"]
    if abs(cooperation_score - generosity_score) >= 20:
        limits.append(
            {
                "topic": "Prosociality mismatch",
                "note": "Cooperative self-report and direct one-shot giving behavior do not fully align for this participant.",
                "evidence_refs": [
                    "Personality::QID25",
                    "Personality::QID233",
                    "Economic preferences::QID117",
                    "Economic preferences::QID231",
                ],
            }
        )

    communication_payload = profile["pgg_relevant_cues"]["communication_coordination"]
    if communication_payload["confidence"] == "low" and score_distance(communication_payload) >= 15:
        limits.append(
            {
                "topic": "Communication uncertainty",
                "note": "Communication/coordination is distinctive here but remains indirect because it comes from self-report and social-sensitivity items rather than observed group discussion.",
                "evidence_refs": communication_payload["evidence_refs"],
            }
        )

    return limits[: mode_setting(detail_mode, "limits_max")]


def build_card(profile: Dict[str, Any], detail_mode: str) -> Dict[str, Any]:
    return {
        "profile_card_version": "twin_extended_profile_card_v4",
        "detail_mode": detail_mode,
        "participant": {
            "pid": profile["participant"]["pid"],
            "source_dataset": profile["participant"]["source_dataset"],
            "source_profile_version": profile["profile_spec_version"],
        },
        "headline": build_headline(profile),
        "summary": build_summary(profile),
        "background": build_background(profile, detail_mode),
        "behavioral_signature": profile["behavioral_signature"][: mode_setting(detail_mode, "signature_max")],
        "social_style": build_social_style_section(profile, detail_mode),
        "decision_style": build_decision_style_section(profile, detail_mode),
        "observed_anchors": build_observed_anchors(profile, detail_mode),
        "transfer_relevance": build_transfer_relevance(profile, detail_mode),
        "limits": build_limits(profile, detail_mode),
    }


def validate_card(card: Dict[str, Any], validator: Draft202012Validator) -> None:
    errors = sorted(validator.iter_errors(card), key=lambda err: list(err.absolute_path))
    if not errors:
        return
    first = errors[0]
    path = ".".join(str(part) for part in first.absolute_path) or "<root>"
    raise ValueError(f"Card failed schema validation at {path}: {first.message}")


def build_markdown(cards: List[Dict[str, Any]], total_count: int, detail_mode: str) -> str:
    lines: List[str] = []
    if detail_mode == "pgg_prompt":
        lines.append("# Behavior Profile Cards For PGG Prompting")
    elif detail_mode == "pgg_prompt_min":
        lines.append("# Minimal Behavior Profile Cards For PGG Prompting")
    else:
        lines.append("# Twin Extended Profile Cards")
    lines.append("")
    lines.append(f"Detail mode: `{detail_mode}`")
    lines.append("")
    if detail_mode in {"pgg_prompt", "pgg_prompt_min"}:
        lines.append("Shared prompt note is written separately to `shared_prompt_notes.md` in this output directory.")
        lines.append("")
    lines.append(f"Preview cards rendered: {len(cards)} of {total_count}")
    lines.append("")
    for idx, card in enumerate(cards, start=1):
        lines.append(f"## Card {idx}")
        lines.append("")
        lines.append(f"**PID**: `{card['participant']['pid']}`")
        lines.append("")
        lines.append(f"**Headline**: {card['headline']}")
        lines.append("")
        lines.append(f"**Summary**: {card['summary']}")
        lines.append("")
        lines.append(f"**Background**: {card['background']['summary']}")
        lines.append("")
        lines.append("**Behavioral Signature**")
        for item in card["behavioral_signature"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("**Social Style**")
        lines.append(f"- {card['social_style']['summary']}")
        if card["social_style"]["constructed_from"]:
            lines.append("- Constructed from:")
            for item in card["social_style"]["constructed_from"]:
                lines.append(f"  - {item}")
        for item in card["social_style"]["cautions"]:
            lines.append(f"- Caution: {item}")
        lines.append("")
        lines.append("**Decision Style**")
        lines.append(f"- {card['decision_style']['summary']}")
        if card["decision_style"]["constructed_from"]:
            lines.append("- Constructed from:")
            for item in card["decision_style"]["constructed_from"]:
                lines.append(f"  - {item}")
        for item in card["decision_style"]["cautions"]:
            lines.append(f"- Caution: {item}")
        lines.append("")
        lines.append("**Observed Anchors**")
        for item in card["observed_anchors"]:
            lines.append(f"- {item['title']}: {item['detail']}")
        lines.append("")
        lines.append("**Transfer-Relevant Cues**")
        for item in card["transfer_relevance"]:
            lines.append(
                f"- `{CUE_DISPLAY_NAMES[item['cue']]}`: {label_pretty(item['label'])} ({item['score_0_to_100']}), confidence {item['confidence']}"
            )
            if item["meaning_here"]:
                lines.append(f"  Meaning here: {item['meaning_here']}")
            if item["constructed_from"]:
                lines.append("  Constructed from:")
                for sub in item["constructed_from"]:
                    lines.append(f"  - {sub}")
            if item["scope_note"]:
                lines.append(f"  Boundary: {item['scope_note']}")
        lines.append("")
        if card["limits"]:
            lines.append("**Limits**")
            for item in card["limits"]:
                lines.append(f"- {item['topic']}: {item['note']}")
            lines.append("")
    return "\n".join(lines)


def build_shared_prompt_notes() -> str:
    lines: List[str] = []
    lines.append("# Shared Prompt Notes")
    lines.append("")
    lines.append("## General Interpretation Note")
    lines.append("")
    for item in PGG_PROMPT_SHARED_NOTE:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Shared Caveats")
    lines.append("")
    for item in PGG_PROMPT_SHARED_CAVEATS:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Cue Glossary")
    lines.append("")
    for cue in CUE_ORDER:
        lines.append(f"### {CUE_DISPLAY_NAMES[cue]}")
        for item in PGG_PROMPT_CUE_GLOSSARY[cue]:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_csv_rows(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for card in cards:
        row: Dict[str, Any] = {
            "pid": card["participant"]["pid"],
            "detail_mode": card["detail_mode"],
            "headline": card["headline"],
            "summary": card["summary"],
            "background_summary": card["background"]["summary"],
            "social_style_summary": card["social_style"]["summary"],
            "decision_style_summary": card["decision_style"]["summary"],
        }
        for idx in range(6):
            row[f"behavioral_signature_{idx + 1}"] = (
                card["behavioral_signature"][idx] if idx < len(card["behavioral_signature"]) else ""
            )
        for cue in CUE_ORDER:
            payload = next(item for item in card["transfer_relevance"] if item["cue"] == cue)
            row[f"{cue}_label"] = payload["label"]
            row[f"{cue}_score"] = payload["score_0_to_100"]
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_filter_profiles(
    profiles: Iterable[Dict[str, Any]],
    *,
    pid: Optional[str],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for profile in profiles:
        if pid is not None and profile["participant"]["pid"] != pid:
            continue
        selected.append(profile)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def write_mode_outputs(
    *,
    cards: List[Dict[str, Any]],
    output_dir: Path,
    detail_mode: str,
    markdown_limit: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "twin_extended_profile_cards.jsonl"
    preview_json = output_dir / "preview_twin_extended_profile_cards.json"
    output_md = output_dir / "preview_twin_extended_profile_cards.md"
    output_csv = output_dir / "twin_extended_profile_cards.csv"
    shared_prompt_notes = output_dir / "shared_prompt_notes.md"

    with output_jsonl.open("w", encoding="utf-8") as f:
        for card in cards:
            f.write(json.dumps(card, ensure_ascii=False) + "\n")
    preview_json.write_text(json.dumps(cards[:3], ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_markdown(cards[: markdown_limit], len(cards), detail_mode), encoding="utf-8")
    write_csv(build_csv_rows(cards), output_csv)
    if detail_mode in {"pgg_prompt", "pgg_prompt_min"}:
        shared_prompt_notes.write_text(build_shared_prompt_notes(), encoding="utf-8")

    print(f"[{detail_mode}] Wrote jsonl:   {output_jsonl}")
    print(f"[{detail_mode}] Wrote preview: {preview_json}")
    print(f"[{detail_mode}] Wrote md:      {output_md}")
    print(f"[{detail_mode}] Wrote csv:     {output_csv}")
    if detail_mode in {"pgg_prompt", "pgg_prompt_min"}:
        print(f"[{detail_mode}] Wrote note:    {shared_prompt_notes}")
    print(f"[{detail_mode}] Cards:         {len(cards)}")


def main() -> int:
    args = parse_args()
    profiles = load_profiles(args.profiles_jsonl)
    profiles = maybe_filter_profiles(profiles, pid=args.pid, limit=args.limit)
    schema = json.loads(args.schema_json.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    detail_modes = DETAIL_MODES if args.all_modes else (args.detail_mode,)
    for detail_mode in detail_modes:
        cards: List[Dict[str, Any]] = []
        for profile in profiles:
            card = build_card(profile, detail_mode)
            validate_card(card, validator)
            cards.append(card)
        write_mode_outputs(
            cards=cards,
            output_dir=args.output_dir / detail_mode,
            detail_mode=detail_mode,
            markdown_limit=args.markdown_limit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
