#!/usr/bin/env python3
"""Render leakage-controlled Twin card ablations for PGG forecasting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
DEFAULT_PROFILES_JSONL = (
    PROJECT_ROOT
    / "non-PGG_generalization"
    / "twin_profiles"
    / "output"
    / "twin_extended_profiles"
    / "twin_extended_profiles.jsonl"
)
DEFAULT_OUTPUT_DIR = THIS_DIR / "output" / "twin_pgg_six_category_ablations"


VARIANTS = (
    "twin_pgg_background_only",
    "twin_pgg_direct_social_only",
    "twin_pgg_self_report_social_only",
    "twin_pgg_non_social_econ_only",
    "twin_pgg_cognitive_only",
    "twin_pgg_misc_heuristics_pricing_text_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-jsonl", type=Path, default=DEFAULT_PROFILES_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--variant",
        action="append",
        choices=VARIANTS,
        help="Variant to render. Can be repeated. Defaults to all variants.",
    )
    parser.add_argument("--pid", type=str, default=None, help="Render only one Twin participant.")
    parser.add_argument("--limit", type=int, default=None, help="Render at most N profiles.")
    return parser.parse_args()


def read_profiles(path: Path, *, pid: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            profile = json.loads(line)
            profile_pid = str((profile.get("participant") or {}).get("pid") or "")
            if pid is not None and profile_pid != str(pid):
                continue
            rows.append(profile)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def feature_map(block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("name")): item
        for item in block.get("summary_features", [])
        if item.get("name")
    }


def feature_score(item: dict[str, Any] | None) -> float | None:
    if not item:
        return None
    value = item.get("value") or {}
    score = value.get("score_0_to_100")
    return float(score) if score is not None else None


def feature_label(item: dict[str, Any] | None, score: float | None = None) -> str:
    if item:
        value = item.get("value") or {}
        label = value.get("label")
        if label:
            return str(label)
    return label_from_score(score)


def feature_refs(item: dict[str, Any] | None) -> list[str]:
    if not item:
        return []
    return list(dict.fromkeys(str(ref) for ref in item.get("evidence_refs", [])))


def feature_raw(item: dict[str, Any] | None) -> Any:
    if not item:
        return None
    value = item.get("value") or {}
    return value.get("raw")


def label_from_score(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score < 15:
        return "very_low"
    if score < 35:
        return "low"
    if score < 45:
        return "mixed"
    if score < 65:
        return "medium"
    if score < 85:
        return "high"
    return "very_high"


def pretty_label(label: str) -> str:
    return str(label).replace("_", " ")


def pct(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except Exception:
        return "NA"
    return f"{numeric * 100:.0f}%"


def money(value: Any) -> str:
    if value is None:
        return "NA"
    numeric = float(value)
    if numeric.is_integer():
        return f"${int(numeric)}"
    return f"${numeric:.2f}"


def mean_score(items: Iterable[dict[str, Any] | None], *, invert: set[str] | None = None) -> tuple[float | None, list[str]]:
    values: list[float] = []
    refs: list[str] = []
    invert = invert or set()
    for item in items:
        if not item:
            continue
        name = str(item.get("name") or "")
        score = feature_score(item)
        if score is None:
            continue
        values.append(100.0 - score if name in invert else score)
        refs.extend(feature_refs(item))
    if not values:
        return None, []
    return sum(values) / len(values), list(dict.fromkeys(refs))


def signal(
    cue: str,
    *,
    score: float | None,
    label: str | None = None,
    confidence: str = "medium",
    evidence_refs: Iterable[str] = (),
) -> dict[str, Any] | None:
    if score is None:
        return None
    return {
        "cue": cue,
        "label": label or label_from_score(score),
        "score_0_to_100": int(round(max(0.0, min(100.0, score)))),
        "meaning_here": "",
        "constructed_from": [],
        "scope_note": "",
        "evidence_refs": list(dict.fromkeys(str(ref) for ref in evidence_refs)),
    }


def signal_from_feature(
    fmap: dict[str, dict[str, Any]],
    feature_name: str,
    cue: str | None = None,
    *,
    confidence: str = "medium",
) -> dict[str, Any] | None:
    item = fmap.get(feature_name)
    score = feature_score(item)
    return signal(
        cue or feature_name,
        score=score,
        label=feature_label(item, score),
        confidence=confidence,
        evidence_refs=feature_refs(item),
    )


def background(profile: dict[str, Any]) -> dict[str, Any]:
    features = {
        str(item.get("name")): (item.get("value") or {}).get("raw")
        for item in profile.get("background_context", {}).get("harmonized_features", [])
        if item.get("name")
    }
    selected = [
        ("Age", features.get("age_bracket")),
        ("Sex assigned at birth", features.get("sex_assigned_at_birth")),
        ("Completed education", features.get("education_completed_raw")),
    ]
    selected_context = [
        {"name": name, "value": str(value)}
        for name, value in selected
        if value is not None and str(value).strip()
    ]
    summary = "; ".join(f"{item['name']}: {item['value']}" for item in selected_context)
    return {"summary": summary + "." if summary else "", "selected_context": selected_context}


def anchor_direct_social(profile: dict[str, Any]) -> dict[str, Any] | None:
    fmap = feature_map(profile["observed_in_twin"]["social_game_behavior"])
    send = feature_raw(fmap.get("trust_send_amount"))
    ret = feature_raw(fmap.get("trust_return_share_mean"))
    ult_offer = feature_raw(fmap.get("ultimatum_offer_to_other"))
    ult_min = feature_raw(fmap.get("ultimatum_min_acceptable_to_self"))
    reject = feature_raw(fmap.get("ultimatum_rejection_rate"))
    dictator = feature_raw(fmap.get("dictator_offer_to_other"))
    if all(value is None for value in [send, ret, ult_offer, ult_min, reject, dictator]):
        return None
    refs: list[str] = []
    for name in [
        "trust_send_amount",
        "trust_return_share_mean",
        "ultimatum_offer_to_other",
        "ultimatum_min_acceptable_to_self",
        "ultimatum_rejection_rate",
        "dictator_offer_to_other",
    ]:
        refs.extend(feature_refs(fmap.get(name)))
    return {
        "title": "Direct social-allocation tasks",
        "detail": (
            f"Trust send {money(send)}/$5, mean trust return {pct(ret)}, "
            f"ultimatum offer {money(ult_offer)}/$5, minimum acceptable {money(ult_min)}/$5, "
            f"ultimatum rejection rate {pct(reject)}, dictator give {money(dictator)}/$5."
        ),
        "evidence_refs": list(dict.fromkeys(refs)),
    }


def anchor_self_report(profile: dict[str, Any]) -> dict[str, Any] | None:
    fmap = feature_map(profile["observed_in_twin"]["personality_and_self_report"])
    names = ["empathy", "cooperation_orientation", "prosocial_values", "uncertainty_aversion"]
    parts: list[str] = []
    refs: list[str] = []
    for name in names:
        item = fmap.get(name)
        score = feature_score(item)
        if score is None:
            continue
        parts.append(f"{name.replace('_', ' ')} {pretty_label(feature_label(item, score))}")
        refs.extend(feature_refs(item))
    if not parts:
        return None
    return {
        "title": "Social/personality self-report",
        "detail": "Self-report block emphasizes " + ", ".join(parts[:4]) + ".",
        "evidence_refs": list(dict.fromkeys(refs)),
    }


def anchor_non_social_econ(profile: dict[str, Any]) -> dict[str, Any] | None:
    fmap = feature_map(profile["observed_in_twin"]["economic_preferences_non_social"])
    later = feature_raw(fmap.get("later_choice_rate"))
    gains = feature_raw(fmap.get("lottery_choice_rate_gains"))
    losses = feature_raw(fmap.get("lottery_choice_rate_losses"))
    mental = feature_raw(fmap.get("mental_accounting_endorsement_rate"))
    if all(value is None for value in [later, gains, losses, mental]):
        return None
    refs: list[str] = []
    for name in [
        "later_choice_rate",
        "lottery_choice_rate_gains",
        "lottery_choice_rate_losses",
        "mental_accounting_endorsement_rate",
    ]:
        refs.extend(feature_refs(fmap.get(name)))
    return {
        "title": "Time, risk, and accounting tasks",
        "detail": (
            f"Later-choice rate {pct(later)}, gain-lottery choice {pct(gains)}, "
            f"loss-lottery choice {pct(losses)}, mental-accounting endorsement {pct(mental)}."
        ),
        "evidence_refs": list(dict.fromkeys(refs)),
    }


def anchor_cognitive(profile: dict[str, Any]) -> dict[str, Any] | None:
    fmap = feature_map(profile["observed_in_twin"]["cognitive_performance"])
    fin = feature_raw(fmap.get("financial_literacy_accuracy"))
    num = feature_raw(fmap.get("numeracy_accuracy"))
    crt = feature_raw(fmap.get("cognitive_reflection_accuracy"))
    logic = feature_raw(fmap.get("logical_reasoning_accuracy"))
    if all(value is None for value in [fin, num, crt, logic]):
        return None
    refs: list[str] = []
    for name in [
        "financial_literacy_accuracy",
        "numeracy_accuracy",
        "cognitive_reflection_accuracy",
        "logical_reasoning_accuracy",
    ]:
        refs.extend(feature_refs(fmap.get(name)))
    return {
        "title": "Cognitive performance",
        "detail": (
            f"Financial literacy {pct(fin)}, numeracy {pct(num)}, "
            f"cognitive reflection {pct(crt)}, logical reasoning {pct(logic)}."
        ),
        "evidence_refs": list(dict.fromkeys(refs)),
    }


def anchor_heuristics(profile: dict[str, Any]) -> dict[str, Any] | None:
    fmap = feature_map(profile["observed_in_twin"]["heuristics_and_biases"])
    search = feature_raw(fmap.get("search_willingness"))
    anchor = feature_score(fmap.get("anchor_pull_average"))
    framing = feature_score(fmap.get("framing_reference_sensitivity"))
    ratio_bias = feature_raw(fmap.get("ratio_bias_small_tray_choice"))
    if all(value is None for value in [search, anchor, framing, ratio_bias]):
        return None
    refs: list[str] = []
    for name in [
        "search_willingness",
        "anchor_pull_average",
        "framing_reference_sensitivity",
        "ratio_bias_small_tray_choice",
    ]:
        refs.extend(feature_refs(fmap.get(name)))
    return {
        "title": "Heuristics and biases",
        "detail": (
            f"Search willingness {pct(search)}, anchor pull {int(round(anchor)) if anchor is not None else 'NA'}/100, "
            f"framing/reference sensitivity {int(round(framing)) if framing is not None else 'NA'}/100, "
            f"ratio-bias small-tray choice {int(ratio_bias) if ratio_bias is not None else 'NA'}."
        ),
        "evidence_refs": list(dict.fromkeys(refs)),
    }


def anchor_pricing(profile: dict[str, Any]) -> dict[str, Any] | None:
    fmap = feature_map(profile["observed_in_twin"]["pricing_and_consumer_choice"])
    purchase_yes = feature_raw(fmap.get("purchase_yes_rate"))
    if purchase_yes is None:
        return None
    return {
        "title": "Pricing and consumer choices",
        "detail": f"Across the product-pricing block, purchase-yes rate is {pct(purchase_yes)}.",
        "evidence_refs": feature_refs(fmap.get("purchase_yes_rate")),
    }


def anchor_open_text(profile: dict[str, Any]) -> dict[str, Any] | None:
    fmap = feature_map(profile["observed_in_twin"]["open_text_responses"])
    actual = bool(feature_raw(fmap.get("actual_text_present")))
    aspire = bool(feature_raw(fmap.get("aspire_text_present")))
    ought = bool(feature_raw(fmap.get("ought_text_present")))
    word_count = feature_raw(fmap.get("forward_flow_word_count"))
    if not any([actual, aspire, ought]) and word_count is None:
        return None
    parts: list[str] = []
    if actual:
        parts.append("actual-self text present")
    if aspire:
        parts.append("aspired-self text present")
    if ought:
        parts.append("ought-self text present")
    if word_count is not None:
        parts.append(f"forward-flow length {int(word_count)} words")
    refs: list[str] = []
    for name in [
        "actual_text_present",
        "aspire_text_present",
        "ought_text_present",
        "forward_flow_word_count",
    ]:
        refs.extend(feature_refs(fmap.get(name)))
    return {
        "title": "Open-text availability",
        "detail": "; ".join(parts) + ".",
        "evidence_refs": list(dict.fromkeys(refs)),
    }


def direct_social_signals(profile: dict[str, Any]) -> list[dict[str, Any]]:
    fmap = feature_map(profile["observed_in_twin"]["social_game_behavior"])
    names = [
        "trust_send_amount",
        "trust_return_share_mean",
        "ultimatum_offer_to_other",
        "ultimatum_min_acceptable_to_self",
        "ultimatum_rejection_rate",
        "dictator_offer_to_other",
    ]
    return [item for item in (signal_from_feature(fmap, name) for name in names) if item]


def self_report_signals(profile: dict[str, Any]) -> list[dict[str, Any]]:
    fmap = feature_map(profile["observed_in_twin"]["personality_and_self_report"])
    items: list[dict[str, Any] | None] = []

    coop_score, coop_refs = mean_score(
        [
            fmap.get("cooperation_orientation"),
            fmap.get("big_five_agreeableness"),
            fmap.get("prosocial_values"),
        ]
    )
    items.append(
        signal(
            "self_report_cooperation_orientation",
            score=coop_score,
            confidence="medium",
            evidence_refs=coop_refs,
        )
    )
    items.append(signal_from_feature(fmap, "empathy", "self_report_empathy"))
    items.append(signal_from_feature(fmap, "uncertainty_aversion", "self_report_uncertainty_aversion"))

    comm_score, comm_refs = mean_score(
        [fmap.get("big_five_extraversion"), fmap.get("empathy"), fmap.get("social_sensitivity")]
    )
    items.append(
        signal(
            "self_report_social_adaptability",
            score=comm_score,
            confidence="low",
            evidence_refs=comm_refs,
        )
    )
    stability_score, stability_refs = mean_score(
        [
            fmap.get("big_five_conscientiousness"),
            fmap.get("self_concept_clarity"),
            fmap.get("big_five_neuroticism"),
            fmap.get("orderliness"),
        ],
        invert={"big_five_neuroticism"},
    )
    items.append(
        signal(
            "self_report_behavioral_stability",
            score=stability_score,
            confidence="medium",
            evidence_refs=stability_refs,
        )
    )
    return [item for item in items if item]


def non_social_econ_signals(profile: dict[str, Any]) -> list[dict[str, Any]]:
    fmap = feature_map(profile["observed_in_twin"]["economic_preferences_non_social"])
    mapping = [
        ("later_choice_rate", "patience_later_choice_rate"),
        ("lottery_choice_rate_gains", "gain_lottery_choice_rate"),
        ("lottery_choice_rate_losses", "loss_lottery_choice_rate"),
        ("mental_accounting_endorsement_rate", "mental_accounting_endorsement_rate"),
    ]
    return [
        item
        for item in (
            signal_from_feature(fmap, feature_name, cue_name, confidence="low")
            for feature_name, cue_name in mapping
        )
        if item
    ]


def cognitive_signals(profile: dict[str, Any]) -> list[dict[str, Any]]:
    fmap = feature_map(profile["observed_in_twin"]["cognitive_performance"])
    mapping = [
        ("financial_literacy_accuracy", "financial_literacy"),
        ("numeracy_accuracy", "numeracy"),
        ("cognitive_reflection_accuracy", "cognitive_reflection"),
        ("logical_reasoning_accuracy", "logical_reasoning"),
    ]
    return [
        item
        for item in (
            signal_from_feature(fmap, feature_name, cue_name, confidence="medium")
            for feature_name, cue_name in mapping
        )
        if item
    ]


def misc_signals(profile: dict[str, Any]) -> list[dict[str, Any]]:
    heuristics = feature_map(profile["observed_in_twin"]["heuristics_and_biases"])
    pricing = feature_map(profile["observed_in_twin"]["pricing_and_consumer_choice"])
    open_text = feature_map(profile["observed_in_twin"]["open_text_responses"])
    items: list[dict[str, Any] | None] = [
        signal_from_feature(heuristics, "search_willingness", "heuristics_search_willingness", confidence="low"),
        signal_from_feature(heuristics, "anchor_pull_average", "heuristics_anchor_pull", confidence="low"),
        signal_from_feature(
            heuristics,
            "framing_reference_sensitivity",
            "heuristics_framing_reference_sensitivity",
            confidence="low",
        ),
        signal_from_feature(pricing, "purchase_yes_rate", "pricing_purchase_yes_rate", confidence="low"),
    ]
    text_presence_items = [
        open_text.get("actual_text_present"),
        open_text.get("aspire_text_present"),
        open_text.get("ought_text_present"),
    ]
    text_presence = [bool(feature_raw(item)) for item in text_presence_items if item]
    open_refs: list[str] = []
    for item in text_presence_items:
        open_refs.extend(feature_refs(item))
    if text_presence:
        items.append(
            signal(
                "open_text_availability",
                score=100.0 * sum(text_presence) / len(text_presence),
                confidence="low",
                evidence_refs=open_refs,
            )
        )
    return [item for item in items if item]


def present(items: Iterable[dict[str, Any] | None]) -> list[dict[str, Any]]:
    return [item for item in items if item is not None]


def empty_background() -> dict[str, Any]:
    return {"summary": "", "selected_context": []}


def variant_payload(profile: dict[str, Any], variant: str) -> tuple[str, str, list[str], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if variant == "twin_pgg_background_only":
        return (
            "Sampled Twin background only",
            "Only sampled Twin demographic background is shown.",
            [],
            [],
            [],
            [],
        )
    if variant == "twin_pgg_direct_social_only":
        return (
            "Direct social-allocation evidence only",
            "Only one-shot trust, ultimatum, and dictator task evidence is shown.",
            [],
            present([anchor_direct_social(profile)]),
            direct_social_signals(profile),
            [],
        )
    if variant == "twin_pgg_self_report_social_only":
        return (
            "Social/personality self-report only",
            "Only social and personality questionnaire evidence is shown.",
            [],
            present([anchor_self_report(profile)]),
            self_report_signals(profile),
            [],
        )
    if variant == "twin_pgg_non_social_econ_only":
        return (
            "Non-social economic preferences only",
            "Only time, risk, and mental-accounting task evidence is shown.",
            [],
            present([anchor_non_social_econ(profile)]),
            non_social_econ_signals(profile),
            [],
        )
    if variant == "twin_pgg_cognitive_only":
        return (
            "Cognitive performance only",
            "Only cognitive task performance evidence is shown.",
            [],
            present([anchor_cognitive(profile)]),
            cognitive_signals(profile),
            [],
        )
    if variant == "twin_pgg_misc_heuristics_pricing_text_only":
        return (
            "Miscellaneous heuristics, pricing, and open-text evidence only",
            "Only residual heuristics/biases, pricing/consumer-choice, and open-text availability evidence is shown.",
            [],
            present([anchor_heuristics(profile), anchor_pricing(profile), anchor_open_text(profile)]),
            misc_signals(profile),
            [],
        )
    raise ValueError(f"Unsupported variant: {variant}")


def build_card(profile: dict[str, Any], variant: str) -> dict[str, Any]:
    participant = profile.get("participant", {})
    headline, summary, signature, anchors, signals, limits = variant_payload(profile, variant)
    return {
        "profile_card_version": "pgg_twin_six_category_ablation_v1",
        "detail_mode": variant,
        "participant": {
            "pid": str(participant.get("pid") or ""),
            "source_dataset": participant.get("source_dataset", "Twin-2k"),
            "source_profile_version": profile.get("profile_spec_version"),
        },
        "headline": "",
        "summary": "",
        "background": background(profile) if variant == "twin_pgg_background_only" else empty_background(),
        "behavioral_signature": signature,
        "social_style": {"summary": "", "constructed_from": [], "cautions": []},
        "decision_style": {"summary": "", "constructed_from": [], "cautions": []},
        "observed_anchors": anchors,
        "transfer_relevance": signals,
        "limits": limits,
    }


VISIBLE_EVIDENCE_GUIDES: dict[str, list[str]] = {
    "twin_pgg_background_only": [
        "#### Demographic Background",
        "- Age bracket, sex assigned at birth, and completed education describe broad background context.",
        "- These fields do not summarize behavior in any Twin task or questionnaire.",
    ],
    "twin_pgg_direct_social_only": [
        "#### Direct Social-Allocation Tasks",
        "- Trust-game send amount: how much the participant sent to another person in a one-shot trust task.",
        "- Trust-game return share: how much the participant returned when placed in the responder role.",
        "- Ultimatum offer: how much the participant offered another person in an ultimatum-style split.",
        "- Ultimatum minimum acceptable amount and rejection rate: how willing the participant was to reject low offers.",
        "- Dictator offer: how much the participant gave when the other person could not reciprocate or reject.",
    ],
    "twin_pgg_self_report_social_only": [
        "#### Social/Personality Self-Report",
        "- Cooperation, prosocial values, agreeableness, empathy, and social-sensitivity items describe stated social orientation.",
        "- Uncertainty aversion and behavioral-stability items describe stated preference for predictability and consistency.",
        "- These are questionnaire summaries rather than observed choices in an interactive group game.",
    ],
    "twin_pgg_non_social_econ_only": [
        "#### Non-Social Economic Preferences",
        "- Later-choice rate summarizes willingness to wait for larger delayed rewards.",
        "- Gain- and loss-lottery choice rates summarize risk-taking in non-social choice tasks.",
        "- Mental-accounting endorsement summarizes how strongly the participant treated money as category-specific.",
    ],
    "twin_pgg_cognitive_only": [
        "#### Cognitive Performance",
        "- Financial literacy, numeracy, cognitive reflection, and logical reasoning summarize task accuracy.",
        "- These cues describe problem-solving performance, not directly observed social preferences.",
    ],
    "twin_pgg_misc_heuristics_pricing_text_only": [
        "#### Heuristics, Pricing, and Open-Text Evidence",
        "- Heuristics and biases summarize responses to anchoring, framing, ratio-bias, and search-style items.",
        "- Pricing and consumer-choice summaries describe willingness to buy listed products at listed prices.",
        "- Open-text availability records whether short self-description text was present and its coarse length.",
    ],
}


def shared_notes_for_variant(variant: str) -> str:
    guide_lines = VISIBLE_EVIDENCE_GUIDES[variant]
    if variant == "twin_pgg_background_only":
        return "\n".join(
            [
                "# Shared Prompt Notes",
                "",
                "## General Interpretation Note",
                "",
                "- These profiles summarize visible information about sampled participants.",
                "- Treat visible information as background context, not a deterministic prediction for any single public-goods-game decision.",
                "",
                "## Visible Evidence Guide",
                "",
                *guide_lines,
                "",
            ]
        )
    lines = [
        "# Shared Prompt Notes",
        "",
        "## General Interpretation Note",
        "",
        "- These profiles summarize visible survey or behavioral-task evidence about sampled participants.",
        "- Treat visible cues as relative tendencies, not deterministic predictions for any single public-goods-game decision.",
        "- Scores use a 0-100 scale where higher values indicate more of the named cue; labels are coarse bins.",
        "",
        "## Visible Evidence Guide",
        "",
        *guide_lines,
        "",
        "### Card Fields",
        "- Observed Anchors are concrete task or questionnaire summaries.",
        "- Transfer-Relevant Cues are compact labels and scores derived from the visible evidence.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    variants = tuple(args.variant or VARIANTS)
    profiles = read_profiles(args.profiles_jsonl, pid=args.pid, limit=args.limit)
    if not profiles:
        raise ValueError("No profiles selected.")

    manifest: dict[str, Any] = {
        "spec_version": "pgg_twin_six_category_ablation_v1",
        "source_profiles_jsonl": str(args.profiles_jsonl),
        "profile_count": len(profiles),
        "variants": [],
    }
    for variant in variants:
        out_dir = args.output_dir / variant
        cards = [build_card(profile, variant) for profile in profiles]
        cards_path = out_dir / "twin_pgg_ablation_cards.jsonl"
        notes_path = out_dir / "shared_prompt_notes.md"
        preview_path = out_dir / "preview_twin_pgg_ablation_cards.json"
        write_jsonl(cards_path, cards)
        notes_path.write_text(shared_notes_for_variant(variant), encoding="utf-8")
        preview_path.write_text(json.dumps(cards[:5], ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["variants"].append(
            {
                "name": variant,
                "cards_file": str(cards_path),
                "shared_notes_file": str(notes_path),
                "preview_file": str(preview_path),
            }
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
