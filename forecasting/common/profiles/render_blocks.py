from __future__ import annotations

from typing import Any


TWIN_TRANSFER_CUE_DISPLAY_NAMES = {
    "cooperation_orientation": "Cooperation orientation",
    "conditional_cooperation": "Conditional cooperation",
    "norm_enforcement": "Norm enforcement",
    "generosity_without_return": "Generosity without return",
    "exploitation_caution": "Exploitation caution",
    "communication_coordination": "Communication/coordination",
    "behavioral_stability": "Behavioral stability",
}


def render_demographic_profile_block(card: dict[str, Any]) -> str:
    lines = [
        "# PARTICIPANT PROFILE",
        "Use this sampled profile as lightweight background context. It was sampled from the study population's observed demographic distribution and is not the real participant.",
        "",
        f"Summary: {card['summary']}",
    ]
    markdown = str(card.get("markdown", "")).strip()
    if markdown:
        lines.extend(["Details:", markdown])
    return "\n".join(lines)


def render_twin_profile_block(
    assignment: dict[str, Any],
    card: dict[str, Any],
    shared_notes: str,
    corrected: bool,
) -> str:
    lines = [
        "# PARTICIPANT PERSONA",
        (
            "Use this sampled persona as a prior about likely tendencies. "
            "It was sampled from Twin to match the target study's observed demographic distribution and is not the real participant."
            if corrected
            else "Use this sampled persona as a prior about likely tendencies. It was sampled from the Twin pool without demographic correction and is not the real participant."
        ),
        "",
        shared_notes.strip(),
        "",
        f"Headline: {card.get('headline', assignment.get('headline', ''))}",
        f"Summary: {card.get('summary', assignment.get('summary', ''))}",
    ]
    background_summary = str((card.get("background") or {}).get("summary", "")).strip()
    if background_summary:
        lines.append(f"Background: {background_summary}")

    behavioral_signature = card.get("behavioral_signature", [])
    if behavioral_signature:
        lines.append("Behavioral Signature:")
        for item in behavioral_signature:
            lines.append(f"- {item}")

    observed_anchors = card.get("observed_anchors", [])
    if observed_anchors:
        lines.append("Observed Anchors:")
        for item in observed_anchors:
            title = str(item.get("title", "")).strip()
            detail = str(item.get("detail", "")).strip()
            if title and detail:
                lines.append(f"- {title}: {detail}")

    transfer_relevance = card.get("transfer_relevance", [])
    if transfer_relevance:
        lines.append("Transfer-Relevant Cues:")
        for item in transfer_relevance:
            cue = str(item.get("cue", "")).strip()
            cue_name = TWIN_TRANSFER_CUE_DISPLAY_NAMES.get(
                cue, cue.replace("_", " ").strip().title()
            )
            label = str(item.get("label", "")).replace("_", " ").strip()
            score = item.get("score_0_to_100", "")
            confidence = str(item.get("confidence", "")).strip()
            lines.append(f"- {cue_name}: {label} ({score}), confidence {confidence}")

    limits = card.get("limits", [])
    if limits:
        lines.append("Limits:")
        for item in limits:
            topic = str(item.get("topic", "")).strip()
            note = str(item.get("note", "")).strip()
            if topic and note:
                lines.append(f"- {topic}: {note}")
            elif note:
                lines.append(f"- {note}")

    return "\n".join(lines)


def render_pgg_persona_block(
    *,
    variant_slug: str,
    shared_prompt_notes: str | None,
    raw_player_order: list[str],
    avatar_order: list[str],
    persona_assignments: dict[str, Any],
    twin_profile_cards: dict[str, dict[str, Any]],
    is_demographic_only_variant: bool,
) -> str:
    is_demographic_only = is_demographic_only_variant
    lines = (
        [
            "# PLAYER PROFILES",
            "Use these provided profiles as player-specific priors when reasoning about motivations and likely choices.",
        ]
        if is_demographic_only
        else [
            "# PLAYER PERSONAS",
            "Use these provided personas as player-specific priors when reasoning about motivations and likely choices.",
        ]
    )
    if shared_prompt_notes:
        note_lines = shared_prompt_notes.splitlines()
        if note_lines and note_lines[0].strip() == "# Shared Prompt Notes":
            note_lines = ["## Shared Prompt Notes", *note_lines[1:]]
        demoted_note_lines: list[str] = []
        for idx, line in enumerate(note_lines):
            if idx > 0 and line.startswith("## "):
                demoted_note_lines.append(f"#{line}")
            else:
                demoted_note_lines.append(line)
        lines.extend(["", *demoted_note_lines, ""])
    for seat_index, (player_id, avatar) in enumerate(zip(raw_player_order, avatar_order), start=1):
        assignment = persona_assignments.get(player_id) or persona_assignments.get(f"seat:{seat_index}")
        if assignment is None:
            raise KeyError(f"Missing persona assignment for game seat {seat_index} ({avatar})")
        card = twin_profile_cards[assignment.profile_id]
        lines.append(f"## {avatar}")

        if is_demographic_only:
            summary = str(card.get("summary", assignment.summary)).strip()
            if summary:
                lines.append(f"Summary: {summary}")
            lines.append("")
            continue

        lines.append(f"Headline: {card.get('headline', assignment.headline)}")
        lines.append(f"Summary: {card.get('summary', assignment.summary)}")

        background = card.get("background", {})
        background_summary = str(background.get("summary", "")).strip()
        if background_summary:
            lines.append(f"Background: {background_summary}")

        behavioral_signature = card.get("behavioral_signature", [])
        if behavioral_signature:
            lines.append("Behavioral Signature:")
            for item in behavioral_signature:
                lines.append(f"- {item}")

        observed_anchors = card.get("observed_anchors", [])
        if observed_anchors:
            lines.append("Observed Anchors:")
            for item in observed_anchors:
                title = str(item.get("title", "")).strip()
                detail = str(item.get("detail", "")).strip()
                if title and detail:
                    lines.append(f"- {title}: {detail}")

        transfer_relevance = card.get("transfer_relevance", [])
        if transfer_relevance:
            lines.append("Transfer-Relevant Cues:")
            for item in transfer_relevance:
                cue = str(item.get("cue", "")).strip()
                cue_name = TWIN_TRANSFER_CUE_DISPLAY_NAMES.get(
                    cue, cue.replace("_", " ").capitalize()
                )
                label = str(item.get("label", "")).replace("_", " ").strip()
                score = item.get("score_0_to_100", "")
                confidence = str(item.get("confidence", "")).strip()
                lines.append(f"- {cue_name}: {label} ({score}), confidence {confidence}")

        limits = card.get("limits", [])
        if limits:
            lines.append("Limits:")
            for item in limits:
                topic = str(item.get("topic", "")).strip()
                note = str(item.get("note", "")).strip()
                if topic and note:
                    lines.append(f"- {topic}: {note}")
                elif note:
                    lines.append(f"- {note}")

            lines.append("")
    return "\n".join(lines)
