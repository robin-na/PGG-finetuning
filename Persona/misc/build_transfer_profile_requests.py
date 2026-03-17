#!/usr/bin/env python3
"""Build OpenAI Batch requests for transfer-oriented PGG profile extraction."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None

try:
    from Persona.misc.transfer_profile_data import build_raw_profiles, write_profiles_jsonl
except ImportError:
    from transfer_profile_data import build_raw_profiles, write_profiles_jsonl  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSONA_DIR = PROJECT_ROOT / "Persona"
DEFAULT_OUTPUT_DIR = PERSONA_DIR / "transfer_profiles" / "output" / "all_waves"

DEFAULT_MODEL = "gpt-5.1"
SUPPORTED_SPLITS = ("learn", "val", "all")
TRANSCRIPT_MODES = ("full", "snippets", "none")

SYSTEM_PROMPT = """You extract transfer-oriented behavioral profiles from public-goods-game evidence.

Your job is to separate:
1. behavior directly observed in this exact PGG environment, and
2. broader latent economic or social-preference traits that may transfer across other games or bargaining settings.

Rules:
- Ground every latent inference in the provided evidence.
- Generalize cautiously. If the evidence is weak or the mechanism was unavailable, say so.
- Use general economic / behavioral language, not only PGG-specific jargon.
- Use `none` only when a behavior or mechanism was available and observed at zero.
- Use `unknown` when a mechanism was unavailable or the relevant behavior was not observed well enough to infer.
- Use `unknown` for `response_to_sanctions` when the player was never punished.
- Do not use rewards as evidence for `response_to_sanctions`.
- Use `none` for `sanctioning_behavior` or `reward_behavior` when the corresponding mechanism was available and total observed use was exactly zero.
- Reserve `very_low` for small but nonzero observed use or responsiveness.
- If the player shows no observable adjustment to others' behavior, prefer `none` or `very_low` over `low` for `response_to_others`.
- Reserve `medium` or higher for `strategic_cooperation` only when there is clear evidence of contingent, incentive-aware adjustment or targeted mechanism use; simple fixed heuristics or mild payoff awareness should stay below that.
- Avoid redundancy across fields. Do not restate the same claim in multiple sections unless the section requires a short cross-reference.
- Keep section roles distinct:
  - `observed_in_pgg` = only what was directly observed in this exact PGG environment.
  - `latent_traits` = broader transferable tendencies inferred from the evidence.
  - `evidence` = only the highest-signal supporting facts, not paraphrases of the same point.
  - `persona_card.behavioral_signature` = 2-3 distinctive transferable patterns.
  - `persona_card.transfer_relevance` = 2-3 retrieval use cases, without repeating the signature verbatim.
  - `persona_card.limits` = 2-3 scope limits on transfer, distinct from `uncertainties`.
- Keep the persona card concise and retrieval-friendly.
- Do not output participant IDs, game IDs, config IDs, or a standalone game-context object.
- Return JSON only, matching the schema exactly.
"""

USER_INSTRUCTIONS = """Create a transfer-oriented profile for downstream retrieval and prompt augmentation.

The output should be useful for retrieving analogous cases from a PGG library when reasoning about behavior in other economic or social-decision games.

Important:
- The persona card should mention that the evidence comes from a repeated public-goods game, but it should describe general latent tendencies rather than only PGG mechanics.
- Much of the direct PGG behavior has already been summarized deterministically from raw game logs. If transcript evidence is provided, use it to refine, qualify, or contextualize those summaries rather than to overwrite them casually.
- If an existing summary is provided, treat it as a lossy qualitative prior. Use it to capture broad patterns, but anchor the final profile in the deterministic summary and transcript when they are more specific.
- If punishment or reward was unavailable in the game, do not infer behavior for that mechanism from thin evidence; mark it as unknown.
- If communication was unavailable, mark communication style as unknown.
- Keep evidence statements concrete and tied to the provided stats, events, existing summary, or transcript evidence when present.
- Prefer transferable features such as generosity, reciprocity, norm enforcement, exploitation caution, conditionality, and behavioral stability over narrow game-by-game predictions.
- Do not generate a separate per-target-game forecast block. Keep transfer claims general and retrieval-oriented.
- Keep the output compact. Use the shortest wording that preserves the claim.
- `summary`: max 2 sentences.
- `behavioral_signature`: 2-3 bullets covering the most distinctive transferable patterns only.
- `transfer_relevance`: 2-3 bullets explaining when this case would be useful to retrieve; do not paraphrase the signature.
- `limits`: 2-3 bullets about transfer scope limits of this evidence or environment.
- `uncertainties`: 2-3 bullets about unresolved ambiguity in the participant's traits or motives; do not repeat `limits`.
- If two statements say nearly the same thing, keep the more general transferable version and drop the duplicate.
- In the persona card, avoid repeating PGG mechanics unless they are necessary to explain why the broader trait was inferred.
- Do not merely paraphrase the existing summary back to us. Reconcile it with the deterministic profile and transcript, then keep only the most transferable claims.
"""

OBSERVED_DIMENSIONS = [
    "contribution_level",
    "contribution_stability",
    "response_to_others",
    "sanctioning_behavior",
    "response_to_sanctions",
    "reward_behavior",
    "communication_style",
]

LATENT_DIMENSIONS = [
    "generalized_prosociality",
    "reciprocity",
    "trust_in_others",
    "strategic_cooperation",
    "fairness_sensitivity",
    "inequality_aversion",
    "caution_about_exploitation",
]

LABEL_ENUM = [
    "very_low",
    "low",
    "mixed",
    "medium",
    "high",
    "very_high",
    "none",
    "unknown",
]

CONFIDENCE_ENUM = ["low", "medium", "high"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SUPPORTED_SPLITS, default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-round-snippets", type=int, default=5)
    parser.add_argument("--transcript-mode", choices=TRANSCRIPT_MODES, default="full")
    parser.add_argument("--include-oracle-summary", action="store_true")
    parser.add_argument("--include-event-evidence", action="store_true")
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def shorten(text: str, limit: int) -> str:
    text = normalize_whitespace(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def splits_from_arg(split: str) -> List[str]:
    if split == "all":
        return ["learn", "val"]
    return [split]


def load_participant_transcripts(split: str) -> Dict[Tuple[str, str, str], Dict]:
    path = PERSONA_DIR / f"transcripts_{'learn' if split == 'learn' else 'val'}.jsonl"
    with path.open("r", encoding="utf-8") as f:
        return {
            (split, str(row["experiment"]), str(row["participant"])): row
            for row in (json.loads(line) for line in f)
        }


def load_oracle_summaries(split: str) -> Dict[Tuple[str, str, str], str]:
    path = PERSONA_DIR / f"archetype_oracle_gpt51_{'learn' if split == 'learn' else 'val'}.jsonl"
    with path.open("r", encoding="utf-8") as f:
        return {
            (split, str(row["experiment"]), str(row["participant"])): str(row["text"])
            for row in (json.loads(line) for line in f)
            if row.get("text")
        }


def build_rule_summary(config: Dict) -> List[str]:
    endowment = config.get("CONFIG_endowment") or 20
    player_count = config.get("CONFIG_playerCount")
    rounds = config.get("CONFIG_numRounds")
    multiplier = config.get("CONFIG_multiplier")
    all_or_nothing = config.get("CONFIG_allOrNothing") is True
    action_space = f"either 0 or {endowment}" if all_or_nothing else f"integer from 0 to {endowment}"

    lines = [
        f"Repeated online public-goods game with configured group size {player_count} and {rounds} rounds.",
        f"Each active player had {endowment} coins per round and chose a contribution amount {action_space}.",
        "Players did not see others' choices before deciding each round.",
        f"The public pot was multiplied by {multiplier}x and split equally across active players.",
    ]
    if config.get("CONFIG_showNRounds") is True:
        lines.append("Players knew the total number of rounds in advance.")
    else:
        lines.append("Players did not know the total number of rounds in advance.")

    if config.get("CONFIG_rewardExists") is True:
        lines.append(
            "After redistribution, players could reward others: "
            f"cost {config.get('CONFIG_rewardCost')} to grant {config.get('CONFIG_rewardMagnitude')}."
        )
    if config.get("CONFIG_punishmentExists") is True:
        lines.append(
            "After redistribution, players could punish others: "
            f"cost {config.get('CONFIG_punishmentCost')} to deduct {config.get('CONFIG_punishmentMagnitude')}."
        )
    if config.get("CONFIG_rewardExists") is not True and config.get("CONFIG_punishmentExists") is not True:
        lines.append("There was no reward or punishment stage after redistribution.")

    if config.get("CONFIG_chat") is True:
        lines.append("Players could optionally send one short group message each round.")
    else:
        lines.append("There was no round-by-round chat channel.")
    return lines


def extract_tag_body(block: str, tag: str) -> Optional[str]:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", block, flags=re.S)
    if not match:
        return None
    return normalize_whitespace(match.group(1))


def extract_self_closing_json(block: str, tag: str) -> Optional[str]:
    match = re.search(rf"<{tag} json='([^']+)'/>", block)
    if not match:
        return None
    return normalize_whitespace(match.group(1))


def extract_round_blocks(text: str) -> List[str]:
    if "# GAME STARTS" in text:
        text = text.split("# GAME STARTS", 1)[1]
    return re.findall(r"(<ROUND i=\".*?</ROUND>)", text, flags=re.S)


def parse_contribution(block: str) -> Optional[int]:
    match = re.search(r"You contributed:\s*([0-9]+)", block)
    return int(match.group(1)) if match else None


def parse_round_label(block: str) -> str:
    match = re.search(r"<ROUND i=\"([^\"]+)\">", block)
    return match.group(1) if match else "unknown"


def round_signal(block: str) -> Dict[str, object]:
    punishment_text = extract_tag_body(block, "PUNISHMENT")
    reward_text = extract_tag_body(block, "REWARD")
    chat_text = extract_tag_body(block, "CHAT")
    return {
        "round_label": parse_round_label(block),
        "contribution": parse_contribution(block),
        "chat_text": chat_text,
        "punishment_text": punishment_text,
        "reward_text": reward_text,
        "punished_by": extract_self_closing_json(block, "PUNISHED_BY"),
        "rewarded_by": extract_self_closing_json(block, "REWARDED_BY"),
        "peers": extract_tag_body(block, "PEERS_CONTRIBUTIONS"),
        "round_info": extract_tag_body(block, "ROUND_INFO"),
        "has_active_punishment": bool(punishment_text and "did not punish anybody" not in punishment_text.lower()),
        "has_active_reward": bool(reward_text and "did not reward anybody" not in reward_text.lower()),
        "has_chat": bool(chat_text),
        "received_punishment": extract_self_closing_json(block, "PUNISHED_BY") is not None,
        "received_reward": extract_self_closing_json(block, "REWARDED_BY") is not None,
    }


def render_round_excerpt(info: Dict[str, object]) -> str:
    lines = [f"Round {info['round_label']}"]
    if info.get("round_info"):
        lines.append(f"Round info: {shorten(str(info['round_info']), 220)}")
    if info.get("chat_text"):
        lines.append(f"Chat: {shorten(str(info['chat_text']), 220)}")
    if info.get("contribution") is not None:
        lines.append(f"Contribution: {info['contribution']}")
    if info.get("peers"):
        lines.append(f"Peers: {shorten(str(info['peers']), 220)}")
    if info.get("punishment_text"):
        lines.append(f"Punishment action: {shorten(str(info['punishment_text']), 220)}")
    if info.get("reward_text"):
        lines.append(f"Reward action: {shorten(str(info['reward_text']), 220)}")
    if info.get("punished_by"):
        lines.append(f"Punished by: {shorten(str(info['punished_by']), 220)}")
    if info.get("rewarded_by"):
        lines.append(f"Rewarded by: {shorten(str(info['rewarded_by']), 220)}")
    return "\n".join(lines)


def select_round_snippets(text: str, max_round_snippets: int) -> List[str]:
    blocks = extract_round_blocks(text)
    if not blocks:
        return []
    infos = [round_signal(block) for block in blocks]
    contributions = [info.get("contribution") for info in infos]
    selected_indices: List[int] = []

    def add_index(idx: Optional[int]) -> None:
        if idx is None or idx < 0 or idx >= len(infos) or idx in selected_indices:
            return
        selected_indices.append(idx)

    add_index(0)
    add_index(next((i for i, info in enumerate(infos) if info["has_chat"]), None))
    add_index(next((i for i, info in enumerate(infos) if info["has_active_punishment"] or info["has_active_reward"]), None))
    add_index(next((i for i, info in enumerate(infos) if info["received_punishment"] or info["received_reward"]), None))

    numeric_contribs = [(i, c) for i, c in enumerate(contributions) if c is not None]
    if numeric_contribs:
        add_index(min(numeric_contribs, key=lambda item: item[1])[0])
        add_index(max(numeric_contribs, key=lambda item: item[1])[0])
        largest_delta_idx = None
        largest_delta = -1
        prev_value = None
        for idx, value in numeric_contribs:
            if prev_value is not None:
                delta = abs(value - prev_value)
                if delta > largest_delta:
                    largest_delta = delta
                    largest_delta_idx = idx
            prev_value = value
        add_index(largest_delta_idx)

    add_index(len(infos) - 1)
    return [render_round_excerpt(infos[idx]) for idx in selected_indices[:max_round_snippets]]


def event_priority(event: Dict) -> Tuple[int, float, int]:
    event_type = str(event.get("event_type"))
    priority = {
        "was_punished": 0,
        "was_rewarded": 1,
        "saw_defection": 2,
        "endgame_phase": 3,
    }.get(event_type, 9)
    response = event.get("response_next", {}) or {}
    magnitude = 0.0
    for value in response.values():
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            magnitude = max(magnitude, abs(float(value)))
    round_index = event.get("round_index") or 10**6
    return priority, -magnitude, int(round_index)


def render_event(event: Dict) -> str:
    event_type = str(event.get("event_type"))
    round_index = event.get("round_index")
    details = event.get("event_details", {}) or {}
    response = event.get("response_next", {}) or {}
    return shorten(
        f"{event_type} at round {round_index}: details={json.dumps(details, separators=(',', ':'))}; "
        f"response={json.dumps(response, separators=(',', ':'))}",
        320,
    )


def select_event_evidence(profile: Dict, limit: int = 6) -> List[str]:
    events = list(profile.get("event_responses", []))
    events.sort(key=event_priority)
    return [render_event(event) for event in events[:limit]]


def trim_profile_for_prompt(profile: Dict) -> Dict:
    return {
        "derived_context": {
            "num_rounds_observed": profile.get("derived_context", {}).get("num_rounds_observed"),
            "num_rounds_configured": profile.get("derived_context", {}).get("num_rounds_configured"),
            "action_space_expected": profile.get("derived_context", {}).get("action_space_expected"),
            "action_space_observed": profile.get("derived_context", {}).get("action_space_observed"),
            "mechanisms_enabled": profile.get("derived_context", {}).get("mechanisms_enabled"),
            "group_size": profile.get("derived_context", {}).get("group_size"),
        },
        "observed_summary": profile.get("observed_summary"),
        "module_card": profile.get("module_card"),
    }


def build_dimension_schema() -> Dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string", "enum": LABEL_ENUM},
            "score_0_to_100": {"type": "integer", "minimum": 0, "maximum": 100},
            "rationale": {"type": "string"},
        },
        "required": ["label", "score_0_to_100", "rationale"],
    }


def build_response_schema() -> Dict:
    observed_properties = {name: build_dimension_schema() for name in OBSERVED_DIMENSIONS}
    latent_properties = {name: build_dimension_schema() for name in LATENT_DIMENSIONS}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "observed_in_pgg": {
                "type": "object",
                "additionalProperties": False,
                "properties": observed_properties,
                "required": OBSERVED_DIMENSIONS,
            },
            "latent_traits": {
                "type": "object",
                "additionalProperties": False,
                "properties": latent_properties,
                "required": LATENT_DIMENSIONS,
            },
            "uncertainties": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 3,
            },
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "source_type": {"type": "string", "enum": ["stat", "event", "summary", "transcript"]},
                        "detail": {"type": "string"},
                    },
                    "required": ["source_type", "detail"],
                },
                "minItems": 2,
                "maxItems": 4,
            },
            "persona_card": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "headline": {"type": "string"},
                    "summary": {"type": "string"},
                    "behavioral_signature": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 3,
                    },
                    "transfer_relevance": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 3,
                    },
                    "limits": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 3,
                    },
                },
                "required": [
                    "headline",
                    "summary",
                    "behavioral_signature",
                    "transfer_relevance",
                    "limits",
                ],
            },
        },
        "required": [
            "observed_in_pgg",
            "latent_traits",
            "uncertainties",
            "evidence",
            "persona_card",
        ],
    }


def json_dumps_pretty(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=True, indent=2, sort_keys=False)


def build_user_prompt(
    *,
    profile: Dict,
    participant_text: str,
    oracle_summary: Optional[str],
    max_round_snippets: int,
    transcript_mode: str,
    include_event_evidence: bool,
) -> str:
    rule_summary = build_rule_summary(profile.get("config", {}))
    trimmed_profile = trim_profile_for_prompt(profile)
    selected_events = select_event_evidence(profile) if include_event_evidence else []
    round_snippets = select_round_snippets(participant_text, max_round_snippets=max_round_snippets)
    if transcript_mode == "full":
        transcript_section = participant_text
    elif transcript_mode == "snippets":
        transcript_section = "\n\n".join(round_snippets) if round_snippets else "None"
    else:
        transcript_section = "Omitted for this run."
    sections = [
        USER_INSTRUCTIONS,
        "\nPGG rule summary derived from the exact game config:\n- " + "\n- ".join(rule_summary),
        "\nDirect profile from raw game data:\n" + json_dumps_pretty(trimmed_profile),
    ]
    if selected_events:
        sections.append("\nSelected event evidence from raw game data:\n" + json_dumps_pretty(selected_events))
    if oracle_summary:
        sections.append("\nExisting summary for this participant:\n" + oracle_summary)
    if transcript_mode != "none":
        sections.append("\nParticipant transcript:\n" + transcript_section)
    sections.append(
        "\nOutput JSON schema follows. Do not include participant_id, playerId, gameId, configId, or a separate game_context field in the output.\n"
    )
    return "\n".join(sections)


def estimate_tokens_for_text(text: str, model: str) -> Tuple[int, str]:
    if tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text)), "tiktoken_model"
        except Exception:
            try:
                encoding = tiktoken.get_encoding("o200k_base")
                return len(encoding.encode(text)), "tiktoken_o200k_base"
            except Exception:
                pass
    return max(1, math.ceil(len(text) / 4)), "char_div4_fallback"


def main() -> None:
    args = parse_args()
    splits = splits_from_arg(args.split)
    output_tag = args.split
    args.output_dir.mkdir(parents=True, exist_ok=True)

    profiles = build_raw_profiles(splits)
    raw_profiles_path = args.output_dir / f"raw_profiles_{output_tag}.jsonl"
    write_profiles_jsonl(profiles, raw_profiles_path)

    participant_transcripts: Dict[Tuple[str, str, str], Dict] = {}
    oracle_summaries: Dict[Tuple[str, str, str], str] = {}
    for split in splits:
        participant_transcripts.update(load_participant_transcripts(split))
        if args.include_oracle_summary:
            oracle_summaries.update(load_oracle_summaries(split))

    eligible_complete_profiles = [
        profile
        for profile in profiles
        if profile.get("played_to_end") is True
    ]
    eligible_complete_profiles.sort(key=lambda row: (row["split"], row["gameId"], row["playerId"]))
    filtered_profiles = eligible_complete_profiles
    if args.limit is not None:
        filtered_profiles = filtered_profiles[: args.limit]

    requests_path = args.output_dir / f"requests_transfer_profiles_{args.model}.jsonl"
    manifest_path = args.output_dir / "manifest_transfer_profiles.jsonl"
    preview_path = args.output_dir / f"preview_transfer_profiles_{args.model}.json"
    token_estimate_path = args.output_dir / f"token_estimate_transfer_profiles_{args.model}.json"

    schema = build_response_schema()
    request_token_estimates: List[int] = []
    tokenizer_source = None
    preview_request: Optional[Dict] = None
    missing_transcript = 0
    missing_oracle_summary = 0

    with requests_path.open("w", encoding="utf-8") as requests_file, manifest_path.open("w", encoding="utf-8") as manifest_file:
        for profile in filtered_profiles:
            key = (profile["split"], profile["gameId"], profile["playerId"])
            transcript_row = participant_transcripts.get(key)
            if args.transcript_mode != "none" and transcript_row is None:
                missing_transcript += 1
                continue
            oracle_summary = None
            if args.include_oracle_summary:
                oracle_summary = oracle_summaries.get(key)
                if oracle_summary is None:
                    missing_oracle_summary += 1

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        profile=profile,
                        participant_text=str(transcript_row["text"]) if transcript_row is not None else "",
                        oracle_summary=oracle_summary,
                        max_round_snippets=args.max_round_snippets,
                        transcript_mode=args.transcript_mode,
                        include_event_evidence=args.include_event_evidence,
                    ),
                },
            ]
            body = {
                "model": args.model,
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "pgg_transfer_profile",
                        "strict": True,
                        "schema": schema,
                    },
                },
            }
            if args.temperature is not None:
                body["temperature"] = args.temperature
            if args.max_completion_tokens is not None:
                body["max_completion_tokens"] = args.max_completion_tokens

            custom_id = f"pgg-transfer-profile::{profile['split']}::{profile['gameId']}::{profile['playerId']}"
            request_obj = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            serialized_request = json.dumps(request_obj, ensure_ascii=True)
            token_estimate, current_source = estimate_tokens_for_text(serialized_request, args.model)
            if tokenizer_source is None:
                tokenizer_source = current_source
            request_token_estimates.append(token_estimate)

            requests_file.write(serialized_request)
            requests_file.write("\n")
            manifest_row = {
                "custom_id": custom_id,
                "split": profile["split"],
                "gameId": profile["gameId"],
                "playerId": profile["playerId"],
                "chat_enabled": bool(profile.get("config", {}).get("CONFIG_chat")),
                "punishment_enabled": bool(profile.get("config", {}).get("CONFIG_punishmentExists")),
                "reward_enabled": bool(profile.get("config", {}).get("CONFIG_rewardExists")),
                "played_to_end": bool(profile.get("played_to_end")),
                "transcript_mode": args.transcript_mode,
                "include_oracle_summary": bool(args.include_oracle_summary),
                "include_event_evidence": bool(args.include_event_evidence),
                "request_token_estimate": token_estimate,
            }
            manifest_file.write(json.dumps(manifest_row, ensure_ascii=True))
            manifest_file.write("\n")
            if preview_request is None:
                preview_request = request_obj

    if request_token_estimates:
        sorted_estimates = sorted(request_token_estimates)
        p95_index = max(0, math.ceil(0.95 * len(sorted_estimates)) - 1)
        token_summary = {
            "split": args.split,
            "model": args.model,
            "tokenizer_source": tokenizer_source,
            "raw_profile_count": len(profiles),
            "eligible_complete_profiles": len(eligible_complete_profiles),
            "requests_targeted_after_limit": len(filtered_profiles),
            "request_count": len(request_token_estimates),
            "transcript_mode": args.transcript_mode,
            "include_oracle_summary": bool(args.include_oracle_summary),
            "include_event_evidence": bool(args.include_event_evidence),
            "total_prompt_tokens_estimate": int(sum(request_token_estimates)),
            "mean_prompt_tokens_estimate": statistics.mean(request_token_estimates),
            "median_prompt_tokens_estimate": statistics.median(request_token_estimates),
            "p95_prompt_tokens_estimate": sorted_estimates[p95_index],
            "max_prompt_tokens_estimate": max(request_token_estimates),
            "missing_participant_transcripts": missing_transcript,
            "missing_oracle_summaries": missing_oracle_summary,
        }
    else:
        token_summary = {
            "split": args.split,
            "model": args.model,
            "tokenizer_source": tokenizer_source,
            "raw_profile_count": len(profiles),
            "eligible_complete_profiles": len(eligible_complete_profiles),
            "requests_targeted_after_limit": len(filtered_profiles),
            "request_count": 0,
            "transcript_mode": args.transcript_mode,
            "include_oracle_summary": bool(args.include_oracle_summary),
            "include_event_evidence": bool(args.include_event_evidence),
            "missing_participant_transcripts": missing_transcript,
            "missing_oracle_summaries": missing_oracle_summary,
        }

    if preview_request is not None:
        preview_path.write_text(json.dumps(preview_request, ensure_ascii=True, indent=2), encoding="utf-8")
    token_estimate_path.write_text(json.dumps(token_summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Wrote raw profiles: {raw_profiles_path}")
    print(f"Wrote requests:     {requests_path}")
    print(f"Wrote manifest:     {manifest_path}")
    if preview_request is not None:
        print(f"Wrote preview:      {preview_path}")
    print(f"Wrote tokens:       {token_estimate_path}")
    print(f"Raw profiles:       {len(profiles)}")
    print(f"Complete profiles:  {len(eligible_complete_profiles)}")
    print(f"Targeted requests:  {len(filtered_profiles)}")
    print(f"Requests kept:      {len(request_token_estimates)}")
    print(f"Skipped (missing transcript): {missing_transcript}")


if __name__ == "__main__":
    main()
