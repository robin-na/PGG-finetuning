from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trajectory_completion.data import GameTrajectory, RoundRecord, load_wave_games


CONTINUATION_SYSTEM_PROMPT = """You forecast the remaining rounds of an observed public goods game.

Start with a careful game-level explanation of what the observed history suggests about each player's incentives, values, beliefs, strategy, and uncertainty. Then continue the game in the same transcript format as the observed history. At the start of each predicted round, include a round explanation before the predicted actions. Keep the stage ordering coherent within each round. Any predicted action lines that are wrapped in the observed history should stay wrapped in <<>> exactly."""

FULL_ROLLOUT_SYSTEM_PROMPT = """You simulate an entire public goods game from round 1.

Start with a game-level explanation of the likely dynamics, incentives, and uncertainties. Then write the game directly in transcript form, starting from round 1. At the start of each round, include a round explanation before the predicted actions. Keep the stage ordering coherent within each round. Any action lines that are wrapped in the scaffold should stay wrapped in <<>> exactly."""


PHASE_ORDER = ("contribution", "outcome", "summary")
TWIN_TRANSFER_CUE_DISPLAY_NAMES = {
    "cooperation_orientation": "Cooperation orientation",
    "conditional_cooperation": "Conditional cooperation",
    "norm_enforcement": "Norm enforcement",
    "generosity_without_return": "Generosity without return",
    "exploitation_caution": "Exploitation caution",
    "communication_coordination": "Communication/coordination",
    "behavioral_stability": "Behavioral stability",
}


@dataclass(frozen=True)
class PromptMetadata:
    game_id: str
    treatment_name: str
    created_at: str
    config_id: int
    num_rounds: int
    endowment: int
    multiplier: float
    all_or_nothing: bool
    chat_enabled: bool
    default_contrib_prop: bool
    punishment_exists: bool
    punishment_cost: int
    punishment_magnitude: int
    reward_exists: bool
    reward_cost: int
    reward_magnitude: int
    show_n_rounds: bool
    show_other_summaries: bool
    show_punishment_id: bool
    show_reward_id: bool
    valid_number_of_starting_players: bool
    chat_log: str


@dataclass(frozen=True)
class PersonaAssignment:
    seat_index: int
    player_id: str | None
    profile_id: str
    headline: str
    summary: str


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _format_num(value: float | int) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def _coin_phrase(value: int) -> str:
    return f"{value} coin" if int(value) == 1 else f"{value} coins"


def _parse_k_values(raw_value: str) -> list[int]:
    values: list[int] = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(int(chunk))
    if not values:
        raise ValueError("At least one k value is required.")
    return values


def _sanitize_token(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return sanitized.strip("_").lower()


def _canonical_variant_name(variant_name: str) -> str:
    variant_slug = _sanitize_token(variant_name)
    aliases = {
        "baseline_direct_transcript": "baseline",
        "twin_sampled_seed_0_unadjusted": "twin_sampled_unadjusted_seed_0",
        "twin_to_pgg_validation_persona_sampling_unadjusted_seed_0": (
            "twin_sampled_unadjusted_seed_0"
        ),
        "pgg_validation_demographic_only_sampling_row_resampled_seed_0": (
            "demographic_only_row_resampled_seed_0"
        ),
    }
    return aliases.get(variant_slug, variant_slug)


def _is_assigned_profile_variant(variant_slug: str) -> bool:
    return variant_slug in {
        "twin_sampled_seed_0",
        "twin_sampled_unadjusted_seed_0",
        "demographic_only_row_resampled_seed_0",
    }


def _is_demographic_only_variant(variant_slug: str) -> bool:
    return variant_slug == "demographic_only_row_resampled_seed_0"


def _twin_assignment_path_for_variant(repo_root: Path, variant_name: str) -> Path | None:
    variant_slug = _canonical_variant_name(variant_name)
    if variant_slug == "twin_sampled_seed_0":
        return (
            repo_root
            / "non-PGG_generalization/task_grounding/output"
            / "twin_to_pgg_validation_persona_sampling/seed_0/game_assignments.jsonl"
        )
    if variant_slug == "twin_sampled_unadjusted_seed_0":
        return (
            repo_root
            / "non-PGG_generalization/task_grounding/output"
            / "twin_to_pgg_validation_persona_sampling_unadjusted/seed_0/game_assignments.jsonl"
        )
    if variant_slug == "demographic_only_row_resampled_seed_0":
        return (
            repo_root
            / "non-PGG_generalization/task_grounding/output"
            / "pgg_validation_demographic_only_sampling_row_resampled/seed_0/game_assignments.jsonl"
        )
    return None


def _twin_prompt_cards_path_for_variant(repo_root: Path, variant_name: str) -> Path | None:
    variant_slug = _canonical_variant_name(variant_name)
    if variant_slug in {"twin_sampled_seed_0", "twin_sampled_unadjusted_seed_0"}:
        return (
            repo_root
            / "non-PGG_generalization/task_grounding/output"
            / "twin_extended_profile_cards/pgg_prompt_min/twin_extended_profile_cards.jsonl"
        )
    if variant_slug == "demographic_only_row_resampled_seed_0":
        return (
            repo_root
            / "non-PGG_generalization/task_grounding/output"
            / "pgg_validation_demographic_only_sampling_row_resampled/seed_0/demographic_profile_cards.jsonl"
        )
    return None


def _twin_shared_notes_path_for_variant(repo_root: Path, variant_name: str) -> Path | None:
    variant_slug = _canonical_variant_name(variant_name)
    if variant_slug in {"twin_sampled_seed_0", "twin_sampled_unadjusted_seed_0"}:
        return (
            repo_root
            / "non-PGG_generalization/task_grounding/output"
            / "twin_extended_profile_cards/pgg_prompt_min/shared_prompt_notes.md"
        )
    if variant_slug == "demographic_only_row_resampled_seed_0":
        return None
    return None


def _default_run_name(args: argparse.Namespace, selected_game_count: int) -> str:
    base_name = f"{_canonical_variant_name(args.variant_name)}_{_sanitize_token(args.model)}"
    suffixes: list[str] = []
    if args.split != "val":
        suffixes.append(_sanitize_token(args.split))
    if args.selection_mode != "one_per_treatment":
        suffixes.append(_sanitize_token(args.selection_mode))
    if not args.require_valid_starting_players:
        suffixes.append("allstarts")
    if args.min_num_rounds_exclusive != 0:
        suffixes.append(f"minrounds_gt_{args.min_num_rounds_exclusive}")
    if args.repeat_count_mode != "match_valid_start_treatment_counts":
        suffixes.append(f"repeat_{args.repeats_per_game}")
    if selected_game_count != 40:
        suffixes.append(f"games_{selected_game_count}")
    if not suffixes:
        return base_name
    return "__".join([base_name, *suffixes])


def _load_avatar_map(players_path: Path) -> dict[str, str]:
    players = pd.read_csv(players_path, usecols=["_id", "data.avatar"])
    players = players.dropna(subset=["_id", "data.avatar"]).drop_duplicates("_id")
    return {
        str(row["_id"]): str(row["data.avatar"]).strip().upper()
        for _, row in players.iterrows()
    }


def _load_avatar_inventory(players_path: Path) -> list[str]:
    players = pd.read_csv(players_path, usecols=["data.avatar"])
    avatars = {
        str(value).strip().upper()
        for value in players["data.avatar"].dropna().tolist()
        if str(value).strip()
    }
    return sorted(avatars)


def _load_game_rows(games_path: Path) -> dict[str, dict[str, Any]]:
    games = pd.read_csv(games_path, usecols=["_id", "playerIds", "createdAt"])
    game_rows: dict[str, dict[str, Any]] = {}
    for _, row in games.iterrows():
        player_ids = [item.strip() for item in str(row["playerIds"]).split(",") if item.strip()]
        game_rows[str(row["_id"])] = {
            "player_order": player_ids,
            "created_at": str(row["createdAt"]),
        }
    return game_rows


def _load_prompt_metadata(processed_path: Path) -> dict[str, PromptMetadata]:
    usecols = [
        "gameId",
        "createdAt",
        "CONFIG_treatmentName",
        "CONFIG_configId",
        "CONFIG_numRounds",
        "CONFIG_endowment",
        "CONFIG_multiplier",
        "CONFIG_allOrNothing",
        "CONFIG_chat",
        "CONFIG_defaultContribProp",
        "CONFIG_punishmentExists",
        "CONFIG_punishmentCost",
        "CONFIG_punishmentMagnitude",
        "CONFIG_rewardExists",
        "CONFIG_rewardCost",
        "CONFIG_rewardMagnitude",
        "CONFIG_showNRounds",
        "CONFIG_showOtherSummaries",
        "CONFIG_showPunishmentId",
        "CONFIG_showRewardId",
        "valid_number_of_starting_players",
        "chat_log",
    ]
    frame = pd.read_csv(processed_path, usecols=usecols).drop_duplicates("gameId")
    metadata: dict[str, PromptMetadata] = {}
    for _, row in frame.iterrows():
        metadata[str(row["gameId"])] = PromptMetadata(
            game_id=str(row["gameId"]),
            treatment_name=str(row["CONFIG_treatmentName"]),
            created_at=str(row["createdAt"]),
            config_id=int(row["CONFIG_configId"]),
            num_rounds=int(row["CONFIG_numRounds"]),
            endowment=int(round(float(row["CONFIG_endowment"]))),
            multiplier=float(row["CONFIG_multiplier"]),
            all_or_nothing=_as_bool(row["CONFIG_allOrNothing"]),
            chat_enabled=_as_bool(row["CONFIG_chat"]),
            default_contrib_prop=_as_bool(row["CONFIG_defaultContribProp"]),
            punishment_exists=_as_bool(row["CONFIG_punishmentExists"]),
            punishment_cost=int(round(float(row["CONFIG_punishmentCost"]))),
            punishment_magnitude=int(round(float(row["CONFIG_punishmentMagnitude"]))),
            reward_exists=_as_bool(row["CONFIG_rewardExists"]),
            reward_cost=int(round(float(row["CONFIG_rewardCost"]))),
            reward_magnitude=int(round(float(row["CONFIG_rewardMagnitude"]))),
            show_n_rounds=_as_bool(row["CONFIG_showNRounds"]),
            show_other_summaries=_as_bool(row["CONFIG_showOtherSummaries"]),
            show_punishment_id=_as_bool(row["CONFIG_showPunishmentId"]),
            show_reward_id=_as_bool(row["CONFIG_showRewardId"]),
            valid_number_of_starting_players=_as_bool(row["valid_number_of_starting_players"]),
            chat_log="" if pd.isna(row["chat_log"]) else str(row["chat_log"]),
        )
    return metadata


def _load_valid_start_treatment_counts(processed_path: Path) -> dict[str, int]:
    frame = pd.read_csv(
        processed_path,
        usecols=["CONFIG_treatmentName", "valid_number_of_starting_players"],
    )
    frame = frame[frame["valid_number_of_starting_players"].apply(_as_bool)].copy()
    counts = frame.groupby("CONFIG_treatmentName").size()
    return {str(treatment_name): int(count) for treatment_name, count in counts.items()}


def _load_twin_game_assignments(
    assignments_path: Path,
) -> dict[str, dict[str, PersonaAssignment]]:
    assignments_by_game: dict[str, dict[str, PersonaAssignment]] = {}
    with assignments_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            game_id = str(row["gameId"])
            player_assignments: dict[str, PersonaAssignment] = {}
            for assignment in row.get("assignments", []):
                player_id = assignment.get("pgg_roster_playerId")
                player_key = (
                    str(player_id) if player_id is not None else f"seat:{int(assignment['seat_index'])}"
                )
                profile_id = (
                    str(assignment.get("twin_pid", "")).strip()
                    or str(assignment.get("profile_id", "")).strip()
                )
                player_assignments[player_key] = PersonaAssignment(
                    seat_index=int(assignment["seat_index"]),
                    player_id=str(player_id) if player_id is not None else None,
                    profile_id=profile_id,
                    headline=str(
                        assignment.get("twin_profile_headline", "")
                        or assignment.get("profile_headline", "")
                    ).strip(),
                    summary=str(
                        assignment.get("twin_profile_summary", "")
                        or assignment.get("profile_summary", "")
                    ).strip(),
                )
            assignments_by_game[game_id] = player_assignments
    return assignments_by_game


def _load_twin_profile_cards(cards_path: Path) -> dict[str, dict[str, Any]]:
    cards_by_pid: dict[str, dict[str, Any]] = {}
    with cards_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            card = json.loads(line)
            participant = card.get("participant", {})
            if isinstance(participant, dict) and participant.get("pid") is not None:
                pid = str(participant["pid"])
            else:
                pid = str(card["profile_id"])
            cards_by_pid[pid] = card
    return cards_by_pid


def _load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _parse_chat_messages(raw_value: str) -> list[dict[str, Any]]:
    if not isinstance(raw_value, str):
        return []
    text = raw_value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except Exception:
        pass

    messages: list[dict[str, Any]] = []
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace < 0 or last_brace < first_brace:
        return messages

    cursor = first_brace
    while cursor <= last_brace:
        while cursor <= last_brace and text[cursor] != "{":
            cursor += 1
        if cursor > last_brace:
            break
        start = cursor
        depth = 0
        while cursor <= last_brace:
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : cursor + 1]
                    try:
                        parsed = ast.literal_eval(chunk)
                    except Exception:
                        try:
                            parsed = json.loads(chunk)
                        except Exception:
                            parsed = None
                    if isinstance(parsed, dict):
                        messages.append(parsed)
                    cursor += 1
                    break
            cursor += 1
        else:
            break
    return messages


def _extract_round_phase(phase_text: str) -> tuple[int | None, str | None]:
    if not isinstance(phase_text, str):
        return None, None
    lowered = phase_text.lower()
    match = re.search(r"round\s+(\d+)", lowered)
    round_index = int(match.group(1)) if match else None
    if "contrib" in lowered:
        phase = "contribution"
    elif "outcome" in lowered:
        phase = "outcome"
    elif "summary" in lowered:
        phase = "summary"
    else:
        phase = None
    return round_index, phase


def _index_chat_log(chat_log: str) -> dict[int, dict[str, list[tuple[str, str]]]]:
    messages = _parse_chat_messages(chat_log)
    parsed: list[tuple[int | None, str | None, str, str]] = []
    saw_round_zero = False

    for message in messages:
        text = str(message.get("text", "")).strip().replace("\n", " ")
        if not text:
            continue
        avatar = str(message.get("avatar", "")).strip().upper()
        round_index, phase = _extract_round_phase(str(message.get("gamePhase", "")))
        if round_index == 0:
            saw_round_zero = True
        parsed.append((round_index, phase, avatar, text))

    shift = 1 if saw_round_zero else 0
    indexed: dict[int, dict[str, list[tuple[str, str]]]] = {}
    last_round_number: int | None = None
    for round_index, phase, avatar, text in parsed:
        if round_index is not None:
            round_number = round_index + shift
            last_round_number = round_number
        else:
            if last_round_number is None:
                continue
            round_number = last_round_number
        resolved_phase = phase or "outcome"
        indexed.setdefault(round_number, {}).setdefault(resolved_phase, []).append((avatar, text))
    return indexed


def _empty_chat_index() -> dict[int, dict[str, list[tuple[str, str]]]]:
    return {}


def _select_one_game_per_treatment(
    games: list[GameTrajectory],
    prompt_metadata: dict[str, PromptMetadata],
    game_rows: dict[str, dict[str, Any]],
) -> list[GameTrajectory]:
    by_treatment: dict[str, list[GameTrajectory]] = {}
    for game in games:
        metadata = prompt_metadata.get(game.game_id)
        game_row = game_rows.get(game.game_id)
        if metadata is None or game_row is None:
            continue
        if metadata.num_rounds != game.num_rounds:
            continue
        by_treatment.setdefault(metadata.treatment_name, []).append(game)

    selected: list[GameTrajectory] = []
    for treatment_name in sorted(by_treatment):
        candidates = by_treatment[treatment_name]
        candidates.sort(
            key=lambda game: (
                game_rows[game.game_id]["created_at"],
                game.game_id,
            )
        )
        selected.append(candidates[0])
    return selected


def _select_games(
    games: list[GameTrajectory],
    prompt_metadata: dict[str, PromptMetadata],
    game_rows: dict[str, dict[str, Any]],
    selection_mode: str,
    require_valid_starting_players: bool,
) -> tuple[list[GameTrajectory], str]:
    filtered_games = [
        game
        for game in games
        if game.game_id in prompt_metadata
        and game.game_id in game_rows
        and (
            not require_valid_starting_players
            or prompt_metadata[game.game_id].valid_number_of_starting_players
        )
    ]
    if selection_mode == "one_per_treatment":
        return (
            _select_one_game_per_treatment(filtered_games, prompt_metadata, game_rows),
            "one earliest-created complete validation game per CONFIG_treatmentName",
        )
    if selection_mode == "full":
        selected_games = list(filtered_games)
        selected_games.sort(key=lambda game: (game_rows[game.game_id]["created_at"], game.game_id))
        return selected_games, "all complete validation games satisfying the round filter"
    raise ValueError(f"Unsupported selection mode: {selection_mode}")


def _player_avatar_order(
    game_id: str,
    game_rows: dict[str, dict[str, Any]],
    avatar_map: dict[str, str],
    avatar_inventory: list[str],
) -> tuple[list[str], list[str]]:
    raw_player_order = list(game_rows[game_id]["player_order"])
    known_avatar_order = [
        avatar_map[player_id].strip().upper()
        for player_id in raw_player_order
        if player_id in avatar_map and avatar_map[player_id].strip()
    ]
    if len(set(known_avatar_order)) != len(known_avatar_order):
        raise ValueError(f"Game {game_id} has duplicate avatar names within the selected roster.")

    avatar_order: list[str] = []
    reserved_avatars = set(known_avatar_order)
    assigned_missing_avatars: set[str] = set()
    for player_id in raw_player_order:
        avatar = avatar_map.get(player_id)
        if avatar:
            normalized_avatar = avatar.strip().upper()
            avatar_order.append(normalized_avatar)
            continue
        replacement = next(
            (
                candidate
                for candidate in avatar_inventory
                if candidate not in reserved_avatars and candidate not in assigned_missing_avatars
            ),
            None,
        )
        if replacement is None:
            raise ValueError(
                f"Game {game_id} has missing avatar metadata and no unused replacement avatar is available."
            )
        avatar_order.append(replacement)
        assigned_missing_avatars.add(replacement)
    if len(set(avatar_order)) != len(avatar_order):
        raise ValueError(f"Game {game_id} has duplicate avatar names within the selected roster.")
    return raw_player_order, avatar_order


def _render_contributions(round_record: RoundRecord, raw_player_order: list[str]) -> str:
    values = [str(round_record.contributions[player_id]) for player_id in raw_player_order]
    return f"<<[{', '.join(values)}]>>"


def _render_interactions(
    round_record: RoundRecord,
    metadata: PromptMetadata,
    raw_player_order: list[str],
    raw_to_avatar: dict[str, str],
) -> str:
    tuples: list[str] = []
    for source_id in raw_player_order:
        for target_id in raw_player_order:
            if target_id == source_id:
                continue
            punish_units = round_record.punished[source_id].get(target_id, 0)
            reward_units = round_record.rewarded[source_id].get(target_id, 0)
            if punish_units > 0:
                unit_value = -int(punish_units) if metadata.reward_exists else int(punish_units)
                tuples.append(
                    f"({raw_to_avatar[source_id]}, {raw_to_avatar[target_id]}, {unit_value})"
                )
            if reward_units > 0:
                tuples.append(
                    f"({raw_to_avatar[source_id]}, {raw_to_avatar[target_id]}, {int(reward_units)})"
                )
    return f"<<[{', '.join(tuples)}]>>" if tuples else "<<[]>>"


def _interaction_tag_name(metadata: PromptMetadata) -> str | None:
    if metadata.punishment_exists and metadata.reward_exists:
        return "PUNISHMENT/REWARD"
    if metadata.punishment_exists:
        return "PUNISHMENT"
    if metadata.reward_exists:
        return "REWARD"
    return None


def _render_chat_lines(
    chat_index: dict[int, dict[str, list[tuple[str, str]]]],
    round_number: int,
    phase: str,
) -> list[str]:
    lines: list[str] = []
    for avatar, text in chat_index.get(round_number, {}).get(phase, []):
        lines.append(f"CHAT from {avatar}: {text}")
    return lines


def _render_round_block(
    *,
    round_number: int,
    round_record: RoundRecord,
    metadata: PromptMetadata,
    raw_player_order: list[str],
    raw_to_avatar: dict[str, str],
    chat_index: dict[int, dict[str, list[tuple[str, str]]]],
    include_chat: bool,
    include_interactions: bool,
    include_explanation: bool,
) -> str:
    lines = [f"## ROUND {round_number} BEGINS"]
    interaction_tag = _interaction_tag_name(metadata)
    if include_explanation:
        lines.extend(
            [
                f"### ROUND {round_number} EXPLANATION",
                f"<Explain how you expect round {round_number} to unfold, including player-level motivations, strategic incentives, and uncertainty.>",
            ]
        )
    if include_chat:
        lines.extend(_render_chat_lines(chat_index, round_number, "contribution"))
    lines.append(f"### CONTRIBUTIONS: {_render_contributions(round_record, raw_player_order)}")
    if include_chat:
        lines.extend(_render_chat_lines(chat_index, round_number, "outcome"))
    if include_interactions and interaction_tag is not None:
        lines.append(
            f"### {interaction_tag}: {_render_interactions(round_record, metadata, raw_player_order, raw_to_avatar)}"
        )
    lines.append(f"### ROUND {round_number} SUMMARY SHOWN TO PLAYERS")
    if include_chat:
        lines.extend(_render_chat_lines(chat_index, round_number, "summary"))
    return "\n".join(lines)


def _build_observed_prefix(
    game: GameTrajectory,
    metadata: PromptMetadata,
    raw_player_order: list[str],
    avatar_order: list[str],
    chat_index: dict[int, dict[str, list[tuple[str, str]]]],
    k: int,
) -> str:
    raw_to_avatar = dict(zip(raw_player_order, avatar_order))
    blocks = [
        _render_round_block(
            round_number=round_record.index + 1,
            round_record=round_record,
            metadata=metadata,
            raw_player_order=raw_player_order,
            raw_to_avatar=raw_to_avatar,
            chat_index=chat_index,
            include_chat=metadata.chat_enabled,
            include_interactions=metadata.punishment_exists or metadata.reward_exists,
            include_explanation=False,
        )
        for round_record in game.rounds[:k]
    ]
    return "\n".join(
        [
            "# GAME STARTS",
            f"<PLAYERS> {', '.join(avatar_order)} </PLAYERS>",
            *blocks,
        ]
    )


def _build_k0_scaffold(avatar_order: list[str]) -> str:
    return "\n".join(
        [
            "# GAME STARTS",
            f"<PLAYERS> {', '.join(avatar_order)} </PLAYERS>",
        ]
    )


def _build_gold_continuation(
    game: GameTrajectory,
    metadata: PromptMetadata,
    raw_player_order: list[str],
    avatar_order: list[str],
    chat_index: dict[int, dict[str, list[tuple[str, str]]]],
    k: int,
) -> str:
    raw_to_avatar = dict(zip(raw_player_order, avatar_order))
    blocks = [
        _render_round_block(
            round_number=round_record.index + 1,
            round_record=round_record,
            metadata=metadata,
            raw_player_order=raw_player_order,
            raw_to_avatar=raw_to_avatar,
            chat_index=chat_index,
            include_chat=metadata.chat_enabled,
            include_interactions=metadata.punishment_exists or metadata.reward_exists,
            include_explanation=False,
        )
        for round_record in game.rounds[k:]
    ]
    return "\n".join(blocks)


def _build_gold_round_payload(
    game: GameTrajectory,
    metadata: PromptMetadata,
    raw_player_order: list[str],
    avatar_order: list[str],
    chat_index: dict[int, dict[str, list[tuple[str, str]]]],
    k: int,
) -> list[dict[str, Any]]:
    raw_to_avatar = dict(zip(raw_player_order, avatar_order))
    rounds: list[dict[str, Any]] = []
    for round_record in game.rounds[k:]:
        round_number = round_record.index + 1
        interactions: list[list[Any]] = []
        for source_id in raw_player_order:
            for target_id in raw_player_order:
                if target_id == source_id:
                    continue
                punish_units = round_record.punished[source_id].get(target_id, 0)
                reward_units = round_record.rewarded[source_id].get(target_id, 0)
                if punish_units > 0:
                    unit_value = -int(punish_units) if metadata.reward_exists else int(punish_units)
                    interactions.append(
                        [raw_to_avatar[source_id], raw_to_avatar[target_id], unit_value]
                    )
                if reward_units > 0:
                    interactions.append(
                        [raw_to_avatar[source_id], raw_to_avatar[target_id], int(reward_units)]
                    )
        chat_messages: list[dict[str, Any]] = []
        for phase in PHASE_ORDER:
            for avatar, text in chat_index.get(round_number, {}).get(phase, []):
                chat_messages.append(
                    {
                        "speaker": avatar,
                        "text": text,
                        "phase": phase,
                    }
                )
        rounds.append(
            {
                "round_number": round_number,
                "contributions": [round_record.contributions[player_id] for player_id in raw_player_order],
                "interactions": interactions,
                "messages": chat_messages,
            }
        )
    return rounds


def _contribution_rule(metadata: PromptMetadata) -> str:
    if metadata.default_contrib_prop:
        if metadata.all_or_nothing:
            return (
                f"Each round, the {metadata.endowment} coins begin in the shared pot. "
                f"Each player chooses how many coins to withdraw for private use, so the resulting contribution must be either 0 or {metadata.endowment}."
            )
        return (
            f"Each round, the {metadata.endowment} coins begin in the shared pot. "
            f"Each player chooses how many coins to withdraw for private use, and the remainder becomes their contribution. "
            f"Contributions are integers from 0 to {metadata.endowment}."
        )
    if metadata.all_or_nothing:
        return (
            f"Each player receives {metadata.endowment} coins in private holdings each round and must contribute either 0 or {metadata.endowment} to the shared pot."
        )
    return (
        f"Each player receives {metadata.endowment} coins in private holdings each round and chooses an integer contribution from 0 to {metadata.endowment}."
    )


def _visibility_rules(metadata: PromptMetadata) -> list[str]:
    lines = []
    if metadata.chat_enabled:
        lines.append("Players may send messages to the group whenever they choose to do so.")
    else:
        lines.append("Players cannot send group chat messages in this game.")
    if metadata.show_n_rounds:
        lines.append(f"Players know that the game lasts {metadata.num_rounds} rounds.")
    else:
        lines.append("Players do not know the total number of rounds while playing.")
    if metadata.punishment_exists and metadata.reward_exists:
        if metadata.show_punishment_id == metadata.show_reward_id:
            if metadata.show_punishment_id:
                lines.append("Players can identify who punished/rewarded them in their summary information.")
            else:
                lines.append("Players cannot identify who punished/rewarded them in their summary information.")
        else:
            if metadata.show_punishment_id:
                lines.append("Players can identify who punished them in their summary information.")
            else:
                lines.append("Players cannot identify who punished them in their summary information.")
            if metadata.show_reward_id:
                lines.append("Players can identify who rewarded them in their summary information.")
            else:
                lines.append("Players cannot identify who rewarded them in their summary information.")
    elif metadata.punishment_exists:
        if metadata.show_punishment_id:
            lines.append("Players can identify who punished them in their summary information.")
        else:
            lines.append("Players cannot identify who punished them in their summary information.")
    elif metadata.reward_exists:
        if metadata.show_reward_id:
            lines.append("Players can identify who rewarded them in their summary information.")
        else:
            lines.append("Players cannot identify who rewarded them in their summary information.")
    if metadata.punishment_exists and metadata.reward_exists:
        lines.append(
            "At the summary stage (`ROUND N SUMMARY SHOWN TO PLAYERS`), each player sees their own net payoff of the round, along with the amount they used for punishing and rewarding other players, amount deducted from punishment, and amount received for reward."
        )
    elif metadata.punishment_exists:
        lines.append(
            "At the summary stage (`ROUND N SUMMARY SHOWN TO PLAYERS`), each player sees their own net payoff of the round, along with the amount they used for punishing other players and amount deducted from punishment."
        )
    elif metadata.reward_exists:
        lines.append(
            "At the summary stage (`ROUND N SUMMARY SHOWN TO PLAYERS`), each player sees their own net payoff of the round, along with the amount they used for rewarding other players and amount received for reward."
        )
    else:
        lines.append(
            "At the summary stage (`ROUND N SUMMARY SHOWN TO PLAYERS`), each player sees their own net payoff of the round."
        )
    if metadata.show_other_summaries:
        lines.append("The corresponding information of the peers are also shown.")
    else:
        lines.append("However, the corresponding information of the peers are not shown.")
    return lines


def _mechanism_rules(metadata: PromptMetadata) -> list[str]:
    lines = []
    if metadata.punishment_exists and metadata.reward_exists:
        lines.append("After contributions are revealed and redistributed, players may punish and reward other players.")
        lines.append(
            f"Each punishment unit costs {_coin_phrase(metadata.punishment_cost)} to the source and deducts {_coin_phrase(metadata.punishment_magnitude)} from the target."
        )
        lines.append(
            f"Each reward unit costs {_coin_phrase(metadata.reward_cost)} to the source and adds {_coin_phrase(metadata.reward_magnitude)} to the target."
        )
    elif metadata.punishment_exists:
        lines.append("After contributions are revealed and redistributed, players may punish other players.")
        lines.append(
            f"Each punishment unit costs {_coin_phrase(metadata.punishment_cost)} to the source and deducts {_coin_phrase(metadata.punishment_magnitude)} from the target."
        )
    elif metadata.reward_exists:
        lines.append("After contributions are revealed and redistributed, players may reward other players.")
        lines.append(
            f"Each reward unit costs {_coin_phrase(metadata.reward_cost)} to the source and adds {_coin_phrase(metadata.reward_magnitude)} to the target."
        )
    return lines


def _behavioral_sparsity_hints(metadata: PromptMetadata) -> list[str]:
    lines: list[str] = []
    if metadata.chat_enabled:
        lines.append("- In real games, players usually send relatively few chat messages. Keep chat sparse and purposeful.")
    if metadata.punishment_exists:
        lines.append("- In real games, players usually punish sparingly. Do not over-predict punishment actions.")
    if metadata.reward_exists:
        lines.append("- In real games, players usually reward sparingly. Do not over-predict reward actions.")
    return lines


def _strict_transcript_template(metadata: PromptMetadata, start_round: int) -> list[str]:
    interaction_tag = _interaction_tag_name(metadata)
    lines = [
        "Strict continuation template:",
        "```text",
        "### GAME EXPLANATION",
        "<Explain your overall prediction for how the game is likely to unfold.>",
        f"## ROUND {start_round} BEGINS",
        f"### ROUND {start_round} EXPLANATION",
        "<Explain why you expect this round to unfold this way.>",
    ]
    if metadata.chat_enabled:
        lines.append("CHAT from AVATAR: <message>")
    lines.append("### CONTRIBUTIONS: <<[c1, c2, ...]>>")
    if metadata.chat_enabled:
        lines.append("CHAT from AVATAR: <message>")
    if interaction_tag is not None:
        lines.append(f"### {interaction_tag}: <<[(SOURCE, TARGET, UNIT), ...]>>")
    lines.append(f"### ROUND {start_round} SUMMARY SHOWN TO PLAYERS")
    if metadata.chat_enabled:
        lines.append("CHAT from AVATAR: <message>")
    lines.append(f"## ROUND {start_round + 1} BEGINS")
    lines.append("...")
    lines.append("```")
    return lines


def _build_persona_block(
    *,
    variant_slug: str,
    shared_prompt_notes: str | None,
    raw_player_order: list[str],
    avatar_order: list[str],
    persona_assignments: dict[str, PersonaAssignment],
    twin_profile_cards: dict[str, dict[str, Any]],
) -> str:
    is_demographic_only = _is_demographic_only_variant(variant_slug)
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


def _build_user_prompt(
    num_rounds: int,
    metadata: PromptMetadata,
    avatar_order: list[str],
    transcript_prefix: str,
    k: int,
    persona_block: str | None = None,
) -> str:
    interaction_tag = _interaction_tag_name(metadata)
    if k == 0:
        output_requirement_lines = [
            "- First write `### GAME EXPLANATION` followed by your explanation of how the game is likely to unfold.",
            "- Then generate the full game directly in transcript form from round 1 onward.",
            "- At the start of each round, immediately after `## ROUND N BEGINS`, write `### ROUND N EXPLANATION`.",
            "- Use the exact avatar order from `<PLAYERS>` for every contribution list.",
            "- Keep the contribution line wrapped as `### CONTRIBUTIONS: <<[...]>>`.",
        ]
    else:
        output_requirement_lines = [
            "- First write `### GAME EXPLANATION` followed by a detailed explanation of the game state, what each player's behavior suggests, and where uncertainty remains.",
            "- After that, continue only the unobserved rounds in the same transcript format as the observed history.",
            "- At the start of each predicted round, immediately after `## ROUND N BEGINS`, write `### ROUND N EXPLANATION`.",
            "- Use the exact avatar order from `<PLAYERS>` for every contribution list.",
            "- Keep the contribution line wrapped as `### CONTRIBUTIONS: <<[...]>>`.",
        ]
    if interaction_tag is not None:
        output_requirement_lines.extend(
            [
            f"- Keep the action line wrapped as `### {interaction_tag}: <<[...]>>`.",
            f"- In `### {interaction_tag}`, use tuples `(source_avatar, target_avatar, unit)`.",
            f"- Include only non-zero action edges. If no one acts in a round, write `### {interaction_tag}: <<[]>>`.",
            ]
        )
        if metadata.punishment_exists and metadata.reward_exists:
            output_requirement_lines.append("- Positive `unit` means reward and negative `unit` means punishment.")
        elif metadata.punishment_exists:
            output_requirement_lines.append("- Because this game only has punishment, every `unit` must be positive.")
        elif metadata.reward_exists:
            output_requirement_lines.append("- Because this game only has reward, every `unit` must be positive.")
    if metadata.chat_enabled:
        output_requirement_lines.append("- Predict chat inline as transcript lines, using `CHAT from AVATAR: ...`.")
    output_requirement_lines.append("- Preserve exact avatar names.")
    intro_line = (
        f"Generate a full game transcript from round 1 through round {num_rounds}."
        if k == 0
        else f"Predict every remaining round from round {k + 1} through round {num_rounds}."
    )
    history_label = "Start from this scaffold:" if k == 0 else "Observed history:"
    lines = [
        intro_line,
        "",
        "# GAME RULES",
        "This is an online public goods game (PGG).",
        _contribution_rule(metadata),
        "Players do not see others' choices before deciding.",
        f"The shared pot is multiplied by {_format_num(metadata.multiplier)} and split equally among all active players.",
        *(_mechanism_rules(metadata)),
        *(_visibility_rules(metadata)),
        *(["", persona_block] if persona_block else []),
        "",
        "Output requirements:",
        *output_requirement_lines,
        *(_behavioral_sparsity_hints(metadata)),
        *(_strict_transcript_template(metadata, k + 1) if k == 0 else []),
        "",
        history_label,
        transcript_prefix,
    ]
    return "\n".join(lines)


def _batch_entry(
    *,
    custom_id: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build k=0 forecasting batch inputs for full-rollout public-goods-game simulation."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", type=str, choices=["val"], default="val")
    parser.add_argument("--k-values", type=str, default="0")
    parser.add_argument("--min-num-rounds-exclusive", type=int, default=0)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--variant-name", type=str, default="baseline_direct_transcript")
    parser.add_argument("--repeats-per-game", type=int, default=1)
    parser.add_argument(
        "--repeat-count-mode",
        type=str,
        choices=["fixed", "match_valid_start_treatment_counts"],
        default="fixed",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=["one_per_treatment", "full"],
        default="one_per_treatment",
    )
    parser.add_argument("--require-valid-starting-players", action="store_true")
    parser.add_argument("--request-file-prefix", type=str, default="request_batch")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    variant_slug = _canonical_variant_name(args.variant_name)

    if args.split != "val":
        raise ValueError("Only validation-wave generation is supported in this compact builder.")
    if _is_assigned_profile_variant(variant_slug):
        if args.selection_mode != "full":
            raise ValueError(
                f"{variant_slug} must be built with --selection-mode full so each request uses "
                "its own assigned validation game."
            )
        if not args.require_valid_starting_players:
            raise ValueError(
                f"{variant_slug} must be built with --require-valid-starting-players."
            )
        if args.repeat_count_mode != "fixed" or args.repeats_per_game != 1:
            raise ValueError(
                f"{variant_slug} must be built with --repeat-count-mode fixed "
                "--repeats-per-game 1 because the 417 assigned games already provide the "
                "within-CONFIG sampling."
            )

    k_values = _parse_k_values(args.k_values)
    if k_values != [0]:
        raise ValueError("forecasting/build_batch_inputs.py is reserved for the k=0 full-rollout task.")
    repo_root = args.repo_root
    forecasting_root = args.forecasting_root
    batch_input_dir = forecasting_root / "batch_input"
    batch_output_dir = forecasting_root / "batch_output"
    metadata_root = forecasting_root / "metadata"
    results_root = forecasting_root / "results"
    for directory in [batch_input_dir, batch_output_dir, metadata_root, results_root]:
        directory.mkdir(parents=True, exist_ok=True)

    complete_games = load_wave_games(
        repo_root=repo_root,
        wave_name="validation_wave",
        processed_suffix="val",
        min_num_rounds_exclusive=args.min_num_rounds_exclusive,
    )
    complete_games_by_id = {game.game_id: game for game in complete_games}
    prompt_metadata = _load_prompt_metadata(repo_root / "data/processed_data/df_analysis_val.csv")
    valid_start_treatment_counts = _load_valid_start_treatment_counts(
        repo_root / "data/processed_data/df_analysis_val.csv"
    )
    players_path = repo_root / "data/raw_data/validation_wave/players.csv"
    avatar_map = _load_avatar_map(players_path)
    avatar_inventory = _load_avatar_inventory(players_path)
    game_rows = _load_game_rows(repo_root / "data/raw_data/validation_wave/games.csv")
    twin_assignments_path = _twin_assignment_path_for_variant(repo_root, args.variant_name)
    twin_cards_path = _twin_prompt_cards_path_for_variant(repo_root, args.variant_name)
    twin_shared_notes_path = _twin_shared_notes_path_for_variant(repo_root, args.variant_name)
    twin_assignments_by_game = (
        _load_twin_game_assignments(twin_assignments_path)
        if twin_assignments_path is not None
        else {}
    )
    twin_profile_cards = (
        _load_twin_profile_cards(twin_cards_path) if twin_cards_path is not None else {}
    )
    twin_shared_prompt_notes = (
        _load_text_file(twin_shared_notes_path) if twin_shared_notes_path is not None else None
    )

    if twin_assignments_path is not None:
        selected_game_ids = sorted(
            twin_assignments_by_game,
            key=lambda game_id: (
                game_rows[game_id]["created_at"],
                game_id,
            ),
        )
        selection_rule = "all valid-start validation games from the assigned profile manifest"
    else:
        selected_games, selection_rule = _select_games(
            games=complete_games,
            prompt_metadata=prompt_metadata,
            game_rows=game_rows,
            selection_mode=args.selection_mode,
            require_valid_starting_players=args.require_valid_starting_players,
        )
        selected_game_ids = [game.game_id for game in selected_games]

    selected_rows: list[dict[str, Any]] = []
    all_batch_rows: list[dict[str, Any]] = []
    all_gold_rows: list[dict[str, Any]] = []
    request_manifest_rows: list[dict[str, Any]] = []

    for game_id in selected_game_ids:
        metadata = prompt_metadata[game_id]
        raw_player_order, avatar_order = _player_avatar_order(
            game_id,
            game_rows,
            avatar_map,
            avatar_inventory,
        )
        if twin_assignments_path is not None:
            persona_assignments = twin_assignments_by_game.get(game_id)
            if persona_assignments is None:
                raise ValueError(
                    f"Missing twin persona assignments for selected game {game_id}."
                )
            missing_assignment_slots = []
            missing_profile_ids = set()
            for seat_index, player_id in enumerate(raw_player_order, start=1):
                assignment = persona_assignments.get(player_id) or persona_assignments.get(
                    f"seat:{seat_index}"
                )
                if assignment is None:
                    missing_assignment_slots.append(player_id)
                    continue
                if assignment.profile_id not in twin_profile_cards:
                    missing_profile_ids.add(assignment.profile_id)
            if missing_assignment_slots:
                raise ValueError(
                    f"Game {game_id} is missing profile assignments for roster players: "
                    f"{missing_assignment_slots}"
                )
            if missing_profile_ids:
                raise ValueError(
                    f"Game {game_id} references profile cards that are missing from "
                    f"{twin_cards_path}: {sorted(missing_profile_ids)}"
                )
        repeat_count = (
            valid_start_treatment_counts.get(metadata.treatment_name, 1)
            if args.repeat_count_mode == "match_valid_start_treatment_counts"
            else args.repeats_per_game
        )
        selected_rows.append(
            {
                "game_id": game_id,
                "treatment_name": metadata.treatment_name,
                "config_id": metadata.config_id,
                "created_at": game_rows[game_id]["created_at"],
                "num_rounds": metadata.num_rounds,
                "num_players": len(raw_player_order),
                "avatars": json.dumps(avatar_order),
                "valid_number_of_starting_players": metadata.valid_number_of_starting_players,
                "has_complete_gold_trajectory": game_id in complete_games_by_id,
                "repeat_count": repeat_count,
            }
        )

    run_name = args.run_name or _default_run_name(args, len(selected_game_ids))
    metadata_dir = metadata_root / run_name
    metadata_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = metadata_dir / "sample_prompts"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for k in k_values:
        batch_rows: list[dict[str, Any]] = []
        gold_rows: list[dict[str, Any]] = []
        sample_written = False

        for game_id in selected_game_ids:
            metadata = prompt_metadata[game_id]
            if metadata.num_rounds <= k:
                continue
            raw_player_order, avatar_order = _player_avatar_order(
                game_id,
                game_rows,
                avatar_map,
                avatar_inventory,
            )
            persona_block = None
            if twin_assignments_path is not None:
                persona_block = _build_persona_block(
                    variant_slug=variant_slug,
                    shared_prompt_notes=twin_shared_prompt_notes,
                    raw_player_order=raw_player_order,
                    avatar_order=avatar_order,
                    persona_assignments=twin_assignments_by_game[game_id],
                    twin_profile_cards=twin_profile_cards,
                )
            chat_index = _index_chat_log(metadata.chat_log) if metadata.chat_enabled else _empty_chat_index()
            complete_game = complete_games_by_id.get(game_id)

            transcript_prefix = (
                _build_observed_prefix(
                    game=complete_game,
                    metadata=metadata,
                    raw_player_order=raw_player_order,
                    avatar_order=avatar_order,
                    chat_index=chat_index,
                    k=k,
                )
                if complete_game is not None and k > 0
                else _build_k0_scaffold(avatar_order)
            )
            user_prompt = _build_user_prompt(
                num_rounds=metadata.num_rounds,
                metadata=metadata,
                avatar_order=avatar_order,
                transcript_prefix=transcript_prefix,
                k=k,
                persona_block=persona_block,
            )
            repeat_count = (
                valid_start_treatment_counts.get(metadata.treatment_name, 1)
                if args.repeat_count_mode == "match_valid_start_treatment_counts"
                else args.repeats_per_game
            )
            for repeat_index in range(1, repeat_count + 1):
                custom_id = (
                    f"trajectory_completion_compact__{metadata.treatment_name}__{game_id}"
                    f"__k{k}__rep{repeat_index}"
                )
                batch_row = _batch_entry(
                    custom_id=custom_id,
                    model=args.model,
                    system_prompt=CONTINUATION_SYSTEM_PROMPT if k > 0 else FULL_ROLLOUT_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
                gold_row = {
                    "custom_id": custom_id,
                    "game_id": game_id,
                    "treatment_name": metadata.treatment_name,
                    "config_id": metadata.config_id,
                    "k": k,
                    "repeat_index": repeat_index,
                    "repeat_count_for_treatment": repeat_count,
                    "players": avatar_order,
                    "has_complete_gold_trajectory": complete_game is not None,
                    "gold_continuation_text": (
                        _build_gold_continuation(
                            game=complete_game,
                            metadata=metadata,
                            raw_player_order=raw_player_order,
                            avatar_order=avatar_order,
                            chat_index=chat_index,
                            k=k,
                        )
                        if complete_game is not None
                        else ""
                    ),
                    "gold_rounds": (
                        _build_gold_round_payload(
                            game=complete_game,
                            metadata=metadata,
                            raw_player_order=raw_player_order,
                            avatar_order=avatar_order,
                            chat_index=chat_index,
                            k=k,
                        )
                        if complete_game is not None
                        else []
                    ),
                }
                batch_rows.append(batch_row)
                gold_rows.append(gold_row)
                request_manifest_rows.append(
                    {
                        "custom_id": custom_id,
                        "game_id": game_id,
                        "treatment_name": metadata.treatment_name,
                        "config_id": metadata.config_id,
                        "k": k,
                        "repeat_index": repeat_index,
                        "repeat_count_for_treatment": repeat_count,
                        "num_rounds": metadata.num_rounds,
                        "num_players": len(raw_player_order),
                        "avatars": json.dumps(avatar_order),
                        "chat_enabled": metadata.chat_enabled,
                        "punishment_enabled": metadata.punishment_exists,
                        "reward_enabled": metadata.reward_exists,
                        "all_or_nothing": metadata.all_or_nothing,
                        "valid_number_of_starting_players": metadata.valid_number_of_starting_players,
                        "has_complete_gold_trajectory": complete_game is not None,
                        "persona_variant": _canonical_variant_name(args.variant_name),
                    }
                )

            if not sample_written:
                sample_text = "\n\n".join(
                    [
                        "=== SYSTEM MESSAGE ===",
                        CONTINUATION_SYSTEM_PROMPT if k > 0 else FULL_ROLLOUT_SYSTEM_PROMPT,
                        "=== USER MESSAGE ===",
                        user_prompt,
                    ]
                )
                (sample_dir / f"sample_prompt_k{k}.txt").write_text(sample_text, encoding="utf-8")
                sample_written = True

        _write_jsonl(batch_input_dir / f"{run_name}.jsonl", batch_rows)
        _write_jsonl(metadata_dir / f"gold_continuations_k{k}.jsonl", gold_rows)
        all_batch_rows.extend(batch_rows)
        all_gold_rows.extend(gold_rows)

    selected_path = metadata_dir / "selected_games.csv"
    with selected_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0].keys()) if selected_rows else [])
        if selected_rows:
            writer.writeheader()
            writer.writerows(selected_rows)

    manifest_path = metadata_dir / "request_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(request_manifest_rows[0].keys()) if request_manifest_rows else [],
        )
        if request_manifest_rows:
            writer.writeheader()
            writer.writerows(request_manifest_rows)

    manifest = {
        "repo_root": str(repo_root),
        "split": args.split,
        "min_num_rounds_exclusive": args.min_num_rounds_exclusive,
        "k_values": k_values,
        "selection_mode": args.selection_mode,
        "selection_rule": selection_rule,
        "require_valid_starting_players": args.require_valid_starting_players,
        "selected_game_count": len(selected_game_ids),
        "run_name": run_name,
        "variant_name": args.variant_name,
        "repeat_count_mode": args.repeat_count_mode,
        "repeats_per_game": args.repeats_per_game,
        "total_request_count": len(request_manifest_rows),
        "treatment_names": [prompt_metadata[game_id].treatment_name for game_id in selected_game_ids],
        "model": args.model,
        "persona_assignment_file": (
            str(twin_assignments_path) if twin_assignments_path is not None else None
        ),
        "persona_cards_file": str(twin_cards_path) if twin_cards_path is not None else None,
        "persona_shared_notes_file": (
            str(twin_shared_notes_path) if twin_shared_notes_path is not None else None
        ),
        "request_file_prefix": args.request_file_prefix,
        "batch_input_file": str(batch_input_dir / f"{run_name}.jsonl"),
        "expected_batch_output_file": str(batch_output_dir / f"{run_name}.jsonl"),
        "metadata_dir": str(metadata_dir),
        "requests_by_k": {
            str(k): int(sum(row["k"] == k for row in request_manifest_rows))
            for k in k_values
        },
        "total_requests": len(all_batch_rows),
    }
    (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote forecasting batch input to {batch_input_dir / f'{run_name}.jsonl'}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
