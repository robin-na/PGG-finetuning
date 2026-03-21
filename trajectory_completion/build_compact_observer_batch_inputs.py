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

from .data import GameTrajectory, RoundRecord, load_wave_games


SYSTEM_PROMPT = """You forecast the remaining rounds of an observed public goods game.

First write a careful free-form reflection about what the observed history suggests about each player's incentives, values, beliefs, strategy, and uncertainty. Be cautious where the evidence is weak.

Then continue the game in the same transcript format as the observed history. Keep the stage ordering coherent within each round. Any predicted action lines that are wrapped in the observed history should stay wrapped in <<>> exactly."""


PHASE_ORDER = ("contribution", "outcome", "summary")


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


def _load_avatar_map(players_path: Path) -> dict[str, str]:
    players = pd.read_csv(players_path, usecols=["_id", "data.avatar"])
    players = players.dropna(subset=["_id", "data.avatar"]).drop_duplicates("_id")
    return {
        str(row["_id"]): str(row["data.avatar"]).strip().upper()
        for _, row in players.iterrows()
    }


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
) -> tuple[list[str], list[str]]:
    raw_player_order = list(game_rows[game_id]["player_order"])
    avatar_order = [avatar_map[player_id] for player_id in raw_player_order]
    if len(set(avatar_order)) != len(avatar_order):
        raise ValueError(f"Game {game_id} has duplicate avatar names within the selected roster.")
    return raw_player_order, avatar_order


def _render_contributions(round_record: RoundRecord, raw_player_order: list[str]) -> str:
    values = [str(round_record.contributions[player_id]) for player_id in raw_player_order]
    return f"<<[{', '.join(values)}]>>"


def _render_interactions(
    round_record: RoundRecord,
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
                tuples.append(
                    f"({raw_to_avatar[source_id]}, {raw_to_avatar[target_id]}, {-int(punish_units)})"
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
            f"### {interaction_tag}: {_render_interactions(round_record, raw_player_order, raw_to_avatar)}"
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
                    interactions.append(
                        [raw_to_avatar[source_id], raw_to_avatar[target_id], -int(punish_units)]
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


def _strict_transcript_template(metadata: PromptMetadata, start_round: int) -> list[str]:
    interaction_tag = _interaction_tag_name(metadata)
    lines = [
        "Strict continuation template:",
        "```text",
        "### OVERALL REFLECTION",
        "<Write a detailed reflection before the transcript continuation.>",
        f"## ROUND {start_round} BEGINS",
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


def _build_user_prompt(
    game: GameTrajectory,
    metadata: PromptMetadata,
    avatar_order: list[str],
    transcript_prefix: str,
    k: int,
) -> str:
    interaction_tag = _interaction_tag_name(metadata)
    output_requirement_lines = [
        "- First write `### OVERALL REFLECTION` followed by a detailed reflection on the state of the game, what each player's behavior suggests, and where uncertainty remains.",
        "- After that, continue only the unobserved rounds in the same transcript format as the observed history.",
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
            output_requirement_lines.append("- Because this game only has punishment, every `unit` must be negative.")
        elif metadata.reward_exists:
            output_requirement_lines.append("- Because this game only has reward, every `unit` must be positive.")
    output_requirement_lines.extend(
        [
            "- Predict chat inline as transcript lines, using `CHAT from AVATAR: ...`.",
            "- Preserve exact avatar names.",
        ]
    )
    lines = [
        f"Predict every remaining round from round {k + 1} through round {game.num_rounds}.",
        "",
        "# GAME RULES",
        "This is an online public goods game (PGG).",
        _contribution_rule(metadata),
        "Players do not see others' choices before deciding.",
        f"The shared pot is multiplied by {_format_num(metadata.multiplier)} and split equally among all active players.",
        *(_mechanism_rules(metadata)),
        *(_visibility_rules(metadata)),
        "",
        "Output requirements:",
        *output_requirement_lines,
        *(_strict_transcript_template(metadata, k + 1) if k == 0 else []),
        "",
        "Observed history:",
        transcript_prefix,
    ]
    return "\n".join(lines)


def _batch_entry(
    *,
    custom_id: str,
    model: str,
    user_prompt: str,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
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
        description="Build compact observer-view OpenAI Batch inputs for trajectory completion."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", type=str, choices=["val"], default="val")
    parser.add_argument("--k-values", type=str, default="1,3,5,8")
    parser.add_argument("--min-num-rounds-exclusive", type=int, default=8)
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=["one_per_treatment", "full"],
        default="one_per_treatment",
    )
    parser.add_argument("--require-valid-starting-players", action="store_true")
    parser.add_argument("--request-file-prefix", type=str, default="request_batch")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "batch_inputs"
        / "validation_wave_one_per_treatment_gt8_k1358_compact_observer",
    )
    args = parser.parse_args()

    if args.split != "val":
        raise ValueError("Only validation-wave generation is supported in this compact builder.")

    k_values = _parse_k_values(args.k_values)
    repo_root = args.repo_root

    games = load_wave_games(
        repo_root=repo_root,
        wave_name="validation_wave",
        processed_suffix="val",
        min_num_rounds_exclusive=args.min_num_rounds_exclusive,
    )
    prompt_metadata = _load_prompt_metadata(repo_root / "data/processed_data/df_analysis_val.csv")
    avatar_map = _load_avatar_map(repo_root / "data/raw_data/validation_wave/players.csv")
    game_rows = _load_game_rows(repo_root / "data/raw_data/validation_wave/games.csv")

    selected_games, selection_rule = _select_games(
        games=games,
        prompt_metadata=prompt_metadata,
        game_rows=game_rows,
        selection_mode=args.selection_mode,
        require_valid_starting_players=args.require_valid_starting_players,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output_dir / "sample_prompts"
    sample_dir.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict[str, Any]] = []
    all_batch_rows: list[dict[str, Any]] = []
    all_gold_rows: list[dict[str, Any]] = []
    request_manifest_rows: list[dict[str, Any]] = []

    for game in selected_games:
        metadata = prompt_metadata[game.game_id]
        raw_player_order, avatar_order = _player_avatar_order(game.game_id, game_rows, avatar_map)
        selected_rows.append(
            {
                "game_id": game.game_id,
                "treatment_name": metadata.treatment_name,
                "config_id": metadata.config_id,
                "created_at": game_rows[game.game_id]["created_at"],
                "num_rounds": game.num_rounds,
                "num_players": len(raw_player_order),
                "avatars": json.dumps(avatar_order),
                "valid_number_of_starting_players": metadata.valid_number_of_starting_players,
            }
        )

    for k in k_values:
        batch_rows: list[dict[str, Any]] = []
        gold_rows: list[dict[str, Any]] = []
        sample_written = False

        for game in selected_games:
            if game.num_rounds <= k:
                continue
            metadata = prompt_metadata[game.game_id]
            raw_player_order, avatar_order = _player_avatar_order(game.game_id, game_rows, avatar_map)
            chat_index = _index_chat_log(metadata.chat_log) if metadata.chat_enabled else _empty_chat_index()

            transcript_prefix = _build_observed_prefix(
                game=game,
                metadata=metadata,
                raw_player_order=raw_player_order,
                avatar_order=avatar_order,
                chat_index=chat_index,
                k=k,
            )
            user_prompt = _build_user_prompt(
                game=game,
                metadata=metadata,
                avatar_order=avatar_order,
                transcript_prefix=transcript_prefix,
                k=k,
            )

            custom_id = f"trajectory_completion_compact__{metadata.treatment_name}__{game.game_id}__k{k}"
            batch_row = _batch_entry(
                custom_id=custom_id,
                model=args.model,
                user_prompt=user_prompt,
            )
            gold_row = {
                "custom_id": custom_id,
                "game_id": game.game_id,
                "treatment_name": metadata.treatment_name,
                "config_id": metadata.config_id,
                "k": k,
                "players": avatar_order,
                "gold_continuation_text": _build_gold_continuation(
                    game=game,
                    metadata=metadata,
                    raw_player_order=raw_player_order,
                    avatar_order=avatar_order,
                    chat_index=chat_index,
                    k=k,
                ),
                "gold_rounds": _build_gold_round_payload(
                    game=game,
                    metadata=metadata,
                    raw_player_order=raw_player_order,
                    avatar_order=avatar_order,
                    chat_index=chat_index,
                    k=k,
                ),
            }
            batch_rows.append(batch_row)
            gold_rows.append(gold_row)
            request_manifest_rows.append(
                {
                    "custom_id": custom_id,
                    "game_id": game.game_id,
                    "treatment_name": metadata.treatment_name,
                    "config_id": metadata.config_id,
                    "k": k,
                    "num_rounds": game.num_rounds,
                    "num_players": len(raw_player_order),
                    "avatars": json.dumps(avatar_order),
                    "chat_enabled": metadata.chat_enabled,
                    "punishment_enabled": metadata.punishment_exists,
                    "reward_enabled": metadata.reward_exists,
                    "all_or_nothing": metadata.all_or_nothing,
                    "valid_number_of_starting_players": metadata.valid_number_of_starting_players,
                }
            )

            if not sample_written:
                sample_text = "\n\n".join(
                    [
                        "=== SYSTEM MESSAGE ===",
                        SYSTEM_PROMPT,
                        "=== USER MESSAGE ===",
                        user_prompt,
                    ]
                )
                (sample_dir / f"sample_prompt_k{k}.txt").write_text(sample_text, encoding="utf-8")
                sample_written = True

        _write_jsonl(args.output_dir / f"{args.request_file_prefix}_k{k}.jsonl", batch_rows)
        _write_jsonl(args.output_dir / f"gold_continuations_k{k}.jsonl", gold_rows)
        all_batch_rows.extend(batch_rows)
        all_gold_rows.extend(gold_rows)

    _write_jsonl(args.output_dir / f"{args.request_file_prefix}_all_k.jsonl", all_batch_rows)
    _write_jsonl(args.output_dir / "gold_continuations_all_k.jsonl", all_gold_rows)

    selected_path = args.output_dir / "selected_games.csv"
    with selected_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0].keys()) if selected_rows else [])
        if selected_rows:
            writer.writeheader()
            writer.writerows(selected_rows)

    manifest_path = args.output_dir / "request_manifest.csv"
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
        "selected_game_count": len(selected_games),
        "treatment_names": [prompt_metadata[game.game_id].treatment_name for game in selected_games],
        "model": args.model,
        "request_file_prefix": args.request_file_prefix,
        "requests_by_k": {
            str(k): int(sum(game.num_rounds > k for game in selected_games))
            for k in k_values
        },
        "total_requests": len(all_batch_rows),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote compact batch inputs to {args.output_dir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
