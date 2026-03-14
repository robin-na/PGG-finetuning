from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from concordia.associative_memory import basic_associative_memory
from concordia.environment import engine as concordia_engine
from concordia.environment.engines import simultaneous as simultaneous_engine
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib

try:
    from .parsers import parse_json_response
    from .prompt_builder import (
        actions_tag,
        mech_info,
        peers_contributions_csv,
        redist_line,
        round_info_line,
        round_open,
        system_header_plain,
    )
    from .simulator import (
        GameContext,
        _append_chat_history_line,
        _apply_continuation_gate,
        _build_initial_macro_transcripts,
        _clamp_probability,
        _json_compact,
        _parse_binary_action_response,
        _resolve_archetype_mode,
        _use_binary_targets,
    )
    from .utils import as_bool, is_nan
except ImportError:
    from parsers import parse_json_response
    from prompt_builder import (
        actions_tag,
        mech_info,
        peers_contributions_csv,
        redist_line,
        round_info_line,
        round_open,
        system_header_plain,
    )
    from simulator import (
        GameContext,
        _append_chat_history_line,
        _apply_continuation_gate,
        _build_initial_macro_transcripts,
        _clamp_probability,
        _json_compact,
        _parse_binary_action_response,
        _resolve_archetype_mode,
        _use_binary_targets,
    )
    from utils import as_bool, is_nan


MAKE_OBSERVATION_CALL = "PGG_MAKE_OBSERVATION::{name}"
NEXT_ACTING_CALL = "PGG_NEXT_ACTING"
NEXT_ACTION_SPEC_CALL = "PGG_NEXT_ACTION_SPEC::{name}"
RESOLVE_CALL = "PGG_RESOLVE"
TERMINATE_CALL = "PGG_TERMINATE"


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class StageSpec:
    round_index: int
    name: str

    @property
    def starts_round(self) -> bool:
        return self.name in {"chat", "contribution"}


def build_simultaneous_engine() -> simultaneous_engine.Simultaneous:
    return simultaneous_engine.Simultaneous(
        call_to_make_observation=MAKE_OBSERVATION_CALL,
        call_to_next_acting=NEXT_ACTING_CALL,
        call_to_next_action_spec=NEXT_ACTION_SPEC_CALL,
        call_to_resolve=RESOLVE_CALL,
        call_to_check_termination=TERMINATE_CALL,
    )


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _serialize_log(value: Any) -> Optional[str]:
    if not value:
        return None
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value).strip()
    text = text.strip()
    return text or None


def _extract_name(call_to_action: str, prefix: str) -> str:
    text = str(call_to_action or "")
    if not text.startswith(prefix):
        raise ValueError(f"Expected call_to_action prefix `{prefix}`, got `{text}`.")
    return text[len(prefix) :].strip()


def _parse_joint_actions(raw_event: str) -> Dict[str, str]:
    actions: Dict[str, str] = {}
    for line in str(raw_event or "").splitlines():
        text = line.strip()
        if not text:
            continue
        if ":" not in text:
            continue
        actor, payload = text.split(":", 1)
        actions[actor.strip()] = payload.strip()
    return actions


def _parse_numeric_text(raw_value: Any) -> tuple[Optional[int], bool]:
    if raw_value is None:
        return None, False
    if isinstance(raw_value, (int, float)):
        if is_nan(raw_value):
            return None, False
        return int(round(float(raw_value))), True
    match = _NUMBER_RE.search(str(raw_value))
    if not match:
        return None, False
    try:
        return int(round(float(match.group(0)))), True
    except Exception:
        return None, False


def _match_choice_value(raw_value: Any, options: Sequence[str]) -> tuple[str, bool]:
    text = str(raw_value or "").strip()
    for option in options:
        if text == str(option):
            return str(option), True
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(options):
            return str(options[idx]), True
        if 1 <= idx <= len(options):
            return str(options[idx - 1]), True
    lowered = text.lower()
    for option in options:
        option_text = str(option).strip().lower()
        if lowered == option_text or (option_text and option_text in lowered):
            return str(option), True
    return str(options[0]), False


def _build_stage_plan(env: Mapping[str, Any]) -> List[StageSpec]:
    num_rounds = int(env.get("CONFIG_numRounds", 0) or 0)
    chat_on = as_bool(env.get("CONFIG_chat", False))
    actions_on = as_bool(env.get("CONFIG_rewardExists", False)) or as_bool(
        env.get("CONFIG_punishmentExists", False)
    )
    stages: List[StageSpec] = []
    for round_index in range(1, num_rounds + 1):
        if chat_on:
            stages.append(StageSpec(round_index=round_index, name="chat"))
        stages.append(StageSpec(round_index=round_index, name="contribution"))
        if actions_on:
            stages.append(StageSpec(round_index=round_index, name="actions"))
    return stages


class PublicGoodsGameMaster(entity_lib.EntityWithLogging):
    def __init__(
        self,
        *,
        name: str,
        ctx: GameContext,
        args: Any,
        entities: Sequence[entity_lib.Entity],
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
        assigned_archetypes: Mapping[str, Dict[str, Any]],
    ):
        self._name = str(name)
        self._ctx = ctx
        self._args = args
        self.entities = list(entities)
        self._memory_bank = memory_bank
        self._assigned_archetypes = dict(assigned_archetypes)
        self._player_names = [ctx.avatar_by_player[pid] for pid in ctx.player_ids]
        self._entity_by_name = {entity.name: entity for entity in self.entities}
        self._pid_by_avatar = {ctx.avatar_by_player[pid]: pid for pid in ctx.player_ids}
        self._stage_plan = _build_stage_plan(ctx.env)
        self._stage_cursor = 0
        self._pending_joint_event = ""
        self._last_log: Dict[str, Any] = {}
        self._transcripts, self._archetype_ids = _build_initial_macro_transcripts(
            ctx,
            assigned_archetypes=assigned_archetypes,
        )
        self._round_data: Dict[int, Dict[str, Any]] = {}
        self._rows: List[Dict[str, Any]] = []
        self._binary_targets = _use_binary_targets(
            str(getattr(args, "action_prompt_mode", "binary_targets") or "binary_targets")
        )
        self._gate_enabled = bool(getattr(args, "action_continuation_gate", True))
        self._punish_keep_prob = _clamp_probability(
            getattr(args, "punish_continuation_keep_prob", 0.5)
        )
        self._reward_keep_prob = _clamp_probability(
            getattr(args, "reward_continuation_keep_prob", 0.35)
        )
        self._previous_round_rewards = {avatar: {} for avatar in self._player_names}
        self._previous_round_punish = {avatar: {} for avatar in self._player_names}
        self._seed_players()

    @property
    def name(self) -> str:
        return self._name

    def get_last_log(self) -> dict[str, Any]:
        return {"__pgg_game_master__": dict(self._last_log)}

    def get_rows(self) -> List[Dict[str, Any]]:
        return list(self._rows)

    def get_transcripts(self) -> Dict[str, List[str]]:
        return {pid: list(lines) for pid, lines in self._transcripts.items()}

    def observe(self, observation: str) -> None:
        text = str(observation or "").strip()
        if not text:
            return
        self._memory_bank.add(text)
        self._last_log = {
            "phase": "observe",
            "stage_index": self._stage_cursor,
            "observation": text,
        }
        if text.startswith(f"{simultaneous_engine.PUTATIVE_EVENT_TAG} "):
            self._pending_joint_event = text[len(simultaneous_engine.PUTATIVE_EVENT_TAG) + 1 :].strip()

    def act(self, action_spec: entity_lib.ActionSpec = entity_lib.DEFAULT_ACTION_SPEC) -> str:
        if action_spec.output_type == entity_lib.OutputType.MAKE_OBSERVATION:
            avatar = _extract_name(action_spec.call_to_action, "PGG_MAKE_OBSERVATION::")
            observation = self._make_observation_for_avatar(avatar)
            self._last_log = {
                "phase": "make_observation",
                "stage": self._current_stage_name(),
                "avatar": avatar,
                "observation": observation,
            }
            return observation
        if action_spec.output_type == entity_lib.OutputType.NEXT_ACTING:
            joined = ",".join(self._player_names)
            self._last_log = {
                "phase": "next_acting",
                "stage": self._current_stage_name(),
                "players": list(self._player_names),
            }
            return joined
        if action_spec.output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
            avatar = _extract_name(action_spec.call_to_action, "PGG_NEXT_ACTION_SPEC::")
            next_spec = self._current_action_spec(avatar)
            payload = concordia_engine.action_spec_to_string(next_spec)
            self._last_log = {
                "phase": "next_action_spec",
                "stage": self._current_stage_name(),
                "avatar": avatar,
                "action_spec": next_spec.to_dict(),
            }
            return payload
        if action_spec.output_type == entity_lib.OutputType.RESOLVE:
            result = self._resolve_stage()
            self._last_log = {
                "phase": "resolve",
                "stage": self._current_stage_name(previous=True),
                "result": result,
            }
            return result
        if action_spec.output_type == entity_lib.OutputType.TERMINATE:
            done = self._stage_cursor >= len(self._stage_plan)
            self._last_log = {
                "phase": "terminate",
                "stage_index": self._stage_cursor,
                "done": done,
            }
            return entity_lib.BINARY_OPTIONS["affirmative" if done else "negative"]
        raise NotImplementedError(f"Unsupported game-master output type: {action_spec.output_type}")

    def _seed_players(self) -> None:
        include_demographics = bool(getattr(self._args, "include_demographics", False))
        include_reasoning = bool(getattr(self._args, "include_reasoning", False))
        for pid in self._ctx.player_ids:
            avatar = self._ctx.avatar_by_player[pid]
            entity = self._entity_by_name[avatar]
            system_text = system_header_plain(
                self._ctx.env,
                self._ctx.demographics_by_player.get(pid, "") if include_demographics else "",
                include_reasoning,
            )
            body = "\n".join(self._transcripts[pid])
            entity.observe(f"{system_text}\n{body}")

    def _current_stage(self) -> StageSpec:
        if self._stage_cursor >= len(self._stage_plan):
            return self._stage_plan[-1]
        return self._stage_plan[self._stage_cursor]

    def _current_stage_name(self, previous: bool = False) -> str:
        if previous:
            if self._stage_cursor == 0:
                return "initial"
            return self._stage_plan[self._stage_cursor - 1].name
        if self._stage_cursor >= len(self._stage_plan):
            return "complete"
        return self._stage_plan[self._stage_cursor].name

    def _ensure_round_data(self, round_index: int) -> Dict[str, Any]:
        if round_index not in self._round_data:
            self._round_data[round_index] = {
                "chat_messages": {avatar: "" for avatar in self._player_names},
                "chat_parsed": {avatar: True for avatar in self._player_names},
                "chat_reasoning": {avatar: None for avatar in self._player_names},
                "contrib_math": {avatar: 0 for avatar in self._player_names},
                "contrib_rec": {avatar: None for avatar in self._player_names},
                "contrib_parsed": {avatar: False for avatar in self._player_names},
                "contrib_reasoning": {avatar: None for avatar in self._player_names},
                "rewards_given": {avatar: {} for avatar in self._player_names},
                "punish_given": {avatar: {} for avatar in self._player_names},
                "actions_parsed": {avatar: None for avatar in self._player_names},
                "actions_reasoning": {avatar: None for avatar in self._player_names},
            }
        return self._round_data[round_index]

    def _make_observation_for_avatar(self, avatar: str) -> str:
        pid = self._pid_by_avatar[avatar]
        stage = self._current_stage()
        parts: List[str] = []
        is_first_stage = self._stage_cursor == 0 or self._stage_plan[self._stage_cursor - 1].round_index != stage.round_index
        if is_first_stage:
            line = round_open(self._ctx.env, stage.round_index)
            self._transcripts[pid].append(line)
            parts.append(line)
        return "\n".join(part for part in parts if str(part).strip())

    def _current_action_spec(self, avatar: str) -> entity_lib.ActionSpec:
        stage = self._current_stage()
        if stage.name == "chat":
            return entity_lib.free_action_spec(
                call_to_action=(
                    f"You are {avatar}. This is the optional group chat at the start of the round. "
                    "Reply with one short message to the group, or reply exactly SILENT."
                ),
                tag="speech",
            )
        if stage.name == "contribution":
            endow = int(self._ctx.env.get("CONFIG_endowment", 0) or 0)
            if as_bool(self._ctx.env.get("CONFIG_allOrNothing", False)):
                return entity_lib.choice_action_spec(
                    call_to_action=(
                        f"{round_info_line(self._ctx.env)} Reply with exactly one of: 0 or {endow}."
                    ),
                    options=(str(0), str(endow)),
                    tag="action",
                )
            return entity_lib.float_action_spec(
                call_to_action=f"{round_info_line(self._ctx.env)} Reply with the number only.",
                tag="action",
            )
        if stage.name == "actions":
            peer_order = [name for name in self._player_names if name != avatar]
            peers_csv = ", ".join(peer_order)
            tag = actions_tag(self._ctx.env) or "PUNISHMENT/REWARD"
            action_prompt_mode = str(
                getattr(self._args, "action_prompt_mode", "binary_targets") or "binary_targets"
            )
            if self._binary_targets:
                if tag == "PUNISHMENT":
                    fmt = '{"stage":"actions","punish":["AVATAR", ...]}'
                elif tag == "REWARD":
                    fmt = '{"stage":"actions","reward":["AVATAR", ...]}'
                else:
                    fmt = '{"stage":"actions","punish":["AVATAR", ...],"reward":["AVATAR", ...]}'
            else:
                if tag == "PUNISHMENT":
                    fmt = '{"stage":"actions","actions":{"AVATAR":1,...}}'
                elif tag == "REWARD":
                    fmt = '{"stage":"actions","actions":{"AVATAR":1,...}}'
                else:
                    fmt = '{"stage":"actions","actions":{"AVATAR":-1|1,...}}'
            prompt = mech_info(self._ctx.env, action_prompt_mode) or "Choose punishments and rewards."
            return entity_lib.free_action_spec(
                call_to_action=(
                    f"You are {avatar}. {prompt}\n"
                    f"Available target avatars: {peers_csv}.\n"
                    "Reply with JSON only in this format:\n"
                    f"{fmt}"
                ),
                tag="action",
            )
        raise NotImplementedError(f"Unsupported stage: {stage.name}")

    def _capture_entity_logs(self) -> Dict[str, Optional[str]]:
        return {
            avatar: _serialize_log(self._entity_by_name[avatar].get_last_log())
            for avatar in self._player_names
        }

    def _resolve_stage(self) -> str:
        stage = self._current_stage()
        if not self._pending_joint_event:
            joint_actions = {}
        else:
            joint_actions = _parse_joint_actions(self._pending_joint_event)
        self._pending_joint_event = ""
        if stage.name == "chat":
            return self._resolve_chat(stage.round_index, joint_actions)
        if stage.name == "contribution":
            return self._resolve_contribution(stage.round_index, joint_actions)
        if stage.name == "actions":
            return self._resolve_actions(stage.round_index, joint_actions)
        raise NotImplementedError(f"Unsupported stage: {stage.name}")

    def _resolve_chat(self, round_index: int, joint_actions: Mapping[str, str]) -> str:
        round_data = self._ensure_round_data(round_index)
        reasoning = self._capture_entity_logs()
        chat_messages: Dict[str, str] = {}
        for avatar in self._player_names:
            raw_text = str(joint_actions.get(avatar, "") or "").strip()
            if raw_text.upper() in {"", "SILENT", "NONE", "NO MESSAGE"}:
                raw_text = ""
            chat_messages[avatar] = raw_text
            round_data["chat_messages"][avatar] = raw_text
            round_data["chat_parsed"][avatar] = True
            round_data["chat_reasoning"][avatar] = (
                reasoning.get(avatar) if bool(getattr(self._args, "include_reasoning", False)) else None
            )
        _append_chat_history_line(
            self._transcripts,
            self._player_names,
            {pid: self._ctx.avatar_by_player[pid] for pid in self._ctx.player_ids},
            chat_messages,
        )
        for pid in self._ctx.player_ids:
            avatar = self._ctx.avatar_by_player[pid]
            self._entity_by_name[avatar].observe(self._transcripts[pid][-1])
        self._stage_cursor += 1
        return f"Resolved round {round_index} chat."

    def _resolve_contribution(self, round_index: int, joint_actions: Mapping[str, str]) -> str:
        round_data = self._ensure_round_data(round_index)
        reasoning = self._capture_entity_logs()
        endow = int(self._ctx.env.get("CONFIG_endowment", 0) or 0)
        all_or_nothing = as_bool(self._ctx.env.get("CONFIG_allOrNothing", False))
        for pid in self._ctx.player_ids:
            avatar = self._ctx.avatar_by_player[pid]
            raw_value = joint_actions.get(avatar, "")
            parsed_ok = False
            if all_or_nothing:
                matched, parsed_ok = _match_choice_value(raw_value, (str(0), str(endow)))
                value = int(matched)
                value = endow if value >= (endow // 2) else 0
            else:
                parsed_value, parsed_ok = _parse_numeric_text(raw_value)
                value = 0 if parsed_value is None else max(0, min(endow, int(parsed_value)))
            round_data["contrib_math"][avatar] = int(value)
            round_data["contrib_rec"][avatar] = int(value) if parsed_ok else None
            round_data["contrib_parsed"][avatar] = parsed_ok
            round_data["contrib_reasoning"][avatar] = (
                reasoning.get(avatar) if bool(getattr(self._args, "include_reasoning", False)) else None
            )
            self._transcripts[pid].append(f'<CONTRIB v="{int(value)}"/>')

        total_contrib = int(sum(round_data["contrib_math"].values()))
        multiplied = float(self._ctx.env.get("CONFIG_multiplier", 0) or 0) * float(total_contrib)
        for pid in self._ctx.player_ids:
            avatar = self._ctx.avatar_by_player[pid]
            self._transcripts[pid].append(redist_line(total_contrib, multiplied, len(self._player_names)))
            peers_csv, _ = peers_contributions_csv(
                self._player_names,
                avatar,
                round_data["contrib_math"],
            )
            self._transcripts[pid].append(f"<PEERS_CONTRIBUTIONS> {peers_csv} </PEERS_CONTRIBUTIONS>")
            self._entity_by_name[avatar].observe(
                "\n".join(self._transcripts[pid][-3:])
            )

        self._stage_cursor += 1
        if not (as_bool(self._ctx.env.get("CONFIG_rewardExists", False)) or as_bool(self._ctx.env.get("CONFIG_punishmentExists", False))):
            self._finalize_round(round_index)
        return f"Resolved round {round_index} contributions."

    def _resolve_actions(self, round_index: int, joint_actions: Mapping[str, str]) -> str:
        round_data = self._ensure_round_data(round_index)
        reasoning = self._capture_entity_logs()
        tag = actions_tag(self._ctx.env) or "PUNISHMENT/REWARD"
        for pid in self._ctx.player_ids:
            avatar = self._ctx.avatar_by_player[pid]
            peer_order = [name for name in self._player_names if name != avatar]
            payload, ok = parse_json_response(str(joint_actions.get(avatar, "") or ""))
            punish_out: Dict[str, int] = {}
            reward_out: Dict[str, int] = {}
            parsed_ok = False
            if ok and isinstance(payload, dict) and payload.get("stage") == "actions":
                parsed_ok = True
                if self._binary_targets:
                    punish_out, reward_out = _parse_binary_action_response(payload, tag, peer_order)
                else:
                    raw_actions = payload.get("actions")
                    if isinstance(raw_actions, dict):
                        for peer in peer_order:
                            units = _safe_int(raw_actions.get(peer, 0))
                            if as_bool(self._ctx.env.get("CONFIG_rewardExists", False)) and not as_bool(
                                self._ctx.env.get("CONFIG_punishmentExists", False)
                            ):
                                if units > 0:
                                    reward_out[peer] = int(units)
                            elif as_bool(self._ctx.env.get("CONFIG_punishmentExists", False)) and not as_bool(
                                self._ctx.env.get("CONFIG_rewardExists", False)
                            ):
                                if units > 0:
                                    punish_out[peer] = int(units)
                            else:
                                if units < 0:
                                    punish_out[peer] = int(abs(units))
                                elif units > 0:
                                    reward_out[peer] = int(units)
            if self._gate_enabled:
                punish_out = _apply_continuation_gate(
                    punish_out,
                    self._previous_round_punish.get(avatar, {}),
                    self._punish_keep_prob,
                    __import__("random").Random(f"{self._ctx.game_id}|{round_index}|{avatar}|punish"),
                )
                reward_out = _apply_continuation_gate(
                    reward_out,
                    self._previous_round_rewards.get(avatar, {}),
                    self._reward_keep_prob,
                    __import__("random").Random(f"{self._ctx.game_id}|{round_index}|{avatar}|reward"),
                )
            round_data["punish_given"][avatar] = {
                target: int(units) for target, units in punish_out.items() if int(units) > 0
            }
            round_data["rewards_given"][avatar] = {
                target: int(units) for target, units in reward_out.items() if int(units) > 0
            }
            round_data["actions_parsed"][avatar] = parsed_ok
            round_data["actions_reasoning"][avatar] = (
                reasoning.get(avatar) if bool(getattr(self._args, "include_reasoning", False)) else None
            )

            if as_bool(self._ctx.env.get("CONFIG_rewardExists", False)) and not as_bool(
                self._ctx.env.get("CONFIG_punishmentExists", False)
            ):
                line = f"<REWARD>{_json_compact(round_data['rewards_given'][avatar])}</REWARD>"
            elif as_bool(self._ctx.env.get("CONFIG_punishmentExists", False)) and not as_bool(
                self._ctx.env.get("CONFIG_rewardExists", False)
            ):
                line = f"<PUNISHMENT>{_json_compact(round_data['punish_given'][avatar])}</PUNISHMENT>"
            else:
                line = (
                    f'<ACTIONS punish="{_json_compact(round_data["punish_given"][avatar])}" '
                    f'reward="{_json_compact(round_data["rewards_given"][avatar])}"/>'
                )
            self._transcripts[pid].append(line)
            self._entity_by_name[avatar].observe(line)

        self._previous_round_rewards = {
            avatar: dict(targets) for avatar, targets in round_data["rewards_given"].items()
        }
        self._previous_round_punish = {
            avatar: dict(targets) for avatar, targets in round_data["punish_given"].items()
        }
        self._stage_cursor += 1
        self._finalize_round(round_index)
        return f"Resolved round {round_index} punishment/reward stage."

    def _finalize_round(self, round_index: int) -> None:
        round_data = self._ensure_round_data(round_index)
        reward_on = as_bool(self._ctx.env.get("CONFIG_rewardExists", False))
        punish_on = as_bool(self._ctx.env.get("CONFIG_punishmentExists", False))
        show_punish_id = as_bool(self._ctx.env.get("CONFIG_showPunishmentId", False))
        show_reward_id = as_bool(self._ctx.env.get("CONFIG_showRewardId", False))
        show_other = as_bool(self._ctx.env.get("CONFIG_showOtherSummaries", False))
        endow = int(self._ctx.env.get("CONFIG_endowment", 0) or 0)

        inbound_reward_units = {avatar: 0 for avatar in self._player_names}
        inbound_punish_units = {avatar: 0 for avatar in self._player_names}
        for source in self._player_names:
            for target, units in round_data["rewards_given"][source].items():
                inbound_reward_units[target] += int(units)
            for target, units in round_data["punish_given"][source].items():
                inbound_punish_units[target] += int(units)

        total_contrib = int(sum(round_data["contrib_math"].values()))
        multiplied = float(self._ctx.env.get("CONFIG_multiplier", 0) or 0) * float(total_contrib)
        share = (float(multiplied) / len(self._player_names)) if self._player_names else 0.0

        for pid in self._ctx.player_ids:
            avatar = self._ctx.avatar_by_player[pid]
            summary_lines_start = len(self._transcripts[pid])
            if show_punish_id and punish_on:
                punishers = {
                    source: units
                    for source in self._player_names
                    for target, units in round_data["punish_given"][source].items()
                    if target == avatar and int(units) > 0
                }
                self._transcripts[pid].append(f"<PUNISHED_BY>{_json_compact(punishers)}</PUNISHED_BY>")
            if show_reward_id and reward_on:
                rewarders = {
                    source: units
                    for source in self._player_names
                    for target, units in round_data["rewards_given"][source].items()
                    if target == avatar and int(units) > 0
                }
                self._transcripts[pid].append(f"<REWARDED_BY>{_json_compact(rewarders)}</REWARDED_BY>")

            spent_pun_units = sum(round_data["punish_given"][avatar].values()) if punish_on else 0
            spent_reward_units = sum(round_data["rewards_given"][avatar].values()) if reward_on else 0
            you: Dict[str, Any] = {}
            if punish_on:
                you["coins_spent_on_punish"] = spent_pun_units * int(
                    self._ctx.env.get("CONFIG_punishmentCost", 0) or 0
                )
                you["coins_deducted_from_you"] = inbound_punish_units[avatar] * int(
                    self._ctx.env.get("CONFIG_punishmentMagnitude", 0) or 0
                )
            if reward_on:
                you["coins_spent_on_reward"] = spent_reward_units * int(
                    self._ctx.env.get("CONFIG_rewardCost", 0) or 0
                )
                you["coins_rewarded_to_you"] = inbound_reward_units[avatar] * int(
                    self._ctx.env.get("CONFIG_rewardMagnitude", 0) or 0
                )
            private_kept = endow - int(round_data["contrib_math"].get(avatar, 0))
            payoff = (
                private_kept
                + share
                - you.get("coins_spent_on_punish", 0)
                - you.get("coins_spent_on_reward", 0)
                - you.get("coins_deducted_from_you", 0)
                + you.get("coins_rewarded_to_you", 0)
            )
            you["payoff"] = int(payoff)

            summary = {f"{avatar} (YOU)": you}
            if show_other:
                for other in self._player_names:
                    if other == avatar:
                        continue
                    other_summary: Dict[str, Any] = {}
                    if punish_on:
                        other_summary["coins_spent_on_punish"] = sum(
                            round_data["punish_given"][other].values()
                        ) * int(self._ctx.env.get("CONFIG_punishmentCost", 0) or 0)
                        other_summary["coins_deducted_from_them"] = (
                            sum(round_data["punish_given"][source].get(other, 0) for source in self._player_names)
                            * int(self._ctx.env.get("CONFIG_punishmentMagnitude", 0) or 0)
                        )
                    if reward_on:
                        other_summary["coins_spent_on_reward"] = sum(
                            round_data["rewards_given"][other].values()
                        ) * int(self._ctx.env.get("CONFIG_rewardCost", 0) or 0)
                        other_summary["coins_rewarded_to_them"] = (
                            sum(round_data["rewards_given"][source].get(other, 0) for source in self._player_names)
                            * int(self._ctx.env.get("CONFIG_rewardMagnitude", 0) or 0)
                        )
                    private_kept_other = endow - int(round_data["contrib_math"].get(other, 0))
                    payoff_other = (
                        private_kept_other
                        + share
                        - other_summary.get("coins_spent_on_punish", 0)
                        - other_summary.get("coins_spent_on_reward", 0)
                        - other_summary.get("coins_deducted_from_them", 0)
                        + other_summary.get("coins_rewarded_to_them", 0)
                    )
                    other_summary["payoff"] = int(payoff_other)
                    summary[other] = other_summary

            self._transcripts[pid].append(f"<ROUND_SUMMARY>{_json_compact(summary)}</ROUND_SUMMARY>")
            self._transcripts[pid].append("</ROUND>")
            observed_summary = "\n".join(self._transcripts[pid][summary_lines_start:-1])
            if observed_summary.strip():
                self._entity_by_name[avatar].observe(observed_summary)

            archetype_record = dict(self._assigned_archetypes.get(pid) or {})
            self._rows.append(
                {
                    "gameId": self._ctx.game_id,
                    "gameName": self._ctx.game_name,
                    "roundIndex": round_index,
                    "playerId": pid,
                    "playerAvatar": avatar,
                    "archetype": self._archetype_ids.get(pid),
                    "persona": self._archetype_ids.get(pid),
                    "archetype_mode": _resolve_archetype_mode(self._args) or "",
                    "archetype_source_gameId": archetype_record.get("experiment"),
                    "archetype_source_playerId": archetype_record.get("participant"),
                    "archetype_source_rank": archetype_record.get("source_rank"),
                    "archetype_source_score": archetype_record.get("source_score"),
                    "archetype_source_weight": archetype_record.get("source_weight"),
                    "demographics": self._ctx.demographics_by_player.get(pid, ""),
                    "data.chat_message": round_data["chat_messages"].get(avatar, ""),
                    "data.chat_parsed": round_data["chat_parsed"].get(avatar),
                    "data.chat_reasoning": round_data["chat_reasoning"].get(avatar),
                    "data.contribution": round_data["contrib_rec"].get(avatar),
                    "data.contribution_clamped": round_data["contrib_math"].get(avatar),
                    "data.contribution_parsed": round_data["contrib_parsed"].get(avatar),
                    "data.contribution_reasoning": round_data["contrib_reasoning"].get(avatar),
                    "data.punished": _json_compact(round_data["punish_given"][avatar]) if punish_on else None,
                    "data.rewarded": _json_compact(round_data["rewards_given"][avatar]) if reward_on else None,
                    "data.actions_parsed": round_data["actions_parsed"].get(avatar),
                    "data.actions_reasoning": round_data["actions_reasoning"].get(avatar),
                }
            )


@dataclass
class PublicGoodsGameMasterPrefab(prefab_lib.Prefab):
    description: str = "A macro PGG game master built for Concordia's simultaneous engine."
    params: Mapping[str, Any] = None
    entities: Sequence[entity_lib.Entity] | None = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        if self.entities is None:
            self.entities = ()

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> PublicGoodsGameMaster:
        del model
        return PublicGoodsGameMaster(
            name=str(self.params.get("name", "macro_pgg_game_master")),
            ctx=self.params["ctx"],
            args=self.params["args"],
            entities=self.entities or (),
            memory_bank=memory_bank,
            assigned_archetypes=self.params.get("assigned_archetypes", {}),
        )
