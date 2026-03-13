from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import re
import time
import ast
import csv
import random
from threading import Lock
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from Simulation_robin.llm_client import LLMClient
from Persona.archetype_sampling.runtime import (
    ArchetypeSummaryPool,
    PrecomputedAssignmentIndex,
    SoftBankSummarySampler,
    SUPPORTED_ARCHETYPE_MODES,
    assign_archetypes_for_game,
    canonicalize_archetype_mode,
    load_finished_summary_pool,
    load_precomputed_assignment_index,
)

try:
    from .debug import build_debug_record, build_full_debug_record
    from .model_loader import load_model
    from .parsers import parse_json_response
    from .prompt_builder import (
        JSON_STOP_SENTINEL,
        actions_format_line,
        actions_tag,
        build_openai_messages,
        contrib_format_line,
        format_contrib_answer,
        max_tokens_reminder_line,
        mech_info,
        peers_contributions_csv,
        redist_line,
        round_info_line,
        round_open,
        system_header_plain,
    )
    from .utils import (
        as_bool,
        demographics_line,
        format_num,
        is_nan,
        json_compact,
        log,
        make_unique_avatar_map,
        normalize_avatar,
        parse_dict,
        relocate_output,
        timestamp_yymmddhhmm,
    )
except ImportError:
    from debug import build_debug_record, build_full_debug_record
    from model_loader import load_model
    from parsers import parse_json_response
    from prompt_builder import (
        JSON_STOP_SENTINEL,
        actions_format_line,
        actions_tag,
        build_openai_messages,
        contrib_format_line,
        format_contrib_answer,
        max_tokens_reminder_line,
        mech_info,
        peers_contributions_csv,
        redist_line,
        round_info_line,
        round_open,
        system_header_plain,
    )
    from utils import (
        as_bool,
        demographics_line,
        format_num,
        is_nan,
        json_compact,
        log,
        make_unique_avatar_map,
        normalize_avatar,
        parse_dict,
        relocate_output,
        timestamp_yymmddhhmm,
    )


CONFIG_KEYS = {
    "CONFIG_allOrNothing",
    "CONFIG_chat",
    "CONFIG_defaultContribProp",
    "CONFIG_endowment",
    "CONFIG_multiplier",
    "CONFIG_numRounds",
    "CONFIG_playerCount",
    "CONFIG_punishmentCost",
    "CONFIG_punishmentExists",
    "CONFIG_punishmentMagnitude",
    "CONFIG_rewardCost",
    "CONFIG_rewardExists",
    "CONFIG_rewardMagnitude",
    "CONFIG_showNRounds",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
}


BOOL_CONFIG_KEYS = {
    "CONFIG_allOrNothing",
    "CONFIG_chat",
    "CONFIG_punishmentExists",
    "CONFIG_rewardExists",
    "CONFIG_showNRounds",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
}


@dataclass
class GameContext:
    game_id: str
    game_name: str
    env: Dict[str, Any]
    player_ids: List[str]
    avatar_by_player: Dict[str, str]
    player_by_avatar: Dict[str, str]
    rounds: List[int]
    round_to_rows: Dict[int, pd.DataFrame]
    chats_by_round_phase: Dict[int, Dict[str, List[Tuple[str, str]]]]
    demographics_by_player: Dict[str, str]


SUMMARY_ARCHETYPE_INTRO = (
    "Below is an archetype summary of how you played a different PGG in the past. "
    "Be aware of this archetype as you make decisions. "
    "Recall that you're probably playing games with different people from the past, and "
    "that the exact rules of this game could differ from the ones you've played before."
)

def _parse_chat_messages(msg_str: Any) -> List[Dict[str, Any]]:
    if not isinstance(msg_str, str):
        return []
    s = msg_str.strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass
    out: List[Dict[str, Any]] = []
    first_brace = s.find("{")
    last_brace = s.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
        return out
    i = first_brace
    while i <= last_brace:
        while i <= last_brace and s[i] != "{":
            i += 1
        if i > last_brace:
            break
        start = i
        depth = 0
        while i <= last_brace:
            c = s[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    chunk = s[start : i + 1]
                    parsed = None
                    try:
                        parsed = ast.literal_eval(chunk)
                    except Exception:
                        try:
                            parsed = json.loads(chunk)
                        except Exception:
                            parsed = None
                    if isinstance(parsed, dict):
                        out.append(parsed)
                    i += 1
                    break
            i += 1
        else:
            break
    return out


def _extract_round_phase(game_phase: Any) -> Tuple[Optional[int], Optional[str]]:
    if not isinstance(game_phase, str):
        return None, None
    lower = game_phase.lower()
    m = re.search(r"round\s+(\d+)", lower)
    round_id = int(m.group(1)) if m else None
    if "contrib" in lower:
        return round_id, "contribution"
    if "outcome" in lower:
        return round_id, "outcome"
    if "summary" in lower:
        return round_id, "summary"
    return round_id, None


def _index_chats_for_game(messages_raw: Any) -> Dict[int, Dict[str, List[Tuple[str, str]]]]:
    chats_by_round_phase: Dict[int, Dict[str, List[Tuple[str, str]]]] = {}
    msgs = _parse_chat_messages(messages_raw)
    parsed: List[Tuple[Optional[int], Optional[str], str, str]] = []
    saw_round_zero = False
    for msg in msgs:
        text = str(msg.get("text", "")).strip()
        avatar = normalize_avatar(msg.get("avatar"))
        r_raw, phase = _extract_round_phase(msg.get("gamePhase"))
        if r_raw == 0:
            saw_round_zero = True
        parsed.append((r_raw, phase, avatar, text))
    shift = 1 if saw_round_zero else 0
    last_round = None
    for r_raw, phase, avatar, text in parsed:
        if not text:
            continue
        if r_raw is not None:
            round_idx = r_raw + shift
            last_round = round_idx
        else:
            if last_round is None:
                continue
            round_idx = last_round
        bucket = phase or "outcome"
        chats_by_round_phase.setdefault(round_idx, {}).setdefault(bucket, []).append((avatar, text))
    return chats_by_round_phase


def _build_round_index(df_rounds: pd.DataFrame) -> pd.DataFrame:
    df = df_rounds.copy()
    df["gameId"] = df["gameId"].astype(str)
    df["roundId"] = df["roundId"].astype(str)
    df["playerId"] = df["playerId"].astype(str)
    df["__row_order"] = range(len(df))
    if "createdAt" in df.columns:
        df["__created_at"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)
    else:
        df["__created_at"] = pd.NaT
    round_order = (
        df.groupby(["gameId", "roundId"], as_index=False)
        .agg(__created_at=("__created_at", "min"), __row_order=("__row_order", "min"))
        .sort_values(["gameId", "__created_at", "__row_order", "roundId"], na_position="last")
    )
    round_order["roundIndex"] = round_order.groupby("gameId").cumcount() + 1
    df = df.merge(round_order[["gameId", "roundId", "roundIndex"]], on=["gameId", "roundId"], how="left")
    df["roundIndex"] = pd.to_numeric(df["roundIndex"], errors="coerce").fillna(0).astype(int)
    return df


def _build_env_lookup(df_analysis: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if "gameId" not in df_analysis.columns:
        return out
    analysis = df_analysis.copy()
    analysis["gameId"] = analysis["gameId"].astype(str)
    analysis = analysis.drop_duplicates(subset=["gameId"], keep="first")
    for _, row in analysis.iterrows():
        gid = str(row.get("gameId"))
        env = {"gameId": gid, "name": row.get("name", gid)}
        for key in CONFIG_KEYS:
            if key in row.index:
                val = row.get(key)
                if key in BOOL_CONFIG_KEYS:
                    val = as_bool(val)
                env[key] = val
        out[gid] = env
    return out


def _build_avatar_lookup(df_players: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if "_id" not in df_players.columns:
        return out
    for _, row in df_players.iterrows():
        pid = str(row.get("_id"))
        avatar = normalize_avatar(row.get("data.avatar"))
        if pid:
            out[pid] = avatar
    return out


def _build_demographics_lookup(
    df_demographics: pd.DataFrame,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    by_game_player: Dict[Tuple[str, str], Dict[str, Any]] = {}
    by_player: Dict[str, Dict[str, Any]] = {}
    if "playerId" not in df_demographics.columns:
        return by_game_player, by_player
    for _, row in df_demographics.iterrows():
        pid = str(row.get("playerId"))
        rec = row.to_dict()
        by_player[pid] = rec
        gid = row.get("gameId")
        if gid is not None and not is_nan(gid):
            by_game_player[(str(gid), pid)] = rec
    return by_game_player, by_player


def build_game_contexts(
    df_rounds: pd.DataFrame,
    df_analysis: pd.DataFrame,
    df_demographics: pd.DataFrame,
    df_players: pd.DataFrame,
    df_games: pd.DataFrame,
) -> List[GameContext]:
    rounds = _build_round_index(df_rounds)
    env_lookup = _build_env_lookup(df_analysis)
    avatar_lookup = _build_avatar_lookup(df_players)
    demo_by_game_player, demo_by_player = _build_demographics_lookup(df_demographics)
    game_messages: Dict[str, Any] = {}
    if "_id" in df_games.columns and "data.messages" in df_games.columns:
        for _, row in df_games.iterrows():
            game_messages[str(row.get("_id"))] = row.get("data.messages", "")

    contexts: List[GameContext] = []
    for game_id, gdf in rounds.groupby("gameId", sort=True):
        env = env_lookup.get(str(game_id))
        if not env:
            log(f"[warn] skipping gameId={game_id}; not found in analysis file")
            continue
        round_ids = sorted(int(x) for x in gdf["roundIndex"].dropna().unique().tolist() if int(x) > 0)
        if not round_ids:
            continue
        gsorted = gdf.sort_values(["roundIndex", "playerId"])
        player_ids: List[str] = list(dict.fromkeys(gsorted["playerId"].astype(str).tolist()))
        avatar_by_player = make_unique_avatar_map(player_ids, avatar_lookup)
        player_by_avatar = {av: pid for pid, av in avatar_by_player.items()}
        demo_lines: Dict[str, str] = {}
        for pid in player_ids:
            demo_row = demo_by_game_player.get((str(game_id), pid))
            if demo_row is None:
                demo_row = demo_by_player.get(pid)
            demo_lines[pid] = demographics_line(demo_row)
        round_to_rows = {
            rid: gdf[gdf["roundIndex"] == rid].copy().sort_values(["playerId"]).reset_index(drop=True) for rid in round_ids
        }
        chats = _index_chats_for_game(game_messages.get(str(game_id), "")) if as_bool(env.get("CONFIG_chat")) else {}
        contexts.append(
            GameContext(
                game_id=str(game_id),
                game_name=str(env.get("name", game_id)),
                env=env,
                player_ids=player_ids,
                avatar_by_player=avatar_by_player,
                player_by_avatar=player_by_avatar,
                rounds=round_ids,
                round_to_rows=round_to_rows,
                chats_by_round_phase=chats,
                demographics_by_player=demo_lines,
            )
        )
    return contexts


def _round_info_tag(env: Mapping[str, Any]) -> str:
    endow = int(env.get("CONFIG_endowment", 0) or 0)
    aon = as_bool(env.get("CONFIG_allOrNothing", False))
    contrib_mode = f"either 0 or {endow}" if aon else f"integer from 0 to {endow}"
    if float(env.get("CONFIG_defaultContribProp", 0.0) or 0.0) > 0.0:
        text = (
            f"{endow} coins are currently in the public fund, and you will contribute the remainder of the coins "
            f"you choose to take for yourself. Choose the amount to contribute ({contrib_mode})."
        )
    else:
        text = f"{endow} coins are currently in your private pocket. Choose the amount to contribute ({contrib_mode})."
    return f"<ROUND_INFO> {text} (multiplier: {env.get('CONFIG_multiplier', 'Unknown')}x). </ROUND_INFO>"


def _compact_outcomes(round_slice: pd.DataFrame, ctx: GameContext, focal_pid: str) -> str:
    row_by_pid = {str(r["playerId"]): r for _, r in round_slice.iterrows()}
    tokens: List[str] = []
    for pid in ctx.player_ids:
        if pid == focal_pid:
            continue
        row = row_by_pid.get(pid)
        contrib = row.get("data.contribution") if row is not None else None
        val = "NA" if contrib is None or is_nan(contrib) else format_num(contrib)
        tokens.append(f"{ctx.avatar_by_player.get(pid, pid)}={val}")
    return ",".join(tokens)


def _sparse_rewards_punishments(
    focal_row: Mapping[str, Any],
    ctx: GameContext,
    rewards_on: bool,
    punish_on: bool,
    binary_targets: bool,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    rewards: Dict[str, int] = {}
    punishments: Dict[str, int] = {}
    if rewards_on:
        data = parse_dict(focal_row.get("data.rewarded"))
        for pid, units in data.items():
            if units > 0:
                avatar = ctx.avatar_by_player.get(str(pid), str(pid))
                rewards[avatar] = 1 if binary_targets else rewards.get(avatar, 0) + int(units)
    if punish_on:
        data = parse_dict(focal_row.get("data.punished"))
        for pid, units in data.items():
            if units > 0:
                avatar = ctx.avatar_by_player.get(str(pid), str(pid))
                punishments[avatar] = 1 if binary_targets else punishments.get(avatar, 0) + int(units)
    return rewards, punishments


def _others_summary(
    round_slice: pd.DataFrame,
    ctx: GameContext,
    focal_pid: str,
    binary_targets: bool,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    row_by_pid = {str(r["playerId"]): r for _, r in round_slice.iterrows()}
    punishment_exists = as_bool(ctx.env.get("CONFIG_punishmentExists"))
    reward_exists = as_bool(ctx.env.get("CONFIG_rewardExists"))
    punishment_cost = int(ctx.env.get("CONFIG_punishmentCost", 0) or 0)
    reward_cost = int(ctx.env.get("CONFIG_rewardCost", 0) or 0)
    for pid in ctx.player_ids:
        if pid == focal_pid:
            continue
        avatar = ctx.avatar_by_player.get(pid, pid)
        row = row_by_pid.get(pid)
        if row is None or is_nan(row.get("data.roundPayoff")):
            out[avatar] = {"status": "exited"}
            continue
        punish_units = (
            _positive_action_units(parse_dict(row.get("data.punished")), binary_targets)
            if punishment_exists
            else 0
        )
        reward_units = (
            _positive_action_units(parse_dict(row.get("data.rewarded")), binary_targets)
            if reward_exists
            else 0
        )
        payload: Dict[str, Any] = {}
        if punishment_exists:
            payload["coins_spent_on_punish"] = int(punish_units) * punishment_cost
            inbound_punish_units = _positive_action_units(parse_dict(row.get("data.punishedBy")), binary_targets)
            payload["coins_deducted_from_them"] = inbound_punish_units * int(
                ctx.env.get("CONFIG_punishmentMagnitude", 0) or 0
            )
        if reward_exists:
            payload["coins_spent_on_reward"] = int(reward_units) * reward_cost
            inbound_reward_units = _positive_action_units(parse_dict(row.get("data.rewardedBy")), binary_targets)
            payload["coins_rewarded_to_them"] = inbound_reward_units * int(
                ctx.env.get("CONFIG_rewardMagnitude", 0) or 0
            )
        payoff = row.get("data.roundPayoff")
        payload["payoff"] = None if is_nan(payoff) else int(float(payoff))
        out[avatar] = payload
    return out


def append_observed_round(
    ctx: GameContext,
    transcripts: Dict[str, List[str]],
    active_history: Dict[str, bool],
    round_idx: int,
    action_prompt_mode: str = "binary_targets",
):
    round_slice = ctx.round_to_rows.get(round_idx)
    if round_slice is None or round_slice.empty:
        return
    row_by_pid = {str(r["playerId"]): r for _, r in round_slice.iterrows()}
    chats_for_round = ctx.chats_by_round_phase.get(round_idx, {})
    punish_on = as_bool(ctx.env.get("CONFIG_punishmentExists"))
    reward_on = as_bool(ctx.env.get("CONFIG_rewardExists"))
    binary_targets = _use_binary_targets(action_prompt_mode)
    show_other = as_bool(ctx.env.get("CONFIG_showOtherSummaries"))
    show_punish_id = as_bool(ctx.env.get("CONFIG_showPunishmentId"))
    show_reward_id = as_bool(ctx.env.get("CONFIG_showRewardId"))
    for pid in ctx.player_ids:
        if not active_history.get(pid, True):
            continue
        row = row_by_pid.get(pid)
        if row is None:
            continue
        if is_nan(row.get("data.roundPayoff")):
            transcripts[pid].append(f'<EXIT round="{round_idx}"/>')
            active_history[pid] = False
            continue
        focal_avatar = ctx.avatar_by_player.get(pid, pid)
        transcripts[pid].append(round_open(ctx.env, round_idx))
        transcripts[pid].append(_round_info_tag(ctx.env))
        for speaker, text in chats_for_round.get("contribution", []):
            label = f"{speaker} (YOU)" if normalize_avatar(speaker) == normalize_avatar(focal_avatar) else speaker
            transcripts[pid].append(f"<CHAT> {{{label}: {text}}} </CHAT>")
        transcripts[pid].append("<CONTRIB>")
        transcripts[pid].append(format_contrib_answer(format_num(row.get("data.contribution"))))
        transcripts[pid].append("</CONTRIB>")
        contrib_series = pd.to_numeric(round_slice["data.contribution"], errors="coerce")
        total_contrib = float(contrib_series.dropna().sum())
        focal_contrib = 0.0 if is_nan(row.get("data.contribution")) else float(row.get("data.contribution"))
        others_total = total_contrib - focal_contrib
        active_players = int(pd.to_numeric(round_slice["data.roundPayoff"], errors="coerce").notna().sum())
        try:
            multiplied = float(ctx.env.get("CONFIG_multiplier", 0) or 0) * total_contrib
        except Exception:
            multiplied = float("nan")
        redistributed_each = multiplied / active_players if active_players > 0 else float("nan")
        others_avg = (others_total / (active_players - 1)) if active_players > 1 else float("nan")
        transcripts[pid].append(
            '<REDIST total_contrib="{}" others_total="{}" others_avg="{}" multiplied_contrib="{}" '
            'active_players="{}" redistributed_each="{}"/>'.format(
                format_num(total_contrib),
                format_num(round(others_total, 3)),
                format_num(round(others_avg, 3)) if not math.isnan(others_avg) else "NA",
                format_num(round(multiplied, 3)) if not math.isnan(multiplied) else "",
                active_players,
                format_num(round(redistributed_each, 3)) if not math.isnan(redistributed_each) else "NA",
            )
        )
        transcripts[pid].append(f"<PEERS_CONTRIBUTIONS> {_compact_outcomes(round_slice, ctx, pid)} </PEERS_CONTRIBUTIONS>")
        for speaker, text in chats_for_round.get("outcome", []):
            label = f"{speaker} (YOU)" if normalize_avatar(speaker) == normalize_avatar(focal_avatar) else speaker
            transcripts[pid].append(f"<CHAT> {{{label}: {text}}} </CHAT>")
        if reward_on or punish_on:
            mech_text = mech_info(ctx.env, action_prompt_mode)
            transcripts[pid].append(f"<MECHANISM_INFO> {mech_text} </MECHANISM_INFO>")
            rewards_sparse, punish_sparse = _sparse_rewards_punishments(
                row,
                ctx,
                reward_on,
                punish_on,
                binary_targets,
            )
            if reward_on and not punish_on:
                transcripts[pid].append("<REWARD>")
                if rewards_sparse:
                    transcripts[pid].append(f"You rewarded: {json_compact(rewards_sparse)}")
                else:
                    transcripts[pid].append("You did not reward anybody.")
                transcripts[pid].append("</REWARD>")
            elif punish_on and not reward_on:
                transcripts[pid].append("<PUNISHMENT>")
                if punish_sparse:
                    transcripts[pid].append(f"You punished: {json_compact(punish_sparse)}")
                else:
                    transcripts[pid].append("You did not punish anybody.")
                transcripts[pid].append("</PUNISHMENT>")
            else:
                transcripts[pid].append("<PUNISHMENT_REWARD>")
                if punish_sparse:
                    transcripts[pid].append(f"You punished: {json_compact(punish_sparse)}")
                else:
                    transcripts[pid].append("You did not punish anybody.")
                if rewards_sparse:
                    transcripts[pid].append(f"You rewarded: {json_compact(rewards_sparse)}")
                else:
                    transcripts[pid].append("You did not reward anybody.")
                transcripts[pid].append("</PUNISHMENT_REWARD>")
        if punish_on and show_punish_id:
            punish_by = parse_dict(row.get("data.punishedBy"))
            if punish_by:
                mapped = {
                    ctx.avatar_by_player.get(str(k), str(k)): (1 if binary_targets else int(v))
                    for k, v in punish_by.items()
                    if int(v) > 0
                }
                if mapped:
                    transcripts[pid].append(f"<PUNISHED_BY json='{json_compact(mapped)}'/>")
        if reward_on and show_reward_id:
            reward_by = parse_dict(row.get("data.rewardedBy"))
            if reward_by:
                mapped = {
                    ctx.avatar_by_player.get(str(k), str(k)): (1 if binary_targets else int(v))
                    for k, v in reward_by.items()
                    if int(v) > 0
                }
                if mapped:
                    transcripts[pid].append(f"<REWARDED_BY json='{json_compact(mapped)}'/>")
        focal_summary: Dict[str, Any] = {}
        if punish_on:
            pun_units = _positive_action_units(parse_dict(row.get("data.punished")), binary_targets)
            focal_summary["coins_spent_on_punish"] = int(pun_units) * int(ctx.env.get("CONFIG_punishmentCost", 0) or 0)
            inbound_punish_units = _positive_action_units(parse_dict(row.get("data.punishedBy")), binary_targets)
            focal_summary["coins_deducted_from_you"] = inbound_punish_units * int(
                ctx.env.get("CONFIG_punishmentMagnitude", 0) or 0
            )
        if reward_on:
            rew_units = _positive_action_units(parse_dict(row.get("data.rewarded")), binary_targets)
            focal_summary["coins_spent_on_reward"] = int(rew_units) * int(ctx.env.get("CONFIG_rewardCost", 0) or 0)
            inbound_reward_units = _positive_action_units(parse_dict(row.get("data.rewardedBy")), binary_targets)
            focal_summary["coins_rewarded_to_you"] = inbound_reward_units * int(
                ctx.env.get("CONFIG_rewardMagnitude", 0) or 0
            )
        payoff = row.get("data.roundPayoff")
        focal_summary["round_payoff"] = None if is_nan(payoff) else int(float(payoff))
        summary_payload: Dict[str, Any] = {f"{focal_avatar} (YOU)": focal_summary}
        if show_other:
            summary_payload.update(_others_summary(round_slice, ctx, pid, binary_targets))
        transcripts[pid].append(f"<ROUND SUMMARY json='{json_compact(summary_payload)}'/>")
        for speaker, text in chats_for_round.get("summary", []):
            label = f"{speaker} (YOU)" if normalize_avatar(speaker) == normalize_avatar(focal_avatar) else speaker
            transcripts[pid].append(f"<CHAT> {{{label}: {text}}} </CHAT>")
        transcripts[pid].append("</ROUND>")


def _empty_prediction() -> Dict[str, Any]:
    return {
        "pred_contribution": 0,
        "pred_contribution_raw": None,
        "pred_contribution_reasoning": None,
        "pred_contribution_parsed": False,
        "pred_punished_avatar": {},
        "pred_rewarded_avatar": {},
        "pred_actions_reasoning": None,
        "pred_actions_parsed": None,
    }


def _avatar_to_player_dict(ctx: GameContext, value: Mapping[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for avatar, units in value.items():
        pid = ctx.player_by_avatar.get(str(avatar))
        if pid is None:
            continue
        out[pid] = int(units)
    return out


def _use_binary_targets(action_prompt_mode: str) -> bool:
    return str(action_prompt_mode or "binary_targets").strip().lower() != "legacy_units"


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _positive_action_units(mapping: Mapping[str, Any], binary_targets: bool) -> int:
    if binary_targets:
        return sum(1 for value in mapping.values() if _safe_int(value) > 0)
    total = 0
    for value in mapping.values():
        units = _safe_int(value)
        if units > 0:
            total += units
    return total


def _coerce_target_list(raw_value: Any, peer_order: Sequence[str]) -> List[str]:
    allowed = set(peer_order)
    targets: List[str] = []
    seen: set[str] = set()

    def add_target(value: Any) -> None:
        avatar = str(value or "").strip()
        if not avatar or avatar not in allowed or avatar in seen:
            return
        seen.add(avatar)
        targets.append(avatar)

    if raw_value is None:
        return targets
    if isinstance(raw_value, list):
        for item in raw_value:
            add_target(item)
        return targets
    if isinstance(raw_value, dict):
        for key, value in raw_value.items():
            if _safe_int(value) > 0 or value is True:
                add_target(key)
        return targets
    if isinstance(raw_value, str):
        raw_text = raw_value.strip()
        if not raw_text:
            return targets
        if raw_text in allowed:
            add_target(raw_text)
            return targets
        for part in raw_text.split(","):
            add_target(part)
    return targets


def _parse_binary_action_response(
    payload: Mapping[str, Any],
    tag: str,
    peer_order: Sequence[str],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    allowed = set(peer_order)
    if tag == "PUNISHMENT":
        punish_targets = _coerce_target_list(
            payload.get("punish", payload.get("targets", payload.get("actions"))),
            peer_order,
        )
        return ({avatar: 1 for avatar in punish_targets}, {})
    if tag == "REWARD":
        reward_targets = _coerce_target_list(
            payload.get("reward", payload.get("targets", payload.get("actions"))),
            peer_order,
        )
        return ({}, {avatar: 1 for avatar in reward_targets})

    raw_actions = payload.get("actions")
    if isinstance(raw_actions, dict) and "punish" not in payload and "reward" not in payload:
        punish_out = {
            str(avatar): 1
            for avatar, units in raw_actions.items()
            if str(avatar) in allowed and _safe_int(units) < 0
        }
        reward_out = {
            str(avatar): 1
            for avatar, units in raw_actions.items()
            if str(avatar) in allowed and _safe_int(units) > 0
        }
        return punish_out, reward_out

    punish_targets = _coerce_target_list(payload.get("punish"), peer_order)
    punish_set = set(punish_targets)
    reward_targets = [
        avatar
        for avatar in _coerce_target_list(payload.get("reward"), peer_order)
        if avatar not in punish_set
    ]
    return (
        {avatar: 1 for avatar in punish_targets},
        {avatar: 1 for avatar in reward_targets},
    )


def _clamp_probability(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 1.0


def _apply_continuation_gate(
    proposed: Mapping[str, int],
    previous_round: Mapping[str, int],
    keep_prob: float,
    rng: random.Random,
) -> Dict[str, int]:
    if not proposed:
        return {}
    if keep_prob >= 1.0 or not previous_round:
        return {str(target): int(units) for target, units in proposed.items() if int(units) > 0}
    kept: Dict[str, int] = {}
    for target, units in proposed.items():
        if int(units) <= 0:
            continue
        if int(previous_round.get(target, 0)) > 0 and rng.random() > keep_prob:
            continue
        kept[str(target)] = int(units)
    return kept


def _previous_observed_actions_by_avatar(
    ctx: GameContext,
    round_idx: int,
    binary_targets: bool,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    previous_punish = {ctx.avatar_by_player.get(pid, pid): {} for pid in ctx.player_ids}
    previous_reward = {ctx.avatar_by_player.get(pid, pid): {} for pid in ctx.player_ids}
    prev_slice = ctx.round_to_rows.get(int(round_idx) - 1)
    if prev_slice is None or prev_slice.empty:
        return previous_punish, previous_reward
    row_by_pid = {str(r["playerId"]): r for _, r in prev_slice.iterrows()}
    for pid in ctx.player_ids:
        avatar = ctx.avatar_by_player.get(pid, pid)
        row = row_by_pid.get(pid)
        if row is None:
            continue
        punish_map: Dict[str, int] = {}
        reward_map: Dict[str, int] = {}
        for target_pid, units in parse_dict(row.get("data.punished")).items():
            if int(units) <= 0:
                continue
            target_avatar = ctx.avatar_by_player.get(str(target_pid), str(target_pid))
            punish_map[target_avatar] = 1 if binary_targets else int(units)
        for target_pid, units in parse_dict(row.get("data.rewarded")).items():
            if int(units) <= 0:
                continue
            target_avatar = ctx.avatar_by_player.get(str(target_pid), str(target_pid))
            reward_map[target_avatar] = 1 if binary_targets else int(units)
        previous_punish[avatar] = punish_map
        previous_reward[avatar] = reward_map
    return previous_punish, previous_reward


def predict_round_behavior(
    ctx: GameContext,
    round_idx: int,
    transcripts: Dict[str, List[str]],
    active_history: Dict[str, bool],
    client: LLMClient,
    tok: Optional[Any],
    args: Any,
    debug_records: List[Dict[str, Any]],
    debug_full_records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    prediction_players = [pid for pid in ctx.player_ids if active_history.get(pid, True)]
    if not prediction_players:
        return {}
    round_slice = ctx.round_to_rows.get(round_idx, pd.DataFrame())
    row_by_pid: Dict[str, Any] = {}
    if not round_slice.empty and "playerId" in round_slice.columns:
        row_by_pid = {str(r["playerId"]): r for _, r in round_slice.iterrows()}
    include_system_in_prompt = client.provider == "local"
    stop_sequences = [JSON_STOP_SENTINEL]
    debug_excerpt_chars = 200
    state = {pid: _empty_prediction() for pid in prediction_players}
    work = {pid: list(transcripts[pid]) for pid in prediction_players}
    system_text_by_pid = {
        pid: system_header_plain(
            ctx.env,
            ctx.demographics_by_player.get(pid, "") if bool(getattr(args, "include_demographics", False)) else "",
            include_reasoning=args.include_reasoning,
        )
        for pid in prediction_players
    }
    for pid in prediction_players:
        work[pid].append(round_open(ctx.env, round_idx))

    contrib_prompts: List[str] = []
    contrib_messages_list: List[List[Dict[str, str]]] = []
    contrib_meta: List[str] = []
    contrib_prompt_by_pid: Dict[str, str] = {}
    for pid in prediction_players:
        chunks = work[pid] + [
            round_info_line(ctx.env),
            max_tokens_reminder_line(args.contrib_max_new_tokens),
            contrib_format_line(ctx.env, args.include_reasoning),
        ]
        if include_system_in_prompt:
            prompt = "\n".join([system_text_by_pid[pid]] + chunks)
        else:
            prompt = "\n".join(chunks)
        contrib_prompts.append(prompt)
        contrib_meta.append(pid)
        contrib_prompt_by_pid[pid] = prompt
        contrib_messages_list.append(build_openai_messages(system_text_by_pid[pid], chunks))
        if args.debug_print:
            if tok is not None:
                token_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
                log(f"[micro] {ctx.game_id} r={round_idx:02d} {ctx.avatar_by_player[pid]} CONTRIB prompt_tokens~{token_len}")
            else:
                log(f"[micro] {ctx.game_id} r={round_idx:02d} {ctx.avatar_by_player[pid]} CONTRIB prompt_chars={len(prompt)}")
    t1 = time.perf_counter()
    contrib_raw = client.generate_batch(
        prompts=contrib_prompts,
        messages_list=contrib_messages_list,
        stop=stop_sequences,
        max_new_tokens=args.contrib_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        async_openai=_remote_async_enabled(args, client.provider),
        max_concurrency=_remote_max_concurrency(args, client.provider),
    )
    dt_contrib = time.perf_counter() - t1
    predicted_contribution_by_avatar: Dict[str, int] = {}
    for pid, gen in zip(contrib_meta, contrib_raw):
        prompt_text = contrib_prompt_by_pid[pid]
        dt_per = dt_contrib / max(1, len(contrib_raw))
        if args.debug_level != "off":
            debug_records.append(
                build_debug_record(
                    game_id=ctx.game_id,
                    round_idx=round_idx,
                    agent=ctx.avatar_by_player.get(pid, pid),
                    phase="contrib",
                    dt_sec=dt_per,
                    prompt=prompt_text,
                    raw_output=gen,
                    debug_level=args.debug_level,
                    excerpt_chars=debug_excerpt_chars,
                )
            )
        if args.debug_full_jsonl_path:
            debug_full_records.append(
                build_full_debug_record(
                    game_id=ctx.game_id,
                    round_idx=round_idx,
                    agent=ctx.avatar_by_player.get(pid, pid),
                    phase="contrib",
                    dt_sec=dt_per,
                    prompt=prompt_text,
                    raw_output=gen,
                )
            )
        payload, ok = parse_json_response(gen)
        val = 0
        parsed_ok = False
        if ok and isinstance(payload, dict) and payload.get("stage") == "contribution":
            raw_val = payload.get("contribution")
            if isinstance(raw_val, int):
                val = raw_val
                parsed_ok = True
            elif isinstance(raw_val, str) and raw_val.strip().lstrip("-").isdigit():
                val = int(raw_val.strip())
                parsed_ok = True
        endow = int(ctx.env.get("CONFIG_endowment", 0) or 0)
        if as_bool(ctx.env.get("CONFIG_allOrNothing", False)):
            val = endow if val >= (endow // 2) else 0
        else:
            val = max(0, min(endow, int(val)))
        state[pid]["pred_contribution"] = int(val)
        state[pid]["pred_contribution_raw"] = int(val) if parsed_ok else None
        state[pid]["pred_contribution_parsed"] = parsed_ok
        if args.include_reasoning and ok and isinstance(payload, dict) and isinstance(payload.get("reasoning"), str):
            state[pid]["pred_contribution_reasoning"] = payload.get("reasoning")
        predicted_contribution_by_avatar[ctx.avatar_by_player.get(pid, pid)] = int(val)

    # For action prompts in this round, condition on observed current-round contributions.
    observed_contribution_by_player: Dict[str, Optional[int]] = {}
    observed_contribution_by_avatar: Dict[str, int] = {}
    for pid in prediction_players:
        avatar = ctx.avatar_by_player.get(pid, pid)
        row = row_by_pid.get(pid)
        observed_val: Optional[int] = None
        if row is not None:
            raw_obs = row.get("data.contribution")
            if not is_nan(raw_obs):
                try:
                    observed_val = int(float(raw_obs))
                except Exception:
                    observed_val = None
        observed_contribution_by_player[pid] = observed_val
        observed_contribution_by_avatar[avatar] = (
            observed_val if observed_val is not None else predicted_contribution_by_avatar.get(avatar, 0)
        )

    total_contrib = sum(observed_contribution_by_avatar.values())
    try:
        multiplied = float(ctx.env.get("CONFIG_multiplier", 0) or 0) * float(total_contrib)
    except Exception:
        multiplied = float("nan")
    if "data.roundPayoff" in round_slice.columns:
        active_players = int(pd.to_numeric(round_slice["data.roundPayoff"], errors="coerce").notna().sum())
    else:
        active_players = len(prediction_players)
    if active_players <= 0:
        active_players = len(prediction_players)
    roster = [ctx.avatar_by_player.get(pid, pid) for pid in prediction_players]
    for pid in prediction_players:
        avatar = ctx.avatar_by_player.get(pid, pid)
        focal_contrib = observed_contribution_by_player.get(pid)
        if focal_contrib is None:
            focal_contrib = predicted_contribution_by_avatar.get(avatar, 0)
        work[pid].append("<CONTRIB>")
        work[pid].append(format_contrib_answer(focal_contrib))
        work[pid].append("</CONTRIB>")
        work[pid].append(redist_line(total_contrib, multiplied, active_players))
        peers_csv, _ = peers_contributions_csv(roster, avatar, observed_contribution_by_avatar)
        work[pid].append(f"<PEERS_CONTRIBUTIONS> {peers_csv} </PEERS_CONTRIBUTIONS>")

    reward_on = as_bool(ctx.env.get("CONFIG_rewardExists", False))
    punish_on = as_bool(ctx.env.get("CONFIG_punishmentExists", False))
    action_prompt_mode = str(getattr(args, "action_prompt_mode", "binary_targets") or "binary_targets")
    binary_targets = _use_binary_targets(action_prompt_mode)
    gate_enabled = bool(getattr(args, "action_continuation_gate", True))
    punish_keep_prob = _clamp_probability(getattr(args, "punish_continuation_keep_prob", 0.5))
    reward_keep_prob = _clamp_probability(getattr(args, "reward_continuation_keep_prob", 0.35))
    prev_punish_by_avatar, prev_reward_by_avatar = _previous_observed_actions_by_avatar(ctx, round_idx, binary_targets)
    round_gate_rng = random.Random(f"{ctx.game_id}|{round_idx}|{args.seed}|micro_continuation_gate")
    if reward_on or punish_on:
        actions_prompts: List[str] = []
        actions_messages_list: List[List[Dict[str, str]]] = []
        actions_meta: List[str] = []
        actions_prompt_by_pid: Dict[str, str] = {}
        peer_orders: Dict[str, List[str]] = {}
        tag = actions_tag(ctx.env) or "PUNISHMENT/REWARD"
        mech = mech_info(ctx.env, action_prompt_mode)
        for pid in prediction_players:
            avatar = ctx.avatar_by_player.get(pid, pid)
            peer_orders[pid] = [ctx.avatar_by_player.get(p, p) for p in prediction_players if p != pid]
            chunks = list(work[pid])
            if mech:
                chunks.append(mech)
            chunks.extend(
                [
                    max_tokens_reminder_line(args.actions_max_new_tokens),
                    actions_format_line(tag, args.include_reasoning, action_prompt_mode),
                ]
            )
            if include_system_in_prompt:
                prompt = "\n".join([system_text_by_pid[pid]] + chunks)
            else:
                prompt = "\n".join(chunks)
            actions_prompts.append(prompt)
            actions_meta.append(pid)
            actions_prompt_by_pid[pid] = prompt
            actions_messages_list.append(build_openai_messages(system_text_by_pid[pid], chunks))
            if args.debug_print:
                if tok is not None:
                    token_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
                    log(f"[micro] {ctx.game_id} r={round_idx:02d} {avatar} ACTIONS prompt_tokens~{token_len}")
                else:
                    log(f"[micro] {ctx.game_id} r={round_idx:02d} {avatar} ACTIONS prompt_chars={len(prompt)}")
        t2 = time.perf_counter()
        actions_raw = client.generate_batch(
            prompts=actions_prompts,
            messages_list=actions_messages_list,
            stop=stop_sequences,
            max_new_tokens=args.actions_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            async_openai=_remote_async_enabled(args, client.provider),
            max_concurrency=_remote_max_concurrency(args, client.provider),
        )
        dt_actions = time.perf_counter() - t2
        for pid, gen in zip(actions_meta, actions_raw):
            prompt_text = actions_prompt_by_pid[pid]
            dt_per = dt_actions / max(1, len(actions_raw))
            if args.debug_level != "off":
                debug_records.append(
                    build_debug_record(
                        game_id=ctx.game_id,
                        round_idx=round_idx,
                        agent=ctx.avatar_by_player.get(pid, pid),
                        phase="actions",
                        dt_sec=dt_per,
                        prompt=prompt_text,
                        raw_output=gen,
                        debug_level=args.debug_level,
                        excerpt_chars=debug_excerpt_chars,
                    )
                )
            if args.debug_full_jsonl_path:
                debug_full_records.append(
                    build_full_debug_record(
                        game_id=ctx.game_id,
                        round_idx=round_idx,
                        agent=ctx.avatar_by_player.get(pid, pid),
                        phase="actions",
                        dt_sec=dt_per,
                        prompt=prompt_text,
                        raw_output=gen,
                    )
                )
            payload, ok = parse_json_response(gen)
            actions_dict: Dict[str, Any] = {}
            parsed_ok = False
            punish_out: Dict[str, int] = {}
            reward_out: Dict[str, int] = {}
            if ok and isinstance(payload, dict) and payload.get("stage") == "actions":
                parsed_ok = True
                if binary_targets:
                    punish_out, reward_out = _parse_binary_action_response(payload, tag, peer_orders[pid])
                else:
                    raw_actions = payload.get("actions")
                    if raw_actions is None:
                        actions_dict = {}
                    elif isinstance(raw_actions, dict):
                        actions_dict = raw_actions
                    arr: List[int] = []
                    for avatar in peer_orders[pid]:
                        raw_v = actions_dict.get(avatar, 0)
                        try:
                            arr.append(int(raw_v))
                        except Exception:
                            arr.append(0)
                    if reward_on and not punish_on:
                        arr = [max(0, int(v)) for v in arr]
                    elif punish_on and not reward_on:
                        arr = [max(0, int(v)) for v in arr]
                    else:
                        arr = [int(v) for v in arr]
                    actions_out = {
                        peer_orders[pid][i]: int(arr[i])
                        for i in range(len(peer_orders[pid]))
                        if int(arr[i]) != 0
                    }
                    if reward_on and not punish_on:
                        reward_out = {a: int(v) for a, v in actions_out.items() if int(v) > 0}
                    elif punish_on and not reward_on:
                        punish_out = {a: int(v) for a, v in actions_out.items() if int(v) > 0}
                    else:
                        punish_out = {a: int(abs(v)) for a, v in actions_out.items() if int(v) < 0}
                        reward_out = {a: int(v) for a, v in actions_out.items() if int(v) > 0}
            if args.include_reasoning and ok and isinstance(payload, dict) and isinstance(payload.get("reasoning"), str):
                state[pid]["pred_actions_reasoning"] = payload.get("reasoning")
                work[pid].append(f"You thought: {state[pid]['pred_actions_reasoning']}")
            avatar = ctx.avatar_by_player.get(pid, pid)
            if gate_enabled:
                punish_out = _apply_continuation_gate(
                    punish_out,
                    prev_punish_by_avatar.get(avatar, {}),
                    punish_keep_prob,
                    round_gate_rng,
                )
                reward_out = _apply_continuation_gate(
                    reward_out,
                    prev_reward_by_avatar.get(avatar, {}),
                    reward_keep_prob,
                    round_gate_rng,
                )
            if reward_on and not punish_on:
                punish_out = {}
                work[pid].append("<REWARD>")
                work[pid].append(f"You rewarded: {json_compact(reward_out)}" if reward_out else "You did not reward anybody.")
                work[pid].append("</REWARD>")
            elif punish_on and not reward_on:
                reward_out = {}
                work[pid].append("<PUNISHMENT>")
                work[pid].append(f"You punished: {json_compact(punish_out)}" if punish_out else "You did not punish anybody.")
                work[pid].append("</PUNISHMENT>")
            else:
                work[pid].append("<PUNISHMENT_REWARD>")
                work[pid].append(f"You punished: {json_compact(punish_out)}" if punish_out else "You did not punish anybody.")
                work[pid].append(f"You rewarded: {json_compact(reward_out)}" if reward_out else "You did not reward anybody.")
                work[pid].append("</PUNISHMENT_REWARD>")
            state[pid]["pred_punished_avatar"] = punish_out
            state[pid]["pred_rewarded_avatar"] = reward_out
            state[pid]["pred_actions_parsed"] = parsed_ok
    else:
        for pid in prediction_players:
            state[pid]["pred_punished_avatar"] = {}
            state[pid]["pred_rewarded_avatar"] = {}
            state[pid]["pred_actions_parsed"] = None
    return state


def build_eval_rows(
    ctx: GameContext,
    round_idx: int,
    predictions: Dict[str, Dict[str, Any]],
    round_slice: pd.DataFrame,
    args: Any,
    assigned_archetypes: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    row_by_pid = {str(r["playerId"]): r for _, r in round_slice.iterrows()}
    rows: List[Dict[str, Any]] = []
    archetype_mode = _resolve_archetype_mode(args)
    assigned_archetypes = assigned_archetypes or {}
    for pid, pred in predictions.items():
        actual_row = row_by_pid.get(pid)
        actual_available = actual_row is not None and not is_nan(actual_row.get("data.roundPayoff"))
        if args.skip_no_actual and not actual_available:
            continue
        avatar = ctx.avatar_by_player.get(pid, pid)
        actual_contrib: Optional[float] = None
        if actual_row is not None:
            c = actual_row.get("data.contribution")
            if c is not None and not is_nan(c):
                actual_contrib = float(c)
        pred_contrib = float(pred.get("pred_contribution", 0))
        contrib_abs_error = None if actual_contrib is None else abs(pred_contrib - actual_contrib)
        actual_punish_pid = parse_dict(actual_row.get("data.punished")) if actual_row is not None else {}
        actual_reward_pid = parse_dict(actual_row.get("data.rewarded")) if actual_row is not None else {}
        actual_punish_avatar = {ctx.avatar_by_player.get(str(k), str(k)): int(v) for k, v in actual_punish_pid.items() if int(v) > 0}
        actual_reward_avatar = {ctx.avatar_by_player.get(str(k), str(k)): int(v) for k, v in actual_reward_pid.items() if int(v) > 0}
        pred_punish_avatar = {str(k): int(v) for k, v in (pred.get("pred_punished_avatar") or {}).items() if int(v) > 0}
        pred_reward_avatar = {str(k): int(v) for k, v in (pred.get("pred_rewarded_avatar") or {}).items() if int(v) > 0}
        pred_punish_pid = _avatar_to_player_dict(ctx, pred_punish_avatar)
        pred_reward_pid = _avatar_to_player_dict(ctx, pred_reward_avatar)
        archetype_record = dict(assigned_archetypes.get(pid) or {})
        row = {
            "gameId": ctx.game_id,
            "gameName": ctx.game_name,
            "roundIndex": int(round_idx),
            "historyRounds": int(round_idx) - 1,
            "playerId": pid,
            "playerAvatar": avatar,
            "archetype": archetype_record.get("participant"),
            "persona": archetype_record.get("participant"),
            "archetype_mode": archetype_mode or "",
            "archetype_source_gameId": archetype_record.get("experiment"),
            "archetype_source_playerId": archetype_record.get("participant"),
            "archetype_source_rank": archetype_record.get("source_rank"),
            "archetype_source_score": archetype_record.get("source_score"),
            "archetype_source_weight": archetype_record.get("source_weight"),
            "actual_available": actual_available,
            "predicted_contribution": pred.get("pred_contribution"),
            "predicted_contribution_raw": pred.get("pred_contribution_raw"),
            "predicted_contribution_reasoning": pred.get("pred_contribution_reasoning"),
            "predicted_contribution_parsed": pred.get("pred_contribution_parsed"),
            "actual_contribution": actual_contrib,
            "contribution_abs_error": contrib_abs_error,
            "predicted_punished_avatar": json_compact(pred_punish_avatar),
            "predicted_rewarded_avatar": json_compact(pred_reward_avatar),
            "predicted_punished_pid": json_compact(pred_punish_pid),
            "predicted_rewarded_pid": json_compact(pred_reward_pid),
            "predicted_actions_reasoning": pred.get("pred_actions_reasoning"),
            "predicted_actions_parsed": pred.get("pred_actions_parsed"),
            "actual_punished_avatar": json_compact(actual_punish_avatar),
            "actual_rewarded_avatar": json_compact(actual_reward_avatar),
            "actual_punished_pid": json_compact(actual_punish_pid),
            "actual_rewarded_pid": json_compact(actual_reward_pid),
            "actual_data.punished_raw": None if actual_row is None else actual_row.get("data.punished"),
            "actual_data.rewarded_raw": None if actual_row is None else actual_row.get("data.rewarded"),
            "demographics": ctx.demographics_by_player.get(pid, ""),
        }
        rows.append(row)
    return rows


def _resolve_archetype_mode(args: Any) -> str:
    archetype_mode = str(getattr(args, "archetype", None) or "").strip()
    if archetype_mode:
        return canonicalize_archetype_mode(archetype_mode)
    return canonicalize_archetype_mode(str(getattr(args, "persona", None) or "").strip())


def _resolve_archetype_pool_path(args: Any) -> str:
    archetype_pool = str(getattr(args, "archetype_summary_pool", None) or "").strip()
    if archetype_pool:
        return archetype_pool
    return str(getattr(args, "persona_summary_pool", None) or "").strip()


def _resolve_archetype_assignment_manifest_path(args: Any) -> str:
    return str(getattr(args, "archetype_assignments_in_path", None) or "").strip()


def _remote_async_enabled(args: Any, provider: str) -> bool:
    provider_name = str(provider or "").strip().lower()
    if provider_name == "vllm":
        return True
    return bool(getattr(args, "openai_async", False))


def _remote_max_concurrency(args: Any, provider: str) -> int:
    provider_name = str(provider or "").strip().lower()
    if provider_name == "vllm":
        return max(1, int(getattr(args, "vllm_max_concurrency", 8)))
    return max(1, int(getattr(args, "openai_max_concurrency", 8)))


def _build_llm_client(
    args: Any,
    provider: str,
    tok: Optional[Any] = None,
    model: Optional[Any] = None,
) -> LLMClient:
    return LLMClient(
        provider=provider,
        tok=tok,
        model=model,
        base_model=getattr(args, "base_model", None),
        openai_model=getattr(args, "openai_model", None),
        openai_api_key=getattr(args, "openai_api_key", None),
        openai_api_key_env=getattr(args, "openai_api_key_env", "OPENAI_API_KEY"),
        vllm_model=getattr(args, "vllm_model", None),
        vllm_api_key=getattr(args, "vllm_api_key", None),
        vllm_api_key_env=getattr(args, "vllm_api_key_env", "VLLM_API_KEY"),
        vllm_base_url=getattr(args, "vllm_base_url", "http://localhost:8000/v1"),
    )


def evaluate_game(
    ctx: GameContext,
    client: LLMClient,
    tok: Optional[Any],
    args: Any,
    assigned_archetypes: Optional[Dict[str, Dict[str, Any]]],
    debug_records: List[Dict[str, Any]],
    debug_full_records: List[Dict[str, Any]],
    on_round_complete: Optional[
        Callable[[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]], None]
    ] = None,
    start_round: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    transcripts: Dict[str, List[str]] = {}
    archetype_mode = _resolve_archetype_mode(args)
    assigned_archetypes = assigned_archetypes or {}

    for pid in ctx.player_ids:
        lines: List[str] = []
        if archetype_mode:
            archetype_record = assigned_archetypes.get(pid)
            if archetype_record and archetype_record.get("text"):
                lines.extend(
                    [
                        "# YOUR ARCHETYPE",
                        SUMMARY_ARCHETYPE_INTRO,
                        "<SUMMARY STARTS>",
                        archetype_record["text"],
                        "<SUMMARY ENDS>",
                    ]
                )
        lines.append("# GAME STARTS")
        transcripts[pid] = lines

    active_history: Dict[str, bool] = {pid: True for pid in ctx.player_ids}
    rows: List[Dict[str, Any]] = []
    effective_start_round = int(args.start_round if start_round is None else start_round)
    for round_idx in ctx.rounds:
        round_debug_start = len(debug_records)
        round_debug_full_start = len(debug_full_records)
        if int(round_idx) >= effective_start_round:
            predictions = predict_round_behavior(
                ctx=ctx,
                round_idx=round_idx,
                transcripts=transcripts,
                active_history=active_history,
                client=client,
                tok=tok,
                args=args,
                debug_records=debug_records,
                debug_full_records=debug_full_records,
            )
            round_slice = ctx.round_to_rows.get(round_idx, pd.DataFrame())
            round_rows = build_eval_rows(
                ctx,
                round_idx,
                predictions,
                round_slice,
                args,
                assigned_archetypes=assigned_archetypes,
            )
            rows.extend(round_rows)
            if on_round_complete is not None and round_rows:
                round_debug_records = debug_records[round_debug_start:]
                round_debug_full_records = debug_full_records[round_debug_full_start:]
                on_round_complete(round_rows, round_debug_records, round_debug_full_records)
        append_observed_round(
            ctx,
            transcripts,
            active_history,
            round_idx,
            getattr(args, "action_prompt_mode", "binary_targets"),
        )
    for pid in ctx.player_ids:
        transcripts[pid].append("# GAME COMPLETE")
    return rows, transcripts


def _serialize_args(args: Any) -> Dict[str, Any]:
    if is_dataclass(args):
        return asdict(args)
    if hasattr(args, "__dict__"):
        return dict(vars(args))
    return dict(args)


def _model_config(args: Any, run_ts: str) -> Dict[str, Any]:
    args_dict = _serialize_args(args)
    provider = args_dict.get("provider")
    base_keys = [
        "provider",
        "openai_model",
        "openai_api_key_env",
        "openai_async",
        "openai_max_concurrency",
        "vllm_model",
        "vllm_base_url",
        "vllm_api_key_env",
        "vllm_max_concurrency",
        "temperature",
        "top_p",
        "seed",
        "contrib_max_new_tokens",
        "actions_max_new_tokens",
        "action_prompt_mode",
        "action_continuation_gate",
        "punish_continuation_keep_prob",
        "reward_continuation_keep_prob",
        "include_reasoning",
        "include_demographics",
        "archetype",
        "archetype_summary_pool",
        "start_round",
        "game_ids",
        "max_games",
        "skip_no_actual",
        "max_parallel_games",
        "debug_level",
        "debug_compact",
    ]
    if provider in {"local", "vllm"}:
        base_keys.append("base_model")
    if provider == "local":
        base_keys.extend(["adapter_path", "use_peft", "load_in_8bit", "load_in_4bit", "quant_compute_dtype"])
    model_payload = {k: args_dict.get(k) for k in base_keys if k in args_dict}
    model_payload["run_timestamp"] = run_ts
    return model_payload


def _write_config_json(config_path: str, payload: Dict[str, Any]) -> None:
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _resolve_run_location(output_root: str, run_id: Optional[str], resume_from_run: Optional[str]) -> Tuple[str, str, bool]:
    resume_spec = str(resume_from_run or "").strip()
    if not resume_spec:
        run_ts = str(run_id or timestamp_yymmddhhmm())
        return run_ts, os.path.join(output_root, run_ts), False

    candidates: List[str] = []
    if os.path.isabs(resume_spec):
        candidates.append(resume_spec)
    else:
        candidates.append(os.path.join(output_root, resume_spec))
        candidates.append(resume_spec)

    run_dir = ""
    for candidate in candidates:
        if os.path.isdir(candidate):
            run_dir = os.path.abspath(candidate)
            break
    if not run_dir:
        raise FileNotFoundError(
            "resume_from_run does not resolve to an existing run directory: "
            f"{resume_spec}"
        )

    run_ts = os.path.basename(os.path.normpath(run_dir))
    requested_run_id = str(run_id or "").strip()
    if requested_run_id and os.path.basename(requested_run_id.rstrip(os.sep)) != run_ts:
        raise ValueError(
            f"--run_id ({requested_run_id}) does not match resumed run directory ({run_ts})."
        )
    return run_ts, run_dir, True


def _load_json_file(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_existing_rows_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        if os.path.getsize(path) == 0:
            return pd.DataFrame()
    except OSError:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()


def _count_nonempty_lines(path: Optional[str]) -> int:
    if not path or not os.path.exists(path):
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _sanitize_resume_value(value: Any) -> Any:
    if isinstance(value, str) and value == "***redacted***":
        return None
    return value


def _apply_resume_args_from_config(args: Any, config_payload: Mapping[str, Any]) -> List[str]:
    stored_args = config_payload.get("args")
    if not isinstance(stored_args, Mapping):
        return []

    preserved_keys = {"resume_from_run", "debug_print"}
    restored_keys: List[str] = []
    for key, raw_value in stored_args.items():
        if key in preserved_keys or not hasattr(args, key):
            continue
        value = _sanitize_resume_value(raw_value)
        if value is None and key in {"openai_api_key", "vllm_api_key"}:
            continue
        current_value = getattr(args, key)
        if current_value != value:
            setattr(args, key, value)
            restored_keys.append(str(key))
    return restored_keys


def _micro_existing_row_counts(existing_rows: pd.DataFrame) -> Dict[str, Dict[int, int]]:
    if existing_rows.empty or "gameId" not in existing_rows.columns or "roundIndex" not in existing_rows.columns:
        return {}
    counts: Dict[str, Dict[int, int]] = {}
    frame = existing_rows.copy()
    frame["gameId"] = frame["gameId"].astype(str)
    frame["roundIndex"] = pd.to_numeric(frame["roundIndex"], errors="coerce")
    frame = frame.dropna(subset=["roundIndex"])
    if frame.empty:
        return {}
    frame["roundIndex"] = frame["roundIndex"].astype(int)
    grouped = frame.groupby(["gameId", "roundIndex"]).size()
    for (game_id, round_idx), count in grouped.items():
        counts.setdefault(str(game_id), {})[int(round_idx)] = int(count)
    return counts


def _expected_micro_round_row_counts(ctx: GameContext, skip_no_actual: bool) -> Dict[int, int]:
    active_history: Dict[str, bool] = {pid: True for pid in ctx.player_ids}
    counts: Dict[int, int] = {}
    for round_idx in ctx.rounds:
        round_slice = ctx.round_to_rows.get(round_idx, pd.DataFrame())
        row_by_pid = {str(r["playerId"]): r for _, r in round_slice.iterrows()} if not round_slice.empty else {}
        count = 0
        for pid in ctx.player_ids:
            if not active_history.get(pid, True):
                continue
            row = row_by_pid.get(pid)
            actual_available = row is not None and not is_nan(row.get("data.roundPayoff"))
            if (not skip_no_actual) or actual_available:
                count += 1
            if row is not None and is_nan(row.get("data.roundPayoff")):
                active_history[pid] = False
        counts[int(round_idx)] = int(count)
    return counts


def _first_unfinished_micro_round(
    ctx: GameContext,
    *,
    start_round: int,
    skip_no_actual: bool,
    existing_counts: Mapping[int, int],
) -> Tuple[Optional[int], Dict[int, int]]:
    expected_counts = _expected_micro_round_row_counts(ctx, skip_no_actual=skip_no_actual)
    for round_idx in ctx.rounds:
        if int(round_idx) < int(start_round):
            continue
        expected = int(expected_counts.get(int(round_idx), 0))
        actual = int(existing_counts.get(int(round_idx), 0))
        if expected > 0 and actual < expected:
            return int(round_idx), expected_counts
    return None, expected_counts


def run_micro_behavior_eval(args: Any) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    run_ts, run_dir, resume_enabled = _resolve_run_location(
        args.output_root,
        getattr(args, "run_id", None),
        getattr(args, "resume_from_run", None),
    )
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, "config.json")
    existing_config = _load_json_file(config_path) if resume_enabled else {}
    restored_arg_keys = _apply_resume_args_from_config(args, existing_config) if resume_enabled else []

    if args.debug_compact:
        args.debug_level = "compact"
    if args.debug_level not in {"full", "compact", "off"}:
        log(f"[warn] unknown debug_level='{args.debug_level}', defaulting to 'full'")
        args.debug_level = "full"
    if args.debug_level == "off":
        args.debug_jsonl_path = None
        args.debug_full_jsonl_path = None

    rows_out_path = relocate_output(args.rows_out_path, run_dir)
    transcripts_out_path = relocate_output(args.transcripts_out_path, run_dir)
    archetype_assignments_out_path = (
        relocate_output(args.archetype_assignments_out_path, run_dir)
        if getattr(args, "archetype_assignments_out_path", None)
        else None
    )
    debug_jsonl_path = relocate_output(args.debug_jsonl_path, run_dir) if args.debug_jsonl_path else None
    debug_full_jsonl_path = relocate_output(args.debug_full_jsonl_path, run_dir) if args.debug_full_jsonl_path else None
    existing_rows_df = _read_existing_rows_csv(rows_out_path) if resume_enabled else pd.DataFrame()
    existing_row_counts = _micro_existing_row_counts(existing_rows_df) if resume_enabled else {}
    existing_assignment_rows = _count_nonempty_lines(archetype_assignments_out_path) if resume_enabled else 0

    df_rounds = pd.read_csv(args.rounds_csv)
    df_analysis = pd.read_csv(args.analysis_csv)
    df_demographics = pd.read_csv(args.demographics_csv) if args.demographics_csv and os.path.exists(args.demographics_csv) else pd.DataFrame()
    df_players = pd.read_csv(args.players_csv) if args.players_csv and os.path.exists(args.players_csv) else pd.DataFrame()
    df_games = pd.read_csv(args.games_csv) if args.games_csv and os.path.exists(args.games_csv) else pd.DataFrame()

    contexts = build_game_contexts(
        df_rounds=df_rounds,
        df_analysis=df_analysis,
        df_demographics=df_demographics,
        df_players=df_players,
        df_games=df_games,
    )
    if args.game_ids:
        wanted = {x.strip() for x in str(args.game_ids).split(",") if x.strip()}
        contexts = [c for c in contexts if c.game_id in wanted or c.game_name in wanted]
    if args.max_games is not None:
        contexts = contexts[: int(args.max_games)]
    if not contexts:
        raise ValueError("No games selected for evaluation.")
    selected_contexts = list(contexts)

    resume_round_by_game: Dict[str, int] = {}
    completed_game_ids: List[str] = []
    resume_warnings: List[str] = []
    if resume_enabled:
        remaining_contexts: List[GameContext] = []
        for ctx in selected_contexts:
            existing_counts = existing_row_counts.get(ctx.game_id, {})
            first_unfinished_round, expected_counts = _first_unfinished_micro_round(
                ctx,
                start_round=int(args.start_round),
                skip_no_actual=bool(args.skip_no_actual),
                existing_counts=existing_counts,
            )
            for round_idx, expected in expected_counts.items():
                actual = int(existing_counts.get(int(round_idx), 0))
                if expected > 0 and actual > expected:
                    resume_warnings.append(
                        f"gameId={ctx.game_id} round={int(round_idx)} has {actual} existing rows; expected at most {expected}."
                    )
            if first_unfinished_round is None:
                completed_game_ids.append(ctx.game_id)
                continue
            resume_round_by_game[ctx.game_id] = int(first_unfinished_round)
            remaining_contexts.append(ctx)
        contexts = remaining_contexts

    args_payload = _serialize_args(args)
    if args_payload.get("openai_api_key") is not None:
        args_payload["openai_api_key"] = "***redacted***"
    if args_payload.get("vllm_api_key") is not None:
        args_payload["vllm_api_key"] = "***redacted***"

    now_iso = datetime.now(timezone.utc).isoformat()
    created_at_utc = str(existing_config.get("created_at_utc") or now_iso)
    pending_game_ids = [ctx.game_id for ctx in contexts]
    resume_event: Optional[Dict[str, Any]] = None
    resume_history = list(existing_config.get("resume_history") or []) if isinstance(existing_config.get("resume_history"), list) else []
    if resume_enabled:
        resume_event = {
            "resumed_at_utc": now_iso,
            "resume_from_run": str(getattr(args, "resume_from_run", "") or run_ts),
            "run_directory": run_dir,
            "previous_status": existing_config.get("status"),
            "existing_rows": int(len(existing_rows_df)),
            "completed_game_ids": completed_game_ids,
            "pending_game_ids": pending_game_ids,
            "game_resume_rounds": {gid: int(round_idx) for gid, round_idx in sorted(resume_round_by_game.items())},
        }
        if restored_arg_keys:
            resume_event["restored_args_from_config"] = restored_arg_keys
        if resume_warnings:
            resume_event["warnings"] = resume_warnings
        resume_history.append(resume_event)

    config_payload = {
        "run_timestamp": run_ts,
        "status": "running",
        "created_at_utc": created_at_utc,
        "inputs": {
            "rounds_csv": args.rounds_csv,
            "analysis_csv": args.analysis_csv,
            "demographics_csv": args.demographics_csv,
            "players_csv": args.players_csv,
            "games_csv": args.games_csv,
        },
        "selection": {
            "num_games": len(selected_contexts),
            "game_ids": [c.game_id for c in selected_contexts],
            "requested_game_ids": args.game_ids,
            "start_round": int(args.start_round),
            "skip_no_actual": bool(args.skip_no_actual),
        },
        "model": _model_config(args, run_ts),
        "args": args_payload,
        "outputs": {
            "directory": run_dir,
            "rows": rows_out_path,
            "transcripts": transcripts_out_path,
            "archetype_assignments": archetype_assignments_out_path,
            "debug": debug_jsonl_path,
            "debug_full": debug_full_jsonl_path,
        },
    }
    if resume_enabled:
        config_payload["resume"] = resume_event
        config_payload["resume_history"] = resume_history
    _write_config_json(config_path, config_payload)

    archetype_mode = _resolve_archetype_mode(args)
    archetype_assignments_in_path = _resolve_archetype_assignment_manifest_path(args)
    if not archetype_mode:
        archetype_assignments_out_path = None
        config_payload["outputs"]["archetype_assignments"] = None
        if archetype_assignments_in_path:
            raise ValueError(
                "--archetype_assignments_in_path requires an archetype mode. "
                "Use --archetype config_bank_archetype (or another supported mode)."
            )
    if archetype_mode and archetype_mode not in SUPPORTED_ARCHETYPE_MODES:
        raise ValueError(
            f"Unsupported archetype mode '{archetype_mode}'. Allowed values: {', '.join(sorted(SUPPORTED_ARCHETYPE_MODES))}."
        )
    archetype_summary_pool: Optional[ArchetypeSummaryPool] = None
    precomputed_assignment_index: Optional[PrecomputedAssignmentIndex] = None
    soft_bank_sampler: Optional[SoftBankSummarySampler] = None
    if archetype_mode in SUPPORTED_ARCHETYPE_MODES:
        if archetype_assignments_in_path:
            precomputed_assignment_index = load_precomputed_assignment_index(archetype_assignments_in_path)
            if args.debug_print:
                log(
                    "[micro] loaded precomputed archetype assignments from",
                    archetype_assignments_in_path,
                )
        else:
            archetype_summary_pool = load_finished_summary_pool(_resolve_archetype_pool_path(args))
            if archetype_mode == "config_bank_archetype":
                soft_bank_sampler = SoftBankSummarySampler(
                    summary_pool_path=_resolve_archetype_pool_path(args),
                    temperature=float(getattr(args, "archetype_soft_bank_temperature", 0.07)),
                )
            if args.debug_print:
                log(
                    "[micro] loaded archetype summaries:",
                    len(archetype_summary_pool.all_records),
                    "from",
                    _resolve_archetype_pool_path(args),
                )

    provider = str(args.provider).lower()
    max_parallel_games = max(1, int(getattr(args, "max_parallel_games", 1) or 1))
    if provider == "local" and max_parallel_games > 1:
        log(
            "[warn] max_parallel_games > 1 is supported only for remote providers; "
            "falling back to sequential local execution."
        )
        max_parallel_games = 1

    tok: Optional[Any] = None
    model: Optional[Any] = None
    client: Optional[LLMClient] = None
    if contexts:
        if provider == "local":
            tok, model = load_model(
                base_model=args.base_model,
                adapter_path=args.adapter_path,
                use_peft=args.use_peft,
                load_in_8bit=getattr(args, "load_in_8bit", False),
                load_in_4bit=getattr(args, "load_in_4bit", False),
                quant_compute_dtype=getattr(args, "quant_compute_dtype", "auto"),
            )
        client = _build_llm_client(args, provider, tok=tok, model=model)

    all_rows: List[Dict[str, Any]] = existing_rows_df.to_dict(orient="records") if not existing_rows_df.empty else []
    assignment_row_count = int(existing_assignment_rows)
    debug_records: List[Dict[str, Any]] = []
    debug_full_records: List[Dict[str, Any]] = []
    rows_fieldnames = list(existing_rows_df.columns) if not existing_rows_df.empty else None
    rows_file = None
    rows_writer: Optional[csv.DictWriter] = None
    rows_has_header = bool(rows_fieldnames) and bool(os.path.exists(rows_out_path)) and os.path.getsize(rows_out_path) > 0

    transcripts_file = None
    assignment_file = None
    debug_file = None
    debug_full_file = None
    write_lock = Lock()

    def _flush_file(handle):
        handle.flush()
        os.fsync(handle.fileno())

    def _write_rows_chunk(chunk: List[Dict[str, Any]]):
        nonlocal rows_writer
        if not chunk:
            return
        with write_lock:
            if rows_writer is None:
                fieldnames = rows_fieldnames or list(chunk[0].keys())
                rows_writer = csv.DictWriter(rows_file, fieldnames=fieldnames)
                if not rows_has_header:
                    rows_writer.writeheader()
            for row in chunk:
                rows_writer.writerow(row)
            _flush_file(rows_file)

    def _write_transcripts(ctx: GameContext, transcripts: Dict[str, List[str]]) -> None:
        if transcripts_file is None:
            return
        with write_lock:
            for pid in ctx.player_ids:
                sys_text = system_header_plain(
                    ctx.env,
                    ctx.demographics_by_player.get(pid, "") if bool(getattr(args, "include_demographics", False)) else "",
                    args.include_reasoning,
                )
                body = "\n".join(transcripts.get(pid, ["# GAME STARTS", "# GAME COMPLETE"]))
                text = f"{sys_text}\n{body}"
                transcripts_file.write(
                    json.dumps(
                        {
                            "experiment": ctx.game_id,
                            "participant": pid,
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            _flush_file(transcripts_file)

    def _write_assignment_chunk(chunk: List[Dict[str, Any]]) -> None:
        nonlocal assignment_row_count
        if assignment_file is None or not chunk:
            return
        with write_lock:
            for rec in chunk:
                assignment_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            assignment_row_count += len(chunk)
            _flush_file(assignment_file)

    def _write_debug_chunk(chunk: List[Dict[str, Any]]) -> None:
        if debug_file is None or not chunk:
            return
        with write_lock:
            for rec in chunk:
                debug_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            _flush_file(debug_file)

    def _write_debug_full_chunk(chunk: List[Dict[str, Any]]) -> None:
        if debug_full_file is None or not chunk:
            return
        with write_lock:
            for rec in chunk:
                debug_full_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            _flush_file(debug_full_file)

    def _on_round_complete(
        round_rows: List[Dict[str, Any]],
        round_dbg: List[Dict[str, Any]],
        round_dbg_full: List[Dict[str, Any]],
    ) -> None:
        _write_rows_chunk(round_rows)
        _write_debug_chunk(round_dbg)
        _write_debug_full_chunk(round_dbg_full)

    def _run_one_game(game_idx: int, ctx: GameContext) -> Tuple[
        int,
        GameContext,
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        Dict[str, List[str]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        game_debug_records: List[Dict[str, Any]] = []
        game_debug_full_records: List[Dict[str, Any]] = []
        assigned = {}
        assignment_rows: List[Dict[str, Any]] = []
        if archetype_mode:
            assigned_batch = assign_archetypes_for_game(
                mode=archetype_mode,
                game_id=ctx.game_id,
                player_ids=ctx.player_ids,
                env=ctx.env,
                seed=int(getattr(args, "seed", 0)),
                summary_pool=archetype_summary_pool,
                summary_pool_path=_resolve_archetype_pool_path(args),
                soft_bank_sampler=soft_bank_sampler,
                precomputed_assignment_index=precomputed_assignment_index,
                log_fn=log,
            )
            assigned = assigned_batch.assignments_by_player
            assignment_rows = assigned_batch.manifest_rows
        worker_client = client if provider == "local" else _build_llm_client(args, provider)
        rows, transcripts = evaluate_game(
            ctx=ctx,
            client=worker_client,
            tok=tok,
            args=args,
            assigned_archetypes=assigned,
            debug_records=game_debug_records,
            debug_full_records=game_debug_full_records,
            on_round_complete=_on_round_complete,
            start_round=resume_round_by_game.get(ctx.game_id, int(args.start_round)),
        )
        return game_idx, ctx, assignment_rows, rows, transcripts, game_debug_records, game_debug_full_records

    if resume_enabled:
        log(
            f"[micro] resume {run_ts}: preserved {len(all_rows)} existing rows, "
            f"completed_games={len(completed_game_ids)}, pending_games={len(contexts)}"
        )
        if restored_arg_keys:
            log(f"[micro] restored {len(restored_arg_keys)} args from {config_path}")
        for warning in resume_warnings:
            log(f"[warn] {warning}")

    if contexts:
        os.makedirs(os.path.dirname(rows_out_path), exist_ok=True)
        rows_file = open(rows_out_path, "a" if resume_enabled else "w", newline="", encoding="utf-8")
        if rows_has_header and rows_fieldnames:
            rows_writer = csv.DictWriter(rows_file, fieldnames=rows_fieldnames)

        if transcripts_out_path:
            os.makedirs(os.path.dirname(transcripts_out_path), exist_ok=True)
            transcripts_file = open(transcripts_out_path, "a" if resume_enabled else "w", encoding="utf-8")

        if archetype_assignments_out_path and archetype_mode:
            os.makedirs(os.path.dirname(archetype_assignments_out_path), exist_ok=True)
            assignment_file = open(archetype_assignments_out_path, "a" if resume_enabled else "w", encoding="utf-8")

        if debug_jsonl_path and args.debug_level != "off":
            os.makedirs(os.path.dirname(debug_jsonl_path), exist_ok=True)
            debug_file = open(debug_jsonl_path, "a" if resume_enabled else "w", encoding="utf-8")

        if debug_full_jsonl_path:
            os.makedirs(os.path.dirname(debug_full_jsonl_path), exist_ok=True)
            debug_full_file = open(debug_full_jsonl_path, "a" if resume_enabled else "w", encoding="utf-8")

        try:
            if provider != "local" and max_parallel_games > 1:
                max_workers = min(max_parallel_games, len(contexts))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    pending_results: Dict[
                        int,
                        Tuple[
                            GameContext,
                            List[Dict[str, Any]],
                            List[Dict[str, Any]],
                            Dict[str, List[str]],
                            List[Dict[str, Any]],
                            List[Dict[str, Any]],
                        ],
                    ] = {}
                    next_game_idx = 1
                    for idx, ctx in enumerate(contexts, start=1):
                        if args.debug_print:
                            start_round = resume_round_by_game.get(ctx.game_id, int(args.start_round))
                            log(f"[micro] start game {ctx.game_id} ({idx}/{len(contexts)}) from round {start_round}")
                        futures[executor.submit(_run_one_game, idx, ctx)] = (idx, ctx)
                    for future in as_completed(futures):
                        game_idx, completed_ctx, assignment_rows, rows, transcripts, dbg, dbg_full = future.result()
                        pending_results[game_idx] = (completed_ctx, assignment_rows, rows, transcripts, dbg, dbg_full)
                        while next_game_idx in pending_results:
                            (
                                ordered_ctx,
                                ordered_assignment_rows,
                                ordered_rows,
                                ordered_transcripts,
                                ordered_dbg,
                                ordered_dbg_full,
                            ) = pending_results.pop(
                                next_game_idx
                            )
                            all_rows.extend(ordered_rows)
                            _write_assignment_chunk(ordered_assignment_rows)
                            _write_transcripts(ordered_ctx, ordered_transcripts)
                            if args.debug_print:
                                log(f"[micro] done game {ordered_ctx.game_id} -> {len(ordered_rows)} eval rows")
                            next_game_idx += 1
            else:
                for idx, ctx in enumerate(contexts, start=1):
                    debug_records.clear()
                    debug_full_records.clear()
                    if args.debug_print:
                        start_round = resume_round_by_game.get(ctx.game_id, int(args.start_round))
                        log(f"[micro] start game {ctx.game_id} ({idx}/{len(contexts)}) from round {start_round}")
                    assigned = {}
                    assignment_rows = []
                    if archetype_mode:
                        assigned_batch = assign_archetypes_for_game(
                            mode=archetype_mode,
                            game_id=ctx.game_id,
                            player_ids=ctx.player_ids,
                            env=ctx.env,
                            seed=int(getattr(args, "seed", 0)),
                            summary_pool=archetype_summary_pool,
                            summary_pool_path=_resolve_archetype_pool_path(args),
                            soft_bank_sampler=soft_bank_sampler,
                            precomputed_assignment_index=precomputed_assignment_index,
                            log_fn=log,
                        )
                        assigned = assigned_batch.assignments_by_player
                        assignment_rows = assigned_batch.manifest_rows
                    rows, transcripts = evaluate_game(
                        ctx=ctx,
                        client=client,
                        tok=tok,
                        args=args,
                        assigned_archetypes=assigned,
                        debug_records=debug_records,
                        debug_full_records=debug_full_records,
                        on_round_complete=_on_round_complete,
                        start_round=resume_round_by_game.get(ctx.game_id, int(args.start_round)),
                    )
                    all_rows.extend(rows)
                    _write_assignment_chunk(assignment_rows)
                    _write_transcripts(ctx, transcripts)
                    if args.debug_print:
                        log(f"[micro] done game {ctx.game_id} -> {len(rows)} eval rows")
        finally:
            rows_file.close()
            if transcripts_file is not None:
                transcripts_file.close()
            if assignment_file is not None:
                assignment_file.close()
            if debug_file is not None:
                debug_file.close()
            if debug_full_file is not None:
                debug_full_file.close()
    elif resume_enabled:
        log(f"[micro] resume {run_ts}: no unfinished games found")

    df_out = pd.DataFrame(all_rows)
    if contexts and not df_out.empty:
        sort_keys = [col for col in ["gameId", "roundIndex", "playerId"] if col in df_out.columns]
        if sort_keys:
            df_out = df_out.sort_values(sort_keys, kind="stable").reset_index(drop=True)
        df_out.to_csv(rows_out_path, index=False)

    config_payload["status"] = "completed"
    config_payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    config_payload["summary"] = {
        "num_rows": int(len(all_rows)),
        "num_games": int(len(selected_contexts)),
        "num_archetype_assignment_rows": int(assignment_row_count),
    }
    _write_config_json(config_path, config_payload)

    if args.debug_print:
        log(f"[micro] wrote rows -> {rows_out_path}")
        if transcripts_out_path:
            log(f"[micro] wrote transcripts -> {transcripts_out_path}")
        if archetype_assignments_out_path:
            log(f"[micro] wrote archetype assignments -> {archetype_assignments_out_path}")
        if debug_jsonl_path:
            log(f"[micro] wrote debug -> {debug_jsonl_path}")
        if debug_full_jsonl_path:
            log(f"[micro] wrote full debug -> {debug_full_jsonl_path}")
        log(f"[micro] wrote config -> {config_path}")

    return df_out, {
        "rows": rows_out_path,
        "transcripts": transcripts_out_path,
        "archetype_assignments": archetype_assignments_out_path,
        "debug": debug_jsonl_path,
        "debug_full": debug_full_jsonl_path,
        "config": config_path,
        "directory": run_dir,
    }
