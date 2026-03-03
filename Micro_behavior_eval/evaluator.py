from __future__ import annotations

import json
import math
import os
import re
import time
import ast
import csv
import random
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from Simulation_robin.llm_client import LLMClient

try:
    from .debug import build_debug_record, build_full_debug_record
    from .model_loader import load_model
    from .parsers import parse_json_response
    from .prompt_builder import (
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


@dataclass
class ArchetypeSummaryPool:
    all_records: List[Dict[str, str]]
    by_game_player: Dict[str, Dict[str, Dict[str, str]]]


SUMMARY_ARCHETYPE_INTRO = (
    "Below is an archetype summary of how you played a different PGG in the past. "
    "Be aware of this archetype as you make decisions. "
    "Recall that you're probably playing games with different people from the past, and "
    "that the exact rules of this game could differ from the ones you've played before."
)


def _load_archetype_summary_pool(path: str) -> ArchetypeSummaryPool:
    if not path:
        raise ValueError("archetype_summary_pool path must be set when archetype mode is enabled")
    if not os.path.exists(path):
        raise FileNotFoundError(f"archetype summary pool file not found: {path}")
    all_records: List[Dict[str, str]] = []
    by_game_player: Dict[str, Dict[str, Dict[str, str]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if rec.get("game_finished") is not True:
                continue
            text = rec.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            participant = str(rec.get("participant") or "").strip()
            experiment = str(rec.get("experiment") or "").strip()
            entry = {
                "participant": participant,
                "experiment": experiment,
                "text": text.strip(),
            }
            all_records.append(entry)
            if participant and experiment:
                game_map = by_game_player.setdefault(experiment, {})
                if participant not in game_map:
                    game_map[participant] = entry
    if not all_records:
        raise ValueError(f"No finished-game archetype summary records found in {path}")
    return ArchetypeSummaryPool(all_records=all_records, by_game_player=by_game_player)


def _assign_summary_archetypes(
    ctx: GameContext,
    mode: str,
    seed: int,
    pool: ArchetypeSummaryPool,
) -> Dict[str, Dict[str, str]]:
    if mode == "matched_summary":
        game_map = pool.by_game_player.get(ctx.game_id, {})
        missing = [pid for pid in ctx.player_ids if pid not in game_map]
        if missing:
            sample_missing = ", ".join(missing[:5])
            log(
                f"[warn] matched_summary missing archetype summaries for gameId={ctx.game_id}, "
                f"missing_playerIds={sample_missing}, total_missing={len(missing)}; "
                "continuing without archetype for missing players."
            )
        return {pid: game_map[pid] for pid in ctx.player_ids if pid in game_map}

    rng = random.Random(f"{seed}|{ctx.game_id}|random_summary")
    records = pool.all_records
    return {pid: records[rng.randrange(len(records))] for pid in ctx.player_ids}


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
) -> Tuple[Dict[str, int], Dict[str, int]]:
    rewards: Dict[str, int] = {}
    punishments: Dict[str, int] = {}
    if rewards_on:
        data = parse_dict(focal_row.get("data.rewarded"))
        for pid, units in data.items():
            if units > 0:
                avatar = ctx.avatar_by_player.get(str(pid), str(pid))
                rewards[avatar] = rewards.get(avatar, 0) + int(units)
    if punish_on:
        data = parse_dict(focal_row.get("data.punished"))
        for pid, units in data.items():
            if units > 0:
                avatar = ctx.avatar_by_player.get(str(pid), str(pid))
                punishments[avatar] = punishments.get(avatar, 0) + int(units)
    return rewards, punishments


def _others_summary(round_slice: pd.DataFrame, ctx: GameContext, focal_pid: str) -> Dict[str, Dict[str, Any]]:
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
        punish_units = sum(parse_dict(row.get("data.punished")).values()) if punishment_exists else 0
        reward_units = sum(parse_dict(row.get("data.rewarded")).values()) if reward_exists else 0
        payload: Dict[str, Any] = {}
        if punishment_exists:
            payload["coins_spent_on_punish"] = int(punish_units) * punishment_cost
            payload["coins_deducted_from_them"] = int(float(row.get("data.penalties", 0) or 0))
        if reward_exists:
            payload["coins_spent_on_reward"] = int(reward_units) * reward_cost
            payload["coins_rewarded_to_them"] = int(float(row.get("data.rewards", 0) or 0))
        payoff = row.get("data.roundPayoff")
        payload["payoff"] = None if is_nan(payoff) else int(float(payoff))
        out[avatar] = payload
    return out


def append_observed_round(
    ctx: GameContext,
    transcripts: Dict[str, List[str]],
    active_history: Dict[str, bool],
    round_idx: int,
):
    round_slice = ctx.round_to_rows.get(round_idx)
    if round_slice is None or round_slice.empty:
        return
    row_by_pid = {str(r["playerId"]): r for _, r in round_slice.iterrows()}
    chats_for_round = ctx.chats_by_round_phase.get(round_idx, {})
    punish_on = as_bool(ctx.env.get("CONFIG_punishmentExists"))
    reward_on = as_bool(ctx.env.get("CONFIG_rewardExists"))
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
            if reward_on and punish_on:
                mech_text = (
                    f"It will cost you, per reward unit, {int(ctx.env.get('CONFIG_rewardCost', 0) or 0)} coins to give a "
                    f"reward of {int(ctx.env.get('CONFIG_rewardMagnitude', 0) or 0)} coins. It will cost you, per punishment unit, "
                    f"{int(ctx.env.get('CONFIG_punishmentCost', 0) or 0)} coins to impose a deduction of "
                    f"{int(ctx.env.get('CONFIG_punishmentMagnitude', 0) or 0)} coins. Choose whom to punish/reward and by how many units."
                )
            elif reward_on:
                mech_text = (
                    f"It will cost you, per unit, {int(ctx.env.get('CONFIG_rewardCost', 0) or 0)} coins to give a reward of "
                    f"{int(ctx.env.get('CONFIG_rewardMagnitude', 0) or 0)} coins. Choose whom to reward and by how many units."
                )
            else:
                mech_text = (
                    f"It will cost you, per unit, {int(ctx.env.get('CONFIG_punishmentCost', 0) or 0)} coins to impose a deduction of "
                    f"{int(ctx.env.get('CONFIG_punishmentMagnitude', 0) or 0)} coins. Choose whom to punish and by how many units."
                )
            transcripts[pid].append(f"<MECHANISM_INFO> {mech_text} </MECHANISM_INFO>")
            rewards_sparse, punish_sparse = _sparse_rewards_punishments(row, ctx, reward_on, punish_on)
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
                mapped = {ctx.avatar_by_player.get(str(k), str(k)): int(v) for k, v in punish_by.items() if int(v) > 0}
                if mapped:
                    transcripts[pid].append(f"<PUNISHED_BY json='{json_compact(mapped)}'/>")
        if reward_on and show_reward_id:
            reward_by = parse_dict(row.get("data.rewardedBy"))
            if reward_by:
                mapped = {ctx.avatar_by_player.get(str(k), str(k)): int(v) for k, v in reward_by.items() if int(v) > 0}
                if mapped:
                    transcripts[pid].append(f"<REWARDED_BY json='{json_compact(mapped)}'/>")
        focal_summary: Dict[str, Any] = {}
        if punish_on:
            pun_units = sum(parse_dict(row.get("data.punished")).values())
            focal_summary["coins_spent_on_punish"] = int(pun_units) * int(ctx.env.get("CONFIG_punishmentCost", 0) or 0)
            focal_summary["coins_deducted_from_you"] = int(float(row.get("data.penalties", 0) or 0))
        if reward_on:
            rew_units = sum(parse_dict(row.get("data.rewarded")).values())
            focal_summary["coins_spent_on_reward"] = int(rew_units) * int(ctx.env.get("CONFIG_rewardCost", 0) or 0)
            focal_summary["coins_rewarded_to_you"] = int(float(row.get("data.rewards", 0) or 0))
        payoff = row.get("data.roundPayoff")
        focal_summary["round_payoff"] = None if is_nan(payoff) else int(float(payoff))
        summary_payload: Dict[str, Any] = {f"{focal_avatar} (YOU)": focal_summary}
        if show_other:
            summary_payload.update(_others_summary(round_slice, ctx, pid))
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
    include_system_in_prompt = client.provider == "local"
    stop_sequences = ["\n\n"]
    debug_excerpt_chars = 200
    state = {pid: _empty_prediction() for pid in prediction_players}
    work = {pid: list(transcripts[pid]) for pid in prediction_players}
    system_text_by_pid = {
        pid: system_header_plain(ctx.env, ctx.demographics_by_player.get(pid, ""), include_reasoning=args.include_reasoning)
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
        async_openai=args.openai_async,
        max_concurrency=args.openai_max_concurrency,
    )
    dt_contrib = time.perf_counter() - t1
    contribution_by_avatar: Dict[str, int] = {}
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
            work[pid].append(f"You thought: {state[pid]['pred_contribution_reasoning']}")
        work[pid].append("<CONTRIB>")
        work[pid].append(format_contrib_answer(state[pid]["pred_contribution"] if parsed_ok else "NaN"))
        work[pid].append("</CONTRIB>")
        contribution_by_avatar[ctx.avatar_by_player.get(pid, pid)] = int(val)
    total_contrib = sum(contribution_by_avatar.values())
    try:
        multiplied = float(ctx.env.get("CONFIG_multiplier", 0) or 0) * float(total_contrib)
    except Exception:
        multiplied = float("nan")
    roster = [ctx.avatar_by_player.get(pid, pid) for pid in prediction_players]
    for pid in prediction_players:
        avatar = ctx.avatar_by_player.get(pid, pid)
        work[pid].append(redist_line(total_contrib, multiplied, len(prediction_players)))
        peers_csv, _ = peers_contributions_csv(roster, avatar, contribution_by_avatar)
        work[pid].append(f"<PEERS_CONTRIBUTIONS> {peers_csv} </PEERS_CONTRIBUTIONS>")

    reward_on = as_bool(ctx.env.get("CONFIG_rewardExists", False))
    punish_on = as_bool(ctx.env.get("CONFIG_punishmentExists", False))
    if reward_on or punish_on:
        actions_prompts: List[str] = []
        actions_messages_list: List[List[Dict[str, str]]] = []
        actions_meta: List[str] = []
        actions_prompt_by_pid: Dict[str, str] = {}
        peer_orders: Dict[str, List[str]] = {}
        tag = actions_tag(ctx.env) or "PUNISHMENT/REWARD"
        mech = mech_info(ctx.env)
        for pid in prediction_players:
            avatar = ctx.avatar_by_player.get(pid, pid)
            peer_orders[pid] = [ctx.avatar_by_player.get(p, p) for p in prediction_players if p != pid]
            chunks = list(work[pid])
            if mech:
                chunks.append(mech)
            chunks.extend(
                [
                    max_tokens_reminder_line(args.actions_max_new_tokens),
                    actions_format_line(tag, args.include_reasoning),
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
            async_openai=args.openai_async,
            max_concurrency=args.openai_max_concurrency,
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
            if ok and isinstance(payload, dict) and payload.get("stage") == "actions":
                raw_actions = payload.get("actions")
                if raw_actions is None:
                    actions_dict = {}
                    parsed_ok = True
                elif isinstance(raw_actions, dict):
                    actions_dict = raw_actions
                    parsed_ok = True
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
            if args.include_reasoning and ok and isinstance(payload, dict) and isinstance(payload.get("reasoning"), str):
                state[pid]["pred_actions_reasoning"] = payload.get("reasoning")
                work[pid].append(f"You thought: {state[pid]['pred_actions_reasoning']}")
            actions_out = {peer_orders[pid][i]: int(arr[i]) for i in range(len(peer_orders[pid])) if int(arr[i]) != 0}
            if reward_on and not punish_on:
                reward_out = {a: int(v) for a, v in actions_out.items() if int(v) > 0}
                punish_out: Dict[str, int] = {}
                work[pid].append("<REWARD>")
                work[pid].append(f"You rewarded: {json_compact(reward_out)}" if reward_out else "You did not reward anybody.")
                work[pid].append("</REWARD>")
            elif punish_on and not reward_on:
                punish_out = {a: int(v) for a, v in actions_out.items() if int(v) > 0}
                reward_out = {}
                work[pid].append("<PUNISHMENT>")
                work[pid].append(f"You punished: {json_compact(punish_out)}" if punish_out else "You did not punish anybody.")
                work[pid].append("</PUNISHMENT>")
            else:
                punish_out = {a: int(abs(v)) for a, v in actions_out.items() if int(v) < 0}
                reward_out = {a: int(v) for a, v in actions_out.items() if int(v) > 0}
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
) -> List[Dict[str, Any]]:
    row_by_pid = {str(r["playerId"]): r for _, r in round_slice.iterrows()}
    rows: List[Dict[str, Any]] = []
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
        row = {
            "gameId": ctx.game_id,
            "gameName": ctx.game_name,
            "roundIndex": int(round_idx),
            "historyRounds": int(round_idx) - 1,
            "playerId": pid,
            "playerAvatar": avatar,
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
        return archetype_mode
    return str(getattr(args, "persona", None) or "").strip()


def _resolve_archetype_pool_path(args: Any) -> str:
    archetype_pool = str(getattr(args, "archetype_summary_pool", None) or "").strip()
    if archetype_pool:
        return archetype_pool
    return str(getattr(args, "persona_summary_pool", None) or "").strip()


def evaluate_game(
    ctx: GameContext,
    client: LLMClient,
    tok: Optional[Any],
    args: Any,
    archetype_summary_pool: Optional[ArchetypeSummaryPool],
    debug_records: List[Dict[str, Any]],
    debug_full_records: List[Dict[str, Any]],
    on_round_rows: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    transcripts: Dict[str, List[str]] = {}
    archetype_mode = _resolve_archetype_mode(args)
    assigned_archetypes: Dict[str, Dict[str, str]] = {}
    if archetype_mode:
        if archetype_mode not in {"matched_summary", "random_summary"}:
            raise ValueError(
                f"Unsupported archetype mode '{archetype_mode}'. Allowed values: matched_summary, random_summary."
            )
        if archetype_summary_pool is None:
            raise ValueError("archetype_summary_pool must be loaded when archetype mode is enabled.")
        assigned_archetypes = _assign_summary_archetypes(
            ctx=ctx,
            mode=archetype_mode,
            seed=int(getattr(args, "seed", 0)),
            pool=archetype_summary_pool,
        )

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
    for round_idx in ctx.rounds:
        if int(round_idx) >= int(args.start_round):
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
            round_rows = build_eval_rows(ctx, round_idx, predictions, round_slice, args)
            rows.extend(round_rows)
            if on_round_rows is not None and round_rows:
                on_round_rows(round_rows)
        append_observed_round(ctx, transcripts, active_history, round_idx)
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
        "temperature",
        "top_p",
        "seed",
        "contrib_max_new_tokens",
        "actions_max_new_tokens",
        "include_reasoning",
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
    if provider != "openai":
        base_keys.extend(["base_model", "adapter_path", "use_peft"])
    model_payload = {k: args_dict.get(k) for k in base_keys if k in args_dict}
    model_payload["run_timestamp"] = run_ts
    return model_payload


def _write_config_json(config_path: str, payload: Dict[str, Any]) -> None:
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_micro_behavior_eval(args: Any) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    run_ts = getattr(args, "run_id", None) or timestamp_yymmddhhmm()
    run_dir = os.path.join(args.output_root, run_ts)
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, "config.json")

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
    debug_jsonl_path = relocate_output(args.debug_jsonl_path, run_dir) if args.debug_jsonl_path else None
    debug_full_jsonl_path = relocate_output(args.debug_full_jsonl_path, run_dir) if args.debug_full_jsonl_path else None

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

    args_payload = _serialize_args(args)
    if args_payload.get("openai_api_key") is not None:
        args_payload["openai_api_key"] = "***redacted***"

    config_payload = {
        "run_timestamp": run_ts,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "rounds_csv": args.rounds_csv,
            "analysis_csv": args.analysis_csv,
            "demographics_csv": args.demographics_csv,
            "players_csv": args.players_csv,
            "games_csv": args.games_csv,
        },
        "selection": {
            "num_games": len(contexts),
            "game_ids": [c.game_id for c in contexts],
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
            "debug": debug_jsonl_path,
            "debug_full": debug_full_jsonl_path,
        },
    }
    _write_config_json(config_path, config_payload)

    if args.max_parallel_games and int(args.max_parallel_games) > 1:
        log("[warn] max_parallel_games is currently not used in this pipeline; running sequentially.")

    archetype_mode = _resolve_archetype_mode(args)
    if archetype_mode and archetype_mode not in {"matched_summary", "random_summary"}:
        raise ValueError(
            f"Unsupported archetype mode '{archetype_mode}'. Allowed values: matched_summary, random_summary."
        )
    archetype_summary_pool: Optional[ArchetypeSummaryPool] = None
    if archetype_mode in {"matched_summary", "random_summary"}:
        archetype_summary_pool = _load_archetype_summary_pool(_resolve_archetype_pool_path(args))
        if args.debug_print:
            log(
                "[micro] loaded archetype summaries:",
                len(archetype_summary_pool.all_records),
                "from",
                _resolve_archetype_pool_path(args),
            )

    tok: Optional[Any] = None
    model: Optional[Any] = None
    provider = str(args.provider).lower()
    if provider == "local":
        tok, model = load_model(base_model=args.base_model, adapter_path=args.adapter_path, use_peft=args.use_peft)
    client = LLMClient(
        provider=provider,
        tok=tok,
        model=model,
        openai_model=args.openai_model,
        openai_api_key=args.openai_api_key,
        openai_api_key_env=args.openai_api_key_env,
    )

    all_rows: List[Dict[str, Any]] = []
    debug_records: List[Dict[str, Any]] = []
    debug_full_records: List[Dict[str, Any]] = []

    os.makedirs(os.path.dirname(rows_out_path), exist_ok=True)
    rows_file = open(rows_out_path, "w", newline="", encoding="utf-8")
    rows_writer: Optional[csv.DictWriter] = None

    transcripts_file = None
    if transcripts_out_path:
        os.makedirs(os.path.dirname(transcripts_out_path), exist_ok=True)
        transcripts_file = open(transcripts_out_path, "w", encoding="utf-8")

    debug_file = None
    if debug_jsonl_path and args.debug_level != "off":
        os.makedirs(os.path.dirname(debug_jsonl_path), exist_ok=True)
        debug_file = open(debug_jsonl_path, "w", encoding="utf-8")

    debug_full_file = None
    if debug_full_jsonl_path:
        os.makedirs(os.path.dirname(debug_full_jsonl_path), exist_ok=True)
        debug_full_file = open(debug_full_jsonl_path, "w", encoding="utf-8")

    def _flush_file(handle):
        handle.flush()
        os.fsync(handle.fileno())

    def _write_rows_chunk(chunk: List[Dict[str, Any]]):
        nonlocal rows_writer
        if not chunk:
            return
        if rows_writer is None:
            rows_writer = csv.DictWriter(rows_file, fieldnames=list(chunk[0].keys()))
            rows_writer.writeheader()
        for row in chunk:
            rows_writer.writerow(row)
        _flush_file(rows_file)

    try:
        for idx, ctx in enumerate(contexts, start=1):
            debug_records.clear()
            debug_full_records.clear()
            if args.debug_print:
                log(f"[micro] start game {ctx.game_id} ({idx}/{len(contexts)})")
            rows, transcripts = evaluate_game(
                ctx=ctx,
                client=client,
                tok=tok,
                args=args,
                archetype_summary_pool=archetype_summary_pool,
                debug_records=debug_records,
                debug_full_records=debug_full_records,
                on_round_rows=_write_rows_chunk,
            )
            all_rows.extend(rows)

            if transcripts_file is not None:
                for pid in ctx.player_ids:
                    sys_text = system_header_plain(ctx.env, ctx.demographics_by_player.get(pid, ""), args.include_reasoning)
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

            if debug_file is not None and debug_records:
                for rec in debug_records:
                    debug_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                _flush_file(debug_file)

            if debug_full_file is not None and debug_full_records:
                for rec in debug_full_records:
                    debug_full_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                _flush_file(debug_full_file)

            if args.debug_print:
                log(f"[micro] done game {ctx.game_id} -> {len(rows)} eval rows")
    finally:
        rows_file.close()
        if transcripts_file is not None:
            transcripts_file.close()
        if debug_file is not None:
            debug_file.close()
        if debug_full_file is not None:
            debug_full_file.close()

    df_out = pd.DataFrame(all_rows)

    config_payload["status"] = "completed"
    config_payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    config_payload["summary"] = {
        "num_rows": int(len(all_rows)),
        "num_games": int(len(contexts)),
    }
    _write_config_json(config_path, config_payload)

    if args.debug_print:
        log(f"[micro] wrote rows -> {rows_out_path}")
        if transcripts_out_path:
            log(f"[micro] wrote transcripts -> {transcripts_out_path}")
        if debug_jsonl_path:
            log(f"[micro] wrote debug -> {debug_jsonl_path}")
        if debug_full_jsonl_path:
            log(f"[micro] wrote full debug -> {debug_full_jsonl_path}")
        log(f"[micro] wrote config -> {config_path}")

    return df_out, {
        "rows": rows_out_path,
        "transcripts": transcripts_out_path,
        "debug": debug_jsonl_path,
        "debug_full": debug_full_jsonl_path,
        "config": config_path,
        "directory": run_dir,
    }
