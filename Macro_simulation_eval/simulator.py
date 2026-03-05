from __future__ import annotations

import ast
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

try:
    from .debug import build_debug_record, build_full_debug_record
    from .llm_client import LLMClient
    from .model_loader import load_model
    from .parsers import parse_json_response
    from .prompt_builder import (
        actions_format_line,
        actions_tag,
        build_openai_messages,
        chat_format_line,
        chat_stage_line,
        contrib_format_line,
        mech_info,
        max_tokens_reminder_line,
        peers_contributions_csv,
        redist_line,
        round_info_line,
        round_open,
        system_header_plain,
    )
    from .utils import (
        as_bool,
        demographics_line,
        is_nan,
        log,
        make_unique_avatar_map,
        normalize_avatar,
        relocate_output,
        timestamp_yymmddhhmm,
    )
except ImportError:
    from debug import build_debug_record, build_full_debug_record
    from llm_client import LLMClient
    from model_loader import load_model
    from parsers import parse_json_response
    from prompt_builder import (
        actions_format_line,
        actions_tag,
        build_openai_messages,
        chat_format_line,
        chat_stage_line,
        contrib_format_line,
        mech_info,
        max_tokens_reminder_line,
        peers_contributions_csv,
        redist_line,
        round_info_line,
        round_open,
        system_header_plain,
    )
    from utils import (
        as_bool,
        demographics_line,
        is_nan,
        log,
        make_unique_avatar_map,
        normalize_avatar,
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


SUMMARY_ARCHETYPE_INTRO = (
    "Below is an archetype summary of how you played a different PGG in the past. "
    "Be aware of this archetype as you make decisions. "
    "Recall that you're probably playing games with different people from the past, and "
    "that the exact rules of this game could differ from the ones you've played before."
)


AVATAR_POOL: List[str] = sorted(
    {
        "CHICK",
        "CHICKEN",
        "COW",
        "CROCODILE",
        "DOG",
        "DUCK",
        "ELEPHANT",
        "FROG",
        "GORILLA",
        "HORSE",
        "MONKEY",
        "MOOSE",
        "OWL",
        "PARROT",
        "PINGUIN",
        "RABBIT",
        "SLOTH",
        "SNAKE",
        "WALRUS",
        "WHALE",
    }
)


@dataclass
class GameContext:
    game_id: str
    game_name: str
    env: Dict[str, Any]
    player_ids: List[str]
    avatar_by_player: Dict[str, str]
    player_by_avatar: Dict[str, str]
    demographics_by_player: Dict[str, str]


@dataclass
class ArchetypeSummaryPool:
    all_records: List[Dict[str, str]]
    by_game_player: Dict[str, Dict[str, Dict[str, str]]]


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _format_num(value: Any) -> str:
    if value is None:
        return "NA"
    if is_nan(value):
        return "NA"
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    return str(value)


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
        "chat_max_new_tokens",
        "actions_max_new_tokens",
        "include_reasoning",
        "archetype",
        "archetype_summary_pool",
        "game_ids",
        "max_games",
        "max_parallel_games",
        "debug_level",
        "debug_compact",
    ]
    if provider != "openai":
        base_keys.extend(["base_model", "adapter_path", "use_peft"])
    payload = {k: args_dict.get(k) for k in base_keys if k in args_dict}
    payload["run_timestamp"] = run_ts
    return payload


def _write_config_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


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
            if key not in row.index:
                continue
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
        if not pid:
            continue
        out[pid] = normalize_avatar(row.get("data.avatar"))
    return out


def _parse_player_ids_cell(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                return [str(x) for x in obj if str(x).strip()]
        except Exception:
            pass
        return [x.strip() for x in s.split(",") if x.strip()]
    return []


def _build_player_ids_by_game(df_rounds: pd.DataFrame, df_analysis: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}

    if {"gameId", "playerId"}.issubset(df_rounds.columns):
        rounds = df_rounds.copy()
        rounds["gameId"] = rounds["gameId"].astype(str)
        rounds["playerId"] = rounds["playerId"].astype(str)
        rounds["__row_order"] = range(len(rounds))
        if "createdAt" in rounds.columns:
            rounds["__created_at"] = pd.to_datetime(rounds["createdAt"], errors="coerce", utc=True)
        else:
            rounds["__created_at"] = pd.NaT
        rounds = rounds.sort_values(["gameId", "__created_at", "__row_order", "playerId"], na_position="last")
        for gid, gdf in rounds.groupby("gameId", sort=True):
            ordered: List[str] = []
            seen: set[str] = set()
            for pid in gdf["playerId"].tolist():
                spid = str(pid)
                if not spid or spid in seen:
                    continue
                ordered.append(spid)
                seen.add(spid)
            if ordered:
                out[str(gid)] = ordered

    if {"gameId", "playerIds"}.issubset(df_analysis.columns):
        analysis = df_analysis.copy()
        analysis["gameId"] = analysis["gameId"].astype(str)
        analysis = analysis.drop_duplicates(subset=["gameId"], keep="first")
        for _, row in analysis.iterrows():
            gid = str(row.get("gameId"))
            if gid in out and out[gid]:
                continue
            parsed = _parse_player_ids_cell(row.get("playerIds"))
            if parsed:
                out[gid] = parsed

    return out


def _ensure_configured_player_count(
    game_id: str,
    player_ids: List[str],
    configured: int,
) -> List[str]:
    out = list(player_ids)
    if configured <= 0 or len(out) >= configured:
        return out
    missing = configured - len(out)
    for i in range(1, missing + 1):
        out.append(f"{game_id}__SYNTH_{i}")
    return out


def _build_avatar_seed_map(
    player_ids: List[str],
    avatar_lookup: Mapping[str, str],
) -> Dict[str, str]:
    seed_map: Dict[str, str] = {}
    used: set[str] = set()
    for pid in player_ids:
        av = normalize_avatar(avatar_lookup.get(pid))
        if av:
            seed_map[pid] = av
            used.add(av)
    pool_available = [av for av in AVATAR_POOL if av not in used]
    for pid in player_ids:
        if pid in seed_map:
            continue
        if pool_available:
            seed_map[pid] = pool_available.pop(0)
        else:
            seed_map[pid] = ""
    return seed_map


def _build_demographics_lookup(
    df_demographics: pd.DataFrame,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    by_game_player: Dict[Tuple[str, str], Dict[str, Any]] = {}
    by_player: Dict[str, Dict[str, Any]] = {}
    if "playerId" not in df_demographics.columns:
        return by_game_player, by_player
    for _, row in df_demographics.iterrows():
        pid = str(row.get("playerId"))
        if not pid:
            continue
        rec = row.to_dict()
        by_player[pid] = rec
        gid = row.get("gameId")
        if gid is not None and not is_nan(gid):
            by_game_player[(str(gid), pid)] = rec
    return by_game_player, by_player


def build_game_contexts(
    df_analysis: pd.DataFrame,
    df_rounds: pd.DataFrame,
    df_players: pd.DataFrame,
    df_demographics: pd.DataFrame,
) -> List[GameContext]:
    env_lookup = _build_env_lookup(df_analysis)
    avatar_lookup = _build_avatar_lookup(df_players)
    player_ids_by_game = _build_player_ids_by_game(df_rounds, df_analysis)
    demo_by_game_player, demo_by_player = _build_demographics_lookup(df_demographics)

    contexts: List[GameContext] = []
    for game_id in sorted(env_lookup.keys()):
        env = dict(env_lookup[game_id])
        configured = int(env.get("CONFIG_playerCount", 0) or 0)
        player_ids = list(player_ids_by_game.get(game_id, []))
        if configured > 0 and len(player_ids) > configured:
            player_ids = player_ids[:configured]
        if configured > 0:
            player_ids = _ensure_configured_player_count(game_id, player_ids, configured)
        if not player_ids:
            log(f"[warn] skipping gameId={game_id}; no player IDs found")
            continue

        env["CONFIG_playerCount"] = len(player_ids)
        avatar_seed_map = _build_avatar_seed_map(player_ids, avatar_lookup)
        avatar_by_player = make_unique_avatar_map(player_ids, avatar_seed_map)
        player_by_avatar = {av: pid for pid, av in avatar_by_player.items()}
        demographics_by_player: Dict[str, str] = {}
        for pid in player_ids:
            row = demo_by_game_player.get((game_id, pid))
            if row is None:
                row = demo_by_player.get(pid)
            demographics_by_player[pid] = demographics_line(row)

        contexts.append(
            GameContext(
                game_id=game_id,
                game_name=str(env.get("name", game_id)),
                env=env,
                player_ids=player_ids,
                avatar_by_player=avatar_by_player,
                player_by_avatar=player_by_avatar,
                demographics_by_player=demographics_by_player,
            )
        )
    return contexts


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
                by_game_player.setdefault(experiment, {}).setdefault(participant, entry)
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
                f"[warn] matched_summary missing archetypes for gameId={ctx.game_id}, "
                f"missing_playerIds={sample_missing}, total_missing={len(missing)}"
            )
        return {pid: game_map[pid] for pid in ctx.player_ids if pid in game_map}
    rng = random.Random(f"{seed}|{ctx.game_id}|random_summary")
    records = pool.all_records
    return {pid: records[rng.randrange(len(records))] for pid in ctx.player_ids}


def _resolve_archetype_mode(args: Any) -> str:
    mode = str(getattr(args, "archetype", None) or "").strip()
    if not mode:
        mode = str(getattr(args, "archetype_mode", None) or "").strip()
    if mode:
        if mode == "none":
            return ""
        return mode
    return str(getattr(args, "persona", None) or "").strip()


def _resolve_archetype_pool_path(args: Any) -> str:
    pool = str(getattr(args, "archetype_summary_pool", None) or "").strip()
    if pool:
        return pool
    return str(getattr(args, "persona_summary_pool", None) or "").strip()


def _empty_debug_lists() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return [], []


def _append_chat_history_line(
    transcripts: Dict[str, List[str]],
    roster: Sequence[str],
    avatar_by_pid: Mapping[str, str],
    chat_messages: Mapping[str, str],
) -> None:
    for pid, focal_avatar in avatar_by_pid.items():
        payload: List[Dict[str, str]] = []
        for speaker in roster:
            msg = str(chat_messages.get(speaker, "") or "").strip()
            if not msg:
                continue
            label = f"{speaker} (YOU)" if speaker == focal_avatar else speaker
            payload.append({"speaker": label, "text": msg})
        transcripts[pid].append(f"<CHAT_LOG>{_json_compact(payload)}</CHAT_LOG>")


def simulate_game(
    ctx: GameContext,
    client: LLMClient,
    tok: Optional[Any],
    args: Any,
    seed: Optional[int] = None,
    assigned_archetypes: Optional[Dict[str, Dict[str, str]]] = None,
    debug_records: Optional[List[Dict[str, Any]]] = None,
    debug_full_records: Optional[List[Dict[str, Any]]] = None,
    on_round_complete: Optional[
        Callable[
            [GameContext, int, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]],
            None,
        ]
    ] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    assigned_archetypes = assigned_archetypes or {}
    if debug_records is None:
        debug_records = []
    if debug_full_records is None:
        debug_full_records = []

    include_system_in_prompt = str(args.provider).lower() == "local"
    game_seed = int(seed if seed is not None else args.seed)
    debug_excerpt_chars = 200
    stop_sequences = ["\n\n"]
    include_reasoning = bool(getattr(args, "include_reasoning", False))
    game_id = ctx.game_id

    roster = [ctx.avatar_by_player[pid] for pid in ctx.player_ids]
    pid_by_avatar = {ctx.avatar_by_player[pid]: pid for pid in ctx.player_ids}
    system_text_by_pid = {
        pid: system_header_plain(ctx.env, ctx.demographics_by_player.get(pid, ""), include_reasoning)
        for pid in ctx.player_ids
    }

    transcripts: Dict[str, List[str]] = {}
    archetype_ids: Dict[str, Optional[str]] = {}
    for pid in ctx.player_ids:
        avatar = ctx.avatar_by_player[pid]
        lines: List[str] = []
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
            archetype_ids[pid] = str(archetype_record.get("participant") or "")
        else:
            archetype_ids[pid] = None
        lines.append("# GAME STARTS")
        lines.append(f"Your avatar is {avatar}.")
        transcripts[pid] = lines

    rows: List[Dict[str, Any]] = []
    actions_reasoning: Dict[str, Optional[str]] = {}
    contrib_reasoning: Dict[str, Optional[str]] = {}
    chat_reasoning: Dict[str, Optional[str]] = {}

    num_rounds = int(ctx.env.get("CONFIG_numRounds", 0) or 0)
    for r in range(1, num_rounds + 1):
        round_debug_start = len(debug_records)
        round_debug_full_start = len(debug_full_records)
        actions_reasoning.clear()
        contrib_reasoning.clear()
        chat_reasoning.clear()

        chat_messages: Dict[str, str] = {}
        chat_parsed: Dict[str, bool] = {}
        chat_on = as_bool(ctx.env.get("CONFIG_chat", False))
        if chat_on:
            chat_prompts: List[str] = []
            chat_meta: List[str] = []
            chat_messages_list: List[List[Dict[str, str]]] = []
            chat_prompt_by_pid: Dict[str, str] = {}

            for pid in ctx.player_ids:
                avatar = ctx.avatar_by_player[pid]
                transcripts[pid].append(round_open(ctx.env, r))
                chunks = transcripts[pid] + [
                    chat_stage_line(ctx.env),
                    max_tokens_reminder_line(int(args.chat_max_new_tokens)),
                    chat_format_line(include_reasoning),
                ]
                if include_system_in_prompt:
                    prompt = "\n".join([system_text_by_pid[pid]] + chunks)
                else:
                    prompt = "\n".join(chunks)
                chat_prompts.append(prompt)
                chat_meta.append(pid)
                chat_prompt_by_pid[pid] = prompt
                chat_messages_list.append(build_openai_messages(system_text_by_pid[pid], chunks))
                if args.debug_print:
                    if tok is not None:
                        tok_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
                        log(f"[macro] {game_id} r={r:02d} {avatar} CHAT prompt_tokens~{tok_len}")
                    else:
                        log(f"[macro] {game_id} r={r:02d} {avatar} CHAT prompt_chars={len(prompt)}")

            t_chat = time.perf_counter()
            chat_raw = client.generate_batch(
                prompts=chat_prompts,
                messages_list=chat_messages_list,
                stop=stop_sequences,
                max_new_tokens=int(args.chat_max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                seed=game_seed,
                async_openai=bool(args.openai_async),
                max_concurrency=int(args.openai_max_concurrency),
            )
            dt_chat = time.perf_counter() - t_chat

            for pid, gen in zip(chat_meta, chat_raw):
                avatar = ctx.avatar_by_player[pid]
                dt_per = dt_chat / max(1, len(chat_raw))
                prompt_text = chat_prompt_by_pid[pid]
                if args.debug_level != "off":
                    debug_records.append(
                        build_debug_record(
                            game_id=game_id,
                            round_idx=r,
                            agent=avatar,
                            phase="chat",
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
                            game_id=game_id,
                            round_idx=r,
                            agent=avatar,
                            phase="chat",
                            dt_sec=dt_per,
                            prompt=prompt_text,
                            raw_output=gen,
                        )
                    )
                payload, ok = parse_json_response(gen)
                msg = ""
                parsed_ok = False
                if ok and isinstance(payload, dict) and payload.get("stage") == "chat":
                    raw_msg = payload.get("chat")
                    if raw_msg is None:
                        msg = ""
                        parsed_ok = True
                    elif isinstance(raw_msg, str):
                        msg = raw_msg.strip()
                        if msg in {"", "...", "SILENT", "silence", "NONE", "none"}:
                            msg = ""
                        parsed_ok = True
                chat_messages[avatar] = msg if parsed_ok else ""
                chat_parsed[avatar] = parsed_ok
                if include_reasoning and ok and isinstance(payload, dict) and isinstance(payload.get("reasoning"), str):
                    chat_reasoning[avatar] = payload.get("reasoning")
                else:
                    chat_reasoning[avatar] = None

            _append_chat_history_line(transcripts, roster, {pid: ctx.avatar_by_player[pid] for pid in ctx.player_ids}, chat_messages)
        else:
            for pid in ctx.player_ids:
                transcripts[pid].append(round_open(ctx.env, r))
                avatar = ctx.avatar_by_player[pid]
                chat_messages[avatar] = ""
                chat_parsed[avatar] = True
                chat_reasoning[avatar] = None

        contrib_math: Dict[str, int] = {}
        contrib_rec: Dict[str, Optional[int]] = {}
        contrib_parsed: Dict[str, bool] = {}

        contrib_prompts: List[str] = []
        contrib_meta: List[str] = []
        contrib_messages_list: List[List[Dict[str, str]]] = []
        contrib_prompt_by_pid: Dict[str, str] = {}
        round_rows: List[Dict[str, Any]] = []
        for pid in ctx.player_ids:
            avatar = ctx.avatar_by_player[pid]
            chunks = transcripts[pid] + [
                round_info_line(ctx.env),
                max_tokens_reminder_line(int(args.contrib_max_new_tokens)),
                contrib_format_line(ctx.env, include_reasoning),
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
                    tok_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
                    log(f"[macro] {game_id} r={r:02d} {avatar} CONTRIB prompt_tokens~{tok_len}")
                else:
                    log(f"[macro] {game_id} r={r:02d} {avatar} CONTRIB prompt_chars={len(prompt)}")

        t0 = time.perf_counter()
        contrib_raw = client.generate_batch(
            prompts=contrib_prompts,
            messages_list=contrib_messages_list,
            stop=stop_sequences,
            max_new_tokens=int(args.contrib_max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            seed=game_seed,
            async_openai=bool(args.openai_async),
            max_concurrency=int(args.openai_max_concurrency),
        )
        dt_contrib = time.perf_counter() - t0

        endow = int(ctx.env.get("CONFIG_endowment", 0) or 0)
        all_or_nothing = as_bool(ctx.env.get("CONFIG_allOrNothing", False))
        for pid, gen in zip(contrib_meta, contrib_raw):
            avatar = ctx.avatar_by_player[pid]
            prompt_text = contrib_prompt_by_pid[pid]
            dt_per = dt_contrib / max(1, len(contrib_raw))
            if args.debug_level != "off":
                debug_records.append(
                    build_debug_record(
                        game_id=game_id,
                        round_idx=r,
                        agent=avatar,
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
                        game_id=game_id,
                        round_idx=r,
                        agent=avatar,
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
            if all_or_nothing:
                val = endow if val >= (endow // 2) else 0
            else:
                val = max(0, min(endow, int(val)))
            contrib_math[avatar] = int(val)
            contrib_rec[avatar] = int(val) if parsed_ok else None
            contrib_parsed[avatar] = parsed_ok
            if include_reasoning and ok and isinstance(payload, dict) and isinstance(payload.get("reasoning"), str):
                contrib_reasoning[avatar] = payload.get("reasoning")
            else:
                contrib_reasoning[avatar] = None
            transcripts[pid].append(f'<CONTRIB v="{int(val)}"/>')

        total_contrib = int(sum(contrib_math.values()))
        try:
            multiplied = float(ctx.env.get("CONFIG_multiplier", 0) or 0) * float(total_contrib)
        except Exception:
            multiplied = float("nan")
        active_players = len(roster)
        for pid in ctx.player_ids:
            avatar = ctx.avatar_by_player[pid]
            transcripts[pid].append(redist_line(total_contrib, multiplied, active_players))
            peers_csv, _ = peers_contributions_csv(roster, avatar, contrib_math)
            transcripts[pid].append(f"<PEERS_CONTRIBUTIONS> {peers_csv} </PEERS_CONTRIBUTIONS>")

        reward_on = as_bool(ctx.env.get("CONFIG_rewardExists", False))
        punish_on = as_bool(ctx.env.get("CONFIG_punishmentExists", False))
        actions_parsed: Dict[str, Optional[bool]] = {av: None for av in roster}
        rewards_given: Dict[str, Dict[str, int]] = {av: {} for av in roster}
        punish_given: Dict[str, Dict[str, int]] = {av: {} for av in roster}

        if reward_on or punish_on:
            tag = actions_tag(ctx.env) or "PUNISHMENT/REWARD"
            mech = mech_info(ctx.env)
            actions_prompts: List[str] = []
            actions_meta: List[str] = []
            actions_messages_list: List[List[Dict[str, str]]] = []
            actions_prompt_by_pid: Dict[str, str] = {}
            peer_orders: Dict[str, List[str]] = {}

            for pid in ctx.player_ids:
                avatar = ctx.avatar_by_player[pid]
                peer_order = [x for x in roster if x != avatar]
                peer_orders[avatar] = peer_order
                chunks = list(transcripts[pid])
                if mech:
                    chunks.append(mech)
                chunks.extend(
                    [
                        max_tokens_reminder_line(int(args.actions_max_new_tokens)),
                        actions_format_line(tag, include_reasoning),
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
                        tok_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
                        log(f"[macro] {game_id} r={r:02d} {avatar} ACTIONS prompt_tokens~{tok_len}")
                    else:
                        log(f"[macro] {game_id} r={r:02d} {avatar} ACTIONS prompt_chars={len(prompt)}")

            t1 = time.perf_counter()
            actions_raw = client.generate_batch(
                prompts=actions_prompts,
                messages_list=actions_messages_list,
                stop=stop_sequences,
                max_new_tokens=int(args.actions_max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                seed=game_seed,
                async_openai=bool(args.openai_async),
                max_concurrency=int(args.openai_max_concurrency),
            )
            dt_actions = time.perf_counter() - t1

            for pid, raw in zip(actions_meta, actions_raw):
                avatar = ctx.avatar_by_player[pid]
                prompt_text = actions_prompt_by_pid[pid]
                dt_per = dt_actions / max(1, len(actions_raw))
                if args.debug_level != "off":
                    debug_records.append(
                        build_debug_record(
                            game_id=game_id,
                            round_idx=r,
                            agent=avatar,
                            phase="actions",
                            dt_sec=dt_per,
                            prompt=prompt_text,
                            raw_output=raw,
                            debug_level=args.debug_level,
                            excerpt_chars=debug_excerpt_chars,
                        )
                    )
                if args.debug_full_jsonl_path:
                    debug_full_records.append(
                        build_full_debug_record(
                            game_id=game_id,
                            round_idx=r,
                            agent=avatar,
                            phase="actions",
                            dt_sec=dt_per,
                            prompt=prompt_text,
                            raw_output=raw,
                        )
                    )

                payload, ok = parse_json_response(raw)
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
                actions_parsed[avatar] = parsed_ok

                peer_order = peer_orders[avatar]
                arr: List[int] = []
                for peer in peer_order:
                    raw_v = actions_dict.get(peer, 0)
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

                if include_reasoning and ok and isinstance(payload, dict) and isinstance(payload.get("reasoning"), str):
                    actions_reasoning[avatar] = payload.get("reasoning")
                else:
                    actions_reasoning[avatar] = None

                actions_out = {peer_order[i]: int(arr[i]) for i in range(len(peer_order)) if int(arr[i]) != 0}
                if reward_on and not punish_on:
                    reward_out = {a: int(v) for a, v in actions_out.items() if int(v) > 0}
                    punish_out: Dict[str, int] = {}
                    transcripts[pid].append(f"<REWARD>{_json_compact(reward_out)}</REWARD>")
                elif punish_on and not reward_on:
                    punish_out = {a: int(v) for a, v in actions_out.items() if int(v) > 0}
                    reward_out = {}
                    transcripts[pid].append(f"<PUNISHMENT>{_json_compact(punish_out)}</PUNISHMENT>")
                else:
                    punish_out = {a: int(abs(v)) for a, v in actions_out.items() if int(v) < 0}
                    reward_out = {a: int(v) for a, v in actions_out.items() if int(v) > 0}
                    transcripts[pid].append(
                        f'<ACTIONS punish="{_json_compact(punish_out)}" reward="{_json_compact(reward_out)}"/>'
                    )

                if reward_on:
                    for tgt, units in reward_out.items():
                        if units > 0:
                            rewards_given[avatar][tgt] = int(units)
                if punish_on:
                    for tgt, units in punish_out.items():
                        if units > 0:
                            punish_given[avatar][tgt] = int(units)
        else:
            for av in roster:
                actions_reasoning[av] = None

        show_punish_id = as_bool(ctx.env.get("CONFIG_showPunishmentId", False))
        show_reward_id = as_bool(ctx.env.get("CONFIG_showRewardId", False))
        show_other = as_bool(ctx.env.get("CONFIG_showOtherSummaries", False))

        inbound_reward_units = {av: 0 for av in roster}
        inbound_punish_units = {av: 0 for av in roster}
        for src in roster:
            for tgt, units in rewards_given[src].items():
                inbound_reward_units[tgt] += int(units)
            for tgt, units in punish_given[src].items():
                inbound_punish_units[tgt] += int(units)

        num_players = len(roster)
        share = (float(multiplied) / num_players) if (not math.isnan(float(multiplied)) and num_players > 0) else 0.0

        for pid in ctx.player_ids:
            avatar = ctx.avatar_by_player[pid]
            if show_punish_id and punish_on:
                punishers = {
                    src: u
                    for src in roster
                    for tgt, u in punish_given[src].items()
                    if tgt == avatar and int(u) > 0
                }
                transcripts[pid].append(f"<PUNISHED_BY>{_json_compact(punishers)}</PUNISHED_BY>")
            if show_reward_id and reward_on:
                rewarders = {
                    src: u
                    for src in roster
                    for tgt, u in rewards_given[src].items()
                    if tgt == avatar and int(u) > 0
                }
                transcripts[pid].append(f"<REWARDED_BY>{_json_compact(rewarders)}</REWARDED_BY>")

            spent_pun_units = sum(punish_given[avatar].values()) if punish_on else 0
            spent_rew_units = sum(rewards_given[avatar].values()) if reward_on else 0
            you: Dict[str, Any] = {}
            if punish_on:
                you["coins_spent_on_punish"] = spent_pun_units * int(ctx.env.get("CONFIG_punishmentCost", 0) or 0)
                you["coins_deducted_from_you"] = inbound_punish_units[avatar] * int(
                    ctx.env.get("CONFIG_punishmentMagnitude", 0) or 0
                )
            if reward_on:
                you["coins_spent_on_reward"] = spent_rew_units * int(ctx.env.get("CONFIG_rewardCost", 0) or 0)
                you["coins_rewarded_to_you"] = inbound_reward_units[avatar] * int(
                    ctx.env.get("CONFIG_rewardMagnitude", 0) or 0
                )

            private_kept = endow - int(contrib_math.get(avatar, 0))
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
                for other in roster:
                    if other == avatar:
                        continue
                    ob: Dict[str, Any] = {}
                    if punish_on:
                        o_pun_units = sum(punish_given[other].values())
                        ob["coins_spent_on_punish"] = o_pun_units * int(ctx.env.get("CONFIG_punishmentCost", 0) or 0)
                        ob["coins_deducted_from_them"] = (
                            sum(punish_given[src].get(other, 0) for src in roster)
                            * int(ctx.env.get("CONFIG_punishmentMagnitude", 0) or 0)
                        )
                    if reward_on:
                        o_rew_units = sum(rewards_given[other].values())
                        ob["coins_spent_on_reward"] = o_rew_units * int(ctx.env.get("CONFIG_rewardCost", 0) or 0)
                        ob["coins_rewarded_to_them"] = (
                            sum(rewards_given[src].get(other, 0) for src in roster)
                            * int(ctx.env.get("CONFIG_rewardMagnitude", 0) or 0)
                        )
                    private_kept_other = endow - int(contrib_math.get(other, 0))
                    payoff_other = (
                        private_kept_other
                        + share
                        - ob.get("coins_spent_on_punish", 0)
                        - ob.get("coins_spent_on_reward", 0)
                        - ob.get("coins_deducted_from_them", 0)
                        + ob.get("coins_rewarded_to_them", 0)
                    )
                    ob["payoff"] = int(payoff_other)
                    summary[other] = ob

            transcripts[pid].append(f"<ROUND_SUMMARY>{_json_compact(summary)}</ROUND_SUMMARY>")
            transcripts[pid].append("</ROUND>")

            rewarded_str = _json_compact(rewards_given[avatar]) if reward_on else None
            punished_str = _json_compact(punish_given[avatar]) if punish_on else None
            round_rows.append(
                {
                    "gameId": ctx.game_id,
                    "gameName": ctx.game_name,
                    "roundIndex": r,
                    "playerId": pid,
                    "playerAvatar": avatar,
                    "archetype": archetype_ids.get(pid),
                    "persona": archetype_ids.get(pid),  # legacy alias
                    "demographics": ctx.demographics_by_player.get(pid, ""),
                    "data.chat_message": chat_messages.get(avatar, "") if chat_on else "",
                    "data.chat_parsed": chat_parsed.get(avatar),
                    "data.chat_reasoning": chat_reasoning.get(avatar),
                    "data.contribution": contrib_rec.get(avatar),
                    "data.contribution_clamped": contrib_math.get(avatar),
                    "data.contribution_parsed": contrib_parsed.get(avatar),
                    "data.contribution_reasoning": contrib_reasoning.get(avatar),
                    "data.punished": punished_str,
                    "data.rewarded": rewarded_str,
                    "data.actions_parsed": actions_parsed.get(avatar),
                    "data.actions_reasoning": actions_reasoning.get(avatar),
                }
            )
        rows.extend(round_rows)

        if on_round_complete is not None and round_rows:
            round_debug_records = debug_records[round_debug_start:]
            round_debug_full_records = debug_full_records[round_debug_full_start:]
            on_round_complete(ctx, r, round_rows, round_debug_records, round_debug_full_records)

    for pid in ctx.player_ids:
        transcripts[pid].append("# GAME COMPLETE")

    return rows, transcripts


def run_macro_simulation_eval(args: Any) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
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
    transcripts_out_path = relocate_output(args.transcripts_out_path, run_dir) if args.transcripts_out_path else None
    debug_jsonl_path = relocate_output(args.debug_jsonl_path, run_dir) if args.debug_jsonl_path else None
    debug_full_jsonl_path = relocate_output(args.debug_full_jsonl_path, run_dir) if args.debug_full_jsonl_path else None

    df_analysis = pd.read_csv(args.analysis_csv)
    df_rounds = pd.read_csv(args.rounds_csv)
    df_players = pd.read_csv(args.players_csv) if args.players_csv and os.path.exists(args.players_csv) else pd.DataFrame()
    df_demographics = (
        pd.read_csv(args.demographics_csv)
        if args.demographics_csv and os.path.exists(args.demographics_csv)
        else pd.DataFrame()
    )

    contexts = build_game_contexts(
        df_analysis=df_analysis,
        df_rounds=df_rounds,
        df_players=df_players,
        df_demographics=df_demographics,
    )
    if args.game_ids:
        wanted = {x.strip() for x in str(args.game_ids).split(",") if x.strip()}
        contexts = [c for c in contexts if c.game_id in wanted or c.game_name in wanted]
    if args.max_games is not None:
        contexts = contexts[: int(args.max_games)]
    if not contexts:
        raise ValueError("No games selected for macro simulation.")

    args_payload = _serialize_args(args)
    if args_payload.get("openai_api_key") is not None:
        args_payload["openai_api_key"] = "***redacted***"

    config_payload = {
        "run_timestamp": run_ts,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "analysis_csv": args.analysis_csv,
            "rounds_csv": args.rounds_csv,
            "players_csv": args.players_csv,
            "demographics_csv": args.demographics_csv,
        },
        "selection": {
            "num_games": len(contexts),
            "game_ids": [c.game_id for c in contexts],
            "requested_game_ids": args.game_ids,
            "max_games": args.max_games,
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
        log("[warn] max_parallel_games is currently not used in macro simulation; running sequentially.")

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
                "[macro] loaded archetype summaries:",
                len(archetype_summary_pool.all_records),
                "from",
                _resolve_archetype_pool_path(args),
            )

    provider = str(args.provider).lower()
    tok: Optional[Any] = None
    model: Optional[Any] = None
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
    debug_records, debug_full_records = _empty_debug_lists()

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

    def _flush_file(handle: Any) -> None:
        handle.flush()
        os.fsync(handle.fileno())

    def _write_rows_chunk(chunk: List[Dict[str, Any]]) -> None:
        nonlocal rows_writer
        if not chunk:
            return
        if rows_writer is None:
            rows_writer = csv.DictWriter(rows_file, fieldnames=list(chunk[0].keys()))
            rows_writer.writeheader()
        for row in chunk:
            rows_writer.writerow(row)
        _flush_file(rows_file)

    def _write_debug_chunk(chunk: List[Dict[str, Any]]) -> None:
        if debug_file is None or not chunk:
            return
        for rec in chunk:
            debug_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _flush_file(debug_file)

    def _write_debug_full_chunk(chunk: List[Dict[str, Any]]) -> None:
        if debug_full_file is None or not chunk:
            return
        for rec in chunk:
            debug_full_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _flush_file(debug_full_file)

    def _on_round_complete(
        _ctx: GameContext,
        _round_idx: int,
        round_rows: List[Dict[str, Any]],
        round_dbg: List[Dict[str, Any]],
        round_dbg_full: List[Dict[str, Any]],
    ) -> None:
        _write_rows_chunk(round_rows)
        _write_debug_chunk(round_dbg)
        _write_debug_full_chunk(round_dbg_full)

    try:
        game_seed_rng = random.Random(int(args.seed))
        for idx, ctx in enumerate(contexts, start=1):
            if args.debug_print:
                log(f"[macro] start game {ctx.game_id} ({idx}/{len(contexts)})")

            debug_records.clear()
            debug_full_records.clear()
            game_seed = game_seed_rng.randrange(0, 2**32 - 1)

            assigned: Dict[str, Dict[str, str]] = {}
            if archetype_mode:
                assert archetype_summary_pool is not None
                assigned = _assign_summary_archetypes(
                    ctx=ctx,
                    mode=archetype_mode,
                    seed=game_seed,
                    pool=archetype_summary_pool,
                )

            rows, transcripts = simulate_game(
                ctx=ctx,
                client=client,
                tok=tok,
                args=args,
                seed=game_seed,
                assigned_archetypes=assigned,
                debug_records=debug_records,
                debug_full_records=debug_full_records,
                on_round_complete=_on_round_complete,
            )
            all_rows.extend(rows)

            if transcripts_file is not None:
                for pid in ctx.player_ids:
                    sys_text = system_header_plain(
                        ctx.env, ctx.demographics_by_player.get(pid, ""), bool(args.include_reasoning)
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

            if args.debug_print:
                log(f"[macro] done game {ctx.game_id} -> {len(rows)} rows")
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
        log(f"[macro] wrote rows -> {rows_out_path}")
        if transcripts_out_path:
            log(f"[macro] wrote transcripts -> {transcripts_out_path}")
        if debug_jsonl_path:
            log(f"[macro] wrote debug -> {debug_jsonl_path}")
        if debug_full_jsonl_path:
            log(f"[macro] wrote full debug -> {debug_full_jsonl_path}")
        log(f"[macro] wrote config -> {config_path}")

    return df_out, {
        "rows": rows_out_path,
        "transcripts": transcripts_out_path,
        "debug": debug_jsonl_path,
        "debug_full": debug_full_jsonl_path,
        "config": config_path,
        "directory": run_dir,
    }
