from __future__ import annotations

import concurrent.futures
import csv
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from debug import build_debug_record, build_full_debug_record
from llm_client import LLMClient
from output_manager import relocate_for_experiment, resolve_experiment_dir, resolve_run_ts, write_config
from parsers import first_int, parse_chat_message, parse_int_dict
from prompt_builder import (
    actions_format_line,
    actions_tag,
    build_openai_messages,
    chat_format_line,
    chat_stage_line,
    contrib_format_line,
    extract_reasoning,
    format_actions_answer,
    format_contrib_answer,
    mech_info,
    max_tokens_reminder_line,
    peers_contributions_csv,
    redist_line,
    round_info_line,
    round_open,
    system_header_plain,
)
from utils import log


AVATAR_POOL = {
    "chick",
    "chicken",
    "cow",
    "crocodile",
    "dog",
    "duck",
    "elephant",
    "frog",
    "gorilla",
    "horse",
    "monkey",
    "moose",
    "owl",
    "parrot",
    "pinguin",
    "rabbit",
    "sloth",
    "snake",
    "walrus",
    "whale",
}


def sample_roster(env: pd.Series, seed: int = 0) -> List[str]:
    rng = random.Random(seed)
    pool = [a.upper() for a in AVATAR_POOL]
    n = int(env["CONFIG_playerCount"])
    if n > len(pool):
        raise ValueError(f"ENV.players={n} exceeds avatar pool size={len(pool)}")
    return rng.sample(pool, k=n)


def simulate_game(
    env: pd.Series,
    client: LLMClient,
    tok: Optional[Any],
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 0,
    contrib_max_new_tokens: int = 12,
    chat_max_new_tokens: int = 96,
    actions_max_new_tokens: int = 192,
    include_reasoning: bool = False,
    rows_out_path: Optional[str] = None,
    transcripts_out_path: Optional[str] = None,
    debug_jsonl_path: Optional[str] = None,
    debug_print: bool = True,
    debug_level: str = "full",
    debug_full_jsonl_path: Optional[str] = None,
    openai_async: bool = False,
    openai_max_concurrency: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    debug_records: List[Dict[str, Any]] = []
    debug_full_records: List[Dict[str, Any]] = []
    debug_excerpt_chars = 200

    roster = sample_roster(env, seed=seed)
    assert len(roster) == env["CONFIG_playerCount"], "Roster length must match ENV.players"

    sys_text_plain = system_header_plain(env, include_reasoning=include_reasoning)
    include_system_in_prompt = client.provider == "local"

    transcripts: Dict[str, List[str]] = {}
    rows: List[Dict[str, Any]] = []
    game_id = env.get("name", "GAME")

    for av in roster:
        transcripts[av] = ["# GAME STARTS"]

    csv_writer = None
    csv_file = None
    if rows_out_path:
        os.makedirs(os.path.dirname(rows_out_path), exist_ok=True)
        csv_file = open(rows_out_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "playerAvatar",
                "roundIndex",
                "gameId",
                "data.chat_message",
                "data.chat_reasoning",
                "data.contribution",
                "data.contribution_reasoning",
                "data.punished",
                "data.rewarded",
                "data.actions_reasoning",
            ],
        )
        csv_writer.writeheader()

    contrib_math: Dict[str, int] = {}
    contrib_rec: Dict[str, Any] = {}
    contrib_reasoning: Dict[str, Optional[str]] = {}
    actions_reasoning: Dict[str, Optional[str]] = {}
    chat_reasoning: Dict[str, Optional[str]] = {}
    chat_messages: Dict[str, str] = {}

    for r in range(1, int(env["CONFIG_numRounds"]) + 1):
        actions_reasoning.clear()
        chat_reasoning.clear()
        chat_messages.clear()

        chat_on = bool(env.get("CONFIG_chat", False))
        if chat_on:
            chat_prompts: List[str] = []
            chat_meta: List[str] = []
            chat_messages_list: List[List[Dict[str, str]]] = []

            for av in roster:
                transcripts[av].append(round_open(env, r))
                transcripts[av].append(chat_stage_line(env))

                chat_chunks = transcripts[av] + [
                    chat_format_line(include_reasoning),
                    max_tokens_reminder_line(chat_max_new_tokens),
                ]
                if include_system_in_prompt:
                    chat_chunks = [sys_text_plain] + chat_chunks
                chat_prompt = "\n".join(chat_chunks)
                chat_prompts.append(chat_prompt)
                chat_meta.append(av)
                chat_messages_list.append(
                    build_openai_messages(
                        sys_text_plain,
                        transcripts[av]
                        + [
                            chat_format_line(include_reasoning),
                            max_tokens_reminder_line(chat_max_new_tokens),
                        ],
                    )
                )

                if debug_print:
                    if tok is not None:
                        tok_len = len(tok(chat_prompt, add_special_tokens=False)["input_ids"])
                        log(f"[ptc] {game_id} r={r:02d} {av} CHAT prompt_tokens≈{tok_len}")
                    else:
                        log(f"[ptc] {game_id} r={r:02d} {av} CHAT prompt_chars={len(chat_prompt)}")
                    log("----- PROMPT (CHAT) -----")
                    log(chat_prompt)

            t_chat = time.perf_counter()
            chat_raw = client.generate_batch(
                prompts=chat_prompts,
                messages_list=chat_messages_list,
                stop=["</CHAT>"],
                max_new_tokens=chat_max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                async_openai=openai_async,
                max_concurrency=openai_max_concurrency,
            )
            dt_chat = time.perf_counter() - t_chat

            for av, gen in zip(chat_meta, chat_raw):
                if debug_print:
                    log(f"[ptc] {game_id} r={r:02d} {av} CHAT dt={dt_chat/len(chat_raw):.3f}s out='{gen}'")
                prompt_text = chat_prompts[chat_meta.index(av)]
                dt_per_agent = dt_chat / len(chat_raw)
                if debug_level != "off":
                    debug_records.append(
                        build_debug_record(
                            game_id=game_id,
                            round_idx=r,
                            agent=av,
                            phase="chat",
                            dt_sec=dt_per_agent,
                            prompt=prompt_text,
                            raw_output=gen,
                            debug_level=debug_level,
                            excerpt_chars=debug_excerpt_chars,
                        )
                    )
                if debug_full_jsonl_path:
                    debug_full_records.append(
                        build_full_debug_record(
                            game_id=game_id,
                            round_idx=r,
                            agent=av,
                            phase="chat",
                            dt_sec=dt_per_agent,
                            prompt=prompt_text,
                            raw_output=gen,
                        )
                    )

                msg, parsed_ok = parse_chat_message(gen)
                chat_messages[av] = msg if parsed_ok else ""
                if include_reasoning:
                    chat_reasoning[av] = extract_reasoning(gen)
                    transcripts[av].append(f"<Reasoning> {chat_reasoning[av]} </Reasoning>")
                else:
                    chat_reasoning[av] = None
                if msg:
                    transcripts[av].append(f"<CHAT> {msg} </CHAT>")

            chat_lines = [f"{av}: {msg}" for av, msg in chat_messages.items() if msg]
            chat_block = (
                "<CHAT_TRANSCRIPT>\n" + "\n".join(chat_lines) + "\n</CHAT_TRANSCRIPT>"
                if chat_lines
                else "<CHAT_TRANSCRIPT> (no messages) </CHAT_TRANSCRIPT>"
            )
            for av in roster:
                transcripts[av].append(chat_block)
                transcripts[av].append(round_info_line(env))
        else:
            for av in roster:
                transcripts[av].append(round_open(env, r))
                transcripts[av].append(round_info_line(env))

        contrib_prompts: List[str] = []
        contrib_meta: List[str] = []
        contrib_messages: List[List[Dict[str, str]]] = []
        for av in roster:
            contrib_chunks = transcripts[av] + [
                contrib_format_line(include_reasoning),
                max_tokens_reminder_line(contrib_max_new_tokens),
            ]
            if include_system_in_prompt:
                contrib_chunks = [sys_text_plain] + contrib_chunks
            prompt = "\n".join(contrib_chunks)
            contrib_prompts.append(prompt)
            contrib_meta.append(av)
            contrib_messages.append(
                build_openai_messages(
                    sys_text_plain,
                    transcripts[av]
                    + [
                        contrib_format_line(include_reasoning),
                        max_tokens_reminder_line(contrib_max_new_tokens),
                    ],
                )
            )

        for av, ptxt in zip(contrib_meta, contrib_prompts):
            if debug_print:
                if tok is not None:
                    tok_len = len(tok(ptxt, add_special_tokens=False)["input_ids"])
                    log(f"[ptc] {game_id} r={r:02d} {av} CONTRIB prompt_tokens≈{tok_len}")
                else:
                    log(f"[ptc] {game_id} r={r:02d} {av} CONTRIB prompt_chars={len(ptxt)}")
                log("----- PROMPT (CONTRIB) -----")
                log(ptxt)

        t0 = time.perf_counter()
        contrib_raw = client.generate_batch(
            prompts=contrib_prompts,
            messages_list=contrib_messages,
            stop=["</CONTRIB>"],
            max_new_tokens=contrib_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            async_openai=openai_async,
            max_concurrency=openai_max_concurrency,
        )
        dt = time.perf_counter() - t0

        contrib_math.clear()
        contrib_rec.clear()
        contrib_reasoning.clear()
        endow = int(env["CONFIG_endowment"])
        for av, gen in zip(contrib_meta, contrib_raw):
            if debug_print:
                log(f"[ptc] {game_id} r={r:02d} {av} CONTRIB dt={dt/len(contrib_raw):.3f}s (avg per agent) out='{gen}'")
            prompt_text = contrib_prompts[contrib_meta.index(av)]
            dt_per_agent = dt / len(contrib_raw)
            if debug_level != "off":
                debug_records.append(
                    build_debug_record(
                        game_id=game_id,
                        round_idx=r,
                        agent=av,
                        phase="contrib",
                        dt_sec=dt_per_agent,
                        prompt=prompt_text,
                        raw_output=gen,
                        debug_level=debug_level,
                        excerpt_chars=debug_excerpt_chars,
                    )
                )
            if debug_full_jsonl_path:
                debug_full_records.append(
                    build_full_debug_record(
                        game_id=game_id,
                        round_idx=r,
                        agent=av,
                        phase="contrib",
                        dt_sec=dt_per_agent,
                        prompt=prompt_text,
                        raw_output=gen,
                    )
                )
            val, parsed_ok = first_int(gen, tag="CONTRIB")
            if env.get("CONFIG_allOrNothing", False):
                val = endow if val >= (endow // 2) else 0
            else:
                val = max(0, min(endow, val))
            contrib_math[av] = int(val)
            contrib_rec[av] = float("nan") if not parsed_ok else int(val)
            if include_reasoning:
                contrib_reasoning[av] = extract_reasoning(gen)
                transcripts[av].append(f"<Reasoning> {contrib_reasoning[av]} </Reasoning>")
            else:
                contrib_reasoning[av] = None
            transcripts[av].append(format_contrib_answer(contrib_rec[av] if parsed_ok else "NaN"))

        total_contrib = sum(contrib_math.values())
        try:
            multiplied = float(env["CONFIG_multiplier"]) * float(total_contrib)
        except Exception:
            multiplied = float("nan")

        active_players = len(roster)
        for av in roster:
            transcripts[av].append(redist_line(total_contrib, multiplied, active_players))
            peers_csv, peer_order = peers_contributions_csv(roster, av, contrib_math)
            transcripts[av].append(f"<PEERS_CONTRIBUTIONS> {peers_csv} </PEERS_CONTRIBUTIONS>")

        reward_on = env.get("CONFIG_rewardExists", False)
        punish_on = env.get("CONFIG_punishmentExists", False)
        rewards_given: Dict[str, Dict[str, int]] = {av: {} for av in roster}
        punish_given: Dict[str, Dict[str, int]] = {av: {} for av in roster}

        if reward_on or punish_on:
            actions_prompts: List[str] = []
            actions_meta: List[str] = []
            peer_orders: Dict[str, List[str]] = {}
            actions_messages: List[List[Dict[str, str]]] = []
            tag = actions_tag(env)
            mech = mech_info(env)

            for av in roster:
                peer_order = [x for x in roster if x != av]
                peer_orders[av] = peer_order

                if mech:
                    transcripts[av].append(f"<MECHANISM_INFO> {mech} </MECHANISM_INFO>")

                actions_chunks = transcripts[av] + [
                    actions_format_line(tag, include_reasoning),
                    max_tokens_reminder_line(actions_max_new_tokens),
                ]
                if include_system_in_prompt:
                    actions_chunks = [sys_text_plain] + actions_chunks
                prompt = "\n".join(actions_chunks)
                actions_prompts.append(prompt)
                actions_meta.append(av)
                actions_messages.append(
                    build_openai_messages(
                        sys_text_plain,
                        transcripts[av]
                        + [
                            actions_format_line(tag, include_reasoning),
                            max_tokens_reminder_line(actions_max_new_tokens),
                        ],
                    )
                )
                if debug_print:
                    if tok is not None:
                        tok_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
                        log(f"[ptc] {game_id} r={r:02d} {av} ACTIONS prompt_tokens≈{tok_len}")
                    else:
                        log(f"[ptc] {game_id} r={r:02d} {av} ACTIONS prompt_chars={len(prompt)}")
                    log("----- PROMPT (ACTIONS) -----")
                    log(prompt)

            t1 = time.perf_counter()
            actions_raw = client.generate_batch(
                prompts=actions_prompts,
                messages_list=actions_messages,
                stop=[f"</{tag}>"],
                max_new_tokens=actions_max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                async_openai=openai_async,
                max_concurrency=openai_max_concurrency,
            )
            dt_actions = time.perf_counter() - t1

            for av, gen in zip(actions_meta, actions_raw):
                raw = gen
                if "{" in raw and "}" not in raw:
                    raw = raw + "}"
                if debug_print:
                    log(f"[ptc] {game_id} r={r:02d} {av} ACTIONS dt={dt_actions/len(actions_raw):.3f}s out='{raw}'")
                prompt_text = actions_prompts[actions_meta.index(av)]
                dt_per_agent = dt_actions / len(actions_raw)
                if debug_level != "off":
                    debug_records.append(
                        build_debug_record(
                            game_id=game_id,
                            round_idx=r,
                            agent=av,
                            phase="actions",
                            dt_sec=dt_per_agent,
                            prompt=prompt_text,
                            raw_output=raw,
                            debug_level=debug_level,
                            excerpt_chars=debug_excerpt_chars,
                        )
                    )
                if debug_full_jsonl_path:
                    debug_full_records.append(
                        build_full_debug_record(
                            game_id=game_id,
                            round_idx=r,
                            agent=av,
                            phase="actions",
                            dt_sec=dt_per_agent,
                            prompt=prompt_text,
                            raw_output=raw,
                        )
                    )

                actions_dict, parsed_ok = parse_int_dict(raw, tag=tag)
                if actions_dict is None:
                    actions_dict = {}
                peer_order = peer_orders[av]
                arr = [actions_dict.get(peer, 0) for peer in peer_order]

                if reward_on and not punish_on:
                    arr = [max(0, int(v)) for v in arr]
                elif punish_on and not reward_on:
                    arr = [max(0, int(v)) for v in arr]
                else:
                    arr = [int(v) for v in arr]

                if include_reasoning:
                    actions_reasoning[av] = extract_reasoning(gen)
                    transcripts[av].append(f"<Reasoning> {actions_reasoning[av]} </Reasoning>")
                else:
                    actions_reasoning[av] = None
                actions_out = {peer: int(arr[idx]) for idx, peer in enumerate(peer_order) if int(arr[idx]) != 0}
                transcripts[av].append(format_actions_answer(tag, actions_out))

                if reward_on:
                    for j, v in enumerate(arr):
                        if v > 0:
                            tgt = peer_order[j]
                            rewards_given[av][tgt] = int(v)
                if punish_on:
                    for j, v in enumerate(arr):
                        if (reward_on and v < 0) or (not reward_on and v > 0):
                            tgt = peer_order[j]
                            punish_given[av][tgt] = int(abs(v))

        show_punish_id = env.get("CONFIG_showPunishmentId", False)
        show_reward_id = env.get("CONFIG_showRewardId", False)

        inbound_reward_units = {av: 0 for av in roster}
        inbound_punish_units = {av: 0 for av in roster}
        for src in roster:
            for tgt, u in rewards_given[src].items():
                inbound_reward_units[tgt] += int(u)
            for tgt, u in punish_given[src].items():
                inbound_punish_units[tgt] += int(u)

        num_players = int(env["CONFIG_playerCount"])
        share = (float(multiplied) / num_players) if (isinstance(multiplied, (int, float)) and num_players > 0) else 0.0

        for av in roster:
            if show_punish_id and env.get("CONFIG_punishmentExists", False):
                punishers = {src: u for src in roster for (tgt, u) in punish_given[src].items() if tgt == av and u > 0}
                if punishers:
                    transcripts[av].append(f"<PUNISHED_BY json='{json.dumps(punishers, separators=(',', ':'))}'/>")
            if show_reward_id and env.get("CONFIG_rewardExists", False):
                rewarders = {src: u for src in roster for (tgt, u) in rewards_given[src].items() if tgt == av and u > 0}
                if rewarders:
                    transcripts[av].append(f"<REWARDED_BY json='{json.dumps(rewarders, separators=(',', ':'))}'/>")

        for av in roster:
            f_pun_units = sum(punish_given[av].values()) if env.get("CONFIG_punishmentExists", False) else 0
            f_rew_units = sum(rewards_given[av].values()) if env.get("CONFIG_rewardExists", False) else 0

            you_dict: Dict[str, Any] = {}
            if env.get("CONFIG_punishmentExists", False):
                you_dict["coins_spent_on_punish"] = f_pun_units * int(env.get("CONFIG_punishmentCost", 0) or 0)
                you_dict["coins_deducted_from_you"] = inbound_punish_units[av] * int(env.get("CONFIG_punishmentMagnitude", 0) or 0)
            if env.get("CONFIG_rewardExists", False):
                you_dict["coins_spent_on_reward"] = f_rew_units * int(env.get("CONFIG_rewardCost", 0) or 0)
                you_dict["coins_rewarded_to_you"] = inbound_reward_units[av] * int(env.get("CONFIG_rewardMagnitude", 0) or 0)

            endow = int(env["CONFIG_endowment"])
            private_kept = endow - (0 if (isinstance(contrib_rec[av], float) and math.isnan(contrib_rec[av])) else int(contrib_rec[av]))
            coins_spent_on_punish = you_dict.get("coins_spent_on_punish", 0)
            coins_spent_on_reward = you_dict.get("coins_spent_on_reward", 0)
            coins_deducted_from_you = you_dict.get("coins_deducted_from_you", 0)
            coins_rewarded_to_you = you_dict.get("coins_rewarded_to_you", 0)
            payoff = (
                private_kept
                + share
                - coins_spent_on_punish
                - coins_spent_on_reward
                - coins_deducted_from_you
                + coins_rewarded_to_you
            )
            you_dict["payoff"] = int(payoff)

            summary_obj = {f"{av} (YOU)": you_dict}

            if env.get("CONFIG_showOtherSummaries", False):
                others_block: Dict[str, Dict[str, int]] = {}
                for other in roster:
                    if other == av:
                        continue
                    ob: Dict[str, int] = {}
                    if env.get("CONFIG_punishmentExists", False):
                        o_pun_units = sum(punish_given[other].values())
                        ob["coins_spent_on_punish"] = o_pun_units * int(env.get("CONFIG_punishmentCost", 0) or 0)
                        coins_deducted_from_them = (
                            sum(punish_given[src].get(other, 0) for src in roster)
                            * int(env.get("CONFIG_punishmentMagnitude", 0) or 0)
                        )
                        ob["coins_deducted_from_them"] = coins_deducted_from_them
                    if env.get("CONFIG_rewardExists", False):
                        o_rew_units = sum(rewards_given[other].values())
                        ob["coins_spent_on_reward"] = o_rew_units * int(env.get("CONFIG_rewardCost", 0) or 0)
                        coins_rewarded_to_them = (
                            sum(rewards_given[src].get(other, 0) for src in roster)
                            * int(env.get("CONFIG_rewardMagnitude", 0) or 0)
                        )
                        ob["coins_rewarded_to_them"] = coins_rewarded_to_them

                    endow = int(env["CONFIG_endowment"])
                    private_kept_other = endow - int(contrib_math[other])
                    spend_pun_other = ob.get("coins_spent_on_punish", 0)
                    spend_rew_other = ob.get("coins_spent_on_reward", 0)
                    payoff_other = (
                        private_kept_other
                        + share
                        - spend_pun_other
                        - spend_rew_other
                        - ob.get("coins_deducted_from_them", 0)
                        + ob.get("coins_rewarded_to_them", 0)
                    )
                    ob["payoff"] = int(payoff_other)
                    others_block[other] = ob

                summary_obj.update(others_block)

            transcripts[av].append(f"<ROUND SUMMARY json='{json.dumps(summary_obj, separators=(',', ':'))}'/>")
            transcripts[av].append("</ROUND>")

        for av in roster:
            punished_str = (
                json.dumps(punish_given[av], separators=(",", ":")) if env.get("CONFIG_punishmentExists", False) else None
            )
            rewarded_str = (
                json.dumps(rewards_given[av], separators=(",", ":")) if env.get("CONFIG_rewardExists", False) else None
            )

            row = {
                "playerAvatar": av,
                "roundIndex": r,
                "gameId": game_id,
                "data.chat_message": chat_messages.get(av, "") if env.get("CONFIG_chat", False) else "",
                "data.chat_reasoning": chat_reasoning.get(av) if env.get("CONFIG_chat", False) else None,
                "data.contribution": contrib_rec[av],
                "data.contribution_reasoning": contrib_reasoning.get(av),
                "data.punished": punished_str,
                "data.rewarded": rewarded_str,
                "data.actions_reasoning": actions_reasoning.get(av),
            }
            if csv_writer:
                csv_writer.writerow(row)
            rows.append(row)

        if csv_file:
            csv_file.flush()
            os.fsync(csv_file.fileno())

    if csv_file:
        csv_file.close()

    df = pd.DataFrame(rows)

    if transcripts_out_path:
        os.makedirs(os.path.dirname(transcripts_out_path), exist_ok=True)
        transcripts_str = {av: "\n".join(chunks) for av, chunks in transcripts.items()}
        with open(transcripts_out_path, "w", encoding="utf-8") as f:
            for av, text in transcripts_str.items():
                f.write(json.dumps({"experiment": game_id, "participant": av, "text": text}, ensure_ascii=False) + "\n")

    if debug_jsonl_path and debug_level != "off":
        os.makedirs(os.path.dirname(debug_jsonl_path), exist_ok=True)
        with open(debug_jsonl_path, "a", encoding="utf-8") as fdbg:
            for rec in debug_records:
                fdbg.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if debug_full_jsonl_path:
        os.makedirs(os.path.dirname(debug_full_jsonl_path), exist_ok=True)
        with open(debug_full_jsonl_path, "a", encoding="utf-8") as fdbg_full:
            for rec in debug_full_records:
                fdbg_full.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return df, transcripts


def simulate_games(
    env_df: pd.DataFrame,
    args: Any,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]], Dict[str, Dict[str, Optional[str]]]]:
    run_ts = resolve_run_ts(getattr(args, "run_id", None))

    all_rows: List[Dict[str, Any]] = []
    all_transcripts: Dict[str, Dict[str, str]] = {}
    output_paths: Dict[str, Dict[str, Optional[str]]] = {}

    provider = args.provider
    tok: Optional[Any] = None
    model: Optional[Any] = None
    if provider == "local":
        from model_loader import load_model

        tok, model = load_model(base_model=args.base_model, adapter_path=args.adapter_path, use_peft=args.use_peft)
        if args.max_parallel_games > 1:
            log("[warn] local HF models should not be parallelized on a single GPU; forcing --max_parallel_games=1")
            args.max_parallel_games = 1

    if not getattr(args, "group_by_game", True):
        log("[warn] --group_by_game=False is overridden to keep outputs under experiment/timestamp folders")

    def build_client() -> LLMClient:
        return LLMClient(
            provider=provider,
            tok=tok,
            model=model,
            openai_model=args.openai_model,
            openai_api_key=args.openai_api_key,
            openai_api_key_env=args.openai_api_key_env,
        )

    client = build_client()

    rng = random.Random(args.seed)
    env_rows = [(idx, env, rng.randrange(0, 2**32 - 1)) for idx, env in env_df.iterrows()]

    def run_one(idx: int, env: pd.Series, game_seed: int):
        game_id = env.get("name", f"GAME_{idx}")
        log(f"[ptc] starting game {game_id} (idx={idx})")
        experiment_dir = resolve_experiment_dir(args.output_root, game_id, run_ts)
        rows_path = relocate_for_experiment(args.rows_out_path, experiment_dir)
        transcripts_path = relocate_for_experiment(args.transcripts_out_path, experiment_dir)
        debug_path = relocate_for_experiment(args.debug_jsonl_path, experiment_dir)
        debug_full_path = relocate_for_experiment(args.debug_full_jsonl_path, experiment_dir)
        config_path = write_config(experiment_dir, env, args, run_ts)

        df_game, transcripts_game = simulate_game(
            env=env,
            client=build_client() if args.max_parallel_games > 1 else client,
            tok=tok,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=game_seed,
            contrib_max_new_tokens=args.contrib_max_new_tokens,
            chat_max_new_tokens=args.chat_max_new_tokens,
            actions_max_new_tokens=args.actions_max_new_tokens,
            include_reasoning=args.include_reasoning,
            rows_out_path=rows_path,
            transcripts_out_path=transcripts_path,
            debug_jsonl_path=debug_path,
            debug_print=args.debug_print,
            debug_level=args.debug_level,
            debug_full_jsonl_path=debug_full_path,
            openai_async=args.openai_async,
            openai_max_concurrency=args.openai_max_concurrency,
        )
        log(f"[ptc] done game {game_id} with {len(df_game)} rows")
        return game_id, df_game, transcripts_game, rows_path, transcripts_path, debug_path, debug_full_path, config_path

    if args.max_parallel_games > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel_games) as executor:
            futures = [executor.submit(run_one, idx, env, game_seed) for idx, env, game_seed in env_rows]
            for fut in concurrent.futures.as_completed(futures):
                game_id, df_game, transcripts_game, rows_path, transcripts_path, debug_path, debug_full_path, config_path = fut.result()
                all_rows.extend(df_game.to_dict("records"))
                all_transcripts[game_id] = {av: "\n".join(t) for av, t in transcripts_game.items()}
                output_paths[game_id] = {
                    "rows": rows_path,
                    "transcripts": transcripts_path,
                    "debug": debug_path,
                    "debug_full": debug_full_path,
                    "config": config_path,
                    "directory": os.path.dirname(config_path),
                }
    else:
        for idx, env, game_seed in env_rows:
            game_id, df_game, transcripts_game, rows_path, transcripts_path, debug_path, debug_full_path, config_path = run_one(
                idx, env, game_seed
            )
            all_rows.extend(df_game.to_dict("records"))
            all_transcripts[game_id] = {av: "\n".join(t) for av, t in transcripts_game.items()}
            output_paths[game_id] = {
                "rows": rows_path,
                "transcripts": transcripts_path,
                "debug": debug_path,
                "debug_full": debug_full_path,
                "config": config_path,
                "directory": os.path.dirname(config_path),
            }

    return pd.DataFrame(all_rows), all_transcripts, output_paths
