from __future__ import annotations

import csv
import json
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from ..common import (
        MacroGameContext,
        as_bool,
        build_macro_game_contexts,
        json_compact,
        load_optional_csv,
        log,
        relocate_output,
        timestamp_yymmddhhmm,
        write_json,
    )
    from ..paths import BENCHMARK_DATA_ROOT, analysis_csv_name_for_wave, demographics_csv_name_for_wave
    from ..policy import RandomBaselinePolicyConfig, sample_random_baseline_action
except ImportError:
    from simulation_statistical.common import (
        MacroGameContext,
        as_bool,
        build_macro_game_contexts,
        json_compact,
        load_optional_csv,
        log,
        relocate_output,
        timestamp_yymmddhhmm,
        write_json,
    )
    from simulation_statistical.paths import BENCHMARK_DATA_ROOT, analysis_csv_name_for_wave, demographics_csv_name_for_wave
    from simulation_statistical.policy import RandomBaselinePolicyConfig, sample_random_baseline_action


DEFAULT_ROUNDS_CSV = os.path.join(BENCHMARK_DATA_ROOT, "raw_data", "validation_wave", "player-rounds.csv")
DEFAULT_ANALYSIS_CSV = os.path.join(
    BENCHMARK_DATA_ROOT,
    "processed_data",
    analysis_csv_name_for_wave("validation_wave"),
)
DEFAULT_DEMOGRAPHICS_CSV = os.path.join(
    BENCHMARK_DATA_ROOT,
    "demographics",
    demographics_csv_name_for_wave("validation_wave"),
)
DEFAULT_PLAYERS_CSV = os.path.join(BENCHMARK_DATA_ROOT, "raw_data", "validation_wave", "players.csv")


def _apply_data_root_paths(args: Any) -> None:
    data_root = str(getattr(args, "data_root", "") or "").strip()
    if not data_root:
        return

    wave = str(getattr(args, "wave", "validation_wave") or "validation_wave").strip()
    if wave not in {"validation_wave", "learning_wave"}:
        raise ValueError(f"Unsupported --wave '{wave}'. Allowed values: validation_wave, learning_wave.")

    analysis_name = analysis_csv_name_for_wave(wave)
    demographics_name = demographics_csv_name_for_wave(wave)
    derived = {
        "analysis_csv": os.path.join(data_root, "processed_data", analysis_name),
        "rounds_csv": os.path.join(data_root, "raw_data", wave, "player-rounds.csv"),
        "players_csv": os.path.join(data_root, "raw_data", wave, "players.csv"),
        "demographics_csv": os.path.join(data_root, "demographics", demographics_name),
    }
    defaults = {
        "analysis_csv": DEFAULT_ANALYSIS_CSV,
        "rounds_csv": DEFAULT_ROUNDS_CSV,
        "players_csv": DEFAULT_PLAYERS_CSV,
        "demographics_csv": DEFAULT_DEMOGRAPHICS_CSV,
    }
    for key, path in derived.items():
        current = str(getattr(args, key, "") or "").strip()
        if (not current) or (current == defaults[key]):
            setattr(args, key, path)


def _serialize_args(args: Any) -> Dict[str, Any]:
    if hasattr(args, "__dict__"):
        return dict(vars(args))
    return dict(args)


def _round_count(ctx: MacroGameContext) -> int:
    try:
        value = int(ctx.env.get("CONFIG_numRounds", 0) or 0)
    except Exception:
        value = 0
    return value if value > 0 else 1


def _compute_round_summary(
    ctx: MacroGameContext,
    contributions: Dict[str, int],
    punish_given: Dict[str, Dict[str, int]],
    reward_given: Dict[str, Dict[str, int]],
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    roster = [ctx.avatar_by_player[player_id] for player_id in ctx.player_ids]
    total_contribution = float(sum(int(value) for value in contributions.values()))
    multiplier = float(ctx.env.get("CONFIG_multiplier", 0) or 0)
    num_players = len(roster)
    share = (total_contribution * multiplier / num_players) if num_players > 0 else 0.0
    endowment = int(ctx.env.get("CONFIG_endowment", 0) or 0)

    punishment_cost = float(ctx.env.get("CONFIG_punishmentCost", 0) or 0)
    punishment_magnitude = float(ctx.env.get("CONFIG_punishmentMagnitude", 0) or 0)
    reward_cost = float(ctx.env.get("CONFIG_rewardCost", 0) or 0)
    reward_magnitude = float(ctx.env.get("CONFIG_rewardMagnitude", 0) or 0)
    punishment_on = as_bool(ctx.env.get("CONFIG_punishmentExists", False))
    reward_on = as_bool(ctx.env.get("CONFIG_rewardExists", False))

    inbound_punishment_units = {avatar: 0 for avatar in roster}
    inbound_reward_units = {avatar: 0 for avatar in roster}
    for src_avatar in roster:
        for target_avatar, units in punish_given.get(src_avatar, {}).items():
            inbound_punishment_units[target_avatar] = inbound_punishment_units.get(target_avatar, 0) + int(units)
        for target_avatar, units in reward_given.get(src_avatar, {}).items():
            inbound_reward_units[target_avatar] = inbound_reward_units.get(target_avatar, 0) + int(units)

    summary: Dict[str, Dict[str, float]] = {}
    for player_id in ctx.player_ids:
        avatar = ctx.avatar_by_player[player_id]
        spent_punish_units = sum(int(v) for v in punish_given.get(avatar, {}).values()) if punishment_on else 0
        spent_reward_units = sum(int(v) for v in reward_given.get(avatar, {}).values()) if reward_on else 0
        private_kept = endowment - int(contributions.get(avatar, 0))
        payoff = float(private_kept + share)

        player_summary: Dict[str, float] = {}
        if punishment_on:
            player_summary["coins_spent_on_punish"] = float(spent_punish_units * punishment_cost)
            player_summary["coins_deducted_from_you"] = float(
                inbound_punishment_units.get(avatar, 0) * punishment_magnitude
            )
            payoff -= player_summary["coins_spent_on_punish"]
            payoff -= player_summary["coins_deducted_from_you"]
        if reward_on:
            player_summary["coins_spent_on_reward"] = float(spent_reward_units * reward_cost)
            player_summary["coins_rewarded_to_you"] = float(
                inbound_reward_units.get(avatar, 0) * reward_magnitude
            )
            payoff -= player_summary["coins_spent_on_reward"]
            payoff += player_summary["coins_rewarded_to_you"]
        player_summary["payoff"] = float(payoff)
        summary[avatar] = player_summary

    return total_contribution, share, summary


def run_macro_statistical_simulation(args: Any) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    _apply_data_root_paths(args)

    run_ts = getattr(args, "run_id", None) or timestamp_yymmddhhmm()
    run_dir = os.path.join(args.output_root, run_ts)
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, "config.json")

    rows_out_path = relocate_output(args.rows_out_path, run_dir)
    transcripts_out_path = relocate_output(args.transcripts_out_path, run_dir) if args.transcripts_out_path else None
    debug_jsonl_path = relocate_output(args.debug_jsonl_path, run_dir) if args.debug_jsonl_path else None

    df_analysis = pd.read_csv(args.analysis_csv)
    df_rounds = pd.read_csv(args.rounds_csv)
    df_players = load_optional_csv(args.players_csv)
    df_demographics = load_optional_csv(args.demographics_csv)

    contexts = build_macro_game_contexts(
        df_analysis=df_analysis,
        df_rounds=df_rounds,
        df_players=df_players,
        df_demographics=df_demographics,
    )
    if args.game_ids:
        wanted = {part.strip() for part in str(args.game_ids).split(",") if part.strip()}
        contexts = [ctx for ctx in contexts if ctx.game_id in wanted or ctx.game_name in wanted]
    if args.max_games is not None:
        contexts = contexts[: int(args.max_games)]
    if not contexts:
        raise ValueError("No games selected for statistical macro simulation.")

    args_payload = _serialize_args(args)
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
            "game_ids": [ctx.game_id for ctx in contexts],
            "requested_game_ids": args.game_ids,
            "max_games": args.max_games,
        },
        "strategy": {
            "name": "random_baseline",
            "contribution_sampler": "uniform_legal_action_space",
            "target_probability": float(args.target_probability),
            "action_magnitude": int(args.action_magnitude),
            "chat_policy": "always_blank",
        },
        "args": args_payload,
        "outputs": {
            "directory": run_dir,
            "rows": rows_out_path,
            "transcripts": transcripts_out_path,
            "debug": debug_jsonl_path,
        },
    }
    write_json(config_path, config_payload)

    os.makedirs(os.path.dirname(rows_out_path), exist_ok=True)
    rows_file = open(rows_out_path, "w", newline="", encoding="utf-8")
    rows_writer: Optional[csv.DictWriter] = None

    transcripts_file = None
    if transcripts_out_path:
        os.makedirs(os.path.dirname(transcripts_out_path), exist_ok=True)
        transcripts_file = open(transcripts_out_path, "w", encoding="utf-8")

    debug_file = None
    if debug_jsonl_path:
        os.makedirs(os.path.dirname(debug_jsonl_path), exist_ok=True)
        debug_file = open(debug_jsonl_path, "w", encoding="utf-8")

    def _flush(handle: Any) -> None:
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
        _flush(rows_file)

    def _write_jsonl_chunk(handle: Any, chunk: List[Dict[str, Any]]) -> None:
        if handle is None or not chunk:
            return
        for record in chunk:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        _flush(handle)

    output_rows: List[Dict[str, Any]] = []
    policy_config = RandomBaselinePolicyConfig(
        target_probability=float(args.target_probability),
        action_magnitude=int(args.action_magnitude),
    )
    game_seed_rng = random.Random(int(args.seed))

    try:
        for index, ctx in enumerate(contexts, start=1):
            if args.debug_print:
                log(f"[stat-macro] start game {ctx.game_id} ({index}/{len(contexts)})")

            game_seed = game_seed_rng.randrange(0, 2**32 - 1)
            game_rng = random.Random(game_seed)
            transcripts = {player_id: ["# GAME STARTS"] for player_id in ctx.player_ids}

            for round_idx in range(1, _round_count(ctx) + 1):
                contributions: Dict[str, int] = {}
                punish_given_avatar: Dict[str, Dict[str, int]] = {}
                reward_given_avatar: Dict[str, Dict[str, int]] = {}
                round_rows: List[Dict[str, Any]] = []
                debug_rows: List[Dict[str, Any]] = []

                for player_id in ctx.player_ids:
                    avatar = ctx.avatar_by_player[player_id]
                    peer_avatars = [ctx.avatar_by_player[peer_id] for peer_id in ctx.player_ids if peer_id != player_id]
                    sampled = sample_random_baseline_action(
                        env=ctx.env,
                        focal_avatar=avatar,
                        peer_avatars=peer_avatars,
                        rng=game_rng,
                        config=policy_config,
                    )
                    contributions[avatar] = int(sampled.contribution)
                    punish_given_avatar[avatar] = dict(sampled.punished_avatar)
                    reward_given_avatar[avatar] = dict(sampled.rewarded_avatar)

                total_contribution, share, round_summary = _compute_round_summary(
                    ctx=ctx,
                    contributions=contributions,
                    punish_given=punish_given_avatar,
                    reward_given=reward_given_avatar,
                )

                for player_id in ctx.player_ids:
                    avatar = ctx.avatar_by_player[player_id]
                    punished_pid = {
                        ctx.player_by_avatar[target_avatar]: int(units)
                        for target_avatar, units in punish_given_avatar.get(avatar, {}).items()
                        if target_avatar in ctx.player_by_avatar and int(units) > 0
                    }
                    rewarded_pid = {
                        ctx.player_by_avatar[target_avatar]: int(units)
                        for target_avatar, units in reward_given_avatar.get(avatar, {}).items()
                        if target_avatar in ctx.player_by_avatar and int(units) > 0
                    }

                    round_rows.append(
                        {
                            "gameId": ctx.game_id,
                            "gameName": ctx.game_name,
                            "roundIndex": int(round_idx),
                            "playerId": player_id,
                            "playerAvatar": avatar,
                            "archetype": None,
                            "persona": None,
                            "demographics": ctx.demographics_by_player.get(player_id, ""),
                            "data.chat_message": "",
                            "data.chat_parsed": None,
                            "data.chat_reasoning": None,
                            "data.contribution": int(contributions[avatar]),
                            "data.contribution_clamped": int(contributions[avatar]),
                            "data.contribution_parsed": True,
                            "data.contribution_reasoning": None,
                            "data.punished": json_compact(punished_pid)
                            if as_bool(ctx.env.get("CONFIG_punishmentExists", False))
                            else None,
                            "data.rewarded": json_compact(rewarded_pid)
                            if as_bool(ctx.env.get("CONFIG_rewardExists", False))
                            else None,
                            "data.punished_avatar": json_compact(punish_given_avatar.get(avatar, {}))
                            if as_bool(ctx.env.get("CONFIG_punishmentExists", False))
                            else None,
                            "data.rewarded_avatar": json_compact(reward_given_avatar.get(avatar, {}))
                            if as_bool(ctx.env.get("CONFIG_rewardExists", False))
                            else None,
                            "data.actions_parsed": True
                            if (
                                as_bool(ctx.env.get("CONFIG_punishmentExists", False))
                                or as_bool(ctx.env.get("CONFIG_rewardExists", False))
                            )
                            else None,
                            "data.actions_reasoning": None,
                        }
                    )

                    transcripts[player_id].append(
                        f"<ROUND>{json_compact({'roundIndex': round_idx, 'contribution': contributions[avatar], 'punished': punish_given_avatar.get(avatar, {}), 'rewarded': reward_given_avatar.get(avatar, {}), 'totalContribution': total_contribution, 'publicShare': share, 'summary': round_summary.get(avatar, {})})}</ROUND>"
                    )
                    debug_rows.append(
                        {
                            "gameId": ctx.game_id,
                            "gameName": ctx.game_name,
                            "roundIndex": int(round_idx),
                            "playerId": player_id,
                            "playerAvatar": avatar,
                            "strategy": "random_baseline",
                            "seed": game_seed,
                            "contribution": int(contributions[avatar]),
                            "punished_avatar": dict(punish_given_avatar.get(avatar, {})),
                            "rewarded_avatar": dict(reward_given_avatar.get(avatar, {})),
                            "round_summary": round_summary.get(avatar, {}),
                        }
                    )

                _write_rows_chunk(round_rows)
                _write_jsonl_chunk(debug_file, debug_rows)
                output_rows.extend(round_rows)

            for player_id in ctx.player_ids:
                transcripts[player_id].append("# GAME COMPLETE")

            if transcripts_file is not None:
                for player_id in ctx.player_ids:
                    transcripts_file.write(
                        json.dumps(
                            {
                                "experiment": ctx.game_id,
                                "participant": player_id,
                                "text": "\n".join(transcripts[player_id]),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                _flush(transcripts_file)

            if args.debug_print:
                log(f"[stat-macro] done game {ctx.game_id}")
    finally:
        rows_file.close()
        if transcripts_file is not None:
            transcripts_file.close()
        if debug_file is not None:
            debug_file.close()

    output_df = pd.DataFrame(output_rows)
    config_payload["status"] = "completed"
    config_payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    config_payload["summary"] = {
        "num_rows": int(len(output_rows)),
        "num_games": int(len(contexts)),
    }
    write_json(config_path, config_payload)

    if args.debug_print:
        log(f"[stat-macro] wrote rows -> {rows_out_path}")
        if transcripts_out_path:
            log(f"[stat-macro] wrote transcripts -> {transcripts_out_path}")
        if debug_jsonl_path:
            log(f"[stat-macro] wrote debug -> {debug_jsonl_path}")
        log(f"[stat-macro] wrote config -> {config_path}")

    return output_df, {
        "rows": rows_out_path,
        "transcripts": transcripts_out_path,
        "debug": debug_jsonl_path,
        "config": config_path,
        "directory": run_dir,
    }
