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
        MicroGameContext,
        as_bool,
        build_micro_game_contexts,
        is_nan,
        json_compact,
        load_optional_csv,
        log,
        parse_dict,
        relocate_output,
        timestamp_yymmddhhmm,
        write_json,
    )
    from ..paths import BENCHMARK_DATA_ROOT, analysis_csv_name_for_wave, demographics_csv_name_for_wave
    from ..policy import RandomBaselinePolicyConfig, sample_random_baseline_action
except ImportError:
    from simulation_statistical.common import (
        MicroGameContext,
        as_bool,
        build_micro_game_contexts,
        is_nan,
        json_compact,
        load_optional_csv,
        log,
        parse_dict,
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
DEFAULT_GAMES_CSV = os.path.join(BENCHMARK_DATA_ROOT, "raw_data", "validation_wave", "games.csv")


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
        "rounds_csv": os.path.join(data_root, "raw_data", wave, "player-rounds.csv"),
        "analysis_csv": os.path.join(data_root, "processed_data", analysis_name),
        "demographics_csv": os.path.join(data_root, "demographics", demographics_name),
        "players_csv": os.path.join(data_root, "raw_data", wave, "players.csv"),
        "games_csv": os.path.join(data_root, "raw_data", wave, "games.csv"),
    }
    defaults = {
        "rounds_csv": DEFAULT_ROUNDS_CSV,
        "analysis_csv": DEFAULT_ANALYSIS_CSV,
        "demographics_csv": DEFAULT_DEMOGRAPHICS_CSV,
        "players_csv": DEFAULT_PLAYERS_CSV,
        "games_csv": DEFAULT_GAMES_CSV,
    }
    for key, path in derived.items():
        current = str(getattr(args, key, "") or "").strip()
        if (not current) or (current == defaults[key]):
            setattr(args, key, path)


def _serialize_args(args: Any) -> Dict[str, Any]:
    if hasattr(args, "__dict__"):
        return dict(vars(args))
    return dict(args)


def _avatar_to_player_dict(ctx: MicroGameContext, values: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for avatar, units in values.items():
        player_id = ctx.player_by_avatar.get(str(avatar))
        if player_id and int(units) > 0:
            out[player_id] = int(units)
    return out


def _append_observed_round_to_transcripts(
    ctx: MicroGameContext,
    transcripts: Dict[str, List[str]],
    round_idx: int,
) -> None:
    round_slice = ctx.round_to_rows.get(round_idx)
    if round_slice is None or round_slice.empty:
        for player_id in ctx.player_ids:
            transcripts[player_id].append(
                f'<OBSERVED_ROUND round="{int(round_idx)}" status="missing"></OBSERVED_ROUND>'
            )
        return

    row_by_player = {str(row["playerId"]): row for _, row in round_slice.iterrows()}
    for player_id in ctx.player_ids:
        row = row_by_player.get(player_id)
        if row is None or is_nan(row.get("data.roundPayoff")):
            payload = {"roundIndex": int(round_idx), "status": "missing"}
        else:
            actual_punish_pid = parse_dict(row.get("data.punished"))
            actual_reward_pid = parse_dict(row.get("data.rewarded"))
            actual_punish_avatar = {
                ctx.avatar_by_player.get(str(key), str(key)): int(value)
                for key, value in actual_punish_pid.items()
                if int(value) > 0
            }
            actual_reward_avatar = {
                ctx.avatar_by_player.get(str(key), str(key)): int(value)
                for key, value in actual_reward_pid.items()
                if int(value) > 0
            }
            contribution = row.get("data.contribution")
            payload = {
                "roundIndex": int(round_idx),
                "contribution": None if is_nan(contribution) else int(float(contribution)),
                "punished": actual_punish_avatar,
                "rewarded": actual_reward_avatar,
                "roundPayoff": int(float(row.get("data.roundPayoff"))) if not is_nan(row.get("data.roundPayoff")) else None,
            }
        transcripts[player_id].append(f"<OBSERVED_ROUND>{json_compact(payload)}</OBSERVED_ROUND>")


def _build_eval_rows(
    ctx: MicroGameContext,
    round_idx: int,
    predictions: Dict[str, Dict[str, Any]],
    round_slice: pd.DataFrame,
    skip_no_actual: bool,
) -> List[Dict[str, Any]]:
    row_by_player = {str(row["playerId"]): row for _, row in round_slice.iterrows()}
    rows: List[Dict[str, Any]] = []

    for player_id, prediction in predictions.items():
        actual_row = row_by_player.get(player_id)
        actual_available = actual_row is not None and not is_nan(actual_row.get("data.roundPayoff"))
        if skip_no_actual and not actual_available:
            continue

        avatar = ctx.avatar_by_player.get(player_id, player_id)
        actual_contribution: Optional[float] = None
        if actual_row is not None:
            raw_contribution = actual_row.get("data.contribution")
            if raw_contribution is not None and not is_nan(raw_contribution):
                actual_contribution = float(raw_contribution)

        predicted_contribution = float(prediction.get("pred_contribution", 0))
        contribution_abs_error = (
            None if actual_contribution is None else abs(predicted_contribution - actual_contribution)
        )

        actual_punish_pid = parse_dict(actual_row.get("data.punished")) if actual_row is not None else {}
        actual_reward_pid = parse_dict(actual_row.get("data.rewarded")) if actual_row is not None else {}
        actual_punish_avatar = {
            ctx.avatar_by_player.get(str(key), str(key)): int(value)
            for key, value in actual_punish_pid.items()
            if int(value) > 0
        }
        actual_reward_avatar = {
            ctx.avatar_by_player.get(str(key), str(key)): int(value)
            for key, value in actual_reward_pid.items()
            if int(value) > 0
        }

        predicted_punish_avatar = {
            str(key): int(value)
            for key, value in (prediction.get("pred_punished_avatar") or {}).items()
            if int(value) > 0
        }
        predicted_reward_avatar = {
            str(key): int(value)
            for key, value in (prediction.get("pred_rewarded_avatar") or {}).items()
            if int(value) > 0
        }
        predicted_punish_pid = _avatar_to_player_dict(ctx, predicted_punish_avatar)
        predicted_reward_pid = _avatar_to_player_dict(ctx, predicted_reward_avatar)

        rows.append(
            {
                "gameId": ctx.game_id,
                "gameName": ctx.game_name,
                "roundIndex": int(round_idx),
                "historyRounds": int(round_idx) - 1,
                "playerId": player_id,
                "playerAvatar": avatar,
                "actual_available": actual_available,
                "predicted_contribution": prediction.get("pred_contribution"),
                "predicted_contribution_raw": prediction.get("pred_contribution_raw"),
                "predicted_contribution_reasoning": prediction.get("pred_contribution_reasoning"),
                "predicted_contribution_parsed": prediction.get("pred_contribution_parsed"),
                "actual_contribution": actual_contribution,
                "contribution_abs_error": contribution_abs_error,
                "predicted_punished_avatar": json_compact(predicted_punish_avatar),
                "predicted_rewarded_avatar": json_compact(predicted_reward_avatar),
                "predicted_punished_pid": json_compact(predicted_punish_pid),
                "predicted_rewarded_pid": json_compact(predicted_reward_pid),
                "predicted_actions_reasoning": prediction.get("pred_actions_reasoning"),
                "predicted_actions_parsed": prediction.get("pred_actions_parsed"),
                "actual_punished_avatar": json_compact(actual_punish_avatar),
                "actual_rewarded_avatar": json_compact(actual_reward_avatar),
                "actual_punished_pid": json_compact(actual_punish_pid),
                "actual_rewarded_pid": json_compact(actual_reward_pid),
                "actual_data.punished_raw": None if actual_row is None else actual_row.get("data.punished"),
                "actual_data.rewarded_raw": None if actual_row is None else actual_row.get("data.rewarded"),
                "demographics": ctx.demographics_by_player.get(player_id, ""),
            }
        )
    return rows


def run_micro_statistical_simulation(args: Any) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    _apply_data_root_paths(args)

    run_ts = getattr(args, "run_id", None) or timestamp_yymmddhhmm()
    run_dir = os.path.join(args.output_root, run_ts)
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, "config.json")

    rows_out_path = relocate_output(args.rows_out_path, run_dir)
    transcripts_out_path = relocate_output(args.transcripts_out_path, run_dir)
    debug_jsonl_path = relocate_output(args.debug_jsonl_path, run_dir) if args.debug_jsonl_path else None

    df_rounds = pd.read_csv(args.rounds_csv)
    df_analysis = pd.read_csv(args.analysis_csv)
    df_demographics = load_optional_csv(args.demographics_csv)
    df_players = load_optional_csv(args.players_csv)

    contexts = build_micro_game_contexts(
        df_rounds=df_rounds,
        df_analysis=df_analysis,
        df_demographics=df_demographics,
        df_players=df_players,
    )
    if args.game_ids:
        wanted = {part.strip() for part in str(args.game_ids).split(",") if part.strip()}
        contexts = [ctx for ctx in contexts if ctx.game_id in wanted or ctx.game_name in wanted]
    if args.max_games is not None:
        contexts = contexts[: int(args.max_games)]
    if not contexts:
        raise ValueError("No games selected for statistical micro simulation.")

    args_payload = _serialize_args(args)
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
            "game_ids": [ctx.game_id for ctx in contexts],
            "requested_game_ids": args.game_ids,
            "start_round": int(args.start_round),
            "skip_no_actual": bool(args.skip_no_actual),
        },
        "strategy": {
            "name": "random_baseline",
            "contribution_sampler": "uniform_legal_action_space",
            "target_probability": float(args.target_probability),
            "action_magnitude": int(args.action_magnitude),
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

    def _write_debug_chunk(chunk: List[Dict[str, Any]]) -> None:
        if debug_file is None or not chunk:
            return
        for record in chunk:
            debug_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        _flush(debug_file)

    all_rows: List[Dict[str, Any]] = []
    policy_config = RandomBaselinePolicyConfig(
        target_probability=float(args.target_probability),
        action_magnitude=int(args.action_magnitude),
    )
    game_seed_rng = random.Random(int(args.seed))

    try:
        for index, ctx in enumerate(contexts, start=1):
            if args.debug_print:
                log(f"[stat-micro] start game {ctx.game_id} ({index}/{len(contexts)})")

            game_seed = game_seed_rng.randrange(0, 2**32 - 1)
            game_rng = random.Random(game_seed)
            transcripts = {player_id: ["# GAME STARTS"] for player_id in ctx.player_ids}

            for round_idx in ctx.rounds:
                round_slice = ctx.round_to_rows.get(round_idx, pd.DataFrame())
                if int(round_idx) >= int(args.start_round):
                    predictions: Dict[str, Dict[str, Any]] = {}
                    debug_records: List[Dict[str, Any]] = []
                    actions_enabled = as_bool(ctx.env.get("CONFIG_punishmentExists")) or as_bool(
                        ctx.env.get("CONFIG_rewardExists")
                    )

                    for player_id in ctx.player_ids:
                        avatar = ctx.avatar_by_player.get(player_id, player_id)
                        peer_avatars = [
                            ctx.avatar_by_player[peer_id] for peer_id in ctx.player_ids if peer_id != player_id
                        ]
                        sampled = sample_random_baseline_action(
                            env=ctx.env,
                            focal_avatar=avatar,
                            peer_avatars=peer_avatars,
                            rng=game_rng,
                            config=policy_config,
                        )
                        prediction = {
                            "pred_contribution": int(sampled.contribution),
                            "pred_contribution_raw": int(sampled.contribution),
                            "pred_contribution_reasoning": None,
                            "pred_contribution_parsed": True,
                            "pred_punished_avatar": dict(sampled.punished_avatar),
                            "pred_rewarded_avatar": dict(sampled.rewarded_avatar),
                            "pred_actions_reasoning": None,
                            "pred_actions_parsed": True if actions_enabled else None,
                        }
                        predictions[player_id] = prediction

                        transcripts[player_id].append(
                            f'<PREDICTED_ROUND round="{int(round_idx)}">'
                            f'{json_compact({"contribution": sampled.contribution, "punished": sampled.punished_avatar, "rewarded": sampled.rewarded_avatar})}'
                            "</PREDICTED_ROUND>"
                        )
                        debug_records.append(
                            {
                                "gameId": ctx.game_id,
                                "gameName": ctx.game_name,
                                "roundIndex": int(round_idx),
                                "playerId": player_id,
                                "playerAvatar": avatar,
                                "strategy": "random_baseline",
                                "seed": game_seed,
                                "contribution": int(sampled.contribution),
                                "predicted_punished_avatar": dict(sampled.punished_avatar),
                                "predicted_rewarded_avatar": dict(sampled.rewarded_avatar),
                            }
                        )

                    round_rows = _build_eval_rows(
                        ctx=ctx,
                        round_idx=int(round_idx),
                        predictions=predictions,
                        round_slice=round_slice,
                        skip_no_actual=bool(args.skip_no_actual),
                    )
                    _write_rows_chunk(round_rows)
                    _write_debug_chunk(debug_records)
                    all_rows.extend(round_rows)

                _append_observed_round_to_transcripts(ctx, transcripts, int(round_idx))

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
                log(f"[stat-micro] done game {ctx.game_id}")
    finally:
        rows_file.close()
        if transcripts_file is not None:
            transcripts_file.close()
        if debug_file is not None:
            debug_file.close()

    output_df = pd.DataFrame(all_rows)
    config_payload["status"] = "completed"
    config_payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    config_payload["summary"] = {
        "num_rows": int(len(all_rows)),
        "num_games": int(len(contexts)),
    }
    write_json(config_path, config_payload)

    if args.debug_print:
        log(f"[stat-micro] wrote rows -> {rows_out_path}")
        if transcripts_out_path:
            log(f"[stat-micro] wrote transcripts -> {transcripts_out_path}")
        if debug_jsonl_path:
            log(f"[stat-micro] wrote debug -> {debug_jsonl_path}")
        log(f"[stat-micro] wrote config -> {config_path}")

    return output_df, {
        "rows": rows_out_path,
        "transcripts": transcripts_out_path,
        "debug": debug_jsonl_path,
        "config": config_path,
        "directory": run_dir,
    }
