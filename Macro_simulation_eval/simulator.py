from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import ast
import csv
import json
import math
import os
import random
from threading import Lock
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
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
    from .llm_client import LLMClient
    from .model_loader import load_model
    from .parsers import parse_json_response
    from .prompt_builder import (
        JSON_STOP_SENTINEL,
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
        JSON_STOP_SENTINEL,
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


def _use_binary_targets(action_prompt_mode: str) -> bool:
    return str(action_prompt_mode or "binary_targets").strip().lower() != "legacy_units"


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


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


def _serialize_args(args: Any) -> Dict[str, Any]:
    if is_dataclass(args):
        return asdict(args)
    if hasattr(args, "__dict__"):
        return dict(vars(args))
    return dict(args)


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
        "chat_max_new_tokens",
        "actions_max_new_tokens",
        "action_prompt_mode",
        "action_continuation_gate",
        "punish_continuation_keep_prob",
        "reward_continuation_keep_prob",
        "include_reasoning",
        "include_demographics",
        "archetype",
        "archetype_summary_pool",
        "game_ids",
        "max_games",
        "max_parallel_games",
        "debug_level",
        "debug_compact",
    ]
    if provider in {"local", "vllm"}:
        base_keys.append("base_model")
    if provider == "local":
        base_keys.extend(["adapter_path", "use_peft", "load_in_8bit", "load_in_4bit", "quant_compute_dtype"])
    payload = {k: args_dict.get(k) for k in base_keys if k in args_dict}
    payload["run_timestamp"] = run_ts
    return payload


def _write_config_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
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


def _parse_sparse_action_map(value: Any) -> Dict[str, int]:
    if isinstance(value, dict):
        return {
            str(key): int(units)
            for key, units in value.items()
            if str(key) and _safe_int(units) > 0
        }
    if value is None or is_nan(value):
        return {}
    if isinstance(value, str):
        raw = value.strip()
        if not raw or raw.lower() in {"nan", "none", "null"}:
            return {}
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(raw)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return {
                    str(key): int(units)
                    for key, units in parsed.items()
                    if str(key) and _safe_int(units) > 0
                }
    return {}


def _build_initial_macro_transcripts(
    ctx: GameContext,
    assigned_archetypes: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, Optional[str]]]:
    assigned_archetypes = assigned_archetypes or {}
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
    return transcripts, archetype_ids


def _completed_macro_rounds(
    ctx: GameContext,
    existing_rows: pd.DataFrame,
) -> Tuple[int, Optional[int], List[str]]:
    warnings: List[str] = []
    if existing_rows.empty or "roundIndex" not in existing_rows.columns or "playerId" not in existing_rows.columns:
        return 0, 1, warnings

    frame = existing_rows.copy()
    frame["playerId"] = frame["playerId"].astype(str)
    frame = frame[frame["playerId"].isin({str(pid) for pid in ctx.player_ids})].copy()
    frame["roundIndex"] = pd.to_numeric(frame["roundIndex"], errors="coerce")
    frame = frame.dropna(subset=["roundIndex"])
    if frame.empty:
        return 0, 1, warnings
    frame["roundIndex"] = frame["roundIndex"].astype(int)

    expected_per_round = int(len(ctx.player_ids))
    num_rounds = int(ctx.env.get("CONFIG_numRounds", 0) or 0)
    completed_rounds = 0
    for round_idx in range(1, num_rounds + 1):
        round_df = frame[frame["roundIndex"] == round_idx]
        if round_df.empty:
            return completed_rounds, round_idx, warnings
        unique_players = int(round_df["playerId"].nunique())
        total_rows = int(len(round_df))
        if total_rows > unique_players:
            warnings.append(
                f"gameId={ctx.game_id} round={round_idx} has duplicate rows "
                f"(rows={total_rows}, unique_players={unique_players})."
            )
        if unique_players < expected_per_round:
            return completed_rounds, round_idx, warnings
        if unique_players > expected_per_round:
            warnings.append(
                f"gameId={ctx.game_id} round={round_idx} has {unique_players} unique players; "
                f"expected {expected_per_round}."
            )
        completed_rounds = round_idx
    return completed_rounds, None, warnings


def _trim_macro_rows_for_resume(
    existing_rows: pd.DataFrame,
    resume_round_by_game: Mapping[str, int],
) -> pd.DataFrame:
    if existing_rows.empty or not resume_round_by_game:
        return existing_rows
    if "gameId" not in existing_rows.columns or "roundIndex" not in existing_rows.columns:
        return existing_rows

    frame = existing_rows.copy()
    game_series = frame["gameId"].astype(str)
    round_series = pd.to_numeric(frame["roundIndex"], errors="coerce")
    keep_mask = pd.Series(True, index=frame.index)
    for game_id, resume_round in resume_round_by_game.items():
        keep_mask &= ~((game_series == str(game_id)) & (round_series >= int(resume_round)))
    return frame.loc[keep_mask].copy()


def _rewrite_jsonl_for_macro_resume(
    path: Optional[str],
    game_key: str,
    round_key: Optional[str],
    resume_round_by_game: Mapping[str, int],
) -> Dict[str, int]:
    if not path or not os.path.exists(path):
        return {"removed": 0, "malformed": 0}
    if not resume_round_by_game:
        return {"removed": 0, "malformed": 0}

    tmp_path = f"{path}.resume_tmp"
    removed = 0
    malformed = 0
    with open(path, "r", encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as dst:
        for line in src:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                malformed += 1
                continue
            game_id = str(rec.get(game_key) or "").strip()
            if game_id not in resume_round_by_game:
                dst.write(line if line.endswith("\n") else f"{line}\n")
                continue
            if round_key is None:
                removed += 1
                continue
            round_idx = _safe_int(rec.get(round_key))
            if round_idx >= int(resume_round_by_game[game_id]):
                removed += 1
                continue
            dst.write(line if line.endswith("\n") else f"{line}\n")
    os.replace(tmp_path, path)
    return {"removed": removed, "malformed": malformed}


def _restore_macro_state_from_rows(
    ctx: GameContext,
    args: Any,
    existing_rows: pd.DataFrame,
    assigned_archetypes: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    transcripts, _ = _build_initial_macro_transcripts(ctx, assigned_archetypes=assigned_archetypes)
    roster = [ctx.avatar_by_player[pid] for pid in ctx.player_ids]
    avatar_by_pid = {pid: ctx.avatar_by_player[pid] for pid in ctx.player_ids}
    previous_round_rewards = {avatar: {} for avatar in roster}
    previous_round_punish = {avatar: {} for avatar in roster}
    if existing_rows.empty:
        return transcripts, previous_round_rewards, previous_round_punish

    frame = existing_rows.copy()
    frame["playerId"] = frame["playerId"].astype(str)
    frame["roundIndex"] = pd.to_numeric(frame["roundIndex"], errors="coerce")
    frame = frame.dropna(subset=["roundIndex"])
    if frame.empty:
        return transcripts, previous_round_rewards, previous_round_punish
    frame["roundIndex"] = frame["roundIndex"].astype(int)

    chat_on = as_bool(ctx.env.get("CONFIG_chat", False))
    reward_on = as_bool(ctx.env.get("CONFIG_rewardExists", False))
    punish_on = as_bool(ctx.env.get("CONFIG_punishmentExists", False))
    show_punish_id = as_bool(ctx.env.get("CONFIG_showPunishmentId", False))
    show_reward_id = as_bool(ctx.env.get("CONFIG_showRewardId", False))
    show_other = as_bool(ctx.env.get("CONFIG_showOtherSummaries", False))
    action_prompt_mode = str(getattr(args, "action_prompt_mode", "binary_targets") or "binary_targets")
    endow = int(ctx.env.get("CONFIG_endowment", 0) or 0)

    for round_idx in sorted(frame["roundIndex"].unique().tolist()):
        round_df = frame[frame["roundIndex"] == int(round_idx)].sort_index(kind="stable")
        round_df = round_df.drop_duplicates(subset=["playerId"], keep="last")
        row_by_pid = {
            str(row["playerId"]): row
            for _, row in round_df.iterrows()
            if str(row["playerId"]) in avatar_by_pid
        }

        chat_messages: Dict[str, str] = {}
        for pid in ctx.player_ids:
            avatar = avatar_by_pid[pid]
            row = row_by_pid.get(pid)
            raw_message = "" if row is None else row.get("data.chat_message", "")
            if raw_message is None or is_nan(raw_message):
                raw_message = ""
            chat_messages[avatar] = str(raw_message)

        for pid in ctx.player_ids:
            transcripts[pid].append(round_open(ctx.env, int(round_idx)))
        if chat_on:
            _append_chat_history_line(transcripts, roster, avatar_by_pid, chat_messages)

        contrib_math: Dict[str, int] = {}
        for pid in ctx.player_ids:
            avatar = avatar_by_pid[pid]
            row = row_by_pid.get(pid)
            contribution = 0
            if row is not None:
                raw_value = row.get("data.contribution_clamped")
                if raw_value is None or is_nan(raw_value):
                    raw_value = row.get("data.contribution")
                contribution = _safe_int(raw_value)
            contrib_math[avatar] = int(max(0, contribution))
            transcripts[pid].append(f'<CONTRIB v="{int(contrib_math[avatar])}"/>')

        total_contrib = int(sum(contrib_math.values()))
        try:
            multiplied = float(ctx.env.get("CONFIG_multiplier", 0) or 0) * float(total_contrib)
        except Exception:
            multiplied = float("nan")
        active_players = len(roster)
        for pid in ctx.player_ids:
            avatar = avatar_by_pid[pid]
            transcripts[pid].append(redist_line(total_contrib, multiplied, active_players))
            peers_csv, _ = peers_contributions_csv(roster, avatar, contrib_math)
            transcripts[pid].append(f"<PEERS_CONTRIBUTIONS> {peers_csv} </PEERS_CONTRIBUTIONS>")

        rewards_given: Dict[str, Dict[str, int]] = {avatar: {} for avatar in roster}
        punish_given: Dict[str, Dict[str, int]] = {avatar: {} for avatar in roster}
        for pid in ctx.player_ids:
            avatar = avatar_by_pid[pid]
            row = row_by_pid.get(pid)
            reward_out = _parse_sparse_action_map(None if row is None else row.get("data.rewarded"))
            punish_out = _parse_sparse_action_map(None if row is None else row.get("data.punished"))
            rewards_given[avatar] = reward_out
            punish_given[avatar] = punish_out

            if reward_on and not punish_on:
                transcripts[pid].append(f"<REWARD>{_json_compact(reward_out)}</REWARD>")
            elif punish_on and not reward_on:
                transcripts[pid].append(f"<PUNISHMENT>{_json_compact(punish_out)}</PUNISHMENT>")
            elif reward_on or punish_on:
                transcripts[pid].append(
                    f'<ACTIONS punish="{_json_compact(punish_out)}" reward="{_json_compact(reward_out)}"/>'
                )

        previous_round_rewards = {avatar: dict(targets) for avatar, targets in rewards_given.items()}
        previous_round_punish = {avatar: dict(targets) for avatar, targets in punish_given.items()}

        inbound_reward_units = {avatar: 0 for avatar in roster}
        inbound_punish_units = {avatar: 0 for avatar in roster}
        for src in roster:
            for tgt, units in rewards_given[src].items():
                inbound_reward_units[tgt] += int(units)
            for tgt, units in punish_given[src].items():
                inbound_punish_units[tgt] += int(units)

        num_players = len(roster)
        share = (float(multiplied) / num_players) if (not math.isnan(float(multiplied)) and num_players > 0) else 0.0
        for pid in ctx.player_ids:
            avatar = avatar_by_pid[pid]
            if show_punish_id and punish_on:
                punishers = {
                    src: units
                    for src in roster
                    for tgt, units in punish_given[src].items()
                    if tgt == avatar and int(units) > 0
                }
                transcripts[pid].append(f"<PUNISHED_BY>{_json_compact(punishers)}</PUNISHED_BY>")
            if show_reward_id and reward_on:
                rewarders = {
                    src: units
                    for src in roster
                    for tgt, units in rewards_given[src].items()
                    if tgt == avatar and int(units) > 0
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
                        ob["coins_spent_on_punish"] = sum(punish_given[other].values()) * int(
                            ctx.env.get("CONFIG_punishmentCost", 0) or 0
                        )
                        ob["coins_deducted_from_them"] = (
                            sum(punish_given[src].get(other, 0) for src in roster)
                            * int(ctx.env.get("CONFIG_punishmentMagnitude", 0) or 0)
                        )
                    if reward_on:
                        ob["coins_spent_on_reward"] = sum(rewards_given[other].values()) * int(
                            ctx.env.get("CONFIG_rewardCost", 0) or 0
                        )
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

    return transcripts, previous_round_rewards, previous_round_punish


def _rewrite_jsonl_excluding_games(path: Optional[str], game_key: str, excluded_game_ids: Sequence[str]) -> Dict[str, int]:
    if not path or not os.path.exists(path):
        return {"removed": 0, "malformed": 0}
    excluded = {str(game_id) for game_id in excluded_game_ids if str(game_id)}
    if not excluded:
        return {"removed": 0, "malformed": 0}

    tmp_path = f"{path}.resume_tmp"
    removed = 0
    malformed = 0
    with open(path, "r", encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as dst:
        for line in src:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                malformed += 1
                continue
            game_id = str(rec.get(game_key) or "").strip()
            if game_id in excluded:
                removed += 1
                continue
            dst.write(line if line.endswith("\n") else f"{line}\n")
    os.replace(tmp_path, path)
    return {"removed": removed, "malformed": malformed}


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


def _resolve_archetype_mode(args: Any) -> str:
    mode = str(getattr(args, "archetype", None) or "").strip()
    if not mode:
        mode = str(getattr(args, "archetype_mode", None) or "").strip()
    if mode:
        if mode == "none":
            return ""
        return canonicalize_archetype_mode(mode)
    return canonicalize_archetype_mode(str(getattr(args, "persona", None) or "").strip())


def _resolve_archetype_pool_path(args: Any) -> str:
    pool = str(getattr(args, "archetype_summary_pool", None) or "").strip()
    if pool:
        return pool
    return str(getattr(args, "persona_summary_pool", None) or "").strip()


def _resolve_archetype_assignment_manifest_path(args: Any) -> str:
    return str(getattr(args, "archetype_assignments_in_path", None) or "").strip()


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
    assigned_archetypes: Optional[Dict[str, Dict[str, Any]]] = None,
    debug_records: Optional[List[Dict[str, Any]]] = None,
    debug_full_records: Optional[List[Dict[str, Any]]] = None,
    start_round: int = 1,
    initial_transcripts: Optional[Dict[str, List[str]]] = None,
    initial_previous_round_rewards: Optional[Dict[str, Dict[str, int]]] = None,
    initial_previous_round_punish: Optional[Dict[str, Dict[str, int]]] = None,
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
    stop_sequences = [JSON_STOP_SENTINEL]
    include_reasoning = bool(getattr(args, "include_reasoning", False))
    game_id = ctx.game_id

    roster = [ctx.avatar_by_player[pid] for pid in ctx.player_ids]
    pid_by_avatar = {ctx.avatar_by_player[pid]: pid for pid in ctx.player_ids}
    system_text_by_pid = {
        pid: system_header_plain(
            ctx.env,
            ctx.demographics_by_player.get(pid, "") if bool(getattr(args, "include_demographics", False)) else "",
            include_reasoning,
        )
        for pid in ctx.player_ids
    }

    base_transcripts, archetype_ids = _build_initial_macro_transcripts(
        ctx,
        assigned_archetypes=assigned_archetypes,
    )
    transcripts: Dict[str, List[str]] = (
        {pid: list(initial_transcripts.get(pid, base_transcripts.get(pid, []))) for pid in ctx.player_ids}
        if initial_transcripts is not None
        else base_transcripts
    )

    rows: List[Dict[str, Any]] = []
    actions_reasoning: Dict[str, Optional[str]] = {}
    contrib_reasoning: Dict[str, Optional[str]] = {}
    chat_reasoning: Dict[str, Optional[str]] = {}
    previous_round_rewards = (
        {avatar: dict(initial_previous_round_rewards.get(avatar, {})) for avatar in roster}
        if initial_previous_round_rewards is not None
        else {avatar: {} for avatar in roster}
    )
    previous_round_punish = (
        {avatar: dict(initial_previous_round_punish.get(avatar, {})) for avatar in roster}
        if initial_previous_round_punish is not None
        else {avatar: {} for avatar in roster}
    )

    num_rounds = int(ctx.env.get("CONFIG_numRounds", 0) or 0)
    for r in range(max(1, int(start_round)), num_rounds + 1):
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
                async_openai=_remote_async_enabled(args, client.provider),
                max_concurrency=_remote_max_concurrency(args, client.provider),
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
            async_openai=_remote_async_enabled(args, client.provider),
            max_concurrency=_remote_max_concurrency(args, client.provider),
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
        action_prompt_mode = str(getattr(args, "action_prompt_mode", "binary_targets") or "binary_targets")
        binary_targets = _use_binary_targets(action_prompt_mode)
        gate_enabled = bool(getattr(args, "action_continuation_gate", True))
        punish_keep_prob = _clamp_probability(getattr(args, "punish_continuation_keep_prob", 0.5))
        reward_keep_prob = _clamp_probability(getattr(args, "reward_continuation_keep_prob", 0.35))
        round_gate_rng = random.Random(f"{game_seed}|{r}|macro_continuation_gate")
        actions_parsed: Dict[str, Optional[bool]] = {av: None for av in roster}
        rewards_given: Dict[str, Dict[str, int]] = {av: {} for av in roster}
        punish_given: Dict[str, Dict[str, int]] = {av: {} for av in roster}

        if reward_on or punish_on:
            tag = actions_tag(ctx.env) or "PUNISHMENT/REWARD"
            mech = mech_info(ctx.env, action_prompt_mode)
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
                        actions_format_line(tag, include_reasoning, action_prompt_mode),
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
                async_openai=_remote_async_enabled(args, client.provider),
                max_concurrency=_remote_max_concurrency(args, client.provider),
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
                punish_out: Dict[str, int] = {}
                reward_out: Dict[str, int] = {}
                if ok and isinstance(payload, dict) and payload.get("stage") == "actions":
                    parsed_ok = True
                    if binary_targets:
                        punish_out, reward_out = _parse_binary_action_response(payload, tag, peer_orders[avatar])
                    else:
                        raw_actions = payload.get("actions")
                        if raw_actions is None:
                            actions_dict = {}
                        elif isinstance(raw_actions, dict):
                            actions_dict = raw_actions
                actions_parsed[avatar] = parsed_ok

                peer_order = peer_orders[avatar]
                if not binary_targets:
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
                    actions_out = {peer_order[i]: int(arr[i]) for i in range(len(peer_order)) if int(arr[i]) != 0}
                    if reward_on and not punish_on:
                        reward_out = {a: int(v) for a, v in actions_out.items() if int(v) > 0}
                    elif punish_on and not reward_on:
                        punish_out = {a: int(v) for a, v in actions_out.items() if int(v) > 0}
                    else:
                        punish_out = {a: int(abs(v)) for a, v in actions_out.items() if int(v) < 0}
                        reward_out = {a: int(v) for a, v in actions_out.items() if int(v) > 0}

                if include_reasoning and ok and isinstance(payload, dict) and isinstance(payload.get("reasoning"), str):
                    actions_reasoning[avatar] = payload.get("reasoning")
                else:
                    actions_reasoning[avatar] = None

                if gate_enabled:
                    punish_out = _apply_continuation_gate(
                        punish_out,
                        previous_round_punish.get(avatar, {}),
                        punish_keep_prob,
                        round_gate_rng,
                    )
                    reward_out = _apply_continuation_gate(
                        reward_out,
                        previous_round_rewards.get(avatar, {}),
                        reward_keep_prob,
                        round_gate_rng,
                    )

                if reward_on and not punish_on:
                    punish_out = {}
                    transcripts[pid].append(f"<REWARD>{_json_compact(reward_out)}</REWARD>")
                elif punish_on and not reward_on:
                    reward_out = {}
                    transcripts[pid].append(f"<PUNISHMENT>{_json_compact(punish_out)}</PUNISHMENT>")
                else:
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

        previous_round_rewards = {avatar: dict(targets) for avatar, targets in rewards_given.items()}
        previous_round_punish = {avatar: dict(targets) for avatar, targets in punish_given.items()}

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
            archetype_record = dict(assigned_archetypes.get(pid) or {})
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
                    "archetype_mode": _resolve_archetype_mode(args) or "",
                    "archetype_source_gameId": None if archetype_record is None else archetype_record.get("experiment"),
                    "archetype_source_playerId": None if archetype_record is None else archetype_record.get("participant"),
                    "archetype_source_rank": None if archetype_record is None else archetype_record.get("source_rank"),
                    "archetype_source_score": None if archetype_record is None else archetype_record.get("source_score"),
                    "archetype_source_weight": None if archetype_record is None else archetype_record.get("source_weight"),
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
    transcripts_out_path = relocate_output(args.transcripts_out_path, run_dir) if args.transcripts_out_path else None
    archetype_assignments_out_path = (
        relocate_output(args.archetype_assignments_out_path, run_dir)
        if getattr(args, "archetype_assignments_out_path", None)
        else None
    )
    debug_jsonl_path = relocate_output(args.debug_jsonl_path, run_dir) if args.debug_jsonl_path else None
    debug_full_jsonl_path = relocate_output(args.debug_full_jsonl_path, run_dir) if args.debug_full_jsonl_path else None
    existing_rows_df = _read_existing_rows_csv(rows_out_path) if resume_enabled else pd.DataFrame()

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
    selected_contexts = list(contexts)

    completed_game_ids: List[str] = []
    resumed_game_ids: List[str] = []
    resume_round_by_game: Dict[str, int] = {}
    resume_warnings: List[str] = []
    purge_stats: Dict[str, Any] = {}
    if resume_enabled:
        remaining_contexts: List[GameContext] = []
        for ctx in selected_contexts:
            game_rows = (
                existing_rows_df[existing_rows_df["gameId"].astype(str) == ctx.game_id].copy()
                if (not existing_rows_df.empty and "gameId" in existing_rows_df.columns)
                else pd.DataFrame()
            )
            completed_rounds, resume_round, game_warnings = _completed_macro_rounds(ctx, game_rows)
            resume_warnings.extend(game_warnings)
            expected_rounds = int(ctx.env.get("CONFIG_numRounds", 0) or 0)
            if completed_rounds >= expected_rounds:
                completed_game_ids.append(ctx.game_id)
                continue
            if resume_round is None:
                resume_round = completed_rounds + 1
            if completed_rounds > 0:
                resumed_game_ids.append(ctx.game_id)
            resume_round_by_game[ctx.game_id] = int(resume_round)
            remaining_contexts.append(ctx)
        contexts = remaining_contexts

        if resume_round_by_game:
            original_row_count = int(len(existing_rows_df))
            existing_rows_df = _trim_macro_rows_for_resume(existing_rows_df, resume_round_by_game)
            os.makedirs(os.path.dirname(rows_out_path), exist_ok=True)
            existing_rows_df.to_csv(rows_out_path, index=False)
            purge_stats["rows_removed"] = original_row_count - int(len(existing_rows_df))
            purge_stats["debug"] = _rewrite_jsonl_for_macro_resume(
                debug_jsonl_path,
                "game",
                "round",
                resume_round_by_game,
            )
            purge_stats["debug_full"] = _rewrite_jsonl_for_macro_resume(
                debug_full_jsonl_path,
                "game",
                "round",
                resume_round_by_game,
            )
            purge_stats["transcripts"] = _rewrite_jsonl_for_macro_resume(
                transcripts_out_path,
                "experiment",
                None,
                resume_round_by_game,
            )
            purge_stats["archetype_assignments"] = _rewrite_jsonl_for_macro_resume(
                archetype_assignments_out_path,
                "target_gameId",
                None,
                resume_round_by_game,
            )

    existing_assignment_rows = _count_nonempty_lines(archetype_assignments_out_path) if resume_enabled else 0

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
        if purge_stats:
            resume_event["purged_partial_outputs"] = purge_stats
        if resume_warnings:
            resume_event["warnings"] = resume_warnings
        resume_history.append(resume_event)

    config_payload = {
        "run_timestamp": run_ts,
        "status": "running",
        "created_at_utc": created_at_utc,
        "inputs": {
            "analysis_csv": args.analysis_csv,
            "rounds_csv": args.rounds_csv,
            "players_csv": args.players_csv,
            "demographics_csv": args.demographics_csv,
        },
        "selection": {
            "num_games": len(selected_contexts),
            "game_ids": [c.game_id for c in selected_contexts],
            "requested_game_ids": args.game_ids,
            "max_games": args.max_games,
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
                    "[macro] loaded precomputed archetype assignments from",
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
                    "[macro] loaded archetype summaries:",
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
    debug_records, debug_full_records = _empty_debug_lists()
    existing_rows_by_game: Dict[str, pd.DataFrame] = {}
    if not existing_rows_df.empty and "gameId" in existing_rows_df.columns:
        grouped_existing_rows = existing_rows_df.groupby(existing_rows_df["gameId"].astype(str), sort=False)
        existing_rows_by_game = {str(game_id): group.copy() for game_id, group in grouped_existing_rows}
    rows_fieldnames = list(existing_rows_df.columns) if not existing_rows_df.empty else None
    rows_file = None
    rows_writer: Optional[csv.DictWriter] = None
    rows_has_header = bool(os.path.exists(rows_out_path)) and os.path.getsize(rows_out_path) > 0

    transcripts_file = None
    assignment_file = None
    debug_file = None
    debug_full_file = None
    write_lock = Lock()

    def _flush_file(handle: Any) -> None:
        handle.flush()
        os.fsync(handle.fileno())

    def _write_rows_chunk(chunk: List[Dict[str, Any]]) -> None:
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

    def _write_transcripts(ctx: GameContext, transcripts: Dict[str, List[str]]) -> None:
        if transcripts_file is None:
            return
        with write_lock:
            for pid in ctx.player_ids:
                sys_text = system_header_plain(
                    ctx.env,
                    ctx.demographics_by_player.get(pid, "") if bool(getattr(args, "include_demographics", False)) else "",
                    bool(args.include_reasoning),
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

    def _run_one_game(game_idx: int, ctx: GameContext, game_seed: int) -> Tuple[
        int,
        GameContext,
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        Dict[str, List[str]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        ]:
        game_debug_records, game_debug_full_records = _empty_debug_lists()
        assigned: Dict[str, Dict[str, Any]] = {}
        assignment_rows: List[Dict[str, Any]] = []
        start_round = int(resume_round_by_game.get(ctx.game_id, 1))
        if archetype_mode:
            assigned_batch = assign_archetypes_for_game(
                mode=archetype_mode,
                game_id=ctx.game_id,
                player_ids=ctx.player_ids,
                env=ctx.env,
                seed=game_seed,
                summary_pool=archetype_summary_pool,
                summary_pool_path=_resolve_archetype_pool_path(args),
                soft_bank_sampler=soft_bank_sampler,
                precomputed_assignment_index=precomputed_assignment_index,
                log_fn=log,
            )
            assigned = assigned_batch.assignments_by_player
            assignment_rows = assigned_batch.manifest_rows
        initial_transcripts = None
        initial_previous_round_rewards = None
        initial_previous_round_punish = None
        if start_round > 1:
            initial_transcripts, initial_previous_round_rewards, initial_previous_round_punish = (
                _restore_macro_state_from_rows(
                    ctx,
                    args,
                    existing_rows_by_game.get(ctx.game_id, pd.DataFrame()),
                    assigned_archetypes=assigned,
                )
            )
        worker_client = client if provider == "local" else _build_llm_client(args, provider)
        rows, transcripts = simulate_game(
            ctx=ctx,
            client=worker_client,
            tok=tok,
            args=args,
            seed=game_seed,
            assigned_archetypes=assigned,
            debug_records=game_debug_records,
            debug_full_records=game_debug_full_records,
            start_round=start_round,
            initial_transcripts=initial_transcripts,
            initial_previous_round_rewards=initial_previous_round_rewards,
            initial_previous_round_punish=initial_previous_round_punish,
            on_round_complete=_on_round_complete,
        )
        return game_idx, ctx, assignment_rows, rows, transcripts, game_debug_records, game_debug_full_records

    game_seed_rng = random.Random(int(args.seed))
    game_seed_by_id = {
        ctx.game_id: game_seed_rng.randrange(0, 2**32 - 1)
        for ctx in selected_contexts
    }

    if resume_enabled:
        log(
            f"[macro] resume {run_ts}: preserved {len(all_rows)} existing rows, "
            f"completed_games={len(completed_game_ids)}, pending_games={len(contexts)}, "
            f"resumed_games={len(resumed_game_ids)}"
        )
        if restored_arg_keys:
            log(f"[macro] restored {len(restored_arg_keys)} args from {config_path}")
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
                        start_round = int(resume_round_by_game.get(ctx.game_id, 1))
                        log(f"[macro] start game {ctx.game_id} ({idx}/{len(contexts)}) from round {start_round}")
                        futures[executor.submit(_run_one_game, idx, ctx, game_seed_by_id[ctx.game_id])] = (idx, ctx)
                    for future in as_completed(futures):
                        game_idx, completed_ctx, assignment_rows, rows, transcripts, dbg, dbg_full = future.result()
                        log(f"[macro] done game {completed_ctx.game_id} -> {len(rows)} rows")
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
                            next_game_idx += 1
            else:
                for idx, ctx in enumerate(contexts, start=1):
                    start_round = int(resume_round_by_game.get(ctx.game_id, 1))
                    log(f"[macro] start game {ctx.game_id} ({idx}/{len(contexts)}) from round {start_round}")

                    debug_records.clear()
                    debug_full_records.clear()
                    game_seed = game_seed_by_id[ctx.game_id]

                    assigned: Dict[str, Dict[str, Any]] = {}
                    assignment_rows = []
                    if archetype_mode:
                        assigned_batch = assign_archetypes_for_game(
                            mode=archetype_mode,
                            game_id=ctx.game_id,
                            player_ids=ctx.player_ids,
                            env=ctx.env,
                            seed=game_seed,
                            summary_pool=archetype_summary_pool,
                            summary_pool_path=_resolve_archetype_pool_path(args),
                            soft_bank_sampler=soft_bank_sampler,
                            precomputed_assignment_index=precomputed_assignment_index,
                            log_fn=log,
                        )
                        assigned = assigned_batch.assignments_by_player
                        assignment_rows = assigned_batch.manifest_rows

                    initial_transcripts = None
                    initial_previous_round_rewards = None
                    initial_previous_round_punish = None
                    if start_round > 1:
                        initial_transcripts, initial_previous_round_rewards, initial_previous_round_punish = (
                            _restore_macro_state_from_rows(
                                ctx,
                                args,
                                existing_rows_by_game.get(ctx.game_id, pd.DataFrame()),
                                assigned_archetypes=assigned,
                            )
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
                        start_round=start_round,
                        initial_transcripts=initial_transcripts,
                        initial_previous_round_rewards=initial_previous_round_rewards,
                        initial_previous_round_punish=initial_previous_round_punish,
                        on_round_complete=_on_round_complete,
                    )
                    all_rows.extend(rows)
                    _write_assignment_chunk(assignment_rows)
                    _write_transcripts(ctx, transcripts)

                    log(f"[macro] done game {ctx.game_id} -> {len(rows)} rows")
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
        log(f"[macro] resume {run_ts}: no unfinished games found")

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
        log(f"[macro] wrote rows -> {rows_out_path}")
        if transcripts_out_path:
            log(f"[macro] wrote transcripts -> {transcripts_out_path}")
        if archetype_assignments_out_path:
            log(f"[macro] wrote archetype assignments -> {archetype_assignments_out_path}")
        if debug_jsonl_path:
            log(f"[macro] wrote debug -> {debug_jsonl_path}")
        if debug_full_jsonl_path:
            log(f"[macro] wrote full debug -> {debug_full_jsonl_path}")
        log(f"[macro] wrote config -> {config_path}")

    return df_out, {
        "rows": rows_out_path,
        "transcripts": transcripts_out_path,
        "archetype_assignments": archetype_assignments_out_path,
        "debug": debug_jsonl_path,
        "debug_full": debug_full_jsonl_path,
        "config": config_path,
        "directory": run_dir,
    }
