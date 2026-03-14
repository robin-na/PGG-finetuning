from __future__ import annotations

import csv
import json
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from Persona.archetype_sampling.runtime import (
    ArchetypeSummaryPool,
    PrecomputedAssignmentIndex,
    SoftBankSummarySampler,
    SUPPORTED_ARCHETYPE_MODES,
    assign_archetypes_for_game,
    load_finished_summary_pool,
    load_precomputed_assignment_index,
)
from concordia.language_model import language_model as concordia_language_model
from concordia.prefabs.entity import basic as concordia_basic_entity
from concordia.prefabs.entity import basic_with_plan as concordia_basic_with_plan_entity
from concordia.prefabs.entity import rational as concordia_rational_entity
from concordia.prefabs.simulation import generic as concordia_generic_simulation
from concordia.typing import prefab as concordia_prefab
from concordia.utils import structured_logging as concordia_structured_logging

try:
    from .concordia_pgg import PublicGoodsGameMasterPrefab, build_simultaneous_engine
    from .concordia_runtime import build_language_model_adapter, build_sentence_embedder
    from .llm_client import LLMClient
    from .model_loader import load_model
    from .prompt_builder import system_header_plain
    from .simulator import (
        GameContext,
        _build_llm_client,
        _model_config,
        _resolve_archetype_assignment_manifest_path,
        _resolve_archetype_mode,
        _resolve_archetype_pool_path,
        _serialize_args,
        _write_config_json,
        build_game_contexts,
    )
    from .utils import log, relocate_output, timestamp_yymmddhhmm
except ImportError:
    from concordia_pgg import PublicGoodsGameMasterPrefab, build_simultaneous_engine
    from concordia_runtime import build_language_model_adapter, build_sentence_embedder
    from llm_client import LLMClient
    from model_loader import load_model
    from prompt_builder import system_header_plain
    from simulator import (
        GameContext,
        _build_llm_client,
        _model_config,
        _resolve_archetype_assignment_manifest_path,
        _resolve_archetype_mode,
        _resolve_archetype_pool_path,
        _serialize_args,
        _write_config_json,
        build_game_contexts,
    )
    from utils import log, relocate_output, timestamp_yymmddhhmm


_DEFAULT_GOAL = (
    "Maximize your cumulative payoff in this public goods game while staying behaviorally "
    "consistent with your persona, recent observations, and the game rules."
)


def _player_prefab_module(prefab_name: str) -> Any:
    normalized = str(prefab_name or "rational").strip().lower()
    if normalized == "rational":
        return concordia_rational_entity
    if normalized == "basic":
        return concordia_basic_entity
    if normalized == "basic_with_plan":
        return concordia_basic_with_plan_entity
    raise ValueError("Unsupported Concordia prefab. Use `rational`, `basic`, or `basic_with_plan`.")


def _write_rows(rows_path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(rows_path), exist_ok=True)
    rows_list = [dict(row) for row in rows]
    with open(rows_path, "w", newline="", encoding="utf-8") as handle:
        if not rows_list:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows_list[0].keys()))
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


def _write_transcripts(
    transcripts_path: Optional[str],
    contexts: Sequence[GameContext],
    transcripts_by_game: Mapping[str, Dict[str, List[str]]],
    *,
    include_demographics: bool,
    include_reasoning: bool,
) -> None:
    if not transcripts_path:
        return
    os.makedirs(os.path.dirname(transcripts_path), exist_ok=True)
    with open(transcripts_path, "w", encoding="utf-8") as handle:
        for ctx in contexts:
            transcripts = transcripts_by_game.get(ctx.game_id, {})
            for pid in ctx.player_ids:
                system_text = system_header_plain(
                    ctx.env,
                    ctx.demographics_by_player.get(pid, "") if include_demographics else "",
                    include_reasoning,
                )
                body = "\n".join(transcripts.get(pid, ["# GAME STARTS", "# GAME COMPLETE"]))
                handle.write(
                    json.dumps(
                        {
                            "experiment": ctx.game_id,
                            "participant": pid,
                            "text": f"{system_text}\n{body}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def _write_assignment_rows(path: Optional[str], rows: Sequence[Mapping[str, Any]]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _write_simulation_logs(
    logs_dir: str,
    game_id: str,
    simulation_log: concordia_structured_logging.SimulationLog,
) -> Dict[str, str]:
    os.makedirs(logs_dir, exist_ok=True)
    json_path = os.path.join(logs_dir, f"{game_id}.simulation_log.json")
    html_path = os.path.join(logs_dir, f"{game_id}.simulation_log.html")
    with open(json_path, "w", encoding="utf-8") as handle:
        handle.write(simulation_log.to_json(indent=2))
        handle.write("\n")
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(simulation_log.to_html(title=f"Concordia Simulation Log: {game_id}"))
    return {"json": json_path, "html": html_path}


def _load_archetype_runtime(args: Any) -> Tuple[
    str,
    Optional[ArchetypeSummaryPool],
    Optional[PrecomputedAssignmentIndex],
    Optional[SoftBankSummarySampler],
]:
    archetype_mode = _resolve_archetype_mode(args)
    manifest_path = _resolve_archetype_assignment_manifest_path(args)
    if not archetype_mode:
        if manifest_path:
            raise ValueError(
                "--archetype_assignments_in_path requires an archetype mode. "
                "Use --archetype config_bank_archetype or another supported mode."
            )
        return "", None, None, None
    if archetype_mode not in SUPPORTED_ARCHETYPE_MODES:
        raise ValueError(
            f"Unsupported archetype mode '{archetype_mode}'. Allowed values: "
            f"{', '.join(sorted(SUPPORTED_ARCHETYPE_MODES))}."
        )
    if manifest_path:
        return archetype_mode, None, load_precomputed_assignment_index(manifest_path), None
    summary_pool = load_finished_summary_pool(_resolve_archetype_pool_path(args))
    sampler = None
    if archetype_mode == "config_bank_archetype":
        sampler = SoftBankSummarySampler(
            summary_pool_path=_resolve_archetype_pool_path(args),
            temperature=float(getattr(args, "archetype_soft_bank_temperature", 0.07)),
        )
    return archetype_mode, summary_pool, None, sampler


def _build_simulation_config(
    *,
    ctx: GameContext,
    args: Any,
    assigned_archetypes: Mapping[str, Dict[str, Any]],
) -> concordia_prefab.Config:
    player_prefab = _player_prefab_module(getattr(args, "concordia_agent_prefab", "rational"))
    prefabs = {
        "player": player_prefab.Entity(),
        "macro_pgg_game_master": PublicGoodsGameMasterPrefab(),
    }
    instances: List[concordia_prefab.InstanceConfig] = []
    goal_text = str(getattr(args, "concordia_goal", "") or _DEFAULT_GOAL).strip()
    for pid in ctx.player_ids:
        instances.append(
            concordia_prefab.InstanceConfig(
                prefab="player",
                role=concordia_prefab.Role.ENTITY,
                params={
                    "name": ctx.avatar_by_player[pid],
                    "goal": goal_text,
                },
            )
        )
    instances.append(
        concordia_prefab.InstanceConfig(
            prefab="macro_pgg_game_master",
            role=concordia_prefab.Role.GAME_MASTER,
            params={
                "name": f"macro_pgg_game_master::{ctx.game_id}",
                "ctx": ctx,
                "args": args,
                "assigned_archetypes": dict(assigned_archetypes),
            },
        )
    )
    return concordia_prefab.Config(
        prefabs=prefabs,
        instances=instances,
        default_premise="",
        default_max_steps=max(8, int(ctx.env.get("CONFIG_numRounds", 0) or 0) * 4),
    )


def _run_one_game_native(
    *,
    ctx: GameContext,
    args: Any,
    client: LLMClient,
    assigned_archetypes: Mapping[str, Dict[str, Any]],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], concordia_structured_logging.SimulationLog]:
    lm_adapter = build_language_model_adapter(
        language_model_lib=concordia_language_model,
        client=client,
        seed=int(seed),
        default_temperature=float(getattr(args, "temperature", 1.0)),
        default_top_p=float(getattr(args, "top_p", 1.0)),
    )
    sentence_embedder = build_sentence_embedder(args, client)
    sim_config = _build_simulation_config(ctx=ctx, args=args, assigned_archetypes=assigned_archetypes)
    simulation = concordia_generic_simulation.Simulation(
        config=sim_config,
        model=lm_adapter,
        embedder=sentence_embedder,
        engine=build_simultaneous_engine(),
    )
    simulation_log = simulation.play(premise="", max_steps=sim_config.default_max_steps)
    game_master = simulation.get_game_masters()[0]
    rows = game_master.get_rows()
    transcripts = game_master.get_transcripts()
    for pid in ctx.player_ids:
        transcripts[pid].append("# GAME COMPLETE")
    return rows, transcripts, simulation_log


def run_macro_simulation_eval_concordia(args: Any) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    if getattr(args, "concordia_import_path", None):
        log(
            "[warn] --concordia_import_path is ignored by the native Concordia backend. "
            "Install `gdm-concordia` in the active environment instead."
        )
    if getattr(args, "resume_from_run", None):
        raise NotImplementedError(
            "The Concordia-backed macro simulator does not support --resume_from_run yet."
        )
    if int(getattr(args, "max_parallel_games", 1) or 1) != 1:
        raise NotImplementedError(
            "The Concordia-backed macro simulator currently supports only --max_parallel_games 1."
        )

    run_ts = str(getattr(args, "run_id", None) or timestamp_yymmddhhmm())
    run_dir = os.path.join(args.output_root, run_ts)
    os.makedirs(run_dir, exist_ok=True)

    rows_out_path = relocate_output(args.rows_out_path, run_dir)
    transcripts_out_path = (
        relocate_output(args.transcripts_out_path, run_dir)
        if getattr(args, "transcripts_out_path", None)
        else None
    )
    archetype_assignments_out_path = (
        relocate_output(args.archetype_assignments_out_path, run_dir)
        if getattr(args, "archetype_assignments_out_path", None)
        else None
    )
    simulation_logs_dir = os.path.join(run_dir, "concordia_logs")
    config_path = os.path.join(run_dir, "config.json")

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
        wanted = {item.strip() for item in str(args.game_ids).split(",") if item.strip()}
        contexts = [ctx for ctx in contexts if ctx.game_id in wanted or ctx.game_name in wanted]
    if args.max_games is not None:
        contexts = contexts[: int(args.max_games)]
    if not contexts:
        raise ValueError("No games selected for Concordia-backed macro simulation.")

    provider = str(args.provider or "local").strip().lower()
    tok: Optional[Any] = None
    model: Optional[Any] = None
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

    archetype_mode, summary_pool, precomputed_index, soft_bank_sampler = _load_archetype_runtime(args)
    all_rows: List[Dict[str, Any]] = []
    transcripts_by_game: Dict[str, Dict[str, List[str]]] = {}
    assignment_rows: List[Dict[str, Any]] = []
    simulation_log_paths_by_game: Dict[str, Dict[str, str]] = {}
    seed_rng = random.Random(int(getattr(args, "seed", 0)))

    for game_idx, ctx in enumerate(contexts):
        assigned: Dict[str, Dict[str, Any]] = {}
        game_seed = seed_rng.randrange(0, 2**32 - 1)
        if archetype_mode:
            assigned_batch = assign_archetypes_for_game(
                mode=archetype_mode,
                game_id=ctx.game_id,
                player_ids=ctx.player_ids,
                env=ctx.env,
                seed=game_seed,
                summary_pool=summary_pool,
                summary_pool_path=_resolve_archetype_pool_path(args),
                soft_bank_sampler=soft_bank_sampler,
                precomputed_assignment_index=precomputed_index,
                log_fn=log,
            )
            assigned = assigned_batch.assignments_by_player
            assignment_rows.extend(assigned_batch.manifest_rows)
        if getattr(args, "debug_print", False):
            log(
                f"[macro-concordia] game {game_idx + 1}/{len(contexts)} "
                f"gameId={ctx.game_id} players={len(ctx.player_ids)}"
            )
        game_rows, game_transcripts, simulation_log = _run_one_game_native(
            ctx=ctx,
            args=args,
            client=client,
            assigned_archetypes=assigned,
            seed=game_seed,
        )
        all_rows.extend(game_rows)
        transcripts_by_game[ctx.game_id] = game_transcripts
        simulation_log_paths_by_game[ctx.game_id] = _write_simulation_logs(
            simulation_logs_dir,
            ctx.game_id,
            simulation_log,
        )

    _write_rows(rows_out_path, all_rows)
    _write_transcripts(
        transcripts_out_path,
        contexts,
        transcripts_by_game,
        include_demographics=bool(getattr(args, "include_demographics", False)),
        include_reasoning=bool(getattr(args, "include_reasoning", False)),
    )
    if archetype_mode:
        _write_assignment_rows(archetype_assignments_out_path, assignment_rows)
    else:
        archetype_assignments_out_path = None

    args_payload = _serialize_args(args)
    if args_payload.get("openai_api_key") is not None:
        args_payload["openai_api_key"] = "***redacted***"
    if args_payload.get("vllm_api_key") is not None:
        args_payload["vllm_api_key"] = "***redacted***"

    config_payload = {
        "run_timestamp": run_ts,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "macro_simulation_eval_concordia_native",
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
        "model": _model_config(args, run_ts),
        "concordia": {
            "engine": "simultaneous",
            "simulation_prefab": "concordia.prefabs.simulation.generic.Simulation",
            "entity_prefab": getattr(args, "concordia_agent_prefab", "rational"),
            "embedder": getattr(args, "concordia_embedder", "hash"),
            "embedding_model": getattr(args, "concordia_embedding_model", "text-embedding-3-small"),
            "hash_dim": getattr(args, "concordia_hash_dim", 384),
            "goal": getattr(args, "concordia_goal", _DEFAULT_GOAL),
        },
        "args": args_payload,
        "outputs": {
            "directory": run_dir,
            "rows": rows_out_path,
            "transcripts": transcripts_out_path,
            "archetype_assignments": archetype_assignments_out_path,
            "concordia_logs": simulation_log_paths_by_game,
            "debug": None,
            "debug_full": None,
        },
    }
    _write_config_json(config_path, config_payload)

    result_df = pd.DataFrame(all_rows)
    output_paths = {
        "directory": run_dir,
        "rows": rows_out_path,
        "transcripts": transcripts_out_path,
        "archetype_assignments": archetype_assignments_out_path,
        "concordia_logs": simulation_logs_dir,
        "debug": None,
        "debug_full": None,
        "config": config_path,
    }
    return result_df, output_paths
