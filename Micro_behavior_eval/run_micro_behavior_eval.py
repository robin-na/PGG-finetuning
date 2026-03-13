from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluator import run_micro_behavior_eval  # noqa: E402
from Persona.archetype_sampling.runtime import canonicalize_archetype_mode, default_summary_pool_for_mode  # noqa: E402
from utils import log  # noqa: E402


DEFAULT_ROUNDS_CSV = "data/raw_data/validation_wave/player-rounds.csv"
DEFAULT_ANALYSIS_CSV = "data/processed_data/df_analysis_val.csv"
DEFAULT_DEMOGRAPHICS_CSV = "demographics/demographics_numeric_val.csv"
DEFAULT_PLAYERS_CSV = "data/raw_data/validation_wave/players.csv"
DEFAULT_GAMES_CSV = "data/raw_data/validation_wave/games.csv"


@dataclass
class Args:
    provider: str = field(default="local", metadata={"help": "Inference provider: local | openai | vllm."})
    openai_model: Optional[str] = field(default=None)
    openai_api_key: Optional[str] = field(default=None)
    openai_api_key_env: str = field(default="OPENAI_API_KEY")
    openai_async: bool = field(
        default=False,
        metadata={"help": "If true, use asyncio.gather with a thread pool for concurrent OpenAI calls."},
    )
    openai_max_concurrency: int = field(
        default=8,
        metadata={"help": "Max concurrent OpenAI calls when --openai_async is enabled."},
    )
    vllm_model: Optional[str] = field(
        default=None,
        metadata={"help": "Served model name for --provider vllm. Defaults to --base_model."},
    )
    vllm_base_url: str = field(
        default="http://localhost:8000/v1",
        metadata={"help": "OpenAI-compatible vLLM base URL when --provider vllm is used."},
    )
    vllm_api_key: Optional[str] = field(default=None)
    vllm_api_key_env: str = field(default="VLLM_API_KEY")
    vllm_max_concurrency: int = field(
        default=8,
        metadata={"help": "Max concurrent vLLM requests per game stage."},
    )

    base_model: str = field(default="meta-llama/Llama-3.1-8B-Instruct")
    adapter_path: Optional[str] = field(default="out/llama31-8b-lora-pgg-ptc/checkpoint-489")
    use_peft: bool = field(default=True)
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "If true, load the local HF model with bitsandbytes 8-bit quantization."},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "If true, load the local HF model with bitsandbytes 4-bit quantization."},
    )
    quant_compute_dtype: str = field(
        default="auto",
        metadata={"help": "bitsandbytes compute dtype for local quantized loading: auto | bf16 | fp16 | fp32."},
    )

    data_root: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional dataset root containing raw_data/ processed_data/ demographics/. "
                "If set, default CSV inputs are auto-resolved from this root."
            )
        },
    )
    wave: str = field(
        default="validation_wave",
        metadata={"help": "Wave used for raw_data path and *_val/*_learn file selection."},
    )
    rounds_csv: str = field(default=DEFAULT_ROUNDS_CSV)
    analysis_csv: str = field(default=DEFAULT_ANALYSIS_CSV)
    demographics_csv: str = field(default=DEFAULT_DEMOGRAPHICS_CSV)
    players_csv: str = field(default=DEFAULT_PLAYERS_CSV)
    games_csv: str = field(default=DEFAULT_GAMES_CSV)

    output_root: str = field(default="outputs/default/runs/source_default/micro_behavior_eval")
    run_id: Optional[str] = field(default=None)
    resume_from_run: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Resume an existing run in place. Accepts either a run ID under --output_root "
                "or an explicit run directory path."
            )
        },
    )
    rows_out_path: str = field(default="output/micro_behavior_eval.csv")
    transcripts_out_path: Optional[str] = field(default="output/history_transcripts.jsonl")
    archetype_assignments_out_path: Optional[str] = field(default="output/archetype_assignments.jsonl")
    debug_jsonl_path: Optional[str] = field(default="output/micro_behavior_debug.jsonl")
    debug_full_jsonl_path: Optional[str] = field(default=None)

    start_round: int = field(
        default=1,
        metadata={"help": "First round index T to evaluate (history uses rounds <= T-1)."},
    )
    game_ids: Optional[str] = field(
        default=None,
        metadata={"help": "Optional comma-separated game IDs (or treatment names) to evaluate."},
    )
    max_games: Optional[int] = field(default=None)
    skip_no_actual: bool = field(
        default=True,
        metadata={"help": "If true, drop rows where round-T actual behavior is unavailable (e.g. exits)."},
    )

    temperature: float = field(default=1.0)
    top_p: float = field(default=1.0)
    seed: int = field(default=0)
    contrib_max_new_tokens: int = field(default=512)
    actions_max_new_tokens: int = field(default=512)
    action_prompt_mode: str = field(
        default="binary_targets",
        metadata={
            "help": "Action prompt/output mode: binary_targets (default, one unit per selected target) or legacy_units."
        },
    )
    action_continuation_gate: bool = field(
        default=True,
        metadata={"help": "If true, probabilistically thin repeated same-dyad punish/reward actions from the previous round."},
    )
    punish_continuation_keep_prob: float = field(
        default=0.5,
        metadata={"help": "Keep probability for repeated punish dyads when continuation gating is enabled."},
    )
    reward_continuation_keep_prob: float = field(
        default=0.35,
        metadata={"help": "Keep probability for repeated reward dyads when continuation gating is enabled."},
    )
    include_reasoning: bool = field(
        default=False,
        metadata={"help": "If true, request a short reasoning field in JSON outputs."},
    )
    include_demographics: bool = field(
        default=False,
        metadata={"help": "If true, include demographic information in the prompt header."},
    )
    archetype: Optional[str] = field(
        default=None,
        metadata={"help": "Optional archetype mode: matched_summary | random_summary | config_bank_archetype"},
    )
    archetype_summary_pool: str = field(
        default="Persona/archetype_oracle_gpt51_val.jsonl",
        metadata={"help": "JSONL file containing archetype summary entries."},
    )
    archetype_assignments_in_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional JSONL file of precomputed target-game archetype assignments."},
    )
    archetype_soft_bank_temperature: float = field(
        default=0.07,
        metadata={"help": "Temperature for config_bank_archetype sampling over the learn-wave persona bank."},
    )
    persona: Optional[str] = field(
        default=None,
        metadata={"help": argparse.SUPPRESS},
    )
    persona_summary_pool: Optional[str] = field(
        default=None,
        metadata={"help": argparse.SUPPRESS},
    )
    max_parallel_games: int = field(
        default=1,
        metadata={"help": "Number of games to process concurrently. Applied only for remote providers."},
    )

    debug_print: bool = field(default=False)
    debug_level: str = field(
        default="full",
        metadata={"help": "Debug output level: full | compact | off"},
    )
    debug_compact: bool = field(
        default=False,
        metadata={"help": "If true, store compact debug records (metadata + excerpt/hash)."},
    )


def main(args: Args):
    _normalize_archetype_args(args)
    _apply_data_root_paths(args)
    _, output_paths = run_micro_behavior_eval(args)
    if args.debug_print:
        log(f"[micro] outputs directory -> {output_paths.get('directory')}")
        log(f"[micro] rows -> {output_paths.get('rows')}")
        log(f"[micro] transcripts -> {output_paths.get('transcripts')}")
        log(f"[micro] debug -> {output_paths.get('debug')}")
        log(f"[micro] config -> {output_paths.get('config')}")


def _apply_data_root_paths(args: Args) -> None:
    data_root = str(getattr(args, "data_root", "") or "").strip()
    if not data_root:
        return

    wave = str(getattr(args, "wave", "validation_wave") or "validation_wave").strip()
    if wave not in {"validation_wave", "learning_wave"}:
        raise ValueError(
            f"Unsupported --wave '{wave}'. Allowed values: validation_wave, learning_wave."
        )

    root = data_root
    analysis_name = "df_analysis_val.csv" if wave == "validation_wave" else "df_analysis_learn.csv"
    demographics_name = (
        "demographics_numeric_val.csv"
        if wave == "validation_wave"
        else "demographics_numeric_learn.csv"
    )

    derived = {
        "rounds_csv": os.path.join(root, "raw_data", wave, "player-rounds.csv"),
        "analysis_csv": os.path.join(root, "processed_data", analysis_name),
        "demographics_csv": os.path.join(root, "demographics", demographics_name),
        "players_csv": os.path.join(root, "raw_data", wave, "players.csv"),
        "games_csv": os.path.join(root, "raw_data", wave, "games.csv"),
    }

    default_values = {
        "rounds_csv": DEFAULT_ROUNDS_CSV,
        "analysis_csv": DEFAULT_ANALYSIS_CSV,
        "demographics_csv": DEFAULT_DEMOGRAPHICS_CSV,
        "players_csv": DEFAULT_PLAYERS_CSV,
        "games_csv": DEFAULT_GAMES_CSV,
    }

    # Respect explicit CSV overrides; auto-fill only when still on the built-in defaults (or blank).
    for key, path in derived.items():
        current = str(getattr(args, key, "") or "").strip()
        if (not current) or (current == default_values[key]):
            setattr(args, key, path)


def _normalize_archetype_args(args: Args) -> None:
    archetype_mode = str(getattr(args, "archetype", "") or "").strip()
    persona_mode = str(getattr(args, "persona", "") or "").strip()
    if archetype_mode and persona_mode and archetype_mode != persona_mode:
        raise ValueError(
            "Conflicting values for --archetype and deprecated --persona."
        )
    if not archetype_mode:
        archetype_mode = persona_mode
    archetype_mode = canonicalize_archetype_mode(archetype_mode)
    args.archetype = archetype_mode or None

    archetype_pool = str(getattr(args, "archetype_summary_pool", "") or "").strip()
    persona_pool = str(getattr(args, "persona_summary_pool", "") or "").strip()
    if archetype_pool and persona_pool and archetype_pool != persona_pool:
        raise ValueError(
            "Conflicting values for --archetype_summary_pool and deprecated --persona_summary_pool."
        )
    if not archetype_pool:
        archetype_pool = persona_pool
    if not archetype_pool:
        archetype_pool = "Persona/archetype_oracle_gpt51_val.jsonl"
    default_pool = default_summary_pool_for_mode(args.archetype or "")
    if args.archetype and archetype_pool == "Persona/archetype_oracle_gpt51_val.jsonl":
        archetype_pool = default_pool
    args.archetype_summary_pool = archetype_pool
    args.persona = None
    args.persona_summary_pool = None


if __name__ == "__main__":
    parser = HfArgumentParser(Args)
    parsed, unknown = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    cfg = parsed[0] if isinstance(parsed, (list, tuple)) else parsed
    if unknown:
        log("[note] unknown args (ignored):", unknown)
    main(cfg)
