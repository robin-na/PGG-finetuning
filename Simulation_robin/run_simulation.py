from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from transformers import HfArgumentParser

from simulator import simulate_games
from utils import log

from pathlib import Path


try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


@dataclass
class Args:
    provider: str = field(default="local")
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

    base_model: str = field(default="meta-llama/Llama-3.1-8B-Instruct")
    adapter_path: Optional[str] = field(default="out/llama31-8b-lora-pgg-ptc/checkpoint-489")
    use_peft: bool = field(default=True)

    env_csv: str = field(default="data/processed_data/df_analysis_val.csv")
    output_root: str = field(
        default="output",
        metadata={"help": "Base output directory to store experiment/timestamp folders."},
    )
    run_id: Optional[str] = field(
        default=None,
        metadata={"help": "Optional run identifier; defaults to timestamp."},
    )
    group_by_game: bool = field(
        default=True,
        metadata={"help": "Retained for CLI compatibility; outputs are always grouped by experiment."},
    )
    rows_out_path: Optional[str] = field(default="output/participant_sim.csv")
    transcripts_out_path: Optional[str] = field(default="output/participant_transcripts.jsonl")
    debug_jsonl_path: Optional[str] = field(default="output/participant_debug.jsonl")

    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 0
    contrib_max_new_tokens: int = 128
    chat_max_new_tokens: int = 128
    actions_max_new_tokens: int = 128
    include_reasoning: bool = field(
        default=False,
        metadata={"help": "If true, request a short Reasoning line followed by a strict Answer line."},
    )
    persona: Optional[str] = field(
        default=None,
        metadata={"help": "Optional persona mode (e.g., random_full_transcript, random_summary, finetuned_summary)."},
    )
    person: Optional[str] = field(
        default=None,
        metadata={"help": "Alias for --persona (backwards compatibility)."},
    )
    persona_pool: str = field(
        default="Persona/transcripts_learn.jsonl",
        metadata={"help": "JSONL file containing persona transcript entries."},
    )
    persona_summary_pool: str = field(
        default="Persona/summary_gpt51_learn.jsonl",
        metadata={"help": "JSONL file containing persona summary entries."},
    )
    persona_finetuned_pool: str = field(
        default="Persona/LLM_mapped/Qwen/Qwen3-8B/persona_type_outputs.jsonl",
        metadata={"help": "JSONL file containing fine-tuned persona outputs keyed by CONFIG_treatmentName."},
    )

    debug_print: bool = False
    debug_level: str = field(
        default="full",
        metadata={"help": "Debug output level: full | compact | off"},
    )
    debug_compact: bool = field(
        default=False,
        metadata={"help": "If true, store compact debug records (metadata + excerpt/hash)."},
    )
    debug_full_jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional JSONL path to store full prompts/outputs when compact logging is enabled."},
    )
    max_parallel_games: int = field(
        default=1,
        metadata={"help": "Run up to N environments concurrently. Use 1 for local HF GPU models."},
    )


@dataclass
class CLIArgs(Args):
    pass


def main(args: CLIArgs):
    # Backwards-compat: some scripts used --person instead of --persona.
    if getattr(args, "person", None) and not getattr(args, "persona", None):
        args.persona = args.person
    if args.debug_print:
        log(args)
    if args.debug_compact:
        args.debug_level = "compact"
    if args.debug_level not in {"full", "compact", "off"}:
        log(f"[warn] unknown debug_level='{args.debug_level}', defaulting to 'full'")
        args.debug_level = "full"
    if args.debug_level == "off":
        args.debug_jsonl_path = None
        args.debug_full_jsonl_path = None

    env_df = pd.read_csv(Path(args.env_csv))
    if "name" in env_df.columns:
        before = len(env_df)
        env_df = env_df.drop_duplicates(subset="name", keep="first")
        if args.debug_print:
            log(f"[env] dedup by name: {before} -> {len(env_df)} rows")

    df_all, transcripts_all, output_paths = simulate_games(
        env_df=env_df,
        args=args,
    )

    if args.debug_print:
        log(f"[ptc] simulated rows: {len(df_all)} across {len(transcripts_all)} experiments")
        for game_id, paths in output_paths.items():
            log(f"[ptc] outputs for {game_id} → {paths.get('directory')}")
            log(f"[ptc]   rows: {paths.get('rows')}")
            log(f"[ptc]   transcripts: {paths.get('transcripts')}")
            if paths.get("debug"):
                log(f"[ptc]   debug: {paths.get('debug')}")
            if paths.get("debug_full") and args.debug_level != "off":
                log(f"[ptc]   debug_full: {paths.get('debug_full')}")
            log(f"[ptc]   config: {paths.get('config')}")


if __name__ == "__main__":
    parser = HfArgumentParser(CLIArgs)
    parsed, unknown = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    cfg = parsed[0] if isinstance(parsed, (list, tuple)) else parsed
    if unknown:
        log("[note] unknown args (ignored):", unknown)
    main(cfg)
