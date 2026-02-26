from __future__ import annotations

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
from utils import log  # noqa: E402


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

    rounds_csv: str = field(default="data/raw_data/validation_wave/player-rounds.csv")
    analysis_csv: str = field(default="data/processed_data/df_analysis_val.csv")
    demographics_csv: str = field(default="demographics/demographics_numeric_val.csv")
    players_csv: str = field(default="data/raw_data/validation_wave/players.csv")
    games_csv: str = field(default="data/raw_data/validation_wave/games.csv")

    output_root: str = field(default="Micro_behavior_eval/output")
    run_id: Optional[str] = field(default=None)
    rows_out_path: str = field(default="output/micro_behavior_eval.csv")
    transcripts_out_path: Optional[str] = field(default="output/history_transcripts.jsonl")
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
    contrib_max_new_tokens: int = field(default=128)
    actions_max_new_tokens: int = field(default=128)
    include_reasoning: bool = field(
        default=False,
        metadata={"help": "If true, request a short reasoning field in JSON outputs."},
    )
    max_parallel_games: int = field(
        default=1,
        metadata={"help": "Reserved for future use; current implementation runs sequentially."},
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
    _, output_paths = run_micro_behavior_eval(args)
    if args.debug_print:
        log(f"[micro] outputs directory -> {output_paths.get('directory')}")
        log(f"[micro] rows -> {output_paths.get('rows')}")
        log(f"[micro] transcripts -> {output_paths.get('transcripts')}")
        log(f"[micro] debug -> {output_paths.get('debug')}")
        log(f"[micro] config -> {output_paths.get('config')}")


if __name__ == "__main__":
    parser = HfArgumentParser(Args)
    parsed, unknown = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    cfg = parsed[0] if isinstance(parsed, (list, tuple)) else parsed
    if unknown:
        log("[note] unknown args (ignored):", unknown)
    main(cfg)
