from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser

try:
    from .concordia_simulator import run_macro_simulation_eval_concordia
    from .run_simulation import Args, _apply_data_root_paths, _normalize_archetype_args
    from .utils import log
except ImportError:
    from concordia_simulator import run_macro_simulation_eval_concordia
    from run_simulation import Args, _apply_data_root_paths, _normalize_archetype_args
    from utils import log


@dataclass
class ConcordiaArgs(Args):
    output_root: str = field(default="outputs/default/runs/source_default/macro_simulation_eval_concordia")
    rows_out_path: str = field(default="output/macro_simulation_eval_concordia.csv")
    transcripts_out_path: Optional[str] = field(default="output/history_transcripts_concordia.jsonl")
    debug_jsonl_path: Optional[str] = field(default=None)
    debug_full_jsonl_path: Optional[str] = field(default=None)

    concordia_import_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional checkout root for importing Concordia when it is not installed in the active env."},
    )
    concordia_agent_prefab: str = field(
        default="rational",
        metadata={"help": "Concordia entity prefab to use: rational | basic | basic_with_plan."},
    )
    concordia_embedder: str = field(
        default="hash",
        metadata={"help": "Sentence embedder for Concordia memory retrieval: hash | openai."},
    )
    concordia_embedding_model: str = field(
        default="text-embedding-3-small",
        metadata={"help": "Embedding model name used when --concordia_embedder openai is selected."},
    )
    concordia_hash_dim: int = field(
        default=384,
        metadata={"help": "Dimensionality of the local hashing embedder used for Concordia memory."},
    )
    concordia_goal: str = field(
        default=(
            "Maximize your cumulative payoff in this public goods game while staying behaviorally "
            "consistent with your persona, recent observations, and the game rules."
        ),
        metadata={"help": "High-level standing goal passed into the Concordia agent prefab."},
    )


def main(args: ConcordiaArgs) -> None:
    _normalize_archetype_args(args)
    _apply_data_root_paths(args)
    _, output_paths = run_macro_simulation_eval_concordia(args)
    if args.debug_print:
        log(f"[macro-concordia] outputs directory -> {output_paths.get('directory')}")
        log(f"[macro-concordia] rows -> {output_paths.get('rows')}")
        log(f"[macro-concordia] transcripts -> {output_paths.get('transcripts')}")
        log(f"[macro-concordia] concordia logs -> {output_paths.get('concordia_logs')}")
        log(f"[macro-concordia] config -> {output_paths.get('config')}")


if __name__ == "__main__":
    parser = HfArgumentParser(ConcordiaArgs)
    parsed, unknown = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    cfg = parsed[0] if isinstance(parsed, (list, tuple)) else parsed
    if unknown:
        log("[note] unknown args (ignored):", unknown)
    main(cfg)
