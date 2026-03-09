from __future__ import annotations

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)
for path in (SCRIPT_DIR, PACKAGE_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from simulator import (
        DEFAULT_ANALYSIS_CSV,
        DEFAULT_DEMOGRAPHICS_CSV,
        DEFAULT_PLAYERS_CSV,
        DEFAULT_ROUNDS_CSV,
        run_macro_statistical_simulation,
    )
    from simulation_statistical.paths import BENCHMARK_DATA_ROOT, MACRO_RUN_ROOT
except ImportError:
    from simulation_statistical.macro.simulator import (
        DEFAULT_ANALYSIS_CSV,
        DEFAULT_DEMOGRAPHICS_CSV,
        DEFAULT_PLAYERS_CSV,
        DEFAULT_ROUNDS_CSV,
        run_macro_statistical_simulation,
    )
    from simulation_statistical.paths import BENCHMARK_DATA_ROOT, MACRO_RUN_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run macro-level statistical PGG simulation.")
    parser.add_argument("--data_root", type=str, default=BENCHMARK_DATA_ROOT)
    parser.add_argument("--wave", type=str, default="validation_wave")
    parser.add_argument("--analysis_csv", type=str, default=DEFAULT_ANALYSIS_CSV)
    parser.add_argument("--rounds_csv", type=str, default=DEFAULT_ROUNDS_CSV)
    parser.add_argument("--players_csv", type=str, default=DEFAULT_PLAYERS_CSV)
    parser.add_argument("--demographics_csv", type=str, default=DEFAULT_DEMOGRAPHICS_CSV)
    parser.add_argument(
        "--output_root",
        type=str,
        default=MACRO_RUN_ROOT,
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--rows_out_path", type=str, default="output/macro_simulation_eval.csv")
    parser.add_argument("--transcripts_out_path", type=str, default="output/history_transcripts.jsonl")
    parser.add_argument("--debug_jsonl_path", type=str, default="output/macro_statistical_debug.jsonl")
    parser.add_argument("--game_ids", type=str, default=None)
    parser.add_argument("--max_games", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_probability", type=float, default=0.10)
    parser.add_argument("--action_magnitude", type=int, default=1)
    parser.add_argument("--debug_print", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_macro_statistical_simulation(args)


if __name__ == "__main__":
    main()
