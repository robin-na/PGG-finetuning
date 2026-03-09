from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)
for path in (SCRIPT_DIR, PACKAGE_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from Micro_behavior_eval.analysis.run_analysis import AnalysisArgs, run_analysis  # noqa: E402
from supplemental import generate_punishment_target_report  # noqa: E402
from simulation_statistical.paths import MICRO_REPORT_ROOT, MICRO_RUN_ROOT  # noqa: E402


def _parse_bool(value: str, default: bool) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze statistical micro simulation outputs.")
    parser.add_argument("--eval_csv", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument(
        "--eval_root",
        type=str,
        default=MICRO_RUN_ROOT,
    )
    parser.add_argument("--compare_run_ids", type=str, default=None)
    parser.add_argument("--compare_labels", type=str, default=None)
    parser.add_argument(
        "--analysis_root",
        type=str,
        default=MICRO_REPORT_ROOT,
    )
    parser.add_argument("--analysis_run_id", type=str, default=None)
    parser.add_argument("--min_round", type=int, default=None)
    parser.add_argument("--max_round", type=int, default=None)
    parser.add_argument("--skip_no_actual", type=str, default="true")
    parser.add_argument("--include_prev_round_baseline", type=str, default="false")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--debug_print", type=str, default="false")
    parser.add_argument("--analysis_csv", type=str, default=None)
    return parser


def _to_analysis_args(ns: argparse.Namespace) -> AnalysisArgs:
    return AnalysisArgs(
        eval_csv=ns.eval_csv,
        run_id=ns.run_id,
        eval_root=ns.eval_root,
        compare_run_ids=ns.compare_run_ids,
        compare_labels=ns.compare_labels,
        analysis_root=ns.analysis_root,
        analysis_run_id=ns.analysis_run_id,
        min_round=ns.min_round,
        max_round=ns.max_round,
        skip_no_actual=_parse_bool(ns.skip_no_actual, default=True),
        include_prev_round_baseline=_parse_bool(ns.include_prev_round_baseline, default=False),
        dpi=int(ns.dpi),
        debug_print=_parse_bool(ns.debug_print, default=False),
    )


def main(argv: Optional[list[str]] = None) -> None:
    ns = build_parser().parse_args(argv)
    result = run_analysis(_to_analysis_args(ns))
    generate_punishment_target_report(
        output_dir=Path(str(result["output_dir"])),
        run_id=ns.run_id,
        eval_root=ns.eval_root,
        analysis_csv=ns.analysis_csv,
        dpi=int(ns.dpi),
    )


if __name__ == "__main__":
    main()
