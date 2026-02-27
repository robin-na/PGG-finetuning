from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from io_utils import (  # noqa: E402
    ACTION_DICT_COLUMNS,
    REQUIRED_COLUMNS,
    apply_filters,
    coerce_base_columns,
    load_eval_csv,
    parse_action_columns,
    parse_bool,
    validate_required_columns,
)
from manifest import write_manifest  # noqa: E402
from metrics import AGG_METRIC_COLUMNS, aggregate_scores, score_rows  # noqa: E402


@dataclass
class AnalysisArgs:
    eval_csv: Optional[str]
    run_id: Optional[str]
    analysis_root: str
    analysis_run_id: Optional[str]
    min_round: Optional[int]
    max_round: Optional[int]
    skip_no_actual: bool
    dpi: int
    debug_print: bool


def _timestamp_id() -> str:
    return datetime.now().strftime("%y%m%d%H%M")


def _resolve_input_eval_csv(eval_csv: Optional[str], run_id: Optional[str]) -> Path:
    if eval_csv:
        return Path(eval_csv).resolve()
    if run_id:
        return (PROJECT_ROOT / "Micro_behavior_eval" / "output" / run_id / "micro_behavior_eval.csv").resolve()
    raise ValueError("Provide --eval_csv or --run_id.")


def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _ensure_output_safety(out_dir: Path) -> None:
    forbidden_root = (PROJECT_ROOT / "Micro_behavior_eval" / "output").resolve()
    if _is_subpath(out_dir.resolve(), forbidden_root):
        raise ValueError(
            f"Unsafe analysis output path: {out_dir}. "
            f"Analysis output must not be under {forbidden_root}."
        )


def _empty_scored_like(df: pd.DataFrame) -> pd.DataFrame:
    scored = score_rows(df.head(0).copy())
    return scored


def run_analysis(args: AnalysisArgs) -> Dict[str, Any]:
    input_csv = _resolve_input_eval_csv(args.eval_csv, args.run_id)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input evaluation CSV not found: {input_csv}")

    analysis_run_id = args.analysis_run_id or _timestamp_id()
    output_dir = Path(args.analysis_root).resolve() / analysis_run_id
    _ensure_output_safety(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_eval_csv(input_csv)
    validate_required_columns(raw_df, REQUIRED_COLUMNS)
    base_df = coerce_base_columns(raw_df)
    parsed_df, parse_summary = parse_action_columns(base_df, ACTION_DICT_COLUMNS)
    filtered_df, filter_summary = apply_filters(
        parsed_df,
        min_round=args.min_round,
        max_round=args.max_round,
        skip_no_actual=args.skip_no_actual,
    )

    if filtered_df.empty:
        scored_df = _empty_scored_like(filtered_df)
    else:
        scored_df = score_rows(filtered_df)

    metrics_overall = aggregate_scores(scored_df)
    metrics_by_round = aggregate_scores(scored_df, group_cols=["roundIndex"])
    metrics_by_game = aggregate_scores(scored_df, group_cols=["gameId"])

    output_files = []
    row_level_path = output_dir / "row_level_scored.csv"
    scored_df.to_csv(row_level_path, index=False)
    output_files.append(str(row_level_path))

    overall_path = output_dir / "metrics_overall.csv"
    metrics_overall.to_csv(overall_path, index=False)
    output_files.append(str(overall_path))

    round_path = output_dir / "metrics_by_round.csv"
    metrics_by_round.to_csv(round_path, index=False)
    output_files.append(str(round_path))

    game_path = output_dir / "metrics_by_game.csv"
    metrics_by_game.to_csv(game_path, index=False)
    output_files.append(str(game_path))

    plot_files = []
    if not scored_df.empty:
        from plots import generate_all_plots  # local import to avoid matplotlib init on CLI --help

        plot_files = generate_all_plots(
            scored_df=scored_df,
            metrics_by_round=metrics_by_round,
            metrics_by_game=metrics_by_game,
            output_dir=output_dir,
            dpi=int(args.dpi),
        )
        output_files.extend(plot_files)

    manifest_target = output_dir / "analysis_manifest.json"
    manifest_payload = {
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_csv),
        "output_path": str(output_dir),
        "filters": {
            "min_round": args.min_round,
            "max_round": args.max_round,
            "skip_no_actual": args.skip_no_actual,
        },
        "row_counts": {
            "pre_filter": filter_summary.pre_filter_rows,
            "post_filter": filter_summary.post_filter_rows,
            "dropped": filter_summary.dropped_rows,
        },
        "malformed_dict_parse_counts": parse_summary.malformed_counts,
        "malformed_dict_rows": parse_summary.malformed_rows,
        "generated_files": output_files + [str(manifest_target)],
        "args": {
            "eval_csv": args.eval_csv,
            "run_id": args.run_id,
            "analysis_root": args.analysis_root,
            "analysis_run_id": analysis_run_id,
            "min_round": args.min_round,
            "max_round": args.max_round,
            "skip_no_actual": args.skip_no_actual,
            "dpi": args.dpi,
            "debug_print": args.debug_print,
        },
        "metrics_columns": AGG_METRIC_COLUMNS,
    }
    manifest_path = write_manifest(output_dir, manifest_payload)
    output_files.append(str(manifest_path))

    if args.debug_print:
        print(f"[analysis] input: {input_csv}")
        print(f"[analysis] output: {output_dir}")
        print(f"[analysis] rows pre/post: {filter_summary.pre_filter_rows}/{filter_summary.post_filter_rows}")
        print(f"[analysis] malformed parse counts: {parse_summary.malformed_counts}")
        print(f"[analysis] generated files: {len(output_files)}")

    return {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "analysis_run_id": analysis_run_id,
        "generated_files": output_files,
        "plot_files": plot_files,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze micro behavior evaluation CSV output.")
    parser.add_argument("--eval_csv", type=str, default=None, help="Path to micro_behavior_eval.csv")
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run id under Micro_behavior_eval/output/<run_id>/micro_behavior_eval.csv",
    )
    parser.add_argument(
        "--analysis_root",
        type=str,
        default="Micro_behavior_eval/analysis_results",
        help="Root directory for analysis artifacts.",
    )
    parser.add_argument("--analysis_run_id", type=str, default=None, help="Output analysis run id (default: timestamp).")
    parser.add_argument("--min_round", type=int, default=None, help="Minimum round index to include.")
    parser.add_argument("--max_round", type=int, default=None, help="Maximum round index to include.")
    parser.add_argument(
        "--skip_no_actual",
        type=str,
        default="true",
        help="Whether to drop rows with no actual observation. true/false (default: true).",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Plot DPI.")
    parser.add_argument(
        "--debug_print",
        type=str,
        default="false",
        help="Print debug summary. true/false (default: false).",
    )
    return parser


def parse_cli_args(argv: Optional[list[str]] = None) -> AnalysisArgs:
    parser = build_parser()
    ns = parser.parse_args(argv)
    return AnalysisArgs(
        eval_csv=ns.eval_csv,
        run_id=ns.run_id,
        analysis_root=ns.analysis_root,
        analysis_run_id=ns.analysis_run_id,
        min_round=ns.min_round,
        max_round=ns.max_round,
        skip_no_actual=bool(parse_bool(ns.skip_no_actual, default=True)),
        dpi=int(ns.dpi),
        debug_print=bool(parse_bool(ns.debug_print, default=False)),
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_cli_args(argv)
    run_analysis(args)


if __name__ == "__main__":
    main()
