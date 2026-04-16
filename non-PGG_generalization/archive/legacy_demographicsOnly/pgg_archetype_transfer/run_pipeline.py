#!/usr/bin/env python3
"""End-to-end orchestration for PGG archetype transfer pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config import OUTPUT_ROOT

SCRIPT_DIR = Path(__file__).resolve().parent

STAGES = [
    "prepare_demographics",
    "embed_archetypes",
    "train_retrieval",
    "retrieve_candidates",
    "summarize_archetypes",
    "predict_archetype",
    "predict_demographics_only",
    "predict_random_archetype",
    "evaluate",
]


def run_stage(stage: str, args: argparse.Namespace) -> None:
    """Run a single pipeline stage."""
    print(f"\n{'='*60}")
    print(f"STAGE: {stage}")
    print(f"{'='*60}\n")

    cmd_base = [sys.executable]

    if stage == "prepare_demographics":
        cmd = cmd_base + [
            str(SCRIPT_DIR / "prepare_twin2k_demographics.py"),
            "--output", str(OUTPUT_ROOT / "twin2k_demographics_pgg_compatible.csv"),
        ]

    elif stage == "embed_archetypes":
        cmd = cmd_base + [
            str(SCRIPT_DIR / "embed_archetypes.py"),
            "--output-dir", str(OUTPUT_ROOT / "archetype_bank"),
        ]

    elif stage == "train_retrieval":
        cmd = cmd_base + [
            str(SCRIPT_DIR / "train_d_only_retrieval.py"),
            "--output-dir", str(OUTPUT_ROOT / "d_only_model"),
        ]

    elif stage == "retrieve_candidates":
        cmd = cmd_base + [
            str(SCRIPT_DIR / "retrieve_candidates.py"),
            "--top-k", str(args.top_k),
            "--output", str(OUTPUT_ROOT / "candidate_archetypes.jsonl"),
        ]

    elif stage == "summarize_archetypes":
        cmd = cmd_base + [
            str(SCRIPT_DIR / "summarize_archetypes.py"),
        ]
        if args.skip_summarize:
            print("Skipping summarization (--skip-summarize)")
            return

    elif stage == "predict_archetype":
        cmd = cmd_base + [
            str(SCRIPT_DIR / "llm_predict.py"),
            "--mode", "archetype",
            "--model", args.model,
            "--top-k", str(args.top_k),
            "--pilot-n", str(args.pilot_n),
        ]

    elif stage == "predict_demographics_only":
        cmd = cmd_base + [
            str(SCRIPT_DIR / "llm_predict.py"),
            "--mode", "demographics_only",
            "--model", args.model,
            "--pilot-n", str(args.pilot_n),
        ]

    elif stage == "predict_random_archetype":
        cmd = cmd_base + [
            str(SCRIPT_DIR / "llm_predict.py"),
            "--mode", "random_archetype",
            "--model", args.model,
            "--top-k", str(args.top_k),
            "--pilot-n", str(args.pilot_n),
        ]

    elif stage == "evaluate":
        pred_dir = OUTPUT_ROOT / "predictions"
        pred_files = sorted(pred_dir.glob("predictions_*.jsonl")) if pred_dir.exists() else []
        if not pred_files:
            print("No prediction files found. Skipping evaluation.")
            return
        cmd = cmd_base + [
            str(SCRIPT_DIR / "evaluate.py"),
            "--predictions", *[str(p) for p in pred_files],
            "--output-dir", str(OUTPUT_ROOT / "evaluation"),
        ]

    else:
        raise ValueError(f"Unknown stage: {stage}")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        print(f"\nERROR: Stage '{stage}' failed with exit code {result.returncode}")
        if not args.continue_on_error:
            sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PGG archetype transfer pipeline.")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        help=f"Stages to run. 'all' runs everything. Available: {STAGES}",
    )
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pilot-n", type=int, default=200)
    parser.add_argument("--skip-summarize", action="store_true",
                        help="Skip archetype summarization (use full text)")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    stages = STAGES if "all" in args.stages else args.stages

    # Validate
    for s in stages:
        if s not in STAGES:
            print(f"Unknown stage: {s}. Available: {STAGES}")
            sys.exit(1)

    print(f"Pipeline stages: {stages}")
    print(f"Model: {args.model}")
    print(f"Top-K: {args.top_k}")
    print(f"Pilot N: {args.pilot_n}")
    print(f"Output: {OUTPUT_ROOT}")

    for stage in stages:
        run_stage(stage, args)

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"Results in: {OUTPUT_ROOT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
