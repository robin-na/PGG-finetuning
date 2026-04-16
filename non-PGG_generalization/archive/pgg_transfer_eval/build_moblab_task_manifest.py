#!/usr/bin/env python3
"""Build sample task manifests for MobLab LLM inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moblab_llm_utils import (
    DEFAULT_TASK_OUTPUT_DIR,
    build_task1_instances,
    build_task2_instances,
    sample_stratified,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("task1", "task2"), required=True)
    parser.add_argument("--task2-mode", choices=("future_mean", "trajectory"), default="future_mean")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TASK_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.task == "task1":
        df = build_task1_instances()
        sampled = sample_stratified(df, "target_measure", args.sample_size, args.seed)
    else:
        df = build_task2_instances(prediction_mode=args.task2_mode)
        sampled = sample_stratified(df, "target_measure", args.sample_size, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_stem = (
        f"{args.task}_{args.task2_mode}_sample_{len(sampled)}"
        if args.task == "task2"
        else f"{args.task}_sample_{len(sampled)}"
    )
    tasks_path = args.output_dir / f"{task_stem}.jsonl"
    preview_path = args.output_dir / f"{task_stem}_preview.json"
    summary_path = args.output_dir / f"{task_stem}_summary.json"

    rows = sampled.to_dict(orient="records")
    write_jsonl(rows, tasks_path)
    if rows:
        preview_path.write_text(json.dumps(rows[0], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "task": args.task,
        "task2_mode": args.task2_mode if args.task == "task2" else None,
        "sample_size": len(rows),
        "seed": args.seed,
        "counts_by_target_measure": sampled["target_measure"].value_counts().sort_index().to_dict(),
        "path": str(tasks_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
