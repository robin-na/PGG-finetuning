#!/usr/bin/env python3
"""
Wait for in-progress LLM predictions to finish, then run:
  1. random_archetype predictions
  2. evaluate.py  (all three modes)
  3. generate_report.py  (charts + description)

Run this in the background; it polls every 60s.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
PRED_DIR = OUTPUT_ROOT / "predictions"
SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
TARGET = 6200  # 200 participants × 31 questions
POLL_INTERVAL = 60  # seconds


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def run(cmd: list, label: str) -> bool:
    print(f"\n[finalize] Starting: {label}")
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        print(f"[finalize] ERROR in {label} (exit {result.returncode})")
        return False
    print(f"[finalize] Done: {label}")
    return True


def wait_for(paths_and_targets: list, label: str) -> None:
    while True:
        done = all(count_lines(p) >= t for p, t in paths_and_targets)
        counts = ", ".join(f"{p.name}={count_lines(p)}/{t}" for p, t in paths_and_targets)
        print(f"[finalize] Waiting for {label}: {counts}")
        if done:
            print(f"[finalize] {label} complete.")
            return
        time.sleep(POLL_INTERVAL)


def main() -> None:
    print("[finalize] Starting finalize.py")
    print(f"[finalize] Polling every {POLL_INTERVAL}s, target={TARGET} lines each")

    # ── Step 1: Wait for demographics_only ────────────────────────────────────
    wait_for(
        [(PRED_DIR / "predictions_demographics_only.jsonl", TARGET)],
        "demographics_only",
    )

    # ── Step 2: Wait for archetype ────────────────────────────────────────────
    wait_for(
        [(PRED_DIR / "predictions_archetype.jsonl", TARGET)],
        "archetype",
    )

    # ── Step 3: Run random_archetype ──────────────────────────────────────────
    ok = run(
        [PYTHON, str(SCRIPT_DIR / "llm_predict.py"),
         "--mode", "random_archetype",
         "--model", "gpt-4o",
         "--pilot-n", "200",
         "--top-k", "10"],
        "random_archetype predictions",
    )
    if not ok:
        print("[finalize] Skipping random_archetype; continuing to evaluate.")

    # ── Step 4: Evaluate all modes ────────────────────────────────────────────
    pred_files = sorted(PRED_DIR.glob("predictions_*.jsonl"))
    if not pred_files:
        print("[finalize] No prediction files found. Exiting.")
        return

    eval_dir = OUTPUT_ROOT / "evaluation"
    ok = run(
        [PYTHON, str(SCRIPT_DIR / "evaluate.py"),
         "--predictions", *[str(p) for p in pred_files],
         "--output-dir", str(eval_dir)],
        "evaluate.py",
    )
    if not ok:
        print("[finalize] evaluate.py failed; will still try report generation.")

    # ── Step 5: Generate report ───────────────────────────────────────────────
    run(
        [PYTHON, str(SCRIPT_DIR / "generate_report.py")],
        "generate_report.py",
    )

    print("\n[finalize] All done.")
    report_dir = OUTPUT_ROOT / "report"
    if report_dir.exists():
        print("Report files:")
        for f in sorted(report_dir.iterdir()):
            print(f"  {f}")


if __name__ == "__main__":
    main()
