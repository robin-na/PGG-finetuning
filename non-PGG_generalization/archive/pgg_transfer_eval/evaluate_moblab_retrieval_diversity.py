#!/usr/bin/env python3
"""Evaluate retrieval diversity and mode-collapse risk for MobLab -> PGG retrieval."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from moblab_llm_utils import DEFAULT_OUTPUT_ROOT, load_jsonl


DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "retrieval_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def entropy_from_counter(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return float("nan")
    probs = [value / total for value in counter.values()]
    return float(-sum(p * math.log(p, 2) for p in probs if p > 0))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.candidates_jsonl)
    expanded: List[Dict[str, Any]] = []
    top1_counter: Counter = Counter()
    topk_counter: Counter = Counter()
    for row in rows:
        instance_id = str(row["instance_id"])
        task_type = str(row["task_type"])
        target_measure = str(row["target_measure"])
        candidates = list(row.get("candidates") or [])
        if candidates:
            top1 = str(candidates[0].get("custom_id", candidates[0].get("filename", "")))
            top1_counter[top1] += 1
        for candidate in candidates:
            cid = str(candidate.get("custom_id", candidate.get("filename", "")))
            topk_counter[cid] += 1
            expanded.append(
                {
                    "instance_id": instance_id,
                    "task_type": task_type,
                    "target_measure": target_measure,
                    "candidate_id": cid,
                    "rank": int(candidate.get("rank", 0)),
                    "score": float(candidate.get("score", 0.0)),
                }
            )

    expanded_df = pd.DataFrame(expanded)
    expanded_df.to_csv(args.output_dir / "retrieval_candidates_long.csv", index=False)

    top1_share = top1_counter.most_common(1)[0][1] / len(rows) if rows and top1_counter else float("nan")
    summary = {
        "num_queries": len(rows),
        "num_queries_with_hits": int(sum(1 for row in rows if row.get("candidates"))),
        "unique_top1_candidates": len(top1_counter),
        "unique_topk_candidates": len(topk_counter),
        "top1_entropy_bits": entropy_from_counter(top1_counter),
        "topk_entropy_bits": entropy_from_counter(topk_counter),
        "top1_max_share": top1_share,
        "top1_mode_collapse_warning": bool(top1_share == top1_share and top1_share >= 0.2),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    top1_table = pd.DataFrame(
        [{"candidate_id": cid, "count": count, "share": count / len(rows)} for cid, count in top1_counter.most_common()]
    )
    top1_table.to_csv(args.output_dir / "top1_frequency.csv", index=False)

    if not expanded_df.empty:
        by_target = (
            expanded_df.groupby(["task_type", "target_measure", "candidate_id"])
            .size()
            .reset_index(name="count")
            .sort_values(["task_type", "target_measure", "count"], ascending=[True, True, False])
        )
        by_target.to_csv(args.output_dir / "candidate_frequency_by_target.csv", index=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
