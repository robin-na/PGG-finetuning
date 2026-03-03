#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


def normalize_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "1", "yes"}:
            return True
        if low in {"false", "0", "no"}:
            return False
    return None


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def record_pair(rec: Dict[str, Any]) -> Tuple[str, str]:
    game_id = normalize_id(rec.get("experiment") or rec.get("gameId"))
    player_id = normalize_id(rec.get("participant") or rec.get("playerId"))
    return game_id, player_id


def to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deduplicated union archetype summary pool from learn+val JSONL."
    )
    parser.add_argument(
        "--learn-jsonl",
        type=Path,
        default=Path("Persona/archetype_oracle_gpt51_learn.jsonl"),
    )
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        default=Path("Persona/archetype_oracle_gpt51_val.jsonl"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(
            "outputs/benchmark/cache/archetype/archetype_oracle_gpt51_learn_val_union_finished.jsonl"
        ),
    )
    parser.add_argument(
        "--include-unfinished",
        action="store_true",
        help="Keep records regardless of game_finished flag (default keeps finished only).",
    )
    parser.add_argument(
        "--prefer-val",
        action="store_true",
        help="If duplicate (game,player) exists in both sources, keep val record instead of learn.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources: List[Tuple[str, Path]] = [("learn", args.learn_jsonl), ("val", args.val_jsonl)]
    if args.prefer_val:
        sources = [("val", args.val_jsonl), ("learn", args.learn_jsonl)]

    seen: set[Tuple[str, str]] = set()
    out_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Dict[str, int]] = {}
    dropped_duplicates = 0
    dropped_unfinished = 0
    dropped_missing_ids = 0

    for source_name, path in sources:
        src_total = 0
        src_kept = 0
        for rec in iter_jsonl(path):
            src_total += 1
            if not args.include_unfinished and parse_bool(rec.get("game_finished")) is not True:
                dropped_unfinished += 1
                continue

            game_id, player_id = record_pair(rec)
            if not game_id or not player_id:
                dropped_missing_ids += 1
                continue
            key = (game_id, player_id)
            if key in seen:
                dropped_duplicates += 1
                continue
            seen.add(key)
            out_rows.append(rec)
            src_kept += 1

        stats[source_name] = {
            "rows_total": int(src_total),
            "rows_kept": int(src_kept),
        }

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for rec in out_rows:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "learn_jsonl": to_repo_rel(args.learn_jsonl),
        "val_jsonl": to_repo_rel(args.val_jsonl),
        "output_jsonl": to_repo_rel(args.output_jsonl),
        "include_unfinished": bool(args.include_unfinished),
        "prefer_val": bool(args.prefer_val),
        "rows_written": int(len(out_rows)),
        "unique_pairs_written": int(len(seen)),
        "dropped_duplicates": int(dropped_duplicates),
        "dropped_unfinished": int(dropped_unfinished),
        "dropped_missing_ids": int(dropped_missing_ids),
        "source_stats": stats,
    }

    summary_path = args.output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
