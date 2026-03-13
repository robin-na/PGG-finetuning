#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple macro shard run directories into one combined run directory."
    )
    parser.add_argument(
        "--output-run-dir",
        type=Path,
        required=True,
        help="Destination combined run directory.",
    )
    parser.add_argument(
        "--shard-run-dir",
        action="append",
        type=Path,
        required=True,
        help="Shard run directory produced by Macro_simulation_eval. Can repeat.",
    )
    parser.add_argument(
        "--source-manifest",
        type=str,
        default="",
        help="Optional full manifest path recorded in the merged config.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _concat_jsonl(sources: List[Path], destination: Path) -> bool:
    wrote = False
    with destination.open("w", encoding="utf-8") as out_handle:
        for source in sources:
            if not source.exists() or source.stat().st_size == 0:
                continue
            with source.open("r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    out_handle.write(line)
                    wrote = True
    if not wrote:
        destination.unlink(missing_ok=True)
    return wrote


def main() -> None:
    args = parse_args()
    shard_dirs = [path.resolve() for path in args.shard_run_dir]
    for shard_dir in shard_dirs:
        if not shard_dir.is_dir():
            raise FileNotFoundError(f"Shard run directory not found: {shard_dir}")

    output_run_dir = args.output_run_dir.resolve()
    output_run_dir.mkdir(parents=True, exist_ok=True)

    shard_configs = [_load_json(shard_dir / "config.json") for shard_dir in shard_dirs]
    first_config = next((cfg for cfg in shard_configs if cfg), {})
    now_iso = datetime.now(timezone.utc).isoformat()

    csv_frames: List[pd.DataFrame] = []
    total_rows = 0
    game_ids: List[str] = []
    requested_game_ids: List[Any] = []

    for shard_dir, cfg in zip(shard_dirs, shard_configs):
        csv_path = shard_dir / "macro_simulation_eval.csv"
        if csv_path.exists() and csv_path.stat().st_size > 0:
            frame = pd.read_csv(csv_path)
            csv_frames.append(frame)
            total_rows += int(len(frame))
        selection = cfg.get("selection") if isinstance(cfg.get("selection"), dict) else {}
        game_ids.extend([str(x) for x in selection.get("game_ids", []) if str(x).strip()])
        requested = selection.get("requested_game_ids")
        if requested not in (None, "", []):
            requested_game_ids.append(requested)

    merged_csv_path = output_run_dir / "macro_simulation_eval.csv"
    if csv_frames:
        merged_frame = pd.concat(csv_frames, ignore_index=True)
        sort_keys = [col for col in ["gameId", "roundIndex", "playerId"] if col in merged_frame.columns]
        if sort_keys:
            merged_frame = merged_frame.sort_values(sort_keys, kind="stable").reset_index(drop=True)
        merged_frame.to_csv(merged_csv_path, index=False)
    else:
        pd.DataFrame().to_csv(merged_csv_path, index=False)

    _concat_jsonl([shard / "history_transcripts.jsonl" for shard in shard_dirs], output_run_dir / "history_transcripts.jsonl")
    _concat_jsonl([shard / "macro_simulation_debug.jsonl" for shard in shard_dirs], output_run_dir / "macro_simulation_debug.jsonl")
    _concat_jsonl([shard / "macro_simulation_debug_full.jsonl" for shard in shard_dirs], output_run_dir / "macro_simulation_debug_full.jsonl")

    merged_config = dict(first_config)
    merged_config["status"] = "completed"
    merged_config["completed_at_utc"] = now_iso
    merged_config["created_at_utc"] = merged_config.get("created_at_utc") or now_iso
    merged_config["inputs"] = dict(first_config.get("inputs") or {})
    if args.source_manifest:
        merged_config["inputs"]["source_manifest"] = args.source_manifest
    merged_config["selection"] = {
        "num_games": len(game_ids),
        "game_ids": game_ids,
        "requested_game_ids": requested_game_ids,
        "source_manifest": args.source_manifest or None,
        "shard_run_dirs": [str(path) for path in shard_dirs],
    }
    merged_config["outputs"] = {
        "directory": str(output_run_dir),
        "rows": str(merged_csv_path),
        "transcripts": str(output_run_dir / "history_transcripts.jsonl") if (output_run_dir / "history_transcripts.jsonl").exists() else None,
        "debug": str(output_run_dir / "macro_simulation_debug.jsonl") if (output_run_dir / "macro_simulation_debug.jsonl").exists() else None,
        "debug_full": str(output_run_dir / "macro_simulation_debug_full.jsonl") if (output_run_dir / "macro_simulation_debug_full.jsonl").exists() else None,
    }
    merged_config["summary"] = {
        "num_rows": total_rows,
        "num_games": len(game_ids),
        "merged_from_shards": len(shard_dirs),
    }

    with (output_run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(merged_config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Merged run written to {output_run_dir}")


if __name__ == "__main__":
    main()
