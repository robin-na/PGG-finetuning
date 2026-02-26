#!/usr/bin/env python3
"""
Verify and move per-tag embedding/jsonl files into learning_wave.

For each tag directory under --source-root:
1) Find one embedding .npy file.
2) Find .jsonl files whose records contain either
   ("gameId", "playerId") or ("experiment", "participant").
3) Verify embedding vector count against rows where game_finished is True/False.
4) Move validated .npy and selected .jsonl files into --dest-root/<TAG>/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


DEFAULT_TAGS = [
    "COMMUNICATION",
    "CONTRIBUTION",
    "PUNISHMENT",
    "RESPONSE_TO_END_GAME",
    "RESPONSE_TO_OTHERS_OUTCOME",
    "RESPONSE_TO_PUNISHER",
    "RESPONSE_TO_REWARDER",
    "REWARD",
]


@dataclass
class JsonlCounts:
    rows_total: int = 0
    game_finished_true: int = 0
    game_finished_false: int = 0
    game_finished_other: int = 0

    @property
    def rows_with_bool_game_finished(self) -> int:
        return self.game_finished_true + self.game_finished_false


def parse_bool(value: Any) -> Optional[bool]:
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


def first_json_obj(path: Path) -> Optional[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                return obj
            return None
    return None


def count_jsonl(path: Path) -> JsonlCounts:
    counts = JsonlCounts()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            counts.rows_total += 1
            finished = parse_bool(row.get("game_finished"))
            if finished is True:
                counts.game_finished_true += 1
            elif finished is False:
                counts.game_finished_false += 1
            else:
                counts.game_finished_other += 1
    return counts


def embedding_vector_count(path: Path) -> int:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim <= 1:
        # A single vector may be stored as shape (D,) instead of (1, D).
        count = 1
    else:
        count = int(arr.shape[0])
    del arr
    return count


def pick_embedding_file(tag_dir: Path) -> Optional[Path]:
    npy_files = sorted(tag_dir.glob("*.npy"))
    if not npy_files:
        return None
    preferred = [p for p in npy_files if "embedding" in p.name.lower()]
    if len(preferred) == 1:
        return preferred[0]
    if len(npy_files) == 1:
        return npy_files[0]
    return None


def find_info_jsonl_files(tag_dir: Path) -> List[Path]:
    files: List[Path] = []
    for path in sorted(tag_dir.glob("*.jsonl")):
        try:
            obj = first_json_obj(path)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        has_old_keys = "experiment" in obj and "participant" in obj
        has_new_keys = "gameId" in obj and "playerId" in obj
        if has_old_keys or has_new_keys:
            files.append(path)
    return files


def ensure_unique_destinations(
    move_items: Iterable[Dict[str, Path]],
    overwrite: bool,
) -> List[str]:
    errors: List[str] = []
    for item in move_items:
        src = item["src"]
        dst = item["dst"]
        if not src.exists():
            errors.append(f"Missing source file: {src}")
            continue
        if dst.exists() and not overwrite:
            errors.append(f"Destination exists (use --overwrite): {dst}")
    return errors


def move_files(move_items: Iterable[Dict[str, Path]], overwrite: bool, dry_run: bool) -> List[Dict[str, str]]:
    moved: List[Dict[str, str]] = []
    for item in move_items:
        src = item["src"]
        dst = item["dst"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and overwrite:
            if dst.is_file():
                dst.unlink()
            else:
                raise RuntimeError(f"Refusing to overwrite non-file destination: {dst}")
        if dry_run:
            moved.append({"src": str(src), "dst": str(dst), "status": "dry_run"})
            continue
        shutil.move(str(src), str(dst))
        moved.append({"src": str(src), "dst": str(dst), "status": "moved"})
    return moved


def relative_to_cwd(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def run_pipeline(
    source_root: Path,
    dest_root: Path,
    tags: List[str],
    verify_only: bool,
    allow_mismatch: bool,
    overwrite: bool,
    dry_run: bool,
) -> int:
    report: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": relative_to_cwd(source_root),
        "dest_root": relative_to_cwd(dest_root),
        "verify_only": verify_only,
        "allow_mismatch": allow_mismatch,
        "overwrite": overwrite,
        "dry_run": dry_run,
        "tags": tags,
        "results": {},
        "errors": [],
    }

    all_move_items: List[Dict[str, Path]] = []
    mismatch_found = False

    for tag in tags:
        tag_dir = source_root / tag
        tag_report: Dict[str, Any] = {
            "tag_dir": relative_to_cwd(tag_dir),
            "status": "ok",
            "errors": [],
            "embedding_file": None,
            "embedding_vectors": None,
            "jsonl_checks": [],
            "move_candidates": [],
        }

        if not tag_dir.is_dir():
            tag_report["status"] = "error"
            tag_report["errors"].append(f"Missing tag directory: {relative_to_cwd(tag_dir)}")
            report["results"][tag] = tag_report
            report["errors"].append(f"{tag}: missing tag directory")
            continue

        emb_file = pick_embedding_file(tag_dir)
        if emb_file is None:
            tag_report["status"] = "error"
            tag_report["errors"].append("Could not uniquely identify embedding .npy file.")
            report["results"][tag] = tag_report
            report["errors"].append(f"{tag}: missing or ambiguous embedding file")
            continue

        jsonl_files = find_info_jsonl_files(tag_dir)
        if not jsonl_files:
            tag_report["status"] = "error"
            tag_report["errors"].append(
                "No JSONL files with gameId+playerId or experiment+participant keys found."
            )
            report["results"][tag] = tag_report
            report["errors"].append(f"{tag}: missing jsonl files")
            continue

        vectors = embedding_vector_count(emb_file)
        tag_report["embedding_file"] = relative_to_cwd(emb_file)
        tag_report["embedding_vectors"] = vectors

        for jsonl_path in jsonl_files:
            counts = count_jsonl(jsonl_path)
            check = {
                "file": relative_to_cwd(jsonl_path),
                **asdict(counts),
                "rows_with_bool_game_finished": counts.rows_with_bool_game_finished,
                "match_vectors_vs_bool_rows": vectors == counts.rows_with_bool_game_finished,
                "match_vectors_vs_total_rows": vectors == counts.rows_total,
            }
            tag_report["jsonl_checks"].append(check)
            if not check["match_vectors_vs_bool_rows"]:
                mismatch_found = True
                tag_report["status"] = "mismatch"

        for src in [emb_file, *jsonl_files]:
            dst = dest_root / tag / src.name
            move_item = {"src": src, "dst": dst, "tag": tag}
            all_move_items.append(move_item)
            tag_report["move_candidates"].append(
                {"src": relative_to_cwd(src), "dst": relative_to_cwd(dst)}
            )

        report["results"][tag] = tag_report

    move_errors = ensure_unique_destinations(all_move_items, overwrite=overwrite)
    if move_errors:
        report["errors"].extend(move_errors)

    can_move = not verify_only and not move_errors
    if mismatch_found and not allow_mismatch:
        can_move = False
        report["errors"].append(
            "One or more vector-count mismatches found against rows where game_finished is True/False."
        )

    if can_move:
        moved = move_files(all_move_items, overwrite=overwrite, dry_run=dry_run)
        report["moved_files"] = moved
    else:
        report["moved_files"] = []

    dest_root.mkdir(parents=True, exist_ok=True)
    report_path = dest_root / "learning_wave_verification_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Report written to: {relative_to_cwd(report_path)}")
    print()
    for tag in tags:
        tag_result = report["results"].get(tag)
        if not tag_result:
            continue
        print(f"[{tag}] status={tag_result['status']}")
        if tag_result["errors"]:
            for err in tag_result["errors"]:
                print(f"  - ERROR: {err}")
            continue
        print(f"  embedding_vectors={tag_result['embedding_vectors']}")
        for check in tag_result["jsonl_checks"]:
            print(
                "  - {name}: total={total}, true={true}, false={false}, other={other}, "
                "match_bool_rows={mb}, match_total_rows={mt}".format(
                    name=Path(check["file"]).name,
                    total=check["rows_total"],
                    true=check["game_finished_true"],
                    false=check["game_finished_false"],
                    other=check["game_finished_other"],
                    mb=check["match_vectors_vs_bool_rows"],
                    mt=check["match_vectors_vs_total_rows"],
                )
            )
        print()

    if report["errors"]:
        print("Pipeline completed with issues:")
        for err in report["errors"]:
            print(f"- {err}")
        return 1

    if verify_only:
        print("Verification passed. No files moved (--verify-only).")
    elif dry_run:
        print("Verification passed. Move step was a dry run.")
    else:
        print("Verification passed. Files moved successfully.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify vector/jsonl consistency and move files into learning_wave."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("Persona/cluster/tag_section_clusters_openai_learn"),
        help="Source root containing per-tag folders.",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path("Persona/archetype_retrieval/learning_wave"),
        help="Destination root for moved files and report.",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=DEFAULT_TAGS,
        help="Tag directories to process.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify counts; do not move files.",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Allow move even when vector count mismatch is found.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not move files; only print/report planned moves.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tags = [str(tag).strip().upper() for tag in args.tags if str(tag).strip()]
    if not tags:
        print("No tags provided.", file=sys.stderr)
        return 2
    return run_pipeline(
        source_root=args.source_root,
        dest_root=args.dest_root,
        tags=tags,
        verify_only=args.verify_only,
        allow_mismatch=args.allow_mismatch,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
