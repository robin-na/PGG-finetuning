#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def classify_variant(config: Dict[str, Any]) -> str:
    model = config.get("model", {}) if isinstance(config, dict) else {}
    mode = str(model.get("archetype") or model.get("persona") or "").strip()
    pool = str(
        model.get("archetype_summary_pool") or model.get("persona_summary_pool") or ""
    ).strip()
    pool_lower = pool.lower()

    if mode in {"", "none"}:
        return "no_archetype"
    if mode == "random_summary":
        return "random_archetype"
    if mode == "matched_summary":
        oracle_candidates = {
            str((REPO_ROOT / "Persona" / "summary_gpt51_val.jsonl").resolve()).lower(),
            str(
                (
                    REPO_ROOT
                    / "outputs"
                    / "benchmark"
                    / "cache"
                    / "archetype"
                    / "summary_gpt51_learn_val_union_finished.jsonl"
                ).resolve()
            ).lower(),
            "persona/summary_gpt51_val.jsonl",
        }
        if pool_lower in oracle_candidates:
            return "oracle_archetype"
        if any(
            marker in pool_lower
            for marker in [
                "synthetic_persona",
                "synthetic_archetype",
                "retrieved",
                "archetype_retrieval/validation_wave",
            ]
        ):
            return "retrieved_archetype"
        return "oracle_archetype"
    return "no_archetype"


def update_config_paths(config_path: Path, run_dir: Path, variant_root: Path) -> None:
    if not config_path.exists():
        return
    payload = load_json(config_path)
    if not payload:
        return

    outputs = payload.get("outputs", {})
    if isinstance(outputs, dict):
        outputs["directory"] = str(run_dir)
        for key in ["rows", "transcripts", "debug", "debug_full", "config"]:
            value = outputs.get(key)
            if isinstance(value, str) and value:
                outputs[key] = str(run_dir / Path(value).name)
        payload["outputs"] = outputs

    for section in ["args", "model"]:
        section_obj = payload.get(section)
        if isinstance(section_obj, dict):
            if "output_root" in section_obj:
                section_obj["output_root"] = str(variant_root)
            payload[section] = section_obj

    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def iter_run_dirs(source_root: Path) -> Iterable[Path]:
    for child in sorted(source_root.iterdir()):
        if child.is_dir() and (child / "micro_behavior_eval.csv").exists():
            yield child


def choose_destination(base_dir: Path, run_id: str) -> Path:
    dst = base_dir / run_id
    if not dst.exists():
        return dst
    suffix = 1
    while True:
        cand = base_dir / f"{run_id}_migrated_{suffix}"
        if not cand.exists():
            return cand
        suffix += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate micro eval run folders into variant-based layout: "
            "<destination-root>/<variant>/<run_id>/"
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("Micro_behavior_eval/output"),
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=Path("outputs/default/runs/source_default/micro_behavior_eval"),
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy runs instead of moving them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without changing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = resolve_repo_path(args.source_root)
    destination_root = resolve_repo_path(args.destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    moves: List[Tuple[Path, Path, str]] = []
    for run_dir in iter_run_dirs(source_root):
        config_path = run_dir / "config.json"
        payload = load_json(config_path) if config_path.exists() else {}
        variant = classify_variant(payload)
        variant_root = destination_root / variant
        dst_dir = choose_destination(variant_root, run_dir.name)
        moves.append((run_dir, dst_dir, variant))

    if not moves:
        print("No run folders found to migrate.")
        return 0

    print("Planned migrations:")
    for src, dst, variant in moves:
        print(f"- {src} -> {dst} ({variant})")

    if args.dry_run:
        return 0

    for src, dst, _ in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if args.copy:
            shutil.copytree(src, dst)
        else:
            shutil.move(str(src), str(dst))
        update_config_paths(dst / "config.json", run_dir=dst, variant_root=dst.parent)

    print(f"Migrated {len(moves)} runs into {destination_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
