#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def split_rel_path(split_root: Path, split_base_root: Path) -> Path:
    split_root_resolved = split_root.resolve()
    split_base_resolved = split_base_root.resolve()
    benchmark_filtered_root = (split_base_resolved / "data").resolve()
    benchmark_ood_root = (split_base_resolved / "data_ood_splits").resolve()
    benchmark_ood_wave_root = (split_base_resolved / "data_ood_splits_wave_anchored").resolve()

    if split_root_resolved == benchmark_filtered_root:
        return Path("benchmark_filtered")
    try:
        rel_ood = split_root_resolved.relative_to(benchmark_ood_root)
        return Path("benchmark_ood") / rel_ood
    except Exception:
        pass
    try:
        rel_ood_wave = split_root_resolved.relative_to(benchmark_ood_wave_root)
        return Path("benchmark_ood_wave_anchored") / rel_ood_wave
    except Exception:
        pass
    try:
        return split_root_resolved.relative_to(split_base_resolved)
    except Exception:
        return Path(split_root.name)


def discover_split_roots(
    split_roots: Sequence[Path],
    include_default: bool,
    split_base_root: Path,
    ood_roots: Sequence[Path],
) -> List[Path]:
    roots: List[Path] = []
    if split_roots:
        roots.extend(split_roots)
        return roots
    if include_default:
        default_root = split_base_root / "data"
        if default_root.is_dir():
            roots.append(default_root)
    for ood_root in ood_roots:
        if not ood_root.is_dir():
            continue
        for config_dir in sorted([p for p in ood_root.iterdir() if p.is_dir()]):
            for direction_dir in sorted([p for p in config_dir.iterdir() if p.is_dir()]):
                roots.append(direction_dir)
    return roots


def run_cmd(cmd: List[str], dry_run: bool) -> None:
    print("+", " ".join(shlex.quote(x) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute split-specific synthetic archetype pools so SLURM simulation jobs "
            "only need to run micro/macro simulation."
        )
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        action="append",
        default=[],
        help="Split root to process. Can be repeated.",
    )
    parser.add_argument(
        "--include-default",
        action="store_true",
        help="Include benchmark/data when discovering splits automatically.",
    )
    parser.add_argument(
        "--ood-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "OOD root for auto-discovery. Can be repeated. "
            "Defaults to benchmark/data_ood_splits and benchmark/data_ood_splits_wave_anchored."
        ),
    )
    parser.add_argument("--split-base-root", type=Path, default=Path("benchmark"))
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/benchmark/runs"))
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--synthetic-model", type=str, default="ridge")
    parser.add_argument(
        "--oracle-archetype-summary",
        type=Path,
        default=Path("Persona/archetype_oracle_gpt51_val.jsonl"),
    )
    parser.add_argument(
        "--auto-train-if-missing",
        action="store_true",
        help="If a split has no latest retrieval run, run archetype-train+validate before synthetic generation.",
    )
    parser.add_argument(
        "--train-arg",
        action="append",
        default=[],
        help="Extra arg string forwarded to run_split_pipeline.py --train-arg. Can repeat.",
    )
    parser.add_argument(
        "--validate-arg",
        action="append",
        default=[],
        help="Extra arg string forwarded to run_split_pipeline.py --validate-arg. Can repeat.",
    )
    parser.add_argument(
        "--fast-default-train",
        action="store_true",
        help=(
            "When auto-training missing splits and no --train-arg is provided, "
            "use a faster training profile: --models mean linear ridge --allow-tag-errors."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild synthetic JSONL even if it already exists.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_base_root = resolve_repo_path(args.split_base_root)
    runs_root = resolve_repo_path(args.runs_root)
    split_roots = [resolve_repo_path(p) for p in args.split_root]
    ood_roots = [
        resolve_repo_path(p)
        for p in (
            args.ood_root
            or [Path("benchmark/data_ood_splits"), Path("benchmark/data_ood_splits_wave_anchored")]
        )
    ]

    roots = discover_split_roots(
        split_roots=split_roots,
        include_default=args.include_default,
        split_base_root=split_base_root,
        ood_roots=ood_roots,
    )
    if not roots:
        raise FileNotFoundError("No split roots selected/discovered.")

    skipped = 0
    processed = 0
    for split_root in roots:
        rel = split_rel_path(split_root, split_base_root)
        split_run_root = runs_root / rel
        latest_run = split_run_root / "archetype_retrieval" / "model_runs" / "latest_run.txt"
        synthetic_jsonl = (
            split_run_root
            / "archetype_retrieval"
            / "validation_wave"
            / f"synthetic_archetype_{args.synthetic_model}_val.jsonl"
        )
        if synthetic_jsonl.exists() and synthetic_jsonl.stat().st_size > 0 and not args.overwrite:
            print(f"[skip] exists: {synthetic_jsonl}")
            skipped += 1
            continue

        if latest_run.exists():
            stages = "archetype-synthetic,index"
        elif args.auto_train_if_missing:
            stages = "archetype-train,archetype-validate,archetype-synthetic,index"
        else:
            print(f"[skip] missing latest run (use --auto-train-if-missing): {latest_run}")
            skipped += 1
            continue

        cmd = [
            args.python,
            "benchmark/scripts/run_split_pipeline.py",
            "--split-root",
            str(split_root),
            "--split-base-root",
            str(split_base_root),
            "--runs-root",
            str(runs_root),
            "--stages",
            stages,
            "--synthetic-model",
            str(args.synthetic_model),
            "--oracle-archetype-summary",
            str(resolve_repo_path(args.oracle_archetype_summary)),
        ]
        train_args = list(args.train_arg or [])
        validate_args = list(args.validate_arg or [])
        if args.auto_train_if_missing and not latest_run.exists() and args.fast_default_train and not train_args:
            train_args = ["--models mean linear ridge --allow-tag-errors"]
        for extra in train_args:
            cmd.extend(["--train-arg", str(extra)])
        for extra in validate_args:
            cmd.extend(["--validate-arg", str(extra)])
        run_cmd(cmd, dry_run=args.dry_run)
        processed += 1

    print(f"Processed splits: {processed}")
    print(f"Skipped splits: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
