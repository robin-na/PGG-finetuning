#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


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


def parse_csv_arg(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_analysis_args(values: List[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        out.extend(shlex.split(value))
    return out


def find_latest_run_dir(variant_root: Path) -> Path:
    candidates = [
        d
        for d in variant_root.iterdir()
        if d.is_dir()
        and (d / "macro_simulation_eval.csv").exists()
        and (d / "macro_simulation_eval.csv").stat().st_size > 0
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No run directories with macro_simulation_eval.csv under {variant_root}"
        )
    numeric = [d for d in candidates if d.name.isdigit()]
    if numeric:
        return max(numeric, key=lambda d: int(d.name))
    return max(candidates, key=lambda d: (d.stat().st_mtime, d.name))


def run_cmd(cmd: List[str], dry_run: bool) -> None:
    print("+", " ".join(shlex.quote(x) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run macro comparison analysis for a split by selecting the latest run "
            "from each named variant folder."
        )
    )
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--split-base-root", type=Path, default=Path("benchmark"))
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/benchmark/runs"))
    parser.add_argument(
        "--variants",
        type=str,
        default="no_archetype,random_archetype,oracle_archetype,retrieved_archetype",
        help="Comma-separated variant folders under macro_simulation_eval.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="no archetype,random archetype,oracle archetype,retrieved archetype",
        help="Comma-separated labels aligned with --variants.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing variants instead of failing.",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("reports/benchmark/macro_simulation_eval"),
    )
    parser.add_argument("--analysis-run-id", type=str, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--binary-factors",
        type=str,
        default=None,
        help="Optional override passed to Macro_simulation_eval/analysis/run_analysis.py",
    )
    parser.add_argument(
        "--median-factors",
        type=str,
        default=None,
        help="Optional override passed to Macro_simulation_eval/analysis/run_analysis.py",
    )
    parser.add_argument("--shared-games-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--analysis-arg",
        action="append",
        default=[],
        help="Extra arg string passed to Macro_simulation_eval/analysis/run_analysis.py. Can repeat.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_root = resolve_repo_path(args.split_root)
    split_base_root = resolve_repo_path(args.split_base_root)
    runs_root = resolve_repo_path(args.runs_root)
    analysis_root = resolve_repo_path(args.analysis_root)

    split_rel = split_rel_path(split_root, split_base_root)
    macro_root = runs_root / split_rel / "macro_simulation_eval"
    if not macro_root.exists():
        raise FileNotFoundError(f"Macro root not found: {macro_root}")

    variants = parse_csv_arg(args.variants)
    labels = parse_csv_arg(args.labels)
    if len(variants) != len(labels):
        raise ValueError("labels length must match variants length.")

    selected_run_refs: List[str] = []
    selected_labels: List[str] = []
    for variant, label in zip(variants, labels):
        variant_root = macro_root / variant
        if not variant_root.exists():
            if args.allow_missing:
                continue
            raise FileNotFoundError(f"Missing variant folder: {variant_root}")
        run_dir = find_latest_run_dir(variant_root)
        selected_run_refs.append(f"{variant}/{run_dir.name}")
        selected_labels.append(label)

    if len(selected_run_refs) < 1:
        raise ValueError("Need at least one variant run to run macro analysis.")

    analysis_run_id = args.analysis_run_id
    if not analysis_run_id:
        split_id = str(split_rel).replace("/", "__")
        analysis_run_id = f"{split_id}__macro_variants_latest"

    human_analysis_csv = split_root / "processed_data" / "df_analysis_val.csv"
    cmd = [
        args.python,
        "Macro_simulation_eval/analysis/run_analysis.py",
        "--eval_root",
        str(macro_root),
        "--compare_run_ids",
        ",".join(selected_run_refs),
        "--compare_labels",
        ",".join(selected_labels),
        "--analysis_root",
        str(analysis_root),
        "--analysis_run_id",
        analysis_run_id,
        "--human_analysis_csv",
        str(human_analysis_csv),
    ]
    if args.binary_factors:
        cmd.extend(["--binary_factors", args.binary_factors])
    if args.median_factors:
        cmd.extend(["--median_factors", args.median_factors])
    if args.shared_games_only:
        cmd.append("--shared_games_only")
    if args.no_plots:
        cmd.append("--no_plots")
    cmd.extend(parse_analysis_args(args.analysis_arg))

    print("Selected runs:")
    for run_ref, label in zip(selected_run_refs, selected_labels):
        print(f"- {label}: {run_ref}")

    run_cmd(cmd, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

