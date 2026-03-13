#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORACLE_VAL_POOL = Path("Persona/archetype_oracle_gpt51_val.jsonl")
DEFAULT_ORACLE_LEARN_POOL = Path("Persona/archetype_oracle_gpt51_learn.jsonl")


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_extra_args(values: List[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        out.extend(shlex.split(value))
    return out


def has_cli_flag(args_list: List[str], flag: str) -> bool:
    return any(str(x).strip() == flag for x in args_list)


def to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def ensure_not_under_benchmark(path: Path, label: str) -> None:
    benchmark_root = (REPO_ROOT / "benchmark").resolve()
    if _is_subpath(path, benchmark_root):
        raise ValueError(
            f"Unsafe {label} path under benchmark/: {to_repo_rel(path)}. "
            "Generated run artifacts must be written under outputs/ (not benchmark/)."
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_run_index(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def write_run_index(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_latest_archetype_run(archetype_output_root: Path) -> Optional[Path]:
    latest_file = archetype_output_root / "latest_run.txt"
    if not latest_file.exists():
        return None
    raw = latest_file.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    candidates: List[Path] = []
    run_dir = Path(raw)
    if run_dir.is_absolute():
        candidates.append(run_dir)
    else:
        candidates.append((REPO_ROOT / run_dir).resolve())
    marker = "outputs/benchmark/runs/"
    if marker in raw:
        suffix = raw.split(marker, 1)[1].lstrip("/\\")
        candidates.append((REPO_ROOT / "outputs" / "benchmark" / "runs" / suffix).resolve())
    marker_default = "outputs/default/runs/"
    if marker_default in raw:
        suffix = raw.split(marker_default, 1)[1].lstrip("/\\")
        candidates.append((REPO_ROOT / "outputs" / "default" / "runs" / suffix).resolve())
    for c in candidates:
        if c.is_dir():
            return c
    return None


def find_latest_variant_run_dir(variant_root: Path, eval_csv_name: str) -> Optional[Path]:
    candidates = [
        d
        for d in variant_root.iterdir()
        if d.is_dir()
        and (d / eval_csv_name).exists()
        and (d / eval_csv_name).stat().st_size > 0
    ]
    if not candidates:
        return None
    numeric = [d for d in candidates if d.name.isdigit()]
    if numeric:
        return max(numeric, key=lambda d: int(d.name))
    return max(candidates, key=lambda d: (d.stat().st_mtime, d.name))


def normalize_path_text(path_text: str) -> str:
    text = str(path_text or "").strip()
    if not text:
        return text
    as_path = Path(text)
    if not as_path.is_absolute():
        as_path = (REPO_ROOT / as_path).resolve()
    return to_repo_rel(as_path)


def read_eval_config_fields(run_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None, None
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    if not isinstance(cfg, dict):
        return None, None

    # Support older and newer config payloads.
    candidates = [cfg.get("args"), cfg.get("model"), cfg]
    mode: Optional[str] = None
    pool: Optional[str] = None
    for rec in candidates:
        if not isinstance(rec, dict):
            continue
        if mode is None:
            raw_mode = str(rec.get("archetype") or rec.get("persona") or "").strip()
            if raw_mode:
                mode = raw_mode
        if pool is None:
            raw_pool = str(
                rec.get("archetype_summary_pool") or rec.get("persona_summary_pool") or ""
            ).strip()
            if raw_pool:
                pool = raw_pool
        if mode is not None and pool is not None:
            break
    return mode, pool


def _collect_eval_section(
    split_run_root: Path,
    eval_root_name: str,
    eval_csv_name: str,
) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    eval_root = split_run_root / eval_root_name
    eval_section: Dict[str, Any] = {
        "root": to_repo_rel(eval_root),
        "variants": {},
    }
    retrieved_summary_jsonl: Optional[str] = None
    retrieved_trace_jsonl: Optional[str] = None

    if eval_root.exists():
        for variant_dir in sorted([p for p in eval_root.iterdir() if p.is_dir()]):
            latest_run = find_latest_variant_run_dir(variant_dir, eval_csv_name)
            if latest_run is None:
                continue
            mode, pool = read_eval_config_fields(latest_run)
            variant_info: Dict[str, Any] = {
                "latest_run_dir": to_repo_rel(latest_run),
                "latest_run_id": latest_run.name,
            }
            if mode:
                variant_info["archetype_mode"] = mode
            if pool:
                variant_info["archetype_summary_pool"] = normalize_path_text(pool)
            eval_section["variants"][variant_dir.name] = variant_info

            if variant_dir.name == "retrieved_archetype" and pool:
                retrieved_summary_jsonl = normalize_path_text(pool)
                pool_path = Path(pool)
                if not pool_path.is_absolute():
                    pool_path = (REPO_ROOT / pool_path).resolve()
                if pool_path.suffix == ".jsonl":
                    trace_candidate = pool_path.with_name(f"{pool_path.stem}_trace.jsonl")
                    if trace_candidate.exists():
                        retrieved_trace_jsonl = to_repo_rel(trace_candidate)

    return eval_section, retrieved_summary_jsonl, retrieved_trace_jsonl


def refresh_run_index(
    run_index_path: Path,
    split_root: Path,
    split_rel: Path,
    runs_root: Path,
    split_run_root: Path,
    archetype_output_root: Path,
) -> None:
    payload = load_run_index(run_index_path)
    payload["updated_at_utc"] = utc_now_iso()
    payload["split"] = {
        "split_root": to_repo_rel(split_root),
        "split_relative_path": str(split_rel),
        "runs_root": to_repo_rel(runs_root),
        "split_run_root": to_repo_rel(split_run_root),
    }

    archetype_section: Dict[str, Any] = {
        "output_root": to_repo_rel(archetype_output_root),
    }
    latest_archetype = load_latest_archetype_run(archetype_output_root)
    if latest_archetype is not None:
        archetype_section["latest_run_dir"] = to_repo_rel(latest_archetype)
        archetype_section["latest_run_id"] = latest_archetype.name
    payload["archetype_retrieval"] = archetype_section

    micro_section, micro_retrieved_summary, micro_retrieved_trace = _collect_eval_section(
        split_run_root=split_run_root,
        eval_root_name="micro_behavior_eval",
        eval_csv_name="micro_behavior_eval.csv",
    )
    payload["micro_behavior_eval"] = micro_section

    macro_section, macro_retrieved_summary, macro_retrieved_trace = _collect_eval_section(
        split_run_root=split_run_root,
        eval_root_name="macro_simulation_eval",
        eval_csv_name="macro_simulation_eval.csv",
    )
    payload["macro_simulation_eval"] = macro_section

    retrieved_summary_jsonl = micro_retrieved_summary or macro_retrieved_summary
    retrieved_trace_jsonl = micro_retrieved_trace or macro_retrieved_trace
    if retrieved_summary_jsonl is None:
        synthetic_candidates = sorted(
            (split_run_root / "archetype_retrieval" / "validation_wave").glob(
                "synthetic_archetype_*_val.jsonl"
            )
        )
        if synthetic_candidates:
            chosen = synthetic_candidates[0]
            retrieved_summary_jsonl = to_repo_rel(chosen)
            trace_candidate = chosen.with_name(f"{chosen.stem}_trace.jsonl")
            if trace_candidate.exists():
                retrieved_trace_jsonl = to_repo_rel(trace_candidate)
    if retrieved_summary_jsonl is not None:
        retrieved_section = dict(payload.get("retrieved_archetype") or {})
        retrieved_section["summary_jsonl"] = retrieved_summary_jsonl
        if retrieved_trace_jsonl is not None:
            retrieved_section["trace_jsonl"] = retrieved_trace_jsonl
        payload["retrieved_archetype"] = retrieved_section

    write_run_index(run_index_path, payload)


def infer_variant_name(archetype_mode: str, archetype_pool: Path) -> str:
    mode = str(archetype_mode or "").strip()
    if mode == "none":
        return "no_archetype"
    if mode == "random_summary":
        return "random_archetype"
    if mode == "config_bank_archetype":
        return "config_bank_archetype"

    pool_resolved = archetype_pool.resolve()
    pool_text = str(pool_resolved).lower()
    oracle_candidates = {
        str((REPO_ROOT / "Persona" / "archetype_oracle_gpt51_val.jsonl").resolve()).lower(),
        str(
            (
                REPO_ROOT
                / "outputs"
                / "benchmark"
                / "cache"
                / "archetype"
                / "archetype_oracle_gpt51_learn_val_union_finished.jsonl"
            ).resolve()
        ).lower(),
    }
    if pool_text in oracle_candidates:
        return "oracle_archetype"

    retrieved_markers = [
        "synthetic_persona",
        "synthetic_archetype",
        "retrieved",
        "archetype_retrieval/validation_wave",
    ]
    if any(marker in pool_text for marker in retrieved_markers):
        return "retrieved_archetype"
    return "oracle_archetype"


def run_cmd(cmd: List[str], dry_run: bool) -> None:
    print("+", " ".join(shlex.quote(x) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def parse_stages(stage_text: str) -> List[str]:
    stages = [s.strip().lower() for s in stage_text.split(",") if s.strip()]
    if "all" in stages:
        return ["micro", "macro", "archetype-train", "archetype-validate", "archetype-synthetic"]
    allowed = {
        "micro",
        "macro",
        "archetype-train",
        "archetype-validate",
        "archetype-synthetic",
        "index",
    }
    bad = [s for s in stages if s not in allowed]
    if bad:
        raise ValueError(f"Unsupported stages: {bad}. Allowed: {sorted(allowed)}")
    return stages


def resolve_archetype_pool_for_mode(archetype_mode: str, pool_arg: Optional[Path]) -> Path:
    if pool_arg is not None:
        return resolve_repo_path(pool_arg)
    if archetype_mode in {"random_summary", "config_bank_archetype"}:
        return (REPO_ROOT / DEFAULT_ORACLE_LEARN_POOL).resolve()
    return (REPO_ROOT / DEFAULT_ORACLE_VAL_POOL).resolve()


def should_auto_build_union_pool(archetype_pool: Path) -> bool:
    text = str(archetype_pool.resolve()).lower()
    return (
        "benchmark/cache/archetype" in text
        or "learn_val_union" in archetype_pool.name.lower()
    )


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


def ensure_split_structure(split_root: Path) -> None:
    required = [
        split_root / "raw_data" / "learning_wave" / "player-rounds.csv",
        split_root / "raw_data" / "validation_wave" / "player-rounds.csv",
        split_root / "processed_data" / "df_analysis_learn.csv",
        split_root / "processed_data" / "df_analysis_val.csv",
        split_root / "demographics" / "demographics_numeric_learn.csv",
        split_root / "demographics" / "demographics_numeric_val.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Split root is missing required files:\n" + "\n".join(missing)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run micro/macro/archetype pipeline using a benchmark OOD split root."
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        required=True,
        help=(
            "Split direction root, e.g. "
            "benchmark/data_ood_splits/all_or_nothing/false_to_true "
            "or benchmark/data_ood_splits_wave_anchored/all_or_nothing/false_to_true"
        ),
    )
    parser.add_argument(
        "--split-base-root",
        type=Path,
        default=Path("benchmark"),
        help="Base directory used to map split roots into --runs-root.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("outputs/benchmark/runs"),
        help=(
            "Root for generated split artifacts. "
            "Default: outputs/benchmark/runs/<split-relative-path>/..."
        ),
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help=(
            "Comma-separated: all | index | micro | macro | "
            "archetype-train | archetype-validate | archetype-synthetic"
        ),
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to run subcommands.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not execute.",
    )

    parser.add_argument(
        "--skip-build-archetype-banks",
        action="store_true",
        help="Do not auto-build split-specific archetype banks before train/validate.",
    )
    parser.add_argument(
        "--overwrite-archetype-banks",
        action="store_true",
        help="Overwrite existing split-specific archetype bank files when auto-building.",
    )

    parser.add_argument(
        "--archetype-mode",
        dest="archetype_mode",
        choices=["none", "matched_summary", "random_summary", "config_bank_archetype"],
        default="matched_summary",
        help=(
            "Archetype mode for micro and macro eval. "
            "Use matched_summary for oracle/retrieved (pool controls source)."
        ),
    )
    parser.add_argument(
        "--persona-mode",
        dest="archetype_mode",
        choices=["none", "matched_summary", "random_summary", "config_bank_archetype"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--archetype-summary-pool",
        dest="archetype_summary_pool",
        type=Path,
        default=None,
        help=(
            "Archetype summary pool JSONL used when archetype mode is enabled. "
            "Default depends on mode: "
            "random_summary -> Persona/archetype_oracle_gpt51_learn.jsonl, "
            "config_bank_archetype -> Persona/archetype_oracle_gpt51_learn.jsonl, "
            "matched_summary -> Persona/archetype_oracle_gpt51_val.jsonl."
        ),
    )
    parser.add_argument(
        "--persona-summary-pool",
        dest="archetype_summary_pool",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-auto-build-archetype-pool",
        dest="no_auto_build_archetype_pool",
        action="store_true",
        help="Do not auto-build the union archetype summary pool if missing.",
    )
    parser.add_argument(
        "--no-auto-build-persona-pool",
        dest="no_auto_build_archetype_pool",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--micro-output-root",
        type=Path,
        default=None,
        help=(
            "Override micro output root. "
            "Default: <runs-root>/<split-relative-path>/micro_behavior_eval/<variant>"
        ),
    )
    parser.add_argument(
        "--micro-variant",
        choices=[
            "auto",
            "no_archetype",
            "random_archetype",
            "oracle_archetype",
            "retrieved_archetype",
            "config_bank_archetype",
        ],
        default="auto",
        help=(
            "Folder name under micro_behavior_eval for this run. "
            "Default auto infers from archetype mode + summary pool."
        ),
    )
    parser.add_argument(
        "--macro-output-root",
        type=Path,
        default=None,
        help=(
            "Override macro output root. "
            "Default: <runs-root>/<split-relative-path>/macro_simulation_eval/<variant>"
        ),
    )
    parser.add_argument(
        "--macro-variant",
        choices=[
            "auto",
            "no_archetype",
            "random_archetype",
            "oracle_archetype",
            "retrieved_archetype",
            "config_bank_archetype",
        ],
        default="auto",
        help=(
            "Folder name under macro_simulation_eval for this run. "
            "Default auto infers from archetype mode + summary pool."
        ),
    )
    parser.add_argument(
        "--archetype-output-root",
        type=Path,
        default=None,
        help=(
            "Override archetype model output root. "
            "Default: <runs-root>/<split-relative-path>/archetype_retrieval/model_runs"
        ),
    )

    parser.add_argument(
        "--micro-arg",
        action="append",
        default=[],
        help='Extra arg string for micro run, e.g. --micro-arg "--provider openai". Can repeat.',
    )
    parser.add_argument(
        "--macro-arg",
        action="append",
        default=[],
        help='Extra arg string for macro run, e.g. --macro-arg "--provider openai". Can repeat.',
    )
    parser.add_argument(
        "--train-arg",
        action="append",
        default=[],
        help='Extra arg string for archetype train. Can repeat.',
    )
    parser.add_argument(
        "--validate-arg",
        action="append",
        default=[],
        help='Extra arg string for archetype validate. Can repeat.',
    )
    parser.add_argument(
        "--game-ids-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV listing game IDs to evaluate. "
            "If set, automatically injects --game_ids into micro/macro runs."
        ),
    )
    parser.add_argument(
        "--game-ids-column",
        type=str,
        default="gameId",
        help="Column name used when --game-ids-csv is set.",
    )
    parser.add_argument(
        "--synthetic-model",
        type=str,
        default="ridge",
        help="Model to use when generating synthetic archetypes.",
    )
    parser.add_argument(
        "--oracle-archetype-summary",
        type=Path,
        default=Path("Persona/archetype_oracle_gpt51_val.jsonl"),
        help="Oracle archetype JSONL used only to transfer game_finished to synthetic rows.",
    )
    parser.add_argument(
        "--synthetic-archetype-output-jsonl",
        type=Path,
        default=None,
        help=(
            "Override synthetic archetype JSONL output path. "
            "Default: <split-run-root>/archetype_retrieval/validation_wave/"
            "synthetic_archetype_<model>_val.jsonl"
        ),
    )
    parser.add_argument(
        "--synthetic-archetype-output-trace-jsonl",
        type=Path,
        default=None,
        help=(
            "Override synthetic archetype trace JSONL output path. "
            "Default: <split-run-root>/archetype_retrieval/validation_wave/"
            "synthetic_archetype_<model>_val_trace.jsonl"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_root = resolve_repo_path(args.split_root)
    split_base_root = resolve_repo_path(args.split_base_root)
    runs_root = resolve_repo_path(args.runs_root)
    ensure_split_structure(split_root)
    stages = parse_stages(args.stages)

    split_rel = split_rel_path(split_root, split_base_root)
    split_run_root = runs_root / split_rel
    run_index_path = split_run_root / "run_index.json"
    ensure_not_under_benchmark(runs_root, "--runs-root")

    selected_game_ids_arg: Optional[str] = None
    if args.game_ids_csv is not None:
        game_ids_csv = resolve_repo_path(args.game_ids_csv)
        if not game_ids_csv.exists():
            raise FileNotFoundError(f"--game-ids-csv not found: {to_repo_rel(game_ids_csv)}")
        import pandas as pd  # local import to avoid extra dependency at import time

        df_ids = pd.read_csv(game_ids_csv)
        col = str(args.game_ids_column or "").strip()
        if not col:
            raise ValueError("--game-ids-column cannot be empty.")
        if col not in df_ids.columns:
            raise KeyError(
                f"--game-ids-column '{col}' not present in {to_repo_rel(game_ids_csv)}; "
                f"available columns: {list(df_ids.columns)}"
            )
        game_ids = [str(x).strip() for x in df_ids[col].tolist() if str(x).strip()]
        if not game_ids:
            raise ValueError(f"No non-empty game IDs found in {to_repo_rel(game_ids_csv)} column '{col}'.")
        selected_game_ids_arg = ",".join(game_ids)

    archetype_root = split_run_root / "archetype_retrieval"
    archetype_learning = archetype_root / "learning_wave"
    archetype_validation = archetype_root / "validation_wave"
    archetype_output_root = (
        resolve_repo_path(args.archetype_output_root)
        if args.archetype_output_root is not None
        else (archetype_root / "model_runs")
    )
    ensure_not_under_benchmark(archetype_output_root, "archetype output")

    need_archetype = any(s in {"archetype-train", "archetype-validate"} for s in stages)
    if need_archetype and not args.skip_build_archetype_banks:
        need_build = not (
            (archetype_learning / "CONTRIBUTION").exists()
            and (archetype_validation / "CONTRIBUTION").exists()
        )
        if need_build or args.overwrite_archetype_banks:
            build_cmd = [
                args.python,
                "benchmark/scripts/build_split_archetype_banks.py",
                "--split-root",
                str(split_root),
                "--output-root",
                str(archetype_root),
            ]
            if args.overwrite_archetype_banks:
                build_cmd.append("--overwrite")
            run_cmd(build_cmd, args.dry_run)

    if "micro" in stages:
        archetype_mode = args.archetype_mode
        archetype_pool = resolve_archetype_pool_for_mode(archetype_mode, args.archetype_summary_pool)
        micro_extra_args = parse_extra_args(args.micro_arg)
        if selected_game_ids_arg and has_cli_flag(micro_extra_args, "--game_ids"):
            raise ValueError("Conflict: --game-ids-csv and --micro-arg containing --game_ids.")
        inferred_variant = infer_variant_name(archetype_mode, archetype_pool)
        variant_name = inferred_variant if args.micro_variant == "auto" else args.micro_variant
        micro_output_root = (
            resolve_repo_path(args.micro_output_root)
            if args.micro_output_root is not None
            else (split_run_root / "micro_behavior_eval" / variant_name)
        )
        ensure_not_under_benchmark(micro_output_root, "micro output")

        if archetype_mode != "none":
            if not archetype_pool.exists():
                if (
                    not args.no_auto_build_archetype_pool
                    and should_auto_build_union_pool(archetype_pool)
                ):
                    run_cmd(
                        [
                            args.python,
                            "benchmark/scripts/build_union_archetype_summary_pool.py",
                            "--output-jsonl",
                            str(archetype_pool),
                        ],
                        args.dry_run,
                    )
                else:
                    raise FileNotFoundError(
                        "archetype summary pool file not found: "
                        f"{to_repo_rel(archetype_pool)}"
                    )

        micro_cmd = [
            args.python,
            "Micro_behavior_eval/run_micro_behavior_eval.py",
            "--data_root",
            str(split_root),
            "--wave",
            "validation_wave",
            "--output_root",
            str(micro_output_root),
        ]
        if archetype_mode != "none":
            micro_cmd.extend(
                [
                    "--archetype",
                    archetype_mode,
                    "--archetype_summary_pool",
                    str(archetype_pool),
                ]
            )
        if selected_game_ids_arg:
            micro_cmd.extend(["--game_ids", selected_game_ids_arg])
        micro_cmd.extend(micro_extra_args)
        run_cmd(micro_cmd, args.dry_run)

    if "macro" in stages:
        archetype_mode = args.archetype_mode
        archetype_pool = resolve_archetype_pool_for_mode(archetype_mode, args.archetype_summary_pool)
        macro_extra_args = parse_extra_args(args.macro_arg)
        if selected_game_ids_arg and has_cli_flag(macro_extra_args, "--game_ids"):
            raise ValueError("Conflict: --game-ids-csv and --macro-arg containing --game_ids.")
        inferred_variant = infer_variant_name(archetype_mode, archetype_pool)
        variant_name = inferred_variant if args.macro_variant == "auto" else args.macro_variant
        macro_output_root = (
            resolve_repo_path(args.macro_output_root)
            if args.macro_output_root is not None
            else (split_run_root / "macro_simulation_eval" / variant_name)
        )
        ensure_not_under_benchmark(macro_output_root, "macro output")

        if archetype_mode != "none":
            if not archetype_pool.exists():
                if (
                    not args.no_auto_build_archetype_pool
                    and should_auto_build_union_pool(archetype_pool)
                ):
                    run_cmd(
                        [
                            args.python,
                            "benchmark/scripts/build_union_archetype_summary_pool.py",
                            "--output-jsonl",
                            str(archetype_pool),
                        ],
                        args.dry_run,
                    )
                else:
                    raise FileNotFoundError(
                        "archetype summary pool file not found: "
                        f"{to_repo_rel(archetype_pool)}"
                    )

        macro_cmd = [
            args.python,
            "Macro_simulation_eval/run_macro_simulation_eval.py",
            "--data_root",
            str(split_root),
            "--wave",
            "validation_wave",
            "--output_root",
            str(macro_output_root),
        ]
        if archetype_mode != "none":
            macro_cmd.extend(
                [
                    "--archetype",
                    archetype_mode,
                    "--archetype_summary_pool",
                    str(archetype_pool),
                ]
            )
        if selected_game_ids_arg:
            macro_cmd.extend(["--game_ids", selected_game_ids_arg])
        macro_cmd.extend(macro_extra_args)
        run_cmd(macro_cmd, args.dry_run)

    if "archetype-train" in stages:
        train_cmd = [
            args.python,
            "Persona/archetype_retrieval/train_archetype_retrieval.py",
            "--learning-wave-root",
            str(archetype_learning),
            "--demographics-csv",
            str(split_root / "demographics" / "demographics_numeric_learn.csv"),
            "--environment-csv",
            str(split_root / "processed_data" / "df_analysis_learn.csv"),
            "--output-root",
            str(archetype_output_root),
        ]
        train_cmd.extend(parse_extra_args(args.train_arg))
        run_cmd(train_cmd, args.dry_run)

    if "archetype-validate" in stages:
        validate_cmd = [
            args.python,
            "Persona/archetype_retrieval/validate_archetype_retrieval.py",
            "--output-root",
            str(archetype_output_root),
            "--validation-wave-root",
            str(archetype_validation),
            "--demographics-csv",
            str(split_root / "demographics" / "demographics_numeric_val.csv"),
            "--environment-csv",
            str(split_root / "processed_data" / "df_analysis_val.csv"),
        ]
        validate_cmd.extend(parse_extra_args(args.validate_arg))
        run_cmd(validate_cmd, args.dry_run)

    if "archetype-synthetic" in stages:
        latest_run_dir = load_latest_archetype_run(archetype_output_root)
        if latest_run_dir is None:
            raise FileNotFoundError(
                f"No latest archetype run found under {to_repo_rel(archetype_output_root)}. "
                "Run archetype-train first."
            )
        synthetic_model = str(args.synthetic_model or "ridge").strip().lower()
        synthetic_output_jsonl = (
            resolve_repo_path(args.synthetic_archetype_output_jsonl)
            if args.synthetic_archetype_output_jsonl is not None
            else (
                split_run_root
                / "archetype_retrieval"
                / "validation_wave"
                / f"synthetic_archetype_{synthetic_model}_val.jsonl"
            )
        )
        ensure_not_under_benchmark(synthetic_output_jsonl, "synthetic archetype output")
        synthetic_output_trace = (
            resolve_repo_path(args.synthetic_archetype_output_trace_jsonl)
            if args.synthetic_archetype_output_trace_jsonl is not None
            else (
                split_run_root
                / "archetype_retrieval"
                / "validation_wave"
                / f"synthetic_archetype_{synthetic_model}_val_trace.jsonl"
            )
        )
        synth_cmd = [
            args.python,
            "Persona/archetype_retrieval/generate_synthetic_persona_val.py",
            "--run-dir",
            str(latest_run_dir),
            "--demographics-csv",
            str(split_root / "demographics" / "demographics_numeric_val.csv"),
            "--environment-csv",
            str(split_root / "processed_data" / "df_analysis_val.csv"),
            "--oracle-summary-jsonl",
            str(resolve_repo_path(args.oracle_archetype_summary)),
            "--model",
            synthetic_model,
            "--output-jsonl",
            str(synthetic_output_jsonl),
            "--output-trace-jsonl",
            str(synthetic_output_trace),
        ]
        run_cmd(synth_cmd, args.dry_run)

    if not args.dry_run:
        refresh_run_index(
            run_index_path=run_index_path,
            split_root=split_root,
            split_rel=split_rel,
            runs_root=runs_root,
            split_run_root=split_run_root,
            archetype_output_root=archetype_output_root,
        )
        print(f"Wrote: {to_repo_rel(run_index_path)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
