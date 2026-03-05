#!/usr/bin/env python3
"""
Run detailed validation plotting (with random baseline + error bars) for each benchmark split,
then build a consolidated cross-split figure.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


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


def resolve_latest_run_path(raw: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        return Path("")
    candidates: List[Path] = []
    p = Path(text)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((REPO_ROOT / p).resolve())

    marker = "outputs/benchmark/runs/"
    if marker in text:
        suffix = text.split(marker, 1)[1].lstrip("/\\")
        candidates.append((REPO_ROOT / "outputs" / "benchmark" / "runs" / suffix).resolve())
    marker_default = "outputs/default/runs/"
    if marker_default in text:
        suffix = text.split(marker_default, 1)[1].lstrip("/\\")
        candidates.append((REPO_ROOT / "outputs" / "default" / "runs" / suffix).resolve())

    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0] if candidates else Path(text)


def discover_split_roots(
    include_default: bool,
    default_root: Path,
    ood_roots: Sequence[Path],
) -> List[Path]:
    roots: List[Path] = []
    if include_default and default_root.is_dir():
        roots.append(default_root)
    for ood_root in ood_roots:
        if ood_root.is_dir():
            for config_dir in sorted([p for p in ood_root.iterdir() if p.is_dir()]):
                for direction_dir in sorted([p for p in config_dir.iterdir() if p.is_dir()]):
                    roots.append(direction_dir)
    return roots


def split_meta(root: Path, default_root: Path, ood_roots: Sequence[Path]) -> Dict[str, str]:
    if root.resolve() == default_root.resolve():
        return {
            "split_group": "default",
            "split_name": "default",
            "config": "default",
            "direction": "default",
        }
    rel = None
    for ood_root in ood_roots:
        try:
            rel = root.resolve().relative_to(ood_root.resolve())
            break
        except Exception:
            continue
    if rel is None:
        rel = Path(root.name)
    parts = list(rel.parts)
    if len(parts) >= 2:
        config = parts[0]
        direction = parts[1]
        name = f"{config}/{direction}"
    else:
        config = rel.as_posix()
        direction = ""
        name = rel.as_posix()
    return {
        "split_group": "ood",
        "split_name": name,
        "config": config,
        "direction": direction,
    }


def find_latest_run(root: Path, runs_root: Path, split_base_root: Path) -> tuple[Path, Path]:
    rel = split_rel_path(root, split_base_root)
    archetype_root = runs_root / rel / "archetype_retrieval"
    latest = archetype_root / "model_runs" / "latest_run.txt"
    if not latest.exists():
        raise FileNotFoundError(f"Missing latest run pointer: {to_repo_rel(latest)}")
    run_dir = resolve_latest_run_path(latest.read_text(encoding="utf-8"))
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing run directory from latest_run.txt: {run_dir}")
    validation_wave_root = archetype_root / "validation_wave"
    return run_dir, validation_wave_root


def run_cmd(cmd: List[str], dry_run: bool) -> None:
    print("+", " ".join(shlex.quote(x) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def infer_hit_col(columns: Sequence[str], requested_k: int) -> str | None:
    preferred = f"oracle_hit_at_{requested_k}"
    if preferred in columns:
        return preferred
    ks: List[int] = []
    for c in columns:
        m = re.fullmatch(r"oracle_hit_at_(\d+)", str(c))
        if m:
            ks.append(int(m.group(1)))
    if not ks:
        return None
    return f"oracle_hit_at_{max(ks)}"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = v.notna() & (w > 0)
    if not mask.any():
        return float(v.mean()) if v.notna().any() else float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def weighted_std(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = v.notna() & (w > 0)
    if not mask.any():
        return float(v.std(ddof=0)) if v.notna().any() else float("nan")
    vv = v[mask].astype(float).to_numpy()
    ww = w[mask].astype(float).to_numpy()
    mean = np.average(vv, weights=ww)
    var = np.average((vv - mean) ** 2, weights=ww)
    return float(np.sqrt(max(0.0, var)))


def aggregate_one_split(
    metrics: pd.DataFrame,
    split_info: Dict[str, str],
    models: set[str],
    hit_col: str,
    weight_by_rows: bool,
) -> pd.DataFrame:
    df = metrics.copy()
    df["model"] = df["model"].astype(str).str.lower()
    df = df[df["status"].astype(str).str.lower() == "ok"].copy()
    df = df[df["model"].isin(models)].copy()
    if df.empty:
        return pd.DataFrame()

    metric_cols = [
        "retrieval_cosine_regret_mean",
        "retrieved_true_cosine_mean",
        "oracle_true_cosine_mean",
        hit_col,
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    if not metric_cols:
        return pd.DataFrame()

    out: List[Dict[str, object]] = []
    for model, g in df.groupby("model", sort=True):
        rec: Dict[str, object] = {
            **split_info,
            "model": model,
            "n_tags": int(g["tag"].nunique()) if "tag" in g.columns else int(len(g)),
            "n_rows_sum": int(pd.to_numeric(g.get("n_validation_rows"), errors="coerce").fillna(0).sum())
            if "n_validation_rows" in g.columns
            else int(len(g)),
        }
        for col in metric_cols:
            if weight_by_rows and "n_validation_rows" in g.columns:
                rec[col] = weighted_mean(g[col], g["n_validation_rows"])
                rec[f"{col}_std"] = weighted_std(g[col], g["n_validation_rows"])
            else:
                arr = pd.to_numeric(g[col], errors="coerce")
                rec[col] = float(arr.mean())
                rec[f"{col}_std"] = float(arr.std(ddof=0))
        out.append(rec)
    return pd.DataFrame(out)


def model_order(models: Iterable[str]) -> List[str]:
    preferred = ["ridge", "linear", "random", "elastic_net", "mlp", "mean"]
    model_list = list(dict.fromkeys([str(m).lower() for m in models]))
    out = [m for m in preferred if m in model_list]
    out.extend([m for m in model_list if m not in out])
    return out


def ordered_split_names(df: pd.DataFrame) -> List[str]:
    names = list(dict.fromkeys(df["split_name"].astype(str).tolist()))
    if "default" in names:
        names.remove("default")
        return ["default", *sorted(names)]
    return sorted(names)


def plot_grouped_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_col: str,
    metric_std_col: str,
    ylabel: str,
    title: str,
) -> None:
    splits = ordered_split_names(df)
    models = model_order(df["model"].unique())
    x = np.arange(len(splits), dtype=float)
    width = 0.8 / max(1, len(models))

    p_mean = (
        df.pivot(index="split_name", columns="model", values=metric_col)
        .reindex(splits)
        .copy()
    )
    p_std = (
        df.pivot(index="split_name", columns="model", values=metric_std_col)
        .reindex(splits)
        .copy()
    )

    for i, model in enumerate(models):
        if model not in p_mean.columns:
            continue
        vals = p_mean[model].to_numpy()
        errs = p_std[model].to_numpy() if model in p_std.columns else None
        ax.bar(
            x + (i - (len(models) - 1) / 2) * width,
            vals,
            width=width,
            yerr=errs,
            capsize=2 if errs is not None else 0,
            label=model,
        )

    ax.set_xticks(x, splits, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)


def plot_consolidated(
    split_model_df: pd.DataFrame,
    hit_col: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(14, split_model_df["split_name"].nunique() * 0.75), 5.8),
        sharex=False,
    )

    plot_grouped_panel(
        axes[0],
        split_model_df,
        metric_col="retrieval_cosine_regret_mean",
        metric_std_col="retrieval_cosine_regret_mean_std",
        ylabel="Regret (lower better)",
        title="Per Split: Retrieval Regret (with random baseline)",
    )
    plot_grouped_panel(
        axes[1],
        split_model_df,
        metric_col=hit_col,
        metric_std_col=f"{hit_col}_std",
        ylabel=f"{hit_col} (higher better)",
        title=f"Per Split: {hit_col} (with random baseline)",
    )
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-split validation plots (random baseline + error bars) and one "
            "consolidated cross-split figure."
        )
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--split-base-root", type=Path, default=Path("benchmark"))
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/benchmark/runs"))
    parser.add_argument("--default-root", type=Path, default=Path("benchmark/data"))
    parser.add_argument(
        "--ood-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "OOD split root. Can be repeated. "
            "Defaults to benchmark/data_ood_splits and "
            "benchmark/data_ood_splits_wave_anchored."
        ),
    )
    parser.add_argument(
        "--include-default",
        action="store_true",
        help="Include benchmark/data alongside OOD splits.",
    )
    parser.add_argument(
        "--run-per-split",
        action="store_true",
        help="Run Persona/archetype_retrieval/plot_validation_performance.py for each split first.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear", "ridge", "random"],
        help="Models to include in consolidated plots.",
    )
    parser.add_argument(
        "--weight-by-rows",
        action="store_true",
        help="Use n_validation_rows as weights when aggregating split-level model metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/benchmark/plots/embeddings"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    split_base_root = resolve_repo_path(args.split_base_root)
    runs_root = resolve_repo_path(args.runs_root)
    default_root = resolve_repo_path(args.default_root)
    ood_roots_raw = args.ood_root or [
        Path("benchmark/data_ood_splits"),
        Path("benchmark/data_ood_splits_wave_anchored"),
    ]
    ood_roots = [resolve_repo_path(p) for p in ood_roots_raw]
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roots = discover_split_roots(
        include_default=args.include_default,
        default_root=default_root,
        ood_roots=ood_roots,
    )
    if not roots:
        raise FileNotFoundError("No split roots found.")

    per_split_index_rows: List[Dict[str, object]] = []
    split_model_rows: List[pd.DataFrame] = []
    skipped: List[str] = []

    for root in roots:
        info = split_meta(root, default_root=default_root, ood_roots=ood_roots)
        try:
            run_dir, validation_wave_root = find_latest_run(root, runs_root, split_base_root)
        except FileNotFoundError as exc:
            skipped.append(f"{info['split_name']}: {exc}")
            continue

        if args.run_per_split:
            plot_cmd = [
                args.python,
                "Persona/archetype_retrieval/plot_validation_performance.py",
                "--run-dir",
                str(run_dir),
                "--validation-wave-root",
                str(validation_wave_root),
                "--demographics-csv",
                str(root / "demographics" / "demographics_numeric_val.csv"),
                "--environment-csv",
                str(root / "processed_data" / "df_analysis_val.csv"),
                "--top-k",
                str(args.top_k),
                "--random-repeats",
                str(args.random_repeats),
                "--bootstrap-repeats",
                str(args.bootstrap_repeats),
                "--random-seed",
                str(args.random_seed),
            ]
            run_cmd(plot_cmd, dry_run=args.dry_run)

        eval_dir = run_dir / "validation_eval"
        metrics_path = eval_dir / "validation_metrics_with_random.csv"
        if not metrics_path.exists():
            skipped.append(f"{info['split_name']}: missing {to_repo_rel(metrics_path)}")
            continue

        metrics = pd.read_csv(metrics_path)
        hit_col = infer_hit_col(metrics.columns, requested_k=args.top_k)
        if hit_col is None:
            skipped.append(f"{info['split_name']}: no oracle_hit_at_* column in {to_repo_rel(metrics_path)}")
            continue

        model_set = {str(m).strip().lower() for m in args.models if str(m).strip()}
        agg = aggregate_one_split(
            metrics=metrics,
            split_info=info,
            models=model_set,
            hit_col=hit_col,
            weight_by_rows=args.weight_by_rows,
        )
        if agg.empty:
            skipped.append(f"{info['split_name']}: no rows after filtering models/status.")
            continue

        agg.insert(0, "split_root", to_repo_rel(root))
        agg.insert(1, "run_dir", to_repo_rel(run_dir))
        agg["hit_col"] = hit_col
        split_model_rows.append(agg)

        per_split_index_rows.append(
            {
                "split_name": info["split_name"],
                "split_root": to_repo_rel(root),
                "run_dir": to_repo_rel(run_dir),
                "metrics_with_random_csv": to_repo_rel(metrics_path),
                "fig_per_tag_regret": to_repo_rel(eval_dir / "figures" / "per_tag_regret_with_random.png"),
                "fig_per_tag_hit_at_5": to_repo_rel(eval_dir / "figures" / "per_tag_hit_at_5_with_random.png"),
                "fig_across_tags_summary": to_repo_rel(eval_dir / "figures" / "across_tags_summary_with_random.png"),
            }
        )

    if not split_model_rows:
        raise RuntimeError("No split-level metrics available to consolidate.")

    consolidated = pd.concat(split_model_rows, ignore_index=True)
    consolidated = consolidated.sort_values(["split_group", "split_name", "model"]).reset_index(drop=True)

    # Use whichever hit column appears most often. With --top-k 5, this should be oracle_hit_at_5.
    hit_counts = consolidated["hit_col"].value_counts()
    hit_col = str(hit_counts.index[0])

    out_csv = output_dir / "consolidated_split_model_metrics_with_random.csv"
    consolidated.to_csv(out_csv, index=False)

    out_index = output_dir / "per_split_plot_index.csv"
    pd.DataFrame(per_split_index_rows).sort_values("split_name").to_csv(out_index, index=False)

    plot_df = consolidated.copy()
    for col in [hit_col, f"{hit_col}_std"]:
        if col not in plot_df.columns:
            plot_df[col] = np.nan
    out_fig = output_dir / "across_splits_regret_and_hit_with_random.png"
    plot_consolidated(plot_df, hit_col=hit_col, out_path=out_fig)

    print(f"Wrote: {to_repo_rel(out_csv)}")
    print(f"Wrote: {to_repo_rel(out_index)}")
    print(f"Wrote: {to_repo_rel(out_fig)}")
    print(f"Splits consolidated: {plot_df['split_name'].nunique()}")
    print(f"Rows consolidated: {len(plot_df)}")
    print(f"Hit column used: {hit_col}")
    if skipped:
        skipped_path = output_dir / "skipped_or_missing.txt"
        skipped_path.write_text("\n".join(skipped) + "\n", encoding="utf-8")
        print(f"Wrote: {to_repo_rel(skipped_path)}")
        print(f"Skipped entries: {len(skipped)}")
    else:
        print("Skipped entries: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
