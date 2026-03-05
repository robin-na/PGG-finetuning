#!/usr/bin/env python3
"""
Plot archetype-retrieval validation performance across benchmark splits.

Focus:
- How close retrieved embeddings are to true embeddings.
- How far retrieved embeddings are from the oracle (best bank match) via regret.
"""

from __future__ import annotations

import argparse
import re
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


def run_dir_for_split(split_root: Path, runs_root: Path, split_base_root: Path) -> Path:
    rel = split_rel_path(split_root, split_base_root)
    latest = runs_root / rel / "archetype_retrieval" / "model_runs" / "latest_run.txt"
    if not latest.exists():
        raise FileNotFoundError(f"missing latest_run.txt: {to_repo_rel(latest)}")
    run_dir = resolve_latest_run_path(latest.read_text(encoding="utf-8"))
    if not run_dir.is_dir():
        raise FileNotFoundError(f"missing run directory from latest_run.txt: {run_dir}")
    return run_dir


def discover_split_roots(
    include_default: bool,
    default_root: Path,
    ood_roots: Sequence[Path],
) -> List[Path]:
    out: List[Path] = []
    if include_default and default_root.is_dir():
        out.append(default_root)
    for ood_root in ood_roots:
        if ood_root.is_dir():
            for config_dir in sorted([p for p in ood_root.iterdir() if p.is_dir()]):
                for direction_dir in sorted([p for p in config_dir.iterdir() if p.is_dir()]):
                    out.append(direction_dir)
    return out


def split_label(root: Path, default_root: Path, ood_roots: Sequence[Path]) -> Dict[str, str]:
    if root.resolve() == default_root.resolve():
        return {"split_group": "default", "split_name": "default", "config": "default", "direction": "default"}

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
        config, direction = parts[0], parts[1]
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


def infer_hitk_column(columns: Sequence[str]) -> str | None:
    ks: List[int] = []
    for c in columns:
        m = re.fullmatch(r"oracle_hit_at_(\d+)", str(c))
        if m:
            ks.append(int(m.group(1)))
    if not ks:
        return None
    return f"oracle_hit_at_{max(ks)}"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    m = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    valid = m.notna() & (w > 0)
    if not valid.any():
        if m.notna().any():
            return float(m.mean())
        return float("nan")
    return float(np.average(m[valid], weights=w[valid]))


def aggregate_split_metrics(
    metrics_df: pd.DataFrame,
    split_meta: Dict[str, str],
    model_whitelist: set[str],
    weight_by_rows: bool,
) -> pd.DataFrame:
    ok = metrics_df.copy()
    ok["model"] = ok["model"].astype(str).str.lower()
    ok = ok[ok["status"].astype(str).str.lower() == "ok"].copy()
    ok = ok[ok["model"].isin(model_whitelist)].copy()
    if ok.empty:
        return pd.DataFrame()

    metric_cols = [
        "retrieved_true_cosine_mean",
        "oracle_true_cosine_mean",
        "retrieval_cosine_regret_mean",
        "oracle_hit_at_1",
    ]
    hitk_col = infer_hitk_column(ok.columns)
    if hitk_col is not None:
        metric_cols.append(hitk_col)
    metric_cols = [c for c in metric_cols if c in ok.columns]

    rows: List[Dict[str, object]] = []
    for model, g in ok.groupby("model", sort=True):
        rec: Dict[str, object] = {
            **split_meta,
            "model": model,
            "n_tags": int(g["tag"].nunique()) if "tag" in g.columns else int(len(g)),
            "n_rows_sum": int(pd.to_numeric(g.get("n_validation_rows"), errors="coerce").fillna(0).sum())
            if "n_validation_rows" in g.columns
            else int(len(g)),
        }
        for col in metric_cols:
            if weight_by_rows and "n_validation_rows" in g.columns:
                rec[col] = weighted_mean(g[col], g["n_validation_rows"])
            else:
                rec[col] = float(pd.to_numeric(g[col], errors="coerce").mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def ensure_order(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ordered = df.copy()
    ordered["split_sort"] = ordered["split_name"].map(lambda x: (0, x) if x == "default" else (1, x))
    ordered = ordered.sort_values(["split_sort", "model"]).drop(columns=["split_sort"]).reset_index(drop=True)
    return ordered


def model_order(models: Iterable[str]) -> List[str]:
    preferred = ["ridge", "linear", "elastic_net", "mlp", "mean", "random"]
    models_list = list(dict.fromkeys([str(m).lower() for m in models]))
    out = [m for m in preferred if m in models_list]
    out.extend([m for m in models_list if m not in out])
    return out


def plot_gap_to_oracle(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.copy()
    pivot = plot_df.pivot(index="split_name", columns="model", values="retrieval_cosine_regret_mean").sort_index()
    splits = list(pivot.index)
    models = model_order(list(pivot.columns))

    x = np.arange(len(splits), dtype=float)
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(max(12, len(splits) * 0.6), 5.5))
    for i, model in enumerate(models):
        ax.bar(
            x + (i - (len(models) - 1) / 2) * width,
            pivot[model].to_numpy(),
            width=width,
            label=model,
        )
    ax.set_xticks(x, splits, rotation=35, ha="right")
    ax.set_ylabel("Oracle Gap / Regret (lower is better)")
    ax.set_title("Benchmark Splits: Gap Between Oracle and Retrieved Similarity")
    ax.legend(frameon=False, ncol=min(6, len(models)))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_retrieved_vs_oracle(df: pd.DataFrame, out_path: Path) -> None:
    models = model_order(df["model"].unique())
    n = len(models)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(max(12, df["split_name"].nunique() * 0.6), 4.0 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        d = df[df["model"] == model].copy().sort_values("split_name")
        x = np.arange(len(d), dtype=float)
        w = 0.38
        ax.bar(x - w / 2, d["retrieved_true_cosine_mean"], width=w, label="retrieved")
        ax.bar(x + w / 2, d["oracle_true_cosine_mean"], width=w, label="oracle")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"{model}: Retrieved vs Oracle Similarity")
        ax.grid(axis="y", alpha=0.2)
        if ax is axes[0]:
            ax.legend(frameon=False, ncol=2)
        ax.set_xticks(x, d["split_name"], rotation=35, ha="right")

    axes[-1].set_xlabel("Split")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot benchmark split validation performance for archetype retrieval."
    )
    parser.add_argument(
        "--split-base-root",
        type=Path,
        default=Path("benchmark"),
        help="Base path used to map split roots into --runs-root.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("outputs/benchmark/runs"),
        help="Root containing generated split run artifacts.",
    )
    parser.add_argument(
        "--default-root",
        type=Path,
        default=Path("benchmark/data"),
    )
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
        help="Include benchmark/data in addition to OOD splits.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear", "ridge"],
        help="Model names to include.",
    )
    parser.add_argument(
        "--weight-by-rows",
        action="store_true",
        help="Use n_validation_rows as weights for split-level aggregation across tags.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/benchmark/plots/embeddings"),
    )
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
        raise FileNotFoundError("No split roots discovered.")

    model_whitelist = {str(m).strip().lower() for m in args.models if str(m).strip()}
    all_rows: List[pd.DataFrame] = []
    missing: List[str] = []

    for root in roots:
        try:
            run_dir = run_dir_for_split(root, runs_root, split_base_root)
        except FileNotFoundError as exc:
            missing.append(str(exc))
            continue
        metrics_path = run_dir / "validation_eval" / "validation_metrics_all_tags.csv"
        if not metrics_path.exists():
            missing.append(f"missing metrics: {to_repo_rel(metrics_path)}")
            continue
        metrics = pd.read_csv(metrics_path)
        split_meta = split_label(root, default_root=default_root, ood_roots=ood_roots)
        split_agg = aggregate_split_metrics(
            metrics_df=metrics,
            split_meta=split_meta,
            model_whitelist=model_whitelist,
            weight_by_rows=args.weight_by_rows,
        )
        if split_agg.empty:
            missing.append(f"no usable rows after filtering: {to_repo_rel(metrics_path)}")
            continue
        split_agg.insert(0, "split_root", to_repo_rel(root))
        split_agg.insert(1, "run_dir", to_repo_rel(run_dir))
        all_rows.append(split_agg)

    if not all_rows:
        raise RuntimeError("No split metrics available to plot.")

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df = ensure_order(all_df)

    csv_path = output_dir / "benchmark_split_model_metrics.csv"
    all_df.to_csv(csv_path, index=False)

    regret_fig = output_dir / "gap_to_oracle_by_split.png"
    sim_fig = output_dir / "retrieved_vs_oracle_by_split.png"

    plot_gap_to_oracle(all_df, regret_fig)
    plot_retrieved_vs_oracle(all_df, sim_fig)

    print(f"Wrote: {to_repo_rel(csv_path)}")
    print(f"Wrote: {to_repo_rel(regret_fig)}")
    print(f"Wrote: {to_repo_rel(sim_fig)}")

    if missing:
        miss_path = output_dir / "missing_or_skipped.txt"
        miss_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"Wrote: {to_repo_rel(miss_path)}")
        print(f"Skipped entries: {len(missing)}")
    else:
        print("Skipped entries: 0")

    print()
    cols = [
        "split_name",
        "model",
        "retrieved_true_cosine_mean",
        "oracle_true_cosine_mean",
        "retrieval_cosine_regret_mean",
    ]
    cols = [c for c in cols if c in all_df.columns]
    print(all_df[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
