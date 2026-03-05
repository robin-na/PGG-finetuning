from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from math import erf, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from io_utils import (  # noqa: E402
    ACTION_DICT_COLUMNS,
    REQUIRED_COLUMNS,
    apply_filters,
    coerce_base_columns,
    load_eval_csv,
    parse_action_columns,
    parse_bool,
    validate_required_columns,
)
from manifest import write_manifest  # noqa: E402
from metrics import (  # noqa: E402
    AGG_METRIC_COLUMNS,
    CONTRIB_REGIME_METRIC_COLUMNS,
    aggregate_contribution_by_regime,
    aggregate_scores,
    score_rows,
)

COMPARE_METRIC_MAP = {
    "contrib_mae": "contrib_abs_error",
    "target_f1": "target_f1",
    "action_exact_match": "action_exact_match",
    "target_hit_any": "target_hit_any",
}

METRIC_DIRECTION = {
    "contrib_mae": "↓ better",
    "target_f1": "↑ better",
    "action_exact_match": "↑ better",
    "target_hit_any": "↑ better",
}


@dataclass
class AnalysisArgs:
    eval_csv: Optional[str]
    run_id: Optional[str]
    eval_root: str
    compare_run_ids: Optional[str]
    compare_labels: Optional[str]
    analysis_root: str
    analysis_run_id: Optional[str]
    min_round: Optional[int]
    max_round: Optional[int]
    skip_no_actual: bool
    dpi: int
    debug_print: bool


def _timestamp_id() -> str:
    return datetime.now().strftime("%y%m%d%H%M")


def _split_csv_arg(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _resolve_run_eval_csv(run_id: str, eval_root: str) -> Path:
    return (Path(eval_root).resolve() / run_id / "micro_behavior_eval.csv").resolve()


def _resolve_input_eval_csv(eval_csv: Optional[str], run_id: Optional[str], eval_root: str) -> Path:
    if eval_csv:
        return Path(eval_csv).resolve()
    if run_id:
        return _resolve_run_eval_csv(run_id, eval_root=eval_root)
    raise ValueError("Provide --eval_csv or --run_id.")


def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _ensure_output_safety(out_dir: Path, eval_root: Optional[Path] = None) -> None:
    forbidden_roots = [
        (PROJECT_ROOT / "Micro_behavior_eval" / "output").resolve(),  # legacy location
        (PROJECT_ROOT / "outputs" / "default" / "runs" / "source_default" / "micro_behavior_eval").resolve(),
    ]
    if eval_root is not None:
        forbidden_roots.append(eval_root.resolve())

    out_resolved = out_dir.resolve()
    for forbidden_root in forbidden_roots:
        if _is_subpath(out_resolved, forbidden_root):
            raise ValueError(
                f"Unsafe analysis output path: {out_dir}. "
                f"Analysis output must not be under {forbidden_root}."
            )


def _prepare_scored(
    input_csv: Path,
    min_round: Optional[int],
    max_round: Optional[int],
    skip_no_actual: bool,
) -> Tuple[pd.DataFrame, Any, Any]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input evaluation CSV not found: {input_csv}")

    raw_df = load_eval_csv(input_csv)
    validate_required_columns(raw_df, REQUIRED_COLUMNS)
    base_df = coerce_base_columns(raw_df)
    parsed_df, parse_summary = parse_action_columns(base_df, ACTION_DICT_COLUMNS)
    filtered_df, filter_summary = apply_filters(
        parsed_df,
        min_round=min_round,
        max_round=max_round,
        skip_no_actual=skip_no_actual,
    )

    if filtered_df.empty:
        scored_df = score_rows(filtered_df.head(0).copy())
    else:
        scored_df = score_rows(filtered_df)
    return scored_df, filter_summary, parse_summary


def _normalize_game_key(value: Any) -> str:
    text = str(value).strip()
    return text if text else ""


def _resolve_existing_path(path_text: str | Path, base_dir: Optional[Path] = None) -> Optional[Path]:
    raw = str(path_text or "").strip()
    if not raw:
        return None

    candidates: List[Path] = []
    p = Path(raw)
    if p.is_absolute():
        candidates.append(p)
    else:
        if base_dir is not None:
            candidates.append((base_dir / p).resolve())
        candidates.append((PROJECT_ROOT / p).resolve())

    marker = "PGG-finetuning/"
    if marker in raw:
        tail = raw.split(marker, 1)[1]
        candidates.append((PROJECT_ROOT / tail).resolve())

    if p.is_absolute():
        stripped = str(p).lstrip("/")
        candidates.append((PROJECT_ROOT / stripped).resolve())

    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand
    return None


def _load_run_config(run_id: str, eval_root: str) -> Dict[str, Any]:
    config_path = (Path(eval_root).resolve() / run_id / "config.json").resolve()
    if not config_path.exists():
        return {}
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _extract_analysis_csv_from_config(run_cfg: Dict[str, Any], eval_root: str, run_id: str) -> Optional[Path]:
    run_dir = (Path(eval_root).resolve() / run_id).resolve()
    inputs = run_cfg.get("inputs") if isinstance(run_cfg.get("inputs"), dict) else {}
    args = run_cfg.get("args") if isinstance(run_cfg.get("args"), dict) else {}
    model = run_cfg.get("model") if isinstance(run_cfg.get("model"), dict) else {}

    candidates = [
        inputs.get("analysis_csv"),
        args.get("analysis_csv"),
        model.get("analysis_csv"),
    ]
    for raw in candidates:
        resolved = _resolve_existing_path(raw, base_dir=run_dir)
        if resolved is not None:
            return resolved
    return None


def _load_all_or_nothing_map(run_id: str, eval_root: str) -> Tuple[Dict[str, Optional[bool]], Optional[Path]]:
    run_cfg = _load_run_config(run_id, eval_root)
    if not run_cfg:
        return {}, None
    analysis_csv = _extract_analysis_csv_from_config(run_cfg, eval_root=eval_root, run_id=run_id)
    if analysis_csv is None:
        return {}, None

    try:
        cfg_df = pd.read_csv(analysis_csv, usecols=["gameId", "CONFIG_allOrNothing"])
    except Exception:
        return {}, analysis_csv

    cfg_df["gameId"] = cfg_df["gameId"].map(_normalize_game_key)
    cfg_df["CONFIG_allOrNothing"] = cfg_df["CONFIG_allOrNothing"].map(lambda x: parse_bool(x, default=None))
    cfg_df = cfg_df.drop_duplicates(subset=["gameId"], keep="first")

    mapping = {
        str(row["gameId"]): row["CONFIG_allOrNothing"]
        for _, row in cfg_df.iterrows()
        if str(row["gameId"]).strip()
    }
    return mapping, analysis_csv


def _attach_all_or_nothing(scored_df: pd.DataFrame, mapping: Dict[str, Optional[bool]]) -> pd.DataFrame:
    out = scored_df.copy()
    if "gameId" not in out.columns:
        out["CONFIG_allOrNothing"] = pd.Series([pd.NA] * len(out), dtype="boolean")
        return out
    if not mapping:
        out["CONFIG_allOrNothing"] = pd.Series([pd.NA] * len(out), dtype="boolean")
        return out
    mapped = out["gameId"].astype(str).map(mapping)
    out["CONFIG_allOrNothing"] = mapped.astype("boolean")
    return out


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _bh_adjust(p_values: List[float]) -> List[float]:
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [1.0] * m
    min_so_far = 1.0
    for rank_rev, (idx, p) in enumerate(reversed(indexed), start=1):
        rank = m - rank_rev + 1
        val = min(1.0, p * m / rank)
        min_so_far = min(min_so_far, val)
        adjusted[idx] = min_so_far
    return adjusted


def _mean_ci(series: pd.Series, z: float = 1.96) -> Dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = int(len(clean))
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "ci_half": float("nan"),
        }
    mean = float(clean.mean())
    if n <= 1:
        se = 0.0
        std = 0.0
    else:
        std = float(clean.std(ddof=1))
        se = float(std / sqrt(n))
    ci_half = float(z * se)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci_low": float(mean - ci_half),
        "ci_high": float(mean + ci_half),
        "ci_half": ci_half,
    }


def _fmt(x: Any, digits: int = 4) -> str:
    try:
        val = float(x)
    except Exception:
        return str(x)
    if val != val:
        return "NaN"
    return f"{val:.{digits}f}"


def _build_errorbar_summary(
    scored_df: pd.DataFrame,
    group_cols: List[str],
) -> pd.DataFrame:
    cols = group_cols + ["metric", "n", "mean", "std", "se", "ci_low", "ci_high", "ci_half"]
    if scored_df.empty:
        return pd.DataFrame(columns=cols)

    rows: List[Dict[str, Any]] = []
    grouped = scored_df.groupby(group_cols, dropna=False, sort=False)
    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_map = {col: val for col, val in zip(group_cols, key_tuple)}
        for metric_name, source_col in COMPARE_METRIC_MAP.items():
            stats = _mean_ci(group[source_col])
            row = dict(key_map)
            row["metric"] = metric_name
            row.update(stats)
            rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def _compute_pairwise_significance(
    scored_df: pd.DataFrame,
    run_ids: List[str],
    labels: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if scored_df.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "run_id_a",
                "run_label_a",
                "run_id_b",
                "run_label_b",
                "n_a",
                "n_b",
                "mean_a",
                "mean_b",
                "diff_a_minus_b",
                "se_diff",
                "z_score",
                "p_value",
                "p_value_bh",
                "significant_0_05",
                "significant_bh_0_05",
            ]
        )

    run_pairs = list(combinations(range(len(run_ids)), 2))
    raw_pvals: List[float] = []
    raw_idx: List[int] = []

    for metric_name, source_col in COMPARE_METRIC_MAP.items():
        for ia, ib in run_pairs:
            run_a, run_b = run_ids[ia], run_ids[ib]
            label_a, label_b = labels[ia], labels[ib]
            a = pd.to_numeric(
                scored_df.loc[scored_df["run_id"] == run_a, source_col],
                errors="coerce",
            ).dropna()
            b = pd.to_numeric(
                scored_df.loc[scored_df["run_id"] == run_b, source_col],
                errors="coerce",
            ).dropna()
            n_a = int(len(a))
            n_b = int(len(b))
            mean_a = float(a.mean()) if n_a > 0 else float("nan")
            mean_b = float(b.mean()) if n_b > 0 else float("nan")

            if n_a <= 1:
                var_a = 0.0
            else:
                var_a = float(a.var(ddof=1))
            if n_b <= 1:
                var_b = 0.0
            else:
                var_b = float(b.var(ddof=1))

            if n_a == 0 or n_b == 0:
                se_diff = float("nan")
                z_score = float("nan")
                p_value = float("nan")
            else:
                se_diff = float(sqrt(var_a / n_a + var_b / n_b))
                if se_diff == 0.0:
                    z_score = float("nan")
                    p_value = float("nan")
                else:
                    z_score = float((mean_a - mean_b) / se_diff)
                    p_value = float(2.0 * (1.0 - _normal_cdf(abs(z_score))))

            row = {
                "metric": metric_name,
                "run_id_a": run_a,
                "run_label_a": label_a,
                "run_id_b": run_b,
                "run_label_b": label_b,
                "n_a": n_a,
                "n_b": n_b,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "diff_a_minus_b": float(mean_a - mean_b) if n_a > 0 and n_b > 0 else float("nan"),
                "se_diff": se_diff,
                "z_score": z_score,
                "p_value": p_value,
            }
            rows.append(row)
            if p_value == p_value:
                raw_idx.append(len(rows) - 1)
                raw_pvals.append(p_value)

    df = pd.DataFrame(rows)
    df["p_value_bh"] = float("nan")
    df["significant_0_05"] = False
    df["significant_bh_0_05"] = False
    if not df.empty:
        df["significant_0_05"] = df["p_value"] < 0.05
        if raw_pvals:
            adj = _bh_adjust(raw_pvals)
            for idx, adj_val in zip(raw_idx, adj):
                df.at[idx, "p_value_bh"] = float(adj_val)
            df["significant_bh_0_05"] = df["p_value_bh"] < 0.05
    return df


def _write_comparison_summary_md(
    output_dir: Path,
    run_summaries: List[Dict[str, Any]],
    comparison_overall: pd.DataFrame,
    significance_df: pd.DataFrame,
    contrib_regime_df: pd.DataFrame,
) -> Path:
    lines: List[str] = []
    lines.append("# Comparison Summary")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    lines.append("| Label | Run ID | Rows (post/pre) |")
    lines.append("|---|---|---:|")
    for rec in run_summaries:
        lines.append(
            f"| {rec['run_label']} | {rec['run_id']} | {rec['post_filter_rows']}/{rec['pre_filter_rows']} |"
        )
    lines.append("")

    lines.append("## Plot Guide")
    lines.append("")
    lines.append("- `compare_contrib_mae.png`: Mean contribution MAE across runs with 95% CI (`↓ better`).")
    lines.append("- `compare_target_f1.png`: Mean typed-target F1 across runs with 95% CI (`↑ better`).")
    lines.append("- `compare_action_exact_match.png`: Mean exact action-match rate across runs with 95% CI (`↑ better`).")
    lines.append("- `compare_target_hit_any.png`: Mean target-hit-any rate across runs with 95% CI (`↑ better`).")
    lines.append("- `compare_by_round_contrib_mae.png`: Round-wise contribution MAE with 95% CI bands (`↓ better`).")
    lines.append("- `compare_by_round_target_f1.png`: Round-wise target F1 with 95% CI bands (`↑ better`).")
    lines.append(
        "- `compare_contrib_mae_by_all_or_nothing.png`: Contribution MAE split by `CONFIG_allOrNothing` (`↓ better`)."
    )
    lines.append(
        "- `compare_contrib_binary_by_all_or_nothing_true.png`: Binary contribution metrics for all-or-nothing games (`↑ better`)."
    )
    lines.append("")

    lines.append("## Standard Error and CI")
    lines.append("")
    lines.append("- For each metric within a run: `SE = s / sqrt(n)`.")
    lines.append("- `s` is sample standard deviation of row-level metric values; `n` is row count.")
    lines.append("- 95% CI is `mean ± 1.96 * SE`.")
    lines.append("")

    if not comparison_overall.empty:
        lines.append("## Overall Means")
        lines.append("")
        lines.append("| Run | contrib_mae | target_f1 | action_exact_match | target_hit_any |")
        lines.append("|---|---:|---:|---:|---:|")
        show_cols = ["run_label", "contrib_mae", "target_f1", "action_exact_match", "target_hit_any"]
        for _, row in comparison_overall[show_cols].iterrows():
            lines.append(
                f"| {row['run_label']} | {_fmt(row['contrib_mae'])} | {_fmt(row['target_f1'])} | {_fmt(row['action_exact_match'])} | {_fmt(row['target_hit_any'])} |"
            )
        lines.append("")

    if not contrib_regime_df.empty:
        lines.append("## Contribution By Game Regime")
        lines.append("")
        lines.append("| Regime | Run | n_rows | contrib_mae | contrib_mae_norm20 | contrib_binary_accuracy | contrib_binary_f1 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        show_cols = [
            "CONFIG_allOrNothing",
            "run_label",
            "n_rows",
            "contrib_mae",
            "contrib_mae_norm20",
            "contrib_binary_accuracy",
            "contrib_binary_f1",
        ]
        ordered = contrib_regime_df.copy()
        ordered["CONFIG_allOrNothing"] = ordered["CONFIG_allOrNothing"].map(
            lambda x: "all-or-nothing" if bool(x) else "continuous"
        )
        ordered["regime_order"] = ordered["CONFIG_allOrNothing"].map(
            lambda x: 0 if x == "continuous" else 1
        )
        ordered = ordered.sort_values(["regime_order", "run_label"]).drop(columns=["regime_order"])
        for _, row in ordered[show_cols].iterrows():
            lines.append(
                f"| {row['CONFIG_allOrNothing']} | {row['run_label']} | {int(row['n_rows'])} | {_fmt(row['contrib_mae'])} | {_fmt(row['contrib_mae_norm20'])} | {_fmt(row['contrib_binary_accuracy'])} | {_fmt(row['contrib_binary_f1'])} |"
            )
        lines.append("")

    lines.append("## Pairwise Significance (BH-adjusted)")
    lines.append("")
    lines.append("| Metric | Comparison | Mean A | Mean B | Diff (A-B) | p (BH) | Sig @0.05 |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    if significance_df.empty:
        lines.append("| - | - | - | - | - | - | - |")
    else:
        for _, row in significance_df.iterrows():
            cmp_name = f"{row['run_label_a']} vs {row['run_label_b']}"
            sig = "yes" if bool(row.get("significant_bh_0_05", False)) else "no"
            metric = str(row["metric"])
            direction = METRIC_DIRECTION.get(metric, "")
            lines.append(
                f"| {metric} ({direction}) | {cmp_name} | {_fmt(row['mean_a'])} | {_fmt(row['mean_b'])} | {_fmt(row['diff_a_minus_b'])} | {_fmt(row['p_value_bh'])} | {sig} |"
            )
    lines.append("")

    out_path = output_dir / "comparison_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _plot_compare_bars(
    overall_errorbars: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    run_order: List[str],
    run_color_map: Dict[str, Any],
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _annotate_better_direction(ax, direction: str | None) -> None:
        if direction not in {"up", "down"}:
            return
        x = 0.985
        y_lo = 0.15
        y_hi = 0.85
        if direction == "up":
            ax.annotate(
                "",
                xy=(x, y_hi),
                xytext=(x, y_lo),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="black"),
            )
        else:
            ax.annotate(
                "",
                xy=(x, y_lo),
                xytext=(x, y_hi),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="black"),
            )
        ax.text(
            x + 0.02,
            0.5,
            "better",
            transform=ax.transAxes,
            rotation=90,
            va="center",
            ha="left",
            fontsize=9,
        )

    generated: List[str] = []
    metrics = [
        ("contrib_mae", "Comparison: Contribution MAE", "compare_contrib_mae.png", "down"),
        ("target_f1", "Comparison: Target F1", "compare_target_f1.png", "up"),
        ("action_exact_match", "Comparison: Action Exact Match", "compare_action_exact_match.png", "up"),
        ("target_hit_any", "Comparison: Target Hit Any", "compare_target_hit_any.png", "up"),
    ]
    for metric, title, filename, direction in metrics:
        data = overall_errorbars[overall_errorbars["metric"] == metric].copy()
        if run_order:
            data["run_label"] = pd.Categorical(data["run_label"], categories=run_order, ordered=True)
            data = data.sort_values("run_label")
        data = data.dropna(subset=["mean"])
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        yerr = pd.to_numeric(data["ci_half"], errors="coerce").fillna(0.0).values
        colors = [run_color_map.get(str(lbl), "#1f77b4") for lbl in data["run_label"].astype(str).tolist()]
        ax.bar(data["run_label"], data["mean"], yerr=yerr, capsize=4, color=colors)
        ax.set_title(title)
        ax.set_ylabel(f"{metric} (mean ± 95% CI)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
        _annotate_better_direction(ax, direction)
        out_path = out_dir / filename
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        generated.append(str(out_path))
    return generated


def _plot_compare_lines(
    by_round_errorbars: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    run_order: List[str],
    run_color_map: Dict[str, Any],
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _annotate_better_direction(ax, direction: str | None) -> None:
        if direction not in {"up", "down"}:
            return
        x = 0.985
        y_lo = 0.15
        y_hi = 0.85
        if direction == "up":
            ax.annotate(
                "",
                xy=(x, y_hi),
                xytext=(x, y_lo),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="black"),
            )
        else:
            ax.annotate(
                "",
                xy=(x, y_lo),
                xytext=(x, y_hi),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="black"),
            )
        ax.text(
            x + 0.02,
            0.5,
            "better",
            transform=ax.transAxes,
            rotation=90,
            va="center",
            ha="left",
            fontsize=9,
        )

    generated: List[str] = []
    if by_round_errorbars.empty:
        return generated
    specs = [
        ("contrib_mae", "Comparison by Round: Contribution MAE", "compare_by_round_contrib_mae.png", "down"),
        ("target_f1", "Comparison by Round: Target F1", "compare_by_round_target_f1.png", "up"),
    ]
    for metric, title, filename, direction in specs:
        sub = by_round_errorbars[by_round_errorbars["metric"] == metric].copy()
        sub = sub.dropna(subset=["roundIndex", "mean"])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        if run_order:
            labels = [x for x in run_order if x in set(sub["run_label"].astype(str).tolist())]
        else:
            labels = list(sub["run_label"].astype(str).unique())
        for run_label in labels:
            g = sub[sub["run_label"].astype(str) == str(run_label)].copy()
            gg = g.sort_values("roundIndex")
            x = pd.to_numeric(gg["roundIndex"], errors="coerce")
            y = pd.to_numeric(gg["mean"], errors="coerce")
            low = pd.to_numeric(gg["ci_low"], errors="coerce")
            high = pd.to_numeric(gg["ci_high"], errors="coerce")
            color = run_color_map.get(str(run_label), "#1f77b4")
            ax.plot(x, y, marker="o", label=str(run_label), color=color)
            ax.fill_between(x, low, high, alpha=0.18, color=color)
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_ylabel(f"{metric} (mean ± 95% CI)")
        ax.grid(alpha=0.3)
        ax.legend()
        _annotate_better_direction(ax, direction)
        out_path = out_dir / filename
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        generated.append(str(out_path))
    return generated


def _plot_compare_contrib_by_regime(
    contrib_regime_df: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    run_order: List[str],
    run_color_map: Dict[str, Any],
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    generated: List[str] = []
    if contrib_regime_df.empty:
        return generated

    df = contrib_regime_df.copy()
    if run_order:
        df["run_label"] = pd.Categorical(df["run_label"], categories=run_order, ordered=True)
        df = df.sort_values(["run_label", "CONFIG_allOrNothing"])

    labels = [str(x) for x in df["run_label"].dropna().unique().tolist()]
    if labels:
        regimes = [False, True]
        x = np.arange(len(labels))
        width = 0.36
        fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(labels)), 4))
        for i, regime in enumerate(regimes):
            sub = df[df["CONFIG_allOrNothing"] == regime].copy()
            sub = sub.set_index("run_label")
            y = [float(sub.loc[lbl, "contrib_mae"]) if lbl in sub.index else float("nan") for lbl in labels]
            offset = (-width / 2.0) if i == 0 else (width / 2.0)
            color = "#7aa6c2" if regime is False else "#f08a5d"
            name = "continuous" if regime is False else "all-or-nothing"
            ax.bar(x + offset, y, width=width, label=name, color=color)
        ax.set_title("Contribution MAE by CONFIG_allOrNothing")
        ax.set_ylabel("contrib_mae (lower better)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        out_path = out_dir / "compare_contrib_mae_by_all_or_nothing.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        generated.append(str(out_path))

    sub_true = df[df["CONFIG_allOrNothing"] == True].copy()  # noqa: E712
    if not sub_true.empty:
        if run_order:
            sub_true["run_label"] = pd.Categorical(sub_true["run_label"], categories=run_order, ordered=True)
            sub_true = sub_true.sort_values("run_label")
        labels2 = [str(x) for x in sub_true["run_label"].dropna().tolist()]
        x2 = np.arange(len(labels2))
        width = 0.36
        acc = pd.to_numeric(sub_true["contrib_binary_accuracy"], errors="coerce").values
        f1 = pd.to_numeric(sub_true["contrib_binary_f1"], errors="coerce").values
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels2)), 4))
        ax.bar(x2 - width / 2, acc, width=width, label="binary accuracy", color="#5b8e7d")
        ax.bar(x2 + width / 2, f1, width=width, label="binary F1", color="#d07a90")
        ax.set_title("All-or-Nothing Contribution Metrics")
        ax.set_ylabel("score (higher better)")
        ax.set_xticks(x2)
        ax.set_xticklabels(labels2, rotation=20)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        out_path = out_dir / "compare_contrib_binary_by_all_or_nothing_true.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        generated.append(str(out_path))
    return generated


def _build_run_color_map(run_order: List[str]) -> Dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    labels = [str(x) for x in run_order]
    n = len(labels)
    if n <= 0:
        return {}

    # Use tab10 first (better separation for small N), then tab20, then sampled HSV.
    if n <= 10:
        cmap = plt.get_cmap("tab10")
        palette = [to_hex(cmap(i)) for i in range(n)]
    elif n <= 20:
        cmap = plt.get_cmap("tab20")
        palette = [to_hex(cmap(i)) for i in range(n)]
    else:
        cmap = plt.get_cmap("hsv")
        palette = [to_hex(cmap(i / n)) for i in range(n)]
    return {labels[i]: palette[i] for i in range(n)}


def _run_single_analysis(args: AnalysisArgs, output_dir: Path) -> Dict[str, Any]:
    input_csv = _resolve_input_eval_csv(args.eval_csv, args.run_id, args.eval_root)
    scored_df, filter_summary, parse_summary = _prepare_scored(
        input_csv=input_csv,
        min_round=args.min_round,
        max_round=args.max_round,
        skip_no_actual=args.skip_no_actual,
    )
    regime_analysis_csv: Optional[Path] = None
    if args.run_id:
        all_or_nothing_map, regime_analysis_csv = _load_all_or_nothing_map(args.run_id, args.eval_root)
        scored_df = _attach_all_or_nothing(scored_df, all_or_nothing_map)
    else:
        scored_df = _attach_all_or_nothing(scored_df, {})

    metrics_overall = aggregate_scores(scored_df)
    metrics_by_round = aggregate_scores(scored_df, group_cols=["roundIndex"])
    metrics_by_game = aggregate_scores(scored_df, group_cols=["gameId"])
    contrib_by_regime = aggregate_contribution_by_regime(scored_df)

    output_files: List[str] = []
    row_level_path = output_dir / "row_level_scored.csv"
    scored_df.to_csv(row_level_path, index=False)
    output_files.append(str(row_level_path))

    overall_path = output_dir / "metrics_overall.csv"
    metrics_overall.to_csv(overall_path, index=False)
    output_files.append(str(overall_path))

    round_path = output_dir / "metrics_by_round.csv"
    metrics_by_round.to_csv(round_path, index=False)
    output_files.append(str(round_path))

    game_path = output_dir / "metrics_by_game.csv"
    metrics_by_game.to_csv(game_path, index=False)
    output_files.append(str(game_path))

    regime_path = output_dir / "metrics_contribution_by_all_or_nothing.csv"
    contrib_by_regime.to_csv(regime_path, index=False)
    output_files.append(str(regime_path))

    plot_files: List[str] = []
    if not scored_df.empty:
        from plots import generate_all_plots  # local import to avoid matplotlib init on --help

        plot_files = generate_all_plots(
            scored_df=scored_df,
            metrics_by_round=metrics_by_round,
            metrics_by_game=metrics_by_game,
            output_dir=output_dir,
            dpi=int(args.dpi),
        )
        output_files.extend(plot_files)

    manifest_target = output_dir / "analysis_manifest.json"
    manifest_payload = {
        "mode": "single",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_csv),
        "output_path": str(output_dir),
        "filters": {
            "min_round": args.min_round,
            "max_round": args.max_round,
            "skip_no_actual": args.skip_no_actual,
        },
        "row_counts": {
            "pre_filter": filter_summary.pre_filter_rows,
            "post_filter": filter_summary.post_filter_rows,
            "dropped": filter_summary.dropped_rows,
        },
        "malformed_dict_parse_counts": parse_summary.malformed_counts,
        "malformed_dict_rows": parse_summary.malformed_rows,
        "all_or_nothing_analysis_csv": str(regime_analysis_csv) if regime_analysis_csv is not None else None,
        "generated_files": output_files + [str(manifest_target)],
        "args": {
            "eval_csv": args.eval_csv,
            "run_id": args.run_id,
            "eval_root": args.eval_root,
            "analysis_root": args.analysis_root,
            "analysis_run_id": output_dir.name,
            "min_round": args.min_round,
            "max_round": args.max_round,
            "skip_no_actual": args.skip_no_actual,
            "dpi": args.dpi,
            "debug_print": args.debug_print,
        },
        "metrics_columns": AGG_METRIC_COLUMNS,
        "contribution_regime_metrics_columns": CONTRIB_REGIME_METRIC_COLUMNS,
    }
    manifest_path = write_manifest(output_dir, manifest_payload)
    output_files.append(str(manifest_path))

    if args.debug_print:
        print(f"[analysis] mode: single")
        print(f"[analysis] input: {input_csv}")
        print(f"[analysis] output: {output_dir}")
        print(f"[analysis] rows pre/post: {filter_summary.pre_filter_rows}/{filter_summary.post_filter_rows}")
        print(f"[analysis] generated files: {len(output_files)}")

    return {
        "mode": "single",
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "analysis_run_id": output_dir.name,
        "generated_files": output_files,
        "plot_files": plot_files,
    }


def _run_comparison_analysis(args: AnalysisArgs, output_dir: Path) -> Dict[str, Any]:
    run_ids = _split_csv_arg(args.compare_run_ids)
    if len(run_ids) < 2:
        raise ValueError("Comparison mode requires at least two run IDs in --compare_run_ids.")

    labels = _split_csv_arg(args.compare_labels)
    if labels and len(labels) != len(run_ids):
        raise ValueError(
            f"compare_labels length ({len(labels)}) must match compare_run_ids length ({len(run_ids)})."
        )
    if not labels:
        labels = run_ids

    overall_frames: List[pd.DataFrame] = []
    by_round_frames: List[pd.DataFrame] = []
    by_game_frames: List[pd.DataFrame] = []
    scored_frames: List[pd.DataFrame] = []
    run_summaries: List[Dict[str, Any]] = []

    for run_id, run_label in zip(run_ids, labels):
        input_csv = _resolve_run_eval_csv(run_id, eval_root=args.eval_root)
        scored_df, filter_summary, parse_summary = _prepare_scored(
            input_csv=input_csv,
            min_round=args.min_round,
            max_round=args.max_round,
            skip_no_actual=args.skip_no_actual,
        )
        all_or_nothing_map, regime_analysis_csv = _load_all_or_nothing_map(run_id, args.eval_root)
        scored_df = _attach_all_or_nothing(scored_df, all_or_nothing_map)
        mapped_rows = int(scored_df["CONFIG_allOrNothing"].notna().sum()) if "CONFIG_allOrNothing" in scored_df else 0

        m_overall = aggregate_scores(scored_df)
        if m_overall.empty:
            empty_row = {k: (0 if k == "n_rows" else float("nan")) for k in AGG_METRIC_COLUMNS}
            m_overall = pd.DataFrame([empty_row], columns=AGG_METRIC_COLUMNS)

        m_overall.insert(0, "run_label", run_label)
        m_overall.insert(0, "run_id", run_id)
        m_overall["input_csv"] = str(input_csv)
        m_overall["pre_filter_rows"] = int(filter_summary.pre_filter_rows)
        m_overall["post_filter_rows"] = int(filter_summary.post_filter_rows)
        m_overall["malformed_dict_rows"] = int(parse_summary.malformed_rows)
        overall_frames.append(m_overall)

        m_round = aggregate_scores(scored_df, group_cols=["roundIndex"])
        if not m_round.empty:
            m_round.insert(0, "run_label", run_label)
            m_round.insert(0, "run_id", run_id)
            by_round_frames.append(m_round)

        m_game = aggregate_scores(scored_df, group_cols=["gameId"])
        if not m_game.empty:
            m_game.insert(0, "run_label", run_label)
            m_game.insert(0, "run_id", run_id)
            by_game_frames.append(m_game)

        scored_with_run = scored_df.copy()
        scored_with_run["run_id"] = run_id
        scored_with_run["run_label"] = run_label
        scored_frames.append(scored_with_run)

        run_summaries.append(
            {
                "run_id": run_id,
                "run_label": run_label,
                "input_csv": str(input_csv),
                "pre_filter_rows": int(filter_summary.pre_filter_rows),
                "post_filter_rows": int(filter_summary.post_filter_rows),
                "dropped_rows": int(filter_summary.dropped_rows),
                "malformed_dict_rows": int(parse_summary.malformed_rows),
                "malformed_dict_parse_counts": parse_summary.malformed_counts,
                "all_or_nothing_analysis_csv": str(regime_analysis_csv) if regime_analysis_csv is not None else None,
                "all_or_nothing_mapped_rows": mapped_rows,
            }
        )

    comparison_overall = pd.concat(overall_frames, ignore_index=True)
    comparison_by_round = pd.concat(by_round_frames, ignore_index=True) if by_round_frames else pd.DataFrame()
    comparison_by_game = pd.concat(by_game_frames, ignore_index=True) if by_game_frames else pd.DataFrame()
    combined_scored = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    contribution_by_regime = aggregate_contribution_by_regime(
        combined_scored,
        group_cols=["run_id", "run_label"],
    )
    run_order = labels
    run_color_map = _build_run_color_map(run_order)

    overall_errorbars = _build_errorbar_summary(
        combined_scored,
        group_cols=["run_id", "run_label"],
    )
    by_round_errorbars = _build_errorbar_summary(
        combined_scored,
        group_cols=["run_id", "run_label", "roundIndex"],
    )
    significance_df = _compute_pairwise_significance(combined_scored, run_ids, labels)

    output_files: List[str] = []
    overall_path = output_dir / "comparison_overall.csv"
    comparison_overall.to_csv(overall_path, index=False)
    output_files.append(str(overall_path))

    round_path = output_dir / "comparison_by_round.csv"
    comparison_by_round.to_csv(round_path, index=False)
    output_files.append(str(round_path))

    game_path = output_dir / "comparison_by_game.csv"
    comparison_by_game.to_csv(game_path, index=False)
    output_files.append(str(game_path))

    contrib_regime_path = output_dir / "comparison_contribution_by_all_or_nothing.csv"
    contribution_by_regime.to_csv(contrib_regime_path, index=False)
    output_files.append(str(contrib_regime_path))

    overall_err_path = output_dir / "comparison_overall_errorbars.csv"
    overall_errorbars.to_csv(overall_err_path, index=False)
    output_files.append(str(overall_err_path))

    round_err_path = output_dir / "comparison_by_round_errorbars.csv"
    by_round_errorbars.to_csv(round_err_path, index=False)
    output_files.append(str(round_err_path))

    signif_path = output_dir / "comparison_significance.csv"
    significance_df.to_csv(signif_path, index=False)
    output_files.append(str(signif_path))

    summary_md_path = _write_comparison_summary_md(
        output_dir=output_dir,
        run_summaries=run_summaries,
        comparison_overall=comparison_overall,
        significance_df=significance_df,
        contrib_regime_df=contribution_by_regime,
    )
    output_files.append(str(summary_md_path))

    plot_files: List[str] = []
    plot_files.extend(
        _plot_compare_bars(
            overall_errorbars,
            output_dir,
            int(args.dpi),
            run_order=run_order,
            run_color_map=run_color_map,
        )
    )
    plot_files.extend(
        _plot_compare_lines(
            by_round_errorbars,
            output_dir,
            int(args.dpi),
            run_order=run_order,
            run_color_map=run_color_map,
        )
    )
    plot_files.extend(
        _plot_compare_contrib_by_regime(
            contribution_by_regime,
            output_dir,
            int(args.dpi),
            run_order=run_order,
            run_color_map=run_color_map,
        )
    )
    output_files.extend(plot_files)

    manifest_target = output_dir / "analysis_manifest.json"
    manifest_payload = {
        "mode": "comparison",
        "timestamp": datetime.now().isoformat(),
        "output_path": str(output_dir),
        "filters": {
            "min_round": args.min_round,
            "max_round": args.max_round,
            "skip_no_actual": args.skip_no_actual,
        },
        "comparison_runs": run_summaries,
        "generated_files": output_files + [str(manifest_target)],
        "args": {
            "compare_run_ids": run_ids,
            "compare_labels": labels,
            "run_colors": run_color_map,
            "eval_root": args.eval_root,
            "analysis_root": args.analysis_root,
            "analysis_run_id": output_dir.name,
            "min_round": args.min_round,
            "max_round": args.max_round,
            "skip_no_actual": args.skip_no_actual,
            "dpi": args.dpi,
            "debug_print": args.debug_print,
        },
        "metrics_columns": AGG_METRIC_COLUMNS,
        "contribution_regime_metrics_columns": CONTRIB_REGIME_METRIC_COLUMNS,
    }
    manifest_path = write_manifest(output_dir, manifest_payload)
    output_files.append(str(manifest_path))

    if args.debug_print:
        print(f"[analysis] mode: comparison")
        print(f"[analysis] output: {output_dir}")
        print(f"[analysis] runs: {len(run_summaries)}")
        for rec in run_summaries:
            print(
                "[analysis] run",
                rec["run_id"],
                f"({rec['run_label']}):",
                f"rows {rec['post_filter_rows']}/{rec['pre_filter_rows']}, malformed={rec['malformed_dict_rows']}",
            )
        print(f"[analysis] generated files: {len(output_files)}")

    return {
        "mode": "comparison",
        "output_dir": str(output_dir),
        "analysis_run_id": output_dir.name,
        "compare_run_ids": run_ids,
        "compare_labels": labels,
        "generated_files": output_files,
        "plot_files": plot_files,
    }


def run_analysis(args: AnalysisArgs) -> Dict[str, Any]:
    analysis_run_id = args.analysis_run_id or _timestamp_id()
    output_dir = Path(args.analysis_root).resolve() / analysis_run_id
    _ensure_output_safety(output_dir, eval_root=Path(args.eval_root))
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_run_ids = _split_csv_arg(args.compare_run_ids)
    if compare_run_ids:
        if args.eval_csv or args.run_id:
            raise ValueError("Use either single-run inputs (--eval_csv/--run_id) OR --compare_run_ids, not both.")
        return _run_comparison_analysis(args, output_dir)
    return _run_single_analysis(args, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze micro behavior evaluation CSV output.")
    parser.add_argument("--eval_csv", type=str, default=None, help="Path to micro_behavior_eval.csv")
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run id under --eval_root/<run_id>/micro_behavior_eval.csv",
    )
    parser.add_argument(
        "--eval_root",
        type=str,
        default="outputs/default/runs/source_default/micro_behavior_eval",
        help="Root directory that contains micro eval run folders.",
    )
    parser.add_argument(
        "--compare_run_ids",
        type=str,
        default=None,
        help="Comma-separated run IDs to compare across experiments.",
    )
    parser.add_argument(
        "--compare_labels",
        type=str,
        default=None,
        help="Comma-separated labels aligned with --compare_run_ids.",
    )
    parser.add_argument(
        "--analysis_root",
        type=str,
        default="reports/default/micro_behavior",
        help="Root directory for analysis artifacts.",
    )
    parser.add_argument("--analysis_run_id", type=str, default=None, help="Output analysis run id (default: timestamp).")
    parser.add_argument("--min_round", type=int, default=None, help="Minimum round index to include.")
    parser.add_argument("--max_round", type=int, default=None, help="Maximum round index to include.")
    parser.add_argument(
        "--skip_no_actual",
        type=str,
        default="true",
        help="Whether to drop rows with no actual observation. true/false (default: true).",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Plot DPI.")
    parser.add_argument(
        "--debug_print",
        type=str,
        default="false",
        help="Print debug summary. true/false (default: false).",
    )
    return parser


def parse_cli_args(argv: Optional[list[str]] = None) -> AnalysisArgs:
    parser = build_parser()
    ns = parser.parse_args(argv)
    return AnalysisArgs(
        eval_csv=ns.eval_csv,
        run_id=ns.run_id,
        eval_root=ns.eval_root,
        compare_run_ids=ns.compare_run_ids,
        compare_labels=ns.compare_labels,
        analysis_root=ns.analysis_root,
        analysis_run_id=ns.analysis_run_id,
        min_round=ns.min_round,
        max_round=ns.max_round,
        skip_no_actual=bool(parse_bool(ns.skip_no_actual, default=True)),
        dpi=int(ns.dpi),
        debug_print=bool(parse_bool(ns.debug_print, default=False)),
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_cli_args(argv)
    run_analysis(args)


if __name__ == "__main__":
    main()
