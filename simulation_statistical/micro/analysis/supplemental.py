from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from Micro_behavior_eval.analysis.io_utils import parse_action_dict
from Micro_behavior_eval.analysis.run_analysis import _extract_analysis_csv_from_config, _load_run_config


def _parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return default
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return default


def _load_punishment_enabled_map(
    run_id: Optional[str],
    eval_root: str,
    analysis_csv: Optional[str],
) -> Tuple[Dict[str, bool], Optional[Path]]:
    resolved_analysis_csv: Optional[Path] = None
    if analysis_csv:
        candidate = Path(analysis_csv).resolve()
        if candidate.exists():
            resolved_analysis_csv = candidate
    if resolved_analysis_csv is None and run_id:
        run_cfg = _load_run_config(run_id, eval_root)
        resolved_analysis_csv = _extract_analysis_csv_from_config(run_cfg, eval_root=eval_root, run_id=run_id)
    if resolved_analysis_csv is None or not resolved_analysis_csv.exists():
        return {}, resolved_analysis_csv

    df = pd.read_csv(resolved_analysis_csv)
    if "gameId" not in df.columns or "CONFIG_punishmentExists" not in df.columns:
        return {}, resolved_analysis_csv
    mapping = {
        str(row["gameId"]): _parse_bool(row["CONFIG_punishmentExists"], default=False)
        for _, row in (
            df[["gameId", "CONFIG_punishmentExists"]]
            .drop_duplicates(subset=["gameId"], keep="first")
            .iterrows()
        )
    }
    return mapping, resolved_analysis_csv


def _coerce_action_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name in df.columns:
        return df[column_name].map(lambda value: parse_action_dict(value)[0])
    return pd.Series([{}] * len(df), index=df.index, dtype=object)


def _punishment_target_f1(predicted: Dict[str, int], actual: Dict[str, int]) -> float:
    pred_targets = {str(key) for key, value in predicted.items() if int(value) > 0}
    actual_targets = {str(key) for key, value in actual.items() if int(value) > 0}
    if not pred_targets and not actual_targets:
        return 1.0
    if not pred_targets or not actual_targets:
        return 0.0
    overlap = pred_targets.intersection(actual_targets)
    precision = float(len(overlap)) / float(len(pred_targets)) if pred_targets else 0.0
    recall = float(len(overlap)) / float(len(actual_targets)) if actual_targets else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _line_plot(df: pd.DataFrame, out_path: Path, dpi: int) -> Optional[str]:
    if df.empty:
        return None
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["roundIndex"], df["punishment_target_f1"], marker="o")
    ax.set_title("Punishment Target F1 by Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Punishment Target F1")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _bar_plot(df: pd.DataFrame, out_path: Path, dpi: int) -> Optional[str]:
    if df.empty:
        return None
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_w = max(8, 0.5 * len(df))
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    ax.bar(df["gameId"].astype(str), df["punishment_target_f1"])
    ax.set_title("Punishment Target F1 by Game")
    ax.set_xlabel("Game")
    ax.set_ylabel("Punishment Target F1")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _update_manifest(manifest_path: Path, generated_files: List[str], analysis_csv: Optional[Path]) -> None:
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing = list(payload.get("generated_files", []))
    payload["generated_files"] = list(dict.fromkeys(existing + generated_files))
    payload["supplemental_punishment_target_f1"] = {
        "analysis_csv": str(analysis_csv) if analysis_csv is not None else None,
        "generated_files": generated_files,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_punishment_target_report(
    *,
    output_dir: Path,
    run_id: Optional[str],
    eval_root: str,
    analysis_csv: Optional[str],
    dpi: int,
) -> Dict[str, Any]:
    row_level_path = output_dir / "row_level_scored.csv"
    if not row_level_path.exists():
        return {"generated_files": [], "analysis_csv": None}

    df = pd.read_csv(row_level_path)
    if df.empty or "gameId" not in df.columns:
        return {"generated_files": [], "analysis_csv": None}

    punishment_enabled_map, resolved_analysis_csv = _load_punishment_enabled_map(run_id, eval_root, analysis_csv)
    if punishment_enabled_map:
        df["punishment_enabled"] = df["gameId"].astype(str).map(lambda game_id: punishment_enabled_map.get(game_id, False))
        df = df[df["punishment_enabled"]].copy()
    if df.empty:
        return {"generated_files": [], "analysis_csv": str(resolved_analysis_csv) if resolved_analysis_csv else None}

    df["predicted_punished_pid_dict"] = _coerce_action_column(df, "predicted_punished_pid_dict")
    df["actual_punished_pid_dict"] = _coerce_action_column(df, "actual_punished_pid_dict")
    df["punishment_target_f1"] = [
        _punishment_target_f1(predicted, actual)
        for predicted, actual in zip(df["predicted_punished_pid_dict"], df["actual_punished_pid_dict"])
    ]

    overall = pd.DataFrame(
        [{"n_rows": int(len(df)), "punishment_target_f1": float(df["punishment_target_f1"].mean())}]
    )
    by_round = (
        df.groupby("roundIndex", as_index=False)["punishment_target_f1"]
        .mean()
        .sort_values("roundIndex")
        .reset_index(drop=True)
    )
    by_game = (
        df.groupby("gameId", as_index=False)["punishment_target_f1"]
        .mean()
        .sort_values("gameId")
        .reset_index(drop=True)
    )

    generated_files: List[str] = []
    overall_path = output_dir / "punishment_target_metrics_overall.csv"
    by_round_path = output_dir / "punishment_target_f1_by_round.csv"
    by_game_path = output_dir / "punishment_target_f1_by_game.csv"
    overall.to_csv(overall_path, index=False)
    by_round.to_csv(by_round_path, index=False)
    by_game.to_csv(by_game_path, index=False)
    generated_files.extend([str(overall_path), str(by_round_path), str(by_game_path)])

    line_plot = _line_plot(by_round, output_dir / "punishment_target_f1_by_round.png", dpi=dpi)
    bar_plot = _bar_plot(by_game, output_dir / "punishment_target_f1_by_game.png", dpi=dpi)
    if line_plot:
        generated_files.append(line_plot)
    if bar_plot:
        generated_files.append(bar_plot)

    _update_manifest(output_dir / "analysis_manifest.json", generated_files, resolved_analysis_csv)
    return {
        "generated_files": generated_files,
        "analysis_csv": str(resolved_analysis_csv) if resolved_analysis_csv is not None else None,
    }
