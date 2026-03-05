from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
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


DEFAULT_BINARY_FACTORS: Tuple[str, ...] = (
    "CONFIG_chat",
    "CONFIG_allOrNothing",
    "CONFIG_defaultContribProp",
    "CONFIG_rewardExists",
    "CONFIG_showNRounds",
    "CONFIG_showPunishmentId",
    "CONFIG_showOtherSummaries",
    "CONFIG_punishmentExists",
)

DEFAULT_MEDIAN_FACTORS: Tuple[str, ...] = (
    "CONFIG_playerCount",
    "CONFIG_numRounds",
    "CONFIG_MPCR",
)


def _timestamp_id() -> str:
    return datetime.now().strftime("%y%m%d%H%M")


def _split_csv_arg(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _resolve_run_eval_csv(run_id: str, eval_root: str) -> Path:
    return (Path(eval_root).resolve() / run_id / "macro_simulation_eval.csv").resolve()


def _resolve_input_eval_csv(eval_csv: Optional[str], run_id: Optional[str], eval_root: str) -> Path:
    if eval_csv:
        return Path(eval_csv).resolve()
    if run_id:
        return _resolve_run_eval_csv(run_id, eval_root=eval_root)
    raise ValueError("Provide --eval_csv or --run_id.")


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


def _extract_analysis_csv_from_config(
    run_cfg: Dict[str, Any], eval_root: str, run_id: str
) -> Optional[Path]:
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


def parse_dict_field(value: object) -> Dict[str, float]:
    if isinstance(value, dict):
        out: Dict[str, float] = {}
        for k, v in value.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    text = str(value).strip()
    if text in {"", "{}", "None", "nan", "null"}:
        return {}
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in parsed.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def parse_binary_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if np.issubdtype(series.dtype, np.number):
        return series.fillna(0).astype(float) != 0.0
    true_values = {"1", "true", "t", "yes", "y"}
    return (
        series.astype(str).str.strip().str.lower().map(lambda x: x in true_values).fillna(False)
    )


def signed(value: float, eps: float = 1e-12) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _coalesce_contribution(df: pd.DataFrame) -> pd.Series:
    if "data.contribution_clamped" in df.columns:
        clamped = pd.to_numeric(df["data.contribution_clamped"], errors="coerce")
        raw = pd.to_numeric(df.get("data.contribution"), errors="coerce")
        return clamped.fillna(raw).fillna(0.0)
    return pd.to_numeric(df.get("data.contribution"), errors="coerce").fillna(0.0)


def load_human_game_table(
    analysis_csv: Path,
    binary_factors: Sequence[str],
    median_factors: Sequence[str],
) -> pd.DataFrame:
    df = pd.read_csv(analysis_csv)
    required = {"gameId", "itt_relative_efficiency"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Human analysis CSV missing required columns: {missing}")

    keep = [
        "gameId",
        "itt_relative_efficiency",
        "CONFIG_endowment",
        "CONFIG_multiplier",
        "CONFIG_numRounds",
        "CONFIG_playerCount",
        "num_actual_players",
        "CONFIG_punishmentCost",
        "CONFIG_punishmentMagnitude",
        "CONFIG_rewardCost",
        "CONFIG_rewardMagnitude",
    ]
    keep += [f for f in binary_factors if f in df.columns]
    keep += [f for f in median_factors if f in df.columns]
    keep = list(dict.fromkeys(keep))

    out = df[keep].copy()
    out["gameId"] = out["gameId"].astype(str)
    out = out.drop_duplicates(subset=["gameId"], keep="first")
    out = out.rename(columns={"itt_relative_efficiency": "human_normalized_efficiency"})
    return out


def compute_sim_game_metrics(rows_csv: Path, human_cfg: pd.DataFrame) -> pd.DataFrame:
    rows = pd.read_csv(rows_csv)
    required = {"gameId", "roundIndex", "playerId", "data.punished", "data.rewarded"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{rows_csv} missing required columns: {missing}")

    rows = rows.copy()
    rows["gameId"] = rows["gameId"].astype(str)
    rows["playerId"] = rows["playerId"].astype(str)
    rows["roundIndex"] = pd.to_numeric(rows["roundIndex"], errors="coerce").fillna(0).astype(int)
    rows["contribution"] = _coalesce_contribution(rows)
    rows["punished_dict"] = rows["data.punished"].map(parse_dict_field)
    rows["rewarded_dict"] = rows["data.rewarded"].map(parse_dict_field)

    incoming_pun: Dict[Tuple[str, int, str], float] = {}
    incoming_rew: Dict[Tuple[str, int, str], float] = {}
    for row in rows.itertuples(index=False):
        game_id = str(getattr(row, "gameId"))
        round_index = int(getattr(row, "roundIndex"))
        punished_dict = getattr(row, "punished_dict")
        rewarded_dict = getattr(row, "rewarded_dict")
        for target, units in punished_dict.items():
            key = (game_id, round_index, str(target))
            incoming_pun[key] = incoming_pun.get(key, 0.0) + float(units)
        for target, units in rewarded_dict.items():
            key = (game_id, round_index, str(target))
            incoming_rew[key] = incoming_rew.get(key, 0.0) + float(units)

    merged = rows.merge(human_cfg, on="gameId", how="left")
    if merged["CONFIG_endowment"].isna().any():
        missing_games = sorted(set(merged.loc[merged["CONFIG_endowment"].isna(), "gameId"]))
        raise ValueError(
            "Missing config rows for simulated games in human analysis CSV: "
            + ", ".join(missing_games[:5])
            + (" ..." if len(missing_games) > 5 else "")
        )

    for col in [
        "CONFIG_punishmentCost",
        "CONFIG_punishmentMagnitude",
        "CONFIG_rewardCost",
        "CONFIG_rewardMagnitude",
    ]:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    punishment_exists = (
        parse_binary_series(merged["CONFIG_punishmentExists"])
        if "CONFIG_punishmentExists" in merged.columns
        else pd.Series(False, index=merged.index)
    )
    reward_exists = (
        parse_binary_series(merged["CONFIG_rewardExists"])
        if "CONFIG_rewardExists" in merged.columns
        else pd.Series(False, index=merged.index)
    )

    merged["punished_units_given"] = merged["punished_dict"].map(lambda d: sum(d.values()))
    merged["rewarded_units_given"] = merged["rewarded_dict"].map(lambda d: sum(d.values()))

    pun_recv: List[float] = []
    rew_recv: List[float] = []
    for row in merged.itertuples(index=False):
        key = (str(getattr(row, "gameId")), int(getattr(row, "roundIndex")), str(getattr(row, "playerId")))
        pun_recv.append(float(incoming_pun.get(key, 0.0)))
        rew_recv.append(float(incoming_rew.get(key, 0.0)))
    merged["punished_units_received"] = pun_recv
    merged["rewarded_units_received"] = rew_recv

    endowment = pd.to_numeric(merged["CONFIG_endowment"], errors="coerce").fillna(0.0)
    multiplier = pd.to_numeric(merged["CONFIG_multiplier"], errors="coerce").fillna(0.0)
    contribution = pd.to_numeric(merged["contribution"], errors="coerce").fillna(0.0)

    merged["base_payoff"] = (contribution * multiplier) + (endowment - contribution)
    merged["sim_row_payoff"] = merged["base_payoff"]
    merged.loc[punishment_exists, "sim_row_payoff"] = (
        merged.loc[punishment_exists, "sim_row_payoff"]
        - merged.loc[punishment_exists, "punished_units_given"]
        * merged.loc[punishment_exists, "CONFIG_punishmentCost"]
        - merged.loc[punishment_exists, "punished_units_received"]
        * merged.loc[punishment_exists, "CONFIG_punishmentMagnitude"]
    )
    merged.loc[reward_exists, "sim_row_payoff"] = (
        merged.loc[reward_exists, "sim_row_payoff"]
        - merged.loc[reward_exists, "rewarded_units_given"]
        * merged.loc[reward_exists, "CONFIG_rewardCost"]
        + merged.loc[reward_exists, "rewarded_units_received"]
        * merged.loc[reward_exists, "CONFIG_rewardMagnitude"]
    )

    merged["sim_contribution_rate"] = np.where(endowment != 0, contribution / endowment, 0.0)
    merged["sim_punishment_flag"] = np.where(
        punishment_exists, merged["punished_dict"].map(lambda d: int(bool(d))), 0
    )
    merged["sim_reward_flag"] = np.where(
        reward_exists, merged["rewarded_dict"].map(lambda d: int(bool(d))), 0
    )

    agg = (
        merged.groupby(["gameId"], as_index=False)
        .agg(
            sim_mean_contribution_rate=("sim_contribution_rate", "mean"),
            sim_punishment_rate=("sim_punishment_flag", "mean"),
            sim_reward_rate=("sim_reward_flag", "mean"),
            sim_total_payoff=("sim_row_payoff", "sum"),
            sim_num_players=("playerId", "nunique"),
            sim_num_rounds=("roundIndex", "nunique"),
        )
        .copy()
    )

    denom_cols = [
        "gameId",
        "CONFIG_endowment",
        "CONFIG_numRounds",
        "CONFIG_playerCount",
        "CONFIG_multiplier",
        "num_actual_players",
    ]
    denom = human_cfg[denom_cols].drop_duplicates(subset=["gameId"], keep="first")
    agg = agg.merge(denom, on="gameId", how="left")

    ref_players = np.where(
        pd.to_numeric(agg["num_actual_players"], errors="coerce").fillna(0) > 0,
        pd.to_numeric(agg["num_actual_players"], errors="coerce").fillna(0),
        np.where(
            pd.to_numeric(agg["CONFIG_playerCount"], errors="coerce").fillna(0) > 0,
            pd.to_numeric(agg["CONFIG_playerCount"], errors="coerce").fillna(0),
            pd.to_numeric(agg["sim_num_players"], errors="coerce").fillna(0),
        ),
    )
    ref_rounds = np.where(
        pd.to_numeric(agg["CONFIG_numRounds"], errors="coerce").fillna(0) > 0,
        pd.to_numeric(agg["CONFIG_numRounds"], errors="coerce").fillna(0),
        pd.to_numeric(agg["sim_num_rounds"], errors="coerce").fillna(0),
    )
    endowment = pd.to_numeric(agg["CONFIG_endowment"], errors="coerce").fillna(0.0)
    multiplier = pd.to_numeric(agg["CONFIG_multiplier"], errors="coerce").fillna(0.0)

    p_full_coop = endowment * ref_rounds * ref_players * multiplier
    p_full_defect = endowment * ref_rounds * ref_players
    denom = p_full_coop - p_full_defect
    agg["sim_normalized_efficiency"] = np.where(
        denom != 0, (agg["sim_total_payoff"] - p_full_defect) / denom, 0.0
    )
    return agg[
        [
            "gameId",
            "sim_mean_contribution_rate",
            "sim_punishment_rate",
            "sim_reward_rate",
            "sim_total_payoff",
            "sim_num_players",
            "sim_num_rounds",
            "sim_normalized_efficiency",
        ]
    ].copy()


def build_directional_rows(
    merged: pd.DataFrame,
    run_id: str,
    label: str,
    binary_factors: Sequence[str],
    median_factors: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for factor in list(binary_factors) + list(median_factors):
        if factor not in merged.columns:
            continue
        sub = merged[[factor, "human_normalized_efficiency", "sim_normalized_efficiency"]].dropna()
        if sub.empty:
            continue

        mode = "binary" if factor in set(binary_factors) else "median"
        if mode == "binary":
            mask_high = parse_binary_series(sub[factor])
            high = sub[mask_high]
            low = sub[~mask_high]
            threshold = None
        else:
            factor_numeric = pd.to_numeric(sub[factor], errors="coerce")
            med = float(factor_numeric.median())
            high = sub[factor_numeric > med]
            low = sub[factor_numeric < med]
            threshold = med

        if high.empty or low.empty:
            continue

        human_delta = float(
            high["human_normalized_efficiency"].mean() - low["human_normalized_efficiency"].mean()
        )
        sim_delta = float(
            high["sim_normalized_efficiency"].mean() - low["sim_normalized_efficiency"].mean()
        )
        human_sign = signed(human_delta)
        sim_sign = signed(sim_delta)

        rows.append(
            {
                "run_id": run_id,
                "label": label,
                "factor": factor,
                "mode": mode,
                "threshold": threshold,
                "n_total": int(len(sub)),
                "n_high": int(len(high)),
                "n_low": int(len(low)),
                "human_delta": human_delta,
                "sim_delta": sim_delta,
                "human_sign": human_sign,
                "sim_sign": sim_sign,
                "sign_match": bool(human_sign == sim_sign),
                "sign_match_nonzero_human": bool(human_sign != 0 and human_sign == sim_sign),
            }
        )
    return pd.DataFrame(rows)


def maybe_plot_directional_effects(
    directional: pd.DataFrame, plot_dir: Path, dpi: int = 160
) -> Optional[List[Path]]:
    if directional.empty:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    plot_paths: List[Path] = []
    plot_dir.mkdir(parents=True, exist_ok=True)
    for (run_id, label), group in directional.groupby(["run_id", "label"], sort=False):
        if group.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, max(3.0, 0.45 * len(group))))
        x = np.arange(len(group))
        width = 0.38
        ax.barh(x - width / 2, group["human_delta"], height=width, label="human")
        ax.barh(x + width / 2, group["sim_delta"], height=width, label="simulated")
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_yticks(x)
        ax.set_yticklabels(group["factor"])
        ax.set_title(f"{label}: normalized-efficiency directional deltas")
        ax.set_xlabel("high-or-True mean minus low-or-False mean")
        ax.legend()
        fig.tight_layout()
        safe_run_id = run_id.replace("/", "__")
        out_path = plot_dir / f"directional_effects_{safe_run_id}.png"
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        plot_paths.append(out_path)
    return plot_paths or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze macro_simulation_eval outputs: normalized-efficiency fit and "
            "directional CONFIG effects (human vs simulation)."
        )
    )
    parser.add_argument("--eval_csv", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument(
        "--eval_root",
        type=str,
        default="outputs/default/runs/source_default/macro_simulation_eval",
    )
    parser.add_argument(
        "--compare_run_ids",
        type=str,
        default=None,
        help="Comma-separated run IDs under eval_root (for example variant/run_id).",
    )
    parser.add_argument(
        "--compare_labels",
        type=str,
        default=None,
        help="Optional comma-separated labels aligned to --compare_run_ids.",
    )
    parser.add_argument(
        "--analysis_root",
        type=str,
        default="reports/default/macro_simulation_eval",
    )
    parser.add_argument("--analysis_run_id", type=str, default=None)
    parser.add_argument(
        "--human_analysis_csv",
        type=str,
        default=None,
        help="If set, use this df_analysis CSV for all runs. Otherwise infer from each run config.",
    )
    parser.add_argument(
        "--binary_factors",
        type=str,
        default=",".join(DEFAULT_BINARY_FACTORS),
    )
    parser.add_argument(
        "--median_factors",
        type=str,
        default=",".join(DEFAULT_MEDIAN_FACTORS),
    )
    parser.add_argument("--shared_games_only", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    compare_run_ids = _split_csv_arg(args.compare_run_ids)
    compare_labels = _split_csv_arg(args.compare_labels)
    if compare_run_ids:
        run_ids = compare_run_ids
        if compare_labels and len(compare_labels) != len(compare_run_ids):
            raise ValueError("If provided, --compare_labels must match --compare_run_ids length.")
        labels = compare_labels if compare_labels else compare_run_ids
        if len(run_ids) < 1:
            raise ValueError("No run IDs provided.")
    else:
        run_id = args.run_id
        if not run_id and not args.eval_csv:
            raise ValueError("Provide --eval_csv or --run_id (or --compare_run_ids).")
        if args.eval_csv and run_id:
            run_ids = [run_id]
        elif run_id:
            run_ids = [run_id]
        else:
            run_ids = ["adhoc_eval_csv"]
        labels = [run_ids[0]]

    binary_factors = _split_csv_arg(args.binary_factors)
    median_factors = _split_csv_arg(args.median_factors)

    analysis_root = Path(args.analysis_root).resolve()
    if args.analysis_run_id:
        analysis_run_id = args.analysis_run_id
    else:
        analysis_run_id = _timestamp_id()
    out_dir = analysis_root / analysis_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run_records: List[Dict[str, Any]] = []
    game_tables: Dict[str, pd.DataFrame] = {}
    human_csv_by_run: Dict[str, Path] = {}

    for idx, run_id in enumerate(run_ids):
        if args.eval_csv and len(run_ids) == 1 and run_id == "adhoc_eval_csv":
            rows_csv = Path(args.eval_csv).resolve()
            run_cfg = {}
        else:
            rows_csv = _resolve_run_eval_csv(run_id, eval_root=args.eval_root)
            run_cfg = _load_run_config(run_id, eval_root=args.eval_root)

        if not rows_csv.exists():
            raise FileNotFoundError(f"Macro eval CSV not found: {rows_csv}")

        if args.human_analysis_csv:
            human_csv = Path(args.human_analysis_csv).resolve()
        else:
            human_csv = _extract_analysis_csv_from_config(run_cfg, eval_root=args.eval_root, run_id=run_id)
            if human_csv is None:
                raise ValueError(
                    f"Could not infer human analysis CSV for run '{run_id}'. "
                    "Pass --human_analysis_csv explicitly."
                )
        if not human_csv.exists():
            raise FileNotFoundError(f"Human analysis CSV not found: {human_csv}")

        human_csv_by_run[run_id] = human_csv
        human = load_human_game_table(human_csv, binary_factors=binary_factors, median_factors=median_factors)
        sim = compute_sim_game_metrics(rows_csv, human_cfg=human)
        merged = human.merge(sim, on="gameId", how="inner")
        merged["run_id"] = run_id
        merged["label"] = labels[idx]
        game_tables[run_id] = merged

        run_records.append(
            {
                "run_id": run_id,
                "label": labels[idx],
                "eval_csv": str(rows_csv),
                "human_analysis_csv": str(human_csv),
                "n_games_simulated": int(sim["gameId"].nunique()),
                "n_games_after_merge": int(len(merged)),
            }
        )

    shared_games: Optional[set[str]] = None
    if args.shared_games_only and len(game_tables) > 1:
        for df in game_tables.values():
            game_set = set(df["gameId"].astype(str).tolist())
            if shared_games is None:
                shared_games = game_set
            else:
                shared_games &= game_set
        shared_games = shared_games or set()

    game_level_frames: List[pd.DataFrame] = []
    aggregate_rows: List[Dict[str, Any]] = []
    directional_frames: List[pd.DataFrame] = []

    for run_id, label in zip(run_ids, labels):
        merged = game_tables[run_id].copy()
        if shared_games is not None:
            merged = merged[merged["gameId"].isin(shared_games)].copy()
        game_level_frames.append(merged)

        abs_err = (merged["sim_normalized_efficiency"] - merged["human_normalized_efficiency"]).abs()
        sq_err = (merged["sim_normalized_efficiency"] - merged["human_normalized_efficiency"]) ** 2
        aggregate_rows.append(
            {
                "run_id": run_id,
                "label": label,
                "n_games": int(len(merged)),
                "mae_normalized_efficiency": float(abs_err.mean()) if len(abs_err) else np.nan,
                "rmse_normalized_efficiency": float(np.sqrt(sq_err.mean())) if len(sq_err) else np.nan,
                "corr_normalized_efficiency": float(
                    merged["sim_normalized_efficiency"].corr(merged["human_normalized_efficiency"])
                )
                if len(merged) >= 2
                else np.nan,
                "mean_human_normalized_efficiency": float(merged["human_normalized_efficiency"].mean())
                if len(merged)
                else np.nan,
                "mean_sim_normalized_efficiency": float(merged["sim_normalized_efficiency"].mean())
                if len(merged)
                else np.nan,
            }
        )

        directional = build_directional_rows(
            merged=merged,
            run_id=run_id,
            label=label,
            binary_factors=binary_factors,
            median_factors=median_factors,
        )
        directional_frames.append(directional)

    selected_df = pd.DataFrame(run_records)
    aggregate_df = pd.DataFrame(aggregate_rows)
    game_level_df = pd.concat(game_level_frames, ignore_index=True) if game_level_frames else pd.DataFrame()
    directional_df = (
        pd.concat(directional_frames, ignore_index=True) if directional_frames else pd.DataFrame()
    )

    directional_summary_rows: List[Dict[str, Any]] = []
    if not directional_df.empty:
        for (run_id, label), group in directional_df.groupby(["run_id", "label"], sort=False):
            directional_summary_rows.append(
                {
                    "run_id": run_id,
                    "label": label,
                    "n_factors_evaluated": int(len(group)),
                    "sign_match_rate_all": float(group["sign_match"].mean()),
                    "sign_match_rate_nonzero_human": float(
                        group.loc[group["human_sign"] != 0, "sign_match"].mean()
                    )
                    if (group["human_sign"] != 0).any()
                    else np.nan,
                }
            )
    directional_summary_df = pd.DataFrame(directional_summary_rows)

    selected_df.to_csv(out_dir / "selected_runs.csv", index=False)
    aggregate_df.to_csv(out_dir / "aggregate_efficiency_metrics.csv", index=False)
    game_level_df.to_csv(out_dir / "game_level_metrics.csv", index=False)
    directional_df.to_csv(out_dir / "directional_effects.csv", index=False)
    directional_summary_df.to_csv(out_dir / "directional_sign_summary.csv", index=False)

    plot_paths = None
    if not args.no_plots:
        plot_paths = maybe_plot_directional_effects(
            directional=directional_df, plot_dir=(out_dir / "figures"), dpi=args.dpi
        )

    manifest = {
        "analysis_run_id": analysis_run_id,
        "analysis_root": str(analysis_root),
        "eval_root": str(Path(args.eval_root).resolve()),
        "run_ids": run_ids,
        "labels": labels,
        "shared_games_only": bool(args.shared_games_only),
        "shared_games_count": int(len(shared_games)) if shared_games is not None else None,
        "human_csv_by_run": {k: str(v) for k, v in human_csv_by_run.items()},
        "binary_factors": binary_factors,
        "median_factors": median_factors,
        "plots": [str(p) for p in plot_paths] if plot_paths else [],
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote macro analysis -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

