#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_COLORS: Dict[str, str] = {
    "human": "#222222",
    "no archetype": "#1d4ed8",
    "no archetype 12B": "#1d4ed8",
    "oracle archetype": "#d97706",
    "oracle archetype 12B": "#d97706",
    "oracle archetype 22B mag1": "#7c3aed",
    "oracle archetype 27B unit-edge": "#7c3aed",
    "random archetype": "#0f766e",
    "retrieved archetype": "#b91c1c",
}

GAME_LEVEL_METRICS: Tuple[Tuple[str, str], ...] = (
    ("contribution_rate", "Contribution Rate"),
    ("punishment_rate", "Punishment Rate"),
    ("reward_rate", "Reward Rate"),
    ("normalized_efficiency", "Normalized Efficiency"),
)

TRAJECTORY_METRICS: Tuple[Tuple[str, str], ...] = (
    ("contribution_rate", "Contribution Rate"),
    ("punishment_rate", "Punishment Rate"),
    ("reward_rate", "Reward Rate"),
    ("round_normalized_efficiency", "Round Normalized Efficiency"),
)

TARGETING_METRICS: Tuple[Tuple[str, str], ...] = (
    ("punishment_received_flag", "P(Receive Punishment)"),
    ("reward_received_flag", "P(Receive Reward)"),
)

SUMMARY_CORRELATION_METRICS: Tuple[Tuple[str, str], ...] = (
    ("contribution_rate", "Contribution Rate"),
    ("normalized_efficiency", "Normalized Efficiency"),
)

PLAYER_RATE_METRICS: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("contribution_rate", "Contribution Rate", None),
    ("punishment_rate", "Punishment Rate", "CONFIG_punishmentExists"),
    ("reward_rate", "Reward Rate", "CONFIG_rewardExists"),
)

DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_BOOTSTRAP_SEED = 0


def split_csv_arg(value: str | None) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_label_float_map(value: str | None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in split_csv_arg(value):
        if ":" not in item:
            raise ValueError(
                "Expected label:value pairs for magnitude overrides, for example "
                "'oracle archetype 22B:1'."
            )
        label, raw_value = item.split(":", 1)
        out[label.strip()] = float(raw_value.strip())
    return out


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


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    if np.issubdtype(series.dtype, np.number):
        return series.fillna(0).astype(float) != 0.0
    true_values = {"1", "true", "t", "yes", "y"}
    return (
        series.astype(str).str.strip().str.lower().map(lambda x: x in true_values).fillna(False)
    )


def safe_corr(x: Sequence[float] | pd.Series, y: Sequence[float] | pd.Series) -> float:
    sx = pd.Series(x, dtype=float)
    sy = pd.Series(y, dtype=float)
    mask = sx.notna() & sy.notna()
    sx = sx.loc[mask]
    sy = sy.loc[mask]
    if len(sx) < 2:
        return np.nan
    if float(sx.std(ddof=0)) == 0.0 or float(sy.std(ddof=0)) == 0.0:
        return np.nan
    return float(sx.corr(sy))


def bootstrap_ci(
    values: Sequence[float] | pd.Series,
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    arr = pd.Series(values, dtype=float).dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    point = float(stat_fn(arr))
    if len(arr) == 1 or n_boot <= 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    stats = np.empty(int(n_boot), dtype=float)
    for idx in range(int(n_boot)):
        sample = arr[rng.integers(0, len(arr), size=len(arr))]
        stats[idx] = float(stat_fn(sample))
    return point, float(np.nanquantile(stats, 0.025)), float(np.nanquantile(stats, 0.975))


def bootstrap_corr_ci(
    x: Sequence[float] | pd.Series,
    y: Sequence[float] | pd.Series,
    n_boot: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    frame = pd.DataFrame({"x": pd.Series(x, dtype=float), "y": pd.Series(y, dtype=float)}).dropna()
    if len(frame) < 2:
        return np.nan, np.nan, np.nan
    point = safe_corr(frame["x"], frame["y"])
    if len(frame) == 2 or n_boot <= 1 or not math.isfinite(float(point)):
        return point, point, point
    rng = np.random.default_rng(seed)
    stats: List[float] = []
    for _ in range(int(n_boot)):
        take = rng.integers(0, len(frame), size=len(frame))
        sample = frame.iloc[take]
        corr = safe_corr(sample["x"], sample["y"])
        if math.isfinite(float(corr)):
            stats.append(float(corr))
    if not stats:
        return point, np.nan, np.nan
    return point, float(np.nanquantile(stats, 0.025)), float(np.nanquantile(stats, 0.975))


def wasserstein_1d(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    xs = np.sort(pd.Series(x, dtype=float).dropna().to_numpy(dtype=float))
    ys = np.sort(pd.Series(y, dtype=float).dropna().to_numpy(dtype=float))
    if len(xs) == 0 or len(ys) == 0:
        return np.nan
    values = np.concatenate([xs, ys])
    values.sort()
    if len(values) <= 1:
        return 0.0
    deltas = np.diff(values)
    if len(deltas) == 0:
        return 0.0
    x_cdf = np.searchsorted(xs, values[:-1], side="right") / float(len(xs))
    y_cdf = np.searchsorted(ys, values[:-1], side="right") / float(len(ys))
    return float(np.sum(np.abs(x_cdf - y_cdf) * deltas))


def series_std(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan
    return float(np.std(arr, ddof=0))


def to_unit_edge_dict(mapping: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in mapping.items():
        try:
            units = float(value)
        except Exception:
            continue
        if units > 0.0:
            out[str(key)] = 1.0
    return out


def _resolve_existing_path(path_text: str | Path | None, base_dir: Optional[Path] = None) -> Optional[Path]:
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


def _load_run_config(run_id: str, eval_root: Path) -> Dict[str, Any]:
    config_path = (eval_root / run_id / "config.json").resolve()
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_analysis_csv_from_config(run_cfg: Dict[str, Any], eval_root: Path, run_id: str) -> Optional[Path]:
    run_dir = (eval_root / run_id).resolve()
    inputs = run_cfg.get("inputs") if isinstance(run_cfg.get("inputs"), dict) else {}
    args = run_cfg.get("args") if isinstance(run_cfg.get("args"), dict) else {}
    model = run_cfg.get("model") if isinstance(run_cfg.get("model"), dict) else {}
    for raw in (inputs.get("analysis_csv"), args.get("analysis_csv"), model.get("analysis_csv")):
        resolved = _resolve_existing_path(raw, base_dir=run_dir)
        if resolved is not None:
            return resolved
    return None


def _infer_rounds_csv_from_analysis(analysis_csv: Path) -> Optional[Path]:
    if analysis_csv.name.startswith("df_analysis_"):
        suffix = analysis_csv.name[len("df_analysis_") :]
        candidate = analysis_csv.with_name(f"df_rounds_{suffix}")
        if candidate.exists():
            return candidate
    return None


def _infer_split_csv(csv_path: Path, target_split: str) -> Optional[Path]:
    name = csv_path.name
    for prefix in ("df_analysis_", "df_rounds_"):
        if name.startswith(prefix):
            remainder = name[len(prefix) :]
            current_split, dot, extension = remainder.partition(".csv")
            if dot and current_split:
                candidate = csv_path.with_name(f"{prefix}{target_split}.csv")
                if candidate.exists():
                    return candidate
    return None


def _coalesce_contribution(df: pd.DataFrame) -> pd.Series:
    if "data.contribution_clamped" in df.columns:
        clamped = pd.to_numeric(df["data.contribution_clamped"], errors="coerce")
        raw = pd.to_numeric(df.get("data.contribution"), errors="coerce")
        return clamped.fillna(raw).fillna(0.0)
    return pd.to_numeric(df.get("data.contribution"), errors="coerce").fillna(0.0)


def _calc_ci(grouped: pd.core.groupby.DataFrameGroupBy, value_col: str) -> pd.DataFrame:
    out = grouped[value_col].agg(["count", "mean", "std"]).reset_index()
    out = out.rename(columns={"count": "n", "std": "std"})
    out["std"] = out["std"].fillna(0.0)
    out["se"] = np.where(out["n"] > 0, out["std"] / np.sqrt(out["n"]), np.nan)
    out["ci_half"] = 1.96 * out["se"]
    out["ci_low"] = out["mean"] - out["ci_half"]
    out["ci_high"] = out["mean"] + out["ci_half"]
    return out


def load_game_config(analysis_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(analysis_csv).drop_duplicates(subset=["gameId"], keep="first")
    keep = [
        "gameId",
        "CONFIG_treatmentName",
        "itt_relative_efficiency",
        "CONFIG_endowment",
        "CONFIG_multiplier",
        "CONFIG_numRounds",
        "CONFIG_playerCount",
        "num_actual_players",
        "CONFIG_punishmentExists",
        "CONFIG_punishmentCost",
        "CONFIG_punishmentMagnitude",
        "CONFIG_rewardExists",
        "CONFIG_rewardCost",
        "CONFIG_rewardMagnitude",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out["gameId"] = out["gameId"].astype(str)
    if "itt_relative_efficiency" in out.columns:
        out = out.rename(columns={"itt_relative_efficiency": "human_normalized_efficiency"})
    else:
        out["human_normalized_efficiency"] = np.nan
    for col, default in (
        ("CONFIG_endowment", 20.0),
        ("CONFIG_multiplier", 1.0),
        ("CONFIG_numRounds", np.nan),
        ("CONFIG_playerCount", np.nan),
        ("CONFIG_treatmentName", ""),
        ("num_actual_players", np.nan),
        ("CONFIG_punishmentCost", 0.0),
        ("CONFIG_punishmentMagnitude", 0.0),
        ("CONFIG_rewardCost", 0.0),
        ("CONFIG_rewardMagnitude", 0.0),
    ):
        if col not in out.columns:
            out[col] = default
    for col in ("CONFIG_punishmentExists", "CONFIG_rewardExists"):
        if col not in out.columns:
            out[col] = False
    return out


def _finalize_behavior_rows(df: pd.DataFrame, source: str, config_df: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(config_df, on="gameId", how="left")
    missing = merged["CONFIG_endowment"].isna()
    if missing.any():
        missing_games = sorted(set(merged.loc[missing, "gameId"].astype(str)))
        raise ValueError(
            "Missing config rows for games: "
            + ", ".join(missing_games[:5])
            + (" ..." if len(missing_games) > 5 else "")
        )

    endowment = pd.to_numeric(merged["CONFIG_endowment"], errors="coerce").fillna(20.0)
    merged["contribution_rate"] = np.where(endowment != 0, merged["contribution"] / endowment, 0.0)
    merged["punishment_given_units"] = merged["punished_out_dict"].map(lambda d: float(sum(d.values())))
    merged["reward_given_units"] = merged["rewarded_out_dict"].map(lambda d: float(sum(d.values())))
    merged["punishment_received_units"] = merged["punished_in_dict"].map(lambda d: float(sum(d.values())))
    merged["reward_received_units"] = merged["rewarded_in_dict"].map(lambda d: float(sum(d.values())))
    merged["punishment_given_flag"] = merged["punished_out_dict"].map(lambda d: int(bool(d)))
    merged["reward_given_flag"] = merged["rewarded_out_dict"].map(lambda d: int(bool(d)))
    merged["punishment_received_flag"] = merged["punished_in_dict"].map(lambda d: int(bool(d)))
    merged["reward_received_flag"] = merged["rewarded_in_dict"].map(lambda d: int(bool(d)))
    merged["source"] = source

    merged = merged.sort_values(["gameId", "roundIndex", "playerId"], kind="stable").reset_index(drop=True)
    merged["contrib_percentile"] = (
        merged.groupby(["gameId", "roundIndex"], sort=False)["contribution_rate"]
        .rank(pct=True, method="average")
        .astype(float)
    )

    by_player = merged.groupby(["gameId", "playerId"], sort=False)
    merged["next_contrib_rate"] = by_player["contribution_rate"].shift(-1)
    merged["next_delta_contrib"] = merged["next_contrib_rate"] - merged["contribution_rate"]
    merged["prev_contrib_percentile"] = by_player["contrib_percentile"].shift(1)

    cols = [
        "source",
        "gameId",
        "roundIndex",
        "playerId",
        "contribution",
        "contribution_rate",
        "round_payoff",
        "punishment_given_units",
        "reward_given_units",
        "punishment_received_units",
        "reward_received_units",
        "punishment_given_flag",
        "reward_given_flag",
        "punishment_received_flag",
        "reward_received_flag",
        "contrib_percentile",
        "prev_contrib_percentile",
        "next_contrib_rate",
        "next_delta_contrib",
        "human_normalized_efficiency",
        "CONFIG_treatmentName",
        "CONFIG_endowment",
        "CONFIG_multiplier",
        "CONFIG_numRounds",
        "CONFIG_playerCount",
        "num_actual_players",
        "CONFIG_punishmentExists",
        "CONFIG_punishmentCost",
        "CONFIG_punishmentMagnitude",
        "CONFIG_rewardExists",
        "CONFIG_rewardCost",
        "CONFIG_rewardMagnitude",
    ]
    return merged[cols].copy()


def load_human_rows(rounds_csv: Path, config_df: pd.DataFrame, game_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    rows = pd.read_csv(rounds_csv)
    rows["gameId"] = rows["gameId"].astype(str)
    if game_ids is not None:
        rows = rows[rows["gameId"].isin(set(map(str, game_ids)))].copy()

    if "player_removed" in rows.columns:
        removed = parse_bool_series(rows["player_removed"])
        rows = rows.loc[~removed].copy()

    if "round_index" in rows.columns:
        round_index = pd.to_numeric(rows["round_index"], errors="coerce").fillna(0).astype(int)
        rows["roundIndex"] = round_index + (1 if int(round_index.min()) == 0 else 0)
    elif "roundIndex" in rows.columns:
        rows["roundIndex"] = pd.to_numeric(rows["roundIndex"], errors="coerce").fillna(0).astype(int)
    else:
        order = (
            rows[["gameId", "roundId", "createdAt"]]
            .drop_duplicates(subset=["gameId", "roundId"], keep="first")
            .sort_values(["gameId", "createdAt", "roundId"], kind="stable")
        )
        order["roundIndex"] = order.groupby("gameId").cumcount() + 1
        rows = rows.merge(order[["gameId", "roundId", "roundIndex"]], on=["gameId", "roundId"], how="left")
        rows["roundIndex"] = pd.to_numeric(rows["roundIndex"], errors="coerce").fillna(0).astype(int)

    rows["playerId"] = rows["playerId"].astype(str)
    rows["contribution"] = pd.to_numeric(rows.get("data.contribution"), errors="coerce").fillna(0.0)
    rows["round_payoff"] = pd.to_numeric(rows.get("data.roundPayoff"), errors="coerce").fillna(0.0)
    rows["punished_out_dict"] = rows.get("data.punished", pd.Series(index=rows.index, dtype=object)).map(parse_dict_field)
    rows["rewarded_out_dict"] = rows.get("data.rewarded", pd.Series(index=rows.index, dtype=object)).map(parse_dict_field)
    rows["punished_in_dict"] = rows.get("data.punishedBy", pd.Series(index=rows.index, dtype=object)).map(parse_dict_field)
    rows["rewarded_in_dict"] = rows.get("data.rewardedBy", pd.Series(index=rows.index, dtype=object)).map(parse_dict_field)
    return _finalize_behavior_rows(rows, source="human", config_df=config_df)


def load_sim_rows(
    eval_csv: Path,
    config_df: pd.DataFrame,
    source: str,
    game_ids: Optional[Iterable[str]] = None,
    magnitude_override: Optional[float] = None,
    unit_edge_actions: bool = False,
) -> pd.DataFrame:
    rows = pd.read_csv(eval_csv)
    rows["gameId"] = rows["gameId"].astype(str)
    if game_ids is not None:
        rows = rows[rows["gameId"].isin(set(map(str, game_ids)))].copy()
    rows["playerId"] = rows["playerId"].astype(str)
    rows["playerAvatar"] = rows.get("playerAvatar", pd.Series(index=rows.index, dtype=object)).astype(str)
    rows["roundIndex"] = pd.to_numeric(rows["roundIndex"], errors="coerce").fillna(0).astype(int)
    rows["contribution"] = _coalesce_contribution(rows)
    rows["punished_out_dict"] = rows.get("data.punished", pd.Series(index=rows.index, dtype=object)).map(parse_dict_field)
    rows["rewarded_out_dict"] = rows.get("data.rewarded", pd.Series(index=rows.index, dtype=object)).map(parse_dict_field)
    if unit_edge_actions:
        rows["punished_out_dict"] = rows["punished_out_dict"].map(to_unit_edge_dict)
        rows["rewarded_out_dict"] = rows["rewarded_out_dict"].map(to_unit_edge_dict)

    valid_player_ids_by_game = (
        rows.groupby("gameId", sort=False)["playerId"].apply(lambda s: set(s.astype(str))).to_dict()
    )
    avatar_to_pid_by_game: Dict[str, Dict[str, str]] = {}
    avatar_pairs = rows[["gameId", "playerAvatar", "playerId"]].drop_duplicates()
    for game_id, avatar, player_id in avatar_pairs.itertuples(index=False, name=None):
        avatar_to_pid_by_game.setdefault(str(game_id), {})[str(avatar)] = str(player_id)

    def resolve_target(game_id: str, target: str) -> str:
        text = str(target)
        if text in valid_player_ids_by_game.get(game_id, set()):
            return text
        return avatar_to_pid_by_game.get(game_id, {}).get(text, text)

    incoming_pun: Dict[Tuple[str, int, str], Dict[str, float]] = {}
    incoming_rew: Dict[Tuple[str, int, str], Dict[str, float]] = {}
    for row in rows.itertuples(index=False):
        game_id = str(getattr(row, "gameId"))
        round_index = int(getattr(row, "roundIndex"))
        source_player = str(getattr(row, "playerId"))
        for target, units in getattr(row, "punished_out_dict").items():
            key = (game_id, round_index, resolve_target(game_id, str(target)))
            bucket = incoming_pun.setdefault(key, {})
            bucket[source_player] = bucket.get(source_player, 0.0) + float(units)
        for target, units in getattr(row, "rewarded_out_dict").items():
            key = (game_id, round_index, resolve_target(game_id, str(target)))
            bucket = incoming_rew.setdefault(key, {})
            bucket[source_player] = bucket.get(source_player, 0.0) + float(units)

    punished_in: List[Dict[str, float]] = []
    rewarded_in: List[Dict[str, float]] = []
    for row in rows.itertuples(index=False):
        key = (str(getattr(row, "gameId")), int(getattr(row, "roundIndex")), str(getattr(row, "playerId")))
        punished_in.append(incoming_pun.get(key, {}).copy())
        rewarded_in.append(incoming_rew.get(key, {}).copy())
    rows["punished_in_dict"] = punished_in
    rows["rewarded_in_dict"] = rewarded_in

    effective_config = config_df.copy()
    if magnitude_override is not None:
        effective_config["CONFIG_punishmentMagnitude"] = float(magnitude_override)
        effective_config["CONFIG_rewardMagnitude"] = float(magnitude_override)
    if unit_edge_actions:
        effective_config["CONFIG_punishmentMagnitude"] = 1.0
        effective_config["CONFIG_rewardMagnitude"] = 1.0

    merged = rows.merge(effective_config, on="gameId", how="left")
    punishment_exists = parse_bool_series(merged["CONFIG_punishmentExists"])
    reward_exists = parse_bool_series(merged["CONFIG_rewardExists"])
    endowment = pd.to_numeric(merged["CONFIG_endowment"], errors="coerce").fillna(20.0)
    multiplier = pd.to_numeric(merged["CONFIG_multiplier"], errors="coerce").fillna(1.0)
    contribution = pd.to_numeric(merged["contribution"], errors="coerce").fillna(0.0)
    base_payoff = (contribution * multiplier) + (endowment - contribution)
    merged["round_payoff"] = base_payoff
    merged.loc[punishment_exists, "round_payoff"] = (
        merged.loc[punishment_exists, "round_payoff"]
        - merged.loc[punishment_exists, "punished_out_dict"].map(lambda d: float(sum(d.values())))
        * pd.to_numeric(merged.loc[punishment_exists, "CONFIG_punishmentCost"], errors="coerce").fillna(0.0)
        - merged.loc[punishment_exists, "punished_in_dict"].map(lambda d: float(sum(d.values())))
        * pd.to_numeric(merged.loc[punishment_exists, "CONFIG_punishmentMagnitude"], errors="coerce").fillna(0.0)
    )
    merged.loc[reward_exists, "round_payoff"] = (
        merged.loc[reward_exists, "round_payoff"]
        - merged.loc[reward_exists, "rewarded_out_dict"].map(lambda d: float(sum(d.values())))
        * pd.to_numeric(merged.loc[reward_exists, "CONFIG_rewardCost"], errors="coerce").fillna(0.0)
        + merged.loc[reward_exists, "rewarded_in_dict"].map(lambda d: float(sum(d.values())))
        * pd.to_numeric(merged.loc[reward_exists, "CONFIG_rewardMagnitude"], errors="coerce").fillna(0.0)
    )

    rows["round_payoff"] = merged["round_payoff"].values
    return _finalize_behavior_rows(rows, source=source, config_df=effective_config)


def compute_game_level_metrics(rows: pd.DataFrame, config_df: pd.DataFrame, use_human_efficiency: bool) -> pd.DataFrame:
    agg = (
        rows.groupby("gameId", as_index=False)
        .agg(
            contribution_rate=("contribution_rate", "mean"),
            punishment_rate=("punishment_given_flag", "mean"),
            reward_rate=("reward_given_flag", "mean"),
            total_payoff=("round_payoff", "sum"),
            sim_num_players=("playerId", "nunique"),
            sim_num_rounds=("roundIndex", "nunique"),
        )
        .copy()
    )
    agg = agg.merge(config_df, on="gameId", how="left")
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
    endowment = pd.to_numeric(agg["CONFIG_endowment"], errors="coerce").fillna(20.0)
    multiplier = pd.to_numeric(agg["CONFIG_multiplier"], errors="coerce").fillna(1.0)
    p_full_coop = endowment * ref_rounds * ref_players * multiplier
    p_full_defect = endowment * ref_rounds * ref_players
    denom = p_full_coop - p_full_defect
    agg["normalized_efficiency_model"] = np.where(
        denom != 0, (agg["total_payoff"] - p_full_defect) / denom, 0.0
    )
    agg["normalized_efficiency"] = (
        agg["human_normalized_efficiency"]
        if use_human_efficiency
        else agg["normalized_efficiency_model"]
    )
    return agg[
        [
            "gameId",
            "CONFIG_treatmentName",
            "CONFIG_punishmentExists",
            "CONFIG_rewardExists",
            "human_normalized_efficiency",
            "contribution_rate",
            "punishment_rate",
            "reward_rate",
            "normalized_efficiency",
            "normalized_efficiency_model",
        ]
    ].copy()


def compute_linear_config_baseline_summary(
    learn_analysis_csv: Path,
    learn_rounds_csv: Path,
    val_analysis_csv: Path,
    val_rounds_csv: Path,
    eval_game_ids: Sequence[str],
) -> pd.DataFrame:
    learn_analysis = pd.read_csv(learn_analysis_csv).drop_duplicates(subset=["gameId"], keep="first")
    val_analysis = pd.read_csv(val_analysis_csv).drop_duplicates(subset=["gameId"], keep="first")
    learn_analysis["gameId"] = learn_analysis["gameId"].astype(str)
    val_analysis["gameId"] = val_analysis["gameId"].astype(str)
    eval_ids = set(map(str, eval_game_ids))
    val_analysis = val_analysis[val_analysis["gameId"].isin(eval_ids)].copy()
    if learn_analysis.empty or val_analysis.empty:
        return pd.DataFrame()

    learn_cfg = load_game_config(learn_analysis_csv)
    val_cfg = load_game_config(val_analysis_csv)
    learn_rows = load_human_rows(learn_rounds_csv, config_df=learn_cfg, game_ids=learn_analysis["gameId"].tolist())
    val_rows = load_human_rows(val_rounds_csv, config_df=val_cfg, game_ids=val_analysis["gameId"].tolist())
    learn_game = compute_game_level_metrics(learn_rows, config_df=learn_cfg, use_human_efficiency=True)
    val_game = compute_game_level_metrics(val_rows, config_df=val_cfg, use_human_efficiency=True)

    config_cols = [c for c in learn_analysis.columns if c.startswith("CONFIG_") and c in val_analysis.columns]
    if not config_cols:
        return pd.DataFrame()

    metrics = [metric for metric, _ in SUMMARY_CORRELATION_METRICS]
    rows: List[Dict[str, object]] = []
    for metric in metrics:
        train_df = learn_analysis.merge(learn_game[["gameId", metric]], on="gameId", how="inner")
        test_df = val_analysis.merge(val_game[["gameId", metric]], on="gameId", how="inner")
        x_train = train_df[config_cols].copy()
        y_train = pd.to_numeric(train_df[metric], errors="coerce")
        x_test = test_df[config_cols].copy()
        y_test = pd.to_numeric(test_df[metric], errors="coerce")

        train_mask = y_train.notna()
        test_mask = y_test.notna()
        x_train = x_train.loc[train_mask].copy()
        y_train = y_train.loc[train_mask].copy()
        x_test = x_test.loc[test_mask].copy()
        y_test = y_test.loc[test_mask].copy()
        if x_train.empty or x_test.empty:
            continue

        num_cols = [c for c in x_train.columns if pd.api.types.is_numeric_dtype(x_train[c])]
        cat_cols = [c for c in x_train.columns if c not in num_cols]
        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat_cols,
                ),
            ],
            remainder="drop",
        )
        model = Pipeline([("pre", pre), ("lr", LinearRegression())])
        model.fit(x_train, y_train)
        pred = pd.Series(model.predict(x_test), dtype=float)
        rows.append(
            {
                "source": "OLS config baseline",
                "metric": metric,
                "n_games": int(len(pred)),
                "rmse": float(np.sqrt(np.mean((pred - y_test.reset_index(drop=True)) ** 2))),
                "corr": safe_corr(y_test.reset_index(drop=True), pred),
            }
        )
    return pd.DataFrame(rows)


def build_parity_table(game_tables: Dict[str, pd.DataFrame], run_labels: Sequence[str]) -> pd.DataFrame:
    human = game_tables["human"][
        [
            "gameId",
            "CONFIG_treatmentName",
            "CONFIG_punishmentExists",
            "CONFIG_rewardExists",
            "contribution_rate",
            "punishment_rate",
            "reward_rate",
            "normalized_efficiency",
        ]
    ].rename(
        columns={
            "contribution_rate": "human_contribution_rate",
            "punishment_rate": "human_punishment_rate",
            "reward_rate": "human_reward_rate",
            "normalized_efficiency": "human_normalized_efficiency",
        }
    )

    frames: List[pd.DataFrame] = []
    for label in run_labels:
        sim = game_tables[label][
            [
                "gameId",
                "contribution_rate",
                "punishment_rate",
                "reward_rate",
                "normalized_efficiency",
            ]
        ].rename(
            columns={
                "contribution_rate": "sim_contribution_rate",
                "punishment_rate": "sim_punishment_rate",
                "reward_rate": "sim_reward_rate",
                "normalized_efficiency": "sim_normalized_efficiency",
            }
        )
        joined = human.merge(sim, on="gameId", how="inner")
        joined["source"] = label
        frames.append(joined)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_parity(parity_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for source, sub in parity_df.groupby("source", sort=False):
        for metric, _ in GAME_LEVEL_METRICS:
            metric_sub = filter_metric_games(sub, metric)
            human_col = f"human_{metric}"
            sim_col = f"sim_{metric}"
            x = pd.to_numeric(metric_sub[human_col], errors="coerce")
            y = pd.to_numeric(metric_sub[sim_col], errors="coerce")
            mask = x.notna() & y.notna()
            x = x.loc[mask]
            y = y.loc[mask]
            rows.append(
                {
                    "source": source,
                    "metric": metric,
                    "n_games": int(len(x)),
                    "mae": float(np.abs(y - x).mean()) if len(x) else np.nan,
                    "rmse": float(np.sqrt(np.mean((y - x) ** 2))) if len(x) else np.nan,
                    "bias": float((y - x).mean()) if len(x) else np.nan,
                    "corr": safe_corr(x, y),
                }
            )
    return pd.DataFrame(rows)


def filter_metric_games(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric == "punishment_rate" and "CONFIG_punishmentExists" in df.columns:
        return df.loc[parse_bool_series(df["CONFIG_punishmentExists"])].copy()
    if metric == "reward_rate" and "CONFIG_rewardExists" in df.columns:
        return df.loc[parse_bool_series(df["CONFIG_rewardExists"])].copy()
    return df.copy()


def short_source_label(source: str) -> str:
    text = str(source).strip().lower()
    mapping = {
        "no archetype": "No",
        "no archetype 12b": "No12",
        "oracle archetype": "Oracle",
        "oracle archetype 12b": "Oracle12",
        "oracle archetype 22b mag1": "Oracle22m1",
        "oracle archetype 27b unit-edge": "Oracle27u",
        "random archetype": "Random",
        "retrieved archetype": "Retrieved",
        "human": "Human",
    }
    return mapping.get(text, str(source))


def summarize_series_alignment_vs_human(
    df: pd.DataFrame,
    x_col: str,
    source_order: Sequence[str],
    value_col: str = "mean",
) -> Dict[str, Dict[str, float]]:
    human = df[df["source"] == "human"][[x_col, value_col]].rename(columns={value_col: "human_value"})
    out: Dict[str, Dict[str, float]] = {}
    for source in source_order:
        if source == "human":
            continue
        sim = df[df["source"] == source][[x_col, value_col]].rename(columns={value_col: "sim_value"})
        joined = human.merge(sim, on=x_col, how="inner")
        if joined.empty:
            continue
        x = pd.to_numeric(joined["human_value"], errors="coerce")
        y = pd.to_numeric(joined["sim_value"], errors="coerce")
        mask = x.notna() & y.notna()
        x = x.loc[mask]
        y = y.loc[mask]
        if x.empty:
            continue
        out[source] = {
            "rmse": float(np.sqrt(np.mean((y - x) ** 2))),
            "bias": float((y - x).mean()),
            "corr": safe_corr(x, y) if len(x) >= 3 else np.nan,
        }
    return out


def align_sorted_values(values: np.ndarray, num_points: int) -> np.ndarray:
    vals = np.sort(np.asarray(values, dtype=float))
    if num_points <= 0:
        return np.array([], dtype=float)
    if len(vals) == 0:
        return np.full(num_points, np.nan, dtype=float)
    if len(vals) == 1:
        return np.full(num_points, float(vals[0]), dtype=float)
    src_x = np.linspace(0.0, 1.0, len(vals))
    dst_x = np.linspace(0.0, 1.0, num_points)
    return np.interp(dst_x, src_x, vals)


def compute_distribution_alignment(human_vals: np.ndarray, sim_vals: np.ndarray) -> Dict[str, float]:
    num_points = max(len(human_vals), len(sim_vals))
    if num_points == 0:
        return {"rmse": np.nan, "corr": np.nan, "w1": np.nan}
    x = align_sorted_values(human_vals, num_points)
    y = align_sorted_values(sim_vals, num_points)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return {"rmse": np.nan, "corr": np.nan, "w1": np.nan}
    return {
        "rmse": float(np.sqrt(np.mean((y - x) ** 2))),
        "corr": safe_corr(x, y) if len(x) >= 3 else np.nan,
        "w1": wasserstein_1d(human_vals, sim_vals),
    }


def build_game_round_table(rows: pd.DataFrame, num_bins: int) -> pd.DataFrame:
    agg = (
        rows.groupby(["source", "gameId", "roundIndex"], as_index=False)
        .agg(
            CONFIG_treatmentName=("CONFIG_treatmentName", "first"),
            contribution_rate=("contribution_rate", "mean"),
            punishment_rate=("punishment_given_flag", "mean"),
            reward_rate=("reward_given_flag", "mean"),
            round_total_payoff=("round_payoff", "sum"),
            round_player_count=("playerId", "nunique"),
            CONFIG_endowment=("CONFIG_endowment", "first"),
            CONFIG_multiplier=("CONFIG_multiplier", "first"),
            CONFIG_numRounds=("CONFIG_numRounds", "first"),
            CONFIG_playerCount=("CONFIG_playerCount", "first"),
            num_actual_players=("num_actual_players", "first"),
        )
        .copy()
    )
    ref_players = np.where(
        pd.to_numeric(agg["num_actual_players"], errors="coerce").fillna(0) > 0,
        pd.to_numeric(agg["num_actual_players"], errors="coerce").fillna(0),
        np.where(
            pd.to_numeric(agg["CONFIG_playerCount"], errors="coerce").fillna(0) > 0,
            pd.to_numeric(agg["CONFIG_playerCount"], errors="coerce").fillna(0),
            pd.to_numeric(agg["round_player_count"], errors="coerce").fillna(0),
        ),
    )
    endowment = pd.to_numeric(agg["CONFIG_endowment"], errors="coerce").fillna(20.0)
    multiplier = pd.to_numeric(agg["CONFIG_multiplier"], errors="coerce").fillna(1.0)
    p_full_coop_round = endowment * ref_players * multiplier
    p_full_defect_round = endowment * ref_players
    denom = p_full_coop_round - p_full_defect_round
    agg["round_normalized_efficiency"] = np.where(
        denom != 0, (agg["round_total_payoff"] - p_full_defect_round) / denom, 0.0
    )

    num_rounds_ref = pd.to_numeric(agg["CONFIG_numRounds"], errors="coerce").fillna(0)
    fallback_rounds = agg.groupby(["source", "gameId"], sort=False)["roundIndex"].transform("max")
    num_rounds_ref = np.where(num_rounds_ref > 0, num_rounds_ref, fallback_rounds)
    agg["round_progress"] = np.where(
        num_rounds_ref > 1,
        (pd.to_numeric(agg["roundIndex"], errors="coerce").fillna(1) - 1.0) / (num_rounds_ref - 1.0),
        0.0,
    )
    agg["round_progress"] = agg["round_progress"].clip(lower=0.0, upper=1.0)
    agg["progress_bin"] = np.minimum((agg["round_progress"] * num_bins).astype(int), num_bins - 1)
    agg["progress_mid"] = (agg["progress_bin"].astype(float) + 0.5) / float(num_bins)
    return agg


def summarize_trajectories(game_round_df: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for metric, _ in TRAJECTORY_METRICS:
        summary = _calc_ci(game_round_df.groupby(["source", "progress_bin", "progress_mid"], sort=False), metric)
        summary["metric"] = metric
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_event_study(rows: pd.DataFrame) -> pd.DataFrame:
    event_rows = rows.dropna(subset=["next_delta_contrib"]).copy()
    frames: List[pd.DataFrame] = []
    for event_type, flag_col, enabled_col in (
        ("received_punishment", "punishment_received_flag", "CONFIG_punishmentExists"),
        ("received_reward", "reward_received_flag", "CONFIG_rewardExists"),
    ):
        enabled_mask = parse_bool_series(event_rows[enabled_col])
        sub = event_rows.loc[enabled_mask].copy()
        if sub.empty:
            continue
        summary = _calc_ci(
            sub.groupby(["source", flag_col], sort=False), "next_delta_contrib"
        ).rename(columns={flag_col: "event_flag"})
        summary["event_type"] = event_type
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_targeting(rows: pd.DataFrame, num_bins: int) -> pd.DataFrame:
    target_rows = rows[rows["prev_contrib_percentile"].notna()].copy()
    if target_rows.empty:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for metric, _, enabled_col in (
        ("punishment_received_flag", "P(Receive Punishment)", "CONFIG_punishmentExists"),
        ("reward_received_flag", "P(Receive Reward)", "CONFIG_rewardExists"),
    ):
        enabled_mask = parse_bool_series(target_rows[enabled_col])
        sub = target_rows.loc[enabled_mask].copy()
        if sub.empty:
            continue
        clipped = sub["prev_contrib_percentile"].astype(float).clip(lower=0.0, upper=1.0)
        sub["prev_contrib_bin"] = np.minimum((clipped * num_bins).astype(int), num_bins - 1)
        sub["prev_contrib_mid"] = (sub["prev_contrib_bin"].astype(float) + 0.5) / float(num_bins)
        summary = _calc_ci(
            sub.groupby(["source", "prev_contrib_bin", "prev_contrib_mid"], sort=False),
            metric,
        )
        summary["metric"] = metric
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_player_game_distribution_table(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["source", "gameId", "playerId"], as_index=False)
        .agg(
            CONFIG_treatmentName=("CONFIG_treatmentName", "first"),
            CONFIG_punishmentExists=("CONFIG_punishmentExists", "first"),
            CONFIG_rewardExists=("CONFIG_rewardExists", "first"),
            contribution_rate=("contribution_rate", "mean"),
            punishment_rate=("punishment_given_flag", "mean"),
            reward_rate=("reward_given_flag", "mean"),
        )
        .copy()
    )


def build_player_heterogeneity_table(player_game_df: pd.DataFrame) -> pd.DataFrame:
    return (
        player_game_df.groupby(["source", "gameId"], as_index=False)
        .agg(
            CONFIG_treatmentName=("CONFIG_treatmentName", "first"),
            CONFIG_punishmentExists=("CONFIG_punishmentExists", "first"),
            CONFIG_rewardExists=("CONFIG_rewardExists", "first"),
            contribution_player_sd=("contribution_rate", series_std),
            punishment_player_sd=("punishment_rate", series_std),
            reward_player_sd=("reward_rate", series_std),
        )
        .copy()
    )


def summarize_player_heterogeneity(
    heterogeneity_df: pd.DataFrame,
    source_order: Sequence[str],
    n_boot: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for metric_idx, (metric, title, enabled_col) in enumerate(PLAYER_RATE_METRICS):
        metric_col = f"{metric.replace('_rate', '')}_player_sd"
        sub = heterogeneity_df.copy()
        if enabled_col is not None:
            sub = sub.loc[parse_bool_series(sub[enabled_col])].copy()
        for source_idx, source in enumerate(source_order):
            vals = pd.to_numeric(
                sub.loc[sub["source"] == source, metric_col],
                errors="coerce",
            ).dropna()
            point, ci_low, ci_high = bootstrap_ci(
                vals,
                n_boot=n_boot,
                seed=seed + (metric_idx * 100) + source_idx,
            )
            rows.append(
                {
                    "source": source,
                    "metric": metric,
                    "metric_title": title,
                    "n_games": int(len(vals)),
                    "mean_player_sd": point,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def build_distribution_alignment_by_game(
    player_game_df: pd.DataFrame,
    run_labels: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    human = player_game_df[player_game_df["source"] == "human"].copy()
    game_meta = human[
        ["gameId", "CONFIG_treatmentName", "CONFIG_punishmentExists", "CONFIG_rewardExists"]
    ].drop_duplicates(subset=["gameId"], keep="first")
    for metric, title, enabled_col in PLAYER_RATE_METRICS:
        enabled_meta = game_meta.copy()
        if enabled_col is not None:
            enabled_meta = enabled_meta.loc[parse_bool_series(enabled_meta[enabled_col])].copy()
        for source in run_labels:
            sim = player_game_df[player_game_df["source"] == source].copy()
            for row in enabled_meta.itertuples(index=False):
                game_id = str(row.gameId)
                human_vals = pd.to_numeric(
                    human.loc[human["gameId"] == game_id, metric],
                    errors="coerce",
                ).dropna().to_numpy(dtype=float)
                sim_vals = pd.to_numeric(
                    sim.loc[sim["gameId"] == game_id, metric],
                    errors="coerce",
                ).dropna().to_numpy(dtype=float)
                stats = compute_distribution_alignment(human_vals, sim_vals)
                rows.append(
                    {
                        "source": source,
                        "gameId": game_id,
                        "CONFIG_treatmentName": str(row.CONFIG_treatmentName or ""),
                        "metric": metric,
                        "metric_title": title,
                        "rmse_sorted": float(stats["rmse"]),
                        "corr_sorted": float(stats["corr"]),
                        "wasserstein_1": float(stats["w1"]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_distribution_alignment(
    dist_df: pd.DataFrame,
    run_labels: Sequence[str],
    n_boot: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for metric_idx, (metric, title, _) in enumerate(PLAYER_RATE_METRICS):
        metric_df = dist_df[dist_df["metric"] == metric].copy()
        for source_idx, source in enumerate(run_labels):
            vals = pd.to_numeric(
                metric_df.loc[metric_df["source"] == source, "wasserstein_1"],
                errors="coerce",
            ).dropna()
            point, ci_low, ci_high = bootstrap_ci(
                vals,
                n_boot=n_boot,
                seed=seed + (metric_idx * 100) + source_idx,
            )
            rows.append(
                {
                    "source": source,
                    "metric": metric,
                    "metric_title": title,
                    "n_games": int(len(vals)),
                    "mean_wasserstein_1": point,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def summarize_game_level_correlations(
    parity_df: pd.DataFrame,
    run_labels: Sequence[str],
    n_boot: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for metric_idx, (metric, title) in enumerate(SUMMARY_CORRELATION_METRICS):
        metric_df = filter_metric_games(parity_df, metric)
        human_col = f"human_{metric}"
        sim_col = f"sim_{metric}"
        for source_idx, source in enumerate(run_labels):
            sub = metric_df.loc[metric_df["source"] == source, ["gameId", human_col, sim_col]].copy()
            point, ci_low, ci_high = bootstrap_corr_ci(
                pd.to_numeric(sub[human_col], errors="coerce"),
                pd.to_numeric(sub[sim_col], errors="coerce"),
                n_boot=n_boot,
                seed=seed + (metric_idx * 100) + source_idx,
            )
            rows.append(
                {
                    "source": source,
                    "metric": metric,
                    "metric_title": title,
                    "n_games": int(sub["gameId"].nunique()),
                    "corr": point,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def summarize_contribution_shift(game_round_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (source, game_id), sub in game_round_df.groupby(["source", "gameId"], sort=False):
        early = sub.loc[sub["round_progress"] <= (1.0 / 3.0), "contribution_rate"]
        late = sub.loc[sub["round_progress"] >= (2.0 / 3.0), "contribution_rate"]
        if early.empty or late.empty:
            continue
        rows.append(
            {
                "source": source,
                "gameId": game_id,
                "CONFIG_treatmentName": sub["CONFIG_treatmentName"].dropna().astype(str).iloc[0]
                if "CONFIG_treatmentName" in sub.columns and not sub["CONFIG_treatmentName"].dropna().empty
                else "",
                "early_contribution_rate": float(early.mean()),
                "late_contribution_rate": float(late.mean()),
                "contribution_shift": float(late.mean() - early.mean()),
            }
        )
    return pd.DataFrame(rows)


def build_contribution_shift_parity(shift_df: pd.DataFrame, run_labels: Sequence[str]) -> pd.DataFrame:
    human = shift_df[shift_df["source"] == "human"][
        ["gameId", "CONFIG_treatmentName", "contribution_shift"]
    ].rename(columns={"contribution_shift": "human_contribution_shift"})
    frames: List[pd.DataFrame] = []
    for label in run_labels:
        sim = shift_df[shift_df["source"] == label][["gameId", "contribution_shift"]].rename(
            columns={"contribution_shift": "sim_contribution_shift"}
        )
        joined = human.merge(sim, on="gameId", how="inner")
        joined["source"] = label
        frames.append(joined)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_contribution_shift_parity(shift_parity_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for source, sub in shift_parity_df.groupby("source", sort=False):
        x = pd.to_numeric(sub["human_contribution_shift"], errors="coerce")
        y = pd.to_numeric(sub["sim_contribution_shift"], errors="coerce")
        mask = x.notna() & y.notna()
        x = x.loc[mask]
        y = y.loc[mask]
        rows.append(
            {
                "source": source,
                "metric": "contribution_shift",
                "n_games": int(len(x)),
                "mae": float(np.abs(y - x).mean()) if len(x) else np.nan,
                "rmse": float(np.sqrt(np.mean((y - x) ** 2))) if len(x) else np.nan,
                "bias": float((y - x).mean()) if len(x) else np.nan,
                "corr": safe_corr(x, y),
            }
        )
    return pd.DataFrame(rows)


def _setup_matplotlib() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")


def make_game_level_parity_plot(
    parity_df: pd.DataFrame,
    parity_summary: pd.DataFrame,
    out_path: Path,
    run_labels: Sequence[str],
    dpi: int,
) -> None:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, (metric, title) in zip(axes, GAME_LEVEL_METRICS):
        metric_df = filter_metric_games(parity_df, metric)
        human_col = f"human_{metric}"
        sim_col = f"sim_{metric}"
        if metric.endswith("_rate"):
            mn, mx = 0.0, 1.0
        else:
            both = pd.concat([metric_df[human_col], metric_df[sim_col]], ignore_index=True)
            mn = float(np.nanmin(both.values)) if len(both) else -1.0
            mx = float(np.nanmax(both.values)) if len(both) else 1.0
            pad = 0.05 * max(1.0, mx - mn)
            mn -= pad
            mx += pad

        for label in run_labels:
            sub = metric_df[metric_df["source"] == label]
            ax.scatter(
                sub[human_col],
                sub[sim_col],
                s=36,
                alpha=0.75,
                color=DEFAULT_COLORS.get(label),
                label=label,
            )

        ax.plot([mn, mx], [mn, mx], linestyle="--", color="#666666", linewidth=1.0)
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        metric_n = int(metric_df["gameId"].nunique()) if "gameId" in metric_df.columns else 0
        suffix = ""
        if metric == "punishment_rate":
            suffix = f" ({metric_n} punishment games)"
        elif metric == "reward_rate":
            suffix = f" ({metric_n} reward games)"
        ax.set_title(f"{title}{suffix}")
        ax.set_xlabel("Human")
        ax.set_ylabel("Simulated")
        ax.grid(alpha=0.25)

        text_lines: List[str] = []
        metric_stats = parity_summary[parity_summary["metric"] == metric]
        for label in run_labels:
            row = metric_stats[metric_stats["source"] == label]
            if row.empty:
                continue
            r = row.iloc[0]
            text_lines.append(
                f"{label}: RMSE={r['rmse']:.3f} corr={r['corr']:.3f} bias={r['bias']:.3f}"
            )
        if text_lines:
            ax.text(
                0.03,
                0.97,
                "\n".join(text_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
            )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(run_labels)))
    fig.suptitle("Macro Alignment: Game-level Parity on Shared Games", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _add_bar_labels(ax: Any, bars: Sequence[Any], values: Sequence[float], y_offset: float = 0.02) -> None:
    for bar, value in zip(bars, values):
        if not math.isfinite(float(value)):
            continue
        height = float(bar.get_height())
        va = "bottom" if height >= 0 else "top"
        offset = y_offset if height >= 0 else -y_offset
        ax.text(
            float(bar.get_x()) + float(bar.get_width()) / 2.0,
            height + offset,
            f"{value:.2f}",
            ha="center",
            va=va,
            fontsize=8,
        )


def _add_side_indicator(ax: Any, mode: str) -> None:
    if mode == "higher":
        xy = (1.02, 0.90)
        xytext = (1.02, 0.10)
        label = "Higher is better"
        arrowstyle = "->"
    elif mode == "lower":
        xy = (1.02, 0.10)
        xytext = (1.02, 0.90)
        label = "Lower is better"
        arrowstyle = "->"
    else:
        xy = (1.02, 0.90)
        xytext = (1.02, 0.10)
        label = "Closer to Human is better"
        arrowstyle = "<->"

    ax.annotate(
        "",
        xy=xy,
        xytext=xytext,
        xycoords="axes fraction",
        arrowprops={"arrowstyle": arrowstyle, "color": "#444444", "linewidth": 1.2},
        annotation_clip=False,
    )
    ax.text(
        1.08,
        0.50,
        label,
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
    )


def _draw_group_baseline_lines(
    ax: Any,
    x_positions: np.ndarray,
    metric_names: Sequence[str],
    baseline_df: pd.DataFrame,
    value_col: str,
    span: float,
) -> None:
    if baseline_df.empty or value_col not in baseline_df.columns:
        return
    first = True
    for xpos, metric in zip(x_positions, metric_names):
        row = baseline_df[baseline_df["metric"] == metric]
        if row.empty:
            continue
        value = float(pd.to_numeric(row.iloc[0][value_col], errors="coerce"))
        if not math.isfinite(value):
            continue
        ax.hlines(
            y=value,
            xmin=float(xpos - span),
            xmax=float(xpos + span),
            colors="#111111",
            linestyles="--",
            linewidth=1.5,
            label="OLS config baseline" if first else None,
        )
        first = False


def make_game_level_correlation_bar_plot(
    corr_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    out_path: Path,
    run_labels: Sequence[str],
    dpi: int,
) -> None:
    if corr_df.empty:
        return

    _setup_matplotlib()
    import matplotlib.pyplot as plt

    metrics = [metric for metric, _ in SUMMARY_CORRELATION_METRICS]
    titles = {metric: title for metric, title in SUMMARY_CORRELATION_METRICS}
    x = np.arange(len(metrics), dtype=float)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(run_labels))

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ymin = min(-0.5, float(corr_df["ci_low"].min(skipna=True)) - 0.05)
    ymax = max(0.8, float(corr_df["ci_high"].max(skipna=True)) + 0.08)
    for idx, source in enumerate(run_labels):
        sub = (
            corr_df[corr_df["source"] == source]
            .set_index("metric")
            .reindex(metrics)
            .reset_index()
        )
        vals = pd.to_numeric(sub["corr"], errors="coerce").to_numpy(dtype=float)
        lows = pd.to_numeric(sub["ci_low"], errors="coerce").to_numpy(dtype=float)
        highs = pd.to_numeric(sub["ci_high"], errors="coerce").to_numpy(dtype=float)
        yerr = np.vstack(
            [
                np.where(np.isfinite(vals - lows), vals - lows, 0.0),
                np.where(np.isfinite(highs - vals), highs - vals, 0.0),
            ]
        )
        bars = ax.bar(
            x + offsets[idx],
            vals,
            width=width,
            color=DEFAULT_COLORS.get(source),
            label=source,
            alpha=0.9,
        )
        _add_bar_labels(ax, bars, vals, y_offset=0.025)

    _draw_group_baseline_lines(ax, x, metrics, baseline_df, value_col="corr", span=0.42)
    ax.axhline(0.0, color="#666666", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([titles[m] for m in metrics])
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Correlation Across Games")
    ax.set_title("Game-level Correlation by Model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", ncol=2)
    _add_side_indicator(ax, mode="higher")
    fig.tight_layout(rect=[0, 0, 0.93, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_game_level_rmse_bar_plot(
    parity_summary: pd.DataFrame,
    baseline_df: pd.DataFrame,
    out_path: Path,
    run_labels: Sequence[str],
    dpi: int,
) -> None:
    if parity_summary.empty:
        return

    _setup_matplotlib()
    import matplotlib.pyplot as plt

    metrics = [metric for metric, _ in SUMMARY_CORRELATION_METRICS]
    titles = {metric: title for metric, title in SUMMARY_CORRELATION_METRICS}
    x = np.arange(len(metrics), dtype=float)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(run_labels))

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ymax = float(parity_summary[parity_summary["metric"].isin(metrics)]["rmse"].max(skipna=True)) + 0.08
    for idx, source in enumerate(run_labels):
        sub = (
            parity_summary[(parity_summary["source"] == source) & (parity_summary["metric"].isin(metrics))]
            .set_index("metric")
            .reindex(metrics)
            .reset_index()
        )
        vals = pd.to_numeric(sub["rmse"], errors="coerce").to_numpy(dtype=float)
        bars = ax.bar(
            x + offsets[idx],
            vals,
            width=width,
            color=DEFAULT_COLORS.get(source),
            label=source,
            alpha=0.9,
        )
        _add_bar_labels(ax, bars, vals, y_offset=0.012)

    _draw_group_baseline_lines(ax, x, metrics, baseline_df, value_col="rmse", span=0.42)
    ax.set_xticks(x)
    ax.set_xticklabels([titles[m] for m in metrics])
    ax.set_ylim(0.0, max(0.3, ymax))
    ax.set_ylabel("RMSE Across Games")
    ax.set_title("Game-level RMSE by Model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", ncol=2)
    _add_side_indicator(ax, mode="lower")
    fig.tight_layout(rect=[0, 0, 0.93, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_player_heterogeneity_plot(
    heterogeneity_summary_df: pd.DataFrame,
    out_path: Path,
    source_order: Sequence[str],
    dpi: int,
) -> None:
    if heterogeneity_summary_df.empty:
        return

    _setup_matplotlib()
    import matplotlib.pyplot as plt

    metrics = [metric for metric, _, _ in PLAYER_RATE_METRICS]
    titles = {metric: title for metric, title, _ in PLAYER_RATE_METRICS}
    x = np.arange(len(metrics), dtype=float)
    width = 0.14
    offsets = np.linspace(-2.0 * width, 2.0 * width, num=len(source_order))

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ymax = float(heterogeneity_summary_df["ci_high"].max(skipna=True)) + 0.04
    for idx, source in enumerate(source_order):
        sub = (
            heterogeneity_summary_df[heterogeneity_summary_df["source"] == source]
            .set_index("metric")
            .reindex(metrics)
            .reset_index()
        )
        vals = pd.to_numeric(sub["mean_player_sd"], errors="coerce").to_numpy(dtype=float)
        lows = pd.to_numeric(sub["ci_low"], errors="coerce").to_numpy(dtype=float)
        highs = pd.to_numeric(sub["ci_high"], errors="coerce").to_numpy(dtype=float)
        yerr = np.vstack(
            [
                np.where(np.isfinite(vals - lows), vals - lows, 0.0),
                np.where(np.isfinite(highs - vals), highs - vals, 0.0),
            ]
        )
        bars = ax.bar(
            x + offsets[idx],
            vals,
            width=width,
            color=DEFAULT_COLORS.get(source),
            label=source,
            alpha=0.9,
        )
        _add_bar_labels(ax, bars, vals, y_offset=0.01)

    ax.set_xticks(x)
    ax.set_xticklabels([titles[m] for m in metrics])
    ax.set_ylim(0.0, max(0.12, ymax))
    ax.set_ylabel("Mean Within-game Player SD")
    ax.set_title("Player-level Heterogeneity Across Games")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", ncol=max(1, min(3, len(source_order))))
    _add_side_indicator(ax, mode="match_human")
    fig.tight_layout(rect=[0, 0, 0.93, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_distribution_wasserstein_plot(
    dist_summary_df: pd.DataFrame,
    out_path: Path,
    run_labels: Sequence[str],
    dpi: int,
) -> None:
    if dist_summary_df.empty:
        return

    _setup_matplotlib()
    import matplotlib.pyplot as plt

    metrics = [metric for metric, _, _ in PLAYER_RATE_METRICS]
    titles = {metric: title for metric, title, _ in PLAYER_RATE_METRICS}
    x = np.arange(len(metrics), dtype=float)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(run_labels))

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ymax = float(dist_summary_df["ci_high"].max(skipna=True)) + 0.03
    for idx, source in enumerate(run_labels):
        sub = (
            dist_summary_df[dist_summary_df["source"] == source]
            .set_index("metric")
            .reindex(metrics)
            .reset_index()
        )
        vals = pd.to_numeric(sub["mean_wasserstein_1"], errors="coerce").to_numpy(dtype=float)
        lows = pd.to_numeric(sub["ci_low"], errors="coerce").to_numpy(dtype=float)
        highs = pd.to_numeric(sub["ci_high"], errors="coerce").to_numpy(dtype=float)
        yerr = np.vstack(
            [
                np.where(np.isfinite(vals - lows), vals - lows, 0.0),
                np.where(np.isfinite(highs - vals), highs - vals, 0.0),
            ]
        )
        bars = ax.bar(
            x + offsets[idx],
            vals,
            width=width,
            color=DEFAULT_COLORS.get(source),
            label=source,
            alpha=0.9,
        )
        _add_bar_labels(ax, bars, vals, y_offset=0.008)

    ax.set_xticks(x)
    ax.set_xticklabels([titles[m] for m in metrics])
    ax.set_ylim(0.0, max(0.12, ymax))
    ax.set_ylabel("Mean Within-game Wasserstein-1 Distance")
    ax.set_title("Distributional Alignment: Mean Within-game Wasserstein-1 Across Games")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", ncol=2)
    _add_side_indicator(ax, mode="lower")
    fig.tight_layout(rect=[0, 0, 0.93, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_player_distribution_panels(
    player_game_df: pd.DataFrame,
    out_path: Path,
    metric: str,
    title: str,
    source_order: Sequence[str],
    dpi: int,
    enabled_col: Optional[str] = None,
) -> None:
    game_meta = player_game_df[player_game_df["source"] == "human"][
        ["gameId", "CONFIG_treatmentName", "CONFIG_punishmentExists", "CONFIG_rewardExists"]
    ].drop_duplicates(subset=["gameId"], keep="first")
    if enabled_col is not None:
        game_meta = game_meta.loc[parse_bool_series(game_meta[enabled_col])].copy()
    game_meta = game_meta.sort_values(["CONFIG_treatmentName", "gameId"], kind="stable").reset_index(drop=True)
    n_games = int(len(game_meta))
    if n_games == 0:
        return

    _setup_matplotlib()
    import matplotlib.pyplot as plt

    ncols = min(5, max(1, math.ceil(math.sqrt(n_games))))
    nrows = int(math.ceil(n_games / float(ncols)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.6 * nrows),
        sharex=True,
        sharey=True,
    )
    axes_array = np.atleast_1d(axes).reshape(-1)

    for ax, row in zip(axes_array, game_meta.itertuples(index=False)):
        game_id = str(row.gameId)
        treatment = str(row.CONFIG_treatmentName or "").strip()
        short_title = treatment if treatment else game_id[:8]
        sub = player_game_df[player_game_df["gameId"] == game_id]
        human_vals = np.sort(
            pd.to_numeric(sub[sub["source"] == "human"][metric], errors="coerce").dropna().to_numpy(dtype=float)
        )
        text_lines: List[str] = []
        for source in source_order:
            src = sub[sub["source"] == source].copy()
            vals = np.sort(pd.to_numeric(src[metric], errors="coerce").dropna().to_numpy(dtype=float))
            if len(vals) == 0:
                continue
            x = np.linspace(0.0, 1.0, len(vals)) if len(vals) > 1 else np.array([0.5], dtype=float)
            ax.plot(
                x,
                vals,
                marker="o",
                linewidth=1.6 if source == "human" else 1.2,
                markersize=2.4,
                color=DEFAULT_COLORS.get(source),
                alpha=0.95 if source == "human" else 0.85,
                label=source,
            )
            if source != "human" and len(human_vals) > 0:
                stats = compute_distribution_alignment(human_vals, vals)
                if math.isfinite(float(stats["w1"])):
                    text_lines.append(f"{short_source_label(source)} W1={stats['w1']:.2f}")
        ax.set_title(short_title, fontsize=9)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)
        if text_lines:
            ax.text(
                0.03,
                0.97,
                "\n".join(text_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=6.7,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#dddddd"},
            )

    for ax in axes_array[n_games:]:
        ax.axis("off")
    for ax in axes_array[-ncols:]:
        ax.set_xlabel("Player Quantile")
    for idx in range(0, len(axes_array), ncols):
        axes_array[idx].set_ylabel("Rate")

    handles, labels = axes_array[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(source_order)))
    fig.suptitle(f"{title} (Per-game Player Distributions)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_contribution_shift_parity_plot(
    shift_parity_df: pd.DataFrame,
    shift_summary: pd.DataFrame,
    out_path: Path,
    run_labels: Sequence[str],
    dpi: int,
) -> None:
    if shift_parity_df.empty:
        return

    _setup_matplotlib()
    import matplotlib.pyplot as plt

    x_all = pd.to_numeric(shift_parity_df["human_contribution_shift"], errors="coerce")
    y_all = pd.to_numeric(shift_parity_df["sim_contribution_shift"], errors="coerce")
    both = pd.concat([x_all, y_all], ignore_index=True)
    mn = float(np.nanmin(both.values)) if len(both) else -1.0
    mx = float(np.nanmax(both.values)) if len(both) else 1.0
    pad = 0.08 * max(0.2, mx - mn)
    mn -= pad
    mx += pad

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for label in run_labels:
        sub = shift_parity_df[shift_parity_df["source"] == label]
        ax.scatter(
            sub["human_contribution_shift"],
            sub["sim_contribution_shift"],
            s=36,
            alpha=0.8,
            color=DEFAULT_COLORS.get(label),
            label=label,
        )
    ax.plot([mn, mx], [mn, mx], linestyle="--", color="#666666", linewidth=1.0)
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.axvline(0.0, color="#999999", linewidth=0.8)
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_xlabel("Human Early-to-late Contribution Shift")
    ax.set_ylabel("Simulated Early-to-late Contribution Shift")
    ax.set_title("Macro Alignment: Early-to-late Contribution Shift")
    ax.grid(alpha=0.25)

    text_lines: List[str] = []
    for label in run_labels:
        row = shift_summary[shift_summary["source"] == label]
        if row.empty:
            continue
        r = row.iloc[0]
        text_lines.append(f"{label}: RMSE={r['rmse']:.3f} corr={r['corr']:.3f} bias={r['bias']:.3f}")
    if text_lines:
        ax.text(
            0.03,
            0.97,
            "\n".join(text_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(run_labels)))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_trajectory_plot(trajectory_df: pd.DataFrame, out_path: Path, source_order: Sequence[str], dpi: int) -> None:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    for ax, (metric, title) in zip(axes, TRAJECTORY_METRICS):
        sub = trajectory_df[trajectory_df["metric"] == metric]
        for source in source_order:
            gg = sub[sub["source"] == source].sort_values("progress_mid")
            if gg.empty:
                continue
            x = gg["progress_mid"].to_numpy(dtype=float)
            y = gg["mean"].to_numpy(dtype=float)
            lo = gg["ci_low"].to_numpy(dtype=float)
            hi = gg["ci_high"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", linewidth=2.0, markersize=4, label=source, color=DEFAULT_COLORS.get(source))
            ax.fill_between(x, lo, hi, alpha=0.18, color=DEFAULT_COLORS.get(source))
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax.set_ylabel("Mean")
        if metric.endswith("_rate"):
            ax.set_ylim(0.0, 1.0)
        stats_map = summarize_series_alignment_vs_human(sub, x_col="progress_mid", source_order=source_order)
        text_lines: List[str] = []
        for source in source_order:
            if source == "human" or source not in stats_map:
                continue
            stats = stats_map[source]
            if math.isfinite(float(stats["corr"])):
                text_lines.append(
                    f"{short_source_label(source)} RMSE={stats['rmse']:.2f} corr={stats['corr']:.2f}"
                )
            else:
                text_lines.append(f"{short_source_label(source)} RMSE={stats['rmse']:.2f}")
        if text_lines:
            ax.text(
                0.03,
                0.97,
                "\n".join(text_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#dddddd"},
            )
    for ax in axes[2:]:
        ax.set_xlabel("Round Progress")
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(source_order)))
    fig.suptitle("Macro Alignment: Round-normalized Trajectories", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_event_plot(event_df: pd.DataFrame, out_path: Path, source_order: Sequence[str], dpi: int) -> None:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (event_type, title) in zip(
        axes,
        (
            ("received_punishment", "Response After Receiving Punishment"),
            ("received_reward", "Response After Receiving Reward"),
        ),
    ):
        sub = event_df[event_df["event_type"] == event_type]
        for source in source_order:
            gg = sub[sub["source"] == source].sort_values("event_flag")
            if gg.empty:
                continue
            x = gg["event_flag"].to_numpy(dtype=float)
            y = gg["mean"].to_numpy(dtype=float)
            yerr = gg["ci_half"].fillna(0.0).to_numpy(dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linewidth=2.0,
                capsize=3,
                label=source,
                color=DEFAULT_COLORS.get(source),
            )
        ax.axhline(0.0, color="#666666", linewidth=1.0)
        ax.set_title(title)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No", "Yes"])
        ax.set_xlabel("Event in Current Round")
        ax.grid(alpha=0.25)
        stats_map = summarize_series_alignment_vs_human(sub, x_col="event_flag", source_order=source_order)
        text_lines: List[str] = []
        for source in source_order:
            if source == "human" or source not in stats_map:
                continue
            stats = stats_map[source]
            text_lines.append(f"{short_source_label(source)} RMSE={stats['rmse']:.2f}")
        if text_lines:
            ax.text(
                0.03,
                0.97,
                "\n".join(text_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#dddddd"},
            )
    axes[0].set_ylabel("Next-round Delta in Contribution Rate")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(source_order)))
    fig.suptitle("Macro Alignment: Sanction-response Event Study", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_targeting_plot(
    targeting_df: pd.DataFrame,
    out_path: Path,
    source_order: Sequence[str],
    num_bins: int,
    dpi: int,
) -> None:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    xticks = [(i + 0.5) / float(num_bins) for i in range(num_bins)]
    xticklabels = [f"{int(100 * i / num_bins)}-{int(100 * (i + 1) / num_bins)}%" for i in range(num_bins)]
    for ax, (metric, title) in zip(axes, TARGETING_METRICS):
        sub = targeting_df[targeting_df["metric"] == metric]
        for source in source_order:
            gg = sub[sub["source"] == source].sort_values("prev_contrib_mid")
            if gg.empty:
                continue
            x = gg["prev_contrib_mid"].to_numpy(dtype=float)
            y = gg["mean"].to_numpy(dtype=float)
            lo = gg["ci_low"].to_numpy(dtype=float)
            hi = gg["ci_high"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", linewidth=2.0, markersize=4, label=source, color=DEFAULT_COLORS.get(source))
            ax.fill_between(x, lo, hi, alpha=0.18, color=DEFAULT_COLORS.get(source))
        ax.set_title(title)
        ax.set_xlabel("Previous-round Contribution Percentile")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        stats_map = summarize_series_alignment_vs_human(sub, x_col="prev_contrib_mid", source_order=source_order)
        text_lines: List[str] = []
        for source in source_order:
            if source == "human" or source not in stats_map:
                continue
            stats = stats_map[source]
            if math.isfinite(float(stats["corr"])):
                text_lines.append(
                    f"{short_source_label(source)} RMSE={stats['rmse']:.2f} corr={stats['corr']:.2f}"
                )
            else:
                text_lines.append(f"{short_source_label(source)} RMSE={stats['rmse']:.2f}")
        if text_lines:
            ax.text(
                0.03,
                0.97,
                "\n".join(text_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#dddddd"},
            )
    axes[0].set_ylabel("Probability")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(source_order)))
    fig.suptitle("Macro Alignment: Targeting by Prior Contribution", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare macro simulation alignment to human data with game-level parity, "
            "round-normalized trajectories, sanction-response event studies, and "
            "targeting by previous-round contribution."
        )
    )
    parser.add_argument(
        "--eval_root",
        type=Path,
        default=Path("outputs/benchmark/runs/benchmark_filtered/macro_simulation_eval"),
    )
    parser.add_argument("--run_ids", type=str, required=True, help="Comma-separated run IDs under eval_root.")
    parser.add_argument("--labels", type=str, required=True, help="Comma-separated labels aligned to --run_ids.")
    parser.add_argument("--human_analysis_csv", type=Path, default=None)
    parser.add_argument("--human_rounds_csv", type=Path, default=None)
    parser.add_argument(
        "--analysis_root",
        type=Path,
        default=Path("reports/benchmark/macro_simulation_eval"),
    )
    parser.add_argument("--analysis_run_id", type=str, required=True)
    parser.add_argument("--shared_games_only", action="store_true", default=True)
    parser.add_argument(
        "--magnitude_override_by_label",
        type=str,
        default=None,
        help=(
            "Comma-separated label:value pairs. For matching simulated labels, override both "
            "punishment and reward magnitudes when recomputing payoffs."
        ),
    )
    parser.add_argument(
        "--unit_edge_by_label",
        type=str,
        default=None,
        help=(
            "Comma-separated labels. For matching simulated labels, collapse each nonzero "
            "punishment/reward edge to unit 1 and recompute payoffs with received magnitude 1."
        ),
    )
    parser.add_argument("--progress_bins", type=int, default=10)
    parser.add_argument("--targeting_bins", type=int, default=5)
    parser.add_argument("--bootstrap_iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    analysis_root = args.analysis_root.resolve()
    out_dir = (analysis_root / args.analysis_run_id).resolve()
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_ids = split_csv_arg(args.run_ids)
    labels = split_csv_arg(args.labels)
    magnitude_override_by_label = parse_label_float_map(args.magnitude_override_by_label)
    unit_edge_labels = set(split_csv_arg(args.unit_edge_by_label))
    if not run_ids:
        raise ValueError("Provide at least one run ID via --run_ids.")
    if len(run_ids) != len(labels):
        raise ValueError("--labels must have the same length as --run_ids.")
    source_order = ["human", *labels]

    selected_rows: List[Dict[str, object]] = []
    sim_game_sets: List[set[str]] = []
    human_analysis_csv = args.human_analysis_csv.resolve() if args.human_analysis_csv else None
    human_rounds_csv = args.human_rounds_csv.resolve() if args.human_rounds_csv else None

    for run_id, label in zip(run_ids, labels):
        eval_csv = (eval_root / run_id / "macro_simulation_eval.csv").resolve()
        if not eval_csv.exists():
            raise FileNotFoundError(f"Missing macro_simulation_eval.csv: {eval_csv}")
        run_cfg = _load_run_config(run_id, eval_root=eval_root)
        inferred_analysis = _extract_analysis_csv_from_config(run_cfg, eval_root=eval_root, run_id=run_id)
        if human_analysis_csv is None:
            if inferred_analysis is None:
                raise ValueError(
                    f"Could not infer human analysis CSV for run '{run_id}'. Pass --human_analysis_csv."
                )
            human_analysis_csv = inferred_analysis
        if human_rounds_csv is None:
            inferred_rounds = _infer_rounds_csv_from_analysis(human_analysis_csv)
            if inferred_rounds is None:
                raise ValueError(
                    "Could not infer human rounds CSV from analysis CSV. Pass --human_rounds_csv."
                )
            human_rounds_csv = inferred_rounds

        sim_df = pd.read_csv(eval_csv, usecols=["gameId"])
        sim_games = set(sim_df["gameId"].astype(str))
        sim_game_sets.append(sim_games)
        selected_rows.append(
            {
                "run_id": run_id,
                "label": label,
                "eval_csv": str(eval_csv),
                "n_games_raw": int(len(sim_games)),
                "magnitude_override": magnitude_override_by_label.get(label),
                "unit_edge_actions": bool(label in unit_edge_labels),
            }
        )

    assert human_analysis_csv is not None
    assert human_rounds_csv is not None
    if not human_analysis_csv.exists():
        raise FileNotFoundError(f"Missing human analysis CSV: {human_analysis_csv}")
    if not human_rounds_csv.exists():
        raise FileNotFoundError(f"Missing human rounds CSV: {human_rounds_csv}")

    learn_analysis_csv = _infer_split_csv(human_analysis_csv, target_split="learn")
    learn_rounds_csv = _infer_split_csv(human_rounds_csv, target_split="learn")

    config_df = load_game_config(human_analysis_csv)
    shared_games = set(config_df["gameId"].astype(str))
    for sim_games in sim_game_sets:
        shared_games &= sim_games
    shared_game_ids = sorted(shared_games)
    if not shared_game_ids:
        raise ValueError("No shared game IDs across the provided runs and the human data.")

    human_rows = load_human_rows(human_rounds_csv, config_df=config_df, game_ids=shared_game_ids)
    run_row_tables: Dict[str, pd.DataFrame] = {"human": human_rows}
    for run_id, label in zip(run_ids, labels):
        eval_csv = (eval_root / run_id / "macro_simulation_eval.csv").resolve()
        run_row_tables[label] = load_sim_rows(
            eval_csv=eval_csv,
            config_df=config_df,
            source=label,
            game_ids=shared_game_ids,
            magnitude_override=magnitude_override_by_label.get(label),
            unit_edge_actions=bool(label in unit_edge_labels),
        )

    game_tables = {
        "human": compute_game_level_metrics(run_row_tables["human"], config_df=config_df, use_human_efficiency=True)
    }
    for label in labels:
        game_tables[label] = compute_game_level_metrics(
            run_row_tables[label],
            config_df=config_df,
            use_human_efficiency=False,
        )

    parity_df = build_parity_table(game_tables=game_tables, run_labels=labels)
    parity_summary = summarize_parity(parity_df)
    ols_baseline_summary = (
        compute_linear_config_baseline_summary(
            learn_analysis_csv=learn_analysis_csv,
            learn_rounds_csv=learn_rounds_csv,
            val_analysis_csv=human_analysis_csv,
            val_rounds_csv=human_rounds_csv,
            eval_game_ids=shared_game_ids,
        )
        if learn_analysis_csv is not None and learn_rounds_csv is not None
        else pd.DataFrame()
    )

    all_rows = pd.concat([run_row_tables["human"]] + [run_row_tables[label] for label in labels], ignore_index=True)
    game_round_df = build_game_round_table(all_rows, num_bins=int(args.progress_bins))
    trajectory_df = summarize_trajectories(game_round_df)
    event_df = summarize_event_study(all_rows)
    targeting_df = summarize_targeting(all_rows, num_bins=int(args.targeting_bins))
    player_game_distribution_df = build_player_game_distribution_table(all_rows)
    player_heterogeneity_df = build_player_heterogeneity_table(player_game_distribution_df)
    player_heterogeneity_summary = summarize_player_heterogeneity(
        player_heterogeneity_df,
        source_order=source_order,
        n_boot=int(args.bootstrap_iterations),
    )
    game_level_corr_summary = summarize_game_level_correlations(
        parity_df,
        run_labels=labels,
        n_boot=int(args.bootstrap_iterations),
    )
    distribution_alignment_by_game_df = build_distribution_alignment_by_game(
        player_game_distribution_df,
        run_labels=labels,
    )
    distribution_alignment_summary = summarize_distribution_alignment(
        distribution_alignment_by_game_df,
        run_labels=labels,
        n_boot=int(args.bootstrap_iterations),
    )
    contribution_shift_df = summarize_contribution_shift(game_round_df)
    contribution_shift_parity_df = build_contribution_shift_parity(contribution_shift_df, run_labels=labels)
    contribution_shift_summary = summarize_contribution_shift_parity(contribution_shift_parity_df)

    pd.DataFrame({"gameId": shared_game_ids}).to_csv(out_dir / "shared_game_ids.csv", index=False)
    pd.DataFrame(selected_rows).assign(n_games_shared=len(shared_game_ids)).to_csv(
        out_dir / "selected_runs.csv", index=False
    )
    parity_df.to_csv(out_dir / "game_level_parity.csv", index=False)
    parity_summary.to_csv(out_dir / "game_level_alignment_summary.csv", index=False)
    ols_baseline_summary.to_csv(out_dir / "game_level_ols_baseline_summary.csv", index=False)
    game_round_df.to_csv(out_dir / "game_round_metrics.csv", index=False)
    trajectory_df.to_csv(out_dir / "trajectory_alignment.csv", index=False)
    event_df.to_csv(out_dir / "event_study_alignment.csv", index=False)
    targeting_df.to_csv(out_dir / "targeting_alignment.csv", index=False)
    player_game_distribution_df.to_csv(out_dir / "player_game_distribution_metrics.csv", index=False)
    player_heterogeneity_df.to_csv(out_dir / "player_heterogeneity_by_game.csv", index=False)
    player_heterogeneity_summary.to_csv(out_dir / "player_heterogeneity_summary.csv", index=False)
    game_level_corr_summary.to_csv(out_dir / "game_level_correlation_bootstrap.csv", index=False)
    distribution_alignment_by_game_df.to_csv(out_dir / "player_distribution_alignment_by_game.csv", index=False)
    distribution_alignment_summary.to_csv(out_dir / "player_distribution_wasserstein_summary.csv", index=False)
    contribution_shift_df.to_csv(out_dir / "contribution_shift_by_game.csv", index=False)
    contribution_shift_parity_df.to_csv(out_dir / "contribution_shift_parity.csv", index=False)
    contribution_shift_summary.to_csv(out_dir / "contribution_shift_alignment.csv", index=False)

    parity_fig = figures_dir / "macro_alignment_game_level_parity.png"
    trajectory_fig = figures_dir / "macro_alignment_round_trajectories.png"
    event_fig = figures_dir / "macro_alignment_event_study.png"
    targeting_fig = figures_dir / "macro_alignment_targeting.png"
    contrib_dist_fig = figures_dir / "macro_alignment_player_distribution_contribution.png"
    punish_dist_fig = figures_dir / "macro_alignment_player_distribution_punishment.png"
    reward_dist_fig = figures_dir / "macro_alignment_player_distribution_reward.png"
    contribution_shift_fig = figures_dir / "macro_alignment_contribution_shift_parity.png"
    corr_bar_fig = figures_dir / "macro_alignment_game_level_correlation_bars.png"
    rmse_bar_fig = figures_dir / "macro_alignment_game_level_rmse_bars.png"
    heterogeneity_fig = figures_dir / "macro_alignment_player_heterogeneity.png"
    wasserstein_fig = figures_dir / "macro_alignment_distribution_wasserstein.png"

    make_game_level_parity_plot(parity_df, parity_summary, parity_fig, run_labels=labels, dpi=int(args.dpi))
    make_game_level_correlation_bar_plot(
        game_level_corr_summary,
        ols_baseline_summary,
        corr_bar_fig,
        run_labels=labels,
        dpi=int(args.dpi),
    )
    make_game_level_rmse_bar_plot(
        parity_summary,
        ols_baseline_summary,
        rmse_bar_fig,
        run_labels=labels,
        dpi=int(args.dpi),
    )
    make_trajectory_plot(trajectory_df, trajectory_fig, source_order=source_order, dpi=int(args.dpi))
    make_event_plot(event_df, event_fig, source_order=source_order, dpi=int(args.dpi))
    if not targeting_df.empty:
        make_targeting_plot(
            targeting_df,
            targeting_fig,
            source_order=source_order,
            num_bins=int(args.targeting_bins),
            dpi=int(args.dpi),
        )
    make_player_distribution_panels(
        player_game_distribution_df,
        contrib_dist_fig,
        metric="contribution_rate",
        title="Contribution Rate",
        source_order=source_order,
        dpi=int(args.dpi),
    )
    make_player_distribution_panels(
        player_game_distribution_df,
        punish_dist_fig,
        metric="punishment_rate",
        title="Punishment Rate",
        source_order=source_order,
        dpi=int(args.dpi),
        enabled_col="CONFIG_punishmentExists",
    )
    make_player_distribution_panels(
        player_game_distribution_df,
        reward_dist_fig,
        metric="reward_rate",
        title="Reward Rate",
        source_order=source_order,
        dpi=int(args.dpi),
        enabled_col="CONFIG_rewardExists",
    )
    make_contribution_shift_parity_plot(
        contribution_shift_parity_df,
        contribution_shift_summary,
        contribution_shift_fig,
        run_labels=labels,
        dpi=int(args.dpi),
    )
    make_player_heterogeneity_plot(
        player_heterogeneity_summary,
        heterogeneity_fig,
        source_order=source_order,
        dpi=int(args.dpi),
    )
    make_distribution_wasserstein_plot(
        distribution_alignment_summary,
        wasserstein_fig,
        run_labels=labels,
        dpi=int(args.dpi),
    )

    manifest = {
        "analysis_root": str(analysis_root),
        "analysis_run_id": args.analysis_run_id,
        "eval_root": str(eval_root),
        "run_ids": run_ids,
        "labels": labels,
        "human_analysis_csv": str(human_analysis_csv),
        "human_rounds_csv": str(human_rounds_csv),
        "learn_analysis_csv": str(learn_analysis_csv) if learn_analysis_csv is not None else None,
        "learn_rounds_csv": str(learn_rounds_csv) if learn_rounds_csv is not None else None,
        "shared_games_only": bool(args.shared_games_only),
        "n_shared_games": int(len(shared_game_ids)),
        "progress_bins": int(args.progress_bins),
        "targeting_bins": int(args.targeting_bins),
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "magnitude_override_by_label": magnitude_override_by_label,
        "unit_edge_by_label": sorted(unit_edge_labels),
        "outputs": {
            "selected_runs_csv": str(out_dir / "selected_runs.csv"),
            "shared_game_ids_csv": str(out_dir / "shared_game_ids.csv"),
            "game_level_parity_csv": str(out_dir / "game_level_parity.csv"),
            "game_level_alignment_summary_csv": str(out_dir / "game_level_alignment_summary.csv"),
            "game_level_ols_baseline_summary_csv": str(out_dir / "game_level_ols_baseline_summary.csv"),
            "game_level_correlation_bootstrap_csv": str(out_dir / "game_level_correlation_bootstrap.csv"),
            "game_level_rmse_csv": str(out_dir / "game_level_alignment_summary.csv"),
            "game_round_metrics_csv": str(out_dir / "game_round_metrics.csv"),
            "trajectory_alignment_csv": str(out_dir / "trajectory_alignment.csv"),
            "event_study_alignment_csv": str(out_dir / "event_study_alignment.csv"),
            "targeting_alignment_csv": str(out_dir / "targeting_alignment.csv"),
            "player_game_distribution_metrics_csv": str(out_dir / "player_game_distribution_metrics.csv"),
            "player_heterogeneity_by_game_csv": str(out_dir / "player_heterogeneity_by_game.csv"),
            "player_heterogeneity_summary_csv": str(out_dir / "player_heterogeneity_summary.csv"),
            "player_distribution_alignment_by_game_csv": str(out_dir / "player_distribution_alignment_by_game.csv"),
            "player_distribution_wasserstein_summary_csv": str(out_dir / "player_distribution_wasserstein_summary.csv"),
            "contribution_shift_by_game_csv": str(out_dir / "contribution_shift_by_game.csv"),
            "contribution_shift_parity_csv": str(out_dir / "contribution_shift_parity.csv"),
            "contribution_shift_alignment_csv": str(out_dir / "contribution_shift_alignment.csv"),
            "figures": [
                str(parity_fig),
                str(corr_bar_fig) if corr_bar_fig.exists() else None,
                str(rmse_bar_fig) if rmse_bar_fig.exists() else None,
                str(trajectory_fig),
                str(event_fig),
                str(targeting_fig) if targeting_fig.exists() else None,
                str(contrib_dist_fig) if contrib_dist_fig.exists() else None,
                str(punish_dist_fig) if punish_dist_fig.exists() else None,
                str(reward_dist_fig) if reward_dist_fig.exists() else None,
                str(heterogeneity_fig) if heterogeneity_fig.exists() else None,
                str(wasserstein_fig) if wasserstein_fig.exists() else None,
                str(contribution_shift_fig) if contribution_shift_fig.exists() else None,
            ],
        },
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote macro alignment comparison -> {out_dir}")
    print(f"Shared games: {len(shared_game_ids)}")
    if not parity_summary.empty:
        print(parity_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
