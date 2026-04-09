from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_sequential.code import llm_prompting, plot_utils
from repo_env import get_env_var

def _import_llm_client():
    try:
        from Simulation_robin.llm_client import LLMClient as _LLMClient

        return _LLMClient, None
    except Exception as first_exc:
        root = Path(__file__).resolve().parents[1]
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        try:
            from Simulation_robin.llm_client import LLMClient as _LLMClient

            return _LLMClient, None
        except Exception as second_exc:  # pragma: no cover
            return None, f"{first_exc}; retry_after_sys_path:{second_exc}"


LLMClient, LLM_IMPORT_ERROR = _import_llm_client()


FIXED_CONFIG_FEATURES = [
    "CONFIG_playerCount",
    "CONFIG_numRounds",
    "CONFIG_showNRounds",
    "CONFIG_allOrNothing",
    "CONFIG_chat",
    "CONFIG_defaultContribProp",
    "CONFIG_punishmentCost",
    "CONFIG_punishmentTech",
    "CONFIG_rewardExists",
    "CONFIG_rewardCost",
    "CONFIG_rewardTech",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
    "CONFIG_MPCR",
]

FEATURES = FIXED_CONFIG_FEATURES + ["control_itt_efficiency"]
TARGET = "treatment_itt_efficiency"
ALLOWED_METHODS = ("random", "bo", "llm")

LLM_SYSTEM_PROMPT = (
    "You are selecting the next most informative experiment for a behavioral science study. "
    "Return ONLY valid JSON matching the required schema. Do not include markdown fences."
)


@dataclass(frozen=True)
class Metrics:
    mse: float
    rmse: float
    r2_custom: float
    train_n_rows: int
    train_rank: int
    train_cond: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build random/BO/LLM learning curves for paired experiment selection."
    )
    parser.add_argument("--n-start", type=int, default=10)
    parser.add_argument("--n-max", type=int, default=150)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--random-runs", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--methods",
        type=str,
        default="random,bo,llm",
        help="Comma-separated methods: random,bo,llm",
    )
    parser.add_argument(
        "--seed-scope",
        choices=["fixed42", "multi20"],
        default="fixed42",
        help="fixed42 for smoke, multi20 for full benchmark across seeds 42..61.",
    )
    parser.add_argument(
        "--regression-model",
        choices=["linear", "ridge"],
        default="ridge",
        help="Downstream predictor used to score RMSE/R^2 on val.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--bo-acquisition",
        choices=["ei", "max_variance"],
        default="ei",
        help="Acquisition function for BO baseline.",
    )

    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--llm-model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--llm-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--llm-shortlist-k", type=int, default=20)
    parser.add_argument("--llm-batch-size", type=int, default=10)
    parser.add_argument(
        "--llm-n-max",
        type=int,
        default=0,
        help="Max n_pairs for LLM-assisted curve only; 0 means use global n-max.",
    )
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-top-p", type=float, default=1.0)
    parser.add_argument("--llm-max-output-tokens", type=int, default=12000)
    parser.add_argument("--llm-max-context-chars", type=int, default=120000)
    parser.add_argument(
        "--llm-overflow-policy",
        choices=["trim_oldest"],
        default="trim_oldest",
    )

    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def normalize_config_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.endswith(".0"):
        maybe_int = text[:-2]
        if maybe_int.lstrip("-").isdigit():
            return maybe_int
    if text.lstrip("-").isdigit():
        return str(int(text))
    try:
        as_float = float(text)
    except ValueError:
        return text
    if as_float.is_integer():
        return str(int(as_float))
    return text


def config_id_sort_key(config_id: str) -> tuple[int, Any]:
    cid = normalize_config_id(config_id)
    if cid.lstrip("-").isdigit():
        return (0, int(cid))
    return (1, cid)


def parse_methods(methods_csv: str) -> list[str]:
    methods = [m.strip() for m in methods_csv.split(",") if m.strip()]
    if not methods:
        raise ValueError("--methods cannot be empty")
    invalid = [m for m in methods if m not in ALLOWED_METHODS]
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}. Allowed: {ALLOWED_METHODS}")
    # preserve user order but dedupe
    ordered = []
    for m in methods:
        if m not in ordered:
            ordered.append(m)
    return ordered


def resolve_methods_for_env(methods: list[str], args: argparse.Namespace) -> tuple[list[str], Optional[str]]:
    if "llm" not in methods:
        return methods, None

    if args.llm_provider.lower() == "openai" and not get_env_var(args.llm_api_key_env):
        resolved = [m for m in methods if m != "llm"]
        reason = f"llm_skipped_missing_api_key_env:{args.llm_api_key_env}"
        return resolved, reason

    return methods, None


def cast_bool_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns and out[col].dtype == bool:
            out[col] = out[col].astype(int)
    return out


def build_regressor(model_name: str, ridge_alpha: float):
    if model_name == "linear":
        return LinearRegression()
    if model_name == "ridge":
        return Ridge(alpha=ridge_alpha, random_state=0)
    raise ValueError(f"Unsupported regression model: {model_name}")


def matrix_diagnostics(x: np.ndarray) -> tuple[int, float]:
    rank = int(np.linalg.matrix_rank(x))
    if rank < x.shape[1]:
        return rank, float("inf")
    cond = float(np.linalg.cond(x))
    return rank, cond


def train_eval(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    mse_base: float,
    regression_model: str,
    ridge_alpha: float,
) -> Metrics:
    x_train = train_df[FEATURES].to_numpy(dtype=float)
    y_train = train_df[TARGET].to_numpy(dtype=float)
    x_val = val_df[FEATURES].to_numpy(dtype=float)
    y_val = val_df[TARGET].to_numpy(dtype=float)

    model = build_regressor(regression_model, ridge_alpha)
    model.fit(x_train, y_train)
    pred = model.predict(x_val)

    mse = float(mean_squared_error(y_val, pred))
    rmse = float(np.sqrt(mse))
    r2_custom = float(1 - mse / mse_base)
    rank, cond = matrix_diagnostics(x_train)
    return Metrics(
        mse=mse,
        rmse=rmse,
        r2_custom=r2_custom,
        train_n_rows=int(x_train.shape[0]),
        train_rank=rank,
        train_cond=cond,
    )


def make_gp(random_state: int) -> Pipeline:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(length_scale=1.0, nu=2.5)
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=random_state,
    )
    return Pipeline([("scale", StandardScaler()), ("gp", gp)])


def acquisition_ei(mu: np.ndarray, std: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    std_safe = np.maximum(std, 1e-12)
    improvement = mu - y_best - xi
    z = improvement / std_safe
    ei = improvement * norm.cdf(z) + std_safe * norm.pdf(z)
    ei[std < 1e-12] = 0.0
    return ei


def build_n_values(n_start: int, n_max: int, step: int) -> list[int]:
    if n_start <= 0 or n_max <= 0 or step <= 0:
        raise ValueError("n-start, n-max, and step must all be positive.")
    if n_start > n_max:
        raise ValueError("n-start must be <= n-max.")
    n_values = list(range(n_start, n_max + 1, step))
    if n_values[-1] != n_max:
        n_values.append(n_max)
    return n_values


def build_shortlist_by_ei(
    train_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    random_state: int,
    shortlist_k: int,
) -> pd.DataFrame:
    gp = make_gp(random_state=random_state)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        gp.fit(train_df[FEATURES], train_df[TARGET])

    mu, std = gp.predict(pool_df[FEATURES], return_std=True)
    ei = acquisition_ei(mu=mu, std=std, y_best=float(train_df[TARGET].max()), xi=0.01)

    shortlist = pool_df[["CONFIG_configId"] + FEATURES].copy()
    shortlist["config_id"] = shortlist["CONFIG_configId"].apply(normalize_config_id)
    shortlist["gp_mu"] = mu
    shortlist["gp_std"] = std
    shortlist["bo_score_ei"] = ei
    shortlist = shortlist.sort_values(["bo_score_ei", "gp_std"], ascending=[False, False]).reset_index(drop=True)
    shortlist = shortlist.head(min(shortlist_k, len(shortlist))).copy()
    shortlist["ei_rank"] = np.arange(1, len(shortlist) + 1)
    return shortlist


def build_seed_sets(
    seed_scope: str,
    all_ids: list[str],
    fixed_seed_ids: list[str],
) -> dict[int, list[str]]:
    if seed_scope == "fixed42":
        return {42: list(fixed_seed_ids)}

    out: dict[int, list[str]] = {}
    for seed_state in range(42, 62):
        if seed_state == 42:
            out[seed_state] = list(fixed_seed_ids)
            continue
        rng = np.random.default_rng(seed_state)
        sampled = rng.choice(all_ids, size=10, replace=False)
        out[seed_state] = sorted([normalize_config_id(x) for x in sampled], key=config_id_sort_key)
    return out


def run_random_curve(
    learn: pd.DataFrame,
    val: pd.DataFrame,
    seed_ids: list[str],
    n_values: list[int],
    random_runs: int,
    random_state: int,
    mse_base: float,
    regression_model: str,
    ridge_alpha: float,
    seed_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_ids = sorted(learn["config_id"].unique(), key=config_id_sort_key)
    seed_set = set(seed_ids)
    remaining_ids = [cid for cid in all_ids if cid not in seed_set]
    n_seed = len(seed_set)

    run_rows: list[dict[str, Any]] = []
    for run_idx in range(random_runs):
        rng = np.random.default_rng(random_state + run_idx)
        perm = list(rng.permutation(remaining_ids))
        for n_pairs in n_values:
            add_k = n_pairs - n_seed
            if add_k < 0:
                continue
            selected = seed_ids + perm[:add_k]
            train_df = learn[learn["config_id"].isin(selected)].copy()
            metrics = train_eval(
                train_df=train_df,
                val_df=val,
                mse_base=mse_base,
                regression_model=regression_model,
                ridge_alpha=ridge_alpha,
            )
            run_rows.append(
                {
                    "method": "random_addition",
                    "seed_state": seed_state,
                    "run": run_idx,
                    "n_pairs": n_pairs,
                    "mse": metrics.mse,
                    "rmse": metrics.rmse,
                    "r2_custom": metrics.r2_custom,
                    "train_n_rows": metrics.train_n_rows,
                    "train_rank": metrics.train_rank,
                    "train_cond": metrics.train_cond,
                }
            )

    run_df = pd.DataFrame(run_rows)
    agg_df = (
        run_df.groupby(["method", "seed_state", "n_pairs"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_p10=("rmse", lambda s: float(np.percentile(s, 10))),
            rmse_p90=("rmse", lambda s: float(np.percentile(s, 90))),
            r2_mean=("r2_custom", "mean"),
            r2_p10=("r2_custom", lambda s: float(np.percentile(s, 10))),
            r2_p90=("r2_custom", lambda s: float(np.percentile(s, 90))),
            train_rank_min=("train_rank", "min"),
            train_cond_median=(
                "train_cond",
                lambda s: float(np.median(s[np.isfinite(s)])) if np.isfinite(s).any() else float("inf"),
            ),
        )
    )
    return run_df, agg_df


def run_bo_curve(
    learn: pd.DataFrame,
    val: pd.DataFrame,
    seed_ids: list[str],
    n_values: list[int],
    random_state: int,
    mse_base: float,
    regression_model: str,
    ridge_alpha: float,
    acquisition: str,
    seed_state: int,
) -> pd.DataFrame:
    selected_ids = set(seed_ids)
    unselected_ids = set(learn["config_id"].unique()) - selected_ids
    max_n = max(n_values)

    rows: list[dict[str, Any]] = []
    current_n = len(selected_ids)

    while current_n <= max_n:
        train_df = learn[learn["config_id"].isin(selected_ids)].copy()
        current_metrics = train_eval(
            train_df=train_df,
            val_df=val,
            mse_base=mse_base,
            regression_model=regression_model,
            ridge_alpha=ridge_alpha,
        )

        if current_n in n_values:
            rows.append(
                {
                    "method": f"adaptive_bo_gp_{acquisition}",
                    "seed_state": seed_state,
                    "run": 0,
                    "n_pairs": current_n,
                    "mse": current_metrics.mse,
                    "rmse": current_metrics.rmse,
                    "r2_custom": current_metrics.r2_custom,
                    "train_n_rows": current_metrics.train_n_rows,
                    "train_rank": current_metrics.train_rank,
                    "train_cond": current_metrics.train_cond,
                }
            )

        if current_n == max_n:
            break

        pool_df = learn[learn["config_id"].isin(unselected_ids)].copy()
        shortlist = build_shortlist_by_ei(
            train_df=train_df,
            pool_df=pool_df,
            random_state=random_state,
            shortlist_k=len(pool_df),
        )

        if acquisition == "ei":
            selected_next = shortlist.sort_values(["bo_score_ei", "gp_std"], ascending=[False, False]).iloc[0][
                "config_id"
            ]
        elif acquisition == "max_variance":
            selected_next = shortlist.sort_values(["gp_std", "bo_score_ei"], ascending=[False, False]).iloc[0][
                "config_id"
            ]
        else:
            raise ValueError(f"Unsupported acquisition: {acquisition}")

        selected_ids.add(selected_next)
        unselected_ids.remove(selected_next)
        current_n += 1

    return pd.DataFrame(rows)


def init_llm_client(args: argparse.Namespace) -> tuple[Optional[Any], Optional[str]]:
    if LLMClient is None:
        return None, f"llm_client_import_error: {LLM_IMPORT_ERROR}"
    if args.llm_provider.lower() == "openai":
        if not get_env_var(args.llm_api_key_env):
            return None, f"missing_api_key_env:{args.llm_api_key_env}"
    try:
        client = LLMClient(
            provider=args.llm_provider,
            openai_model=args.llm_model,
            openai_api_key_env=args.llm_api_key_env,
        )
    except BaseException as exc:  # includes SystemExit from missing OPENAI_API_KEY
        return None, f"llm_client_init_error: {exc}"
    return client, None


def build_history_table(learn: pd.DataFrame, selected_order: list[str]) -> pd.DataFrame:
    order_map = {cid: i + 1 for i, cid in enumerate(selected_order)}
    hist = learn[learn["config_id"].isin(selected_order)][["config_id"] + FEATURES + [TARGET]].copy()
    hist["selection_order"] = hist["config_id"].map(order_map)
    hist = hist.sort_values("selection_order").reset_index(drop=True)
    hist["observed_treatment_effect"] = hist[TARGET] - hist["control_itt_efficiency"]
    cols = ["selection_order", "config_id"] + FEATURES + [TARGET, "observed_treatment_effect"]
    return hist[cols]


def call_llm_selection(
    llm_client: Any,
    prompt: str,
    args: argparse.Namespace,
) -> str:
    max_output_tokens = int(args.llm_max_output_tokens) if int(args.llm_max_output_tokens) > 0 else 12000
    messages_list = [
        [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    ]
    output = llm_client.generate_batch(
        prompts=None,
        messages_list=messages_list,
        stop=None,
        max_new_tokens=max_output_tokens,
        temperature=args.llm_temperature,
        top_p=args.llm_top_p,
        seed=args.random_state,
        async_openai=False,
        max_concurrency=1,
    )
    return str(output[0])


def run_llm_curve(
    learn: pd.DataFrame,
    val: pd.DataFrame,
    seed_ids: list[str],
    n_values: list[int],
    random_state: int,
    mse_base: float,
    regression_model: str,
    ridge_alpha: float,
    args: argparse.Namespace,
    seed_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], Optional[str]]:
    llm_client, llm_init_error = init_llm_client(args)
    llm_batch_size = max(1, int(args.llm_batch_size))

    selected_order = list(seed_ids)
    selected_set = set(selected_order)
    unselected_set = set(learn["config_id"].unique()) - selected_set

    max_n_global = max(n_values)
    max_n = max_n_global if args.llm_n_max <= 0 else min(max_n_global, int(args.llm_n_max))
    llm_n_values = [n for n in n_values if n <= max_n]
    if not llm_n_values:
        return pd.DataFrame(), pd.DataFrame(), [], llm_init_error

    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    prompt_records: list[dict[str, Any]] = []

    current_n = len(selected_order)
    while current_n <= max_n:
        train_df = learn[learn["config_id"].isin(selected_set)].copy()
        current_metrics = train_eval(
            train_df=train_df,
            val_df=val,
            mse_base=mse_base,
            regression_model=regression_model,
            ridge_alpha=ridge_alpha,
        )

        if current_n in llm_n_values:
            rows.append(
                {
                    "method": "adaptive_llm_rerank_gp_ei",
                    "seed_state": seed_state,
                    "run": 0,
                    "n_pairs": current_n,
                    "mse": current_metrics.mse,
                    "rmse": current_metrics.rmse,
                    "r2_custom": current_metrics.r2_custom,
                    "train_n_rows": current_metrics.train_n_rows,
                    "train_rank": current_metrics.train_rank,
                    "train_cond": current_metrics.train_cond,
                }
            )

        if current_n == max_n:
            break

        pool_df = learn[learn["config_id"].isin(unselected_set)].copy()
        shortlist = build_shortlist_by_ei(
            train_df=train_df,
            pool_df=pool_df,
            random_state=random_state,
            shortlist_k=args.llm_shortlist_k,
        )
        shortlist_ids = shortlist["config_id"].tolist()
        k_select = min(llm_batch_size, max_n - current_n, len(shortlist_ids))
        if k_select <= 0:
            break
        top_bo_score_ids = shortlist_ids[:k_select]

        history_df = build_history_table(learn=learn, selected_order=selected_order)
        prompt, history_rows_used, dropped_rows, prompt_error = llm_prompting.build_prompt_with_overflow(
            n_pairs=current_n,
            n_unselected=len(unselected_set),
            k_select=k_select,
            features=FEATURES,
            target=TARGET,
            history_df=history_df,
            shortlist_df=shortlist,
            max_chars=args.llm_max_context_chars,
            overflow_policy=args.llm_overflow_policy,
        )

        fallback_reason: Optional[str] = None
        raw_response = ""
        parsed: Optional[dict[str, Any]] = None

        if prompt_error:
            fallback_reason = prompt_error
        elif llm_client is None:
            fallback_reason = llm_init_error or "llm_unavailable"
        else:
            try:
                raw_response = call_llm_selection(
                    llm_client=llm_client,
                    prompt=prompt,
                    args=args,
                )
            except Exception as exc:
                fallback_reason = f"llm_api_error:{exc}"
            else:
                try:
                    parsed = llm_prompting.extract_json_dict(raw_response)
                except Exception:
                    fallback_reason = "llm_parse_error"

        api_model_reported = None if llm_client is None else getattr(llm_client, "last_openai_model_actual", None)
        selection = llm_prompting.finalize_llm_selection(
            parsed=parsed,
            shortlist_ids=shortlist_ids,
            top_bo_score_ids=top_bo_score_ids,
            k_select=k_select,
            raw_response=raw_response,
            fallback_reason=fallback_reason,
            normalize_config_id=normalize_config_id,
        )

        final_selected_ids = [cid for cid in selection.final_selected_config_ids if cid in unselected_set]
        if not final_selected_ids:
            final_selected_ids = top_bo_score_ids[:1]
            selection = llm_prompting.LLMSelectionResult(
                final_selected_config_ids=final_selected_ids,
                selected_config_ids_raw=selection.selected_config_ids_raw,
                confidence=selection.confidence,
                reasoning=selection.reasoning,
                fallback_reason=selection.fallback_reason or "empty_selection_after_validation",
                raw_response=selection.raw_response,
                parsed_response=selection.parsed_response,
            )

        for cid in final_selected_ids:
            selected_set.add(cid)
            selected_order.append(cid)
            unselected_set.remove(cid)

        decision_row = {
            "method": "adaptive_llm_rerank_gp_ei",
            "seed_state": seed_state,
            "n_pairs_before_selection": current_n,
            "requested_batch_size": k_select,
            "realized_batch_size": len(final_selected_ids),
            "selected_config_ids_final": json.dumps(final_selected_ids),
            "selected_config_ids_raw": json.dumps(selection.selected_config_ids_raw),
            "confidence": selection.confidence,
            "reasoning": selection.reasoning,
            "fallback_reason": selection.fallback_reason,
            "shortlist_top_bo_score_ids": json.dumps(top_bo_score_ids),
            "shortlist_ids": json.dumps(shortlist_ids),
            "prompt_chars": len(prompt) if prompt is not None else 0,
            "history_rows_used": history_rows_used,
            "dropped_history_rows": dropped_rows,
            "llm_init_error": llm_init_error,
            "llm_model_requested": args.llm_model,
            "llm_model_reported_by_api": api_model_reported,
            "current_rmse": current_metrics.rmse,
            "current_r2_custom": current_metrics.r2_custom,
        }
        decisions.append(decision_row)

        prompt_records.append(
            {
                "seed_state": seed_state,
                "n_pairs_before_selection": current_n,
                "requested_batch_size": k_select,
                "selected_config_ids_final": final_selected_ids,
                "fallback_reason": selection.fallback_reason,
                "prompt_chars": len(prompt) if prompt is not None else 0,
                "history_rows_used": history_rows_used,
                "dropped_history_rows": dropped_rows,
                "shortlist": shortlist[
                    ["ei_rank", "config_id", "gp_mu", "gp_std", "bo_score_ei"] + FEATURES
                ].to_dict(orient="records"),
                "prompt": prompt,
                "raw_response": selection.raw_response,
                "parsed_response": selection.parsed_response,
                "llm_model_requested": args.llm_model,
                "llm_model_reported_by_api": api_model_reported,
                "decision_row": decision_row,
            }
        )

        current_n += len(final_selected_ids)

    curve_df = pd.DataFrame(rows)
    decisions_df = pd.DataFrame(decisions)
    return curve_df, decisions_df, prompt_records, llm_init_error


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def infer_output_root(root: Path, args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)

    if args.seed_scope == "fixed42":
        run_name = (
            f"learning_curve_{args.regression_model}_random_bo_llm_"
            f"gp_{args.bo_acquisition}_rs42_"
            f"n{args.n_start}_to_{args.n_max}_step{args.step}_{args.seed_scope}"
        )
        return root / "benchmark_sequential" / "results" / run_name

    batch_name = (
        f"learning_curve_{args.regression_model}_random_bo_llm_"
        f"gp_{args.bo_acquisition}_rs42_to_61_"
        f"n{args.n_start}_to_{args.n_max}_step{args.step}_{args.seed_scope}"
    )
    return root / "benchmark_sequential" / "results" / batch_name


def attach_timestamp_run_id(base_dir: Path, args: argparse.Namespace) -> Path:
    # Keep explicit output_dir behavior unchanged; timestamp only for auto-generated paths.
    if args.output_dir:
        return base_dir
    base_run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = base_dir / base_run_id
    if not candidate.exists():
        return candidate

    idx = 2
    while True:
        candidate = base_dir / f"{base_run_id}_{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def infer_seed_run_dir(output_root: Path, args: argparse.Namespace, seed_state: int) -> Path:
    run_name = (
        f"learning_curve_{args.regression_model}_random_bo_llm_"
        f"gp_{args.bo_acquisition}_rs{seed_state}_"
        f"n{args.n_start}_to_{args.n_max}_step{args.step}_{args.seed_scope}"
    )
    if args.seed_scope == "fixed42":
        return output_root
    return output_root / "runs" / run_name


def run_single_seed(
    learn: pd.DataFrame,
    val: pd.DataFrame,
    n_values: list[int],
    seed_ids: list[str],
    seed_state: int,
    mse_base: float,
    full_anchor: Metrics,
    methods: list[str],
    requested_methods: list[str],
    llm_skipped_reason: Optional[str],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    random_run_df: Optional[pd.DataFrame] = None
    random_agg_df: Optional[pd.DataFrame] = None
    bo_df: Optional[pd.DataFrame] = None
    llm_df: Optional[pd.DataFrame] = None
    llm_decisions_df = pd.DataFrame()
    llm_prompt_records: list[dict[str, Any]] = []
    llm_init_error: Optional[str] = None

    method_curves: dict[str, pd.DataFrame] = {}

    if "random" in methods:
        random_run_df, random_agg_df = run_random_curve(
            learn=learn,
            val=val,
            seed_ids=seed_ids,
            n_values=n_values,
            random_runs=args.random_runs,
            random_state=args.random_state,
            mse_base=mse_base,
            regression_model=args.regression_model,
            ridge_alpha=args.ridge_alpha,
            seed_state=seed_state,
        )
        random_run_df.to_csv(out_dir / "random_runs_raw.csv", index=False)
        random_agg_df.to_csv(out_dir / "random_runs_aggregated.csv", index=False)

    if "bo" in methods:
        bo_df = run_bo_curve(
            learn=learn,
            val=val,
            seed_ids=seed_ids,
            n_values=n_values,
            random_state=args.random_state,
            mse_base=mse_base,
            regression_model=args.regression_model,
            ridge_alpha=args.ridge_alpha,
            acquisition=args.bo_acquisition,
            seed_state=seed_state,
        )
        bo_df.to_csv(out_dir / "adaptive_bo_gp_ei_curve.csv", index=False)
        method_curves["bo"] = bo_df

    if "llm" in methods:
        llm_df, llm_decisions_df, llm_prompt_records, llm_init_error = run_llm_curve(
            learn=learn,
            val=val,
            seed_ids=seed_ids,
            n_values=n_values,
            random_state=args.random_state,
            mse_base=mse_base,
            regression_model=args.regression_model,
            ridge_alpha=args.ridge_alpha,
            args=args,
            seed_state=seed_state,
        )
        llm_df.to_csv(out_dir / "adaptive_llm_rerank_gp_ei_curve.csv", index=False)
        llm_decisions_df.to_csv(out_dir / "llm_decisions.csv", index=False)
        save_jsonl(llm_prompt_records, out_dir / "llm_prompts_responses.jsonl")
        method_curves["llm"] = llm_df

    plot_data = plot_utils.collect_seed_plot_data(
        n_values=n_values,
        random_agg=random_agg_df,
        bo_df=bo_df,
        llm_df=llm_df,
        full_anchor_rmse=full_anchor.rmse,
        full_anchor_r2=full_anchor.r2_custom,
    )
    plot_data.to_csv(out_dir / "learning_curve_plot_data.csv", index=False)

    plot_utils.make_plot(
        random_agg=random_agg_df,
        method_curves=method_curves,
        full_anchor_rmse=full_anchor.rmse,
        full_anchor_r2=full_anchor.r2_custom,
        bo_acquisition=args.bo_acquisition,
        out_path=out_dir / "learning_curve_rmse_r2_all_methods.png",
        zoom=False,
    )
    plot_utils.make_plot(
        random_agg=random_agg_df,
        method_curves=method_curves,
        full_anchor_rmse=full_anchor.rmse,
        full_anchor_r2=full_anchor.r2_custom,
        bo_acquisition=args.bo_acquisition,
        out_path=out_dir / "learning_curve_rmse_r2_all_methods_zoom.png",
        zoom=True,
    )

    run_summary = {
        "seed_state": seed_state,
        "seed_ids": seed_ids,
        "n_values": n_values,
        "step": args.step,
        "requested_methods": requested_methods,
        "methods": methods,
        "llm_skipped_reason": llm_skipped_reason,
        "random_runs": args.random_runs if "random" in methods else 0,
        "features": FEATURES,
        "target": TARGET,
        "regression_model": args.regression_model,
        "ridge_alpha": args.ridge_alpha if args.regression_model == "ridge" else None,
        "bo_acquisition": args.bo_acquisition,
        "llm": {
            "provider": args.llm_provider,
            "model": args.llm_model,
            "shortlist_k": args.llm_shortlist_k,
            "batch_size": args.llm_batch_size,
            "n_max": args.llm_n_max,
            "temperature": args.llm_temperature,
            "top_p": args.llm_top_p,
            "max_output_tokens": args.llm_max_output_tokens,
            "max_context_chars": args.llm_max_context_chars,
            "overflow_policy": args.llm_overflow_policy,
            "init_error": llm_init_error,
        },
        "baseline_mse_zero_treatment_effect": mse_base,
        "full_data_anchor": {
            "mse": full_anchor.mse,
            "rmse": full_anchor.rmse,
            "r2_custom": full_anchor.r2_custom,
        },
        "output_dir": str(out_dir),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    curve_frames = []
    if random_agg_df is not None:
        tmp = random_agg_df[["n_pairs", "rmse_mean", "r2_mean"]].copy()
        tmp = tmp.rename(columns={"rmse_mean": "rmse", "r2_mean": "r2_custom"})
        tmp["method"] = "random_addition_mean"
        tmp["seed_state"] = seed_state
        curve_frames.append(tmp[["seed_state", "method", "n_pairs", "rmse", "r2_custom"]])
    if bo_df is not None:
        tmp = bo_df[["n_pairs", "rmse", "r2_custom"]].copy()
        tmp["method"] = "adaptive_bo_gp_ei"
        tmp["seed_state"] = seed_state
        curve_frames.append(tmp[["seed_state", "method", "n_pairs", "rmse", "r2_custom"]])
    if llm_df is not None:
        tmp = llm_df[["n_pairs", "rmse", "r2_custom"]].copy()
        tmp["method"] = "adaptive_llm_rerank_gp_ei"
        tmp["seed_state"] = seed_state
        curve_frames.append(tmp[["seed_state", "method", "n_pairs", "rmse", "r2_custom"]])

    anchor = pd.DataFrame(
        {
            "seed_state": seed_state,
            "method": "full_data_anchor",
            "n_pairs": n_values,
            "rmse": [full_anchor.rmse] * len(n_values),
            "r2_custom": [full_anchor.r2_custom] * len(n_values),
        }
    )
    curve_frames.append(anchor)

    return {
        "seed_state": seed_state,
        "out_dir": str(out_dir),
        "curve_df": pd.concat(curve_frames, ignore_index=True),
        "run_summary": run_summary,
    }


def run_batch_aggregation(
    seed_results: list[dict[str, Any]],
    output_root: Path,
    full_anchor: Metrics,
    args: argparse.Namespace,
    methods: list[str],
    requested_methods: list[str],
    llm_skipped_reason: Optional[str],
) -> None:
    all_curves = pd.concat([r["curve_df"] for r in seed_results], ignore_index=True)

    method_curve = all_curves[all_curves["method"] != "full_data_anchor"].copy()
    agg = (
        method_curve.groupby(["method", "n_pairs"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_median=("rmse", "median"),
            rmse_p10=("rmse", lambda s: float(np.percentile(s, 10))),
            rmse_p90=("rmse", lambda s: float(np.percentile(s, 90))),
            r2_mean=("r2_custom", "mean"),
            r2_median=("r2_custom", "median"),
            r2_p10=("r2_custom", lambda s: float(np.percentile(s, 10))),
            r2_p90=("r2_custom", lambda s: float(np.percentile(s, 90))),
        )
    )
    agg.to_csv(output_root / "batch_aggregated_curves.csv", index=False)

    plot_utils.make_batch_plot(
        agg_df=agg,
        full_anchor_rmse=full_anchor.rmse,
        full_anchor_r2=full_anchor.r2_custom,
        out_path=output_root / "batch_plots_rmse_r2_all_methods.png",
        zoom=False,
    )
    plot_utils.make_batch_plot(
        agg_df=agg,
        full_anchor_rmse=full_anchor.rmse,
        full_anchor_r2=full_anchor.r2_custom,
        out_path=output_root / "batch_plots_rmse_r2_all_methods_zoom.png",
        zoom=True,
    )

    batch_summary = {
        "seed_scope": args.seed_scope,
        "seed_states": [int(r["seed_state"]) for r in seed_results],
        "requested_methods": requested_methods,
        "methods": methods,
        "llm_skipped_reason": llm_skipped_reason,
        "n_values": build_n_values(args.n_start, args.n_max, args.step),
        "regression_model": args.regression_model,
        "ridge_alpha": args.ridge_alpha if args.regression_model == "ridge" else None,
        "bo_acquisition": args.bo_acquisition,
        "llm_model": args.llm_model,
        "llm_shortlist_k": args.llm_shortlist_k,
        "llm_batch_size": args.llm_batch_size,
        "llm_n_max": args.llm_n_max,
        "random_runs": args.random_runs if "random" in methods else 0,
        "full_data_anchor": {
            "mse": full_anchor.mse,
            "rmse": full_anchor.rmse,
            "r2_custom": full_anchor.r2_custom,
        },
        "run_dirs": [r["out_dir"] for r in seed_results],
        "output_dir": str(output_root),
    }
    (output_root / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.llm_batch_size <= 0:
        raise ValueError("--llm-batch-size must be >= 1")
    if args.llm_n_max < 0:
        raise ValueError("--llm-n-max must be >= 0")
    requested_methods = parse_methods(args.methods)
    methods, llm_skipped_reason = resolve_methods_for_env(requested_methods, args)
    if not methods:
        raise ValueError("No runnable methods remain after environment checks.")

    root = Path(__file__).resolve().parents[1]
    learn_path = root / "benchmark_sequential" / "data" / "processed_data" / "df_paired_learn.csv"
    val_path = root / "benchmark_sequential" / "data" / "processed_data" / "df_paired_val.csv"
    seed_summary_path = (
        root
        / "benchmark_sequential"
        / "data_seed_random_learning_paired300_pairs10_rs42"
        / "seed_selection_summary.json"
    )

    output_root_base = infer_output_root(root, args)
    output_root = attach_timestamp_run_id(output_root_base, args)
    output_root.mkdir(parents=True, exist_ok=True)

    n_values = build_n_values(args.n_start, args.n_max, args.step)

    learn = pd.read_csv(learn_path)
    val = pd.read_csv(val_path)
    learn = cast_bool_features(learn, FEATURES)
    val = cast_bool_features(val, FEATURES)

    for col in FEATURES + [TARGET, "CONFIG_configId"]:
        if col not in learn.columns or col not in val.columns:
            raise ValueError(f"Missing required column `{col}` in paired learn/val tables")

    learn["config_id"] = learn["CONFIG_configId"].apply(normalize_config_id)
    val["config_id"] = val["CONFIG_configId"].apply(normalize_config_id)

    if learn["config_id"].nunique() != len(learn):
        raise ValueError("df_paired_learn.csv must have one row per CONFIG_configId pair")

    all_ids = sorted(learn["config_id"].tolist(), key=config_id_sort_key)

    seed_summary = json.loads(seed_summary_path.read_text(encoding="utf-8"))
    fixed_seed_ids = sorted(
        [normalize_config_id(x) for x in seed_summary["seed_pair_bases"]],
        key=config_id_sort_key,
    )
    if len(fixed_seed_ids) != 10:
        raise ValueError(f"Expected 10 seed ids in seed summary, found {len(fixed_seed_ids)}")

    if args.n_start < len(fixed_seed_ids):
        raise ValueError(f"n-start must be >= seed size ({len(fixed_seed_ids)}).")
    if args.n_max > len(all_ids):
        raise ValueError("n-max cannot exceed number of available paired learning rows.")

    seed_sets = build_seed_sets(
        seed_scope=args.seed_scope,
        all_ids=all_ids,
        fixed_seed_ids=fixed_seed_ids,
    )

    mse_base = float(mean_squared_error(val[TARGET], val["control_itt_efficiency"]))
    full_anchor = train_eval(
        train_df=learn.copy(),
        val_df=val,
        mse_base=mse_base,
        regression_model=args.regression_model,
        ridge_alpha=args.ridge_alpha,
    )

    seed_results: list[dict[str, Any]] = []

    for seed_state in sorted(seed_sets.keys()):
        seed_ids = seed_sets[seed_state]
        run_out_dir = infer_seed_run_dir(output_root=output_root, args=args, seed_state=seed_state)
        result = run_single_seed(
            learn=learn,
            val=val,
            n_values=n_values,
            seed_ids=seed_ids,
            seed_state=seed_state,
            mse_base=mse_base,
            full_anchor=full_anchor,
            methods=methods,
            requested_methods=requested_methods,
            llm_skipped_reason=llm_skipped_reason,
            args=args,
            out_dir=run_out_dir,
        )
        seed_results.append(result)

    if args.seed_scope == "multi20":
        run_batch_aggregation(
            seed_results=seed_results,
            output_root=output_root,
            full_anchor=full_anchor,
            args=args,
            methods=methods,
            requested_methods=requested_methods,
            llm_skipped_reason=llm_skipped_reason,
        )

    final_summary = {
        "seed_scope": args.seed_scope,
        "seed_states": sorted(seed_sets.keys()),
        "requested_methods": requested_methods,
        "methods": methods,
        "llm_skipped_reason": llm_skipped_reason,
        "n_values": n_values,
        "output_root": str(output_root),
        "per_seed_run_dirs": [r["out_dir"] for r in seed_results],
    }
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
