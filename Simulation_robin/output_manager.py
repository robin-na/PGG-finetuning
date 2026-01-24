from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import pandas as pd

from utils import log, relocate_output, safe_name, timestamp_yymmddhhmm


USED_CONFIG_KEYS = {
    "CONFIG_allOrNothing",
    "CONFIG_chat",
    "CONFIG_defaultContribProp",
    "CONFIG_endowment",
    "CONFIG_multiplier",
    "CONFIG_numRounds",
    "CONFIG_playerCount",
    "CONFIG_punishmentCost",
    "CONFIG_punishmentExists",
    "CONFIG_punishmentMagnitude",
    "CONFIG_rewardCost",
    "CONFIG_rewardExists",
    "CONFIG_rewardMagnitude",
    "CONFIG_showNRounds",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
}


def resolve_run_ts(run_id: Optional[str]) -> str:
    return run_id or timestamp_yymmddhhmm()


def resolve_experiment_dir(output_root: str, experiment_name: str, run_ts: str) -> str:
    return os.path.join(output_root, safe_name(experiment_name), run_ts)


def _config_from_env(env: pd.Series) -> Dict[str, Any]:
    return {k: env[k] for k in USED_CONFIG_KEYS if k in env.index}


def _model_config(args: Any, run_ts: str) -> Dict[str, Any]:
    args_dict = asdict(args) if is_dataclass(args) else dict(args)
    provider = args_dict.get("provider")

    base_keys = [
        "provider",
        "openai_model",
        "openai_api_key_env",
        "openai_async",
        "openai_max_concurrency",
        "temperature",
        "top_p",
        "seed",
        "contrib_max_new_tokens",
        "chat_max_new_tokens",
        "actions_max_new_tokens",
        "include_reasoning",
        "max_parallel_games",
        "debug_print",
        "env_csv"
    ]
    if provider != "openai":
        base_keys.extend(["base_model", "adapter_path", "use_peft"])

    model_config = {k: args_dict.get(k) for k in base_keys if k in args_dict}
    model_config["run_timestamp"] = run_ts
    return model_config


def write_config(experiment_dir: str, env: pd.Series, args: Any, run_ts: str) -> str:
    os.makedirs(experiment_dir, exist_ok=True)
    payload = {
        "experiment": env.get("name", safe_name(env.get("id", "GAME"))),
        "run_timestamp": run_ts,
        "environment": _config_from_env(env),
        "model": _model_config(args, run_ts),
    }
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    log(f"[ptc] saved config JSON → {config_path}")
    return config_path


def relocate_for_experiment(path: Optional[str], experiment_dir: str) -> Optional[str]:
    return relocate_output(path, experiment_dir)
