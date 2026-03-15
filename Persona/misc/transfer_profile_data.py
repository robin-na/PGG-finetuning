#!/usr/bin/env python3
"""Build direct-from-raw-data PGG transfer profiles for learning and validation waves."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from Persona.misc.build_person_cards_noncomm import (
        _action_space_expected,
        _action_space_observed,
        _clean_scalar,
        _conditionality,
        _coop_baseline_level,
        _endgame_shift,
        _ensure_jsonable,
        _hhi,
        _intensity_level,
        _mean,
        _median,
        _parse_round_numeric,
        _parse_units_dict,
        _punish_propensity_level,
        _response_style,
        _reward_response_style,
        _std,
        _summarize_transfer,
        _volatility_level,
    )
except ImportError:
    from build_person_cards_noncomm import (  # type: ignore
        _action_space_expected,
        _action_space_observed,
        _clean_scalar,
        _conditionality,
        _coop_baseline_level,
        _endgame_shift,
        _ensure_jsonable,
        _hhi,
        _intensity_level,
        _mean,
        _median,
        _parse_round_numeric,
        _parse_units_dict,
        _punish_propensity_level,
        _response_style,
        _reward_response_style,
        _std,
        _summarize_transfer,
        _volatility_level,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FIELDS = [
    "CONFIG_configId",
    "CONFIG_playerCount",
    "CONFIG_numRounds",
    "CONFIG_showNRounds",
    "CONFIG_endowment",
    "CONFIG_multiplier",
    "CONFIG_allOrNothing",
    "CONFIG_chat",
    "CONFIG_defaultContribProp",
    "CONFIG_punishmentExists",
    "CONFIG_punishmentCost",
    "CONFIG_punishmentMagnitude",
    "CONFIG_rewardExists",
    "CONFIG_rewardCost",
    "CONFIG_rewardMagnitude",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
]

FIELD_TYPES = {
    "CONFIG_playerCount": "int",
    "CONFIG_numRounds": "int",
    "CONFIG_endowment": "int",
    "CONFIG_showNRounds": "bool",
    "CONFIG_allOrNothing": "bool",
    "CONFIG_chat": "bool",
    "CONFIG_punishmentExists": "bool",
    "CONFIG_rewardExists": "bool",
    "CONFIG_showOtherSummaries": "bool",
    "CONFIG_showPunishmentId": "bool",
    "CONFIG_showRewardId": "bool",
    "CONFIG_multiplier": "float",
    "CONFIG_defaultContribProp": "float",
    "CONFIG_punishmentCost": "float",
    "CONFIG_punishmentMagnitude": "float",
    "CONFIG_rewardCost": "float",
    "CONFIG_rewardMagnitude": "float",
}

SPLIT_PATHS = {
    "learn": {
        "player_rounds": PROJECT_ROOT / "data" / "raw_data" / "learning_wave" / "player-rounds.csv",
        "config": PROJECT_ROOT / "data" / "processed_data" / "df_analysis_learn.csv",
    },
    "val": {
        "player_rounds": PROJECT_ROOT / "data" / "raw_data" / "validation_wave" / "player-rounds.csv",
        "config": PROJECT_ROOT / "data" / "processed_data" / "df_analysis_val.csv",
    },
}


def _load_config_map(config_path: Path) -> Dict[str, Dict]:
    df_config = pd.read_csv(config_path)
    config_map: Dict[str, Dict] = {}
    for _, row in df_config.iterrows():
        game_id = str(row.get("gameId"))
        config = {}
        for field in CONFIG_FIELDS:
            if field not in df_config.columns:
                config[field] = None
                continue
            config[field] = _clean_scalar(row.get(field), desired_type=FIELD_TYPES.get(field))
        if config.get("CONFIG_punishmentExists") is False:
            config["CONFIG_punishmentCost"] = None
            config["CONFIG_punishmentMagnitude"] = None
        if config.get("CONFIG_rewardExists") is False:
            config["CONFIG_rewardCost"] = None
            config["CONFIG_rewardMagnitude"] = None
        config_map[game_id] = config
    return config_map


def _prepare_player_frame(player_rounds_path: Path) -> pd.DataFrame:
    df_player = pd.read_csv(player_rounds_path)
    df_player = df_player.reset_index().rename(columns={"index": "row_order"})
    df_player["gameId_key"] = df_player["gameId"].astype(str)
    df_player["playerId_key"] = df_player["playerId"].astype(str)
    df_player["roundId_key"] = df_player["roundId"].astype(str)
    df_player["punished_dict"] = df_player["data.punished"].apply(_parse_units_dict)
    df_player["punishedBy_dict"] = df_player["data.punishedBy"].apply(_parse_units_dict)
    df_player["rewarded_dict"] = df_player["data.rewarded"].apply(_parse_units_dict)
    df_player["rewardedBy_dict"] = df_player["data.rewardedBy"].apply(_parse_units_dict)
    return df_player


def _build_round_views(df_player: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, Dict[str, int]], Dict[Tuple[str, str], Dict], Dict[str, Dict]]:
    round_order_by_game: Dict[str, List[str]] = {}
    round_order_source_by_game: Dict[str, str] = {}
    round_index_map_by_game: Dict[str, Dict[str, int]] = {}

    for game_id, game_df in df_player.groupby("gameId_key", sort=False):
        round_ids_appearance: List[str] = []
        seen = set()
        for rid in game_df["roundId_key"]:
            if rid not in seen:
                seen.add(rid)
                round_ids_appearance.append(rid)
        numeric_pairs = []
        all_numeric = True
        for rid in round_ids_appearance:
            num = _parse_round_numeric(rid)
            if num is None:
                all_numeric = False
                break
            numeric_pairs.append((rid, num))
        if all_numeric:
            appearance_index = {rid: idx for idx, rid in enumerate(round_ids_appearance)}
            numeric_pairs.sort(key=lambda x: (x[1], appearance_index[x[0]]))
            ordered_rounds = [rid for rid, _ in numeric_pairs]
            round_order_source_by_game[game_id] = "roundId_numeric"
        else:
            ordered_rounds = round_ids_appearance
            round_order_source_by_game[game_id] = "appearance"
        round_order_by_game[game_id] = ordered_rounds
        round_index_map_by_game[game_id] = {rid: idx + 1 for idx, rid in enumerate(ordered_rounds)}

    round_info: Dict[Tuple[str, str], Dict] = {}
    for game_id, ordered_rounds in round_order_by_game.items():
        game_df = df_player[df_player["gameId_key"] == game_id]
        for rid in ordered_rounds:
            round_df = game_df[game_df["roundId_key"] == rid]
            contribs = {}
            contrib_list = []
            for _, row in round_df.iterrows():
                pid = row["playerId_key"]
                c = row.get("data.contribution")
                if c is None or (isinstance(c, float) and math.isnan(c)):
                    contribs[pid] = None
                else:
                    contribs[pid] = float(c)
                    contrib_list.append(float(c))
            active_players = int(len(round_df))
            group_total = float(np.sum(contrib_list)) if contrib_list else None
            group_mean = _mean(contrib_list)
            group_median = _median(contrib_list)
            others_mean = {}
            others_median = {}
            for pid, _ in contribs.items():
                others_vals = [v for opid, v in contribs.items() if opid != pid and v is not None]
                if others_vals:
                    others_mean[pid] = float(np.mean(np.array(others_vals, dtype=float)))
                    others_median[pid] = _median(others_vals)
                else:
                    others_mean[pid] = None
                    others_median[pid] = None
            round_info[(game_id, rid)] = {
                "active_players": active_players,
                "contribs": contribs,
                "group_total": group_total,
                "group_mean": group_mean,
                "group_median": group_median,
                "others_mean": others_mean,
                "others_median": others_median,
            }

    game_group_sizes: Dict[str, Dict] = {}
    for game_id, ordered_rounds in round_order_by_game.items():
        active_counts = [round_info[(game_id, rid)]["active_players"] for rid in ordered_rounds]
        game_group_sizes[game_id] = {
            "mean_active": float(np.mean(active_counts)) if active_counts else None,
            "min_active": int(np.min(active_counts)) if active_counts else None,
            "max_active": int(np.max(active_counts)) if active_counts else None,
        }

    return (
        round_order_by_game,
        round_order_source_by_game,
        round_index_map_by_game,
        round_info,
        game_group_sizes,
    )


def build_raw_profiles_for_split(split: str) -> List[Dict]:
    paths = SPLIT_PATHS[split]
    config_map = _load_config_map(paths["config"])
    df_player = _prepare_player_frame(paths["player_rounds"])
    (
        round_order_by_game,
        round_order_source_by_game,
        round_index_map_by_game,
        round_info,
        game_group_sizes,
    ) = _build_round_views(df_player)

    output_records: List[Dict] = []

    for (game_id, player_id), group_df in df_player.groupby(["gameId_key", "playerId_key"], sort=False):
        config = config_map.get(game_id, {field: None for field in CONFIG_FIELDS})
        punishment_enabled = config.get("CONFIG_punishmentExists") is True
        reward_enabled = config.get("CONFIG_rewardExists") is True
        show_n_rounds = config.get("CONFIG_showNRounds") is True
        endowment = config.get("CONFIG_endowment")

        round_index_map = round_index_map_by_game.get(game_id, {})
        round_order_source = round_order_source_by_game.get(game_id, "unknown")
        group_df = group_df.copy()
        group_df["round_index"] = group_df["roundId_key"].map(round_index_map)
        group_df = group_df.sort_values("round_index")

        contribs: List[float | None] = []
        round_ids: List[str] = []
        round_indices: List[int | None] = []
        punished_dicts = []
        punished_by_dicts = []
        rewarded_dicts = []
        rewarded_by_dicts = []
        costs = []
        penalties = []
        rewards = []
        remaining_endowment = []
        round_payoffs = []
        others_mean_series = []

        for _, row in group_df.iterrows():
            c = row.get("data.contribution")
            contribs.append(None if c is None or (isinstance(c, float) and math.isnan(c)) else float(c))
            rid = row["roundId_key"]
            round_ids.append(rid)
            round_indices.append(int(row.get("round_index")) if row.get("round_index") is not None else None)
            punished_dicts.append(row["punished_dict"])
            punished_by_dicts.append(row["punishedBy_dict"])
            rewarded_dicts.append(row["rewarded_dict"])
            rewarded_by_dicts.append(row["rewardedBy_dict"])
            costs.append(row.get("data.costs"))
            penalties.append(row.get("data.penalties"))
            rewards.append(row.get("data.rewards"))
            remaining_endowment.append(row.get("data.remainingEndowment"))
            round_payoffs.append(row.get("data.roundPayoff"))
            info = round_info.get((game_id, rid), {})
            others_mean_series.append(info.get("others_mean", {}).get(player_id))

        missing_round_indices = [
            round_indices[idx]
            for idx, value in enumerate(contribs)
            if value is None and round_indices[idx] is not None
        ]
        played_to_end = len(missing_round_indices) == 0
        has_missing_contribution_before_final_round = any(
            idx is not None and idx < len(round_ids) for idx in missing_round_indices
        )

        num_rounds_observed = len(round_ids)
        contribs_nonnull = [c for c in contribs if c is not None]
        contrib_mean = _mean(contribs_nonnull)
        contrib_min = min(contribs_nonnull) if contribs_nonnull else None
        contrib_max = max(contribs_nonnull) if contribs_nonnull else None
        contrib_median = _median(contribs_nonnull)
        contrib_std = _std(contribs_nonnull)

        always_constant = None
        always_max = None
        always_min = None
        if contribs_nonnull:
            always_constant = len(set(contribs_nonnull)) == 1
            if endowment is not None:
                always_max = all(c == endowment for c in contribs_nonnull)
                always_min = all(c == 0 for c in contribs_nonnull)

        switch_rate = None
        large_jump_rate = None
        if num_rounds_observed >= 2:
            switches = 0
            large_jumps = 0
            for idx in range(1, num_rounds_observed):
                if contribs[idx] is None or contribs[idx - 1] is None:
                    continue
                if contribs[idx] != contribs[idx - 1]:
                    switches += 1
                if endowment is not None and abs(contribs[idx] - contribs[idx - 1]) >= math.ceil(endowment / 2):
                    large_jumps += 1
            switch_rate = switches / (num_rounds_observed - 1)
            if endowment is not None:
                large_jump_rate = large_jumps / (num_rounds_observed - 1)

        contrib_summary = {
            "mean": contrib_mean,
            "min": contrib_min,
            "max": contrib_max,
            "median": contrib_median,
            "std": contrib_std,
            "always_constant": always_constant,
            "always_max": always_max,
            "always_min": always_min,
            "switch_rate": switch_rate,
            "large_jump_rate": large_jump_rate,
        }

        if punishment_enabled:
            total_units, rounds_used, units_by_target, _ = _summarize_transfer(list(zip(round_ids, punished_dicts)))
            punishment_given = {
                "used_any": total_units > 0,
                "rounds_used": rounds_used,
                "total_units": int(total_units),
                "total_cost_implied": total_units * config.get("CONFIG_punishmentCost")
                if config.get("CONFIG_punishmentCost") is not None
                else None,
                "total_harm_implied": total_units * config.get("CONFIG_punishmentMagnitude")
                if config.get("CONFIG_punishmentMagnitude") is not None
                else None,
                "mean_units_per_round": total_units / num_rounds_observed if num_rounds_observed else None,
                "mean_units_when_used": total_units / len(rounds_used) if rounds_used else None,
                "unique_targets_count": len(units_by_target),
                "target_units_hhi": _hhi(units_by_target),
            }
            total_units_in, rounds_received, units_by_source, _ = _summarize_transfer(
                list(zip(round_ids, punished_by_dicts))
            )
            punishment_received = {
                "received_any": total_units_in > 0,
                "rounds_received": rounds_received,
                "total_units": int(total_units_in),
                "total_penalty_implied": total_units_in * config.get("CONFIG_punishmentMagnitude")
                if config.get("CONFIG_punishmentMagnitude") is not None
                else None,
                "mean_units_per_round": total_units_in / num_rounds_observed if num_rounds_observed else None,
                "mean_units_when_received": total_units_in / len(rounds_received) if rounds_received else None,
                "unique_sources_count": len(units_by_source),
                "source_units_hhi": _hhi(units_by_source),
            }
        else:
            punishment_given = None
            punishment_received = None

        if reward_enabled:
            total_units_r, rounds_used_r, units_by_target_r, _ = _summarize_transfer(list(zip(round_ids, rewarded_dicts)))
            reward_given = {
                "used_any": total_units_r > 0,
                "rounds_used": rounds_used_r,
                "total_units": int(total_units_r),
                "total_cost_implied": total_units_r * config.get("CONFIG_rewardCost")
                if config.get("CONFIG_rewardCost") is not None
                else None,
                "total_benefit_implied": total_units_r * config.get("CONFIG_rewardMagnitude")
                if config.get("CONFIG_rewardMagnitude") is not None
                else None,
                "mean_units_per_round": total_units_r / num_rounds_observed if num_rounds_observed else None,
                "mean_units_when_used": total_units_r / len(rounds_used_r) if rounds_used_r else None,
                "unique_targets_count": len(units_by_target_r),
                "target_units_hhi": _hhi(units_by_target_r),
            }
            total_units_in_r, rounds_received_r, units_by_source_r, _ = _summarize_transfer(
                list(zip(round_ids, rewarded_by_dicts))
            )
            reward_received = {
                "received_any": total_units_in_r > 0,
                "rounds_received": rounds_received_r,
                "total_units": int(total_units_in_r),
                "total_bonus_implied": total_units_in_r * config.get("CONFIG_rewardMagnitude")
                if config.get("CONFIG_rewardMagnitude") is not None
                else None,
                "mean_units_per_round": total_units_in_r / num_rounds_observed if num_rounds_observed else None,
                "mean_units_when_received": total_units_in_r / len(rounds_received_r) if rounds_received_r else None,
                "unique_sources_count": len(units_by_source_r),
                "source_units_hhi": _hhi(units_by_source_r),
            }
        else:
            reward_given = None
            reward_received = None

        def _safe_vals(vals: Iterable) -> List[float]:
            cleaned = []
            for value in vals:
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                cleaned.append(float(value))
            return cleaned

        payoffs = {
            "round_payoff_mean": _mean(_safe_vals(round_payoffs)),
            "round_payoff_std": _std(_safe_vals(round_payoffs)),
            "costs_mean": _mean(_safe_vals(costs)),
            "penalties_mean": _mean(_safe_vals(penalties)),
            "rewards_mean": _mean(_safe_vals(rewards)),
            "remaining_endowment_mean": _mean(_safe_vals(remaining_endowment)),
        }

        action_space_expected = _action_space_expected(config.get("CONFIG_allOrNothing"), endowment)
        action_space_observed = _action_space_observed(contribs_nonnull, endowment)
        coop_level = _coop_baseline_level(action_space_observed, endowment, contrib_mean, always_max, always_min)
        volatility_level = _volatility_level(switch_rate, num_rounds_observed)

        module_base = {
            "coop_baseline": {
                "level": coop_level,
                "evidence": {
                    "mean_contribution": contrib_mean,
                    "mean_contribution_frac_of_endowment": contrib_mean / endowment if contrib_mean is not None and endowment else None,
                    "first_round_contribution": contribs[0] if contribs else None,
                },
            },
            "volatility": {
                "level": volatility_level,
                "evidence": {"switch_rate": switch_rate, "std_contribution": contrib_std},
            },
            "endgame_shift": _endgame_shift(contribs_nonnull, show_n_rounds, endowment, always_constant),
        }

        conditionality = {
            "responds_to_others_mean": _conditionality(contribs, [None] + others_mean_series[:-1], always_constant)
        }

        punishment_module = None
        if punishment_enabled and punishment_given is not None:
            punish_any_rate = len(punishment_given["rounds_used"]) / num_rounds_observed if num_rounds_observed else None
            mean_units_when_used = punishment_given["mean_units_when_used"]
            p75_units_when_used = None
            if punishment_given["rounds_used"]:
                units_used = [sum(d.values()) for d in punished_dicts if sum(d.values()) > 0]
                if units_used:
                    p75_units_when_used = float(np.percentile(units_used, 75))

            low_units = 0
            retaliate_units = 0
            for idx, rid in enumerate(round_ids):
                outgoing = punished_dicts[idx]
                if not outgoing:
                    continue
                info = round_info.get((game_id, rid), {})
                round_median = info.get("group_median")
                contribs_map = info.get("contribs", {})
                for target, units in outgoing.items():
                    target_contrib = contribs_map.get(str(target))
                    if target_contrib is not None and round_median is not None and target_contrib < round_median:
                        low_units += units
                if idx >= 1:
                    prev_punishers = set(punished_by_dicts[idx - 1].keys())
                    retaliate_units += sum(units for target, units in outgoing.items() if target in prev_punishers)

            delta_next = []
            punish_back = []
            for idx in range(len(round_ids) - 1):
                incoming = punished_by_dicts[idx]
                if sum(incoming.values()) <= 0:
                    continue
                if contribs[idx] is None or contribs[idx + 1] is None:
                    continue
                delta_next.append(contribs[idx + 1] - contribs[idx])
                sources = set(incoming.keys())
                next_out = punished_dicts[idx + 1]
                punish_back.append(bool(sources.intersection(set(next_out.keys()))))

            punishment_module = {
                "punish_propensity": {
                    "level": _punish_propensity_level(punish_any_rate),
                    "evidence": {
                        "punish_any_rate": punish_any_rate,
                        "mean_units_per_round": punishment_given["mean_units_per_round"],
                    },
                },
                "intensity_style": {
                    "level": _intensity_level(mean_units_when_used),
                    "evidence": {
                        "mean_units_when_used": mean_units_when_used,
                        "p75_units_when_used": p75_units_when_used,
                    },
                },
                "targeting_rule": {
                    "style": "low_contributor"
                    if punishment_given["total_units"] and low_units / punishment_given["total_units"] >= 0.7
                    else "retaliation"
                    if punishment_given["total_units"] and retaliate_units / punishment_given["total_units"] >= 0.4
                    else "unknown",
                    "evidence": {
                        "low_target_fraction": low_units / punishment_given["total_units"] if punishment_given["total_units"] else None,
                        "retaliation_fraction": retaliate_units / punishment_given["total_units"] if punishment_given["total_units"] else None,
                        "unique_targets_count": punishment_given["unique_targets_count"],
                        "target_units_hhi": punishment_given["target_units_hhi"],
                    },
                },
                "response_when_punished": {
                    "style": _response_style(
                        _mean(delta_next),
                        _mean([1.0 if value else 0.0 for value in punish_back]) if punish_back else None,
                        endowment,
                    ),
                    "evidence": {
                        "delta_contrib_next_mean": _mean(delta_next),
                        "n_events": len(delta_next),
                        "punish_back_rate_next": _mean([1.0 if value else 0.0 for value in punish_back]) if punish_back else None,
                    },
                },
            }

        reward_module = None
        if reward_enabled and reward_given is not None:
            reward_any_rate = len(reward_given["rounds_used"]) / num_rounds_observed if num_rounds_observed else None
            mean_units_when_used_r = reward_given["mean_units_when_used"]
            p75_units_when_used_r = None
            if reward_given["rounds_used"]:
                units_used_r = [sum(d.values()) for d in rewarded_dicts if sum(d.values()) > 0]
                if units_used_r:
                    p75_units_when_used_r = float(np.percentile(units_used_r, 75))

            high_units = 0
            retaliate_units_r = 0
            for idx, rid in enumerate(round_ids):
                outgoing = rewarded_dicts[idx]
                if not outgoing:
                    continue
                info = round_info.get((game_id, rid), {})
                round_median = info.get("group_median")
                contribs_map = info.get("contribs", {})
                for target, units in outgoing.items():
                    target_contrib = contribs_map.get(str(target))
                    if target_contrib is not None and round_median is not None and target_contrib >= round_median:
                        high_units += units
                if idx >= 1:
                    prev_rewarders = set(rewarded_by_dicts[idx - 1].keys())
                    retaliate_units_r += sum(units for target, units in outgoing.items() if target in prev_rewarders)

            delta_next_r = []
            for idx in range(len(round_ids) - 1):
                incoming_r = rewarded_by_dicts[idx]
                if sum(incoming_r.values()) <= 0:
                    continue
                if contribs[idx] is None or contribs[idx + 1] is None:
                    continue
                delta_next_r.append(contribs[idx + 1] - contribs[idx])

            reward_module = {
                "reward_propensity": {
                    "level": _punish_propensity_level(reward_any_rate),
                    "evidence": {
                        "reward_any_rate": reward_any_rate,
                        "mean_units_per_round": reward_given["mean_units_per_round"],
                    },
                },
                "intensity_style": {
                    "level": _intensity_level(mean_units_when_used_r),
                    "evidence": {
                        "mean_units_when_used": mean_units_when_used_r,
                        "p75_units_when_used": p75_units_when_used_r,
                    },
                },
                "targeting_rule": {
                    "style": "high_contributor"
                    if reward_given["total_units"] and high_units / reward_given["total_units"] >= 0.7
                    else "retaliation"
                    if reward_given["total_units"] and retaliate_units_r / reward_given["total_units"] >= 0.4
                    else "unknown",
                    "evidence": {
                        "high_target_fraction": high_units / reward_given["total_units"] if reward_given["total_units"] else None,
                        "retaliation_fraction": retaliate_units_r / reward_given["total_units"] if reward_given["total_units"] else None,
                        "unique_targets_count": reward_given["unique_targets_count"],
                        "target_units_hhi": reward_given["target_units_hhi"],
                    },
                },
                "response_when_rewarded": {
                    "style": _reward_response_style(_mean(delta_next_r), endowment),
                    "evidence": {
                        "delta_contrib_next_mean": _mean(delta_next_r),
                        "n_events": len(delta_next_r),
                    },
                },
            }

        event_responses = []
        defection_threshold = 0 if config.get("CONFIG_allOrNothing") is True else max(2, math.floor(0.1 * endowment)) if endowment is not None else 0
        for idx, rid in enumerate(round_ids):
            round_index = round_indices[idx]
            info = round_info.get((game_id, rid), {})
            others = {pid: value for pid, value in info.get("contribs", {}).items() if pid != player_id}
            other_vals = [value for value in others.values() if value is not None]
            defectors = []
            for pid, value in others.items():
                if value is None:
                    continue
                if config.get("CONFIG_allOrNothing") is True:
                    if value == 0:
                        defectors.append(pid)
                elif value <= defection_threshold:
                    defectors.append(pid)
            if defectors:
                delta_next = None
                if idx + 1 < len(contribs) and contribs[idx] is not None and contribs[idx + 1] is not None:
                    delta_next = contribs[idx + 1] - contribs[idx]
                outgoing = punished_dicts[idx] if punishment_enabled else {}
                event_responses.append(
                    {
                        "event_type": "saw_defection",
                        "roundId": rid,
                        "round_index": round_index,
                        "event_details": {
                            "num_defectors": len(defectors),
                            "min_other_contrib": min(other_vals) if other_vals else None,
                            "round_median_contrib": info.get("group_median"),
                        },
                        "response_next": {
                            "delta_contribution_next": delta_next,
                            "punished_defectors_same_round_units": sum(
                                units for target, units in outgoing.items() if target in defectors
                            ) if punishment_enabled else None,
                            "total_punish_units_same_round": sum(outgoing.values()) if punishment_enabled else None,
                        },
                    }
                )

            incoming = punished_by_dicts[idx]
            if sum(incoming.values()) > 0:
                delta_next = None
                if idx + 1 < len(contribs) and contribs[idx] is not None and contribs[idx + 1] is not None:
                    delta_next = contribs[idx + 1] - contribs[idx]
                event_responses.append(
                    {
                        "event_type": "was_punished",
                        "roundId": rid,
                        "round_index": round_index,
                        "event_details": {
                            "units_received": sum(incoming.values()),
                            "num_sources": len([key for key, value in incoming.items() if value > 0]),
                        },
                        "response_next": {"delta_contribution_next": delta_next},
                    }
                )

            incoming_r = rewarded_by_dicts[idx]
            if reward_enabled and sum(incoming_r.values()) > 0:
                delta_next = None
                if idx + 1 < len(contribs) and contribs[idx] is not None and contribs[idx + 1] is not None:
                    delta_next = contribs[idx + 1] - contribs[idx]
                event_responses.append(
                    {
                        "event_type": "was_rewarded",
                        "roundId": rid,
                        "round_index": round_index,
                        "event_details": {
                            "units_received": sum(incoming_r.values()),
                            "num_sources": len([key for key, value in incoming_r.items() if value > 0]),
                        },
                        "response_next": {"delta_contribution_next": delta_next},
                    }
                )

        if show_n_rounds and num_rounds_observed > 0:
            k = min(3, max(1, num_rounds_observed // 3))
            for idx, rid in enumerate(round_ids[-k:]):
                position = len(round_ids) - k + idx
                event_responses.append(
                    {
                        "event_type": "endgame_phase",
                        "roundId": rid,
                        "round_index": round_indices[position],
                        "event_details": {"k": k},
                        "response_next": {"contribution": contribs[position]},
                    }
                )

        record = {
            "split": split,
            "gameId": game_id,
            "playerId": player_id,
            "played_to_end": played_to_end,
            "has_missing_contribution": len(missing_round_indices) > 0,
            "missing_contribution_round_indices": missing_round_indices,
            "completion_reason": "complete" if played_to_end else "missing_contribution",
            "config": config,
            "derived_context": {
                "num_rounds_observed": num_rounds_observed,
                "num_rounds_configured": config.get("CONFIG_numRounds"),
                "round_order_source": round_order_source,
                "round_index_by_roundId": {str(k): int(v) for k, v in round_index_map.items()},
                "action_space_expected": action_space_expected,
                "action_space_observed": action_space_observed,
                "mechanisms_enabled": {
                    "punishment": config.get("CONFIG_punishmentExists"),
                    "reward": config.get("CONFIG_rewardExists"),
                },
                "group_size": {
                    "configured": config.get("CONFIG_playerCount"),
                    "mean_active": game_group_sizes.get(game_id, {}).get("mean_active"),
                    "min_active": game_group_sizes.get(game_id, {}).get("min_active"),
                    "max_active": game_group_sizes.get(game_id, {}).get("max_active"),
                },
            },
            "observed_summary": {
                "contribution": contrib_summary,
                "punishment_given": punishment_given,
                "punishment_received": punishment_received,
                "reward_given": reward_given,
                "reward_received": reward_received,
                "payoffs": payoffs,
            },
            "module_card": {
                "base": module_base,
                "conditionality": conditionality,
                "punishment_module": punishment_module,
                "reward_module": reward_module,
            },
            "event_responses": event_responses,
        }
        output_records.append(_ensure_jsonable(record))

    return output_records


def build_raw_profiles(splits: Iterable[str]) -> List[Dict]:
    records: List[Dict] = []
    for split in splits:
        records.extend(build_raw_profiles_for_split(split))
    return records


def write_profiles_jsonl(records: Iterable[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True))
            f.write("\n")
