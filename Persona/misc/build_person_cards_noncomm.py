import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _clean_scalar(val, desired_type=None):
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, (np.floating, np.integer)):
        val = val.item()
    if isinstance(val, str):
        s = val.strip()
        if s == "" or s.lower() == "nan":
            return None
        if s.lower() in {"true", "false"}:
            val = s.lower() == "true"
        else:
            return s
    if desired_type is None:
        return val
    if desired_type == "bool":
        if isinstance(val, bool):
            return val
        return None
    if desired_type == "int":
        try:
            if isinstance(val, bool):
                return int(val)
            return int(val)
        except Exception:
            return None
    if desired_type == "float":
        try:
            return float(val)
        except Exception:
            return None
    return val


def _parse_units_dict(val):
    if val is None:
        return {}
    if isinstance(val, float) and math.isnan(val):
        return {}
    if isinstance(val, dict):
        raw = val
    else:
        s = str(val).strip()
        if s == "" or s.lower() == "nan" or s == "{}":
            return {}
        try:
            raw = json.loads(s)
        except Exception:
            try:
                raw = ast.literal_eval(s)
            except Exception:
                return {}
    if not isinstance(raw, dict):
        return {}
    cleaned = {}
    for k, v in raw.items():
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        try:
            cleaned[str(k)] = int(v)
        except Exception:
            try:
                cleaned[str(k)] = int(float(v))
            except Exception:
                continue
    return cleaned


def _parse_round_numeric(val):
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    s = str(val)
    match = re.search(r"\d+", s)
    if not match:
        return None
    try:
        return int(match.group())
    except Exception:
        return None


def _median(values):
    if not values:
        return None
    return float(np.median(np.array(values, dtype=float)))


def _std(values):
    if not values:
        return None
    return float(np.std(np.array(values, dtype=float), ddof=0))


def _mean(values):
    if not values:
        return None
    return float(np.mean(np.array(values, dtype=float)))


def _hhi(units_by_key):
    total = sum(units_by_key.values())
    if total <= 0:
        return None
    return float(sum((u / total) ** 2 for u in units_by_key.values()))


def _pearson_corr(x, y):
    if len(x) < 2:
        return None
    try:
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)
        if np.std(x_arr) == 0 or np.std(y_arr) == 0:
            return 0.0
        return float(np.corrcoef(x_arr, y_arr)[0, 1])
    except Exception:
        return None


def _ols_slope(x, y):
    if len(x) < 2:
        return None
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    var_x = np.var(x_arr)
    if var_x == 0:
        return 0.0
    cov_xy = np.mean((x_arr - np.mean(x_arr)) * (y_arr - np.mean(y_arr)))
    return float(cov_xy / var_x)


def _action_space_observed(contribs, endowment):
    if not contribs:
        return "unknown"
    if endowment is None:
        return "unknown"
    vals = [c for c in contribs if c is not None]
    if not vals:
        return "unknown"
    unique_vals = set(vals)
    if unique_vals.issubset({0, endowment}):
        return "binary"
    all_int = all(float(v).is_integer() for v in vals)
    if all_int and all(0 <= v <= endowment for v in vals):
        return "0-20"
    return "other"


def _action_space_expected(all_or_nothing, endowment):
    if all_or_nothing is True:
        return "binary"
    if endowment is not None:
        return "0-20"
    return "unknown"


def _coop_baseline_level(action_space_observed, endowment, mean_contrib, always_max, always_min):
    if action_space_observed == "other" or mean_contrib is None:
        return "unknown"
    if endowment is None:
        return "unknown"
    if always_max:
        return "max"
    if always_min:
        return "min"
    frac = mean_contrib / endowment if endowment else None
    if frac is None:
        return "unknown"
    if frac < 0.33:
        return "low"
    if frac < 0.67:
        return "med"
    return "high"


def _volatility_level(switch_rate, num_rounds):
    if num_rounds is None or num_rounds < 3 or switch_rate is None:
        return "unknown"
    if switch_rate == 0:
        return "very_low"
    if 0 < switch_rate <= 0.1:
        return "low"
    if 0.1 < switch_rate <= 0.3:
        return "med"
    if switch_rate > 0.3:
        return "high"
    return "unknown"


def _endgame_shift(contribs, show_n_rounds, endowment, always_constant):
    if show_n_rounds is not True:
        return {
            "direction": "unknown",
            "evidence": {
                "k": None,
                "mean_first_k": None,
                "mean_last_k": None,
                "delta_last_minus_first": None,
            },
        }
    t = len(contribs)
    if t < 6 or always_constant:
        return {
            "direction": "unknown",
            "evidence": {
                "k": None,
                "mean_first_k": None,
                "mean_last_k": None,
                "delta_last_minus_first": None,
            },
        }
    k = min(3, max(1, t // 3))
    first = contribs[:k]
    last = contribs[-k:]
    mean_first = _mean(first)
    mean_last = _mean(last)
    delta = None if mean_first is None or mean_last is None else mean_last - mean_first
    threshold = 1.0
    if endowment is not None:
        threshold = max(1.0, 0.05 * endowment)
    if delta is None:
        direction = "unknown"
    elif abs(delta) <= threshold:
        direction = "none"
    elif delta > threshold:
        direction = "up"
    else:
        direction = "down"
    return {
        "direction": direction,
        "evidence": {
            "k": k,
            "mean_first_k": mean_first,
            "mean_last_k": mean_last,
            "delta_last_minus_first": delta,
        },
    }


def _conditionality(contribs, others_mean_prev, always_constant):
    x = []
    y = []
    for t in range(1, len(contribs)):
        prev_other = others_mean_prev[t]
        if prev_other is None:
            continue
        if contribs[t] is None:
            continue
        x.append(prev_other)
        y.append(contribs[t])
    n_obs = len(x)
    if n_obs < 5 or always_constant:
        return {
            "direction": "unknown",
            "strength": "unknown",
            "evidence": {
                "method": "ols",
                "b_slope": None,
                "corr": None,
                "n_obs": n_obs,
            },
        }
    b_slope = _ols_slope(x, y)
    corr = _pearson_corr(x, y)
    small_threshold = 0.1
    if b_slope is None:
        direction = "unknown"
    elif b_slope > small_threshold:
        direction = "positive"
    elif b_slope < -small_threshold:
        direction = "negative"
    else:
        direction = "none"
    strength = "unknown"
    if corr is not None:
        abs_corr = abs(corr)
        if 0.1 <= abs_corr < 0.3:
            strength = "weak"
        elif 0.3 <= abs_corr < 0.6:
            strength = "mod"
        elif abs_corr >= 0.6:
            strength = "strong"
        else:
            strength = "weak" if abs_corr > 0 else "unknown"
    return {
        "direction": direction,
        "strength": strength,
        "evidence": {
            "method": "ols",
            "b_slope": b_slope,
            "corr": corr,
            "n_obs": n_obs,
        },
    }


def _summarize_transfer(dicts_by_round):
    total_units = 0
    rounds_used = []
    units_by_target = {}
    per_round_units = []
    for round_id, d in dicts_by_round:
        round_total = sum(d.values()) if d else 0
        per_round_units.append(round_total)
        if round_total > 0:
            rounds_used.append(round_id)
        total_units += round_total
        for k, v in (d or {}).items():
            units_by_target[k] = units_by_target.get(k, 0) + v
    return total_units, rounds_used, units_by_target, per_round_units


def _punish_propensity_level(punish_any_rate):
    if punish_any_rate is None:
        return "unknown"
    if punish_any_rate == 0:
        return "none"
    if punish_any_rate <= 0.2:
        return "low"
    if punish_any_rate <= 0.5:
        return "med"
    return "high"


def _intensity_level(mean_units_when_used):
    if mean_units_when_used is None:
        return "unknown"
    if mean_units_when_used < 2:
        return "light"
    if mean_units_when_used < 5:
        return "moderate"
    return "heavy"


def _response_style(delta_mean, punish_back_rate, endowment):
    if delta_mean is None:
        return "unknown"
    threshold = 2.0
    if endowment is not None:
        threshold = max(2.0, 0.1 * endowment)
    if delta_mean > threshold:
        return "compliance"
    if delta_mean < -threshold or (punish_back_rate is not None and punish_back_rate > 0.3):
        return "retaliation"
    if abs(delta_mean) <= threshold and (punish_back_rate is None or punish_back_rate <= 0.1):
        return "no_change"
    return "mixed"


def _reward_response_style(delta_mean, endowment):
    if delta_mean is None:
        return "unknown"
    threshold = 2.0
    if endowment is not None:
        threshold = max(2.0, 0.1 * endowment)
    if delta_mean > threshold:
        return "increase"
    if delta_mean < -threshold:
        return "decrease"
    if abs(delta_mean) <= threshold:
        return "no_change"
    return "mixed"


def _ensure_jsonable(val):
    if isinstance(val, (np.integer, np.floating)):
        val = val.item()
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, dict):
        return {str(k): _ensure_jsonable(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_ensure_jsonable(v) for v in val]
    return val


def main():
    base_dir = Path(__file__).resolve().parents[1]
    player_path = base_dir / "data" / "raw_data" / "learning_wave" / "player-rounds.csv"
    config_path = base_dir / "data" / "processed_data" / "df_analysis_learn.csv"
    output_path = base_dir / "Persona" / "person_cards_noncomm.jsonl"

    df_config = pd.read_csv(config_path)
    df_player = pd.read_csv(player_path)
    df_player = df_player.reset_index().rename(columns={"index": "row_order"})

    df_player["gameId_key"] = df_player["gameId"].astype(str)
    df_player["playerId_key"] = df_player["playerId"].astype(str)
    df_player["roundId_key"] = df_player["roundId"].astype(str)

    config_fields = [
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
    field_types = {
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

    config_map = {}
    for _, row in df_config.iterrows():
        game_id = str(row.get("gameId"))
        config = {}
        for field in config_fields:
            if field not in df_config.columns:
                config[field] = None
                continue
            desired = field_types.get(field)
            config[field] = _clean_scalar(row.get(field), desired_type=desired)
        if config.get("CONFIG_punishmentExists") is False:
            config["CONFIG_punishmentCost"] = None
            config["CONFIG_punishmentMagnitude"] = None
        if config.get("CONFIG_rewardExists") is False:
            config["CONFIG_rewardCost"] = None
            config["CONFIG_rewardMagnitude"] = None
        config_map[game_id] = config

    df_player["punished_dict"] = df_player["data.punished"].apply(_parse_units_dict)
    df_player["punishedBy_dict"] = df_player["data.punishedBy"].apply(_parse_units_dict)
    df_player["rewarded_dict"] = df_player["data.rewarded"].apply(_parse_units_dict)
    df_player["rewardedBy_dict"] = df_player["data.rewardedBy"].apply(_parse_units_dict)

    round_order_by_game = {}
    round_order_source_by_game = {}
    round_index_map_by_game = {}

    for game_id, game_df in df_player.groupby("gameId_key", sort=False):
        round_ids_appearance = []
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

    round_info = {}
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
            for pid, c in contribs.items():
                others_vals = [v for opid, v in contribs.items() if opid != pid and v is not None]
                if len(others_vals) == 0:
                    others_mean[pid] = None
                    others_median[pid] = None
                else:
                    others_mean[pid] = float(np.mean(np.array(others_vals, dtype=float)))
                    others_median[pid] = _median(others_vals)
            round_info[(game_id, rid)] = {
                "active_players": active_players,
                "contribs": contribs,
                "group_total": group_total,
                "group_mean": group_mean,
                "group_median": group_median,
                "others_mean": others_mean,
                "others_median": others_median,
            }

    round_index_col = []
    group_median_col = []
    group_mean_col = []
    group_total_col = []
    active_players_col = []
    others_mean_col = []
    others_median_col = []

    for _, row in df_player.iterrows():
        game_id = row["gameId_key"]
        round_id = row["roundId_key"]
        pid = row["playerId_key"]
        round_index = round_index_map_by_game.get(game_id, {}).get(round_id)
        info = round_info.get((game_id, round_id), {})
        round_index_col.append(round_index)
        group_median_col.append(info.get("group_median"))
        group_mean_col.append(info.get("group_mean"))
        group_total_col.append(info.get("group_total"))
        active_players_col.append(info.get("active_players"))
        others_mean_col.append(info.get("others_mean", {}).get(pid))
        others_median_col.append(info.get("others_median", {}).get(pid))

    df_player["round_index"] = round_index_col
    df_player["group_median"] = group_median_col
    df_player["group_mean"] = group_mean_col
    df_player["group_total"] = group_total_col
    df_player["active_players"] = active_players_col
    df_player["others_mean"] = others_mean_col
    df_player["others_median"] = others_median_col

    game_group_sizes = {}
    for game_id, ordered_rounds in round_order_by_game.items():
        active_counts = [round_info[(game_id, rid)]["active_players"] for rid in ordered_rounds]
        if active_counts:
            game_group_sizes[game_id] = {
                "mean_active": float(np.mean(active_counts)),
                "min_active": int(np.min(active_counts)),
                "max_active": int(np.max(active_counts)),
            }
        else:
            game_group_sizes[game_id] = {
                "mean_active": None,
                "min_active": None,
                "max_active": None,
            }

    output_records = []

    for (game_id, player_id), group_df in df_player.groupby(["gameId_key", "playerId_key"], sort=False):
        config = config_map.get(game_id)
        if config is None:
            config = {field: None for field in config_fields}
        punishment_enabled = config.get("CONFIG_punishmentExists") is True
        reward_enabled = config.get("CONFIG_rewardExists") is True
        show_n_rounds = config.get("CONFIG_showNRounds") is True
        endowment = config.get("CONFIG_endowment")

        ordered_rounds = round_order_by_game.get(game_id, [])
        round_index_map = round_index_map_by_game.get(game_id, {})
        round_order_source = round_order_source_by_game.get(game_id, "unknown")

        group_df = group_df.sort_values("round_index")
        contribs = []
        round_ids = []
        round_indices = []
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
            if c is None or (isinstance(c, float) and math.isnan(c)):
                contribs.append(None)
            else:
                contribs.append(float(c))
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
            others_mean_series.append(row.get("others_mean"))

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
            else:
                always_max = all(c == contrib_max for c in contribs_nonnull) if contrib_max is not None else None
                always_min = all(c == contrib_min for c in contribs_nonnull) if contrib_min is not None else None

        switch_rate = None
        large_jump_rate = None
        if num_rounds_observed >= 2:
            switches = 0
            large_jumps = 0
            for i in range(1, num_rounds_observed):
                if contribs[i] is None or contribs[i - 1] is None:
                    continue
                if contribs[i] != contribs[i - 1]:
                    switches += 1
                if endowment is not None:
                    if abs(contribs[i] - contribs[i - 1]) >= math.ceil(endowment / 2):
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
            total_units, rounds_used, units_by_target, per_round_units = _summarize_transfer(
                list(zip(round_ids, punished_dicts))
            )
            total_cost_implied = (
                total_units * config.get("CONFIG_punishmentCost")
                if config.get("CONFIG_punishmentCost") is not None
                else None
            )
            total_harm_implied = (
                total_units * config.get("CONFIG_punishmentMagnitude")
                if config.get("CONFIG_punishmentMagnitude") is not None
                else None
            )
            punishment_given = {
                "used_any": total_units > 0,
                "rounds_used": rounds_used,
                "total_units": int(total_units),
                "total_cost_implied": total_cost_implied,
                "total_harm_implied": total_harm_implied,
                "mean_units_per_round": total_units / num_rounds_observed if num_rounds_observed else None,
                "mean_units_when_used": total_units / len(rounds_used) if rounds_used else None,
                "unique_targets_count": len(units_by_target),
                "target_units_hhi": _hhi(units_by_target),
            }
        else:
            punishment_given = None

        if punishment_enabled:
            total_units_in, rounds_received, units_by_source, per_round_units_in = _summarize_transfer(
                list(zip(round_ids, punished_by_dicts))
            )
            total_penalty_implied = (
                total_units_in * config.get("CONFIG_punishmentMagnitude")
                if config.get("CONFIG_punishmentMagnitude") is not None
                else None
            )
            punishment_received = {
                "received_any": total_units_in > 0,
                "rounds_received": rounds_received,
                "total_units": int(total_units_in),
                "total_penalty_implied": total_penalty_implied,
                "mean_units_per_round": total_units_in / num_rounds_observed if num_rounds_observed else None,
                "mean_units_when_received": total_units_in / len(rounds_received) if rounds_received else None,
                "unique_sources_count": len(units_by_source),
                "source_units_hhi": _hhi(units_by_source),
            }
        else:
            punishment_received = None

        if reward_enabled:
            total_units_r, rounds_used_r, units_by_target_r, per_round_units_r = _summarize_transfer(
                list(zip(round_ids, rewarded_dicts))
            )
            total_cost_implied_r = (
                total_units_r * config.get("CONFIG_rewardCost") if config.get("CONFIG_rewardCost") is not None else None
            )
            total_benefit_implied_r = (
                total_units_r * config.get("CONFIG_rewardMagnitude")
                if config.get("CONFIG_rewardMagnitude") is not None
                else None
            )
            reward_given = {
                "used_any": total_units_r > 0,
                "rounds_used": rounds_used_r,
                "total_units": int(total_units_r),
                "total_cost_implied": total_cost_implied_r,
                "total_benefit_implied": total_benefit_implied_r,
                "mean_units_per_round": total_units_r / num_rounds_observed if num_rounds_observed else None,
                "mean_units_when_used": total_units_r / len(rounds_used_r) if rounds_used_r else None,
                "unique_targets_count": len(units_by_target_r),
                "target_units_hhi": _hhi(units_by_target_r),
            }
        else:
            reward_given = None

        if reward_enabled:
            total_units_in_r, rounds_received_r, units_by_source_r, per_round_units_in_r = _summarize_transfer(
                list(zip(round_ids, rewarded_by_dicts))
            )
            total_bonus_implied = (
                total_units_in_r * config.get("CONFIG_rewardMagnitude")
                if config.get("CONFIG_rewardMagnitude") is not None
                else None
            )
            reward_received = {
                "received_any": total_units_in_r > 0,
                "rounds_received": rounds_received_r,
                "total_units": int(total_units_in_r),
                "total_bonus_implied": total_bonus_implied,
                "mean_units_per_round": total_units_in_r / num_rounds_observed if num_rounds_observed else None,
                "mean_units_when_received": total_units_in_r / len(rounds_received_r) if rounds_received_r else None,
                "unique_sources_count": len(units_by_source_r),
                "source_units_hhi": _hhi(units_by_source_r),
            }
        else:
            reward_received = None

        def _safe_vals(vals):
            cleaned = []
            for v in vals:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                cleaned.append(float(v))
            return cleaned

        costs_vals = _safe_vals(costs)
        penalties_vals = _safe_vals(penalties)
        rewards_vals = _safe_vals(rewards)
        remaining_vals = _safe_vals(remaining_endowment)
        payoff_vals = _safe_vals(round_payoffs)

        payoffs = {
            "round_payoff_mean": _mean(payoff_vals),
            "round_payoff_std": _std(payoff_vals),
            "costs_mean": _mean(costs_vals),
            "penalties_mean": _mean(penalties_vals),
            "rewards_mean": _mean(rewards_vals),
            "remaining_endowment_mean": _mean(remaining_vals),
        }

        action_space_expected = _action_space_expected(config.get("CONFIG_allOrNothing"), endowment)
        action_space_observed = _action_space_observed(contribs_nonnull, endowment)

        derived_context = {
            "num_rounds_observed": num_rounds_observed,
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
        }

        coop_level = _coop_baseline_level(
            action_space_observed, endowment, contrib_mean, always_max, always_min
        )
        volatility_level = _volatility_level(switch_rate, num_rounds_observed)
        endgame_shift = _endgame_shift(contribs_nonnull, show_n_rounds, endowment, always_constant)

        conditionality = {
            "responds_to_others_mean": _conditionality(contribs, [None] + others_mean_series[:-1], always_constant)
        }

        module_base = {
            "coop_baseline": {
                "level": coop_level,
                "evidence": {
                    "mean_contribution": contrib_mean,
                    "mean_contribution_frac_of_endowment": (
                        contrib_mean / endowment if contrib_mean is not None and endowment else None
                    ),
                    "first_round_contribution": contribs[0] if contribs else None,
                },
            },
            "volatility": {
                "level": volatility_level,
                "evidence": {
                    "switch_rate": switch_rate,
                    "std_contribution": contrib_std,
                },
            },
            "endgame_shift": endgame_shift,
        }

        punishment_module = None
        if punishment_enabled:
            punish_any_rate = (
                len(punishment_given["rounds_used"]) / num_rounds_observed if num_rounds_observed else None
            )
            mean_units_per_round = punishment_given["mean_units_per_round"]
            mean_units_when_used = punishment_given["mean_units_when_used"]
            p75_units_when_used = None
            if punishment_given["rounds_used"]:
                units_used = [sum(d.values()) for d in punished_dicts if sum(d.values()) > 0]
                if units_used:
                    p75_units_when_used = float(np.percentile(units_used, 75))
            total_units_out = punishment_given["total_units"]

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
                    if target_contrib is not None and round_median is not None:
                        if target_contrib < round_median:
                            low_units += units
                if idx >= 1:
                    prev_punishers = set(punished_by_dicts[idx - 1].keys())
                    for target, units in outgoing.items():
                        if target in prev_punishers:
                            retaliate_units += units
            low_target_fraction = low_units / total_units_out if total_units_out else None
            retaliation_fraction = retaliate_units / total_units_out if total_units_out else None

            targeting_style = "unknown"
            if total_units_out > 0 and low_target_fraction is not None and low_target_fraction >= 0.7:
                targeting_style = "low_contributor"
            elif total_units_out > 0 and retaliation_fraction is not None and retaliation_fraction >= 0.4:
                targeting_style = "retaliation"
            elif (
                total_units_out > 0
                and punishment_given["unique_targets_count"] >= 3
                and (punishment_given["target_units_hhi"] is not None and punishment_given["target_units_hhi"] <= 0.4)
                and targeting_style == "unknown"
            ):
                targeting_style = "diffuse"

            delta_next = []
            punish_back = []
            for idx, rid in enumerate(round_ids[:-1]):
                incoming = punished_by_dicts[idx]
                if sum(incoming.values()) <= 0:
                    continue
                if contribs[idx] is None or contribs[idx + 1] is None:
                    continue
                delta_next.append(contribs[idx + 1] - contribs[idx])
                if punishment_enabled:
                    sources = set(incoming.keys())
                    next_out = punished_dicts[idx + 1]
                    punish_back.append(bool(sources.intersection(set(next_out.keys()))))
            delta_mean = _mean(delta_next)
            punish_back_rate = _mean([1.0 if v else 0.0 for v in punish_back]) if punish_back else None

            response_style = "unknown"
            if len(delta_next) >= 2:
                response_style = _response_style(delta_mean, punish_back_rate, endowment)

            punishment_module = {
                "punish_propensity": {
                    "level": _punish_propensity_level(punish_any_rate),
                    "evidence": {
                        "punish_any_rate": punish_any_rate,
                        "mean_units_per_round": mean_units_per_round,
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
                    "style": targeting_style,
                    "evidence": {
                        "low_target_fraction": low_target_fraction,
                        "retaliation_fraction": retaliation_fraction,
                        "unique_targets_count": punishment_given["unique_targets_count"],
                        "target_units_hhi": punishment_given["target_units_hhi"],
                    },
                },
                "response_when_punished": {
                    "style": response_style,
                    "evidence": {
                        "delta_contrib_next_mean": delta_mean,
                        "n_events": len(delta_next),
                        "punish_back_rate_next": punish_back_rate,
                    },
                },
            }

        reward_module = None
        if reward_enabled:
            reward_any_rate = (
                len(reward_given["rounds_used"]) / num_rounds_observed if num_rounds_observed else None
            )
            mean_units_per_round_r = reward_given["mean_units_per_round"]
            mean_units_when_used_r = reward_given["mean_units_when_used"]
            p75_units_when_used_r = None
            if reward_given["rounds_used"]:
                units_used_r = [sum(d.values()) for d in rewarded_dicts if sum(d.values()) > 0]
                if units_used_r:
                    p75_units_when_used_r = float(np.percentile(units_used_r, 75))
            total_units_out_r = reward_given["total_units"]

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
                    if target_contrib is not None and round_median is not None:
                        if target_contrib >= round_median:
                            high_units += units
                if idx >= 1:
                    prev_rewarders = set(rewarded_by_dicts[idx - 1].keys())
                    for target, units in outgoing.items():
                        if target in prev_rewarders:
                            retaliate_units_r += units
            high_target_fraction = high_units / total_units_out_r if total_units_out_r else None
            retaliation_fraction_r = retaliate_units_r / total_units_out_r if total_units_out_r else None

            targeting_style_r = "unknown"
            if total_units_out_r > 0 and high_target_fraction is not None and high_target_fraction >= 0.7:
                targeting_style_r = "high_contributor"
            elif total_units_out_r > 0 and retaliation_fraction_r is not None and retaliation_fraction_r >= 0.4:
                targeting_style_r = "retaliation"
            elif (
                total_units_out_r > 0
                and reward_given["unique_targets_count"] >= 3
                and (reward_given["target_units_hhi"] is not None and reward_given["target_units_hhi"] <= 0.4)
                and targeting_style_r == "unknown"
            ):
                targeting_style_r = "diffuse"

            delta_next_r = []
            for idx, rid in enumerate(round_ids[:-1]):
                incoming = rewarded_by_dicts[idx]
                if sum(incoming.values()) <= 0:
                    continue
                if contribs[idx] is None or contribs[idx + 1] is None:
                    continue
                delta_next_r.append(contribs[idx + 1] - contribs[idx])
            delta_mean_r = _mean(delta_next_r)

            response_style_r = "unknown"
            if len(delta_next_r) >= 2:
                response_style_r = _reward_response_style(delta_mean_r, endowment)

            reward_module = {
                "reward_propensity": {
                    "level": _punish_propensity_level(reward_any_rate),
                    "evidence": {
                        "reward_any_rate": reward_any_rate,
                        "mean_units_per_round": mean_units_per_round_r,
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
                    "style": targeting_style_r,
                    "evidence": {
                        "high_target_fraction": high_target_fraction,
                        "retaliation_fraction": retaliation_fraction_r,
                        "unique_targets_count": reward_given["unique_targets_count"],
                        "target_units_hhi": reward_given["target_units_hhi"],
                    },
                },
                "response_when_rewarded": {
                    "style": response_style_r,
                    "evidence": {
                        "delta_contrib_next_mean": delta_mean_r,
                        "n_events": len(delta_next_r),
                    },
                },
            }

        observed_summary = {
            "contribution": contrib_summary,
            "punishment_given": punishment_given,
            "punishment_received": punishment_received,
            "reward_given": reward_given,
            "reward_received": reward_received,
            "payoffs": payoffs,
        }

        event_responses = []
        defection_threshold = None
        if config.get("CONFIG_allOrNothing") is True:
            defection_threshold = 0
        elif endowment is not None:
            defection_threshold = max(2, math.floor(0.1 * endowment))
        else:
            defection_threshold = 0

        for idx, rid in enumerate(round_ids):
            round_index = round_indices[idx]
            info = round_info.get((game_id, rid), {})
            contribs_map = info.get("contribs", {})
            others = {pid: c for pid, c in contribs_map.items() if pid != player_id}
            other_vals = [c for c in others.values() if c is not None]
            defectors = []
            for pid, c in others.items():
                if c is None:
                    continue
                if config.get("CONFIG_allOrNothing") is True:
                    if c == 0:
                        defectors.append(pid)
                else:
                    if c <= defection_threshold:
                        defectors.append(pid)
            if defectors:
                outgoing = punished_dicts[idx] if punishment_enabled else {}
                punished_defectors_units = 0
                total_out_units = sum(outgoing.values()) if punishment_enabled else None
                if punishment_enabled:
                    for target, units in outgoing.items():
                        if target in defectors:
                            punished_defectors_units += units
                delta_next = None
                if idx + 1 < len(contribs) and contribs[idx] is not None and contribs[idx + 1] is not None:
                    delta_next = contribs[idx + 1] - contribs[idx]
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
                            "punished_defectors_same_round_units": (
                                punished_defectors_units if punishment_enabled else None
                            ),
                            "total_punish_units_same_round": total_out_units,
                        },
                    }
                )

            incoming = punished_by_dicts[idx]
            total_in = sum(incoming.values()) if incoming else 0
            if total_in > 0:
                delta_next = None
                if idx + 1 < len(contribs) and contribs[idx] is not None and contribs[idx + 1] is not None:
                    delta_next = contribs[idx + 1] - contribs[idx]
                punished_any_source_next = None
                if punishment_enabled and idx + 1 < len(contribs):
                    sources = set(incoming.keys())
                    next_out = punished_dicts[idx + 1]
                    punished_any_source_next = bool(sources.intersection(set(next_out.keys())))
                event_responses.append(
                    {
                        "event_type": "was_punished",
                        "roundId": rid,
                        "round_index": round_index,
                        "event_details": {
                            "units_received": total_in,
                            "num_sources": len([k for k, v in incoming.items() if v > 0]),
                        },
                        "response_next": {
                            "delta_contribution_next": delta_next,
                            "punished_any_source_next": punished_any_source_next,
                        },
                    }
                )

            if reward_enabled:
                incoming_r = rewarded_by_dicts[idx]
                total_in_r = sum(incoming_r.values()) if incoming_r else 0
                if total_in_r > 0:
                    delta_next = None
                    if idx + 1 < len(contribs) and contribs[idx] is not None and contribs[idx + 1] is not None:
                        delta_next = contribs[idx + 1] - contribs[idx]
                    event_responses.append(
                        {
                            "event_type": "was_rewarded",
                            "roundId": rid,
                            "round_index": round_index,
                            "event_details": {
                                "units_received": total_in_r,
                                "num_sources": len([k for k, v in incoming_r.items() if v > 0]),
                            },
                            "response_next": {
                                "delta_contribution_next": delta_next,
                            },
                        }
                    )

        if show_n_rounds and num_rounds_observed > 0:
            k = min(3, max(1, num_rounds_observed // 3))
            endgame_rounds = set(round_ids[-k:])
            for idx, rid in enumerate(round_ids):
                if rid not in endgame_rounds:
                    continue
                event_responses.append(
                    {
                        "event_type": "endgame_phase",
                        "roundId": rid,
                        "round_index": round_indices[idx],
                        "event_details": {"k": k},
                        "response_next": {"contribution": contribs[idx]},
                    }
                )

        person_card = {
            "gameId": game_id,
            "playerId": player_id,
            "config": config,
            "derived_context": derived_context,
            "observed_summary": observed_summary,
            "module_card": {
                "base": module_base,
                "conditionality": conditionality,
                "punishment_module": punishment_module,
                "reward_module": reward_module,
            },
            "event_responses": event_responses,
        }

        output_records.append(_ensure_jsonable(person_card))

    with output_path.open("w") as f:
        for record in output_records:
            f.write(json.dumps(record))
            f.write("\n")

    unique_pairs = df_player[["gameId_key", "playerId_key"]].drop_duplicates()
    if len(unique_pairs) != len(output_records):
        raise RuntimeError(
            f"Output count {len(output_records)} does not match unique pairs {len(unique_pairs)}"
        )


if __name__ == "__main__":
    main()
