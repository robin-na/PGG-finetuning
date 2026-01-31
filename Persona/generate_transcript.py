import json
import math
import ast
import re
import pandas as pd

# --------------------------------------------------------------------------------------------------
# Load the same source data you use elsewhere. If you already have DataFrames in memory, you can
# comment these three lines and pass the DataFrames directly at the bottom call.
# --------------------------------------------------------------------------------------------------
df_rounds_learn = pd.read_csv("data/raw_data/learning_wave/player-rounds.csv")
df_rounds_val = pd.read_csv("data/raw_data/validation_wave/player-rounds.csv")
df_players_learn = pd.read_csv("data/raw_data/learning_wave/players.csv")
df_players_val = pd.read_csv("data/raw_data/validation_wave/players.csv")
df_demographic_learn = pd.read_csv("data/raw_data/learning_wave/player-inputs.csv")
df_demographic_val = pd.read_csv("data/raw_data/validation_wave/player-inputs.csv")
df_analysis_learn = pd.read_csv("data/processed_data/df_analysis_learn.csv")
df_analysis_val = pd.read_csv("data/processed_data/df_analysis_val.csv")
df_chats_learn = pd.read_csv("data/raw_data/learning_wave/games.csv")
df_chats_val = pd.read_csv("data/raw_data/validation_wave/games.csv")

# --------------------------------------------------------------------------------------------------
# Helper Functions (verbatim-compatible with your existing style; included here to be self-contained)
# --------------------------------------------------------------------------------------------------


def build_avatar_map(players_df):
    """
    Given players DataFrame with columns: _id, data.avatar
    create a dict: { str(_id): <uppercase-avatar> } (e.g., '101': 'DOG')
    """
    mapping = {}
    for _, row in players_df.iterrows():
        pid_str = str(row['_id'])
        avatar_name = str(row['data.avatar']).upper()
        mapping[pid_str] = avatar_name
    return mapping

def parse_dict(value):
    """Parse a possibly-stringified dict into a Python dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return {}
    return {}

def format_num(x):
    """Return integer-like floats as int strings, else as original numeric string."""
    if isinstance(x, (int, float)):
        if float(x).is_integer():
            return str(int(x))
        else:
            return str(x)
    return str(x)


def _parse_chat_messages(msg_str: str):
    """
    Robustly parse df_chats['data.messages'] into a list[dict] without
    dropping the FIRST or LAST item. Tries JSON first, then falls back
    to a brace-scan that collects top-level {...} blocks.
    """
    if not isinstance(msg_str, str):
        return []
    s = msg_str.strip()
    if not s:
        return []

    # 1) Fast path: strict JSON list of dicts
    try:
        obj = json.loads(s)
        if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass

    # 2) Relax: sometimes it's Python-literal compatible
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass

    # 3) Robust fallback: scan and extract top-level {...} by brace counting.
    out = []
    i, n = 0, len(s)
    # Optional: clamp to the region that actually contains braces
    first_brace = s.find('{')
    last_brace  = s.rfind('}')
    if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
        return out
    i = first_brace
    while i <= last_brace:
        # seek next '{'
        while i <= last_brace and s[i] != '{':
            i += 1
        if i > last_brace:
            break
        start = i
        depth = 0
        while i <= last_brace:
            c = s[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1  # inclusive slice end
                    chunk = s[start:end]
                    # Try literal_eval then JSON for this chunk
                    parsed = None
                    try:
                        parsed = ast.literal_eval(chunk)
                    except Exception:
                        try:
                            parsed = json.loads(chunk)
                        except Exception:
                            parsed = None
                    if isinstance(parsed, dict):
                        out.append(parsed)
                    # move past this chunk
                    i = end
                    break
            i += 1
        else:
            # unbalanced braces; stop
            break
    return out

def _index_chats_for_game(df_chats, game_id):
    """
    Build chats_by_round_phase[round_idx][phase] = list of (AVATAR_UP, text),
    preserving order. Applies 1-based shift only if any message had round 0.
    """
    chats_by_round_phase = {}
    if df_chats is None or df_chats.empty:
        return chats_by_round_phase

    row = df_chats[df_chats["_id"] == game_id]
    if row.empty:
        return chats_by_round_phase

    msgs_raw = row.iloc[0].get("data.messages", "")
    msgs = _parse_chat_messages(msgs_raw)

    # Extract round & phase; tolerate extras like "(timer)"
    def _extract_round_phase(s: str):
        if not isinstance(s, str):
            return (None, None)
        sl = s.lower()
        m = re.search(r"round\s+(\d+)", sl)
        r = int(m.group(1)) if m else None
        if "contrib" in sl:
            ph = "contribution"
        elif "outcome" in sl:
            ph = "outcome"
        elif "summary" in sl:
            ph = "summary"
        else:
            ph = None
        return (r, ph)

    parsed = []
    saw_round0 = False
    for d in msgs:
        txt = str(d.get("text", "")).strip()
        av  = str(d.get("avatar", "")).strip().upper()
        r_raw, phase = _extract_round_phase(d.get("gamePhase", ""))
        if r_raw == 0:
            saw_round0 = True
        parsed.append((r_raw, phase, av, txt))

    # shift +1 only if we actually saw Round 0
    shift = 1 if saw_round0 else 0

    last_r = None
    for r_raw, phase, av, txt in parsed:
        if not txt:
            continue
        if r_raw is not None:
            r = r_raw + shift
            last_r = r
        else:
            if last_r is None:
                # if absolutely no round seen yet, drop; extremely rare
                continue
            r = last_r
        ph = phase or "outcome"
        chats_by_round_phase.setdefault(r, {}).setdefault(ph, []).append((av, txt))

    return chats_by_round_phase

# ===================== DROP-IN: merged <PUNISHMENTS/REWARDS> only when both enabled =====================

def _is_nan(x):
    try:
        return math.isnan(float(x))
    except Exception:
        return False

def _avatars(df_players: pd.DataFrame):
    return build_avatar_map(df_players)

def _peer_order(round_slice: pd.DataFrame, focal_pid, avatars: dict):
    """Stable peer order (by playerId ascending), returns avatar names excluding the focal."""
    peers = []
    for _, ro in round_slice.sort_values("playerId").iterrows():
        if ro["playerId"] == focal_pid:
            continue
        peers.append(avatars.get(str(ro["playerId"]), f"Player {ro['playerId']}"))
    return peers

def _compact_outcomes(round_slice, focal_pid, avatars):
    """DOG=20,PARROT=NA,... (others only)"""
    tokens = []
    for _, ro in round_slice.sort_values("playerId").iterrows():
        if ro["playerId"] == focal_pid:
            continue
        av = avatars.get(str(ro["playerId"]), f"Player {ro['playerId']}")
        c  = ro.get("data.contribution")
        tokens.append(f"{av}={'NA' if _is_nan(c) else format_num(c)}")
    return ",".join(tokens)

def _sparse_rewards_punishments(focal_row, round_slice, focal_pid, avatars,
                                rewards_on: bool, punish_on: bool):
    """
    Return (rewards_sparse, punishments_sparse) with ONLY non-zero peers.
    Keys: avatar; Values: non-negative integer units. If none, return {}.
    """
    rewards = {}
    punishs = {}
    if rewards_on:
        rd = parse_dict(focal_row.get("data.rewarded", {})) or {}
        for pid, u in rd.items():
            units = int(u) if u else 0
            if units > 0:
                av = avatars.get(str(pid), f"Player {pid}")
                rewards[av] = rewards.get(av, 0) + units
    if punish_on:
        pdict = parse_dict(focal_row.get("data.punished", {})) or {}
        for pid, u in pdict.items():
            units = int(u) if u else 0
            if units > 0:
                av = avatars.get(str(pid), f"Player {pid}")
                punishs[av] = punishs.get(av, 0) + units
    return rewards, punishs

def _others_summary(round_slice, avatars, punishment_exists, punishment_cost, reward_exists, reward_cost, focal_pid):
    """Per-other summary keyed by avatar."""
    out = {}
    for _, ro in round_slice.sort_values("playerId").iterrows():
        if ro["playerId"] == focal_pid:
            continue
        av = avatars.get(str(ro["playerId"]), f"Player {ro['playerId']}")
        payoff = ro.get("data.roundPayoff")
        if _is_nan(payoff):
            out[av] = {"status": "exited"}
            continue

        pun_dict = parse_dict(ro.get("data.punished", {})) or {}
        rew_dict = parse_dict(ro.get("data.rewarded", {})) or {}
        pun_units = int(sum(pun_dict.values())) if pun_dict else 0
        rew_units = int(sum(rew_dict.values())) if rew_dict else 0
        coins_spent_on_punish = pun_units * int(punishment_cost or 0)
        coins_spent_on_reward = rew_units * int(reward_cost or 0)

        coins_deducted_from_them = int(ro.get("data.penalties", 0) or 0)
        coins_rewarded_to_them   = int(ro.get("data.rewards",   0) or 0)

        out[av] = {}
        if punishment_exists:
            out[av]["coins_spent_on_punish"] = coins_spent_on_punish
            out[av][ "coins_deducted_from_them"] = coins_deducted_from_them
        if reward_exists:
            out[av]["coins_spent_on_reward"] = coins_spent_on_reward
            out[av]["coins_rewarded_to_them"] = coins_rewarded_to_them
        out[av]["payoff"] = int(float(payoff)) if not _is_nan(payoff) else None

        #{
            #"contrib": None if _is_nan(ro.get("data.contribution")) else int(float(ro.get("data.contribution"))),
            #"coins_spent_on_punish": coins_spent_on_punish,
            #"coins_spent_on_reward": coins_spent_on_reward,
            #"coins_deducted_from_them": coins_deducted_from_them,
            #"coins_rewarded_to_them": coins_rewarded_to_them,
            #"payoff": int(float(payoff)) if not _is_nan(payoff) else None
        #}
    return out


# --- your existing helpers are assumed to exist: _is_nan, _avatars, _peer_order, _compact_outcomes,
#     _sparse_rewards_punishments, parse_dict, format_num  ---

def generate_participant_transcript_chat(
    df_rounds: pd.DataFrame,
    df_players: pd.DataFrame,
    df_analysis: pd.DataFrame,
    out_path: str = "prompts_xml_full.jsonl",
    df_chats: pd.DataFrame = None,          # NEW: optional chats dataframe
):
    avatars = _avatars(df_players)

    # ----- configs per game -----
    cfg_by_game = {}
    for gid, g in df_analysis.groupby("gameId"):
        r0 = g.iloc[0]
        cfg_by_game[gid] = {
            "players":              int(r0.get("CONFIG_playerCount", 0) or 0),
            "numRounds":            int(r0.get("CONFIG_numRounds", 0) or 0),
            "showNRounds":          bool(r0.get("CONFIG_showNRounds", False)),
            "endowment":            int(r0.get("CONFIG_endowment", 0) or 0),
            "multiplier":           r0.get("CONFIG_multiplier", "Unknown"),
            "all_or_nothing":       bool(r0.get("CONFIG_allOrNothing", False)),
            "chat":                 bool(r0.get("CONFIG_chat", False)),
            "defaultContribProp":   bool(r0.get("CONFIG_defaultContribProp", False)),  # True => starts in public fund
            "punishment_exists":    bool(r0.get("CONFIG_punishmentExists", False)),
            "punishment_cost":      int(r0.get("CONFIG_punishmentCost", 0) or 0),
            "punishment_magnitude": int(r0.get("CONFIG_punishmentMagnitude", 0) or 0),
            "reward_exists":        bool(r0.get("CONFIG_rewardExists", False)),
            "reward_cost":          int(r0.get("CONFIG_rewardCost", 0) or 0),
            "reward_magnitude":     int(r0.get("CONFIG_rewardMagnitude", 0) or 0),
            "showOtherSummaries":   bool(r0.get("CONFIG_showOtherSummaries", False)),
            "showPunishmentId":     bool(r0.get("CONFIG_showPunishmentId", False)),
            "showRewardId":         bool(r0.get("CONFIG_showRewardId", False)),
        }

    df_rounds_sorted = df_rounds.sort_values(["gameId", "roundId", "playerId"]).copy()

    out_lines = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for game_id, gdf in df_rounds_sorted.groupby("gameId"):
            if game_id not in cfg_by_game:
                continue
            cfg = cfg_by_game[game_id]

            rounds_order = list(gdf["roundId"].unique())
            players = list(gdf["playerId"].unique())

            # Pre-index chats for this game once (by 1-based round index and phase)
            chats_idx = _index_chats_for_game(df_chats, game_id) if (df_chats is not None and cfg["chat"]) else {}

            # ------------------ SYSTEM (add a chat note if enabled) ------------------
            sys_lines = []
            sys_lines.append("<|begin_of_text|><|start_header_id|>system<|end_header_id|>")
            sys_lines.append("You are playing an online Public Goods Game (PGG).")
            sys_lines.append("Each round you have a fixed endowment of coins. You choose how many coins to contribute to a shared public fund.")
            sys_lines.append("After everyone has decided, all contributions are added up and multiplied by the round's multiplier.")
            sys_lines.append("The multiplied total is divided equally among all active players as the round payoff.")
            sys_lines.append("You keep whatever you did not contribute in your private pocket, plus your share from the public fund, minus any costs you pay.")
            if cfg["reward_exists"] and cfg["punishment_exists"]:
                sys_lines.append("Each round includes a peer feedback stage after contributions where you can either reward or punish other players by assigning units to them.")
            elif cfg["reward_exists"]:
                sys_lines.append("Each round includes a peer feedback stage after contributions where you can reward other players by assigning units to them.")
            elif cfg["punishment_exists"]:
                sys_lines.append("Each round includes a peer feedback stage after contributions where you can punish other players by assigning units to them.")

            # NEW: mention chat only when enabled
            if cfg["chat"]:
                sys_lines.append("You can chat with other players during the round. Chat messages appear as <CHAT> {AVATAR: text}.")

            sys_lines.append("")
            sys_lines.append("For each round:")
            sys_lines.append("1) Decide how much to contribute at the <CONTRIB> tag — output ONLY a single integer (no text).")
            if cfg["reward_exists"] and cfg["punishment_exists"]:
                sys_lines.append("2) Decide whom to punish/reward and how many units at the <PUNISHMENTS_REWARDS> tag — output ONLY an array of integers (no text).")
                sys_lines.append("   The array order MUST match the avatar order shown in <PEERS_CONTRIBUTIONS> for that round.")
                sys_lines.append("   Negative = punish (-n means punish by n units), positive = reward (n means reward by n units), 0 = neither.")
            elif cfg["reward_exists"]:
                sys_lines.append("2) Decide whom to reward and how many units at the <REWARDS> tag — output ONLY an array of integers (no text), each ≥ 0.")
                sys_lines.append("   The array order MUST match the avatar order shown in <PEERS_CONTRIBUTIONS> for that round.")
            elif cfg["punishment_exists"]:
                sys_lines.append("2) Decide whom to punish and how many units at the <PUNISHMENTS> tag — output ONLY an array of integers (no text), each ≤ 0 (-n means punish by n units).")
                sys_lines.append("   The array order MUST match the avatar order shown in <PEERS_CONTRIBUTIONS> for that round.")
            sys_lines.append("<|eot_id|>")
            sys_text = "\n".join(sys_lines)

            for pid in players:
                lines = [sys_text, "# GAME STARTS"]
                pslice = gdf[gdf["playerId"] == pid]
                if pslice.empty:
                    continue

                focal_avatar = avatars.get(str(pid), f"Player {pid}")

                for idx, rid in enumerate(rounds_order, start=1):
                    fr = pslice[pslice["roundId"] == rid]
                    if fr.empty:
                        continue
                    row = fr.iloc[0]

                    if _is_nan(row.get("data.roundPayoff")):
                        lines.append(f'<EXIT round="{idx}"/>')
                        break

                    # Open round tag
                    if cfg["showNRounds"]:
                        lines.append(f'<ROUND i="{idx} of {cfg["numRounds"]}">')
                    else:
                        lines.append(f'<ROUND i="{idx}">')

                    # ----- Pre-contribution reminder -----
                    endow = cfg["endowment"]
                    contrib_mode = (f"either 0 or {endow}") if cfg["all_or_nothing"] else (f"integer from 0 to {endow}")
                    if cfg["defaultContribProp"]:
                        pre_contrib_sentence = (
                            f"{endow} coins are currently in the public fund, and you will contribute the remainder of the coins you choose to take for yourself. "
                            f"Choose the amount to contribute ({contrib_mode})."
                        )
                    else:
                        pre_contrib_sentence = (
                            f"{endow} coins are currently in your private pocket. "
                            f"Choose the amount to contribute ({contrib_mode})."
                        )
                    lines.append(f"<ROUND_INFO> {pre_contrib_sentence} (multiplier: {cfg['multiplier']}×). </ROUND_INFO>")

                    # ===== CHATS: contribution phase (before/during contribution) =====
                    if chats_idx and idx in chats_idx and "contribution" in chats_idx[idx]:
                        for av, txt in chats_idx[idx]["contribution"]:
                            speaker = f"{av}{' (YOU)' if av == focal_avatar else ''}"
                            lines.append(f"<CHAT> {{{speaker}: {txt}}} </CHAT>")

                    # ----- CONTRIB decision -----
                    contrib_val = row.get("data.contribution", 0)
                    lines.append(f"<CONTRIB> <<{format_num(contrib_val)}>> </CONTRIB>")

                    # ----- Redistribution stats -----
                    rs = gdf[gdf["roundId"] == rid]
                    total_contrib = rs["data.contribution"].dropna().sum()
                    focal_contrib = 0 if _is_nan(contrib_val) else float(contrib_val)
                    others_total = float(total_contrib) - float(focal_contrib)
                    active_players = int(rs["data.roundPayoff"].notna().sum())
                    try:
                        multiplied = float(cfg["multiplier"]) * float(total_contrib)
                    except Exception:
                        multiplied = float("nan")
                    redistributed_each = multiplied / active_players if active_players > 0 else float("nan")
                    others_avg = (others_total / (active_players - 1)) if active_players > 1 else float("nan")

                    lines.append(
                        '<REDIST total_contrib="{}" others_total="{}" others_avg="{}" multiplied_contrib="{}" '
                        'active_players="{}" redistributed_each="{}"/>'.format(
                            format_num(total_contrib),
                            format_num(round(others_total, 3)),
                            format_num(round(others_avg, 3)) if not math.isnan(others_avg) else "NA",
                            format_num(round(multiplied, 3)) if not math.isnan(multiplied) else "",
                            active_players,
                            format_num(round(redistributed_each, 3)) if not math.isnan(redistributed_each) else "NA",
                        )
                    )

                    # ----- Peers' contributions (defines the array order) -----
                    peers_csv = _compact_outcomes(rs, pid, avatars)  # e.g., DOG=20,CAT=NA,SLOTH=7
                    lines.append(f"<PEERS_CONTRIBUTIONS> {peers_csv} </PEERS_CONTRIBUTIONS>")

                    # ===== CHATS: outcome phase (after contribs revealed; before P/R decision) =====
                    if chats_idx and idx in chats_idx and "outcome" in chats_idx[idx]:
                        for av, txt in chats_idx[idx]["outcome"]:
                            speaker = f"{av}{' (YOU)' if av == focal_avatar else ''}"
                            lines.append(f"<CHAT> {{{speaker}: {txt}}} </CHAT>")

                    # ----- Mechanism info JUST BEFORE the action tag -----
                    if cfg["reward_exists"] or cfg["punishment_exists"]:
                        if cfg["reward_exists"] and cfg["punishment_exists"]:
                            mech_info = (
                                f"It will cost you, per reward unit, {cfg['reward_cost']} coins to give a reward of {cfg['reward_magnitude']} coins. "
                                f"It will cost you, per punishment unit, {cfg['punishment_cost']} coins to impose a deduction of {cfg['punishment_magnitude']} coins. "
                                "Choose whom to punish/reward and by how many units."
                            )
                            lines.append(f"<MECHANISM_INFO> {mech_info} </MECHANISM_INFO>")
                        elif cfg["reward_exists"]:
                            mech_info = (
                                f"It will cost you, per unit, {cfg['reward_cost']} coins to give a reward of {cfg['reward_magnitude']} coins. "
                                "Choose whom to reward and by how many units."
                            )
                            lines.append(f"<MECHANISM_INFO> {mech_info} </MECHANISM_INFO>")
                        elif cfg["punishment_exists"]:
                            mech_info = (
                                f"It will cost you, per unit, {cfg['punishment_cost']} coins to impose a deduction of {cfg['punishment_magnitude']} coins. "
                                "Choose whom to punish and by how many units."
                            )
                            lines.append(f"<MECHANISM_INFO> {mech_info} </MECHANISM_INFO>")

                        # ----- Actions -> fixed-order array matching PEERS_CONTRIBUTIONS -----
                        rewards_sparse, punishs_sparse = _sparse_rewards_punishments(
                            focal_row=row, round_slice=rs, focal_pid=pid, avatars=avatars,
                            rewards_on=cfg["reward_exists"], punish_on=cfg["punishment_exists"]
                        )

                        peer_order = _peer_order(rs, pid, avatars)
                        signed_vec = [0] * len(peer_order)

                        for av, u in (rewards_sparse or {}).items():
                            if av in peer_order and u:
                                j = peer_order.index(av)
                                signed_vec[j] += int(u)
                        for av, u in (punishs_sparse or {}).items():
                            if av in peer_order and u:
                                j = peer_order.index(av)
                                signed_vec[j] -= int(u)

                        if cfg["reward_exists"] and not cfg["punishment_exists"]:
                            signed_vec = [max(0, v) for v in signed_vec]
                            lines.append(f"<REWARDS> <<{json.dumps(signed_vec)}>> </REWARDS>")
                        elif cfg["punishment_exists"] and not cfg["reward_exists"]:
                            signed_vec = [min(0, v) for v in signed_vec]
                            lines.append(f"<PUNISHMENTS> <<{json.dumps(signed_vec)}>> </PUNISHMENTS>")
                        else:
                            lines.append(f"<PUNISHMENTS_REWARDS> <<{json.dumps(signed_vec)}>> </PUNISHMENTS_REWARDS>")

                    # ----- Optional inbound identifiers -----
                    if cfg["punishment_exists"] and cfg["showPunishmentId"]:
                        pun_by = parse_dict(row.get("data.punishedBy", {})) or {}
                        if pun_by:
                            by_av = {avatars.get(str(k), f"Player {k}"): int(v) for k, v in pun_by.items() if v and int(v) > 0}
                            if by_av:
                                lines.append(f"<PUNISHED_BY json='{json.dumps(by_av, separators=(',', ':'))}'/>")
                    if cfg["reward_exists"] and cfg["showRewardId"]:
                        rew_by = parse_dict(row.get("data.rewardedBy", {})) or {}
                        if rew_by:
                            by_av = {avatars.get(str(k), f"Player {k}"): int(v) for k, v in rew_by.items() if v and int(v) > 0}
                            if by_av:
                                lines.append(f"<REWARDED_BY json='{json.dumps(by_av, separators=(',', ':'))}'/>")

                    # ----- ROUND SUMMARY -----
                    if cfg["punishment_exists"]:
                        f_pun_units = int(sum((parse_dict(row.get("data.punished", {})) or {}).values()))
                    else:
                        f_pun_units = 0
                    if cfg["reward_exists"]:
                        f_rew_units = int(sum((parse_dict(row.get("data.rewarded", {})) or {}).values()))
                    else:
                        f_rew_units = 0

                    summary_dict_focal, summary_dict = {}, {}
                    if cfg["punishment_exists"]:
                        summary_dict_focal["coins_spent_on_punish"] = f_pun_units * int(cfg["punishment_cost"] or 0)
                        summary_dict_focal["coins_deducted_from_you"] = int(row.get("data.penalties", 0) or 0)
                    if cfg["reward_exists"]:
                        summary_dict_focal["coins_spent_on_reward"] = f_rew_units * int(cfg["reward_cost"] or 0)
                        summary_dict_focal["coins_rewarded_to_you"] = int(row.get("data.rewards", 0) or 0)
                    summary_dict_focal["round_payoff"] = None if _is_nan(row.get("data.roundPayoff")) else int(float(row.get("data.roundPayoff")))
                    summary_dict[f"{focal_avatar} (YOU)"] = summary_dict_focal

                    if cfg["showOtherSummaries"]:
                        others_info = _others_summary(
                            rs, avatars,
                            punishment_exists=cfg["punishment_exists"],
                            punishment_cost=cfg["punishment_cost"],
                            reward_exists=cfg["reward_exists"],
                            reward_cost=cfg["reward_cost"],
                            focal_pid=pid
                        )
                        summary_dict.update(others_info)

                    lines.append(f"<ROUND SUMMARY json='{json.dumps(summary_dict, separators=(',', ':'))}'/>")

                    # ===== CHATS: summary phase (after round summary; before next round) =====
                    if chats_idx and idx in chats_idx and "summary" in chats_idx[idx]:
                        for av, txt in chats_idx[idx]["summary"]:
                            speaker = f"{av}{' (YOU)' if av == focal_avatar else ''}"
                            lines.append(f"<CHAT> {{{speaker}: {txt}}} </CHAT>")

                    lines.append("</ROUND>")

                lines.append("# GAME COMPLETE")
                text = "\n".join(lines)
                fout.write(json.dumps({"experiment": game_id, "participant": str(pid), "text": text}, ensure_ascii=False) + "\n")
                out_lines += 1

    print(f"Wrote {out_lines} transcripts to {out_path}")
    return out_path

if __name__ == "__main__":
    _ = generate_participant_transcript_chat(
        df_rounds=df_rounds_learn,
        df_players=df_players_learn,
        df_analysis=df_analysis_learn,
        df_chats=df_chats_learn,
        out_path="prompts_learn_chat.jsonl"
    )
