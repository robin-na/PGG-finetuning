from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple


def system_header_lines(env: Dict, include_reasoning: bool) -> List[str]:
    lines: List[str] = []
    lines.append("You are playing an online public goods game (PGG).")
    if include_reasoning:
        lines.append("For contributions, respond with two lines: Reasoning: <short rationale> then Answer: <CONTRIB> 3 </CONTRIB>.")
    else:
        lines.append("For contributions, output ONLY a single integer at the <CONTRIB> tag (no extra text).")
    if env.get("CONFIG_chat", False):
        lines.append("You can chat with other players during the round.")

    if env.get("CONFIG_punishmentExists", False) and env.get("CONFIG_rewardExists", False):
        lines.append("After contributions, decide whom to punish/reward and by how many units.")
        if include_reasoning:
            lines.append("Respond with two lines: Reasoning: <short rationale> then Answer: <PUNISHMENTS_REWARDS> <<[...]>> </PUNISHMENTS_REWARDS>.")
        lines.append(
            "At the <PUNISHMENTS_REWARDS> tag, output ONLY an array of integers aligned to the avatar order shown in <PEERS_CONTRIBUTIONS> (positive=rewards, negative=punishments, 0=neither)."
        )
    elif env.get("CONFIG_punishmentExists", False):
        lines.append("After contributions, decide whom to punish and by how many units.")
        if include_reasoning:
            lines.append("Respond with two lines: Reasoning: <short rationale> then Answer: <PUNISHMENTS> <<[...]>> </PUNISHMENTS>.")
        lines.append(
            "At the <PUNISHMENTS> tag, output ONLY an array of integers aligned to the avatar order shown in <PEERS_CONTRIBUTIONS>, each ≤ 0 (−n means punish by n units)."
        )
    elif env.get("CONFIG_rewardExists", False):
        lines.append("After contributions, decide whom to reward and by how many units.")
        if include_reasoning:
            lines.append("Respond with two lines: Reasoning: <short rationale> then Answer: <REWARDS> <<[...]>> </REWARDS>.")
        lines.append(
            "At the <REWARDS> tag, output ONLY an array of integers aligned to the avatar order shown in <PEERS_CONTRIBUTIONS>, each ≥ 0."
        )
    return lines


def system_header(env: Dict, include_reasoning: bool) -> str:
    lines = ["<|begin_of_text|><|start_header_id|>system<|end_header_id|>"]
    lines.extend(system_header_lines(env, include_reasoning))
    lines.append("<|eot_id|>")
    return "\n".join(lines)


def system_header_plain(env: Dict, include_reasoning: bool) -> str:
    return "\n".join(system_header_lines(env, include_reasoning))


def build_openai_messages(system_text: str, history_chunks: List[str]) -> List[Dict[str, str]]:
    history = "\n".join(history_chunks)
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": history},
    ]


def round_open(env: Dict, r: int) -> str:
    if env.get("CONFIG_showNRounds", False):
        return f'<ROUND i="{r} of {env["CONFIG_numRounds"]}">'
    return f'<ROUND i="{r}">'


def round_info_line(env: Dict) -> str:
    endow = int(env.get("CONFIG_endowment", 0) or 0)
    aon = bool(env.get("CONFIG_allOrNothing", False))
    contrib_mode = f"either 0 or {endow}" if aon else f"integer from 0 to {endow}"
    if env.get("CONFIG_defaultContribProp", False):
        pre = (
            f"{endow} coins are currently in the public fund, and you will contribute the remainder of the coins "
            f"you choose to take for yourself. Choose the amount to contribute ({contrib_mode})."
        )
    else:
        pre = f"{endow} coins are currently in your private pocket. Choose the amount to contribute ({contrib_mode})."
    mult = env.get("CONFIG_multiplier", "Unknown")
    return f"<ROUND_INFO> {pre} (multiplier: {mult}×). </ROUND_INFO>"


def contrib_open() -> str:
    return "<CONTRIB>"


def contrib_close_filled(val) -> str:
    return f"<CONTRIB> {val} </CONTRIB>"


def format_contrib_answer(val, include_reasoning: bool) -> str:
    base = contrib_close_filled(val)
    return f"Answer: {base}" if include_reasoning else base


def contrib_format_line() -> str:
    return "FORMAT: Reasoning: <short rationale> Answer: <CONTRIB> <<...>> </CONTRIB>"


def actions_format_line(tag: str) -> str:
    return f"FORMAT: Reasoning: <short rationale> Answer: <{tag}> <<[...]>> </{tag}>"


def extract_reasoning(gen: str) -> str:
    if not isinstance(gen, str):
        return ""
    text = gen
    if "Reasoning:" in text:
        text = text.split("Reasoning:", 1)[1]
    if "Answer:" in text:
        text = text.split("Answer:", 1)[0]
    return text.strip()


def redist_line(total_contrib: int, multiplied: float, active_players: int) -> str:
    m_str = f"{multiplied:.1f}" if isinstance(multiplied, (int, float)) and not math.isnan(multiplied) else ""
    per = (multiplied / active_players) if active_players > 0 else float("nan")
    per_str = f"{per:.1f}" if isinstance(per, (int, float)) and not math.isnan(per) else "NA"
    return (
        f'<REDIST total_contrib="{total_contrib}" multiplied_contrib="{m_str}" '
        f'active_players="{active_players}" redistributed_each="{per_str}"/>'
    )


def peers_contributions_csv(roster: List[str], focal: str, contrib_math: Dict[str, int]) -> Tuple[str, List[str]]:
    peer_order = [av for av in roster if av != focal]
    parts = [f"{av}={contrib_math.get(av, 'NA')}" for av in peer_order]
    return ",".join(parts), peer_order


def mech_info(env: Dict) -> Optional[str]:
    r_on = env.get("CONFIG_rewardExists", False)
    p_on = env.get("CONFIG_punishmentExists", False)
    if not (r_on or p_on):
        return None
    if r_on and p_on:
        return (
            f"It will cost you, per reward unit, {env['CONFIG_rewardCost']} coins to give a reward of {env['CONFIG_rewardMagnitude']} coins. "
            f"It will cost you, per punishment unit, {env['CONFIG_punishmentCost']} coins to impose a deduction of {env['CONFIG_punishmentMagnitude']} coins. "
            "Choose whom to punish/reward and by how many units."
        )
    if r_on:
        return (
            f"It will cost you, per unit, {env['CONFIG_rewardCost']} coins to give a reward of {env['CONFIG_rewardMagnitude']} coins. "
            "Choose whom to reward and by how many units."
        )
    return (
        f"It will cost you, per unit, {env['CONFIG_punishmentCost']} coins to impose a deduction of {env['CONFIG_punishmentMagnitude']} coins. "
        "Choose whom to punish and by how many units."
    )


def actions_tag(env: Dict) -> Optional[str]:
    if env.get("CONFIG_punishmentExists", False) and env.get("CONFIG_rewardExists", False):
        return "PUNISHMENTS_REWARDS"
    if env.get("CONFIG_punishmentExists", False):
        return "PUNISHMENTS"
    if env.get("CONFIG_rewardExists", False):
        return "REWARDS"
    return None


def actions_open_array(tag: str) -> str:
    return f"<{tag}> <<"


def actions_close_filled_array(tag: str, vec: List[int]) -> str:
    return f"{json.dumps([int(x) for x in vec])}>> </{tag}>"


def format_actions_answer(tag: str, vec: List[int], include_reasoning: bool) -> str:
    base = actions_close_filled_array(tag, vec)
    return f"Answer: {base}" if include_reasoning else base
