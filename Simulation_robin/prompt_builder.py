from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple


def system_header_lines(env: Dict, include_reasoning: bool) -> List[str]:
    endow = int(env.get("CONFIG_endowment", 0) or 0)
    mult = env.get("CONFIG_multiplier", "Unknown")
    lines: List[str] = []
    aon = bool(env.get("CONFIG_allOrNothing", False))
    contrib_mode = f"either 0 or {endow}" if aon else f"integer from 0 to {endow}"
    lines.append(
        "You are playing an online public goods game (PGG). Each round, you are given "
        f"{endow} coins and need to decide how many coins to put into the shared pot "
        f"({contrib_mode})."
    )
    lines.append("You will not see others' choices before you decide.")
    lines.append(f"The pot is multiplied by {mult}× and split equally among all players.")
    lines.append("Your round payoff is: coins you kept + your equal share of the multiplied pot.")
    if env.get("CONFIG_punishmentExists", False) or env.get("CONFIG_rewardExists", False):
        lines.append("After contributions are redistributed, players may punish and/or reward each other.")
    lines.append("Follow the stage instructions exactly.")
    lines.append("The required response format will be shown at the END of each prompt.")
    if include_reasoning:
        lines.append("When asked for reasoning, keep it brief and strategic.")
    if env.get("CONFIG_chat", False):
        lines.append("At the start of each round, you may optionally send ONE short message to the group.")
        lines.append("It is valid to stay silent; only speak when it helps.")
    return lines


def system_header(env: Dict, include_reasoning: bool) -> str:
    # Plain text header (no special tokens) for transcript compatibility.
    return "\n".join(system_header_lines(env, include_reasoning))


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
            f"{endow} coins start in the shared pot. Choose how many coins to take for yourself "
            f"(your contribution is what remains). Valid choice: {contrib_mode}."
        )
    else:
        pre = f"You have {endow} coins in your pocket. Choose how many to contribute to the shared pot ({contrib_mode})."
    mult = env.get("CONFIG_multiplier", "Unknown")
    return f"<ROUND_INFO> {pre} The pot multiplier is {mult}×. </ROUND_INFO>"


def chat_stage_line(env: Dict) -> str:
    return "<CHAT_STAGE> Would you like to send a message to the group? You may stay silent. </CHAT_STAGE>"


def format_contrib_answer(val, include_reasoning: bool) -> str:
    base = str(val)
    return f"Answer: {base}" if include_reasoning else base


def contrib_format_line(include_reasoning: bool) -> str:
    if include_reasoning:
        return "FORMAT: Reasoning: <short rationale> Answer: <single integer>"
    return "FORMAT: Output a single integer and nothing else."


def actions_format_line(tag: str, include_reasoning: bool) -> str:
    if include_reasoning:
        return "FORMAT: Reasoning: <short rationale> Answer: <JSON array of integers>"
    return "FORMAT: Output a JSON array of integers and nothing else."


def chat_format_line(include_reasoning: bool) -> str:
    if include_reasoning:
        return "FORMAT: Reasoning: <short rationale> Answer: <short message or SILENT>"
    return "FORMAT: Output a single short message, or SILENT if you choose to stay quiet."


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


def format_actions_answer(tag: str, vec: List[int], include_reasoning: bool) -> str:
    base = json.dumps([int(x) for x in vec])
    return f"Answer: {base}" if include_reasoning else base
