from __future__ import annotations

import json
import math
import re
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
    punish_on = bool(env.get("CONFIG_punishmentExists", False))
    reward_on = bool(env.get("CONFIG_rewardExists", False))
    if punish_on and reward_on:
        lines.append("After contributions are redistributed, players may punish or reward each other.")
    elif punish_on:
        lines.append("After contributions are redistributed, players may punish each other.")
    elif reward_on:
        lines.append("After contributions are redistributed, players may reward each other.")
    lines.append("Always respond with ONLY valid single-line JSON per the required format at the END of each prompt.")
    if env.get("CONFIG_chat", False):
        lines.append("At the start of each round, you may optionally send ONE short message to the group.")
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
    if float(env.get("CONFIG_defaultContribProp", 0.0) or 0.0) > 0.0:
        pre = (
            f"{endow} coins start in the shared pot and you can choose to take some for yourself. "
            f"Choose how many coins to leave in the pot for contribution. Valid choice: {contrib_mode}."
        )
    else:
        pre = f"You have {endow} coins in your pocket. Choose how many to contribute to the shared pot ({contrib_mode})."
    mult = env.get("CONFIG_multiplier", "Unknown")
    return f"<ROUND_INFO> {pre} The pot multiplier is {mult}×. </ROUND_INFO>"


def chat_stage_line(env: Dict) -> str:
    return "<CHAT_STAGE> Would you like to send a message to the group? You may stay silent. Speak only when it helps. </CHAT_STAGE>"


def format_contrib_answer(val) -> str:
    base = str(val)
    return f"<CONTRIB> {base} </CONTRIB>"


def contrib_format_line(env: Dict, include_reasoning: bool) -> str:
    contrib_hint = "Set 'contribution' to the amount left in the pot (your contribution)."
    if include_reasoning:
        reasoning_hint = "Provide reasons for your action, keep it brief."
        fmt = '{"stage":"contribution","reasoning":<string>,"contribution":<int>}'
    else:
        reasoning_hint = ""
        fmt = '{"stage":"contribution","contribution":<int>}'
    return (
        f"RULES: {contrib_hint} {reasoning_hint}\n"
        "FORMAT (JSON ONLY): "
        f"{fmt}\n"
        "YOUR RESPONSE:"
    )


def actions_format_line(tag: str, include_reasoning: bool) -> str:
    if tag == "PUNISHMENT":
        dict_hint = "Use 'actions' as a dict mapping avatar -> nonnegative punishment units; omit zeros."
    elif tag == "REWARD":
        dict_hint = "Use 'actions' as a dict mapping avatar -> nonnegative reward units; omit zeros."
    else:
        dict_hint = (
            "Use 'actions' as a dict mapping avatar -> integer units; omit zeros. "
            "Negative=punishment, positive=reward."
        )
    if include_reasoning:
        reasoning_hint = "Provide reasons for your action, keep it brief."
        fmt = '{"stage":"actions","reasoning":<string>,"actions":{...}}'
    else:
        reasoning_hint = ""
        fmt = '{"stage":"actions","actions":{...}}'
    return (
        f"RULES: {dict_hint} {reasoning_hint}\n"
        "FORMAT (JSON ONLY): "
        f"{fmt}\n"
        "YOUR RESPONSE:"
    )


def chat_format_line(include_reasoning: bool) -> str:
    if include_reasoning:
        reasoning_hint = "Provide reasons for your action, keep it brief."
        fmt = '{"stage":"chat","reasoning":<string>,"chat":<string|null>}'
    else:
        reasoning_hint = ""
        fmt = '{"stage":"chat","chat":<string|null>}'
    return (
        "RULES: Set 'chat' to a short message, or null/empty string if silent. "
        f"{reasoning_hint}\n"
        "FORMAT (JSON ONLY): "
        f"{fmt}\n"
        "YOUR RESPONSE:"
    )


def max_tokens_reminder_line(max_tokens: int) -> str:
    return f"MAX_TOKENS: You may generate at most {max_tokens} tokens."


def extract_reasoning(gen: str) -> str:
    if not isinstance(gen, str):
        return ""
    match = re.search(r"<\s*Reasoning\s*>(.*?)</\s*Reasoning\s*>", gen, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


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
        return "PUNISHMENT/REWARD"
    if env.get("CONFIG_punishmentExists", False):
        return "PUNISHMENT"
    if env.get("CONFIG_rewardExists", False):
        return "REWARD"
    return None


def format_actions_answer(tag: str, actions: Dict[str, int]) -> str:
    base = json.dumps(actions, separators=(",", ":"))
    return f"<{tag}> {base} </{tag}>"
