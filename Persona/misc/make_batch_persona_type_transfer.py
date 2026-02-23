import argparse
import json
from pathlib import Path

import pandas as pd


SYSTEM_PROMPT = (
    "You are an expert behavioral analyst of Public Goods Game (PGG) players. "
    "Write a detailed, evidence-grounded persona type for the focal player in this game. "
    "Use only the provided source persona and configurations; do not invent facts. "
    "When uncertain, say 'unknown' or note low confidence."
)

USER_INSTRUCTIONS = (
    "Task: given a source persona from a prior PGG environment, generate an alternative persona for the same player type "
    "under the TARGET_GAME_CONFIG. Preserve stable traits and motivations, but adapt behavior, incentives, and expectations "
    "to the new environment. Do not contradict the source persona; if adaptation is unclear, say 'unknown' or note low "
    "confidence. Ground every claim in the source persona and the target configuration, but DO NOT cite specific rounds, "
    "quotes, or numeric evidence. The generated type is intended to serve as part of a prompt defining a player's type in new "
    "PGG environments. Use present tense throughout. Avoid redundancy; do not repeat the same trait or rationale across "
    "sections. Focus on the focal player described in SOURCE_PERSONA.\n"
    "Output format: free-form text, but you MUST include each header line listed under REQUIRED_HEADERS. "
    "Each required header must appear exactly once as its own line, followed by relevant details.\n"
    "Suggested content to cover (free-form): a short type label/name, a detailed overview, incentives, strategy across rounds, "
    "beliefs about others, social/personality style, response to others' actions, learning/adaptation."
)


def _clean_bool(val):
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        if val == 1:
            return True
        if val == 0:
            return False
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
    return None


def load_config_map(config_path: Path):
    df = pd.read_csv(config_path)
    cfg_map = {}
    for _, row in df.iterrows():
        game_id = str(row.get("gameId"))
        cfg_map[game_id] = {
            "playerCount": row.get("CONFIG_playerCount"),
            "numRounds": row.get("CONFIG_numRounds"),
            "showNRounds": _clean_bool(row.get("CONFIG_showNRounds")),
            "endowment": row.get("CONFIG_endowment"),
            "multiplier": row.get("CONFIG_multiplier"),
            "MPCR": row.get("CONFIG_MPCR"),
            "allOrNothing": _clean_bool(row.get("CONFIG_allOrNothing")),
            "chat": _clean_bool(row.get("CONFIG_chat")),
            "punishmentExists": _clean_bool(row.get("CONFIG_punishmentExists")),
            "punishmentCost": row.get("CONFIG_punishmentCost"),
            "punishmentMagnitude": row.get("CONFIG_punishmentMagnitude"),
            "punishmentTech": row.get("CONFIG_punishmentTech"),
            "rewardExists": _clean_bool(row.get("CONFIG_rewardExists")),
            "rewardCost": row.get("CONFIG_rewardCost"),
            "rewardMagnitude": row.get("CONFIG_rewardMagnitude"),
            "rewardTech": row.get("CONFIG_rewardTech"),
            "showOtherSummaries": _clean_bool(row.get("CONFIG_showOtherSummaries")),
            "showPunishmentId": _clean_bool(row.get("CONFIG_showPunishmentId")),
            "showRewardId": _clean_bool(row.get("CONFIG_showRewardId")),
        }
    return cfg_map


def build_required_headers(cfg):
    headers = ["<CONTRIBUTION>"]
    if cfg.get("chat") is True:
        headers.append("<COMMUNICATION>")
    if cfg.get("showNRounds") is True:
        headers.append("<RESPONSE_TO_END_GAME>")
    if cfg.get("punishmentExists") is True:
        headers.append("<PUNISHMENT>")
        if cfg.get("showPunishmentId") is True:
            headers.append("<RESPONSE_TO_PUNISHER>")
    if cfg.get("rewardExists") is True:
        headers.append("<REWARD>")
        if cfg.get("showRewardId") is True:
            headers.append("<RESPONSE_TO_REWARDER>")
    if cfg.get("showOtherSummaries") is True:
        headers.append("<RESPONSE_TO_OTHERS_OUTCOME>")
    return headers


def build_header_expectations(required_headers):
    mapping = {
        "<CONTRIBUTION>": (
            "general contribution tendency, stability vs volatility, conditionality, fairness norms, risk tolerance."
        ),
        "<COMMUNICATION>": (
            "tone, coordination attempts, persuasion, responsiveness to others, avoidance/engagement style."
        ),
        "<PUNISHMENT>": (
            "willingness to punish, triggers, selectivity, deterrence vs retaliation motives."
        ),
        "<REWARD>": (
            "willingness to reward, triggers, strategic reinforcement, gratitude signaling."
        ),
        "<RESPONSE_TO_END_GAME>": (
            "end-game effects on strategy, cooperation decay or escalation, horizon awareness."
        ),
        "<RESPONSE_TO_OTHERS_OUTCOME>": (
            "reactions to others' payoff outcomes, including costs spent and deducted/granted via punishment/reward."
        ),
        "<RESPONSE_TO_PUNISHER>": (
            "response when punished and when punishers are identifiable (forgiveness, retaliation, compliance)."
        ),
        "<RESPONSE_TO_REWARDER>": (
            "response when rewarded and when rewarders are identifiable (reciprocity, alliance building)."
        ),
    }
    lines = ["Header expectations (include but not limited to):"]
    for header in required_headers:
        detail = mapping.get(header)
        if detail:
            lines.append(f"- {header}: {detail}")
    return "\n".join(lines)


def build_config_description(cfg):
    parts = []
    parts.append(
        f"This is a {cfg.get('playerCount')}-player PGG with {cfg.get('numRounds')} rounds."
    )
    parts.append(
        f"Each round, the endowment is {cfg.get('endowment')} coins. "
        f"Contributions are multiplied by {cfg.get('multiplier')} "
        f"(MPCR {cfg.get('MPCR')})."
    )
    if cfg.get("chat") is True:
        parts.append("Chat is enabled.")
    else:
        parts.append("Chat is disabled.")
    if cfg.get("punishmentExists") is True:
        parts.append(
            f"Punishment is available: it costs {cfg.get('punishmentCost')} coins "
            f"to impose {cfg.get('punishmentMagnitude')} coins of deduction "
            f"(tech {cfg.get('punishmentTech')})."
        )
    else:
        parts.append("Punishment is not available.")
    if cfg.get("rewardExists") is True:
        parts.append(
            f"Reward is available: it costs {cfg.get('rewardCost')} coins "
            f"to grant {cfg.get('rewardMagnitude')} coins of reward "
            f"(tech {cfg.get('rewardTech')})."
        )
    else:
        parts.append("Reward is not available.")
    if cfg.get("showNRounds") is True:
        parts.append("Players can see how many rounds remain.")
    else:
        parts.append("Players cannot see how many rounds remain.")
    if cfg.get("showOtherSummaries") is True:
        outcome_details = ["payoffs"]
        if cfg.get("punishmentExists") is True:
            outcome_details.append("punishment costs and deductions")
        if cfg.get("rewardExists") is True:
            outcome_details.append("reward costs and grants")
        details_text = ", ".join(outcome_details)
        parts.append(
            "Players can see each other's end-of-round outcomes, including "
            f"{details_text}."
        )
    else:
        parts.append("Players cannot see other players' end-of-round outcomes.")
    if cfg.get("showPunishmentId") is True:
        parts.append("Punishers are identifiable to the punished.")
    else:
        parts.append("Punishers are not identifiable.")
    if cfg.get("showRewardId") is True:
        parts.append("Rewarders are identifiable to the rewarded.")
    else:
        parts.append("Rewarders are not identifiable.")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Generate transfer persona prompts for all validation configs."
    )
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument(
        "--summaries",
        default="/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/Persona/summary_gpt51_learn.jsonl",
        help="Path to summary_gpt51_learn.jsonl.",
    )
    parser.add_argument(
        "--learn-config",
        default="/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/data/processed_data/df_analysis_learn.csv",
        help="Path to df_analysis_learn.csv.",
    )
    parser.add_argument(
        "--val-config",
        default="/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/data/processed_data/df_analysis_val_dedup.csv",
        help="Path to df_analysis_val_dedup.csv.",
    )
    parser.add_argument(
        "--output",
        default="/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/Persona/batch_persona_type_input_val_dedup_from_learn.jsonl",
        help="Output JSONL path for generated transfer persona prompts.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summaries)
    learn_cfg_path = Path(args.learn_config)
    val_cfg_path = Path(args.val_config)
    output_path = Path(args.output)

    learn_cfg_map = load_config_map(learn_cfg_path)
    val_cfg_map = load_config_map(val_cfg_path)

    val_cfg_list = []
    for game_id, cfg in val_cfg_map.items():
        required_headers = build_required_headers(cfg)
        header_block = "\n".join(required_headers) if required_headers else "None"
        val_cfg_list.append(
            {
                "game_id": game_id,
                "config_description": build_config_description(cfg),
                "header_block": header_block,
                "header_expectations": build_header_expectations(required_headers),
            }
        )

    val_cfg_list.sort(key=lambda x: x["game_id"])

    total = 0
    kept = 0
    missing_cfg = 0

    with summary_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            total += 1
            obj = json.loads(line)
            if obj.get("game_finished") is not True:
                continue
            old_game_id = str(obj.get("experiment"))
            old_cfg = learn_cfg_map.get(old_game_id)
            if old_cfg is None:
                missing_cfg += 1
                continue
            old_desc = build_config_description(old_cfg)
            persona_text = obj.get("text", "")
            participant = str(obj.get("participant"))

            for val_cfg in val_cfg_list:
                new_game_id = val_cfg["game_id"]
                user_parts = [
                    "\nSOURCE_GAME_CONFIG:\n" + old_desc,
                    "\nTARGET_GAME_CONFIG:\n" + val_cfg["config_description"],
                    "\nREQUIRED_HEADERS:\n" + val_cfg["header_block"],
                    "\nSOURCE_PERSONA:\n" + persona_text,
                    "\n" + val_cfg["header_expectations"],
                    "\n" + USER_INSTRUCTIONS,
                ]

                request_obj = {
                    "custom_id": f"{old_game_id}__{participant}__to__{new_game_id}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": "\n".join(user_parts)},
                        ],
                    },
                }

                fout.write(json.dumps(request_obj))
                fout.write("\n")
                kept += 1

    print(f"Total summary rows: {total}")
    print(f"Kept (finished) x val configs: {kept}")
    print(f"Missing learn config: {missing_cfg}")
    print(f"Val configs: {len(val_cfg_list)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
