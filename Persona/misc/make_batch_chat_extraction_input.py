import argparse
import json
import re
from pathlib import Path

import pandas as pd


SYSTEM_PROMPT = (
    "You are a careful annotator extracting chat behavior features from a Public Goods Game transcript.\n"
    "Return ONLY valid JSON that strictly follows the schema provided.\n"
    "Do not speculate. Use only evidence present in the transcript.\n"
    "When a field cannot be determined, use null or 'unknown' (as specified)."
)

USER_INSTRUCTIONS = (
    "Identify the focal avatar as the one labeled '(YOU)'. Only focal's own chat counts toward chat_summary and chat_acts.\n"
    "Define phase within a round:\n"
    "- round_start: chat lines before focal's <CONTRIB> in that round\n"
    "- post_outcome_pre_sanction: after <PEERS_CONTRIBUTIONS> but before punishment/reward action lines\n"
    "- post_sanction: after punishment/reward action lines\n"
    "- unknown otherwise\n"
    "Defection observed event: any peer contribution equal to 0 (in all-or-nothing transcripts) or visibly much lower than others; use <PEERS_CONTRIBUTIONS>.\n"
    "Punished received: evidence via round summary json coins_deducted_from_you > 0 OR presence of punishedBy info in the transcript.\n"
    "Called out: another speaker mentions focal avatar in a negative/accusatory way (insult/shaming/threat directed at focal).\n"
    "Examples must be literal excerpts from chat messages (truncate to <= 20 words). Do not invent.\n"
    "If something cannot be determined from transcript, use null or 'unknown'. Do not guess.\\n"
    "For reaction_to_communication: exposures are ONLY other players' messages in that round.\\n"
    "Follow window for contribution is next round; for punishment is same round or next round.\\n"
    "Only use focal's messages for hypocrisy; only use others' messages for exposure detection.\\n"
    "Do not invent endowment; if not in transcript, treat endowment as unknown for defection definitions.\\n"
    "Set reaction_to_communication.definitions exactly as:\\n"
    "- follow_window_for_contribution: next_round\\n"
    "- follow_window_for_punishment: same_round_or_next_round\\n"
    "- exposure_definition: communication_by_others_in_round\\n"
    "- delta_definition: c_{t+1} - c_t\\n"
    "- defector_definition: peer contrib==0 if all-or-nothing; else contrib <= max(2, floor(0.1*endowment)) when endowment is present in transcript\\n"
    "Include reaction_to_communication and hypocrisy sections per the schema."
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


def load_chat_games(config_path: Path):
    df = pd.read_csv(config_path)
    if "gameId" not in df.columns or "CONFIG_chat" not in df.columns:
        raise ValueError("Config file missing gameId or CONFIG_chat columns")
    chat_games = set()
    for _, row in df.iterrows():
        game_id = str(row.get("gameId"))
        chat = _clean_bool(row.get("CONFIG_chat"))
        if chat is True:
            chat_games.add(game_id)
    return chat_games


def infer_focal_avatar_hint(text: str):
    # Look for patterns like "SLOTH (YOU)" in ROUND SUMMARY or other blocks
    patterns = [
        r'"([A-Z][A-Z_ ]+) \(YOU\)"',
        r'([A-Z][A-Z_ ]+) \(YOU\)',
    ]
    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            name = matches[0].strip()
            return name
    return None


def build_chat_focused_view(text: str):
    lines = text.splitlines()
    keep = []
    in_block = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("<ROUND "):
            keep.append(stripped)
            continue
        if stripped.startswith("</ROUND"):
            keep.append(stripped)
            continue

        if stripped.startswith("<CONTRIB>"):
            in_block = "CONTRIB"
            keep.append(stripped)
            continue
        if stripped.startswith("</CONTRIB>"):
            keep.append(stripped)
            in_block = None
            continue

        if stripped.startswith("<PUNISHMENT_REWARD>"):
            in_block = "PUNISHMENT_REWARD"
            keep.append(stripped)
            continue
        if stripped.startswith("</PUNISHMENT_REWARD>"):
            keep.append(stripped)
            in_block = None
            continue

        if stripped.startswith("<PEERS_CONTRIBUTIONS>"):
            keep.append(stripped)
            continue
        if stripped.startswith("<ROUND SUMMARY"):
            keep.append(stripped)
            continue
        if "<CHAT>" in stripped:
            keep.append(stripped)
            continue

        if in_block:
            keep.append(stripped)

    return "\n".join(keep)


def build_json_schema():
    # JSON Schema for Structured Outputs
    string_or_null = {"type": ["string", "null"]}
    number_or_null = {"type": ["number", "null"]}
    integer_or_null = {"type": ["integer", "null"]}

    example_obj = {
        "type": "object",
        "properties": {
            "round": {"type": "integer"},
            "quote": {"type": "string"},
        },
        "required": ["round", "quote"],
        "additionalProperties": False,
    }

    chat_act_example = {
        "type": "object",
        "properties": {
            "round": {"type": "integer"},
            "phase": {"type": "string"},
            "quote": {"type": "string"},
        },
        "required": ["round", "phase", "quote"],
        "additionalProperties": False,
    }

    per_round_message = {
        "type": "object",
        "properties": {
            "round": {"type": "integer"},
            "phase": {
                "type": "string",
                "enum": [
                    "round_start",
                    "post_outcome_pre_sanction",
                    "post_sanction",
                    "unknown",
                ],
            },
            "message_index_in_round": {"type": "integer"},
            "quote": {"type": "string"},
            "acts": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "coordination_call",
                        "commitment",
                        "threat",
                        "call_for_punishment",
                        "call_for_reward",
                        "encouragement_praise",
                        "insult_shaming",
                        "fairness_moralizing",
                        "coalition_framing",
                        "confusion_disagreement",
                        "callout_targeting",
                    ],
                },
            },
        },
        "required": ["round", "phase", "message_index_in_round", "quote", "acts"],
        "additionalProperties": False,
    }

    comm_type_enum = [
        "coordination_call",
        "commitment",
        "threat",
        "call_for_punishment",
        "encouragement_praise",
        "insult_shaming",
        "fairness_moralizing",
        "coalition_framing",
        "confusion_disagreement",
        "callout_targeting",
    ]

    reaction_contrib_example = {
        "type": "object",
        "properties": {
            "exposure_round": {"type": "integer"},
            "exposure_quote": {"type": "string"},
            "c_t": integer_or_null,
            "c_t1": integer_or_null,
            "delta": integer_or_null,
        },
        "required": ["exposure_round", "exposure_quote", "c_t", "c_t1", "delta"],
        "additionalProperties": False,
    }

    reaction_punish_example = {
        "type": "object",
        "properties": {
            "exposure_round": {"type": "integer"},
            "exposure_quote": {"type": "string"},
            "named_targets": {"type": "array", "items": {"type": "string"}},
            "punish_action_quote": string_or_null,
        },
        "required": ["exposure_round", "exposure_quote", "named_targets", "punish_action_quote"],
        "additionalProperties": False,
    }

    reaction_to_communication = {
        "type": "object",
        "properties": {
            "definitions": {
                "type": "object",
                "properties": {
                    "follow_window_for_contribution": {"type": "string"},
                    "follow_window_for_punishment": {"type": "string"},
                    "exposure_definition": {"type": "string"},
                    "delta_definition": {"type": "string"},
                    "defector_definition": {"type": "string"},
                },
                "required": [
                    "follow_window_for_contribution",
                    "follow_window_for_punishment",
                    "exposure_definition",
                    "delta_definition",
                    "defector_definition",
                ],
                "additionalProperties": False,
            },
            "per_comm_type": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "comm_type": {"type": "string", "enum": comm_type_enum},
                        "exposure_rounds": {"type": "array", "items": {"type": "integer"}},
                        "n_exposures": {"type": "integer"},
                        "contribution_response": {
                            "type": "object",
                            "properties": {
                                "n_usable": integer_or_null,
                                "delta_mean": number_or_null,
                                "delta_median": number_or_null,
                                "direction": {
                                    "type": "string",
                                    "enum": ["up", "down", "none", "mixed", "unknown"],
                                },
                                "alignment": {
                                    "type": "string",
                                    "enum": ["follow", "ignore", "oppose", "mixed", "unknown"],
                                },
                                "evidence_examples": {
                                    "type": "array",
                                    "items": reaction_contrib_example,
                                },
                            },
                            "required": [
                                "n_usable",
                                "delta_mean",
                                "delta_median",
                                "direction",
                                "alignment",
                                "evidence_examples",
                            ],
                            "additionalProperties": False,
                        },
                        "punishment_response": {
                            "type": "object",
                            "properties": {
                                "n_usable": integer_or_null,
                                "alignment": {
                                    "type": "string",
                                    "enum": ["follow", "ignore", "oppose", "mixed", "unknown"],
                                },
                                "follow_metrics": {
                                    "type": "object",
                                    "properties": {
                                        "punished_named_targets_same_round": {
                                            "type": "string",
                                            "enum": ["yes", "no", "unknown"],
                                        },
                                        "punished_defectors_same_round": {
                                            "type": "string",
                                            "enum": ["yes", "no", "unknown"],
                                        },
                                        "punish_units_same_round": integer_or_null,
                                        "punish_units_next_round": integer_or_null,
                                    },
                                    "required": [
                                        "punished_named_targets_same_round",
                                        "punished_defectors_same_round",
                                        "punish_units_same_round",
                                        "punish_units_next_round",
                                    ],
                                    "additionalProperties": False,
                                },
                                "evidence_examples": {
                                    "type": "array",
                                    "items": reaction_punish_example,
                                },
                            },
                            "required": ["n_usable", "alignment", "follow_metrics", "evidence_examples"],
                            "additionalProperties": False,
                        },
                    },
                    "required": [
                        "comm_type",
                        "exposure_rounds",
                        "n_exposures",
                        "contribution_response",
                        "punishment_response",
                    ],
                    "additionalProperties": False,
                },
            },
            "summary": {
                "type": "object",
                "properties": {
                    "overall_chat_responsiveness": {
                        "type": "string",
                        "enum": ["high", "medium", "low", "unknown"],
                    },
                    "most_influential_comm_types": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "overall_chat_responsiveness",
                    "most_influential_comm_types",
                    "notes",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["definitions", "per_comm_type", "summary"],
        "additionalProperties": False,
    }

    hypocrisy = {
        "type": "object",
        "properties": {
            "definition": {"type": "string"},
            "contribution_hypocrisy": {
                "type": "object",
                "properties": {
                    "commitments_total": integer_or_null,
                    "mismatches_total": integer_or_null,
                    "mismatch_rate": number_or_null,
                    "instances": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "round": {"type": "integer"},
                                "commitment_quote": {"type": "string"},
                                "promised_action": {
                                    "type": "string",
                                    "enum": [
                                        "max",
                                        "specific_amount",
                                        "contribute_more",
                                        "contribute_less",
                                        "unknown",
                                    ],
                                },
                                "observed_contribution": integer_or_null,
                                "endowment_hint": integer_or_null,
                                "mismatch": {"type": "boolean"},
                            },
                            "required": [
                                "round",
                                "commitment_quote",
                                "promised_action",
                                "observed_contribution",
                                "endowment_hint",
                                "mismatch",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["commitments_total", "mismatches_total", "mismatch_rate", "instances"],
                "additionalProperties": False,
            },
            "punishment_hypocrisy": {
                "type": "object",
                "properties": {
                    "commitments_total": integer_or_null,
                    "mismatches_total": integer_or_null,
                    "mismatch_rate": number_or_null,
                    "instances": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "round": {"type": "integer"},
                                "commitment_quote": {"type": "string"},
                                "promised_action": {
                                    "type": "string",
                                    "enum": [
                                        "punish_defectors",
                                        "punish_named_target",
                                        "punish_non_maxers",
                                        "unknown",
                                    ],
                                },
                                "named_targets": {"type": "array", "items": {"type": "string"}},
                                "observed_punishment": string_or_null,
                                "punished_any": {
                                    "type": "string",
                                    "enum": ["yes", "no", "unknown"],
                                },
                                "punished_named_targets": {
                                    "type": "string",
                                    "enum": ["yes", "no", "unknown"],
                                },
                                "mismatch": {"type": "boolean"},
                            },
                            "required": [
                                "round",
                                "commitment_quote",
                                "promised_action",
                                "named_targets",
                                "observed_punishment",
                                "punished_any",
                                "punished_named_targets",
                                "mismatch",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["commitments_total", "mismatches_total", "mismatch_rate", "instances"],
                "additionalProperties": False,
            },
            "note": {"type": "string"},
        },
        "required": ["definition", "contribution_hypocrisy", "punishment_hypocrisy", "note"],
        "additionalProperties": False,
    }

    schema = {
        "type": "object",
        "properties": {
            "gameId": {"type": "string"},
            "playerId": {"type": "string"},
            "focal_avatar": string_or_null,
            "rounds_total": integer_or_null,
            "chat_summary": {
                "type": "object",
                "properties": {
                    "messages_total": {"type": "integer"},
                    "rounds_with_message": {"type": "integer"},
                    "messages_per_round": number_or_null,
                    "messages_by_phase": {
                        "type": "object",
                        "properties": {
                            "round_start": {"type": "integer"},
                            "post_outcome_pre_sanction": {"type": "integer"},
                            "post_sanction": {"type": "integer"},
                            "unknown": {"type": "integer"},
                        },
                        "required": [
                            "round_start",
                            "post_outcome_pre_sanction",
                            "post_sanction",
                            "unknown",
                        ],
                        "additionalProperties": False,
                    },
                    "early_initiation": {
                        "type": "object",
                        "properties": {
                            "rounds_where_first_speaker": integer_or_null,
                            "rate_first_speaker_given_spoke": number_or_null,
                        },
                        "required": [
                            "rounds_where_first_speaker",
                            "rate_first_speaker_given_spoke",
                        ],
                        "additionalProperties": False,
                    },
                    "target_mentions": {
                        "type": "object",
                        "properties": {
                            "mentioned_avatars": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "counts_by_avatar": {
                                "type": "object",
                                "additionalProperties": {"type": "integer"},
                            },
                        },
                        "required": ["mentioned_avatars", "counts_by_avatar"],
                        "additionalProperties": False,
                    },
                },
                "required": [
                    "messages_total",
                    "rounds_with_message",
                    "messages_per_round",
                    "messages_by_phase",
                    "early_initiation",
                    "target_mentions",
                ],
                "additionalProperties": False,
            },
            "chat_conditions": {
                "type": "object",
                "properties": {
                    "style_overall": {
                        "type": "string",
                        "enum": [
                            "mostly_proactive",
                            "mostly_reactive",
                            "mixed",
                            "mostly_silent",
                            "completely_silent",
                            "unknown",
                        ],
                    },
                    "trigger_rates": {
                        "type": "object",
                        "properties": {
                            "after_defection_observed": number_or_null,
                            "after_punished_received": number_or_null,
                            "after_focal_punished_someone": number_or_null,
                            "after_called_out": number_or_null,
                            "after_all_cooperated": number_or_null,
                            "no_clear_trigger": number_or_null,
                        },
                        "required": [
                            "after_defection_observed",
                            "after_punished_received",
                            "after_focal_punished_someone",
                            "after_called_out",
                            "after_all_cooperated",
                            "no_clear_trigger",
                        ],
                        "additionalProperties": False,
                    },
                    "trigger_evidence_examples": {
                        "type": "object",
                        "properties": {
                            "after_defection_observed": {
                                "type": "array",
                                "items": example_obj,
                            },
                            "after_punished_received": {
                                "type": "array",
                                "items": example_obj,
                            },
                            "after_called_out": {
                                "type": "array",
                                "items": example_obj,
                            },
                            "proactive_coordination": {
                                "type": "array",
                                "items": example_obj,
                            },
                        },
                        "required": [
                            "after_defection_observed",
                            "after_punished_received",
                            "after_called_out",
                            "proactive_coordination",
                        ],
                        "additionalProperties": False,
                    },
                },
                "required": [
                    "style_overall",
                    "trigger_rates",
                    "trigger_evidence_examples",
                ],
                "additionalProperties": False,
            },
            "chat_acts": {
                "type": "object",
                "properties": {
                    "coordination_call": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "rounds": {"type": "array", "items": {"type": "integer"}},
                            "targets": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "max",
                                        "specific_amount",
                                        "fair_share",
                                        "unknown",
                                    ],
                                },
                            },
                            "examples": {"type": "array", "items": chat_act_example},
                        },
                        "required": ["count", "rounds", "targets", "examples"],
                        "additionalProperties": False,
                    },
                    "commitment": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "rounds": {"type": "array", "items": {"type": "integer"}},
                            "commitment_types": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "contribute",
                                        "max",
                                        "punish",
                                        "reward",
                                        "other",
                                        "unknown",
                                    ],
                                },
                            },
                            "examples": {"type": "array", "items": chat_act_example},
                        },
                        "required": ["count", "rounds", "commitment_types", "examples"],
                        "additionalProperties": False,
                    },
                    "threat": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "rounds": {"type": "array", "items": {"type": "integer"}},
                            "threat_targets": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "defectors",
                                        "specific_person",
                                        "group",
                                        "unknown",
                                    ],
                                },
                            },
                            "examples": {"type": "array", "items": chat_act_example},
                        },
                        "required": ["count", "rounds", "threat_targets", "examples"],
                        "additionalProperties": False,
                    },
                    "call_for_punishment": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "rounds": {"type": "array", "items": {"type": "integer"}},
                            "examples": {"type": "array", "items": chat_act_example},
                        },
                        "required": ["count", "rounds", "examples"],
                        "additionalProperties": False,
                    },
                    "call_for_reward": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "rounds": {"type": "array", "items": {"type": "integer"}},
                            "examples": {"type": "array", "items": chat_act_example},
                        },
                        "required": ["count", "rounds", "examples"],
                        "additionalProperties": False,
                    },
                    "encouragement_praise": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "rounds": {"type": "array", "items": {"type": "integer"}},
                            "examples": {"type": "array", "items": chat_act_example},
                        },
                        "required": ["count", "rounds", "examples"],
                        "additionalProperties": False,
                    },
                    "insult_shaming": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "rounds": {"type": "array", "items": {"type": "integer"}},
                            "examples": {"type": "array", "items": chat_act_example},
                        },
                        "required": ["count", "rounds", "examples"],
                        "additionalProperties": False,
                    },
                    "fairness_moralizing": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "rounds": {"type": "array", "items": {"type": "integer"}},
                            "examples": {"type": "array", "items": chat_act_example},
                        },
                        "required": ["count", "rounds", "examples"],
                        "additionalProperties": False,
                    },
                    "coalition_framing": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "rounds": {"type": "array", "items": {"type": "integer"}},
                            "examples": {"type": "array", "items": chat_act_example},
                        },
                        "required": ["count", "rounds", "examples"],
                        "additionalProperties": False,
                    },
                    "confusion_disagreement": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "rounds": {"type": "array", "items": {"type": "integer"}},
                            "examples": {"type": "array", "items": chat_act_example},
                        },
                        "required": ["count", "rounds", "examples"],
                        "additionalProperties": False,
                    },
                    "callout_targeting": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "rounds": {"type": "array", "items": {"type": "integer"}},
                            "named_targets": {"type": "array", "items": {"type": "string"}},
                            "examples": {"type": "array", "items": chat_act_example},
                        },
                        "required": ["count", "rounds", "named_targets", "examples"],
                        "additionalProperties": False,
                    },
                },
                "required": [
                    "coordination_call",
                    "commitment",
                    "threat",
                    "call_for_punishment",
                    "call_for_reward",
                    "encouragement_praise",
                    "insult_shaming",
                    "fairness_moralizing",
                    "coalition_framing",
                    "confusion_disagreement",
                    "callout_targeting",
                ],
                "additionalProperties": False,
            },
            "per_round_focal_messages": {
                "type": "array",
                "items": per_round_message,
            },
            "reaction_to_communication": reaction_to_communication,
            "hypocrisy": hypocrisy,
            "quality_flags": {
                "type": "object",
                "properties": {
                    "focal_avatar_found": {"type": "boolean"},
                    "round_parsing_confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "notes": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "focal_avatar_found",
                    "round_parsing_confidence",
                    "notes",
                ],
                "additionalProperties": False,
            },
        },
        "required": [
            "gameId",
            "playerId",
            "focal_avatar",
            "rounds_total",
            "chat_summary",
            "chat_conditions",
            "chat_acts",
            "per_round_focal_messages",
            "reaction_to_communication",
            "hypocrisy",
            "quality_flags",
        ],
        "additionalProperties": False,
    }

    return schema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument(
        "--transcripts",
        default="/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/Persona/transcripts_learn.jsonl",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional explicit config CSV path; default uses df_analysis_val.csv if present, else df_analysis_learn.csv",
    )
    parser.add_argument(
        "--output",
        default="/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/Persona/batch_chat_features_input.jsonl",
    )
    args = parser.parse_args()

    base_dir = Path("/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning")
    if args.config:
        config_path = Path(args.config)
    else:
        candidate = base_dir / "data" / "processed_data" / "df_analysis_val.csv"
        if candidate.exists():
            config_path = candidate
        else:
            config_path = base_dir / "data" / "processed_data" / "df_analysis_learn.csv"

    transcripts_path = Path(args.transcripts)
    transcripts_game_ids = set()
    with transcripts_path.open() as fin:
        for line in fin:
            obj = json.loads(line)
            transcripts_game_ids.add(str(obj.get("experiment")))

    chat_games = load_chat_games(config_path)
    if (
        config_path.name == "df_analysis_val.csv"
        and len(chat_games.intersection(transcripts_game_ids)) == 0
    ):
        fallback = base_dir / "data" / "processed_data" / "df_analysis_learn.csv"
        if fallback.exists():
            config_path = fallback
            chat_games = load_chat_games(config_path)

    schema = build_json_schema()

    output_path = Path(args.output)

    total = 0
    kept = 0
    skipped_missing_config = 0

    with transcripts_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            total += 1
            obj = json.loads(line)
            game_id = str(obj.get("experiment"))
            player_id = str(obj.get("participant"))
            text = obj.get("text", "")

            if game_id not in chat_games:
                skipped_missing_config += 1
                continue

            focal_hint = infer_focal_avatar_hint(text)
            chat_view = build_chat_focused_view(text)

            user_parts = [
                f"gameId: {game_id}",
                f"playerId: {player_id}",
            ]
            if focal_hint:
                user_parts.append(f"focal_avatar_hint: {focal_hint}")
            user_parts.append("\nTRANSCRIPT (chat-focused view):\n" + chat_view)
            user_parts.append("\nOUTPUT JSON schema follows.\n" + USER_INSTRUCTIONS)

            request_obj = {
                "custom_id": f"{game_id}__{player_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": "\n".join(user_parts)},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "chat_features",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                },
            }

            fout.write(json.dumps(request_obj))
            fout.write("\n")
            kept += 1

    print(f"Total transcripts: {total}")
    print(f"Kept (chat enabled): {kept}")
    print(f"Skipped (not in chat games): {skipped_missing_config}")
    print(f"Config file used: {config_path}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
