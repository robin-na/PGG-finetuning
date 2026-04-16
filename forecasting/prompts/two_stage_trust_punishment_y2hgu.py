from __future__ import annotations

import pandas as pd


def build_prompt(row: pd.Series, profile_block: str | None) -> tuple[str, str]:
    system = (
        "You forecast one participant's structured responses in a two-stage trustworthiness experiment. "
        "Use the experiment rules, role, condition, and any provided profile as priors. Return only valid JSON."
    )
    experiment_code = str(row["experiment_code"])
    role = str(row["role"])
    visibility = str(row["visibility"])
    action_context = str(row["action_context"])
    deliberation_dimension = str(row["deliberation_dimension"])
    schema_type = str(row["schema_type"])

    if experiment_code in {"E1", "E4"}:
        action_phrase = "help a victim of unfair treatment"
    else:
        action_phrase = "punish the unfair player"

    if deliberation_dimension == "cost":
        stage1_note = (
            "Before acting, Player A can either check the personal cost or decide without checking. "
            "The unknown personal cost lies between 0p and 10p, and if checked it is revealed to be 5p."
        )
    elif deliberation_dimension == "impact":
        stage1_note = (
            "Before acting, Player A can either check the impact on the target or decide without checking. "
            "The personal cost is fixed at 5p, the unknown impact lies between 1p and 30p, and if checked it is revealed to be 15p."
        )
    else:
        stage1_note = (
            "Player A decides whether to punish after the 5p cost is already known. "
            "The relevant process cue is whether the decision is fast or slow."
        )

    visibility_line = (
        "- Process observable: Player B can see how Player A arrived at the stage-1 decision."
        if visibility == "observable"
        else "- Process hidden: Player B sees only whether Player A acted, not the process by which that choice was made."
    )

    if role == "A":
        if schema_type == "role_a_check":
            output_lines = [
                "Return only JSON in this exact schema:",
                "{",
                '  "check": "YES",',
                '  "act": "YES",',
                '  "return_pct": 50',
                "}",
                "- `check` must be `YES` or `NO`.",
                f"- `act` must be `YES` or `NO`, meaning whether Player A chooses to {action_phrase}.",
                "- `return_pct` must be an integer from 0 to 100.",
            ]
        else:
            output_lines = [
                "Return only JSON in this exact schema:",
                "{",
                '  "decision_time_bucket": "FAST",',
                '  "act": "YES",',
                '  "return_pct": 50',
                "}",
                "- `decision_time_bucket` must be `FAST` or `SLOW`.",
                f"- `act` must be `YES` or `NO`, meaning whether Player A chooses to {action_phrase}.",
                "- `return_pct` must be an integer from 0 to 100.",
            ]
        user = "\n".join(
            [
                f"Forecast one participant's decisions in {experiment_code} as Player A.",
                "",
                "# STAGE 1 BACKGROUND",
                "- Player 1 trusted Player 2 with 10p.",
                "- The amount tripled to 30p.",
                "- Player 2 returned 0p.",
                f"- Player A has 10p and can choose whether to {action_phrase}.",
                f"- If Player A acts, the effect on the target is 15p in the checking variants.",
                stage1_note,
                "",
                "# STAGE 2 TRUST GAME",
                "- Player B has 10p and decides how much to send to Player A.",
                "- Any amount sent is tripled.",
                "- Player A returns some percentage of the tripled amount.",
                visibility_line,
                *([ "", profile_block ] if profile_block else []),
                "",
                "# OUTPUT",
                *output_lines,
            ]
        )
        return system, user

    if schema_type == "role_b_observable_check":
        output_lines = [
            "Return only JSON in this exact schema:",
            "{",
            '  "send_if_act_without_check": 5,',
            '  "send_if_act_after_check": 5,',
            '  "send_if_no_act_without_check": 5,',
            '  "send_if_no_act_after_check": 5',
            "}",
            "- Each value must be an integer from 0 to 10.",
        ]
    elif schema_type == "role_b_hidden_check":
        output_lines = [
            "Return only JSON in this exact schema:",
            "{",
            '  "send_if_act": 5,',
            '  "send_if_no_act": 5',
            "}",
            "- Each value must be an integer from 0 to 10.",
        ]
    elif schema_type == "role_b_observable_time":
        output_lines = [
            "Return only JSON in this exact schema:",
            "{",
            '  "send_if_act_fast": 5,',
            '  "send_if_no_act_fast": 5,',
            '  "send_if_act_slow": 5,',
            '  "send_if_no_act_slow": 5',
            "}",
            "- Each value must be an integer from 0 to 10.",
        ]
    else:
        output_lines = [
            "Return only JSON in this exact schema:",
            "{",
            '  "send_if_act": 5,',
            '  "send_if_no_act": 5',
            "}",
            "- Each value must be an integer from 0 to 10.",
        ]

    role_b_stage1 = [
        f"- Player A previously had the option to {action_phrase} after an unfair interaction.",
        stage1_note,
        visibility_line,
        "- Player B now plays a trust game with Player A using the strategy method.",
        "- Player B has 10p and chooses how much to send to Player A.",
        "- Any amount sent is tripled.",
        "- Player A then returns some percentage of the tripled amount.",
    ]

    user = "\n".join(
        [
            f"Forecast one participant's conditional trust decisions in {experiment_code} as Player B.",
            "",
            "# TASK",
            *role_b_stage1,
            *([ "", profile_block ] if profile_block else []),
            "",
            "# OUTPUT",
            *output_lines,
        ]
    )
    return system, user
