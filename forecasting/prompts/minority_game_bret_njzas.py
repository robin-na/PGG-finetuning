from __future__ import annotations

import pandas as pd


def build_prompt(row: pd.Series, profile_block: str | None) -> tuple[str, str]:
    system = (
        "You forecast one participant's full response in a repeated minority-game study. "
        "Use the task rules, condition, and any provided profile as priors. Return only valid JSON."
    )
    profile_lines = ["", profile_block] if profile_block else []
    user = "\n".join(
        [
            "Forecast one participant's behavior in this study.",
            "",
            "# TASKS",
            "This participant completed two tasks:",
            "1. An 11-round bonus game.",
            "2. A BRET risk task.",
            "",
            "# BONUS GAME RULES",
            "- On each round, the participant chooses `A` or `B`.",
            "- Round 1 always has pot 0.",
            "- From round 2 onward, the current pot depends only on the previous round's choice.",
            "- If the previous choice was `A`, the next-round pots are: 84, 88, 92, 96, 100, 104, 108, 112, 116, 116.",
            "- If the previous choice was `B`, the next-round pots are: 20, 40, 60, 80, 100, 120, 140, 160, 180, 180.",
            "- The crossover is at round 6, where both branches yield 100.",
            "- One round from rounds 2 to 11 is randomly selected for payment.",
            "",
            "# CONDITION",
            (
                "- Deception condition: the participant believed they were playing with real people, although the environment was scripted."
                if int(row["deception_condition"]) == 1
                else "- Non-deception condition: the scripted nature of the environment was not hidden in the same way."
            ),
            *profile_lines,
            "",
            "# BRET RULES",
            "- There are 100 boxes and exactly one hidden bomb.",
            "- The participant chooses how many boxes to collect, from 0 to 100.",
            "- Each collected box earns points, but if the bomb is among the collected boxes, the BRET payoff is 0.",
            "",
            "# OUTPUT",
            "Return only JSON in this exact schema:",
            "{",
            '  "bonus_game_choices": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],',
            '  "bret_boxes": 50',
            "}",
            "- `bonus_game_choices` must contain exactly 11 letters, each either `A` or `B`.",
            "- `bret_boxes` must be an integer from 0 to 100.",
        ]
    )
    return system, user
