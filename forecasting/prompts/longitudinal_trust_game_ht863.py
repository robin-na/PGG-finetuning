from __future__ import annotations

import pandas as pd


def build_prompt(row: pd.Series, profile_block: str | None) -> tuple[str, str]:
    system = (
        "You forecast one participant's full response panel in a repeated trust-game study. "
        "Use the task rules, repeated design, and any provided profile as priors. Return only valid JSON."
    )
    profile_lines = ["", profile_block] if profile_block else []
    user = "\n".join(
        [
            "Forecast one participant's full 10-session response panel.",
            "",
            "# TASK RULES",
            "- The same participant completes 10 sessions over 3 weeks.",
            "- Each session contains the same 16 trust-game trials.",
            "- On each trial, the participant sees a partner's past sharing rate and the number of tokens they would need to give.",
            "- The participant responds on a 1 to 9 scale: `1 = Not at all` and `9 = Extremely`.",
            "- If the participant does not play, both sides keep 5 tokens.",
            "- If the participant plays, they give `Y` tokens to the partner and the partner receives `2Y`.",
            "- If the partner shares, both sides end with `5 + Y / 2`.",
            "- If the partner keeps everything, the participant ends with `5 - Y`.",
            "",
            "# TRIAL ORDER",
            "Each day must use this exact 16-trial order:",
            "1.  partner shared 80%, stake 1",
            "2.  partner shared 80%, stake 2",
            "3.  partner shared 80%, stake 4",
            "4.  partner shared 80%, stake 5",
            "5.  partner shared 75%, stake 1",
            "6.  partner shared 75%, stake 2",
            "7.  partner shared 75%, stake 4",
            "8.  partner shared 75%, stake 5",
            "9.  partner shared 70%, stake 1",
            "10. partner shared 70%, stake 2",
            "11. partner shared 70%, stake 4",
            "12. partner shared 70%, stake 5",
            "13. partner shared 65%, stake 1",
            "14. partner shared 65%, stake 2",
            "15. partner shared 65%, stake 4",
            "16. partner shared 65%, stake 5",
            *profile_lines,
            "",
            "# OUTPUT",
            "Return only JSON in this exact schema:",
            "{",
            '  "days": [',
            '    {"day": 1, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},',
            '    {"day": 2, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},',
            '    {"day": 3, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},',
            '    {"day": 4, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},',
            '    {"day": 5, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},',
            '    {"day": 6, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},',
            '    {"day": 7, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},',
            '    {"day": 8, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},',
            '    {"day": 9, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},',
            '    {"day": 10, "ratings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}',
            "  ]",
            "}",
            "- There must be exactly 10 day objects, for days 1 through 10.",
            "- Each `ratings` list must contain exactly 16 integers.",
            "- Every rating must be an integer from 1 to 9.",
        ]
    )
    return system, user
