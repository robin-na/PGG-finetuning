from __future__ import annotations

import json

import pandas as pd


def build_prompt(row: pd.Series, profile_block: str | None) -> tuple[str, str]:
    system = (
        "You forecast one participant's full decision battery in a multi-game experiment with possible AI delegation. "
        "Use the treatment description, scenario structure, and any provided profile as priors. Return only valid JSON."
    )
    treatment = str(row["Treatment"])
    treatment_code = str(row["TreatmentCode"])
    personalized = int(row["PersonalizedTreatment"])
    scenario_manifest = json.loads(str(row["scenario_manifest_json"]))

    treatment_lines: list[str] = []
    if treatment == "TransparentRandom":
        treatment_lines.extend(
            [
                "- TransparentRandom: AI support or replacement is introduced transparently and is not chosen by the participant.",
                "- The participant does not make voluntary delegation decisions in this treatment.",
            ]
        )
    elif treatment == "TransparentDelegation":
        treatment_lines.extend(
            [
                "- TransparentDelegation: the participant can voluntarily delegate each role to AI.",
                "- If the participant delegates a role, the counterpart knows that delegation occurred.",
            ]
        )
    else:
        treatment_lines.extend(
            [
                "- OpaqueDelegation: the participant can voluntarily delegate each role to AI.",
                "- If the participant delegates a role, the counterpart does not know whether AI was used.",
            ]
        )
    treatment_lines.append(
        "- The AI support is personalized to the participant."
        if personalized == 1
        else "- The AI support is generic rather than personalized."
    )

    scenario_lines = [
        "- Predict the participant's outputs for every scenario below in the listed order.",
    ]
    for item in scenario_manifest:
        scenario_text = str(item["scenario"])
        case_text = str(item["case"])
        if scenario_text == "AISupport":
            support_line = "AI support is present."
        else:
            support_line = "AI support is not present."
        if case_text == "AgainstHuman":
            case_line = "The counterpart is human."
        elif case_text == "AgainstAI":
            case_line = "The counterpart is AI."
        else:
            case_line = "The interaction is opaque with respect to whether the counterpart used AI."
        scenario_lines.append(
            f"- {item['order']}. scenario `{scenario_text}`, case `{case_text}`. {support_line} {case_line}"
        )

    user = "\n".join(
        [
            "Forecast one participant's full decision battery for this experimental run.",
            "",
            "# TREATMENT",
            f"- Treatment code: {treatment_code}",
            *treatment_lines,
            "",
            "# SCENARIOS TO PREDICT",
            *scenario_lines,
            "",
            "# GAME RULES",
            "- Ultimatum Game proposer: choose an offer from 0 to 10.",
            "- Ultimatum Game responder: choose a minimum acceptable offer from 0 to 10.",
            "- Trust Game sender: choose `YES` or `NO` for sending 2 ECU. If sent, it becomes 6 ECU for the receiver.",
            "- Trust Game receiver: if the sender sends, choose how much of 6 ECU to return, from 0 to 6.",
            "- Prisoner's Dilemma: choose `A` or `B` with payoffs (A,A)=5/5, (A,B)=1/8, (B,A)=8/1, (B,B)=3/3.",
            "- Stag Hunt: choose `X` or `Y` with payoffs (X,X)=8/8, (X,Y)=1/5, (Y,X)=5/1, (Y,Y)=4/4.",
            "- Coordination Game: choose one of Mercury, Venus, Earth, Mars, or Saturn. Matching the counterpart yields 5 each; mismatch yields 2 each.",
            *([ "", profile_block ] if profile_block else []),
            "",
            "# OUTPUT",
            "Return only JSON in this exact schema:",
            "{",
            '  "UGProposer_delegated": 0,',
            '  "UGResponder_delegated": 0,',
            '  "TGSender_delegated": 0,',
            '  "TGReceiver_delegated": 0,',
            '  "PD_delegated": 0,',
            '  "SH_delegated": 0,',
            '  "C_delegated": 0,',
            '  "scenario_outputs": [',
            '    {',
            '      "scenario": "AISupport",',
            '      "case": "AgainstHuman",',
            '      "UGProposer_decision": 5,',
            '      "UGResponder_decision": 3,',
            '      "TGSender_decision": "YES",',
            '      "TGReceiver_decision": 2,',
            '      "PD_decision": "A",',
            '      "SH_decision": "X",',
            '      "C_decision": "Earth"',
            "    }",
            "  ]",
            "}",
            "- In voluntary-delegation treatments, each `*_delegated` field records whether the participant delegated that role in the AI-support version of the treatment and must be `0` or `1`.",
            "- In `TransparentRandom`, every `*_delegated` field must be `null` because delegation is not the participant's choice.",
            f"- `scenario_outputs` must contain exactly {int(row['num_scenarios'])} objects in the order listed under `# SCENARIOS TO PREDICT`.",
            "- Each scenario object must repeat the exact `scenario` and `case` labels it is predicting.",
            "- In `AISupport` scenarios, if a role is delegated, set the corresponding `*_decision` field to `null`.",
            "- In `NoAISupport` scenarios, direct decisions may still be non-null even when the corresponding `*_delegated` field is `1`, because the row asks for the participant's own no-AI choice in that scenario.",
            "- `UGProposer_decision` and `UGResponder_decision` must be integers from 0 to 10 or `null`.",
            "- `TGSender_decision` must be `YES`, `NO`, or `null`.",
            "- `TGReceiver_decision` must be an integer from 0 to 6 or `null`.",
            "- `PD_decision` must be `A`, `B`, or `null`.",
            "- `SH_decision` must be `X`, `Y`, or `null`.",
            "- `C_decision` must be one of Mercury, Venus, Earth, Mars, Saturn, or `null`.",
        ]
    )
    return system, user
