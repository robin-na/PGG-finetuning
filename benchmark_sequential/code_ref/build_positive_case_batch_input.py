from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd


SYSTEM_PROMPT = """We have conducted multiple public goods game experiments with varying experimental designs, to measure the effect of punishment in cooperative settings under various environments.
Your task is to predict how enabling a punishment mechanism to a specific game changes the ***efficiency*** compared to the same game with punishment disabled.
According to our experiments, whether punishment increases efficiency or not is highly dependent on a lot of dimensions in experiment design, and it is your job to navigate this heterogeneity and make accurate predictions.

***Efficiency*** is the ratio between the game players' behavior and that of a fully-cooperative group (i.e. a group in which all members contribute their full endowment in every round)
In other words, efficiency measures how close a group's total payoff is, compared to that of a group that always cooperated (i.e. always contributes the entire endowment, and benefits maximally from the multiplier).
An efficiency value of 100% means that a group earned the same amount of coins as a hypothetical group that always cooperated.

For example, let's say a game has 5 players playing 10 rounds where 20 coins are given to each player per round and the multiplier for each contributed coin is 3.
In this case, the earning of a hypothetical "always cooperating" group is 5*10*20*3=3000 coins, while the earning of a hypothetical "never cooperating" group is 1000 coins.
Hence, the efficiency is 100% for the always cooperating group and 33% for the never cooperating group.

Your output should strictly be a prediction value with integer only (e.g., 33% should output 33 and nothing else)."""

SYSTEM_PROMPT_REASONING = """We have conducted multiple public goods game experiments with varying experimental designs, to measure the effect of punishment in cooperative settings under various environments.
Your task is to predict how enabling a punishment mechanism to a specific game changes the ***efficiency*** compared to the same game with punishment disabled.
According to our experiments, whether punishment increases efficiency or not is highly dependent on a lot of dimensions in experiment design, and it is your job to navigate this heterogeneity and make accurate predictions.

***Efficiency*** is the ratio between the game players' behavior and that of a fully-cooperative group (i.e. a group in which all members contribute their full endowment in every round)
In other words, efficiency measures how close a group's total payoff is, compared to that of a group that always cooperated (i.e. always contributes the entire endowment, and benefits maximally from the multiplier).
An efficiency value of 100% means that a group earned the same amount of coins as a hypothetical group that always cooperated.

For example, let's say a game has 5 players playing 10 rounds where 20 coins are given to each player per round and the multiplier for each contributed coin is 3.
In this case, the earning of a hypothetical "always cooperating" group is 5*10*20*3=3000 coins, while the earning of a hypothetical "never cooperating" group is 1000 coins.
Hence, the efficiency is 100% for the always cooperating group and 33% for the never cooperating group.

Respond with a JSON object only:
{
  "reasoning": "brief explanation of how you derived the prediction",
  "prediction": <integer efficiency percent>
}
The value in "prediction" must be an integer with no percent sign."""

REPORT_PROMPTS = {
    "both": """Below is a prediction-support report synthesized from both data analysis and paper evidence, discussing how configuration parameters affect punishment treatment effects on efficiency.
Make predictions based faithfully on implications from the report.""",
    "data_only": """Below is a prediction-support report synthesized from data analysis only, discussing how configuration parameters affect punishment treatment effects on efficiency.
Make predictions based faithfully on implications from the report.""",
    "paper_only": """Below is a prediction-support report synthesized from paper evidence only, discussing how configuration parameters affect punishment treatment effects on efficiency.
Make predictions based faithfully on implications from the report.""",
}

DEFAULT_REPORT_PROMPT = """Below is a prediction-support report discussing how configuration parameters affect punishment treatment effects on efficiency.
Make predictions based faithfully on implications from the report."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build OpenAI batch JSONL prediction inputs from "
            "input/pgg_CONFIGmerged_validation.csv, augmented by agentic reports."
        )
    )
    parser.add_argument(
        "--df-pgg",
        type=Path,
        default=Path("input/pgg_CONFIGmerged_validation.csv"),
        help="Path to validation configuration CSV (df_pgg equivalent).",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("positive_cases/output"),
        help="Directory containing both/data_only/paper_only agentic reports.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["both", "data_only", "paper_only"],
        help="Report variants to use (default: both data_only paper_only).",
    )
    parser.add_argument(
        "--report-filename",
        default="agentic_report.md",
        help="Report filename inside each variant folder.",
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=None,
        help=(
            "Path for merged JSONL containing all selected variants. "
            "If omitted, uses a platform-specific default path under openAI_batch_input/."
        ),
    )
    parser.add_argument(
        "--model-tag",
        default=None,
        help=(
            "Tag embedded in default output filename suffix. "
            "If omitted, inferred from model/platform."
        ),
    )
    parser.add_argument(
        "--platform",
        choices=["openai", "anthropic"],
        default="openai",
        help="Target batch platform/output format.",
    )
    parser.add_argument(
        "--custom-id-prefix",
        default="science-paper",
        help="Prefix for custom IDs. Final format: <prefix>_<variant>/Q<instance>.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name for the selected platform. "
            "Defaults: openai=gpt-5.2, anthropic=claude-sonnet-4-6."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for completion requests.",
    )
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help=(
            "If set, require JSON output with keys 'reasoning' and 'prediction', "
            "disable logprobs, and force temperature to 1.0."
        ),
    )
    parser.add_argument(
        "--anthropic-max-tokens",
        type=int,
        default=None,
        help=(
            "Anthropic-only: max_tokens for each request. "
            "Defaults: 512 in reasoning mode, else 64."
        ),
    )
    parser.add_argument(
        "--anthropic-version",
        default="2023-06-01",
        help="Anthropic API version header to use when submitting batches.",
    )
    return parser.parse_args()


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return bool(text)


def make_predict_prompt(
    config: str, augmented_text: str = "", include_reasoning: bool = False
) -> str:
    if include_reasoning:
        closing = f"""Now, predict the efficiency of the game below when punishment is to be enabled.
### Game Information ###
{config}

Return JSON with exactly these fields:
- "reasoning": concise rationale grounded in the report and game configuration
- "prediction": integer efficiency percentage only (no % sign)
"""
        return augmented_text + closing

    closing = f"""Now, predict the efficiency of the game below when punishment is to be enabled.
### Game Information ###
{config}

You predict that enabling punishment will cause the efficiency percentage to change to (output should be an integer and nothing else):
"""
    return augmented_text + closing


def make_config(row: pd.Series) -> str:
    game_structure = f"""
***The efficiency of this game with punishment disabled was:*** {int(round(100 * row['efficiency_np'], 0))}%

[CONFIGURATION]

*** Game Structure ***
Number of players: {int(row['CONFIG_playerCount'])}
Number of rounds: {int(row['CONFIG_numRounds'])}
Is chat enabled among players?: {_as_bool(row['CONFIG_chat'])}
Is the contribution "all or nothing" i.e., binary instead of continuous?: {_as_bool(row['CONFIG_allOrNothing'])}
Is contribution the default i.e., does each player's endowment start in the public fund for them to opt-out?: {_as_bool(row['CONFIG_defaultContribProp'])}

*** Monetary Stakes ***
Marginal per capita return (MPCR): {row['CONFIG_MPCR']}

*** Peer Incentives ***
    """

    punishment = f"""
Punishment cost to impose a single unit of punishment: {int(row['CONFIG_punishmentCost'])} coin(s)
Punishment impact (number of coins deducted from the punished player per coin spent punishing): {float(row['CONFIG_punishmentTech'])}
    """

    reward = f"""
Reward cost to grant a single unit of reward: {int(row['CONFIG_rewardCost'])} coin(s)
Reward impact (the coins awarded to a player per coin spent rewarding): {float(row['CONFIG_rewardTech'])}
    """

    information_display = f"""
*** Information Display ***
Is the number of rounds known to players (do they know when the game ends)?: {_as_bool(row['CONFIG_showNRounds'])}
Are peer outcomes shown (do players know how much their peers gained at the end of each round)?: {_as_bool(row['CONFIG_showOtherSummaries'])}"""

    information_punishment = f"""
When a player is punished/rewarded, are the punishers/rewarders known?: {_as_bool(row['CONFIG_showPunishmentId'])}
    """

    no_reward = """
Reward mechanism is not enabled.
    """

    if _as_bool(row["CONFIG_rewardExists"]):
        return game_structure + punishment + reward + information_display + information_punishment
    return game_structure + punishment + no_reward + information_display + information_punishment


def append_prompt(dataframe: pd.DataFrame, include_reasoning: bool = False) -> list[str]:
    return [
        make_predict_prompt(make_config(row), include_reasoning=include_reasoning)
        for _, row in dataframe.iterrows()
    ]


def append_from_text(
    text: str,
    messages: list[str],
    instruction: str,
    section_label: str = "Report",
) -> list[str]:
    appended_messages: list[str] = []
    for message in messages:
        appended = f"""{instruction}
----------{section_label} Starts----------

{text}

----------{section_label} Ends----------
{message}"""
        appended_messages.append(appended)
    return appended_messages


def build_prediction_requests(
    prompts: list[str],
    report_text: str,
    variant_name: str,
    platform: str,
    model: str,
    temperature: float,
    instruction: str,
    custom_id_prefix: str,
    include_reasoning: bool,
    anthropic_max_tokens: int,
) -> list[dict]:
    requests: list[dict] = []
    augmented_prompts = append_from_text(
        text=report_text,
        messages=prompts,
        instruction=instruction,
        section_label=f"{variant_name} agentic report",
    )
    custom_id_head = f"{custom_id_prefix}_{variant_name}".strip("_")
    for index, prompt in enumerate(augmented_prompts, start=1):
        if platform == "anthropic":
            safe_head = re.sub(r"[^a-zA-Z0-9_-]", "_", custom_id_head)
            suffix = f"_Q{index}"
            # Anthropic requires <=64 chars for custom_id.
            max_head_len = max(1, 64 - len(suffix))
            custom_id = f"{safe_head[:max_head_len]}{suffix}"
        else:
            custom_id = f"{custom_id_head}/Q{index}"
        temp = 1.0 if include_reasoning else temperature
        system_prompt = SYSTEM_PROMPT_REASONING if include_reasoning else SYSTEM_PROMPT

        if platform == "anthropic":
            requests.append(
                {
                    "custom_id": custom_id,
                    "params": {
                        "model": model,
                        "max_tokens": anthropic_max_tokens,
                        "temperature": temp,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                }
            )
        else:
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temp,
            }
            if include_reasoning:
                body["response_format"] = {"type": "json_object"}
            else:
                body["logprobs"] = True
                body["top_logprobs"] = 20

            requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )
    return requests


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _required_columns() -> set[str]:
    return {
        "efficiency_np",
        "CONFIG_playerCount",
        "CONFIG_numRounds",
        "CONFIG_chat",
        "CONFIG_allOrNothing",
        "CONFIG_defaultContribProp",
        "CONFIG_MPCR",
        "CONFIG_punishmentCost",
        "CONFIG_punishmentTech",
        "CONFIG_rewardCost",
        "CONFIG_rewardTech",
        "CONFIG_showNRounds",
        "CONFIG_showOtherSummaries",
        "CONFIG_showPunishmentId",
        "CONFIG_rewardExists",
    }


def _validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = sorted(_required_columns() - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {', '.join(missing)}"
        )


def _instruction_for_variant(variant: str) -> str:
    return REPORT_PROMPTS.get(variant, DEFAULT_REPORT_PROMPT)


def _load_report_text(report_path: Path) -> str:
    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")
    return report_path.read_text(encoding="utf-8")


def _sanitize_model_tag(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "", value).lower()
    return cleaned or "model"


def _resolve_model(args: argparse.Namespace) -> str:
    if args.model:
        return args.model
    if args.platform == "anthropic":
        return "claude-sonnet-4-6"
    return "gpt-5.2"


def _resolve_model_tag(args: argparse.Namespace, model: str) -> str:
    if args.model_tag:
        return args.model_tag
    if args.platform == "openai" and model == "gpt-5.2":
        return "52"
    if args.platform == "anthropic":
        if model == "claude-opus-4-6":
            return "opus46"
        if model == "claude-sonnet-4-6":
            return "sonnet46"
        return _sanitize_model_tag(model)
    return _sanitize_model_tag(model)


def _resolve_anthropic_max_tokens(args: argparse.Namespace) -> int:
    if args.anthropic_max_tokens is not None:
        if args.anthropic_max_tokens < 1:
            raise ValueError("--anthropic-max-tokens must be at least 1.")
        return args.anthropic_max_tokens
    return 512 if args.include_reasoning else 64


def _resolve_merged_output_path(
    args: argparse.Namespace, model_tag: str, platform: str
) -> Path:
    if args.merged_output is not None:
        return args.merged_output
    if platform == "anthropic":
        base_name = "prediction_positive_cases_reasoning_anthropic_merged"
        if not args.include_reasoning:
            base_name = "prediction_positive_cases_anthropic_merged"
        return Path("openAI_batch_input") / f"{base_name}_{model_tag}.json"
    base_name = "prediction_positive_cases_reasoning_merged"
    if not args.include_reasoning:
        base_name = "prediction_positive_cases_merged"
    return Path("openAI_batch_input") / f"{base_name}_{model_tag}.jsonl"


def main() -> None:
    args = parse_args()
    model = _resolve_model(args)
    model_tag = _resolve_model_tag(args, model)
    anthropic_max_tokens = _resolve_anthropic_max_tokens(args)

    df_pgg = pd.read_csv(args.df_pgg)
    _validate_columns(df_pgg, args.df_pgg)
    base_prompts = append_prompt(df_pgg, include_reasoning=args.include_reasoning)

    all_requests: list[dict] = []

    for variant in args.variants:
        input_name = variant.strip()
        report_path = args.reports_root / input_name / args.report_filename
        report_text = _load_report_text(report_path)
        instruction = _instruction_for_variant(input_name)
        requests = build_prediction_requests(
            prompts=base_prompts,
            report_text=report_text,
            variant_name=input_name,
            platform=args.platform,
            model=model,
            temperature=args.temperature,
            instruction=instruction,
            custom_id_prefix=args.custom_id_prefix,
            include_reasoning=args.include_reasoning,
            anthropic_max_tokens=anthropic_max_tokens,
        )
        all_requests.extend(requests)
        print(f"Collected {len(requests)} requests for {input_name}")

    merged_output_path = _resolve_merged_output_path(
        args=args, model_tag=model_tag, platform=args.platform
    )
    if args.platform == "anthropic":
        write_json(merged_output_path, {"requests": all_requests})
        print(f"Wrote Anthropic batch payload with {len(all_requests)} requests to {merged_output_path}")
        print("Submit with:")
        print(
            "curl https://api.anthropic.com/v1/messages/batches "
            f"--header \"x-api-key: $ANTHROPIC_API_KEY\" "
            f"--header \"anthropic-version: {args.anthropic_version}\" "
            "--header \"content-type: application/json\" "
            f"--data '@{merged_output_path}'"
        )
    else:
        write_jsonl(merged_output_path, all_requests)
        print(f"Wrote {len(all_requests)} merged requests to {merged_output_path}")


if __name__ == "__main__":
    main()
