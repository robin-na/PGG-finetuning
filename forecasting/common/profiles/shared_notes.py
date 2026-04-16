from __future__ import annotations

from pathlib import Path


def _common_shared_note_lines() -> list[str]:
    return [
        "# Shared Prompt Notes",
        "",
        "## General Interpretation Note",
        "",
        "- These profiles summarize prior survey and behavioral-task evidence about each participant.",
        "- Treat the cues as relative tendencies, not deterministic predictions for any single decision in this study.",
        "- Unless a player-specific limit is listed, shared methodological caveats apply to all profiles.",
        "",
        "## Cue Glossary",
        "",
        "### Cooperation orientation",
        "- Meaning: blend of one-shot sharing behavior and cooperation/prosocial self-report.",
        "- Built from: trust-game sending, dictator giving, cooperation/competition items, agreeableness/helpfulness items, and prosocial values.",
        "",
        "### Conditional cooperation",
        "- Meaning: reciprocity and fairness-threshold sensitivity rather than a repeated-game reaction function.",
        "- Built from: trust-game return behavior and ultimatum acceptance-threshold signals.",
        "",
        "### Norm enforcement",
        "- Meaning: resistance to unfair splits and revenge/low-forgiveness cues in ultimatum-like contexts.",
        "- Built from: ultimatum minimum acceptable amounts plus revenge/forgiveness self-report.",
        "",
        "### Generosity without return",
        "- Meaning: willingness to give when repayment incentives are weak or absent.",
        "- Built from: dictator giving, trust-game sending, and prosocial/helpfulness cues.",
        "",
        "### Exploitation caution",
        "- Meaning: guardedness against being taken advantage of.",
        "- Built from: lower trustingness, stricter acceptance thresholds, uncertainty aversion, self-reliance, and revenge tendency.",
        "",
        "### Communication/coordination",
        "- Meaning: indirect cue for likely social expressiveness and coordination readiness.",
        "- Built from: empathy, social-sensitivity/self-monitoring, and extraversion-related self-report.",
        "",
        "### Behavioral stability",
        "- Meaning: rule-like internal consistency across self-regulation items.",
        "- Built from: conscientiousness-related items, self-concept clarity, and lower volatility-related personality items.",
    ]


def _dataset_specific_caveats(dataset_key: str) -> list[str]:
    if dataset_key == "minority_game_bret_njzas":
        return [
            "- Twin does not directly observe repeated minority-game switching, herding, or BRET-style box collection.",
            "- Trust, ultimatum, dictator, uncertainty-aversion, and self-regulation evidence may still transfer as broad priors about cooperation, guardedness, and consistency.",
            "- Communication/coordination is indirect: the source tasks do not directly observe repeated strategic group play.",
        ]
    if dataset_key == "longitudinal_trust_game_ht863":
        return [
            "- Twin includes direct one-shot trust-game evidence, which is relevant here, but this benchmark asks for repeated 1-9 willingness-to-play ratings rather than a single binary trust choice.",
            "- Norm-enforcement cues are secondary in this task because the focal decision is whether to enter a trust interaction, not whether to punish unfairness.",
            "- The repeated ten-session format is not directly observed in Twin, so use these cues as priors for the participant rather than as a literal trajectory template.",
        ]
    if dataset_key == "two_stage_trust_punishment_y2hgu":
        return [
            "- Twin has relevant trust, fairness, dictator, empathy, and uncertainty-aversion evidence, but it does not directly observe checking a cost or impact before acting.",
            "- Twin also does not directly observe deliberation-speed signaling, so fast versus slow should be treated as a coarse behavioral style inference rather than a measured trait.",
            "- Norm-enforcement and generosity cues are especially relevant here, but they are still indirect priors rather than exact predictions for punishment or helping in this design.",
        ]
    if dataset_key == "multi_game_llm_fvk2c":
        return [
            "- Twin overlaps closely with trust, ultimatum, and dictator-style evidence, but it does not directly observe AI delegation, stag-hunt choice, or five-option coordination choice.",
            "- AI-use decisions in this benchmark are therefore transfer tasks from broader social and decision-style evidence, not direct matches to Twin items.",
            "- Communication/coordination remains indirect because Twin does not directly observe interactive AI-supported social play.",
        ]
    raise ValueError(f"Unsupported dataset key: {dataset_key}")


def write_shared_notes_file(dataset_key: str, output_path: Path) -> None:
    lines = _common_shared_note_lines()
    caveats = _dataset_specific_caveats(dataset_key)
    insert_at = lines.index("## Cue Glossary")
    lines = [
        *lines[:insert_at],
        "## Shared Caveats",
        "",
        *[f"- {line[2:]}" if line.startswith("- ") else line for line in caveats],
        "",
        *lines[insert_at:],
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

