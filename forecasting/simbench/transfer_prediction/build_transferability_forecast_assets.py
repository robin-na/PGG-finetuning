from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

SELECTED_ROWS_PATH = (
    REPO_ROOT
    / "forecasting"
    / "simbench"
    / "metadata"
    / "simbenchpop__twin_persona_summary_batched_seed_0__n64__gpt_5_mini__us_only"
    / "selected_rows.csv"
)
OVERLAP_SUMMARY_PATH = (
    REPO_ROOT
    / "forecasting"
    / "simbench"
    / "results"
    / "simbenchpop__baseline_vs_persona_summary_overlap__gpt_5_mini"
    / "dataset_overlap_summary.csv"
)
SIGNIFICANCE_PATH = (
    REPO_ROOT
    / "forecasting"
    / "simbench"
    / "results"
    / "simbenchpop__baseline_vs_persona_summary_overlap__gpt_5_mini"
    / "simbench_score_significance_summary.csv"
)


TWIN_SOURCE_CARD = {
    "dataset_name": "Twin-2K-500",
    "evaluation_variant_name": "twin_persona_summary",
    "paper_title": (
        "Database Report: Twin-2K-500: A Data Set for Building Digital Twins of over "
        "2,000 People Based on Their Answers to over 500 Questions"
    ),
    "abstract_summary": (
        "The Twin-2K-500 study introduces a large, U.S.-based multiwave dataset designed "
        "for building digital twins of individual people. Participants answer more than "
        "500 questions spanning demographics, personality, cognition, economic preferences, "
        "behavioral games, and open-ended self-description. The paper argues that this "
        "breadth makes it possible to test whether a model can use rich prior information "
        "about a person to predict their future responses on held-out tasks."
    ),
    "source_population": (
        "N=2,058 U.S. participants who completed all four waves, recruited on Prolific "
        "from a U.S.-representative intake sample by age, sex, and ethnicity."
    ),
    "collection_schedule": (
        "Four survey waves collected in early 2025. Waves 1-3 contain the persona "
        "information used to build digital twins; wave 4 repeats selected tasks to "
        "measure test-retest accuracy."
    ),
    "measurement_scope": [
        "14 demographic questions",
        "19 personality tests spanning 26 constructs and 279 questions",
        "11 cognitive ability tests spanning 85 questions",
        "10 economic preference tests spanning 34 questions",
        "11 between-subject behavioral-economics experiments",
        "5 within-subject behavioral-economics experiments",
        "1 pricing study with 40 purchase decisions",
    ],
    "what_the_model_gets": [
        "A dry compressed summary of one Twin participant based on many earlier survey responses",
        "Demographics, trait scores, cognitive summaries, economic-preference summaries, and selected open-ended content",
        "No access to the held-out target-study answers at forecast time",
    ],
}


TASK_METADATA = {
    "ChaosNLI": {
        "source_title": "What Can We Learn from Collective Human Opinions on Natural Language Inference Data?",
        "abstract_summary": (
            "ChaosNLI was created to study collective human opinion in natural-language "
            "inference rather than forcing a single majority label. The dataset gathers 100 "
            "annotations per example for ambiguous items drawn from SNLI, MNLI, and alphaNLI, "
            "so the target is an empirical distribution of human judgments over entailment-style labels."
        ),
        "study_population": (
            "Qualified U.S. Mechanical Turk annotators. ChaosNLI collects 100 labels per "
            "example for ambiguous examples from SNLI, MNLI, and alphaNLI."
        ),
        "task_format": (
            "Three-way NLI judgment: definitely correct, definitely incorrect, or neither."
        ),
        "study_details": [
            "Built to preserve disagreement rather than collapse to a single gold label",
            "Focuses on ambiguous examples where human distributions are intrinsically broad",
            "The central object of prediction is a distribution of human judgments rather than a single correct label",
        ],
    },
    "Choices13k": {
        "source_title": (
            "Using large-scale experiments and machine learning to discover theories of human decision-making"
        ),
        "abstract_summary": (
            "This study runs the largest risky-choice experiment of its kind and uses the "
            "resulting data to test and improve interpretable theories of human decision-making. "
            "Participants repeatedly choose between lotteries with varying outcomes and probabilities, "
            "producing fine-grained evidence about how local gamble structure shapes decisions."
        ),
        "study_population": (
            "U.S.-based Amazon Mechanical Turk workers. The original dataset contains "
            "13,006 risky choice problems; participants saw 20 problems each and were paid "
            "a base amount plus a performance bonus tied to a realized gamble."
        ),
        "task_format": "Binary choice between gamble A and gamble B on each item.",
        "study_details": [
            "Each item changes outcome magnitudes and probabilities",
            "The participant goal is explicit bonus maximization",
            "The benchmark target is the population-level choice distribution for each gamble",
        ],
    },
    "ConspiracyCorr": {
        "source_title": "The sociodemographic correlates of conspiracism",
        "abstract_summary": (
            "This cross-national study examines how conspiracy beliefs vary with demographic "
            "and political factors across many countries. Rather than testing factual knowledge, "
            "it measures graded endorsement or rejection of conspiracy claims, creating response "
            "distributions that reflect both prior beliefs and uncertainty."
        ),
        "study_population": (
            "Cross-national survey data covering 20 countries and 26,416 participants in "
            "the study summarized by SimBench."
        ),
        "task_format": (
            "Five-way truth judgment about conspiracy statements: definitely true, probably "
            "true, probably false, definitely false, or don't know."
        ),
        "study_details": [
            "Measures endorsement of widely circulated conspiracy beliefs",
            "Target items are belief judgments, not fact quizzes with a single correct answer",
            "The source study is cross-national, but the current target slice is limited to U.S.-eligible cases",
        ],
    },
    "DICES": {
        "source_title": "DICES Dataset: Diversity in Conversational AI Evaluation for Safety",
        "abstract_summary": (
            "DICES studies diversity in conversational-AI safety evaluation by collecting "
            "crowd judgments about whether chatbot responses are unsafe because of identity-related bias. "
            "The study emphasizes that safety judgments vary systematically across raters and contexts, "
            "so evaluation should preserve disagreement rather than reduce everything to one label."
        ),
        "study_population": (
            "173 raters balanced across country (U.S. and India) and gender in the original "
            "dataset. The current SimBench run keeps the U.S. context rows."
        ),
        "task_format": (
            "Three-way safety judgment on a chatbot's last response: Yes / No / Unsure."
        ),
        "study_details": [
            "Raters assess whether the response is unsafe overall due to bias-related harms",
            "The original paper emphasizes variance and ambiguity in safety judgments rather than a single binary gold label",
            "The relevant target is the response distribution over raters, not a single adjudicated label",
        ],
    },
    "GlobalOpinionQA": {
        "source_title": "Towards Measuring the Representation of Subjective Global Opinions in Language Models",
        "abstract_summary": (
            "GlobalOpinionQA evaluates whether language models reflect subjective public opinions "
            "from many countries using survey questions from sources such as World Values Survey and "
            "Pew Global Attitudes. The core target is cross-country opinion distributions on controversial "
            "or value-laden questions rather than single correct answers."
        ),
        "study_population": (
            "Cross-national survey questions adapted from World Values Survey and Pew Global "
            "Attitudes Survey. The current SimBench evaluation uses the U.S.-eligible rows."
        ),
        "task_format": (
            "Multiple-choice public-opinion questions on social, political, and geopolitical topics."
        ),
        "study_details": [
            "Built to compare language-model opinions to country-level human opinion distributions",
            "Question content is subjective and norm-laden rather than objectively correct",
            "Country context is part of the original study design and shapes the opinion distribution",
        ],
    },
    "ISSP": {
        "source_title": "International Social Survey Programme",
        "abstract_summary": (
            "ISSP is a long-running cross-national survey program designed for comparability "
            "across countries and years on topics such as religion, inequality, work, health care, "
            "and social networks. It is fundamentally a comparative survey instrument, so question "
            "meaning is tied to institutional context, country, and survey wave."
        ),
        "study_population": (
            "Cross-national annual surveys coordinated across many countries since 1984. "
            "The current SimBench evaluation uses the U.S.-eligible ISSP rows only."
        ),
        "task_format": (
            "Multiple-choice survey responses on politics, religion, health care, inequality, and social life."
        ),
        "study_details": [
            "Designed for cross-national and cross-time comparison",
            "Questions often include country-specific institutional context and survey-year context",
            "Interpretation depends heavily on country and survey-wave context",
        ],
    },
    "Jester": {
        "source_title": "Jester Datasets for Recommender Systems and Collaborative Filtering Research",
        "abstract_summary": (
            "Jester is a large-scale joke-rating dataset collected through a live recommender "
            "system. Users rate jokes on a continuous funniness scale, making the task a direct "
            "measurement of taste and entertainment preference rather than reasoning or knowledge."
        ),
        "study_population": (
            "Users of the UC Berkeley Jester joke recommender system; millions of ratings "
            "from a large volunteer user base."
        ),
        "task_format": (
            "Continuous joke-funniness ratings from -10 to +10, binned by SimBench into 10 ranges."
        ),
        "study_details": [
            "A taste and preference task rather than a knowledge or reasoning task",
            "The original platform was a live recommender system, not a one-shot survey",
            "The response variable is hedonic taste rather than correctness or ideology",
        ],
    },
    "MoralMachine": {
        "source_title": "The Moral Machine experiment",
        "abstract_summary": (
            "The Moral Machine experiment collects large-scale judgments about autonomous-vehicle "
            "moral dilemmas from participants around the world. Its main contribution is showing that "
            "moral preferences vary systematically across scenarios and cultures, with distributions "
            "that encode value tradeoffs rather than factual performance."
        ),
        "study_population": (
            "Large-scale global online participants on the Moral Machine website. The "
            "current SimBench evaluation uses the U.S. country slice."
        ),
        "task_format": (
            "Binary choice between two accident outcomes in autonomous-vehicle moral dilemmas."
        ),
        "study_details": [
            "The task asks participants to choose between two harms under imminent crash conditions",
            "Country context matters in the original study and is part of the moral-preference interpretation",
            "Response distributions reflect moral tradeoffs rather than factual correctness",
        ],
    },
    "NumberGame": {
        "source_title": "A Large Dataset of Generalization Patterns in the Number Game",
        "abstract_summary": (
            "The Number Game dataset studies how people generalize from a small set of numbers "
            "to a hidden rule. Participants judge whether a new number belongs in the same concept, "
            "capturing a mix of rule induction, similarity-based reasoning, and uncertainty."
        ),
        "study_population": (
            "U.S. participants in a numerical generalization task; SimBench describes 575 "
            "U.S. participants for the current source."
        ),
        "task_format": (
            "Binary judgment of whether a target number likely follows the hidden rule that generated example numbers."
        ),
        "study_details": [
            "Responses reflect both rule-based and similarity-based generalization",
            "Items vary by the seed set and the candidate target number",
            "The task depends on inductive reasoning under uncertainty rather than on explicit social values",
        ],
    },
    "OpinionQA": {
        "source_title": "Whose Opinions Do Language Models Reflect?",
        "abstract_summary": (
            "OpinionQA introduces a benchmark for comparing language-model responses to public-opinion "
            "distributions from many U.S. demographic groups. The study shows that model opinions can be "
            "substantially misaligned with human groups and that even explicit demographic steering does not "
            "fully close the gap."
        ),
        "study_population": (
            "U.S. public opinion questions derived from Pew Research Center's American Trends "
            "Panel, spanning many demographics and survey waves."
        ),
        "task_format": (
            "Multiple-choice survey questions on politics, society, technology, religion, and public affairs."
        ),
        "study_details": [
            "The original OpinionQA benchmark studies alignment with U.S. demographic groups",
            "Question content is broad and spans politics, technology, religion, and social life",
            "The current overlap comparison uses only the rows available in both baseline and persona-summary runs",
        ],
    },
    "OSPsychBig5": {
        "source_title": "Open-Source Psychometrics Project: Big Five Personality Test",
        "abstract_summary": (
            "This task comes from the OpenPsychometrics IPIP Big-Five self-assessment, where "
            "participants rate agreement with statements intended to measure extraversion, "
            "agreeableness, conscientiousness, neuroticism, and openness. It is a self-report "
            "instrument, so sample frame and item wording matter alongside latent trait differences."
        ),
        "study_population": (
            "Self-selected users of OpenPsychometrics completing the IPIP Big-Five Factor Markers."
        ),
        "task_format": (
            "Five-point agreement ratings on personality self-description statements."
        ),
        "study_details": [
            "Uses IPIP Big-Five Factor Markers based on Goldberg (1992)",
            "OpenPsych users are not a representative U.S. sample and the website is explicitly educational/entertainment-oriented",
            "Sample frame and instrument wording matter alongside the underlying trait distribution",
        ],
    },
    "OSPsychMACH": {
        "source_title": "Open-Source Psychometrics Project: MACH-IV Machiavellianism Test",
        "abstract_summary": (
            "This task uses the MACH-IV scale to measure self-reported manipulativeness, cynicism, "
            "and strategic amorality. The data come from self-selected OpenPsychometrics users rather "
            "than a representative panel, so the benchmark reflects both trait structure and the "
            "idiosyncrasies of that sample frame."
        ),
        "study_population": "Self-selected OpenPsychometrics users completing the MACH-IV scale.",
        "task_format": "Five-point agreement ratings on Machiavellianism statements.",
        "study_details": [
            "Based on Christie and Geis (1970)",
            "Measures manipulativeness, cynicism, and amoral pragmatism through self-report",
            "Only one U.S.-eligible overlap row is present in the current SimBench comparison",
        ],
    },
    "OSPsychMGKT": {
        "source_title": "Open-Source Psychometrics Project: Multifactor General Knowledge Test",
        "abstract_summary": (
            "The Multifactor General Knowledge Test measures broad factual and cultural knowledge "
            "across many domains. In SimBench, multi-answer questions are broken into yes/no subitems, "
            "so the task becomes a sequence of knowledge judgments with an objectively correct structure."
        ),
        "study_population": (
            "Self-selected OpenPsychometrics users. The website notes that the MGKT is most "
            "valid for internet users from the United States."
        ),
        "task_format": (
            "Binary yes/no judgments derived from multi-answer general-knowledge questions."
        ),
        "study_details": [
            "The original MGKT uses multi-select questions with penalties for wrong choices",
            "SimBench converts subquestions into yes/no items",
            "Question content ranges across culture, geography, medicine, computing, and history",
        ],
    },
    "OSPsychRWAS": {
        "source_title": "Open-Source Psychometrics Project: Right Wing Authoritarianism Scale",
        "abstract_summary": (
            "The RWAS instrument measures authoritarianism-related attitudes through agreement "
            "ratings on statements about obedience, tradition, punishment, and social order. "
            "It is a value-laden self-report task in which ideological orientation is central."
        ),
        "study_population": "Self-selected OpenPsychometrics users completing the RWAS instrument.",
        "task_format": (
            "Nine-point agreement ratings on authoritarianism, conformity, and social-order statements."
        ),
        "study_details": [
            "Based on Altemeyer (1981, 2007)",
            "The instrument directly targets ideology-adjacent value positions",
            "A person's political and moral orientation is central to item responses",
        ],
    },
    "TISP": {
        "source_title": (
            "Perceptions of science, science communication, and climate change attitudes in 68 countries - the TISP dataset"
        ),
        "abstract_summary": (
            "TISP is a large cross-national survey about trust in science, science communication, "
            "and climate-related attitudes. It combines value-laden and institution-laden items, "
            "with strong country and cultural context in both sampling and interpretation."
        ),
        "study_population": (
            "71,922 participants across 68 countries in the Many Labs TISP survey; the current "
            "SimBench evaluation uses the U.S.-eligible rows only."
        ),
        "task_format": (
            "Likert-style questions about trust in scientists, science communication, and climate attitudes."
        ),
        "study_details": [
            "Global cross-national survey with translations and quota-weighted sampling",
            "Includes scientific trust, populist attitudes toward science, media use, and policy attitudes",
            "Country and cultural context are deeply entangled with the response distributions",
        ],
    },
    "WisdomOfCrowds": {
        "source_title": "Stanford Policy Lab: wisdom-of-crowds study repository",
        "abstract_summary": (
            "The wisdom-of-crowds dataset studies how collective human judgments perform across "
            "many tasks, including analogies, arithmetic, and common-sense questions. The slice "
            "used in SimBench is closer to problem-solving and factual inference than to opinion polling."
        ),
        "study_population": (
            "Large online study with nearly 2,000 participants and over 500,000 responses "
            "across multiple domains; SimBench uses the U.S. MTurk multiple-choice slice."
        ),
        "task_format": (
            "Multiple-choice problem-solving items such as analogies, arithmetic, and common-sense questions."
        ),
        "study_details": [
            "The broader study spans text, image, video, and audio tasks",
            "SimBench uses a subset of the multiple-choice text-style questions",
            "Unlike survey tasks, many items have objectively correct answers",
        ],
    },
}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _trim(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _collect_examples(selected_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    examples_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_payloads: dict[str, set[str]] = defaultdict(set)
    for row in selected_rows:
        dataset_name = row["dataset_name"]
        if len(examples_by_task[dataset_name]) >= 2:
            continue
        payload = row["question_payload"]
        if payload in seen_payloads[dataset_name]:
            continue
        seen_payloads[dataset_name].add(payload)
        examples_by_task[dataset_name].append(
            {
                "question_excerpt": _trim(payload, 420),
                "question_body_excerpt": _trim(row["question_body"], 420),
                "options": json.loads(row["option_text_map_json"]),
            }
        )
    return dict(examples_by_task)


def _build_task_cards(
    selected_rows: list[dict[str, str]],
    overlap_rows: list[dict[str, str]],
    significance_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    examples_by_task = _collect_examples(selected_rows)
    rows_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected_rows:
        rows_by_task[row["dataset_name"]].append(row)

    overlap_by_task = {row["dataset_name"]: row for row in overlap_rows}
    significance_by_task = {row["dataset_name"]: row for row in significance_rows}

    task_cards: list[dict[str, object]] = []
    ground_truth: list[dict[str, object]] = []

    for dataset_name in sorted(overlap_by_task):
        if dataset_name not in TASK_METADATA:
            raise KeyError(f"Missing metadata for task: {dataset_name}")
        metadata = TASK_METADATA[dataset_name]
        rows = rows_by_task[dataset_name]
        overlap = overlap_by_task[dataset_name]
        significance = significance_by_task.get(dataset_name, {})

        group_sizes = [int(float(row["group_size"])) for row in rows if row.get("group_size")]
        labels = significance.get("significant_bh_0_05", "")
        direction = significance.get("direction", "")
        if labels == "True" and direction == "persona_summary_better":
            true_label = "positive"
        elif labels == "True" and direction == "baseline_better":
            true_label = "negative"
        else:
            true_label = "insignificant"

        delta = float(overlap["delta_simbench_score_persona_summary_minus_baseline"])
        task_cards.append(
            {
                "dataset_name": dataset_name,
                "source_title": metadata["source_title"],
                "abstract_summary": metadata["abstract_summary"],
                "study_population": metadata["study_population"],
                "task_format": metadata["task_format"],
                "study_details": metadata["study_details"],
                "us_eval_rows": int(float(overlap["n_rows"])),
                "us_group_size_range": {
                    "min": min(group_sizes),
                    "max": max(group_sizes),
                },
                "examples": examples_by_task.get(dataset_name, []),
            }
        )
        ground_truth.append(
            {
                "dataset_name": dataset_name,
                "delta_simbench_score_persona_summary_minus_baseline": delta,
                "true_label": true_label,
                "significant_bh_0_05": labels == "True",
            }
        )

    ground_truth.sort(
        key=lambda row: row["delta_simbench_score_persona_summary_minus_baseline"],
        reverse=True,
    )
    for rank, row in enumerate(ground_truth, start=1):
        row["true_rank"] = rank

    return task_cards, ground_truth


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def _render_twin_source_card() -> str:
    lines = [
        "# Twin Source Card",
        "",
        f"- Dataset: `{TWIN_SOURCE_CARD['dataset_name']}`",
        f"- Evaluation variant: `{TWIN_SOURCE_CARD['evaluation_variant_name']}`",
        f"- Paper title: {TWIN_SOURCE_CARD['paper_title']}",
        "",
        "## Study Frame",
        "",
        f"- Abstract-style summary: {TWIN_SOURCE_CARD['abstract_summary']}",
        f"- Source population: {TWIN_SOURCE_CARD['source_population']}",
        f"- Collection schedule: {TWIN_SOURCE_CARD['collection_schedule']}",
        "",
        "## Measurement Scope",
        "",
    ]
    lines.extend([f"- {entry}" for entry in TWIN_SOURCE_CARD["measurement_scope"]])
    lines.extend(
        [
            "",
            "## Information Available About Each Twin Person",
            "",
        ]
    )
    lines.extend([f"- {entry}" for entry in TWIN_SOURCE_CARD["what_the_model_gets"]])
    return "\n".join(lines) + "\n"


def _render_global_prompt(task_cards: list[dict[str, object]]) -> str:
    lines = [
        "# Global Ranking Prompt",
        "",
        "## System Prompt",
        "",
        "```text",
        "You are forecasting whether an external persona prior from one human study will help or hurt response-distribution prediction in another human-study benchmark.",
        "```",
        "",
        "## User Prompt",
        "",
        "```text",
        "You are given:",
        "1. A source-study description for Twin-2K-500 and the kind of information available about each person in that source study.",
        "2. A set of target benchmark task cards from the current US-only SimBenchPop comparison.",
        "",
        "Goal:",
        "Predict which target tasks will benefit from the external Twin persona-summary prior relative to the direct baseline, and which will not.",
        "",
        "Important:",
        "- Do not try to guess the exact numeric SimBench score.",
        "- Instead, produce:",
        "  (a) a ranking from most helped to most harmed, and",
        "  (b) one label for each task: positive, negative, or insignificant.",
        "- 'positive' means you expect the persona-summary transfer pipeline to improve mean SimBench score versus the baseline by a clearly meaningful amount for that task.",
        "- 'negative' means you expect the persona-summary transfer pipeline to hurt mean SimBench score versus the baseline by a clearly meaningful amount for that task.",
        "- 'insignificant' means you expect the difference to be small, noisy, mixed, or not clearly distinguishable.",
        "- Use only the study information provided below. Do not rely on remembered benchmark results.",
        "",
        "Output format:",
        "1. First write an Explanation section in plain text. Keep it concrete and task-specific.",
        "2. Then write a Final JSON section containing only valid JSON with this schema:",
        "{",
        '  "ranking_most_helpful_to_most_harmful": ["task1", "task2", "..."],',
        '  "task_predictions": [',
        '    {"dataset_name": "Choices13k", "predicted_label": "negative", "confidence": 0.0},',
        '    {"dataset_name": "ChaosNLI", "predicted_label": "positive", "confidence": 0.0}',
        "  ]",
        "}",
        "",
        "The confidence values should be between 0 and 1.",
        "",
        "Twin source-study information:",
        f"- Paper title: {TWIN_SOURCE_CARD['paper_title']}",
        f"- Abstract-style summary: {TWIN_SOURCE_CARD['abstract_summary']}",
        f"- Source population: {TWIN_SOURCE_CARD['source_population']}",
        f"- Collection schedule: {TWIN_SOURCE_CARD['collection_schedule']}",
        "- Measurement scope:",
    ]
    lines.extend([f"  - {entry}" for entry in TWIN_SOURCE_CARD["measurement_scope"]])
    lines.extend(
        [
            "- Information available about each Twin person:",
        ]
    )
    lines.extend([f"  - {entry}" for entry in TWIN_SOURCE_CARD["what_the_model_gets"]])
    lines.extend(["", "Target task cards:"])

    for card in task_cards:
        lines.extend(
            [
                "",
                f"Task: {card['dataset_name']}",
                f"- Source title: {card['source_title']}",
                f"- Abstract-style summary: {card['abstract_summary']}",
                f"- Study population: {card['study_population']}",
                f"- Task format: {card['task_format']}",
                f"- Current US-only evaluation rows: {card['us_eval_rows']}",
                (
                    f"- Human group-size range in the current US-only evaluation: "
                    f"{card['us_group_size_range']['min']} to {card['us_group_size_range']['max']}"
                ),
                "- Study details:",
            ]
        )
        lines.extend([f"  - {entry}" for entry in card["study_details"]])
        lines.append("- Example items:")
        for index, example in enumerate(card["examples"], start=1):
            option_preview = ", ".join(
                f"{key}: {value}" for key, value in list(example["options"].items())[:6]
            )
            lines.extend(
                [
                    f"  - Example {index} question: {example['question_excerpt']}",
                    f"  - Example {index} options: {option_preview}",
                ]
            )

    lines.extend(["```", ""])
    return "\n".join(lines)


def _render_pairwise_prompt(task_cards: list[dict[str, object]]) -> str:
    lines = [
        "# Pairwise Comparison Prompt",
        "",
        "## System Prompt",
        "",
        "```text",
        "You are comparing two target tasks and forecasting which one is more likely to benefit from an external persona prior from another human study.",
        "```",
        "",
        "## User Prompt Template",
        "",
        "```text",
        "You are given a source-study card for Twin-2K-500 and two target task cards.",
        "",
        "Goal:",
        "Decide which target task is more likely to benefit from transferring a Twin-based persona prior, relative to a direct baseline with no such prior.",
        "",
        "Output format:",
        "1. First write an Explanation section in plain text.",
        "2. Then write a Final JSON section containing only valid JSON with this schema:",
        "{",
        '  "more_likely_positive": "TaskA" | "TaskB" | "Tie",',
        '  "relative_confidence": 0.0,',
        '  "task_a_label": "positive" | "negative" | "insignificant",',
        '  "task_b_label": "positive" | "negative" | "insignificant"',
        "}",
        "",
        "Twin source-study information:",
        f"- Paper title: {TWIN_SOURCE_CARD['paper_title']}",
        f"- Abstract-style summary: {TWIN_SOURCE_CARD['abstract_summary']}",
        f"- Source population: {TWIN_SOURCE_CARD['source_population']}",
        f"- Collection schedule: {TWIN_SOURCE_CARD['collection_schedule']}",
        "- Measurement scope:",
    ]
    lines.extend([f"  - {entry}" for entry in TWIN_SOURCE_CARD["measurement_scope"]])
    lines.extend(["- Information available about each Twin person:"])
    lines.extend([f"  - {entry}" for entry in TWIN_SOURCE_CARD["what_the_model_gets"]])
    lines.extend(["", "Task A card:", "[Insert Task A card here]", "", "Task B card:", "[Insert Task B card here]", "```", ""])
    lines.extend(
        [
            "## Suggested Use",
            "",
            "- Use the global ranking prompt when you want a single-call ranking across all tasks.",
            "- Use this pairwise prompt when you want more stable relative judgments and are willing to aggregate many pairwise calls into a final ranking.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_readme(task_cards: list[dict[str, object]], ground_truth: list[dict[str, object]]) -> str:
    lines = [
        "# Transferability Forecast Assets",
        "",
        "This directory contains prompt assets for asking an LLM to forecast, before running a transfer experiment, which SimBench tasks will benefit from the Twin persona-summary prior.",
        "",
        "The forecast target is the current US-only `SimBenchPop` comparison:",
        "- Baseline: direct group-level prediction",
        "- Transfer arm: Twin `persona_summary` micro-simulation with `n=64` sampled personas",
        "",
        "The intended prediction target is not raw score regression. It is:",
        "- a ranking from most helped to most harmed, and",
        "- a 3-way class label per task: `positive`, `negative`, or `insignificant`.",
        "",
        "Ground truth is derived from the corrected paper-style SimBench score using the current overlap comparison in:",
        "- `forecasting/simbench/results/simbenchpop__baseline_vs_persona_summary_overlap__gpt_5_mini/`",
        "",
        "Files:",
        "- `twin_source_card.md`: detailed source-study card for Twin-2K-500 and the exact representation used here",
        "- `us16_task_cards.json`: factual study cards for the 16 US-only SimBenchPop tasks in the current comparison",
        "- `us16_ground_truth.json`: actual task deltas, ranks, and 3-way labels for evaluation",
        "- `us16_global_ranking_prompt.md`: a fully materialized prompt for ranking + 3-way labeling",
        "- `pairwise_comparison_prompt.md`: a pairwise prompt template for more stable relative judgments",
        "- `evaluate_transferability_forecast.py`: evaluates an LLM forecast against the ground truth",
        "",
        "Current task counts:",
    ]
    for card in task_cards:
        lines.append(f"- `{card['dataset_name']}`: {card['us_eval_rows']} overlap rows")
    lines.extend(
        [
            "",
            "Current realized labels:",
        ]
    )
    for row in ground_truth:
        lines.append(
            f"- `{row['dataset_name']}`: rank {row['true_rank']}, label `{row['true_label']}`, "
            f"delta {row['delta_simbench_score_persona_summary_minus_baseline']:.2f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    selected_rows = _load_rows(SELECTED_ROWS_PATH)
    overlap_rows = _load_rows(OVERLAP_SUMMARY_PATH)
    significance_rows = _load_rows(SIGNIFICANCE_PATH)

    task_cards, ground_truth = _build_task_cards(selected_rows, overlap_rows, significance_rows)

    _write_json(SCRIPT_DIR / "us16_task_cards.json", task_cards)
    _write_json(SCRIPT_DIR / "us16_ground_truth.json", ground_truth)
    (SCRIPT_DIR / "twin_source_card.md").write_text(_render_twin_source_card())
    (SCRIPT_DIR / "us16_global_ranking_prompt.md").write_text(
        _render_global_prompt(task_cards)
    )
    (SCRIPT_DIR / "pairwise_comparison_prompt.md").write_text(
        _render_pairwise_prompt(task_cards)
    )
    (SCRIPT_DIR / "README.md").write_text(_render_readme(task_cards, ground_truth))


if __name__ == "__main__":
    main()
