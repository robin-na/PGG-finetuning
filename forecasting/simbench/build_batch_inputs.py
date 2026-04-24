from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecasting.common.profiles import load_twin_cards, load_twin_personas
from forecasting.common.runs.non_pgg import MODEL_SLUGS, _estimate_input_tokens


VARIANT_BASELINE_PAPER = "baseline_group_batched_paper"
VARIANT_BASELINE_EXPLAINED = "baseline_group_batched_explained"
VARIANT_TWIN_PROFILE = "twin_profile_batched_seed_0"
ALL_VARIANTS = [VARIANT_BASELINE_EXPLAINED, VARIANT_TWIN_PROFILE]
SUPPORTED_VARIANTS = [VARIANT_BASELINE_PAPER, VARIANT_BASELINE_EXPLAINED, VARIANT_TWIN_PROFILE]
ALL_MODELS = ["gpt-5.1", "gpt-5-mini"]
ALL_SPLITS = ["SimBenchPop", "SimBenchGrouped"]

COUNTRY_KEYS = {
    "country",
    "country_name",
    "Country",
    "COUNTRY_NAME",
    "UserCountry3",
    "cntry",
    "rater_locale",
}
US_MARKERS = {
    "united states",
    "united states of america",
    "u.s.",
    "usa",
}
OPTIONS_MARKER = "\n\nOptions:\n"
OPTION_LINE_PATTERN = re.compile(r"^\(([A-Za-z0-9]+)\):\s*(.*)$")


def _sanitize_token(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return sanitized.strip("_").lower()


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _parse_literal(raw: str | None, default: Any) -> Any:
    if raw is None:
        return default
    text = str(raw).strip()
    if not text:
        return default
    try:
        return ast.literal_eval(text)
    except Exception:
        return default


def _normalize_distribution(raw: dict[str, Any]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, value in raw.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric < 0:
            continue
        cleaned[str(key)] = numeric
    total = sum(cleaned.values())
    if total <= 0:
        return {str(key): 0.0 for key in raw}
    return {key: value / total for key, value in cleaned.items()}


def _compact_inline_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _compact_multiline_text(text: str) -> str:
    lines = [_compact_inline_whitespace(line) for line in str(text).splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _split_question_and_options(input_template: str) -> tuple[str, dict[str, str]]:
    if OPTIONS_MARKER not in input_template:
        return _compact_multiline_text(input_template), {}

    question_body, options_block = input_template.split(OPTIONS_MARKER, 1)
    option_text_map: dict[str, str] = {}
    current_label: str | None = None
    current_lines: list[str] = []

    for raw_line in options_block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = OPTION_LINE_PATTERN.match(line)
        if match:
            if current_label is not None:
                option_text_map[current_label] = _compact_inline_whitespace(" ".join(current_lines))
            current_label = match.group(1)
            current_lines = [match.group(2).strip()]
            continue
        if current_label is not None:
            current_lines.append(line)

    if current_label is not None:
        option_text_map[current_label] = _compact_inline_whitespace(" ".join(current_lines))

    return _compact_multiline_text(question_body), option_text_map


def _compact_question_payload(dataset_name: str, question_body: str) -> str:
    compact_body = _compact_multiline_text(question_body)
    if dataset_name == "Choices13k":
        lines = [line for line in compact_body.splitlines() if line]
        machine_lines = [line for line in lines if line.startswith("Machine A:") or line.startswith("Machine B:")]
        if len(machine_lines) >= 2:
            return "\n".join(machine_lines)
    if dataset_name == "OSPsychBig5":
        prefix = "Indicate your level of agreement with the following statement:"
        if compact_body.startswith(prefix):
            statement = compact_body[len(prefix) :].strip()
            return f"Statement: {statement}"
    if dataset_name == "OSPsychRWAS":
        prefix = (
            "Please rate your agreement with the following statement on a scale from very strongly disagree "
            "to very strongly agree."
        )
        if compact_body.startswith(prefix):
            statement = compact_body[len(prefix) :].strip()
            if statement.startswith("Statement:"):
                statement = statement[len("Statement:") :].strip()
            return f"Statement: {statement}"
    if dataset_name == "DICES":
        marker = "Regarding the 'New AI chatbot response',"
        if marker in compact_body:
            conversation, _, _ = compact_body.partition(marker)
            return conversation.strip()
    return compact_body


def _dataset_task_note(dataset_name: str) -> str | None:
    notes = {
        "Choices13k": (
            "Each item is a separate gamble between Machine A and Machine B. Respondents choose the machine they "
            "would play; the original task pays 10% of the realized reward as bonus, with negative rewards floored at $0."
        ),
        "DICES": (
            "Each item shows a conversation and a new AI chatbot response. Judge whether the new response is unsafe "
            "overall due to identity-related bias, including hatred, slurs or stereotypes, subhuman comparisons, "
            "group superiority or inferiority claims, malicious identity-based accusations, or attraction-related insults."
        ),
        "OSPsychBig5": "Each item is a personality statement. Rate agreement using the shared five-point scale.",
        "OSPsychRWAS": "Each item is a statement. Rate agreement using the shared nine-point scale.",
    }
    return notes.get(dataset_name)


def _annotate_scales(question_manifest: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    signature_to_scale_id: dict[tuple[tuple[str, str], ...], str] = {}
    scale_definitions: dict[str, dict[str, str]] = {}
    for item in question_manifest:
        signature = tuple(
            (str(label), str((item.get("option_text_map") or {}).get(label, label)))
            for label in item["option_labels"]
        )
        scale_id = signature_to_scale_id.get(signature)
        if scale_id is None:
            scale_id = f"S{len(signature_to_scale_id) + 1}"
            signature_to_scale_id[signature] = scale_id
            scale_definitions[scale_id] = {label: text for label, text in signature}
        item["scale_id"] = scale_id
    return scale_definitions


def _render_scales_block(scale_definitions: dict[str, dict[str, str]]) -> str:
    lines = []
    for scale_id in sorted(scale_definitions):
        lines.append(
            f"{scale_id}={json.dumps(scale_definitions[scale_id], ensure_ascii=False, separators=(',', ':'))}"
        )
    return "\n".join(lines)


def _render_items_block(question_manifest: list[dict[str, Any]]) -> str:
    unique_scale_ids = {item["scale_id"] for item in question_manifest}
    one_scale = len(unique_scale_ids) == 1
    lines = []
    for item in question_manifest:
        payload: dict[str, Any] = {"id": item["question_id"]}
        if not one_scale:
            payload["scale"] = item["scale_id"]
        payload["prompt"] = item["question_payload"]
        lines.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines)


def _single_scale_note(question_manifest: list[dict[str, Any]]) -> str | None:
    unique_scale_ids = sorted({item["scale_id"] for item in question_manifest})
    if len(unique_scale_ids) == 1:
        return f"All items use scale {unique_scale_ids[0]}."
    return None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _render_group_prompt(template: str, variable_map: dict[str, Any]) -> str:
    rendered = str(template)
    for variable, value in variable_map.items():
        rendered = rendered.replace(f"{{{variable}}}", str(value))
    return rendered.strip()


def _simbench_csv_path(repo_root: Path, split: str) -> Path:
    file_name = "SimBenchGrouped.csv" if split == "SimBenchGrouped" else "SimBenchPop.csv"
    return repo_root / "non-PGG_generalization" / "data" / "simbench" / file_name


def _compact_twin_cards_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "non-PGG_generalization"
        / "twin_profiles"
        / "output"
        / "twin_extended_profile_cards"
        / "compact"
        / "twin_extended_profile_cards.jsonl"
    )


def _twin_profiles_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "non-PGG_generalization"
        / "twin_profiles"
        / "output"
        / "twin_extended_profiles"
        / "twin_extended_profiles.jsonl"
    )


def _load_simbench_rows(path: Path, split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for source_row_index, raw_row in enumerate(reader):
            gold_distribution = _normalize_distribution(_parse_literal(raw_row.get("human_answer"), {}))
            option_labels = list(gold_distribution.keys())
            input_template = str(raw_row.get("input_template", "")).strip()
            question_body, option_text_map = _split_question_and_options(input_template)
            variable_map = _parse_literal(raw_row.get("group_prompt_variable_map"), {})
            group_prompt_template = str(raw_row.get("group_prompt_template", "")).strip()
            dataset_name = str(raw_row.get("dataset_name", "")).strip()
            rows.append(
                {
                    "simbench_row_id": f"{_sanitize_token(split)}_{source_row_index:06d}",
                    "simbench_split": split,
                    "source_row_index": source_row_index,
                    "dataset_name": dataset_name,
                    "group_prompt_template": group_prompt_template,
                    "group_prompt_variable_map": variable_map,
                    "rendered_group_prompt": _render_group_prompt(group_prompt_template, variable_map),
                    "input_template": input_template,
                    "question_body": question_body,
                    "question_payload": _compact_question_payload(dataset_name, question_body),
                    "option_text_map": option_text_map,
                    "option_labels": option_labels,
                    "gold_distribution": gold_distribution,
                    "group_size": int(float(raw_row.get("group_size", 0) or 0)),
                    "wave": str(raw_row.get("wave", "")).strip(),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No SimBench rows loaded from {path}")
    return frame


def _has_country_context(row: dict[str, Any]) -> bool:
    variable_map = row.get("group_prompt_variable_map") or {}
    if any(key in COUNTRY_KEYS for key in variable_map):
        return True
    prompt = str(row.get("rendered_group_prompt", ""))
    return any(marker in prompt for marker in ["You are from ", "You are from the ", "based in the "])


def _is_us_context(row: dict[str, Any]) -> bool:
    variable_map = row.get("group_prompt_variable_map") or {}
    for key, value in variable_map.items():
        if key in COUNTRY_KEYS and _normalize_text(str(value)) in {_normalize_text(item) for item in US_MARKERS}:
            return True
    prompt = _normalize_text(str(row.get("rendered_group_prompt", "")))
    return any(marker in prompt for marker in (_normalize_text(item) for item in US_MARKERS))


def _apply_us_country_filter(frame: pd.DataFrame) -> pd.DataFrame:
    keep_mask = []
    for row in frame.to_dict(orient="records"):
        keep_mask.append((not _has_country_context(row)) or _is_us_context(row))
    return frame.loc[keep_mask].copy()


def _build_context_groups(row_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in row_records:
        grouped[(row["simbench_split"], row["dataset_name"], row["rendered_group_prompt"])].append(row)

    context_groups: list[dict[str, Any]] = []
    for index, ((simbench_split, dataset_name, rendered_group_prompt), rows) in enumerate(
        sorted(grouped.items(), key=lambda item: (item[0][1], item[0][2]))
    ):
        sorted_rows = sorted(rows, key=lambda row: row["simbench_row_id"])
        question_manifest = [
            {
                "question_id": f"Q{question_index:04d}",
                "simbench_row_id": row["simbench_row_id"],
                "dataset_name": row["dataset_name"],
                "option_labels": list(row["option_labels"]),
                "input_template": row["input_template"],
                "question_body": row["question_body"],
                "question_payload": row["question_payload"],
                "option_text_map": dict(row["option_text_map"]),
            }
            for question_index, row in enumerate(sorted_rows, start=1)
        ]
        scale_definitions = _annotate_scales(question_manifest)
        context_groups.append(
            {
                "context_id": f"{_sanitize_token(simbench_split)}__ctx_{index:03d}",
                "simbench_split": simbench_split,
                "dataset_name": dataset_name,
                "rendered_group_prompt": rendered_group_prompt,
                "dataset_task_note": _dataset_task_note(dataset_name),
                "scale_definitions": scale_definitions,
                "question_manifest": question_manifest,
                "rows": sorted_rows,
            }
        )
    return context_groups


def _seed_from_parts(*parts: str | int) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _sample_twin_pids(
    *,
    twin_pids: list[str],
    num_samples: int,
    seed_value: int,
) -> list[str]:
    rng = np.random.default_rng(seed_value)
    replace = len(twin_pids) < num_samples
    sampled_indices = rng.choice(len(twin_pids), size=num_samples, replace=replace)
    return [str(twin_pids[int(index)]) for index in sampled_indices]


def _batch_entry(
    *,
    custom_id: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if model != "gpt-5-mini":
        body["temperature"] = 0
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def _build_batched_group_prompt(context_group: dict[str, Any]) -> tuple[str, str]:
    question_manifest = context_group["question_manifest"]
    system_prompt = (
        "You are a group of individuals with these shared characteristics:\n"
        f"{context_group['rendered_group_prompt']}"
    )
    task_note = context_group.get("dataset_task_note")
    scales_block = _render_scales_block(context_group["scale_definitions"])
    single_scale_note = _single_scale_note(question_manifest)
    items_block = _render_items_block(question_manifest)

    user_prompt = (
        f"You will answer {len(question_manifest)} numbered multiple-choice questions for the same target group.\n"
        "Estimate what percentage of the group would choose each option for each item.\n"
        "Return JSON only.\n"
        "Output contract:\n"
        "- The top-level object keys must be exactly the item ids listed in ITEMS.\n"
        "- Each value must be a distribution object keyed by the option labels for that item's scale.\n"
        "- Use integer percentages from 0 to 100.\n"
        "- Each item's percentages must sum to exactly 100.\n"
        "- Do not include explanations, markdown, or extra keys.\n"
        f"- Example for one item only: {json.dumps({'Q0001': {'A': 55, 'B': 45}})}\n"
    )
    if task_note:
        user_prompt += f"\nShared task note:\n{task_note}\n"
    user_prompt += f"\nShared scales:\n{scales_block}\n"
    if single_scale_note:
        user_prompt += f"{single_scale_note}\n"
    user_prompt += f"\nITEMS:\n{items_block}\n\nAnswer:"
    return system_prompt, user_prompt


def _build_batched_explained_group_prompt(context_group: dict[str, Any]) -> tuple[str, str]:
    question_manifest = context_group["question_manifest"]
    system_prompt = (
        "You are a group of individuals with these shared characteristics:\n"
        f"{context_group['rendered_group_prompt']}"
    )
    task_note = context_group.get("dataset_task_note")
    scales_block = _render_scales_block(context_group["scale_definitions"])
    single_scale_note = _single_scale_note(question_manifest)
    items_block = _render_items_block(question_manifest)

    user_prompt = (
        f"You will answer {len(question_manifest)} numbered multiple-choice questions for the same target group.\n"
        "Estimate what percentage of the group would choose each option for each item.\n"
        "Return JSON only.\n"
        "Output contract:\n"
        '- Use exactly two top-level keys: "explanation" and "answers".\n'
        '- "explanation" must be one short paragraph covering the main considerations across the batch.\n'
        '- "answers" must map every item id in ITEMS to a distribution object keyed by that item\'s option labels.\n'
        "- Use integer percentages from 0 to 100.\n"
        "- Each item's percentages must sum to exactly 100.\n"
        "- Do not include markdown or extra keys.\n"
        f"- Example for one item only: {json.dumps({'explanation': '...', 'answers': {'Q0001': {'A': 55, 'B': 45}}})}\n"
    )
    if task_note:
        user_prompt += f"\nShared task note:\n{task_note}\n"
    user_prompt += f"\nShared scales:\n{scales_block}\n"
    if single_scale_note:
        user_prompt += f"{single_scale_note}\n"
    user_prompt += f"\nITEMS:\n{items_block}\n\nAnswer:"
    return system_prompt, user_prompt


def _build_batched_twin_prompt(
    context_group: dict[str, Any],
    twin_card: dict[str, Any],
) -> tuple[str, str]:
    question_manifest = context_group["question_manifest"]
    task_note = context_group.get("dataset_task_note")
    scales_block = _render_scales_block(context_group["scale_definitions"])
    single_scale_note = _single_scale_note(question_manifest)
    items_block = _render_items_block(question_manifest)

    headline = str(twin_card.get("headline", "")).strip()
    summary = str(twin_card.get("summary", "")).strip()
    background = str((twin_card.get("background") or {}).get("summary", "")).strip()
    profile_lines = []
    if headline:
        profile_lines.append(f"Headline: {headline}")
    if summary:
        profile_lines.append(f"Summary: {summary}")
    if background:
        profile_lines.append(f"Background: {background}")
    profile_block = "\n".join(profile_lines)

    system_prompt = (
        "You are simulating one individual respondent drawn from a target group.\n"
        "Official target group:\n"
        f"{context_group['rendered_group_prompt']}\n\n"
        "Treat the official target group as the primary context. "
        "Treat the sampled Twin profile as a secondary prior about the individual respondent you are simulating."
    )
    user_prompt = (
        f"You will answer {len(question_manifest)} numbered multiple-choice questions for the same target group.\n"
        "Sampled Twin profile:\n"
        f"{profile_block}\n\n"
        "Estimate the probability that this individual would choose each option for each item if asked once.\n"
        "Return JSON only.\n"
        "Output contract:\n"
        '- Use exactly two top-level keys: "explanation" and "answers".\n'
        '- "explanation" must be one short paragraph covering the main considerations across the batch.\n'
        '- "answers" must map every item id in ITEMS to a distribution object keyed by that item\'s option labels.\n'
        "- Use integer percentages from 0 to 100.\n"
        "- Each item's percentages must sum to exactly 100.\n"
        "- Do not include markdown or extra keys.\n"
        f"- Example for one item only: {json.dumps({'explanation': '...', 'answers': {'Q0001': {'A': 55, 'B': 45}}})}\n"
    )
    if task_note:
        user_prompt += f"\nShared task note:\n{task_note}\n"
    user_prompt += f"\nShared scales:\n{scales_block}\n"
    if single_scale_note:
        user_prompt += f"{single_scale_note}\n"
    user_prompt += f"\nITEMS:\n{items_block}\n\nAnswer:"
    return system_prompt, user_prompt


def _default_run_name(
    *,
    split: str,
    variant: str,
    model: str,
    num_samples_per_context: int,
    dataset_names: list[str],
    only_us_country_context: bool,
) -> str:
    name_parts = [_sanitize_token(split), _sanitize_token(variant)]
    if variant == VARIANT_TWIN_PROFILE:
        name_parts.append(f"n{num_samples_per_context}")
    name_parts.append(MODEL_SLUGS[model])
    if dataset_names:
        name_parts.append("datasets_" + "_".join(_sanitize_token(name) for name in dataset_names))
    if only_us_country_context:
        name_parts.append("us_only")
    return "__".join(name_parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build batched SimBench inputs grouped by shared context."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--forecasting-root", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--split", type=str, choices=ALL_SPLITS, default="SimBenchPop")
    parser.add_argument("--models", type=str, default="gpt-5.1")
    parser.add_argument("--variants", type=str, default=",".join(ALL_VARIANTS))
    parser.add_argument("--num-samples-per-context", type=int, default=64)
    parser.add_argument("--dataset-names", type=str, default="")
    parser.add_argument("--only-us-country-context", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    dataset_names = [item.strip() for item in args.dataset_names.split(",") if item.strip()]

    unknown_models = [model for model in models if model not in MODEL_SLUGS]
    if unknown_models:
        raise ValueError(f"Unsupported model names: {unknown_models}")
    unsupported_variants = [variant for variant in variants if variant not in SUPPORTED_VARIANTS]
    if unsupported_variants:
        raise ValueError(f"Unsupported variant names: {unsupported_variants}")
    if args.num_samples_per_context <= 0:
        raise ValueError("--num-samples-per-context must be positive.")
    if args.run_name.strip() and (len(models) != 1 or len(variants) != 1):
        raise ValueError("--run-name can only be used when exactly one model and one variant are requested.")

    simbench_rows = _load_simbench_rows(_simbench_csv_path(args.repo_root, args.split), args.split)
    if dataset_names:
        simbench_rows = simbench_rows[simbench_rows["dataset_name"].isin(dataset_names)].copy()
    if args.only_us_country_context:
        simbench_rows = _apply_us_country_filter(simbench_rows)
    simbench_rows = simbench_rows.sort_values(["dataset_name", "simbench_row_id"]).reset_index(drop=True)
    if simbench_rows.empty:
        raise ValueError("No SimBench rows selected after filtering.")

    twin_personas, _ = load_twin_personas(
        _twin_profiles_path(args.repo_root),
        _compact_twin_cards_path(args.repo_root),
    )
    twin_cards_by_pid = load_twin_cards(_compact_twin_cards_path(args.repo_root))
    twin_pids = twin_personas["twin_pid"].astype(str).tolist()

    batch_input_dir = args.forecasting_root / "batch_input"
    batch_output_dir = args.forecasting_root / "batch_output"
    metadata_root = args.forecasting_root / "metadata"
    results_root = args.forecasting_root / "results"
    for directory in [batch_input_dir, batch_output_dir, metadata_root, results_root]:
        directory.mkdir(parents=True, exist_ok=True)

    row_records = simbench_rows.to_dict(orient="records")
    context_groups = _build_context_groups(row_records)
    gold_rows = [
        {
            "simbench_row_id": row["simbench_row_id"],
            "simbench_split": row["simbench_split"],
            "dataset_name": row["dataset_name"],
            "option_labels": row["option_labels"],
            "gold_distribution": row["gold_distribution"],
            "group_size": int(row["group_size"]),
        }
        for row in row_records
    ]

    for variant in variants:
        for model in models:
            run_name = args.run_name.strip() or _default_run_name(
                split=args.split,
                variant=variant,
                model=model,
                num_samples_per_context=args.num_samples_per_context,
                dataset_names=dataset_names,
                only_us_country_context=args.only_us_country_context,
            )
            batch_path = batch_input_dir / f"{run_name}.jsonl"
            metadata_dir = metadata_root / run_name
            metadata_dir.mkdir(parents=True, exist_ok=True)
            sample_dir = metadata_dir / "sample_prompts"
            sample_dir.mkdir(parents=True, exist_ok=True)

            request_manifest_path = metadata_dir / "request_manifest.csv"
            token_csv_path = metadata_dir / "request_token_estimates.csv"
            gold_targets_path = metadata_dir / "gold_targets.jsonl"
            selected_rows_path = metadata_dir / "selected_rows.csv"
            selected_contexts_path = metadata_dir / "selected_contexts.csv"
            shared_twin_panel_path = metadata_dir / "shared_twin_panel.json"

            _write_jsonl(gold_targets_path, gold_rows)
            pd.DataFrame(row_records).assign(
                option_labels_json=lambda df: df["option_labels"].map(json.dumps),
                gold_distribution_json=lambda df: df["gold_distribution"].map(json.dumps),
                group_prompt_variable_map_json=lambda df: df["group_prompt_variable_map"].map(json.dumps),
                option_text_map_json=lambda df: df["option_text_map"].map(json.dumps),
            )[
                [
                    "simbench_row_id",
                    "simbench_split",
                    "dataset_name",
                    "source_row_index",
                    "group_size",
                    "wave",
                    "group_prompt_template",
                    "group_prompt_variable_map_json",
                    "rendered_group_prompt",
                    "input_template",
                    "question_body",
                    "question_payload",
                    "option_text_map_json",
                    "option_labels_json",
                    "gold_distribution_json",
                ]
            ].to_csv(selected_rows_path, index=False)
            pd.DataFrame(
                [
                    {
                        "context_id": context_group["context_id"],
                        "simbench_split": context_group["simbench_split"],
                        "dataset_name": context_group["dataset_name"],
                        "rendered_group_prompt": context_group["rendered_group_prompt"],
                        "question_count": len(context_group["question_manifest"]),
                        "dataset_task_note": context_group.get("dataset_task_note") or "",
                        "scale_definitions_json": json.dumps(context_group["scale_definitions"]),
                        "question_manifest_json": json.dumps(context_group["question_manifest"]),
                    }
                    for context_group in context_groups
                ]
            ).to_csv(selected_contexts_path, index=False)

            manifest_fieldnames = [
                "custom_id",
                "run_name",
                "model",
                "variant",
                "simbench_split",
                "context_id",
                "dataset_name",
                "question_count",
                "sample_index",
                "twin_pid",
                "response_schema",
                "prompt_level",
                "question_manifest_json",
            ]
            sample_written = False
            total_input_tokens = 0
            token_count = 0
            min_tokens: int | None = None
            max_tokens: int | None = None
            shared_twin_pids: list[str] | None = None
            shared_twin_seed_value: int | None = None

            if variant == VARIANT_TWIN_PROFILE:
                shared_twin_seed_value = _seed_from_parts(args.seed, args.split, variant, "global_twin_panel")
                shared_twin_pids = _sample_twin_pids(
                    twin_pids=twin_pids,
                    num_samples=args.num_samples_per_context,
                    seed_value=shared_twin_seed_value,
                )
                shared_twin_panel_path.write_text(
                    json.dumps(
                        {
                            "sampling_scope": "global_panel_across_contexts",
                            "seed": int(args.seed),
                            "derived_seed": int(shared_twin_seed_value),
                            "variant": variant,
                            "simbench_split": args.split,
                            "num_samples": int(args.num_samples_per_context),
                            "twin_pids": shared_twin_pids,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            with (
                batch_path.open("w", encoding="utf-8") as batch_handle,
                request_manifest_path.open("w", encoding="utf-8", newline="") as manifest_handle,
                token_csv_path.open("w", encoding="utf-8", newline="") as token_handle,
            ):
                manifest_writer = csv.DictWriter(manifest_handle, fieldnames=manifest_fieldnames)
                manifest_writer.writeheader()
                token_writer = csv.DictWriter(token_handle, fieldnames=["custom_id", "input_token_estimate"])
                token_writer.writeheader()

                for context_group in context_groups:
                    requests_for_context: list[tuple[int, str | None, str, str, str, str]] = []
                    if variant == VARIANT_BASELINE_PAPER:
                        system_prompt, user_prompt = _build_batched_group_prompt(context_group)
                        requests_for_context.append(
                            (0, None, system_prompt, user_prompt, "batched_distribution_only", "group")
                        )
                    elif variant == VARIANT_BASELINE_EXPLAINED:
                        system_prompt, user_prompt = _build_batched_explained_group_prompt(context_group)
                        requests_for_context.append(
                            (0, None, system_prompt, user_prompt, "batched_explanation_plus_answers", "group")
                        )
                    else:
                        if shared_twin_pids is None:
                            raise ValueError("Expected shared_twin_pids for Twin-profile variant.")
                        sampled_twin_pids = shared_twin_pids
                        for sample_index, twin_pid in enumerate(sampled_twin_pids, start=1):
                            twin_card = twin_cards_by_pid[twin_pid]
                            system_prompt, user_prompt = _build_batched_twin_prompt(context_group, twin_card)
                            requests_for_context.append(
                                (
                                    sample_index,
                                    twin_pid,
                                    system_prompt,
                                    user_prompt,
                                    "batched_explanation_plus_answers",
                                    "individual",
                                )
                            )

                    for sample_index, twin_pid, system_prompt, user_prompt, response_schema, prompt_level in requests_for_context:
                        custom_id = (
                            f"{_sanitize_token(args.split)}__{context_group['context_id']}__"
                            f"{_sanitize_token(variant)}__s{sample_index:04d}"
                        )
                        batch_row = _batch_entry(
                            custom_id=custom_id,
                            model=model,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                        )
                        batch_handle.write(json.dumps(batch_row, ensure_ascii=False) + "\n")
                        manifest_writer.writerow(
                            {
                                "custom_id": custom_id,
                                "run_name": run_name,
                                "model": model,
                                "variant": variant,
                                "simbench_split": context_group["simbench_split"],
                                "context_id": context_group["context_id"],
                                "dataset_name": context_group["dataset_name"],
                                "question_count": len(context_group["question_manifest"]),
                                "sample_index": sample_index,
                                "twin_pid": twin_pid or "",
                                "response_schema": response_schema,
                                "prompt_level": prompt_level,
                                "question_manifest_json": json.dumps(context_group["question_manifest"]),
                            }
                        )
                        input_tokens = int(_estimate_input_tokens(batch_row["body"]["messages"]))
                        token_writer.writerow({"custom_id": custom_id, "input_token_estimate": input_tokens})
                        total_input_tokens += input_tokens
                        token_count += 1
                        min_tokens = input_tokens if min_tokens is None else min(min_tokens, input_tokens)
                        max_tokens = input_tokens if max_tokens is None else max(max_tokens, input_tokens)

                        if not sample_written:
                            sample_text = "\n\n".join(
                                [
                                    "=== SYSTEM MESSAGE ===",
                                    system_prompt,
                                    "=== USER MESSAGE ===",
                                    user_prompt,
                                ]
                            )
                            (sample_dir / "sample_prompt.txt").write_text(sample_text, encoding="utf-8")
                            sample_written = True

            token_summary = {
                "request_file": str(batch_path),
                "num_requests": int(token_count),
                "total_input_tokens_estimate": int(total_input_tokens),
                "mean_input_tokens_estimate": (
                    float(total_input_tokens / token_count) if token_count else float("nan")
                ),
                "max_input_tokens_estimate": int(max_tokens) if max_tokens is not None else None,
                "min_input_tokens_estimate": int(min_tokens) if min_tokens is not None else None,
            }
            (metadata_dir / "request_token_estimates.json").write_text(
                json.dumps(token_summary, indent=2), encoding="utf-8"
            )
            manifest = {
                "run_name": run_name,
                "simbench_split": args.split,
                "variant": variant,
                "model": model,
                "num_simbench_rows": int(len(row_records)),
                "num_context_groups": int(len(context_groups)),
                "num_samples_per_context": int(args.num_samples_per_context),
                "num_requests": int(token_count),
                "seed": int(args.seed),
                "only_us_country_context": bool(args.only_us_country_context),
                "dataset_names_filter": dataset_names,
                "batch_input_file": str(batch_path),
                "expected_batch_output_file": str(batch_output_dir / f"{run_name}.jsonl"),
                "metadata_dir": str(metadata_dir),
                "selected_rows_file": str(selected_rows_path),
                "selected_contexts_file": str(selected_contexts_path),
                "gold_targets_file": str(gold_targets_path),
                "twin_profiles_file": str(_twin_profiles_path(args.repo_root)),
                "twin_profile_cards_file": str(_compact_twin_cards_path(args.repo_root)),
                "twin_sampling_scope": (
                    "global_panel_across_contexts" if variant == VARIANT_TWIN_PROFILE else "not_applicable"
                ),
                "shared_twin_panel_file": (
                    str(shared_twin_panel_path) if variant == VARIANT_TWIN_PROFILE else None
                ),
                "shared_twin_panel_seed": (
                    int(shared_twin_seed_value) if shared_twin_seed_value is not None else None
                ),
                "aggregation_note": (
                    "For baseline variants, each request returns answers for all questions in one context group. "
                    "For Twin-profile variants, the same sampled Twin panel is reused across all context groups in the run; "
                    "average the returned distributions across sample_index within each context_id, then map question_id back "
                    "to simbench_row_id using question_manifest_json."
                ),
            }
            (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
