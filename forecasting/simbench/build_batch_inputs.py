from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from itertools import combinations
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


VARIANT_TWIN_CORRECTED = "twin_sampled_seed_0"
VARIANT_TWIN_UNADJUSTED = "twin_sampled_unadjusted_seed_0"
ALL_VARIANTS = [VARIANT_TWIN_CORRECTED, VARIANT_TWIN_UNADJUSTED]
ALL_MODELS = ["gpt-5.1", "gpt-5-mini"]
ALL_SPLITS = ["SimBenchPop", "SimBenchGrouped"]

def _sanitize_token(value: str) -> str:
    import re

    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return sanitized.strip("_").lower()


def _batch_entry(custom_id: str, model: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


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
            rows.append(
                {
                    "simbench_row_id": f"{_sanitize_token(split)}_{source_row_index:06d}",
                    "simbench_split": split,
                    "source_row_index": source_row_index,
                    "dataset_name": str(raw_row.get("dataset_name", "")).strip(),
                    "group_prompt_template": str(raw_row.get("group_prompt_template", "")).strip(),
                    "group_prompt_variable_map": _parse_literal(raw_row.get("group_prompt_variable_map"), {}),
                    "grouping_keys": _parse_literal(raw_row.get("grouping_keys"), raw_row.get("grouping_keys", "")),
                    "num_grouping_vars": int(float(raw_row.get("num_grouping_vars", 0) or 0)),
                    "input_template": str(raw_row.get("input_template", "")).strip(),
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


def _opinionqa_education_to_twin(value: str) -> str | None:
    mapping = {
        "less than high school": "high school",
        "high school graduate": "high school",
        "some college, no degree": "college/postsecondary",
        "associate's degree": "college/postsecondary",
        "college graduate/some postgrad": "college/postsecondary",
        "postgraduate": "postgraduate",
    }
    return mapping.get(value.strip().lower())


def _ess_education_to_twin(value: str) -> str | None:
    lowered = value.strip().lower()
    if any(token in lowered for token in ["less than lower secondary", "lower secondary", "upper secondary"]):
        return "high school"
    if any(token in lowered for token in ["advanced vocational", "sub-degree", "bachelor"]):
        return "college/postsecondary"
    if "master" in lowered or "higher tertiary" in lowered:
        return "postgraduate"
    return None


def _afro_education_to_twin(value: str) -> str | None:
    lowered = value.strip().lower()
    if "post-secondary" in lowered or "polytechnic" in lowered or "college" in lowered:
        return "college/postsecondary"
    if any(
        token in lowered
        for token in [
            "no formal schooling",
            "primary",
            "secondary",
            "high school",
            "intermediate school",
        ]
    ):
        return "high school"
    return None


def _latino_education_to_twin(value: str) -> str | None:
    lowered = value.strip().lower()
    if "universitary" in lowered:
        return "college/postsecondary"
    if "basic education" in lowered or "secondary education" in lowered:
        return "high school"
    return None


def _canonical_marital(value: str) -> str | None:
    lowered = value.strip().lower()
    if "living with a partner" in lowered:
        return "living with a partner"
    if "married" in lowered:
        return "married"
    if "widow" in lowered:
        return "widowed"
    if "divorc" in lowered:
        return "divorced"
    if "separat" in lowered:
        return "separated"
    if "never" in lowered or "single" in lowered:
        return "never been married"
    return None


def _canonical_religion(value: str) -> str | None:
    lowered = value.strip().lower()
    if "roman catholic" in lowered or lowered == "catholic":
        return "roman catholic"
    if any(
        token in lowered
        for token in ["protestant", "anglican", "calvinist", "pentecostal", "evangelical", "christian"]
    ):
        return "protestant"
    if "muslim" in lowered or "islam" in lowered:
        return "muslim"
    if "orthodox" in lowered:
        return "orthodox"
    if "agnostic" in lowered:
        return "agnostic"
    if "atheist" in lowered:
        return "atheist"
    if any(token in lowered for token in ["nothing in particular", "no religion"]):
        return "nothing in particular"
    if "jewish" in lowered:
        return "jewish"
    if "hindu" in lowered:
        return "hindu"
    if "buddhist" in lowered:
        return "buddhist"
    if "mormon" in lowered:
        return "mormon"
    return None


def _canonical_party(value: str) -> str | None:
    lowered = value.strip().lower()
    mapping = {
        "democrat": "Democrat",
        "republican": "Republican",
        "independent": "Independent",
        "other": "Other",
    }
    return mapping.get(lowered)


def _canonical_political_views(value: str) -> str | None:
    lowered = value.strip().lower()
    if lowered in {"very liberal", "liberal", "moderate", "conservative", "very conservative"}:
        return lowered
    return None


def _ideology_from_numeric(value: str) -> str | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 3:
        return "liberal"
    if numeric >= 7:
        return "conservative"
    return "moderate"


def _canonical_income(value: str) -> str | None:
    allowed = {
        "$100,000 or more",
        "$30,000-$50,000",
        "$50,000-$75,000",
        "$75,000-$100,000",
        "Less than $30,000",
    }
    return value if value in allowed else None


def _canonical_religattend(value: str) -> str | None:
    lowered = value.strip().lower()
    allowed = {
        "a few times a year",
        "more than once a week",
        "never",
        "once a week",
        "once or twice a month",
        "seldom",
    }
    return lowered if lowered in allowed else None


def _canonical_employment_broad(value: str) -> str | None:
    lowered = value.strip().lower()
    if any(token in lowered for token in ["paid work", "full time", "part time", "private company", "self-employed"]):
        return "employed"
    if "retired" in lowered:
        return "retired"
    if any(token in lowered for token in ["housework", "looking after children", "shopping and housework"]):
        return "homemaker"
    if "student" in lowered:
        return "student"
    if any(token in lowered for token in ["never had paid work", "not in paid work", "no (looking)", "no (not looking)"]):
        return "unemployed"
    return None


def _derive_matching_criteria(row: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    if row["simbench_split"] != "SimBenchGrouped":
        return {}, {}

    dataset_name = str(row["dataset_name"])
    variables = row["group_prompt_variable_map"] or {}
    criteria: dict[str, str] = {}
    unsupported: dict[str, str] = {}

    for key, raw_value in variables.items():
        value = str(raw_value).strip()
        if dataset_name == "OpinionQA":
            if key == "AGE":
                criteria["matching_age_bracket"] = value
            elif key == "SEX":
                criteria["matching_sex"] = value.lower()
            elif key == "EDUCATION":
                mapped = _opinionqa_education_to_twin(value)
                if mapped:
                    criteria["matching_education"] = mapped
            elif key == "RACE":
                criteria["matching_race"] = value.lower()
            elif key == "CREGION":
                criteria["matching_region"] = value
            elif key == "MARITAL":
                mapped = _canonical_marital(value)
                if mapped:
                    criteria["matching_marital"] = mapped
            elif key == "RELIG":
                mapped = _canonical_religion(value)
                if mapped:
                    criteria["matching_religion"] = mapped
            elif key == "RELIGATTEND":
                mapped = _canonical_religattend(value)
                if mapped:
                    criteria["matching_religattend"] = mapped
            elif key == "POLPARTY":
                mapped = _canonical_party(value)
                if mapped:
                    criteria["matching_party"] = mapped
            elif key == "POLIDEOLOGY":
                mapped = _canonical_political_views(value)
                if mapped:
                    criteria["matching_political_views"] = mapped
            elif key == "INCOME":
                mapped = _canonical_income(value)
                if mapped:
                    criteria["matching_income"] = mapped
        elif dataset_name == "ESS":
            if key == "age_group":
                criteria["matching_age_bracket"] = value
            elif key == "gndr":
                criteria["matching_sex"] = value.lower()
            elif key == "eisced":
                mapped = _ess_education_to_twin(value)
                if mapped:
                    criteria["matching_education"] = mapped
            elif key == "maritalb":
                mapped = _canonical_marital(value)
                if mapped:
                    criteria["matching_marital"] = mapped
            elif key == "mnactic":
                mapped = _canonical_employment_broad(value)
                if mapped:
                    criteria["matching_employment_broad"] = mapped
            elif key == "lrscale":
                mapped = _ideology_from_numeric(value)
                if mapped:
                    criteria["matching_ideology_broad"] = mapped
        elif dataset_name == "Afrobarometer":
            if key == "age_group":
                criteria["matching_age_bracket"] = value
            elif key == "gender":
                criteria["matching_sex"] = value.lower()
            elif key == "education":
                mapped = _afro_education_to_twin(value)
                if mapped:
                    criteria["matching_education"] = mapped
            elif key == "employment":
                mapped = _canonical_employment_broad(value)
                if mapped:
                    criteria["matching_employment_broad"] = mapped
            elif key == "religion":
                mapped = _canonical_religion(value)
                if mapped:
                    criteria["matching_religion"] = mapped
        elif dataset_name == "LatinoBarometro":
            if key == "age_group":
                criteria["matching_age_bracket"] = value
            elif key == "gender":
                criteria["matching_sex"] = value.lower()
            elif key == "highest_education":
                mapped = _latino_education_to_twin(value)
                if mapped:
                    criteria["matching_education"] = mapped
            elif key == "employment_status":
                mapped = _canonical_employment_broad(value)
                if mapped:
                    criteria["matching_employment_broad"] = mapped
            elif key == "religion":
                mapped = _canonical_religion(value)
                if mapped:
                    criteria["matching_religion"] = mapped
            elif key == "political_group":
                mapped = _ideology_from_numeric(value)
                if mapped:
                    criteria["matching_ideology_broad"] = mapped
        elif dataset_name == "ISSP":
            if key == "marital_status":
                mapped = _canonical_marital(value)
                if mapped:
                    criteria["matching_marital"] = mapped
            elif key == "religion":
                mapped = _canonical_religion(value)
                if mapped:
                    criteria["matching_religion"] = mapped
            elif key == "work_status":
                mapped = _canonical_employment_broad(value)
                if mapped:
                    criteria["matching_employment_broad"] = mapped

        if key not in {
            "AGE",
            "SEX",
            "EDUCATION",
            "RACE",
            "CREGION",
            "MARITAL",
            "RELIG",
            "RELIGATTEND",
            "POLPARTY",
            "POLIDEOLOGY",
            "INCOME",
            "age_group",
            "gndr",
            "eisced",
            "maritalb",
            "mnactic",
            "lrscale",
            "gender",
            "education",
            "employment",
            "religion",
            "highest_education",
            "employment_status",
            "political_group",
            "marital_status",
            "work_status",
        }:
            unsupported[key] = value

    return criteria, unsupported


def _best_candidate_pool(
    criteria: dict[str, str],
    twin_personas: pd.DataFrame,
    all_pids: list[str],
) -> tuple[list[str], str, list[str]]:
    available_fields = sorted(criteria.keys())
    for size in range(len(available_fields), 0, -1):
        matches: list[tuple[int, tuple[str, ...], list[str]]] = []
        for subset in combinations(available_fields, size):
            subset_df = twin_personas
            for field in subset:
                subset_df = subset_df[subset_df[field] == criteria[field]]
                if subset_df.empty:
                    break
            if not subset_df.empty:
                candidates = subset_df["twin_pid"].astype(str).tolist()
                matches.append((len(candidates), tuple(subset), candidates))
        if matches:
            _, subset, candidates = min(matches, key=lambda item: (item[0], item[1]))
            return candidates, "+".join(subset), list(subset)
    return list(all_pids), "full_pool_fallback", []


def _row_seed(base_seed: int, split: str, variant: str, simbench_row_id: str) -> int:
    payload = f"{base_seed}::{split}::{variant}::{simbench_row_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _sample_profiles_for_row(
    *,
    row: dict[str, Any],
    variant: str,
    num_samples: int,
    seed: int,
    twin_personas: pd.DataFrame,
    all_pids: list[str],
) -> list[dict[str, Any]]:
    criteria, unsupported = _derive_matching_criteria(row)
    if variant == VARIANT_TWIN_UNADJUSTED:
        candidate_pool = list(all_pids)
        match_level = "full_pool_unadjusted"
        matched_fields: list[str] = []
    else:
        candidate_pool, match_level, matched_fields = _best_candidate_pool(criteria, twin_personas, all_pids)
    rng = np.random.default_rng(_row_seed(seed, row["simbench_split"], variant, row["simbench_row_id"]))
    replace = len(candidate_pool) < num_samples
    sampled_indices = rng.choice(len(candidate_pool), size=num_samples, replace=replace)
    assignments: list[dict[str, Any]] = []
    for sample_index, pool_index in enumerate(sampled_indices, start=1):
        twin_pid = str(candidate_pool[int(pool_index)])
        assignments.append(
            {
                "sample_index": sample_index,
                "twin_pid": twin_pid,
                "match_level": match_level,
                "matched_fields": list(matched_fields),
                "candidate_pool_size": int(len(candidate_pool)),
                "sampled_with_replacement": bool(replace),
                "matched_criteria": dict(criteria),
                "unsupported_criteria": dict(unsupported),
            }
        )
    return assignments


def _build_prompt(
    *,
    row: dict[str, Any],
    variant: str,
    assignment: dict[str, Any],
    twin_card: dict[str, Any],
) -> tuple[str, str]:
    option_labels = [str(label) for label in row["option_labels"]]
    system_prompt = (
        "You answer one multiple-choice question as a single simulated human respondent. "
        "Use the official target-group description as the primary context. "
        "Use the sampled Twin profile only as a secondary prior about one individual drawn from that group. "
        "Return exactly one option identifier and nothing else."
    )
    sampling_note = (
        "This sampled Twin profile was selected to match overlapping demographics from the target group."
        if variant == VARIANT_TWIN_CORRECTED
        else "This sampled Twin profile was drawn from the Twin pool without demographic correction."
    )
    lines = [
        "Answer as one respondent sampled for this target population or group.",
        "",
        "# OFFICIAL TARGET GROUP",
        str(row["group_prompt_template"]),
        "",
        "# SAMPLED TWIN PRIOR",
        sampling_note,
    ]
    matched_fields = assignment.get("matched_fields") or []
    if matched_fields and variant == VARIANT_TWIN_CORRECTED:
        pretty_fields = ", ".join(str(field).replace("matching_", "") for field in matched_fields)
        lines.append(f"Matched fields: {pretty_fields}.")
    lines.extend(
        [
            f"Headline: {str(twin_card.get('headline', '')).strip()}",
            f"Summary: {str(twin_card.get('summary', '')).strip()}",
        ]
    )
    background_summary = str((twin_card.get("background") or {}).get("summary", "")).strip()
    if background_summary:
        lines.append(f"Background: {background_summary}")
    lines.extend(
        [
            "",
            "# QUESTION",
            str(row["input_template"]),
            "",
            "# RESPONSE RULES",
            f"- Select exactly one option identifier from: {', '.join(option_labels)}",
            "- Return only the identifier, with no explanation.",
        ]
    )
    return system_prompt, "\n".join(lines)


def _default_run_name(
    *,
    split: str,
    variant: str,
    model: str,
    num_samples_per_row: int,
    dataset_names: list[str],
    row_offset: int,
    max_rows: int | None,
) -> str:
    name_parts = [
        _sanitize_token(split),
        _sanitize_token(variant),
        f"n{num_samples_per_row}",
        MODEL_SLUGS[model],
    ]
    if dataset_names:
        name_parts.append("datasets_" + "_".join(_sanitize_token(name) for name in dataset_names))
    if row_offset:
        name_parts.append(f"offset_{row_offset}")
    if max_rows is not None:
        name_parts.append(f"rows_{max_rows}")
    return "__".join(name_parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SimBench individual-level Twin-sampled batch inputs."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--forecasting-root", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--split", type=str, choices=ALL_SPLITS, default="SimBenchGrouped")
    parser.add_argument("--models", type=str, default=ALL_MODELS[0])
    parser.add_argument("--variants", type=str, default=VARIANT_TWIN_UNADJUSTED)
    parser.add_argument("--num-samples-per-row", type=int, default=200)
    parser.add_argument("--dataset-names", type=str, default="")
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
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
    unsupported_variants = [variant for variant in variants if variant not in ALL_VARIANTS]
    if unsupported_variants:
        raise ValueError(f"Unsupported variant names: {unsupported_variants}")
    if args.num_samples_per_row <= 0:
        raise ValueError("--num-samples-per-row must be positive.")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive when provided.")

    simbench_rows = _load_simbench_rows(_simbench_csv_path(args.repo_root, args.split), args.split)
    if dataset_names:
        simbench_rows = simbench_rows[simbench_rows["dataset_name"].isin(dataset_names)].copy()
    simbench_rows = simbench_rows.sort_values(["dataset_name", "simbench_row_id"]).reset_index(drop=True)
    if args.row_offset:
        simbench_rows = simbench_rows.iloc[args.row_offset :].copy()
    if args.max_rows is not None:
        simbench_rows = simbench_rows.iloc[: args.max_rows].copy()
    if simbench_rows.empty:
        raise ValueError("No SimBench rows selected after filtering.")

    twin_personas, _ = load_twin_personas(
        _twin_profiles_path(args.repo_root),
        _compact_twin_cards_path(args.repo_root),
    )
    twin_cards_by_pid = load_twin_cards(_compact_twin_cards_path(args.repo_root))
    all_pids = twin_personas["twin_pid"].astype(str).tolist()

    batch_input_dir = args.forecasting_root / "batch_input"
    batch_output_dir = args.forecasting_root / "batch_output"
    metadata_root = args.forecasting_root / "metadata"
    results_root = args.forecasting_root / "results"
    for directory in [batch_input_dir, batch_output_dir, metadata_root, results_root]:
        directory.mkdir(parents=True, exist_ok=True)

    row_records = simbench_rows.to_dict(orient="records")
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
        if variant == VARIANT_TWIN_CORRECTED and args.split != "SimBenchGrouped":
            raise ValueError("Corrected Twin sampling is only supported for SimBenchGrouped.")

        for model in models:
            run_name = args.run_name.strip() or _default_run_name(
                split=args.split,
                variant=variant,
                model=model,
                num_samples_per_row=args.num_samples_per_row,
                dataset_names=dataset_names,
                row_offset=args.row_offset,
                max_rows=args.max_rows,
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

            _write_jsonl(gold_targets_path, gold_rows)
            pd.DataFrame(row_records).assign(
                option_labels_json=lambda df: df["option_labels"].map(json.dumps),
                gold_distribution_json=lambda df: df["gold_distribution"].map(json.dumps),
                group_prompt_variable_map_json=lambda df: df["group_prompt_variable_map"].map(json.dumps),
            )[
                [
                    "simbench_row_id",
                    "simbench_split",
                    "dataset_name",
                    "source_row_index",
                    "group_size",
                    "wave",
                    "group_prompt_template",
                    "input_template",
                    "option_labels_json",
                    "gold_distribution_json",
                    "group_prompt_variable_map_json",
                    "num_grouping_vars",
                ]
            ].to_csv(selected_rows_path, index=False)

            manifest_fieldnames = [
                "custom_id",
                "run_name",
                "model",
                "variant",
                "simbench_split",
                "simbench_row_id",
                "dataset_name",
                "sample_index",
                "option_labels_json",
                "twin_pid",
                "match_level",
                "matched_fields_json",
                "matched_criteria_json",
                "unsupported_criteria_json",
                "candidate_pool_size",
                "sampled_with_replacement",
            ]
            token_fieldnames = ["custom_id", "input_token_estimate"]
            sample_written = False
            total_input_tokens = 0
            token_count = 0
            min_tokens: int | None = None
            max_tokens: int | None = None

            with (
                batch_path.open("w", encoding="utf-8") as batch_handle,
                request_manifest_path.open("w", encoding="utf-8", newline="") as manifest_handle,
                token_csv_path.open("w", encoding="utf-8", newline="") as token_handle,
            ):
                manifest_writer = csv.DictWriter(manifest_handle, fieldnames=manifest_fieldnames)
                manifest_writer.writeheader()
                token_writer = csv.DictWriter(token_handle, fieldnames=token_fieldnames)
                token_writer.writeheader()

                for row in row_records:
                    assignments = _sample_profiles_for_row(
                        row=row,
                        variant=variant,
                        num_samples=args.num_samples_per_row,
                        seed=args.seed,
                        twin_personas=twin_personas,
                        all_pids=all_pids,
                    )
                    for assignment in assignments:
                        twin_pid = str(assignment["twin_pid"])
                        twin_card = twin_cards_by_pid[twin_pid]
                        system_prompt, user_prompt = _build_prompt(
                            row=row,
                            variant=variant,
                            assignment=assignment,
                            twin_card=twin_card,
                        )
                        custom_id = (
                            f"{_sanitize_token(args.split)}__{row['simbench_row_id']}__"
                            f"s{int(assignment['sample_index']):04d}"
                        )
                        batch_row = _batch_entry(custom_id, model, system_prompt, user_prompt)
                        batch_handle.write(json.dumps(batch_row, ensure_ascii=False) + "\n")

                        manifest_writer.writerow(
                            {
                                "custom_id": custom_id,
                                "run_name": run_name,
                                "model": model,
                                "variant": variant,
                                "simbench_split": row["simbench_split"],
                                "simbench_row_id": row["simbench_row_id"],
                                "dataset_name": row["dataset_name"],
                                "sample_index": int(assignment["sample_index"]),
                                "option_labels_json": json.dumps(row["option_labels"]),
                                "twin_pid": twin_pid,
                                "match_level": assignment["match_level"],
                                "matched_fields_json": json.dumps(assignment["matched_fields"]),
                                "matched_criteria_json": json.dumps(assignment["matched_criteria"]),
                                "unsupported_criteria_json": json.dumps(assignment["unsupported_criteria"]),
                                "candidate_pool_size": int(assignment["candidate_pool_size"]),
                                "sampled_with_replacement": int(bool(assignment["sampled_with_replacement"])),
                            }
                        )

                        input_tokens = int(_estimate_input_tokens(batch_row["body"]["messages"]))
                        token_writer.writerow(
                            {
                                "custom_id": custom_id,
                                "input_token_estimate": input_tokens,
                            }
                        )
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
                "num_samples_per_row": int(args.num_samples_per_row),
                "num_requests": int(token_count),
                "seed": int(args.seed),
                "row_offset": int(args.row_offset),
                "max_rows": int(args.max_rows) if args.max_rows is not None else None,
                "dataset_names_filter": dataset_names,
                "batch_input_file": str(batch_path),
                "expected_batch_output_file": str(batch_output_dir / f"{run_name}.jsonl"),
                "metadata_dir": str(metadata_dir),
                "selected_rows_file": str(selected_rows_path),
                "gold_targets_file": str(gold_targets_path),
                "twin_profiles_file": str(_twin_profiles_path(args.repo_root)),
                "twin_profile_cards_file": str(_compact_twin_cards_path(args.repo_root)),
            }
            (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
