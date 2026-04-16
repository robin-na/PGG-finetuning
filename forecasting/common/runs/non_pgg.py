from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import tiktoken

from forecasting.common.profiles import (
    load_twin_personas,
    render_demographic_profile_block,
    render_twin_profile_block,
    sample_demographic_profiles,
    sample_twin_profiles,
    write_shared_notes_file,
)
from forecasting.datasets import DatasetBundle, build_dataset_bundle
from forecasting.prompts import get_prompt_builder

VARIANT_BASELINE = "baseline"
VARIANT_DEMOGRAPHIC_ONLY = "demographic_only_row_resampled_seed_0"
VARIANT_TWIN_CORRECTED = "twin_sampled_seed_0"
VARIANT_TWIN_UNADJUSTED = "twin_sampled_unadjusted_seed_0"
ALL_VARIANTS = [
    VARIANT_BASELINE,
    VARIANT_DEMOGRAPHIC_ONLY,
    VARIANT_TWIN_CORRECTED,
    VARIANT_TWIN_UNADJUSTED,
]

MODEL_SLUGS = {"gpt-5.1": "gpt_5_1", "gpt-5-mini": "gpt_5_mini"}
ALL_MODELS = ["gpt-5.1", "gpt-5-mini"]

TOKEN_ENCODING = tiktoken.get_encoding("o200k_base")


@dataclass(frozen=True)
class SamplingSpec:
    max_records_per_treatment: int | None
    sampling_seed: int


@dataclass
class ArchiveContext:
    base_root: Path
    archive_root: Path | None = None

    def archive_path_for(self, target_path: Path) -> Path:
        if self.archive_root is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.archive_root = self.base_root / "archive" / "non_pgg" / timestamp
        return self.archive_root / target_path.relative_to(self.base_root)


def _twin_profiles_jsonl(repo_root: Path) -> Path:
    return (
        repo_root
        / "non-PGG_generalization"
        / "twin_profiles"
        / "output"
        / "twin_extended_profiles"
        / "twin_extended_profiles.jsonl"
    )


def _twin_cards_jsonl(repo_root: Path) -> Path:
    return (
        repo_root
        / "non-PGG_generalization"
        / "twin_profiles"
        / "output"
        / "twin_extended_profile_cards"
        / "pgg_prompt_min"
        / "twin_extended_profile_cards.jsonl"
    )


def _sanitize_token(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return sanitized.strip("_").lower()


def _estimate_input_tokens(messages: list[dict[str, str]]) -> int:
    tokens = 3
    for message in messages:
        tokens += 3
        tokens += len(TOKEN_ENCODING.encode(message.get("role", "")))
        tokens += len(TOKEN_ENCODING.encode(message.get("content", "")))
    return int(tokens)


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


def _write_bytes_with_archive(path: Path, content: bytes, archive_context: ArchiveContext) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing == content:
            return
        archive_path = archive_context.archive_path_for(path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(existing)
    path.write_bytes(content)


def _write_text_with_archive(path: Path, text: str, archive_context: ArchiveContext) -> None:
    _write_bytes_with_archive(path, text.encode("utf-8"), archive_context)


def _write_jsonl_with_archive(path: Path, rows: list[dict[str, Any]], archive_context: ArchiveContext) -> None:
    payload = "".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in rows)
    _write_text_with_archive(path, payload, archive_context)


def _write_dataframe_csv_with_archive(path: Path, df: pd.DataFrame, archive_context: ArchiveContext) -> None:
    _write_text_with_archive(path, df.to_csv(index=False), archive_context)


def _maybe_sample_bundle(
    bundle: DatasetBundle,
    *,
    max_records_per_treatment: int | None,
    sampling_seed: int,
) -> DatasetBundle:
    if max_records_per_treatment is None:
        return bundle
    if max_records_per_treatment <= 0:
        raise ValueError("max_records_per_treatment must be positive when provided.")

    sampled_groups: list[pd.DataFrame] = []
    for _, group in bundle.records.groupby("treatment_name", sort=True):
        if len(group) <= max_records_per_treatment:
            sampled = group.copy()
        else:
            sampled = group.sample(n=max_records_per_treatment, random_state=sampling_seed)
        sampled_groups.append(sampled)

    sampled_records = (
        pd.concat(sampled_groups, ignore_index=True)
        .sort_values(["treatment_name", "record_id"])
        .reset_index(drop=True)
    )
    sampled_unit_ids = set(sampled_records["unit_id"].astype(str))
    sampled_units = (
        bundle.units[bundle.units["unit_id"].astype(str).isin(sampled_unit_ids)]
        .copy()
        .sort_values("unit_id")
        .reset_index(drop=True)
    )
    return DatasetBundle(
        dataset_key=bundle.dataset_key,
        display_name=bundle.display_name,
        records=sampled_records,
        units=sampled_units,
        demographic_source=bundle.demographic_source,
        twin_matching_fields=bundle.twin_matching_fields,
    )


def _prepare_profile_assets(
    bundle: DatasetBundle,
    forecasting_root: Path,
    seed: int,
    *,
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    profile_sampling_root = forecasting_root / "profile_sampling"
    twin_notes_path = profile_sampling_root / "twin_prompt_assets" / "shared_prompt_notes.md"
    write_shared_notes_file(bundle.dataset_key, twin_notes_path)
    shared_notes = twin_notes_path.read_text(encoding="utf-8").strip()

    twin_profiles_jsonl = _twin_profiles_jsonl(repo_root)
    twin_cards_jsonl = _twin_cards_jsonl(repo_root)
    twin_personas, twin_cards_by_pid = load_twin_personas(twin_profiles_jsonl, twin_cards_jsonl)

    demographic_cards_by_unit, demographic_assignments_path = sample_demographic_profiles(
        units=bundle.units,
        demographic_source=bundle.demographic_source,
        seed=seed,
        output_dir=profile_sampling_root / VARIANT_DEMOGRAPHIC_ONLY / "seed_0",
        dataset_key=bundle.dataset_key,
    )
    twin_corrected_assignments_by_unit, twin_corrected_assignments_path = sample_twin_profiles(
        units=bundle.units,
        demographic_source=bundle.demographic_source,
        twin_personas=twin_personas,
        match_fields=bundle.twin_matching_fields,
        seed=seed,
        output_dir=profile_sampling_root / VARIANT_TWIN_CORRECTED / "seed_0",
        dataset_key=bundle.dataset_key,
        corrected=True,
        cards_path=twin_cards_jsonl,
        profiles_path=twin_profiles_jsonl,
    )
    twin_unadjusted_assignments_by_unit, twin_unadjusted_assignments_path = sample_twin_profiles(
        units=bundle.units,
        demographic_source=bundle.demographic_source,
        twin_personas=twin_personas,
        match_fields=bundle.twin_matching_fields,
        seed=seed,
        output_dir=profile_sampling_root / VARIANT_TWIN_UNADJUSTED / "seed_0",
        dataset_key=bundle.dataset_key,
        corrected=False,
        cards_path=twin_cards_jsonl,
        profiles_path=twin_profiles_jsonl,
    )

    demographic_blocks = {
        unit_id: render_demographic_profile_block(card)
        for unit_id, card in demographic_cards_by_unit.items()
    }
    twin_corrected_blocks = {
        unit_id: render_twin_profile_block(
            assignment=twin_corrected_assignments_by_unit[unit_id],
            card=twin_cards_by_pid[twin_corrected_assignments_by_unit[unit_id]["twin_pid"]],
            shared_notes=shared_notes,
            corrected=True,
        )
        for unit_id in twin_corrected_assignments_by_unit
    }
    twin_unadjusted_blocks = {
        unit_id: render_twin_profile_block(
            assignment=twin_unadjusted_assignments_by_unit[unit_id],
            card=twin_cards_by_pid[twin_unadjusted_assignments_by_unit[unit_id]["twin_pid"]],
            shared_notes=shared_notes,
            corrected=False,
        )
        for unit_id in twin_unadjusted_assignments_by_unit
    }

    return {
        VARIANT_BASELINE: {
            "profile_block_by_unit": {},
            "assignment_file": None,
            "cards_file": None,
            "shared_notes_file": None,
        },
        VARIANT_DEMOGRAPHIC_ONLY: {
            "profile_block_by_unit": demographic_blocks,
            "assignment_file": str(demographic_assignments_path),
            "cards_file": str(
                profile_sampling_root
                / VARIANT_DEMOGRAPHIC_ONLY
                / "seed_0"
                / "demographic_profile_cards.jsonl"
            ),
            "shared_notes_file": None,
        },
        VARIANT_TWIN_CORRECTED: {
            "profile_block_by_unit": twin_corrected_blocks,
            "assignment_file": str(twin_corrected_assignments_path),
            "cards_file": str(twin_cards_jsonl),
            "shared_notes_file": str(twin_notes_path),
        },
        VARIANT_TWIN_UNADJUSTED: {
            "profile_block_by_unit": twin_unadjusted_blocks,
            "assignment_file": str(twin_unadjusted_assignments_path),
            "cards_file": str(twin_cards_jsonl),
            "shared_notes_file": str(twin_notes_path),
        },
    }


def _build_run(
    *,
    bundle: DatasetBundle,
    forecasting_root: Path,
    variant: str,
    model: str,
    profile_assets: dict[str, dict[str, Any]],
    sampling_spec: SamplingSpec,
    archive_context: ArchiveContext,
) -> None:
    model_slug = MODEL_SLUGS[model]
    run_name = f"{variant}_{model_slug}"

    batch_input_dir = forecasting_root / "batch_input"
    batch_output_dir = forecasting_root / "batch_output"
    metadata_root = forecasting_root / "metadata"
    results_root = forecasting_root / "results"
    for directory in [batch_input_dir, batch_output_dir, metadata_root, results_root]:
        directory.mkdir(parents=True, exist_ok=True)

    metadata_dir = metadata_root / run_name
    metadata_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = metadata_dir / "sample_prompts"
    sample_dir.mkdir(parents=True, exist_ok=True)

    prompt_builder = get_prompt_builder(bundle.dataset_key)
    profile_block_by_unit = profile_assets[variant]["profile_block_by_unit"]

    batch_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    request_manifest_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []

    sample_written = False
    for row in bundle.records.itertuples(index=False):
        record_series = pd.Series(row._asdict())
        profile_block = profile_block_by_unit.get(str(row.unit_id))
        system_prompt, user_prompt = prompt_builder(record_series, profile_block)
        custom_id = f"{bundle.dataset_key}__{_sanitize_token(str(row.record_id))}"
        batch_row = _batch_entry(custom_id, model, system_prompt, user_prompt)
        batch_rows.append(batch_row)
        gold_rows.append(
            {
                "custom_id": custom_id,
                "record_id": str(row.record_id),
                "unit_id": str(row.unit_id),
                "treatment_name": str(row.treatment_name),
                "gold_target": json.loads(str(row.gold_target_json)),
            }
        )
        manifest_row = {
            "custom_id": custom_id,
            "record_id": str(row.record_id),
            "unit_id": str(row.unit_id),
            "treatment_name": str(row.treatment_name),
            "variant": variant,
            "model": model,
        }
        for column in bundle.records.columns:
            if column in {"record_id", "unit_id", "treatment_name", "gold_target_json"}:
                continue
            manifest_row[column] = record_series[column]
        request_manifest_rows.append(manifest_row)
        token_rows.append(
            {
                "custom_id": custom_id,
                "input_token_estimate": _estimate_input_tokens(batch_row["body"]["messages"]),
            }
        )

        if not sample_written:
            sample_text = "\n\n".join(
                [
                    "=== SYSTEM MESSAGE ===",
                    system_prompt,
                    "=== USER MESSAGE ===",
                    user_prompt,
                ]
            )
            _write_text_with_archive(sample_dir / "sample_prompt.txt", sample_text, archive_context)
            sample_written = True

    batch_path = batch_input_dir / f"{run_name}.jsonl"
    _write_jsonl_with_archive(batch_path, batch_rows, archive_context)
    _write_jsonl_with_archive(metadata_dir / "gold_targets.jsonl", gold_rows, archive_context)
    _write_dataframe_csv_with_archive(metadata_dir / "selected_records.csv", bundle.records, archive_context)
    _write_dataframe_csv_with_archive(metadata_dir / "selected_units.csv", bundle.units, archive_context)
    _write_dataframe_csv_with_archive(
        metadata_dir / "request_manifest.csv",
        pd.DataFrame(request_manifest_rows),
        archive_context,
    )
    token_df = pd.DataFrame(token_rows)
    _write_dataframe_csv_with_archive(metadata_dir / "request_token_estimates.csv", token_df, archive_context)
    token_summary = {
        "request_file": str(batch_path),
        "num_requests": int(len(token_df)),
        "total_input_tokens_estimate": int(token_df["input_token_estimate"].sum()),
        "mean_input_tokens_estimate": float(token_df["input_token_estimate"].mean()),
        "median_input_tokens_estimate": float(token_df["input_token_estimate"].median()),
        "max_input_tokens_estimate": int(token_df["input_token_estimate"].max()),
        "min_input_tokens_estimate": int(token_df["input_token_estimate"].min()),
    }
    _write_text_with_archive(
        metadata_dir / "request_token_estimates.json",
        json.dumps(token_summary, indent=2),
        archive_context,
    )

    manifest = {
        "dataset_key": bundle.dataset_key,
        "display_name": bundle.display_name,
        "run_name": run_name,
        "variant_name": variant,
        "model": model,
        "record_count": int(len(bundle.records)),
        "unit_count": int(len(bundle.units)),
        "max_records_per_treatment": sampling_spec.max_records_per_treatment,
        "sampling_seed": sampling_spec.sampling_seed,
        "batch_input_file": str(batch_path),
        "expected_batch_output_file": str(batch_output_dir / f"{run_name}.jsonl"),
        "metadata_dir": str(metadata_dir),
        "profile_assignment_file": profile_assets[variant]["assignment_file"],
        "profile_cards_file": profile_assets[variant]["cards_file"],
        "profile_shared_notes_file": profile_assets[variant]["shared_notes_file"],
    }
    _write_text_with_archive(
        metadata_dir / "manifest.json",
        json.dumps(manifest, indent=2),
        archive_context,
    )


def build_dataset_runs(
    *,
    dataset_key: str,
    forecasting_root: Path,
    repo_root: Path,
    models: list[str],
    variants: list[str],
    seed: int,
    max_records_per_treatment: int | None,
) -> None:
    archive_context = ArchiveContext(base_root=forecasting_root)
    sampling_spec = SamplingSpec(
        max_records_per_treatment=max_records_per_treatment,
        sampling_seed=seed,
    )
    bundle = build_dataset_bundle(dataset_key, repo_root)
    bundle = _maybe_sample_bundle(
        bundle,
        max_records_per_treatment=sampling_spec.max_records_per_treatment,
        sampling_seed=sampling_spec.sampling_seed,
    )
    profile_assets = _prepare_profile_assets(
        bundle,
        forecasting_root,
        seed,
        repo_root=repo_root,
    )
    for variant in variants:
        for model in models:
            _build_run(
                bundle=bundle,
                forecasting_root=forecasting_root,
                variant=variant,
                model=model,
                profile_assets=profile_assets,
                sampling_spec=sampling_spec,
                archive_context=archive_context,
            )
