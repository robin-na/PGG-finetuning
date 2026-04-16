from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VARIANT_DEMOGRAPHIC_ONLY = "demographic_only_row_resampled_seed_0"
VARIANT_TWIN_CORRECTED = "twin_sampled_seed_0"
VARIANT_TWIN_UNADJUSTED = "twin_sampled_unadjusted_seed_0"


def _clean_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nan", "none", "na", "n/a", "data_expired", "consent_revoked"}:
        return None
    return text


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _build_candidate_maps(
    personas: pd.DataFrame,
    match_fields: list[str],
) -> dict[tuple[str, ...], dict[tuple[Any, ...], list[str]]]:
    candidate_maps: dict[tuple[str, ...], dict[tuple[Any, ...], list[str]]] = {}
    for size in range(1, len(match_fields) + 1):
        for subset in combinations(match_fields, size):
            subset_df = personas.dropna(subset=list(subset)).copy()
            grouped = subset_df.groupby(list(subset), dropna=False)["twin_pid"].apply(list)
            candidate_maps[subset] = {
                key if isinstance(key, tuple) else (key,): list(value)
                for key, value in grouped.to_dict().items()
            }
    return candidate_maps


def _choose_pid(
    candidate_pids: list[str],
    reuse_counter: Counter[str],
    rng: np.random.Generator,
    weighted_by_inverse_reuse: bool,
) -> str:
    if not weighted_by_inverse_reuse:
        return str(candidate_pids[int(rng.integers(len(candidate_pids)))])
    weights = np.array(
        [1.0 / (1.0 + float(reuse_counter.get(pid, 0))) for pid in candidate_pids],
        dtype=float,
    )
    probs = weights / weights.sum()
    return str(candidate_pids[int(rng.choice(len(candidate_pids), p=probs))])


def sample_demographic_profiles(
    units: pd.DataFrame,
    demographic_source: pd.DataFrame,
    seed: int,
    output_dir: Path,
    dataset_key: str,
) -> tuple[dict[str, dict[str, Any]], Path]:
    rng = np.random.default_rng(seed)
    if demographic_source.empty:
        raise ValueError(f"{dataset_key}: demographic source rows are empty.")
    sampled_positions = rng.choice(demographic_source.index.to_numpy(), size=len(units), replace=True)
    sampled = demographic_source.loc[sampled_positions].reset_index(drop=True)

    assignments: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    unit_to_card: dict[str, dict[str, Any]] = {}

    for idx, unit_row in enumerate(units.itertuples(index=False), start=1):
        source_row = sampled.iloc[idx - 1].to_dict()
        profile_id = f"{dataset_key}_demo_profile_{idx:05d}"
        card = {
            "profile_card_version": f"{dataset_key}_demographic_only_v1",
            "profile_type": f"{dataset_key}_demographic_only",
            "profile_id": profile_id,
            "headline": "",
            "summary": source_row["summary"],
            "markdown": source_row["markdown"],
            "matching_age_bracket": source_row.get("matching_age_bracket"),
            "matching_sex": source_row.get("matching_sex"),
            "matching_education": source_row.get("matching_education"),
            "source": {
                "sampling_mode": "row_resampled_dataset_rows",
                "seed": seed,
                "source_row_id": source_row.get("source_row_id"),
            },
        }
        assignment = {
            "unit_id": str(unit_row.unit_id),
            "profile_id": profile_id,
            "summary": source_row["summary"],
            "source_row_id": source_row.get("source_row_id"),
            "matching_age_bracket": source_row.get("matching_age_bracket"),
            "matching_sex": source_row.get("matching_sex"),
            "matching_education": source_row.get("matching_education"),
        }
        cards.append(card)
        assignments.append(assignment)
        unit_to_card[str(unit_row.unit_id)] = card

    output_dir.mkdir(parents=True, exist_ok=True)
    cards_path = output_dir / "demographic_profile_cards.jsonl"
    assignments_path = output_dir / "unit_assignments.jsonl"
    _write_jsonl(cards_path, cards)
    _write_jsonl(assignments_path, assignments)
    pd.DataFrame(assignments).to_csv(output_dir / "unit_assignments.csv", index=False)
    summary = {
        "dataset_key": dataset_key,
        "variant": VARIANT_DEMOGRAPHIC_ONLY,
        "seed": seed,
        "unit_count": len(units),
        "source_row_count": int(len(demographic_source)),
        "unique_source_rows_used": int(len(set(sampled["source_row_id"].tolist()))),
        "cards_file": str(cards_path),
        "assignments_file": str(assignments_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return unit_to_card, assignments_path


def sample_twin_profiles(
    *,
    units: pd.DataFrame,
    demographic_source: pd.DataFrame,
    twin_personas: pd.DataFrame,
    match_fields: list[str],
    seed: int,
    output_dir: Path,
    dataset_key: str,
    corrected: bool,
    cards_path: Path,
    profiles_path: Path,
) -> tuple[dict[str, dict[str, Any]], Path]:
    rng = np.random.default_rng(seed)
    all_pids = twin_personas["twin_pid"].tolist()
    persona_by_pid = {
        str(row["twin_pid"]): row
        for row in twin_personas.to_dict(orient="records")
    }
    candidate_maps = _build_candidate_maps(twin_personas, match_fields) if corrected else {}
    reuse_counter: Counter[str] = Counter()

    sampled_targets = (
        demographic_source.loc[
            rng.choice(demographic_source.index.to_numpy(), size=len(units), replace=True)
        ].reset_index(drop=True)
        if corrected
        else None
    )

    assignments: list[dict[str, Any]] = []
    unit_to_assignment: dict[str, dict[str, Any]] = {}

    for idx, unit_row in enumerate(units.itertuples(index=False), start=1):
        match_level = "full_pool_unadjusted" if not corrected else "full_pool"
        target_row = sampled_targets.iloc[idx - 1].to_dict() if corrected else None
        chosen_pid: str | None = None

        if corrected and target_row is not None:
            available_fields = [
                field for field in match_fields if _clean_text(target_row.get(field)) is not None
            ]
            for size in range(len(available_fields), 0, -1):
                found = False
                for subset in combinations(available_fields, size):
                    key = tuple(target_row[field] for field in subset)
                    candidates = candidate_maps.get(tuple(subset), {}).get(key, [])
                    if not candidates:
                        continue
                    chosen_pid = _choose_pid(
                        candidates,
                        reuse_counter,
                        rng,
                        weighted_by_inverse_reuse=True,
                    )
                    match_level = "+".join(subset)
                    found = True
                    break
                if found:
                    break

        if chosen_pid is None:
            chosen_pid = _choose_pid(
                all_pids,
                reuse_counter,
                rng,
                weighted_by_inverse_reuse=corrected,
            )

        reuse_before = int(reuse_counter.get(chosen_pid, 0))
        reuse_counter[chosen_pid] += 1
        persona = persona_by_pid[chosen_pid]
        assignment = {
            "unit_id": str(unit_row.unit_id),
            "twin_pid": chosen_pid,
            "headline": persona["headline"],
            "summary": persona["summary"],
            "background_summary": persona["background_summary"],
            "age_bracket": persona["age_bracket"],
            "sex_assigned_at_birth": persona["sex_assigned_at_birth"],
            "education_harmonized": persona["education_harmonized"],
            "match_level": match_level,
            "reuse_count_before": reuse_before,
            "reuse_count_after": int(reuse_counter[chosen_pid]),
        }
        if target_row is not None:
            assignment["sampled_target_source_row_id"] = target_row.get("source_row_id")
            for field in match_fields:
                assignment[f"target_{field}"] = target_row.get(field)
        assignments.append(assignment)
        unit_to_assignment[str(unit_row.unit_id)] = assignment

    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "unit_assignments.jsonl"
    _write_jsonl(assignments_path, assignments)
    pd.DataFrame(assignments).to_csv(output_dir / "unit_assignments.csv", index=False)
    summary = {
        "dataset_key": dataset_key,
        "variant": VARIANT_TWIN_CORRECTED if corrected else VARIANT_TWIN_UNADJUSTED,
        "seed": seed,
        "unit_count": len(units),
        "twin_pool_size": int(len(twin_personas)),
        "unique_twin_profiles_used": int(len(set(a["twin_pid"] for a in assignments))),
        "mean_reuse_count": float(np.mean([a["reuse_count_after"] for a in assignments])),
        "match_level_counts": dict(Counter(a["match_level"] for a in assignments)),
        "assignments_file": str(assignments_path),
        "source_profile_cards_file": str(cards_path),
        "source_profiles_file": str(profiles_path),
    }
    if corrected:
        summary["source_row_count"] = int(len(demographic_source))
        summary["matching_fields"] = list(match_fields)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return unit_to_assignment, assignments_path

