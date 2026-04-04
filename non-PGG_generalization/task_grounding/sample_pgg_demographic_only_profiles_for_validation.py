#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demographics.generate_demo_table import _normalize_gender


DEFAULT_ANALYSIS_CSV = PROJECT_ROOT / "data" / "processed_data" / "df_analysis_val.csv"
DEFAULT_MERGED_CSV = PROJECT_ROOT / "demographics" / "merged_demographcs_prolific.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "pgg_validation_demographic_only_sampling"
DEFAULT_ROW_RESAMPLED_OUTPUT_DIR = (
    SCRIPT_DIR / "output" / "pgg_validation_demographic_only_sampling_row_resampled"
)

SAMPLING_MODE_INDEPENDENT = "independent_field_marginals"
SAMPLING_MODE_ROW_RESAMPLED = "row_resampled_validation_rows"

UNAVAILABLE = "__UNAVAILABLE__"
PROFILE_FIELDS = [
    "age",
    "sex_or_gender",
    "education",
    "ethnicity",
    "country_of_birth",
    "country_of_residence",
    "nationality",
    "employment_status",
]
FIELD_DISPLAY_NAMES = {
    "age": "Age",
    "sex_or_gender": "Sex/gender",
    "education": "Education",
    "ethnicity": "Ethnicity",
    "country_of_birth": "Country of birth",
    "country_of_residence": "Country of residence",
    "nationality": "Nationality",
    "employment_status": "Employment",
}
CSV_FIELDNAMES = [
    "profile_id",
    "headline",
    "summary",
    "age",
    "sex_or_gender",
    "education",
    "ethnicity",
    "country_of_birth",
    "country_of_residence",
    "nationality",
    "employment_status",
    "missing_field_count",
]


def repo_rooted_path(path: Path) -> str:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return resolved.as_posix()
    return f"{root.name}/{relative.as_posix()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create validation-wave demographic-only synthetic profile cards and seat assignments "
            "using only PGG-side demographic distributions."
        )
    )
    parser.add_argument("--analysis-csv", type=Path, default=DEFAULT_ANALYSIS_CSV)
    parser.add_argument("--merged-csv", type=Path, default=DEFAULT_MERGED_CSV)
    parser.add_argument(
        "--sampling-mode",
        choices=[SAMPLING_MODE_INDEPENDENT, SAMPLING_MODE_ROW_RESAMPLED],
        default=SAMPLING_MODE_INDEPENDENT,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def is_truthy(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_numeric(value: Any) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"data_expired", "consent_revoked", "nan", "none"}:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def clean_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"data_expired", "consent_revoked", "nan", "none", "not applicable"}:
        return None
    return text


def canonicalize_age(prolific_age: Any, exit_age: Any) -> str | None:
    for candidate in [prolific_age, exit_age]:
        parsed = parse_numeric(candidate)
        if parsed is not None:
            rounded = int(round(parsed))
            return str(rounded)
    return None


def canonicalize_sex_or_gender(prolific_sex: Any, exit_gender: Any) -> str | None:
    text = clean_text(prolific_sex)
    if text is not None:
        lowered = text.lower()
        if lowered == "male":
            return "Male"
        if lowered == "female":
            return "Female"
        if lowered == "prefer not to say":
            return "Prefer not to say"
    normalized = _normalize_gender(exit_gender)
    mapping = {
        "man": "Male",
        "woman": "Female",
        "non_binary": "Non-binary",
        "unknown": None,
    }
    return mapping.get(normalized)


def canonicalize_education(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    mapping = {
        "high-school": "Completed high school",
        "bachelor": "Has a college degree",
        "master": "Has a postgraduate degree",
        "other": "Other or nonstandard education background",
    }
    return mapping.get(text.lower())


def canonicalize_ethnicity(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    mapping = {
        "Mixed": "Mixed ethnicity",
        "Other": "Other ethnicity",
    }
    return mapping.get(text, text)


def canonicalize_country(value: Any) -> str | None:
    return clean_text(value)


def canonicalize_employment(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    mapping = {
        "Full-Time": "Full-time employed",
        "Part-Time": "Part-time employed",
        "Unemployed (and job seeking)": "Unemployed and job seeking",
        "Not in paid work (e.g. homemaker', 'retired or disabled)": "Not in paid work (e.g. homemaker, retired, or disabled)",
        "Other": "Other employment status",
        "Due to start a new job within the next month": "Starting a new job within the next month",
    }
    return mapping.get(text, text)


def load_valid_games(analysis_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(analysis_csv)
    valid = df[df["valid_number_of_starting_players"].map(is_truthy)].copy()
    valid["actual_player_id_count"] = valid["playerIds"].fillna("").map(
        lambda raw: len([tok.strip() for tok in str(raw).split(",") if tok.strip()])
    )
    valid["config_player_count_mismatch"] = valid["actual_player_id_count"] != valid["CONFIG_playerCount"]
    columns = [
        "gameId",
        "CONFIG_configId",
        "CONFIG_treatmentName",
        "CONFIG_playerCount",
        "actual_player_id_count",
        "config_player_count_mismatch",
    ]
    return valid[columns].reset_index(drop=True)


def expand_games_to_seats(valid_games: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for game in valid_games.itertuples(index=False):
        configured_count = int(getattr(game, "CONFIG_playerCount"))
        for seat_index in range(1, configured_count + 1):
            rows.append(
                {
                    "gameId": str(getattr(game, "gameId")),
                    "CONFIG_configId": int(getattr(game, "CONFIG_configId")),
                    "CONFIG_treatmentName": str(getattr(game, "CONFIG_treatmentName")),
                    "CONFIG_playerCount": configured_count,
                    "actual_player_id_count": int(getattr(game, "actual_player_id_count")),
                    "config_player_count_mismatch": bool(getattr(game, "config_player_count_mismatch")),
                    "seat_index": int(seat_index),
                }
            )
    return pd.DataFrame(rows)


def load_validation_demographic_source(merged_csv: Path, valid_game_ids: set[str]) -> pd.DataFrame:
    usecols = [
        "PGGEXIT_playerId",
        "PGGEXIT_gameId",
        "PGGEXIT_data.age",
        "PGGEXIT_data.gender",
        "PGGEXIT_data.education",
        "PROLIFIC_Age",
        "PROLIFIC_Sex",
        "PROLIFIC_Ethnicity simplified",
        "PROLIFIC_Country of birth",
        "PROLIFIC_Country of residence",
        "PROLIFIC_Nationality",
        "PROLIFIC_Employment status",
    ]
    df = pd.read_csv(merged_csv, usecols=usecols)
    df = df[df["PGGEXIT_gameId"].astype(str).isin(valid_game_ids)].copy()
    if df.empty:
        raise ValueError("No merged PGG demographic rows found for the valid-start validation games.")

    df["age"] = [
        canonicalize_age(prolific_age, exit_age)
        for prolific_age, exit_age in zip(df["PROLIFIC_Age"], df["PGGEXIT_data.age"])
    ]
    df["sex_or_gender"] = [
        canonicalize_sex_or_gender(prolific_sex, exit_gender)
        for prolific_sex, exit_gender in zip(df["PROLIFIC_Sex"], df["PGGEXIT_data.gender"])
    ]
    df["education"] = df["PGGEXIT_data.education"].map(canonicalize_education)
    df["ethnicity"] = df["PROLIFIC_Ethnicity simplified"].map(canonicalize_ethnicity)
    df["country_of_birth"] = df["PROLIFIC_Country of birth"].map(canonicalize_country)
    df["country_of_residence"] = df["PROLIFIC_Country of residence"].map(canonicalize_country)
    df["nationality"] = df["PROLIFIC_Nationality"].map(canonicalize_country)
    df["employment_status"] = df["PROLIFIC_Employment status"].map(canonicalize_employment)

    return df


def allocate_counts(series: pd.Series, total_seats: int, rng: np.random.Generator) -> Dict[str, int]:
    source = series.fillna(UNAVAILABLE).astype(str)
    counts = source.value_counts()
    observed_total = float(counts.sum())
    expected = {value: count / observed_total * float(total_seats) for value, count in counts.items()}
    allocated = {value: int(np.floor(weight)) for value, weight in expected.items()}
    remaining = int(total_seats - sum(allocated.values()))
    if remaining > 0:
        values = list(expected.keys())
        remainders = np.array([expected[value] - allocated[value] for value in values], dtype=float)
        tie_break = rng.random(len(values)) * 1e-9
        order = np.argsort(-(remainders + tie_break))
        for idx in order[:remaining]:
            allocated[values[int(idx)]] += 1
    return allocated


def choose_candidate_index(
    candidate_indices: List[int],
    reuse_counter: Dict[int, int],
    rng: np.random.Generator,
) -> int:
    weights = np.array(
        [1.0 / (1.0 + float(reuse_counter.get(idx, 0))) for idx in candidate_indices],
        dtype=float,
    )
    probabilities = weights / weights.sum()
    return int(candidate_indices[int(rng.choice(len(candidate_indices), p=probabilities))])


def sample_field_values(
    source_df: pd.DataFrame,
    total_seats: int,
    rng: np.random.Generator,
) -> tuple[Dict[str, List[str]], List[Dict[str, Any]]]:
    sampled_by_field: Dict[str, List[str]] = {}
    comparison_rows: List[Dict[str, Any]] = []
    for field in PROFILE_FIELDS:
        allocated = allocate_counts(source_df[field], total_seats, rng)
        sampled_values: List[str] = []
        for value, count in allocated.items():
            sampled_values.extend([value] * int(count))
        if len(sampled_values) != total_seats:
            raise ValueError(f"Field {field} built {len(sampled_values)} sampled values for {total_seats} seats.")
        rng.shuffle(sampled_values)
        sampled_by_field[field] = sampled_values

        source_counts = source_df[field].fillna(UNAVAILABLE).astype(str).value_counts()
        sampled_counts = pd.Series(sampled_values).value_counts()
        categories = sorted(set(source_counts.index) | set(sampled_counts.index), key=lambda x: (x == UNAVAILABLE, x))
        for category in categories:
            source_count = int(source_counts.get(category, 0))
            sampled_count = int(sampled_counts.get(category, 0))
            comparison_rows.append(
                {
                    "field": field,
                    "category": "Unavailable" if category == UNAVAILABLE else category,
                    "source_count": source_count,
                    "source_pct": round(source_count / len(source_df) * 100.0, 4),
                    "sampled_count": sampled_count,
                    "sampled_pct": round(sampled_count / total_seats * 100.0, 4),
                    "sampled_minus_source_pp": round(
                        (sampled_count / total_seats - source_count / len(source_df)) * 100.0,
                        4,
                    ),
                }
            )
    return sampled_by_field, comparison_rows


def sample_row_profiles(
    source_df: pd.DataFrame,
    seats: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[List[Dict[str, str]], List[Dict[str, Any]], pd.DataFrame]:
    source_records = source_df[PROFILE_FIELDS].fillna(UNAVAILABLE).astype(str).to_dict(orient="records")
    all_indices = list(range(len(source_records)))
    reuse_counter: Dict[int, int] = {}
    sampled_records: List[Dict[str, str]] = []
    selection_rows: List[Dict[str, Any]] = []

    for game_id, group in seats.groupby("gameId", sort=False):
        used_in_game: set[int] = set()
        ordered = group.sort_values("seat_index")
        for row in ordered.itertuples(index=False):
            candidates = [idx for idx in all_indices if idx not in used_in_game]
            allow_within_game_reuse = False
            if not candidates:
                candidates = list(all_indices)
                allow_within_game_reuse = True
            chosen_idx = choose_candidate_index(candidates, reuse_counter, rng)
            reuse_before = int(reuse_counter.get(chosen_idx, 0))
            reuse_counter[chosen_idx] = reuse_before + 1
            used_in_game.add(chosen_idx)
            sampled_records.append(dict(source_records[chosen_idx]))
            selection_rows.append(
                {
                    "gameId": str(getattr(row, "gameId")),
                    "seat_index": int(getattr(row, "seat_index")),
                    "source_row_index": chosen_idx,
                    "reuse_count_before": reuse_before,
                    "reuse_count_after": int(reuse_counter[chosen_idx]),
                    "within_game_reuse": bool(allow_within_game_reuse),
                }
            )

    source_row_usage = (
        pd.DataFrame(
            [
                {"source_row_index": idx, "num_assignments": count}
                for idx, count in sorted(reuse_counter.items(), key=lambda item: (-item[1], item[0]))
            ]
        )
        if reuse_counter
        else pd.DataFrame(columns=["source_row_index", "num_assignments"])
    )
    return sampled_records, selection_rows, source_row_usage


def render_summary(fields: Dict[str, str | None]) -> str:
    parts: List[str] = []
    if fields.get("age") is not None:
        parts.append(f"{fields['age']}-year-old")
    if fields.get("sex_or_gender") is not None:
        parts.append(fields["sex_or_gender"].lower())
    if fields.get("ethnicity") is not None:
        parts.append(fields["ethnicity"])
    if fields.get("education") is not None:
        parts.append(fields["education"].lower())
    if fields.get("employment_status") is not None:
        parts.append(fields["employment_status"].lower())
    birth_country = fields.get("country_of_birth")
    nationality = fields.get("nationality")
    residence = fields.get("country_of_residence")
    if nationality is not None and residence is not None:
        parts.append(f"{nationality} citizen living in {residence}")
    elif residence is not None:
        parts.append(f"living in {residence}")
    elif nationality is not None:
        parts.append(f"{nationality} citizen")
    if birth_country is not None and birth_country not in {nationality, residence}:
        parts.append(f"born in {birth_country}")
    if not parts:
        return "No demographic information available."
    return ", ".join(parts) + "."


def render_markdown(profile_id: str, fields: Dict[str, str | None], summary: str) -> str:
    lines = [f"- {summary}"]
    for field in PROFILE_FIELDS:
        value = fields.get(field)
        if value is None:
            continue
        lines.append(f"- {FIELD_DISPLAY_NAMES[field]}: {value}")
    return "\n".join(lines) + "\n"


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if args.sampling_mode == SAMPLING_MODE_ROW_RESAMPLED:
        return DEFAULT_ROW_RESAMPLED_OUTPUT_DIR
    return DEFAULT_OUTPUT_DIR


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    output_dir = resolve_output_dir(args) / f"seed_{int(args.seed)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_games = load_valid_games(args.analysis_csv)
    seats = expand_games_to_seats(valid_games)
    total_seats = int(len(seats))
    source_df = load_validation_demographic_source(
        args.merged_csv,
        valid_game_ids=set(valid_games["gameId"].astype(str)),
    )
    source_row_selection_df = pd.DataFrame()
    if args.sampling_mode == SAMPLING_MODE_INDEPENDENT:
        sampled_by_field, distribution_rows = sample_field_values(source_df, total_seats, rng)
        sampled_records = []
        for seat_idx in range(total_seats):
            sampled_records.append(
                {
                    field: sampled_by_field[field][seat_idx]
                    for field in PROFILE_FIELDS
                }
            )
    else:
        sampled_records, selection_rows, source_row_usage_df = sample_row_profiles(source_df, seats, rng)
        source_row_selection_df = pd.DataFrame(selection_rows)
        distribution_rows = []
        for field in PROFILE_FIELDS:
            source_counts = source_df[field].fillna(UNAVAILABLE).astype(str).value_counts()
            sampled_counts = pd.Series([record[field] for record in sampled_records]).value_counts()
            categories = sorted(set(source_counts.index) | set(sampled_counts.index), key=lambda x: (x == UNAVAILABLE, x))
            for category in categories:
                source_count = int(source_counts.get(category, 0))
                sampled_count = int(sampled_counts.get(category, 0))
                distribution_rows.append(
                    {
                        "field": field,
                        "category": "Unavailable" if category == UNAVAILABLE else category,
                        "source_count": source_count,
                        "source_pct": round(source_count / len(source_df) * 100.0, 4),
                        "sampled_count": sampled_count,
                        "sampled_pct": round(sampled_count / total_seats * 100.0, 4),
                        "sampled_minus_source_pp": round(
                            (sampled_count / total_seats - source_count / len(source_df)) * 100.0,
                            4,
                        ),
                    }
                )

    cards: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    assignment_rows: List[Dict[str, Any]] = []
    game_rows: List[Dict[str, Any]] = []

    cards_jsonl = output_dir / "demographic_profile_cards.jsonl"
    cards_csv = output_dir / "demographic_profile_cards.csv"
    preview_json = output_dir / "preview_demographic_profile_cards.json"
    preview_md = output_dir / "preview_demographic_profile_cards.md"
    seat_csv = output_dir / "seat_assignments.csv"
    seat_jsonl = output_dir / "seat_assignments.jsonl"
    game_jsonl = output_dir / "game_assignments.jsonl"
    source_row_selection_csv = output_dir / "source_row_selection.csv"
    source_row_usage_csv = output_dir / "source_row_usage_summary.csv"

    for index, row in enumerate(seats.itertuples(index=False), start=1):
        profile_id = f"pgg_demo_profile_{index:05d}"
        fields: Dict[str, str | None] = {}
        sampled_record = sampled_records[index - 1]
        for field in PROFILE_FIELDS:
            value = sampled_record[field]
            fields[field] = None if value == UNAVAILABLE else value
        missing_fields = [field for field, value in fields.items() if value is None]
        summary = render_summary(fields)
        headline = ""
        markdown = render_markdown(profile_id, fields, summary)
        card = {
            "profile_card_version": "pgg_validation_demographic_only_v1",
            "profile_type": "pgg_validation_demographic_only",
            "profile_id": profile_id,
            "headline": headline,
            "summary": summary,
            "demographics": fields,
            "missing_fields": missing_fields,
            "source": {
                "sampling_frame": "valid_start_validation_wave_only",
                "sampling_mode": args.sampling_mode,
                "seed": int(args.seed),
            },
            "markdown": markdown,
        }
        cards.append(card)
        csv_rows.append(
            {
                "profile_id": profile_id,
                "headline": headline,
                "summary": summary,
                "age": fields["age"] or "",
                "sex_or_gender": fields["sex_or_gender"] or "",
                "education": fields["education"] or "",
                "ethnicity": fields["ethnicity"] or "",
                "country_of_birth": fields["country_of_birth"] or "",
                "country_of_residence": fields["country_of_residence"] or "",
                "nationality": fields["nationality"] or "",
                "employment_status": fields["employment_status"] or "",
                "missing_field_count": len(missing_fields),
            }
        )
        assignment_row = {
            "gameId": getattr(row, "gameId"),
            "CONFIG_configId": int(getattr(row, "CONFIG_configId")),
            "CONFIG_treatmentName": getattr(row, "CONFIG_treatmentName"),
            "CONFIG_playerCount": int(getattr(row, "CONFIG_playerCount")),
            "actual_player_id_count": int(getattr(row, "actual_player_id_count")),
            "config_player_count_mismatch": bool(getattr(row, "config_player_count_mismatch")),
            "seat_index": int(getattr(row, "seat_index")),
            "assignment_source": (
                "synthetic_independent_validation_marginals"
                if args.sampling_mode == SAMPLING_MODE_INDEPENDENT
                else "synthetic_row_resampled_validation_demographics"
            ),
            "profile_id": profile_id,
            "profile_headline": headline,
            "profile_summary": summary,
            "profile_cards_jsonl": repo_rooted_path(cards_jsonl),
        }
        if args.sampling_mode == SAMPLING_MODE_ROW_RESAMPLED:
            selection_row = source_row_selection_df.iloc[index - 1]
            assignment_row["source_row_index"] = int(selection_row["source_row_index"])
            assignment_row["source_row_reuse_count_before"] = int(selection_row["reuse_count_before"])
            assignment_row["source_row_reuse_count_after"] = int(selection_row["reuse_count_after"])
            assignment_row["within_game_source_row_reuse"] = bool(selection_row["within_game_reuse"])
        for field in PROFILE_FIELDS:
            assignment_row[field] = fields[field]
        assignment_rows.append(assignment_row)

    write_jsonl(cards_jsonl, cards)
    write_csv(cards_csv, csv_rows)
    preview_json.write_text(json.dumps(cards[:5], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preview_md.write_text("\n\n".join(card["markdown"].rstrip() for card in cards[:5]) + "\n", encoding="utf-8")

    assignments_df = pd.DataFrame(assignment_rows)
    assignments_df.to_csv(seat_csv, index=False)
    write_jsonl(seat_jsonl, assignment_rows)

    for game_id, group in assignments_df.groupby("gameId", sort=False):
        ordered = group.sort_values("seat_index")
        game_rows.append(
            {
                "gameId": game_id,
                "CONFIG_configId": int(ordered["CONFIG_configId"].iloc[0]),
                "CONFIG_treatmentName": str(ordered["CONFIG_treatmentName"].iloc[0]),
                "CONFIG_playerCount": int(ordered["CONFIG_playerCount"].iloc[0]),
                "actual_player_id_count": int(ordered["actual_player_id_count"].iloc[0]),
                "config_player_count_mismatch": bool(ordered["config_player_count_mismatch"].iloc[0]),
                "assignments": ordered[
                    ["seat_index", "profile_id", "profile_summary", *PROFILE_FIELDS]
                ].to_dict(orient="records"),
            }
        )
    write_jsonl(game_jsonl, game_rows)

    distribution_df = pd.DataFrame(distribution_rows)
    distribution_csv = output_dir / "field_distribution_comparison.csv"
    distribution_df.to_csv(distribution_csv, index=False)
    stale_shared_notes = output_dir / "shared_prompt_notes.md"
    if stale_shared_notes.exists():
        stale_shared_notes.unlink()
    if args.sampling_mode == SAMPLING_MODE_ROW_RESAMPLED:
        source_row_selection_df.to_csv(source_row_selection_csv, index=False)
        source_row_usage_df.to_csv(source_row_usage_csv, index=False)

    summary = {
        "seed": int(args.seed),
        "analysis_csv": repo_rooted_path(args.analysis_csv),
        "merged_csv": repo_rooted_path(args.merged_csv),
        "sampling_frame": "valid_start_validation_wave_only",
        "sampling_mode": args.sampling_mode,
        "uses_twin_data": False,
        "uses_behavioral_information": False,
        "uses_individual_pgg_profiles": bool(args.sampling_mode == SAMPLING_MODE_ROW_RESAMPLED),
        "links_profiles_to_actual_players": False,
        "num_games": int(valid_games["gameId"].nunique()),
        "configured_player_count_total": int(valid_games["CONFIG_playerCount"].sum()),
        "actual_player_id_count_total": int(valid_games["actual_player_id_count"].sum()),
        "config_player_count_mismatch_games": int(valid_games["config_player_count_mismatch"].sum()),
        "num_source_rows": int(len(source_df)),
        "num_profiles_generated": int(len(cards)),
        "num_seat_assignments": int(len(assignment_rows)),
        "fields": PROFILE_FIELDS,
        "notes": (
            [
                "Distribution estimates use merged demographic rows from valid-start validation-wave games only.",
                "Each field is sampled independently from its own marginal distribution.",
                "Synthetic profiles are not real individuals and are not linked to the actual players who played each game.",
                "No Twin data and no behavioral-task information are used anywhere in this pipeline.",
                "Game pJFEFMc5YWW7XyLuN is valid-start with CONFIG_playerCount=19 but only 18 listed playerIds; this sampler still uses 19 seats.",
            ]
            if args.sampling_mode == SAMPLING_MODE_INDEPENDENT
            else [
                "Source rows come from merged demographic rows in valid-start validation-wave games only.",
                "Whole demographic rows are resampled intact, so cross-field correlations are preserved.",
                "The sampled rows are not linked back to the actual players who played each game.",
                "No Twin data and no behavioral-task information are used anywhere in this pipeline.",
                "Game pJFEFMc5YWW7XyLuN is valid-start with CONFIG_playerCount=19 but only 18 listed playerIds; this sampler still uses 19 seats.",
            ]
        ),
    }
    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "seed_dir": repo_rooted_path(output_dir),
        "demographic_profile_cards_jsonl": repo_rooted_path(cards_jsonl),
        "demographic_profile_cards_csv": repo_rooted_path(cards_csv),
        "preview_demographic_profile_cards_json": repo_rooted_path(preview_json),
        "preview_demographic_profile_cards_md": repo_rooted_path(preview_md),
        "seat_assignments_csv": repo_rooted_path(seat_csv),
        "seat_assignments_jsonl": repo_rooted_path(seat_jsonl),
        "game_assignments_jsonl": repo_rooted_path(game_jsonl),
        "field_distribution_comparison_csv": repo_rooted_path(distribution_csv),
        "summary_json": repo_rooted_path(summary_json),
    }
    if args.sampling_mode == SAMPLING_MODE_ROW_RESAMPLED:
        manifest["source_row_selection_csv"] = repo_rooted_path(source_row_selection_csv)
        manifest["source_row_usage_summary_csv"] = repo_rooted_path(source_row_usage_csv)
    manifest_json = output_dir / "manifest.json"
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
