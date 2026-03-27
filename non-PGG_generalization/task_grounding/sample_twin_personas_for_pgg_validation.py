#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demographics.generate_demo_table import _normalize_gender


DEFAULT_ANALYSIS_CSV = PROJECT_ROOT / "data" / "processed_data" / "df_analysis_val.csv"
DEFAULT_PLAYER_INPUTS_CSV = PROJECT_ROOT / "data" / "raw_data" / "validation_wave" / "player-inputs.csv"
DEFAULT_TWIN_PROFILES_JSONL = (
    SCRIPT_DIR / "output" / "twin_extended_profiles" / "twin_extended_profiles.jsonl"
)
DEFAULT_TWIN_CARDS_JSONL = (
    SCRIPT_DIR
    / "output"
    / "twin_extended_profile_cards"
    / "pgg_prompt_min"
    / "twin_extended_profile_cards.jsonl"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "twin_to_pgg_validation_persona_sampling"

AGE_ORDER = ["18-29", "30-49", "50-64", "65+"]
EDUCATION_ORDER = ["high school", "college/postsecondary", "postgraduate"]
SEX_ORDER = ["male", "female"]


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
            "Sample Twin personas for the validation-wave PGG games using the aggregate "
            "PGG age-by-education-by-sex distribution, not individual-level matching."
        )
    )
    parser.add_argument("--analysis-csv", type=Path, default=DEFAULT_ANALYSIS_CSV)
    parser.add_argument("--player-inputs-csv", type=Path, default=DEFAULT_PLAYER_INPUTS_CSV)
    parser.add_argument("--twin-profiles-jsonl", type=Path, default=DEFAULT_TWIN_PROFILES_JSONL)
    parser.add_argument("--twin-cards-jsonl", type=Path, default=DEFAULT_TWIN_CARDS_JSONL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def is_truthy(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_player_ids(raw: Any) -> List[str]:
    if pd.isna(raw):
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def harmonize_pgg_age(age: Any) -> Optional[str]:
    if pd.isna(age):
        return None
    try:
        age_value = float(age)
    except (TypeError, ValueError):
        return None
    if age_value < 30:
        return "18-29"
    if age_value < 50:
        return "30-49"
    if age_value < 65:
        return "50-64"
    return "65+"


def harmonize_pgg_education(raw: Any) -> Optional[str]:
    if pd.isna(raw):
        return None
    value = str(raw).strip().lower()
    mapping = {
        "high-school": "high school",
        "bachelor": "college/postsecondary",
        "other": "college/postsecondary",
        "master": "postgraduate",
    }
    return mapping.get(value)


def harmonize_pgg_sex(value: Any) -> Optional[str]:
    normalized = _normalize_gender(value)
    mapping = {"man": "male", "woman": "female"}
    return mapping.get(normalized)


def extract_harmonized_feature_map(profile: Dict[str, Any]) -> Dict[str, Any]:
    features = (
        profile.get("background_context", {}).get("harmonized_features", [])
        if isinstance(profile, dict)
        else []
    )
    out: Dict[str, Any] = {}
    for feature in features:
        name = feature.get("name")
        value = feature.get("value", {})
        raw = value.get("raw") if isinstance(value, dict) else value
        if name:
            out[str(name)] = raw
    return out


def load_twin_cards(path: Path) -> Dict[str, Dict[str, str]]:
    cards: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            participant = row.get("participant", {})
            pid = str(participant.get("pid") or "").strip()
            if not pid:
                continue
            cards[pid] = {
                "headline": str(row.get("headline") or ""),
                "summary": str(row.get("summary") or ""),
                "background_summary": str((row.get("background") or {}).get("summary") or ""),
            }
    return cards


def load_twin_personas(
    profiles_path: Path,
    cards_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]], Dict[Tuple[str, str, str], List[str]], Dict[Tuple[str, str], List[str]], Dict[Tuple[str, str], List[str]], Dict[Tuple[str, str], List[str]]]:
    cards = load_twin_cards(cards_path)
    personas: List[Dict[str, Any]] = []
    age_to_pids: Dict[str, List[str]] = defaultdict(list)
    education_to_pids: Dict[str, List[str]] = defaultdict(list)
    sex_to_pids: Dict[str, List[str]] = defaultdict(list)
    joint3_to_pids: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    age_edu_to_pids: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    age_sex_to_pids: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    edu_sex_to_pids: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    with profiles_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            profile = json.loads(line)
            participant = profile.get("participant", {})
            pid = str(participant.get("pid") or "").strip()
            if not pid:
                continue
            features = extract_harmonized_feature_map(profile)
            age_bracket = features.get("age_bracket")
            education = features.get("education_completed_harmonized")
            sex = str(features.get("sex_assigned_at_birth") or "").strip().lower()
            if age_bracket not in AGE_ORDER or education not in EDUCATION_ORDER or sex not in SEX_ORDER:
                continue
            card = cards.get(pid, {})
            row = {
                "twin_pid": pid,
                "age_bracket": age_bracket,
                "education_harmonized": education,
                "sex_assigned_at_birth": sex,
                "profile_path": repo_rooted_path(profiles_path),
                "profile_card_path": repo_rooted_path(cards_path),
                "headline": card.get("headline", ""),
                "summary": card.get("summary", ""),
                "background_summary": card.get("background_summary", ""),
            }
            personas.append(row)
            age_to_pids[age_bracket].append(pid)
            education_to_pids[education].append(pid)
            sex_to_pids[sex].append(pid)
            joint3_to_pids[(age_bracket, education, sex)].append(pid)
            age_edu_to_pids[(age_bracket, education)].append(pid)
            age_sex_to_pids[(age_bracket, sex)].append(pid)
            edu_sex_to_pids[(education, sex)].append(pid)

    if not personas:
        raise ValueError("No Twin personas loaded.")
    return (
        personas,
        dict(age_to_pids),
        dict(education_to_pids),
        dict(sex_to_pids),
        dict(joint3_to_pids),
        dict(age_edu_to_pids),
        dict(age_sex_to_pids),
        dict(edu_sex_to_pids),
    )


def load_valid_games(analysis_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(analysis_csv)
    valid = df[df["valid_number_of_starting_players"].map(is_truthy)].copy()
    valid["roster_player_ids"] = valid["playerIds"].map(parse_player_ids)
    valid["actual_player_id_count"] = valid["roster_player_ids"].map(len)
    valid["config_player_count_mismatch"] = (
        valid["actual_player_id_count"] != valid["CONFIG_playerCount"]
    )
    columns = [
        "gameId",
        "CONFIG_configId",
        "CONFIG_treatmentName",
        "CONFIG_playerCount",
        "actual_player_id_count",
        "config_player_count_mismatch",
        "roster_player_ids",
    ]
    return valid[columns].reset_index(drop=True)


def expand_games_to_config_seats(valid_games: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for game in valid_games.itertuples(index=False):
        roster = list(getattr(game, "roster_player_ids"))
        configured_count = int(getattr(game, "CONFIG_playerCount"))
        for seat_index in range(1, configured_count + 1):
            roster_pid = roster[seat_index - 1] if seat_index <= len(roster) else None
            rows.append(
                {
                    "gameId": str(getattr(game, "gameId")),
                    "CONFIG_configId": int(getattr(game, "CONFIG_configId")),
                    "CONFIG_treatmentName": str(getattr(game, "CONFIG_treatmentName")),
                    "CONFIG_playerCount": configured_count,
                    "actual_player_id_count": int(getattr(game, "actual_player_id_count")),
                    "config_player_count_mismatch": bool(getattr(game, "config_player_count_mismatch")),
                    "seat_index": int(seat_index),
                    "pgg_roster_playerId": roster_pid,
                }
            )
    if not rows:
        raise ValueError("No valid-start PGG seats were found.")
    return pd.DataFrame(rows)


def load_observed_pgg_demographics(path: Path, valid_game_ids: set[str]) -> Dict[str, Any]:
    df = pd.read_csv(
        path,
        usecols=["gameId", "playerId", "data.age", "data.education", "data.gender"],
    )
    df = df[df["gameId"].astype(str).isin(valid_game_ids)].copy()
    df["age_bracket"] = df["data.age"].map(harmonize_pgg_age)
    df["education_harmonized"] = df["data.education"].map(harmonize_pgg_education)
    df["sex_label"] = df["data.gender"].map(harmonize_pgg_sex)
    df["gender_norm"] = df["data.gender"].map(_normalize_gender)

    complete_age_education = df.dropna(subset=["age_bracket", "education_harmonized"]).copy()
    targetable = df.dropna(subset=["age_bracket", "education_harmonized", "sex_label"]).copy()
    if targetable.empty:
        raise ValueError("No complete observed PGG age+education+male/female rows found for valid games.")

    excluded_non_targetable = complete_age_education[complete_age_education["sex_label"].isna()].copy()
    return {
        "all_valid_player_input_rows": int(len(df)),
        "complete_age_education_rows": int(len(complete_age_education)),
        "targetable_complete_rows": int(len(targetable)),
        "missing_age_only_rows": int(
            (df["age_bracket"].isna() & df["education_harmonized"].notna()).sum()
        ),
        "missing_education_only_rows": int(
            (df["age_bracket"].notna() & df["education_harmonized"].isna()).sum()
        ),
        "missing_both_age_education_rows": int(
            (df["age_bracket"].isna() & df["education_harmonized"].isna()).sum()
        ),
        "excluded_non_targetable_complete_rows": int(len(excluded_non_targetable)),
        "excluded_non_targetable_gender_counts": {
            str(key): int(value)
            for key, value in Counter(excluded_non_targetable["gender_norm"]).items()
        },
        "complete_age_education_rows_frame": complete_age_education[
            ["gameId", "playerId", "age_bracket", "education_harmonized", "gender_norm"]
        ].copy(),
        "targetable_rows_frame": targetable[
            ["gameId", "playerId", "age_bracket", "education_harmonized", "sex_label"]
        ].copy(),
    }


def allocate_joint_counts(
    observed_targetable: pd.DataFrame,
    total_seats: int,
    rng: np.random.Generator,
) -> Dict[Tuple[str, str, str], int]:
    joint_counts = Counter(
        zip(
            observed_targetable["age_bracket"],
            observed_targetable["education_harmonized"],
            observed_targetable["sex_label"],
        )
    )
    if not joint_counts:
        raise ValueError("Observed targetable joint distribution is empty.")

    observed_total = float(sum(joint_counts.values()))
    expected: Dict[Tuple[str, str, str], float] = {
        cell: count / observed_total * float(total_seats)
        for cell, count in joint_counts.items()
    }
    allocated: Dict[Tuple[str, str, str], int] = {
        cell: int(np.floor(value))
        for cell, value in expected.items()
    }
    remaining = int(total_seats - sum(allocated.values()))
    if remaining > 0:
        cells = list(expected.keys())
        remainders = np.array([expected[cell] - allocated[cell] for cell in cells], dtype=float)
        tie_break = rng.random(len(cells)) * 1e-9
        order = np.argsort(-(remainders + tie_break))
        for idx in order[:remaining]:
            allocated[cells[int(idx)]] += 1

    for age in AGE_ORDER:
        for education in EDUCATION_ORDER:
            for sex in SEX_ORDER:
                allocated.setdefault((age, education, sex), 0)
    return allocated


def assign_target_cells_to_seats(
    seats: pd.DataFrame,
    target_joint_counts: Dict[Tuple[str, str, str], int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    targets: List[Tuple[str, str, str]] = []
    for age in AGE_ORDER:
        for education in EDUCATION_ORDER:
            for sex in SEX_ORDER:
                targets.extend([(age, education, sex)] * int(target_joint_counts.get((age, education, sex), 0)))
    if len(targets) != len(seats):
        raise ValueError(
            f"Target seat count mismatch: built {len(targets)} targets for {len(seats)} seats."
        )

    rng.shuffle(targets)
    out = seats.copy()
    out["target_age_bracket"] = [age for age, _, _ in targets]
    out["target_education_harmonized"] = [education for _, education, _ in targets]
    out["target_sex"] = [sex for _, _, sex in targets]
    out["target_source"] = "aggregate_weighted_joint_distribution_age_education_sex"
    return out


def choose_pid(
    candidate_pids: Sequence[str],
    reuse_counter: Counter[str],
    rng: np.random.Generator,
) -> str:
    weights = np.array(
        [1.0 / (1.0 + float(reuse_counter.get(pid, 0))) for pid in candidate_pids],
        dtype=float,
    )
    probabilities = weights / weights.sum()
    return str(candidate_pids[int(rng.choice(len(candidate_pids), p=probabilities))])


def total_variation_distance(left_counts: Counter[Any], right_counts: Counter[Any]) -> float:
    categories = sorted(set(left_counts) | set(right_counts))
    left_total = sum(left_counts.values())
    right_total = sum(right_counts.values())
    if left_total == 0 or right_total == 0:
        return 0.0
    distance = 0.0
    for category in categories:
        left_p = left_counts.get(category, 0) / left_total
        right_p = right_counts.get(category, 0) / right_total
        distance += abs(left_p - right_p)
    return 0.5 * distance


def make_distribution_rows(
    dimension: str,
    categories: Iterable[Any],
    observed_counts: Counter[Any],
    target_counts: Counter[Any],
    assigned_counts: Counter[Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    observed_total = sum(observed_counts.values())
    target_total = sum(target_counts.values())
    assigned_total = sum(assigned_counts.values())
    for category in categories:
        observed_count = int(observed_counts.get(category, 0))
        target_count = int(target_counts.get(category, 0))
        assigned_count = int(assigned_counts.get(category, 0))
        rows.append(
            {
                "dimension": dimension,
                "category": category if not isinstance(category, tuple) else " | ".join(category),
                "observed_count": observed_count,
                "observed_pct": round((observed_count / observed_total * 100.0) if observed_total else 0.0, 4),
                "target_count": target_count,
                "target_pct": round((target_count / target_total * 100.0) if target_total else 0.0, 4),
                "assigned_count": assigned_count,
                "assigned_pct": round((assigned_count / assigned_total * 100.0) if assigned_total else 0.0, 4),
                "target_minus_observed_pp": round(
                    (
                        ((target_count / target_total) if target_total else 0.0)
                        - ((observed_count / observed_total) if observed_total else 0.0)
                    )
                    * 100.0,
                    4,
                ),
                "assigned_minus_target_pp": round(
                    (
                        ((assigned_count / assigned_total) if assigned_total else 0.0)
                        - ((target_count / target_total) if target_total else 0.0)
                    )
                    * 100.0,
                    4,
                ),
            }
        )
    return rows


def build_distribution_checks(
    observed_targetable: pd.DataFrame,
    assignments: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    observed_age = Counter(observed_targetable["age_bracket"])
    observed_education = Counter(observed_targetable["education_harmonized"])
    observed_sex = Counter(observed_targetable["sex_label"])
    observed_joint3 = Counter(
        zip(
            observed_targetable["age_bracket"],
            observed_targetable["education_harmonized"],
            observed_targetable["sex_label"],
        )
    )

    target_age = Counter(assignments["target_age_bracket"])
    target_education = Counter(assignments["target_education_harmonized"])
    target_sex = Counter(assignments["target_sex"])
    target_joint3 = Counter(
        zip(
            assignments["target_age_bracket"],
            assignments["target_education_harmonized"],
            assignments["target_sex"],
        )
    )

    assigned_age = Counter(assignments["twin_age_bracket"])
    assigned_education = Counter(assignments["twin_education_harmonized"])
    assigned_sex = Counter(assignments["twin_sex_assigned_at_birth"])
    assigned_joint3 = Counter(
        zip(
            assignments["twin_age_bracket"],
            assignments["twin_education_harmonized"],
            assignments["twin_sex_assigned_at_birth"],
        )
    )

    rows: List[Dict[str, Any]] = []
    rows.extend(make_distribution_rows("age", AGE_ORDER, observed_age, target_age, assigned_age))
    rows.extend(
        make_distribution_rows(
            "education",
            EDUCATION_ORDER,
            observed_education,
            target_education,
            assigned_education,
        )
    )
    rows.extend(make_distribution_rows("sex_male_female", SEX_ORDER, observed_sex, target_sex, assigned_sex))
    joint3_categories = [
        (age, education, sex)
        for age in AGE_ORDER
        for education in EDUCATION_ORDER
        for sex in SEX_ORDER
    ]
    rows.extend(
        make_distribution_rows(
            "age_x_education_x_sex",
            joint3_categories,
            observed_joint3,
            target_joint3,
            assigned_joint3,
        )
    )

    metrics = {
        "age_observed_to_target_tvd": round(total_variation_distance(observed_age, target_age), 8),
        "age_target_to_assigned_tvd": round(total_variation_distance(target_age, assigned_age), 8),
        "education_observed_to_target_tvd": round(
            total_variation_distance(observed_education, target_education), 8
        ),
        "education_target_to_assigned_tvd": round(
            total_variation_distance(target_education, assigned_education), 8
        ),
        "sex_observed_to_target_tvd": round(total_variation_distance(observed_sex, target_sex), 8),
        "sex_target_to_assigned_tvd": round(total_variation_distance(target_sex, assigned_sex), 8),
        "age_x_education_x_sex_observed_to_target_tvd": round(
            total_variation_distance(observed_joint3, target_joint3), 8
        ),
        "age_x_education_x_sex_target_to_assigned_tvd": round(
            total_variation_distance(target_joint3, assigned_joint3), 8
        ),
    }
    return pd.DataFrame(rows), metrics


def assign_personas(
    seats: pd.DataFrame,
    personas: List[Dict[str, Any]],
    age_to_pids: Dict[str, List[str]],
    education_to_pids: Dict[str, List[str]],
    sex_to_pids: Dict[str, List[str]],
    joint3_to_pids: Dict[Tuple[str, str, str], List[str]],
    age_edu_to_pids: Dict[Tuple[str, str], List[str]],
    age_sex_to_pids: Dict[Tuple[str, str], List[str]],
    edu_sex_to_pids: Dict[Tuple[str, str], List[str]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    persona_by_pid = {row["twin_pid"]: row for row in personas}
    all_pids = [row["twin_pid"] for row in personas]
    reuse_counter: Counter[str] = Counter()
    assignment_rows: List[Dict[str, Any]] = []

    for game_id, game_rows in seats.groupby("gameId", sort=False):
        used_in_game: set[str] = set()
        for row in game_rows.sort_values("seat_index").itertuples(index=False):
            target_age = getattr(row, "target_age_bracket")
            target_education = getattr(row, "target_education_harmonized")
            target_sex = getattr(row, "target_sex")

            search_specs: List[Tuple[str, Sequence[str], bool]] = [
                (
                    "exact_joint3_no_within_game",
                    joint3_to_pids.get((target_age, target_education, target_sex), []),
                    False,
                ),
                (
                    "exact_joint3_allow_within_game",
                    joint3_to_pids.get((target_age, target_education, target_sex), []),
                    True,
                ),
                ("age_education_no_within_game", age_edu_to_pids.get((target_age, target_education), []), False),
                ("age_education_allow_within_game", age_edu_to_pids.get((target_age, target_education), []), True),
                ("age_sex_no_within_game", age_sex_to_pids.get((target_age, target_sex), []), False),
                ("age_sex_allow_within_game", age_sex_to_pids.get((target_age, target_sex), []), True),
                ("education_sex_no_within_game", edu_sex_to_pids.get((target_education, target_sex), []), False),
                ("education_sex_allow_within_game", edu_sex_to_pids.get((target_education, target_sex), []), True),
                ("age_only_no_within_game", age_to_pids.get(target_age, []), False),
                ("age_only_allow_within_game", age_to_pids.get(target_age, []), True),
                ("education_only_no_within_game", education_to_pids.get(target_education, []), False),
                ("education_only_allow_within_game", education_to_pids.get(target_education, []), True),
                ("sex_only_no_within_game", sex_to_pids.get(target_sex, []), False),
                ("sex_only_allow_within_game", sex_to_pids.get(target_sex, []), True),
                ("full_pool_no_within_game", all_pids, False),
                ("full_pool_allow_within_game", all_pids, True),
            ]

            chosen_pid: Optional[str] = None
            match_level = ""
            within_game_reuse = False
            for level, base_candidates, allow_within_game_reuse in search_specs:
                if not base_candidates:
                    continue
                candidates = list(base_candidates) if allow_within_game_reuse else [
                    pid for pid in base_candidates if pid not in used_in_game
                ]
                if not candidates:
                    continue
                chosen_pid = choose_pid(candidates, reuse_counter, rng)
                match_level = level
                within_game_reuse = allow_within_game_reuse and chosen_pid in used_in_game
                break

            if chosen_pid is None:
                raise ValueError(
                    f"Could not assign Twin persona for game {game_id} seat {getattr(row, 'seat_index')}."
                )

            persona = persona_by_pid[chosen_pid]
            reuse_before = int(reuse_counter.get(chosen_pid, 0))
            reuse_counter[chosen_pid] += 1
            used_in_game.add(chosen_pid)

            assignment_rows.append(
                {
                    "gameId": getattr(row, "gameId"),
                    "CONFIG_configId": int(getattr(row, "CONFIG_configId")),
                    "CONFIG_treatmentName": getattr(row, "CONFIG_treatmentName"),
                    "CONFIG_playerCount": int(getattr(row, "CONFIG_playerCount")),
                    "actual_player_id_count": int(getattr(row, "actual_player_id_count")),
                    "config_player_count_mismatch": bool(getattr(row, "config_player_count_mismatch")),
                    "seat_index": int(getattr(row, "seat_index")),
                    "pgg_roster_playerId": getattr(row, "pgg_roster_playerId"),
                    "target_age_bracket": target_age,
                    "target_education_harmonized": target_education,
                    "target_sex": target_sex,
                    "target_source": getattr(row, "target_source"),
                    "twin_pid": chosen_pid,
                    "twin_age_bracket": persona["age_bracket"],
                    "twin_education_harmonized": persona["education_harmonized"],
                    "twin_sex_assigned_at_birth": persona["sex_assigned_at_birth"],
                    "match_level": match_level,
                    "within_game_reuse": bool(within_game_reuse),
                    "reuse_count_before": reuse_before,
                    "reuse_count_after": int(reuse_counter[chosen_pid]),
                    "twin_profile_headline": persona["headline"],
                    "twin_profile_summary": persona["summary"],
                    "twin_background_summary": persona["background_summary"],
                    "twin_profiles_jsonl": persona["profile_path"],
                    "twin_cards_jsonl": persona["profile_card_path"],
                }
            )
    return pd.DataFrame(assignment_rows)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_sampling_note(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Twin-to-PGG Validation Persona Sampling",
        "",
        "- Sampling targets the aggregate validation-wave PGG age-by-education-by-male/female distribution rather than matching Twin personas to individual PGG participants.",
        f"- Valid-start games used: {summary['num_games']}. Configured seat total: {summary['configured_player_count_total']}.",
        f"- Observed valid-game PGG rows with complete age+education: {summary['observed_complete_age_education_rows']}.",
        f"- Observed valid-game PGG rows used for age+education+male/female targeting: {summary['observed_targetable_age_education_sex_rows']}.",
        "- Target quotas were allocated from the observed PGG joint age x education x male/female distribution using largest-remainder rounding, then Twin personas were sampled within each target cell with inverse-reuse weighting.",
        "- Within-game Twin reuse was disallowed unless the pool forced a fallback.",
        "",
        "## Data Notes",
        "",
        "- Twin only provides sex assigned at birth with `Male/Female`, so PGG rows normalized to `non_binary` or `unknown` are excluded from the sex-targeted quota estimation.",
        "- One game (`pJFEFMc5YWW7XyLuN`) is marked as a valid-start game with `CONFIG_playerCount = 19` but only `18` listed `playerIds` in `df_analysis_val.csv`.",
        "- Per user instruction, this sampler uses the configured count (`19`) for that game and leaves the extra seat without a roster-linked `pgg_roster_playerId`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir / f"seed_{int(args.seed)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for legacy_name in [
        "target_distribution_final_with_imputation.csv",
        "target_distribution_observed_only.csv",
        "observed_pgg_distribution_complete_rows.csv",
        "target_distribution_allocated.csv",
    ]:
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()
    rng = np.random.default_rng(int(args.seed))

    valid_games = load_valid_games(args.analysis_csv)
    seats = expand_games_to_config_seats(valid_games)
    observed_info = load_observed_pgg_demographics(
        args.player_inputs_csv,
        valid_game_ids=set(valid_games["gameId"].astype(str)),
    )
    observed_complete_age_education = observed_info["complete_age_education_rows_frame"]
    observed_targetable = observed_info["targetable_rows_frame"]
    target_joint_counts = allocate_joint_counts(
        observed_targetable=observed_targetable,
        total_seats=int(seats["CONFIG_playerCount"].groupby(seats["gameId"]).first().sum()),
        rng=rng,
    )
    seats = assign_target_cells_to_seats(seats, target_joint_counts, rng)

    (
        personas,
        age_to_pids,
        education_to_pids,
        sex_to_pids,
        joint3_to_pids,
        age_edu_to_pids,
        age_sex_to_pids,
        edu_sex_to_pids,
    ) = load_twin_personas(
        args.twin_profiles_jsonl,
        args.twin_cards_jsonl,
    )
    assignments = assign_personas(
        seats=seats,
        personas=personas,
        age_to_pids=age_to_pids,
        education_to_pids=education_to_pids,
        sex_to_pids=sex_to_pids,
        joint3_to_pids=joint3_to_pids,
        age_edu_to_pids=age_edu_to_pids,
        age_sex_to_pids=age_sex_to_pids,
        edu_sex_to_pids=edu_sex_to_pids,
        rng=rng,
    )

    seat_csv = output_dir / "seat_assignments.csv"
    seat_jsonl = output_dir / "seat_assignments.jsonl"
    assignments.to_csv(seat_csv, index=False)
    write_jsonl(seat_jsonl, assignments.to_dict(orient="records"))

    game_rows: List[Dict[str, Any]] = []
    for game_id, group in assignments.groupby("gameId", sort=False):
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
                    [
                        "seat_index",
                        "pgg_roster_playerId",
                        "target_age_bracket",
                        "target_education_harmonized",
                        "target_sex",
                        "target_source",
                        "twin_pid",
                        "twin_profile_headline",
                        "twin_profile_summary",
                    ]
                ].to_dict(orient="records"),
            }
        )
    game_jsonl = output_dir / "game_assignments.jsonl"
    write_jsonl(game_jsonl, game_rows)

    observed_targetable_dist = (
        observed_targetable.groupby(["age_bracket", "education_harmonized", "sex_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["age_bracket", "education_harmonized", "sex_label"])
    )
    observed_targetable_dist.to_csv(output_dir / "observed_pgg_distribution_targetable_rows.csv", index=False)

    target_dist = (
        assignments.groupby(["target_age_bracket", "target_education_harmonized", "target_sex"])
        .size()
        .reset_index(name="count")
        .rename(
            columns={
                "target_age_bracket": "age_bracket",
                "target_education_harmonized": "education_harmonized",
                "target_sex": "sex_label",
            }
        )
        .sort_values(["age_bracket", "education_harmonized", "sex_label"])
    )
    target_dist.to_csv(output_dir / "target_distribution_allocated.csv", index=False)

    assigned_dist = (
        assignments.groupby(
            ["twin_age_bracket", "twin_education_harmonized", "twin_sex_assigned_at_birth"]
        )
        .size()
        .reset_index(name="count")
        .rename(
            columns={
                "twin_age_bracket": "age_bracket",
                "twin_education_harmonized": "education_harmonized",
                "twin_sex_assigned_at_birth": "sex_label",
            }
        )
        .sort_values(["age_bracket", "education_harmonized", "sex_label"])
    )
    assigned_dist.to_csv(output_dir / "assigned_twin_distribution.csv", index=False)

    twin_usage = (
        assignments.groupby(
            ["twin_pid", "twin_age_bracket", "twin_education_harmonized", "twin_sex_assigned_at_birth"]
        )
        .size()
        .reset_index(name="num_assignments")
        .sort_values(["num_assignments", "twin_pid"], ascending=[False, True])
    )
    twin_usage.to_csv(output_dir / "twin_usage_summary.csv", index=False)

    distribution_checks, divergence_metrics = build_distribution_checks(
        observed_targetable=observed_targetable,
        assignments=assignments,
    )
    distribution_checks.to_csv(output_dir / "distribution_checks.csv", index=False)

    summary = {
        "seed": int(args.seed),
        "analysis_csv": repo_rooted_path(args.analysis_csv),
        "player_inputs_csv": repo_rooted_path(args.player_inputs_csv),
        "twin_profiles_jsonl": repo_rooted_path(args.twin_profiles_jsonl),
        "twin_cards_jsonl": repo_rooted_path(args.twin_cards_jsonl),
        "matching_unit": "aggregate_population_distribution_not_individuals",
        "quota_method": "largest_remainder_from_observed_joint_distribution",
        "target_dimensions": ["age_bracket", "education_harmonized", "sex_male_female"],
        "num_games": int(assignments["gameId"].nunique()),
        "configured_player_count_total": int(assignments.drop_duplicates("gameId")["CONFIG_playerCount"].sum()),
        "actual_player_id_count_total": int(assignments.drop_duplicates("gameId")["actual_player_id_count"].sum()),
        "config_player_count_mismatch_games": int(
            assignments.drop_duplicates("gameId")["config_player_count_mismatch"].sum()
        ),
        "num_seats_assigned": int(len(assignments)),
        "observed_valid_player_input_rows": int(observed_info["all_valid_player_input_rows"]),
        "observed_complete_age_education_rows": int(observed_info["complete_age_education_rows"]),
        "observed_targetable_age_education_sex_rows": int(observed_info["targetable_complete_rows"]),
        "observed_missing_age_only_rows": int(observed_info["missing_age_only_rows"]),
        "observed_missing_education_only_rows": int(observed_info["missing_education_only_rows"]),
        "observed_missing_both_age_education_rows": int(observed_info["missing_both_age_education_rows"]),
        "observed_excluded_non_targetable_complete_rows": int(
            observed_info["excluded_non_targetable_complete_rows"]
        ),
        "observed_excluded_non_targetable_gender_counts": observed_info[
            "excluded_non_targetable_gender_counts"
        ],
        "num_unique_twin_personas_used": int(assignments["twin_pid"].nunique()),
        "max_reuse_count": int(assignments["twin_pid"].value_counts().max()),
        "within_game_reuse_count": int(assignments["within_game_reuse"].sum()),
        "match_level_counts": {
            str(key): int(value) for key, value in assignments["match_level"].value_counts().items()
        },
        "distribution_divergence": divergence_metrics,
        "notes": [
            "Sampling targets the aggregate validation-wave PGG age x education x male/female distribution, not individual PGG participants.",
            "PGG rows normalized to non_binary or unknown are excluded from the sex-targeted quota estimation because Twin only provides Male/Female sex assigned at birth.",
            "Game pJFEFMc5YWW7XyLuN is valid-start with CONFIG_playerCount=19 but only 18 listed playerIds; this sampler uses 19 seats per user instruction.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    note_path = output_dir / "sampling_notes.md"
    write_sampling_note(note_path, summary)

    manifest = {
        "seed_dir": repo_rooted_path(output_dir),
        "seat_assignments_csv": repo_rooted_path(seat_csv),
        "seat_assignments_jsonl": repo_rooted_path(seat_jsonl),
        "game_assignments_jsonl": repo_rooted_path(game_jsonl),
        "summary_json": repo_rooted_path(summary_path),
        "sampling_notes_md": repo_rooted_path(note_path),
        "observed_pgg_distribution_targetable_rows_csv": repo_rooted_path(
            output_dir / "observed_pgg_distribution_targetable_rows.csv"
        ),
        "target_distribution_allocated_csv": repo_rooted_path(
            output_dir / "target_distribution_allocated.csv"
        ),
        "assigned_twin_distribution_csv": repo_rooted_path(
            output_dir / "assigned_twin_distribution.csv"
        ),
        "twin_usage_summary_csv": repo_rooted_path(output_dir / "twin_usage_summary.csv"),
        "distribution_checks_csv": repo_rooted_path(output_dir / "distribution_checks.csv"),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
