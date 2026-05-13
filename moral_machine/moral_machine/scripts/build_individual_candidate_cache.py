from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

from build_individual_demographic_batch import (
    DEFAULT_DB,
    fetch_scenarios,
    valid_demographics,
)
from build_ordered_distribution_alignment_batch import chosen_action, left_action
from build_stay_swerve_batch import (
    DEFAULT_FULL_INPUT,
    DEFAULT_SURVEY_INPUT,
    DEMOGRAPHIC_FIELDS,
    iter_csv_rows,
)
from count_scenario_repeats import paired_rows
from count_unique_scenarios import value, visible_signature


SCRIPT_DIR = Path(__file__).resolve().parent
MM_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = (
    MM_ROOT
    / "processed"
    / "individual_candidates"
    / "global_n_gt_10000_demographic_complete_actual_order"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reusable row-level cache for individual Moral Machine prediction: all target "
            "scenario response pairs, plus the subset with complete survey demographics."
        )
    )
    parser.add_argument("--survey-input", type=Path, default=DEFAULT_SURVEY_INPUT)
    parser.add_argument("--full-input", type=Path, default=DEFAULT_FULL_INPUT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-global-count", type=int, default=10_000)
    parser.add_argument(
        "--strictly-greater",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use global_count > threshold instead of global_count >= threshold.",
    )
    parser.add_argument("--age-min", type=int, default=18)
    parser.add_argument("--age-max", type=int, default=75)
    parser.add_argument(
        "--exclude-slider-midpoints",
        action="store_true",
        help=(
            "Drop Review_political/Review_religious values of exactly 0.5. The MM codebook says "
            "0.5 is also the no-answer default, but it is indistinguishable from a true midpoint."
        ),
    )
    parser.add_argument(
        "--require-country",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require nonblank UserCountry3 for the demographic-complete candidate subset.",
    )
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Write manifest/count summaries from an existing cache without rescanning source data.",
    )
    return parser.parse_args()


def connect_cache(path: Path, *, force: bool) -> sqlite3.Connection:
    if path.exists():
        if not force:
            raise FileExistsError(f"Cache already exists: {path}. Pass --force to rebuild.")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    return connection


def init_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE scenarios (
            scenario_hash TEXT PRIMARY KEY,
            global_count INTEGER NOT NULL,
            scenario_type TEXT,
            scenario_type_strict TEXT,
            attribute_level TEXT,
            pedped TEXT
        );

        CREATE TABLE target_pairs (
            response_id TEXT PRIMARY KEY,
            scenario_hash TEXT NOT NULL,
            extended_session_id TEXT,
            user_id TEXT,
            scenario_order TEXT,
            user_country3_raw TEXT,
            left_action TEXT NOT NULL,
            right_action TEXT NOT NULL,
            chosen_action TEXT NOT NULL,
            gold_choice TEXT NOT NULL,
            FOREIGN KEY (scenario_hash) REFERENCES scenarios (scenario_hash)
        );

        CREATE TABLE candidates (
            response_id TEXT PRIMARY KEY,
            scenario_hash TEXT NOT NULL,
            extended_session_id TEXT,
            user_id TEXT,
            scenario_order TEXT,
            user_country3_raw TEXT,
            left_action TEXT NOT NULL,
            right_action TEXT NOT NULL,
            chosen_action TEXT NOT NULL,
            gold_choice TEXT NOT NULL,
            user_country3 TEXT,
            review_age TEXT,
            review_education TEXT,
            review_gender TEXT,
            review_income TEXT,
            review_political TEXT,
            review_religious TEXT,
            FOREIGN KEY (scenario_hash) REFERENCES scenarios (scenario_hash)
        );
        """
    )
    connection.commit()


def insert_scenarios(connection: sqlite3.Connection, scenarios: list[dict[str, Any]]) -> None:
    connection.executemany(
        """
        INSERT INTO scenarios (
            scenario_hash,
            global_count,
            scenario_type,
            scenario_type_strict,
            attribute_level,
            pedped
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                scenario["scenario_hash"],
                scenario["global_count"],
                scenario["ScenarioType"],
                scenario["ScenarioTypeStrict"],
                scenario["AttributeLevel"],
                scenario["PedPed"],
            )
            for scenario in scenarios
        ],
    )
    connection.commit()


def build_target_pairs(
    *,
    connection: sqlite3.Connection,
    full_input: Path,
    scenarios: list[dict[str, Any]],
    progress_every: int,
) -> dict[str, Any]:
    target_hashes = {bytes.fromhex(scenario["scenario_hash"]) for scenario in scenarios}
    insert_rows: list[tuple[str, ...]] = []
    counts_by_scenario: Counter[str] = Counter()
    anomalies: Counter[str] = Counter()
    matched_target_pairs = 0
    inserted_target_pairs = 0
    start = time.time()

    insert_sql = """
        INSERT OR IGNORE INTO target_pairs (
            response_id,
            scenario_hash,
            extended_session_id,
            user_id,
            scenario_order,
            user_country3_raw,
            left_action,
            right_action,
            chosen_action,
            gold_choice
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    def flush() -> None:
        nonlocal inserted_target_pairs
        if not insert_rows:
            return
        connection.executemany(insert_sql, insert_rows)
        inserted_target_pairs += len(insert_rows)
        insert_rows.clear()
        connection.commit()

    for stay_row, swerve_row, index in paired_rows(full_input, progress_every=progress_every):
        scenario_hash_bytes = visible_signature(stay_row, swerve_row, index)
        if scenario_hash_bytes not in target_hashes:
            continue
        matched_target_pairs += 1
        left = left_action(stay_row, swerve_row, index)
        if left is None:
            anomalies["invalid_left_right_order"] += 1
            continue
        choice = chosen_action(stay_row, swerve_row, index)
        if choice is None:
            anomalies["invalid_choice"] += 1
            continue

        scenario_hash = scenario_hash_bytes.hex()
        counts_by_scenario[scenario_hash] += 1
        insert_rows.append(
            (
                value(stay_row, index, "ResponseID"),
                scenario_hash,
                value(stay_row, index, "ExtendedSessionID"),
                value(stay_row, index, "UserID"),
                value(stay_row, index, "ScenarioOrder"),
                value(stay_row, index, "UserCountry3"),
                left,
                "swerve" if left == "stay" else "stay",
                choice,
                "A" if choice == left else "B",
            )
        )
        if len(insert_rows) >= 50_000:
            flush()

        if progress_every and matched_target_pairs % progress_every == 0:
            print(
                json.dumps(
                    {
                        "pass": "target_pair_cache",
                        "matched_target_pairs": matched_target_pairs,
                        "inserted_target_pairs": inserted_target_pairs + len(insert_rows),
                        "elapsed_seconds": round(time.time() - start, 1),
                    }
                ),
                flush=True,
            )

    flush()
    connection.execute("CREATE INDEX idx_target_pairs_scenario ON target_pairs (scenario_hash)")
    connection.commit()
    return {
        "matched_target_pairs": matched_target_pairs,
        "inserted_target_pairs": connection.execute("SELECT COUNT(*) FROM target_pairs").fetchone()[0],
        "counts_by_scenario_hash": dict(sorted(counts_by_scenario.items())),
        "anomalies": dict(sorted(anomalies.items())),
        "full_scan_summary": dict(paired_rows.summary),
    }


def build_candidate_subset(
    *,
    connection: sqlite3.Connection,
    survey_input: Path,
    age_min: int,
    age_max: int,
    require_country: bool,
    exclude_slider_midpoints: bool,
) -> dict[str, Any]:
    insert_rows: list[tuple[str, ...]] = []
    stats = {
        "survey_rows_seen": 0,
        "survey_stay_rows_seen": 0,
        "complete_demographic_stay_rows_seen": 0,
        "complete_demographic_target_rows_seen": 0,
    }
    target_lookup = connection.cursor()
    insert_sql = """
        INSERT OR IGNORE INTO candidates (
            response_id,
            scenario_hash,
            extended_session_id,
            user_id,
            scenario_order,
            user_country3_raw,
            left_action,
            right_action,
            chosen_action,
            gold_choice,
            user_country3,
            review_age,
            review_education,
            review_gender,
            review_income,
            review_political,
            review_religious
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    def flush() -> None:
        if not insert_rows:
            return
        connection.executemany(insert_sql, insert_rows)
        insert_rows.clear()
        connection.commit()

    for row in iter_csv_rows(survey_input):
        stats["survey_rows_seen"] += 1
        if row.get("Intervention") != "0":
            continue
        stats["survey_stay_rows_seen"] += 1
        if not valid_demographics(
            row,
            age_min=age_min,
            age_max=age_max,
            require_country=require_country,
            exclude_slider_midpoints=exclude_slider_midpoints,
        ):
            continue
        stats["complete_demographic_stay_rows_seen"] += 1
        response_id = row.get("ResponseID", "")
        target = target_lookup.execute(
            """
            SELECT
                response_id,
                scenario_hash,
                extended_session_id,
                user_id,
                scenario_order,
                user_country3_raw,
                left_action,
                right_action,
                chosen_action,
                gold_choice
            FROM target_pairs
            WHERE response_id = ?
            """,
            (response_id,),
        ).fetchone()
        if target is None:
            continue
        stats["complete_demographic_target_rows_seen"] += 1
        insert_rows.append(
            (
                *target,
                row.get("UserCountry3", ""),
                row.get("Review_age", ""),
                row.get("Review_education", ""),
                row.get("Review_gender", ""),
                row.get("Review_income", ""),
                row.get("Review_political", ""),
                row.get("Review_religious", ""),
            )
        )
        if len(insert_rows) >= 50_000:
            flush()

    flush()
    connection.execute("CREATE INDEX idx_candidates_scenario ON candidates (scenario_hash)")
    connection.execute("CREATE INDEX idx_candidates_gold_choice ON candidates (gold_choice)")
    connection.execute("CREATE INDEX idx_candidates_left_action ON candidates (left_action)")
    connection.commit()
    stats["inserted_candidates"] = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    return stats


def write_counts_csv(connection: sqlite3.Connection, path: Path) -> list[dict[str, Any]]:
    rows = [
        {
            "scenario_hash": row[0],
            "global_count": row[1],
            "target_pair_count": row[2],
            "candidate_count": row[3],
            "left_stay_count": row[4],
            "left_swerve_count": row[5],
            "gold_a_count": row[6],
            "gold_b_count": row[7],
        }
        for row in connection.execute(
            """
            WITH target_counts AS (
                SELECT
                    scenario_hash,
                    COUNT(*) AS target_pair_count
                FROM target_pairs
                GROUP BY scenario_hash
            ),
            candidate_counts AS (
                SELECT
                    scenario_hash,
                    COUNT(*) AS candidate_count,
                    SUM(CASE WHEN left_action = 'stay' THEN 1 ELSE 0 END) AS left_stay_count,
                    SUM(CASE WHEN left_action = 'swerve' THEN 1 ELSE 0 END) AS left_swerve_count,
                    SUM(CASE WHEN gold_choice = 'A' THEN 1 ELSE 0 END) AS gold_a_count,
                    SUM(CASE WHEN gold_choice = 'B' THEN 1 ELSE 0 END) AS gold_b_count
                FROM candidates
                GROUP BY scenario_hash
            )
            SELECT
                s.scenario_hash,
                s.global_count,
                COALESCE(t.target_pair_count, 0) AS target_pair_count,
                COALESCE(c.candidate_count, 0) AS candidate_count,
                COALESCE(c.left_stay_count, 0) AS left_stay_count,
                COALESCE(c.left_swerve_count, 0) AS left_swerve_count,
                COALESCE(c.gold_a_count, 0) AS gold_a_count,
                COALESCE(c.gold_b_count, 0) AS gold_b_count
            FROM scenarios s
            LEFT JOIN target_counts t ON s.scenario_hash = t.scenario_hash
            LEFT JOIN candidate_counts c ON s.scenario_hash = c.scenario_hash
            ORDER BY s.global_count DESC, s.scenario_hash
            """
        )
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def cache_table_count(connection: sqlite3.Connection, table: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def write_manifest(
    *,
    connection: sqlite3.Connection,
    manifest_path: Path,
    sqlite_path: Path,
    counts_path: Path,
    args: argparse.Namespace,
    counts: list[dict[str, Any]],
    target_pair_stats: dict[str, Any] | None,
    candidate_stats: dict[str, Any] | None,
    finalized_from_existing_cache: bool,
) -> dict[str, Any]:
    candidate_counts = [int(row["candidate_count"]) for row in counts]
    manifest = {
        "sqlite_file": str(sqlite_path),
        "scenario_candidate_counts_file": str(counts_path),
        "source_file": str(args.full_input),
        "survey_source_file": str(args.survey_input),
        "scenario_db": str(args.db),
        "threshold": {
            "field": "global_count",
            "operator": ">" if args.strictly_greater else ">=",
            "value": args.min_global_count,
        },
        "num_scenarios": cache_table_count(connection, "scenarios"),
        "target_pair_stats": target_pair_stats
        or {
            "inserted_target_pairs": cache_table_count(connection, "target_pairs"),
            "source": "existing_cache_table_count",
        },
        "candidate_stats": candidate_stats
        or {
            "inserted_candidates": cache_table_count(connection, "candidates"),
            "source": "existing_cache_table_count",
        },
        "candidate_count_min": min(candidate_counts),
        "candidate_count_max": max(candidate_counts),
        "candidate_count_total": sum(candidate_counts),
        "demographic_filter": {
            "source": "SharedResponsesSurvey.csv",
            "required_fields": [*DEMOGRAPHIC_FIELDS, "UserCountry3"]
            if args.require_country
            else list(DEMOGRAPHIC_FIELDS),
            "age_min": args.age_min,
            "age_max": args.age_max,
            "education_gender_income": "nonblank and not default",
            "political_religious": (
                "valid 0-1 slider values; 0.5 retained because MM uses 0.5 both as midpoint "
                "and as the documented no-answer default"
                if not args.exclude_slider_midpoints
                else "valid 0-1 slider values excluding exactly 0.5, the documented no-answer default"
            ),
            "require_country": args.require_country,
        },
        "tables": {
            "scenarios": "One row per selected exact visible scenario.",
            "target_pairs": (
                "All raw individual response pairs for selected scenarios with valid LeftHand order "
                "and valid Saved-derived choice."
            ),
            "candidates": (
                "Subset of target_pairs whose respondent has complete survey demographics under "
                "the documented filter."
            ),
        },
        "finalized_from_existing_cache": finalized_from_existing_cache,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    sqlite_path = args.output_dir / "individual_candidates.sqlite"
    manifest_path = args.output_dir / "manifest.json"
    counts_path = args.output_dir / "scenario_candidate_counts.csv"
    if args.finalize_only:
        if not sqlite_path.is_file():
            raise FileNotFoundError(f"Cache not found: {sqlite_path}")
        connection = sqlite3.connect(sqlite_path)
        try:
            counts = write_counts_csv(connection, counts_path)
            manifest = write_manifest(
                connection=connection,
                manifest_path=manifest_path,
                sqlite_path=sqlite_path,
                counts_path=counts_path,
                args=args,
                counts=counts,
                target_pair_stats=None,
                candidate_stats=None,
                finalized_from_existing_cache=True,
            )
        finally:
            connection.close()
        print(
            json.dumps(
                {
                    "sqlite_file": str(sqlite_path),
                    "manifest_file": str(manifest_path),
                    "scenario_candidate_counts_file": str(counts_path),
                    "num_scenarios": manifest["num_scenarios"],
                    "target_pairs": manifest["target_pair_stats"]["inserted_target_pairs"],
                    "candidates": manifest["candidate_stats"]["inserted_candidates"],
                    "candidate_count_min": manifest["candidate_count_min"],
                    "candidate_count_max": manifest["candidate_count_max"],
                    "finalized_from_existing_cache": True,
                },
                indent=2,
            )
        )
        return

    scenarios = fetch_scenarios(args.db, args.min_global_count, args.strictly_greater)
    if not scenarios:
        raise ValueError("No scenarios matched the requested threshold.")

    connection = connect_cache(sqlite_path, force=args.force)
    try:
        init_schema(connection)
        insert_scenarios(connection, scenarios)
        print(
            json.dumps(
                {
                    "pass": "target_pair_cache",
                    "target_scenarios": len(scenarios),
                    "full_input": str(args.full_input),
                    "sqlite": str(sqlite_path),
                }
            ),
            flush=True,
        )
        target_pair_stats = build_target_pairs(
            connection=connection,
            full_input=args.full_input,
            scenarios=scenarios,
            progress_every=args.progress_every,
        )
        print(
            json.dumps(
                {
                    "pass": "candidate_demographic_join",
                    "survey_input": str(args.survey_input),
                    "target_pairs": target_pair_stats["inserted_target_pairs"],
                }
            ),
            flush=True,
        )
        candidate_stats = build_candidate_subset(
            connection=connection,
            survey_input=args.survey_input,
            age_min=args.age_min,
            age_max=args.age_max,
            require_country=args.require_country,
            exclude_slider_midpoints=args.exclude_slider_midpoints,
        )
        counts = write_counts_csv(connection, counts_path)
        manifest = write_manifest(
            connection=connection,
            manifest_path=manifest_path,
            sqlite_path=sqlite_path,
            counts_path=counts_path,
            args=args,
            counts=counts,
            target_pair_stats=target_pair_stats,
            candidate_stats=candidate_stats,
            finalized_from_existing_cache=False,
        )
    finally:
        connection.close()

    print(
        json.dumps(
            {
                "sqlite_file": str(sqlite_path),
                "manifest_file": str(manifest_path),
                "scenario_candidate_counts_file": str(counts_path),
                "num_scenarios": manifest["num_scenarios"],
                "target_pairs": target_pair_stats["inserted_target_pairs"],
                "candidates": candidate_stats["inserted_candidates"],
                "candidate_count_min": manifest["candidate_count_min"],
                "candidate_count_max": manifest["candidate_count_max"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
