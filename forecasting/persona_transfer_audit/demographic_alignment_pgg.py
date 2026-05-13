"""Analyze demographic alignment of PGG persona-transfer matches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


DEFAULT_METADATA_DIR = Path(
    "forecasting/persona_transfer_audit/metadata/"
    "twin_direct_summary_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2"
)
DEFAULT_DEMOGRAPHICS = Path("demographics/merged_demographcs_prolific.csv")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _clean(value: str | None) -> str:
    value = (value or "").strip()
    if not value:
        return "Unknown"
    if value in {"CONSENT_REVOKED", "DATA_EXPIRED"}:
        return "Unavailable"
    return value


def _normalize_gender(value: str | None) -> str:
    value = _clean(value)
    lowered = value.strip().lower().replace("_", " ").replace("-", " ")
    if lowered in {"male", "m", "man"}:
        return "Man"
    if lowered in {"female", "f", "woman"}:
        return "Woman"
    if "non binary" in lowered or "nonbinary" in lowered:
        return "Non-binary"
    if lowered in {"prefer not to say", "prefer not to answer"}:
        return "Prefer not to say"
    if value == "Unavailable":
        return value
    return "Other/unknown"


def _normalize_education(value: str | None) -> str:
    value = _clean(value)
    lowered = value.strip().lower()
    mapping = {
        "high-school": "High school",
        "high school": "High school",
        "bachelor": "Bachelor",
        "bachelors": "Bachelor",
        "bachelor's": "Bachelor",
        "master": "Master",
        "masters": "Master",
        "master's": "Master",
        "other": "Other",
    }
    return mapping.get(lowered, value if value != "Unknown" else "Unknown")


def _normalize_yes_no(value: str | None) -> str:
    value = _clean(value)
    lowered = value.lower()
    if lowered == "yes":
        return "Yes"
    if lowered == "no":
        return "No"
    return value


CATEGORICAL_FIELDS: dict[str, tuple[str, Callable[[str | None], str]]] = {
    "gender_self_report": ("PGGEXIT_data.gender", _normalize_gender),
    "education_self_report": ("PGGEXIT_data.education", _normalize_education),
    "prolific_sex": ("PROLIFIC_Sex", _normalize_gender),
    "ethnicity": ("PROLIFIC_Ethnicity simplified", _clean),
    "country_of_birth": ("PROLIFIC_Country of birth", _clean),
    "country_of_residence": ("PROLIFIC_Country of residence", _clean),
    "nationality": ("PROLIFIC_Nationality", _clean),
    "language": ("PROLIFIC_Language", _clean),
    "student_status": ("PROLIFIC_Student status", _normalize_yes_no),
    "employment_status": ("PROLIFIC_Employment status", _clean),
}

NUMERIC_FIELDS = {
    "age_self_report": "PGGEXIT_data.age",
    "age_prolific": "PROLIFIC_Age",
    "prolific_total_approvals": "PROLIFIC_Total approvals",
}


def _parse_float(value: str | None) -> float | None:
    value = (value or "").strip()
    if not value or value in {"CONSENT_REVOKED", "DATA_EXPIRED"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _load_demographics(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = _read_csv(path)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        game_id = row.get("PGGEXIT_gameId", "")
        avatar = (row.get("PGG_data.avatar", "") or "").upper()
        if not game_id or not avatar:
            continue
        demo: dict[str, Any] = {
            "game_id": game_id,
            "player": avatar,
            "player_id": row.get("PGGEXIT_playerId", ""),
        }
        for field, (source_col, normalizer) in CATEGORICAL_FIELDS.items():
            demo[field] = normalizer(row.get(source_col))
        for field, source_col in NUMERIC_FIELDS.items():
            value = _parse_float(row.get(source_col))
            demo[field] = "" if value is None else value
        by_key[(game_id, avatar)] = demo
    return by_key


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv(path):
        row = dict(row)
        row["players"] = json.loads(row["players"])
        rows.append(row)
    return rows


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _build_weighted_rows(
    manifest_rows: list[dict[str, Any]],
    match_rows: list[dict[str, str]],
    demographics: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    weighted_rows: list[dict[str, Any]] = []
    missing_keys: set[tuple[str, str]] = set()
    candidate_keys: set[tuple[str, str]] = set()

    def add_row(distribution: str, base: dict[str, Any], player: str, weight: float) -> None:
        key = (base["game_id"], player.upper())
        demo = demographics.get(key)
        if demo is None:
            missing_keys.add(key)
            return
        weighted_rows.append(
            {
                "distribution": distribution,
                "custom_id": base.get("custom_id", ""),
                "persona_pid": base.get("persona_pid", ""),
                "game_id": base["game_id"],
                "treatment_name": base.get("treatment_name", ""),
                "player": player.upper(),
                "weight": weight,
                **demo,
            }
        )

    for row in manifest_rows:
        players = [str(player).upper() for player in row["players"]]
        if not players:
            continue
        uniform_weight = 1.0 / len(players)
        for player in players:
            candidate_keys.add((row["game_id"], player))
            add_row("candidate_uniform", row, player, uniform_weight)

    for game_id, player in sorted(candidate_keys):
        add_row(
            "unique_human_candidate",
            {"custom_id": "", "persona_pid": "", "game_id": game_id, "treatment_name": ""},
            player,
            1.0,
        )

    manifest_by_id = {row["custom_id"]: row for row in manifest_rows}
    for row in match_rows:
        manifest_row = manifest_by_id[row["custom_id"]]
        player = row["player"].upper()
        probability = float(row["probability"])
        add_row("matched_probability", manifest_row, player, probability)
        if _truthy(row["is_top1"]):
            add_row("matched_top1", manifest_row, player, 1.0)

    summary = {
        "candidate_unique_game_players": len(candidate_keys),
        "missing_demographic_keys": len(missing_keys),
        "missing_examples": [f"{game_id}::{player}" for game_id, player in sorted(missing_keys)[:10]],
    }
    return weighted_rows, summary


def _weighted_counts(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    counts: dict[str, float] = defaultdict(float)
    for row in rows:
        counts[str(row[field])] += float(row["weight"])
    return dict(counts)


def _entropy(shares: dict[str, float]) -> float:
    value = 0.0
    for share in shares.values():
        if share > 0:
            value -= share * math.log(share)
    return value


def _categorical_distribution_rows(weighted_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_distribution: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in weighted_rows:
        by_distribution[row["distribution"]].append(row)

    output = []
    for distribution, rows in sorted(by_distribution.items()):
        for field in CATEGORICAL_FIELDS:
            counts = _weighted_counts(rows, field)
            total = sum(counts.values())
            for category, weight in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                output.append(
                    {
                        "distribution": distribution,
                        "field": field,
                        "category": category,
                        "weight": weight,
                        "share": weight / total if total else float("nan"),
                    }
                )
    return output


def _numeric_summary_rows(weighted_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_distribution: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in weighted_rows:
        by_distribution[row["distribution"]].append(row)

    output = []
    for distribution, rows in sorted(by_distribution.items()):
        for field in NUMERIC_FIELDS:
            values = [
                (float(row[field]), float(row["weight"]))
                for row in rows
                if row.get(field) not in {"", None}
            ]
            total_weight = sum(weight for _, weight in values)
            if total_weight <= 0:
                output.append(
                    {
                        "distribution": distribution,
                        "field": field,
                        "n_nonmissing_weight": 0,
                        "mean": "",
                        "sd": "",
                        "missing_weight": sum(float(row["weight"]) for row in rows),
                        "missing_share": 1.0,
                    }
                )
                continue
            mean = sum(value * weight for value, weight in values) / total_weight
            variance = sum(weight * (value - mean) ** 2 for value, weight in values) / total_weight
            total_rows_weight = sum(float(row["weight"]) for row in rows)
            output.append(
                {
                    "distribution": distribution,
                    "field": field,
                    "n_nonmissing_weight": total_weight,
                    "mean": mean,
                    "sd": math.sqrt(max(variance, 0.0)),
                    "missing_weight": total_rows_weight - total_weight,
                    "missing_share": (total_rows_weight - total_weight) / total_rows_weight
                    if total_rows_weight
                    else float("nan"),
                }
            )
    return output


def _alignment_summary_rows(
    categorical_rows: list[dict[str, Any]],
    numeric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_share: dict[tuple[str, str], float] = {}
    by_dist_field: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in categorical_rows:
        key = (row["distribution"], row["field"])
        by_dist_field[key][row["category"]] = float(row["share"])
        if row["distribution"] == "candidate_uniform":
            candidate_share[(row["field"], row["category"])] = float(row["share"])

    output = []
    for (distribution, field), shares in sorted(by_dist_field.items()):
        if distribution == "candidate_uniform":
            continue
        categories = sorted(set(shares) | {category for f, category in candidate_share if f == field})
        diffs = {
            category: shares.get(category, 0.0) - candidate_share.get((field, category), 0.0)
            for category in categories
        }
        max_category = max(categories, key=lambda category: abs(diffs[category])) if categories else ""
        tv_distance = 0.5 * sum(abs(diff) for diff in diffs.values())
        output.append(
            {
                "distribution": distribution,
                "field": field,
                "metric_type": "categorical",
                "total_variation_from_candidate_uniform": tv_distance,
                "entropy": _entropy(shares),
                "max_abs_shift_category": max_category,
                "max_abs_shift": diffs.get(max_category, ""),
                "matched_value": "",
                "candidate_uniform_value": "",
                "matched_minus_candidate_uniform": "",
            }
        )

    numeric_by_key = {
        (row["distribution"], row["field"]): row
        for row in numeric_rows
        if row.get("mean") not in {"", None}
    }
    for field in NUMERIC_FIELDS:
        candidate = numeric_by_key.get(("candidate_uniform", field))
        if not candidate:
            continue
        candidate_mean = float(candidate["mean"])
        for distribution in sorted({row["distribution"] for row in numeric_rows} - {"candidate_uniform"}):
            matched = numeric_by_key.get((distribution, field))
            if not matched:
                continue
            matched_mean = float(matched["mean"])
            output.append(
                {
                    "distribution": distribution,
                    "field": field,
                    "metric_type": "numeric",
                    "total_variation_from_candidate_uniform": "",
                    "entropy": "",
                    "max_abs_shift_category": "",
                    "max_abs_shift": "",
                    "matched_value": matched_mean,
                    "candidate_uniform_value": candidate_mean,
                    "matched_minus_candidate_uniform": matched_mean - candidate_mean,
                }
            )
    return output


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_ci(values: list[float], rng: random.Random, iterations: int) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    n = len(values)
    means = []
    for _ in range(iterations):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return (_percentile(means, 0.025), _percentile(means, 0.975))


def _paired_sign_p(values: list[float]) -> float:
    positives = sum(value > 0 for value in values)
    negatives = sum(value < 0 for value in values)
    n = positives + negatives
    if n == 0:
        return float("nan")
    observed = min(positives, negatives)
    probability = sum(math.comb(n, k) for k in range(observed + 1)) / (2**n)
    return min(1.0, 2 * probability)


def _values_by_request(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        custom_id = str(row.get("custom_id", ""))
        if custom_id:
            by_request[custom_id].append(row)
    return by_request


def _request_level_difference_rows(
    weighted_rows: list[dict[str, Any]],
    categorical_rows: list[dict[str, Any]],
    iterations: int,
    seed: int,
    min_category_share: float,
) -> list[dict[str, Any]]:
    by_distribution = {
        distribution: _values_by_request(
            [row for row in weighted_rows if row["distribution"] == distribution]
        )
        for distribution in ["candidate_uniform", "matched_probability", "matched_top1"]
    }
    request_ids = sorted(
        set(by_distribution["candidate_uniform"])
        & set(by_distribution["matched_probability"])
        & set(by_distribution["matched_top1"])
    )

    candidate_or_matched_shares: dict[tuple[str, str], float] = defaultdict(float)
    for row in categorical_rows:
        if row["distribution"] in {"candidate_uniform", "matched_probability", "matched_top1"}:
            candidate_or_matched_shares[(row["field"], row["category"])] = max(
                candidate_or_matched_shares[(row["field"], row["category"])],
                float(row["share"]),
            )
    categories_by_field: dict[str, list[str]] = defaultdict(list)
    for (field, category), share in candidate_or_matched_shares.items():
        if share >= min_category_share or category in {"Man", "Woman", "Bachelor", "High school", "Master"}:
            categories_by_field[field].append(category)

    rng = random.Random(seed)
    output: list[dict[str, Any]] = []

    def numeric_value(rows: list[dict[str, Any]], field: str) -> float | None:
        values = [
            (float(row[field]), float(row["weight"]))
            for row in rows
            if row.get(field) not in {"", None}
        ]
        total = sum(weight for _, weight in values)
        if total <= 0:
            return None
        return sum(value * weight for value, weight in values) / total

    def categorical_share(rows: list[dict[str, Any]], field: str, category: str) -> float:
        total = sum(float(row["weight"]) for row in rows)
        if total <= 0:
            return 0.0
        return (
            sum(float(row["weight"]) for row in rows if str(row[field]) == category)
            / total
        )

    for distribution in ["matched_probability", "matched_top1"]:
        for field in NUMERIC_FIELDS:
            diffs = []
            for request_id in request_ids:
                candidate = numeric_value(by_distribution["candidate_uniform"][request_id], field)
                matched = numeric_value(by_distribution[distribution][request_id], field)
                if candidate is None or matched is None:
                    continue
                diffs.append(matched - candidate)
            ci_low, ci_high = _bootstrap_ci(diffs, rng, iterations)
            output.append(
                {
                    "distribution": distribution,
                    "field": field,
                    "category": "",
                    "metric_type": "numeric",
                    "n_requests": len(diffs),
                    "mean_diff": _mean(diffs),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "paired_sign_p": _paired_sign_p(diffs),
                }
            )

        for field, categories in sorted(categories_by_field.items()):
            for category in sorted(categories):
                diffs = []
                for request_id in request_ids:
                    candidate = categorical_share(
                        by_distribution["candidate_uniform"][request_id], field, category
                    )
                    matched = categorical_share(by_distribution[distribution][request_id], field, category)
                    diffs.append(matched - candidate)
                ci_low, ci_high = _bootstrap_ci(diffs, rng, iterations)
                output.append(
                    {
                        "distribution": distribution,
                        "field": field,
                        "category": category,
                        "metric_type": "categorical",
                        "n_requests": len(diffs),
                        "mean_diff": _mean(diffs),
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "paired_sign_p": _paired_sign_p(diffs),
                    }
                )
    return output


def run(args: argparse.Namespace) -> None:
    metadata_dir = args.metadata_dir.expanduser().resolve()
    demographics_path = args.demographics.expanduser().resolve()
    manifest_rows = _load_manifest(metadata_dir / "request_manifest.csv")
    match_rows = _read_csv(metadata_dir / "parsed_matches_long.csv")
    demographics = _load_demographics(demographics_path)

    weighted_rows, join_summary = _build_weighted_rows(manifest_rows, match_rows, demographics)
    categorical_rows = _categorical_distribution_rows(weighted_rows)
    numeric_rows = _numeric_summary_rows(weighted_rows)
    alignment_rows = _alignment_summary_rows(categorical_rows, numeric_rows)
    request_diff_rows = _request_level_difference_rows(
        weighted_rows=weighted_rows,
        categorical_rows=categorical_rows,
        iterations=args.iterations,
        seed=args.seed,
        min_category_share=args.min_category_share,
    )

    output_prefix = metadata_dir / "pgg_demographic"
    _write_csv(output_prefix.with_name("pgg_demographic_weighted_rows.csv"), weighted_rows)
    _write_csv(output_prefix.with_name("pgg_demographic_categorical_distributions.csv"), categorical_rows)
    _write_csv(output_prefix.with_name("pgg_demographic_numeric_summaries.csv"), numeric_rows)
    _write_csv(output_prefix.with_name("pgg_demographic_alignment_summary.csv"), alignment_rows)
    _write_csv(output_prefix.with_name("pgg_demographic_request_level_differences.csv"), request_diff_rows)

    totals = defaultdict(float)
    for row in weighted_rows:
        totals[row["distribution"]] += float(row["weight"])
    summary = {
        **join_summary,
        "metadata_dir": str(metadata_dir),
        "demographics_path": str(demographics_path),
        "requests": len(manifest_rows),
        "match_rows": len(match_rows),
        "weighted_rows": len(weighted_rows),
        "total_weight_by_distribution": dict(sorted(totals.items())),
        "categorical_fields": list(CATEGORICAL_FIELDS),
        "numeric_fields": list(NUMERIC_FIELDS),
    }
    _write_json(output_prefix.with_name("pgg_demographic_join_summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--demographics", type=Path, default=DEFAULT_DEMOGRAPHICS)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-category-share", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
