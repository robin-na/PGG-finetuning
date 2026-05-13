from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MM_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = MM_ROOT.parent
WVS_ROOT = PROJECT_ROOT / "world_values_survey"

DEFAULT_MM_RUN = (
    "mm_individual_demographic_complete_5_per_scenario_n_gt_10000_actual_order_seed0_"
    "gpt_5_mini_ab_no_logprobs"
)
DEFAULT_WVS_CSV = WVS_ROOT / "raw" / "WVS_Time_Series_1981-2022_csv_v5_0.csv"
DEFAULT_RUN_NAME = (
    "mm_individual_wvs_wave6_1donor_neutral_profile_no_overlap_exact_seed0_gpt_5_mini_ab_no_logprobs"
)

WAVE_LABELS = {
    1: "1981-1984",
    2: "1989-1993",
    3: "1994-1998",
    4: "1999-2004",
    5: "2005-2009",
    6: "2010-2014",
    7: "2017-2022",
}

MATCH_FIELDS = ["country", "gender2", "age_bin", "educ3", "income3", "political3", "religion3"]

WVS_FIELDS = [
    "S002VS",
    "COUNTRY_ALPHA",
    "S007",
    "X001",
    "X003",
    "X003R",
    "X025R",
    "X047R_WVS",
    "A001",
    "A009",
    "A029",
    "A030",
    "A032",
    "A034",
    "A038",
    "A039",
    "A041",
    "A042",
    "A043B",
    "A071",
    "A103",
    "A124_01",
    "A124_02",
    "A124_06",
    "A124_07",
    "A124_08",
    "A124_09",
    "E033",
    "A165",
    "A173",
    "A208",
    "A209",
    "A210",
    "B008",
    "C001",
    "C001_01",
    "C002",
    "C038",
    "D054",
    "D059",
    "D060",
    "D061",
    "D078",
    "E016",
    "E018",
    "E035",
    "E037",
    "E039",
    "E040",
    "E069_06",
    "E069_17",
    "E217",
    "E218",
    "A006",
    "F028",
    "F034",
    "F050",
    "F063",
    "F114B",
    "F115",
    "F116",
    "F117",
    "F120",
    "F122",
    "F123",
    "Y001",
    "Y002",
    "Y020",
]

RELEVANT_VALUE_FIELDS = [
    "A001",
    "A009",
    "A029",
    "A030",
    "A032",
    "A034",
    "A038",
    "A039",
    "A041",
    "A042",
    "A043B",
    "A071",
    "A103",
    "A124_01",
    "A124_02",
    "A124_06",
    "A124_07",
    "A124_08",
    "A124_09",
    "A165",
    "A173",
    "A208",
    "A209",
    "A210",
    "B008",
    "C001",
    "C001_01",
    "C002",
    "C038",
    "D054",
    "D059",
    "D060",
    "D061",
    "D078",
    "E016",
    "E018",
    "E035",
    "E037",
    "E039",
    "E040",
    "E069_06",
    "E069_17",
    "E217",
    "E218",
    "F114B",
    "F115",
    "F116",
    "F117",
    "F120",
    "F122",
    "F123",
    "Y001",
    "Y002",
    "Y020",
]

SYSTEM_PROMPT = (
    "You need to predict a participant's decision in the following moral dilemma scenario. "
    "Return only A or B."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an individual-level Moral Machine batch augmented with one matched WVS donor."
        )
    )
    parser.add_argument("--mm-run-name", default=DEFAULT_MM_RUN)
    parser.add_argument("--wvs-csv", type=Path, default=DEFAULT_WVS_CSV)
    parser.add_argument("--wave", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--model", default="gpt-5-mini")
    return parser.parse_args()


def parse_int(value: Any) -> int | None:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed


def parse_float(value: Any) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def estimate_tokens(text: str, *, chars_per_token: float) -> int:
    return math.ceil(len(text) / chars_per_token)


def request_input_text(system_prompt: str, user_prompt: str) -> str:
    return f"system: {system_prompt}\nuser: {user_prompt}\n"


def stable_index(key: str, n: int, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % n


def age_bin_from_age(age: Any) -> str | None:
    parsed = parse_int(age)
    if parsed is None:
        return None
    if parsed < 25:
        return "15-24"
    if parsed < 35:
        return "25-34"
    if parsed < 45:
        return "35-44"
    if parsed < 55:
        return "45-54"
    if parsed < 65:
        return "55-64"
    return "65+"


def age_bin_from_x003r(value: Any) -> str | None:
    return {
        1: "15-24",
        2: "25-34",
        3: "35-44",
        4: "45-54",
        5: "55-64",
        6: "65+",
    }.get(parse_int(value))


def thirds_bin(value: Any) -> str | None:
    parsed = parse_float(value)
    if parsed is None or parsed < 0 or parsed > 1:
        return None
    if parsed < 1.0 / 3.0:
        return "low"
    if parsed <= 2.0 / 3.0:
        return "mid"
    return "high"


def mm_gender(raw: str) -> str | None:
    return {"male": "male", "female": "female"}.get(str(raw))


def mm_education(raw: str) -> str | None:
    return {
        "underHigh": "lower",
        "high": "middle",
        "vocational": "middle",
        "college": "middle",
        "bachelor": "upper",
        "graduate": "upper",
    }.get(str(raw))


def mm_income(raw: str) -> str | None:
    return {
        "under5000": "low",
        "5000": "low",
        "10000": "low",
        "15000": "medium",
        "25000": "medium",
        "35000": "medium",
        "50000": "medium",
        "80000": "high",
        "above100000": "high",
    }.get(str(raw))


def wvs_gender(raw: Any) -> str | None:
    return {1: "male", 2: "female"}.get(parse_int(raw))


def wvs_education(raw: Any) -> str | None:
    return {1: "lower", 2: "middle", 3: "upper"}.get(parse_int(raw))


def wvs_income(raw: Any) -> str | None:
    return {1: "low", 2: "medium", 3: "high"}.get(parse_int(raw))


def wvs_political_score(raw: Any) -> float | None:
    parsed = parse_int(raw)
    if parsed is None or parsed < 1 or parsed > 10:
        return None
    return (10 - parsed) / 9.0


def wvs_religion_score(raw: Any) -> float | None:
    parsed = parse_int(raw)
    if parsed is None or parsed < 1 or parsed > 4:
        return None
    return (4 - parsed) / 3.0


def mm_match_key(row: dict[str, str]) -> tuple[str, ...] | None:
    values = {
        "country": row.get("user_country3"),
        "gender2": mm_gender(row.get("review_gender", "")),
        "age_bin": age_bin_from_age(row.get("review_age")),
        "educ3": mm_education(row.get("review_education", "")),
        "income3": mm_income(row.get("review_income", "")),
        "political3": thirds_bin(row.get("review_political")),
        "religion3": thirds_bin(row.get("review_religious")),
    }
    if any(values[field] is None for field in MATCH_FIELDS):
        return None
    return tuple(str(values[field]) for field in MATCH_FIELDS)


def wvs_match_key(row: dict[str, str]) -> tuple[str, ...] | None:
    values = {
        "country": row.get("COUNTRY_ALPHA"),
        "gender2": wvs_gender(row.get("X001")),
        "age_bin": age_bin_from_x003r(row.get("X003R")),
        "educ3": wvs_education(row.get("X025R")),
        "income3": wvs_income(row.get("X047R_WVS")),
        "political3": thirds_bin(wvs_political_score(row.get("E033"))),
        "religion3": thirds_bin(wvs_religion_score(row.get("A006"))),
    }
    if any(values[field] is None for field in MATCH_FIELDS):
        return None
    return tuple(str(values[field]) for field in MATCH_FIELDS)


def valid_positive_int(row: dict[str, str], field: str) -> int | None:
    parsed = parse_int(row.get(field))
    if parsed is None or parsed < 0:
        return None
    return parsed


def label_choice(value: int | None, labels: dict[int, str]) -> str | None:
    if value is None:
        return None
    return labels.get(value)


def coded_choice_line(label: str, value: int | None, labels: dict[int, str]) -> str | None:
    text = label_choice(value, labels)
    return f"{label}: {value} ({text})" if text else None


def value_in_range(row: dict[str, str], field: str, low: int, high: int) -> int | None:
    value = valid_positive_int(row, field)
    if value is None or value < low or value > high:
        return None
    return value


def add_section(lines: list[str], title: str, section_lines: list[str | None]) -> None:
    present = [line for line in section_lines if line]
    if not present:
        return
    lines.append(f"{title}:")
    lines.extend(f"- {line}" for line in present)


def importance_4_line(label: str, value: int | None) -> str | None:
    return coded_choice_line(
        label,
        value,
        {
            1: "very important",
            2: "rather important",
            3: "not very important",
            4: "not at all important",
        },
    )


def agree_4_line(label: str, value: int | None) -> str | None:
    return coded_choice_line(
        label,
        value,
        {
            1: "agree strongly",
            2: "agree",
            3: "disagree",
            4: "strongly disagree",
        },
    )


def agree_5_line(label: str, value: int | None) -> str | None:
    return coded_choice_line(
        label,
        value,
        {
            1: "strongly agree",
            2: "agree",
            3: "neither agree nor disagree",
            4: "disagree",
            5: "strongly disagree",
        },
    )


def confidence_line(label: str, value: int | None) -> str | None:
    return coded_choice_line(
        label,
        value,
        {1: "a great deal", 2: "quite a lot", 3: "not very much", 4: "none at all"},
    )


def binary_yes_line(label: str, value: int | None) -> str | None:
    return coded_choice_line(label, value, {0: "no", 1: "yes"})


def member_line(label: str, value: int | None) -> str | None:
    return coded_choice_line(label, value, {0: "not a member", 1: "inactive member", 2: "active member"})


def mentioned_line(label: str, value: int | None) -> str | None:
    return coded_choice_line(label, value, {0: "not mentioned", 1: "mentioned"})


def ten_point_line(label: str, value: int | None, left_anchor: str, right_anchor: str) -> str | None:
    if value is None or value < 1 or value > 10:
        return None
    return f"{label}: {value} (1={left_anchor}, 10={right_anchor})"


def future_change_line(label: str, value: int | None) -> str | None:
    return coded_choice_line(label, value, {1: "good thing", 2: "do not mind", 3: "bad thing"})


def jobs_scarce_3_line(label: str, value: int | None) -> str | None:
    return coded_choice_line(label, value, {1: "agree", 2: "disagree", 3: "neither"})


def environment_priority_line(value: int | None) -> str | None:
    return coded_choice_line(
        "Environment versus growth",
        value,
        {
            1: "protecting environment should be priority",
            2: "economic growth and jobs should be priority",
            3: "other answer",
        },
    )


def postmaterialist_12_line(value: int | None) -> str | None:
    if value is None or value < 0 or value > 5:
        return None
    return f"Post-materialist index, 12-item (0=materialist, 5=post-materialist): {value}"


def postmaterialist_4_line(value: int | None) -> str | None:
    return coded_choice_line(
        "Post-materialist index, 4-item",
        value,
        {1: "materialist", 2: "mixed", 3: "post-materialist"},
    )


def numeric_index_line(label: str, value: float | None) -> str | None:
    if value is None or value < 0:
        return None
    return f"{label}: {value:g}"


def justifiability_line(field_label: str, value: int | None) -> str | None:
    if value is None or value < 1 or value > 10:
        return None
    return f"{field_label}: {value}"


def child_quality_line(field_label: str, value: int | None) -> str | None:
    if value not in {0, 1}:
        return None
    return coded_choice_line(field_label, value, {0: "no", 1: "yes"})


def subjective_health_line(value: int | None) -> str | None:
    return coded_choice_line(
        "Subjective health",
        value,
        {1: "very good", 2: "good", 3: "fair", 4: "poor", 5: "very poor"},
    )


def wvs_value_present(row: dict[str, str], field: str) -> bool:
    parsed = parse_float(row.get(field))
    return parsed is not None and parsed >= 0


def render_wvs_profile(donor: dict[str, str], _wave: int) -> str:
    lines = [
        "Additional questionnaire responses:",
        (
            "Scale notes: justifiability ratings are 1=never justifiable, "
            "10=always justifiable. Other coded items show code and label; "
            "other numeric scales show endpoints."
        ),
    ]

    add_section(
        lines,
        "Justifiability ratings (1-10)",
        [
            justifiability_line("Stealing property", value_in_range(donor, "F114B", 1, 10)),
            justifiability_line("Avoiding a fare on public transport", value_in_range(donor, "F115", 1, 10)),
            justifiability_line("Cheating on taxes", value_in_range(donor, "F116", 1, 10)),
            justifiability_line("Accepting a bribe", value_in_range(donor, "F117", 1, 10)),
            justifiability_line("Abortion", value_in_range(donor, "F120", 1, 10)),
            justifiability_line("Euthanasia", value_in_range(donor, "F122", 1, 10)),
            justifiability_line("Suicide", value_in_range(donor, "F123", 1, 10)),
        ],
    )

    add_section(
        lines,
        "Confidence and authority",
        [
            confidence_line("Confidence in police", value_in_range(donor, "E069_06", 1, 4)),
            confidence_line("Confidence in courts", value_in_range(donor, "E069_17", 1, 4)),
            future_change_line("Greater respect for authority would be", value_in_range(donor, "E018", 1, 3)),
        ],
    )

    jobs_gender = agree_5_line(
        "When jobs are scarce, men should have more right to a job than women",
        value_in_range(donor, "C001_01", 1, 5),
    )
    if jobs_gender is None:
        jobs_gender = jobs_scarce_3_line(
            "When jobs are scarce, men should have more right to a job than women",
            value_in_range(donor, "C001", 1, 3),
        )
    add_section(
        lines,
        "Statements about gender, work, and family",
        [
            agree_4_line("Men make better political leaders than women", value_in_range(donor, "D059", 1, 4)),
            agree_4_line("University more important for a boy than a girl", value_in_range(donor, "D060", 1, 4)),
            agree_4_line("Men make better business executives than women", value_in_range(donor, "D078", 1, 4)),
            jobs_gender,
            agree_4_line("When a mother works for pay, the children suffer", value_in_range(donor, "D061", 1, 4)),
            agree_4_line("Main goal in life has been to make parents proud", value_in_range(donor, "D054", 1, 4)),
            importance_4_line("Family in life", value_in_range(donor, "A001", 1, 4)),
        ],
    )

    add_section(
        lines,
        "Child qualities selected as important",
        [
            child_quality_line("Independence", valid_positive_int(donor, "A029")),
            child_quality_line("Hard work", valid_positive_int(donor, "A030")),
            child_quality_line("Feeling of responsibility", valid_positive_int(donor, "A032")),
            child_quality_line("Imagination", valid_positive_int(donor, "A034")),
            child_quality_line("Thrift", valid_positive_int(donor, "A038")),
            child_quality_line("Determination and perseverance", valid_positive_int(donor, "A039")),
            child_quality_line("Unselfishness", valid_positive_int(donor, "A041")),
            child_quality_line("Obedience", valid_positive_int(donor, "A042")),
            child_quality_line("Self-expression", valid_positive_int(donor, "A043B")),
        ],
    )

    add_section(
        lines,
        "Statements about older people",
        [
            agree_4_line("Older people are not respected much these days", value_in_range(donor, "A208", 1, 4)),
            agree_4_line("Older people get more than their fair share from government", value_in_range(donor, "A209", 1, 4)),
            agree_4_line("Older people are a burden on society", value_in_range(donor, "A210", 1, 4)),
        ],
    )

    add_section(
        lines,
        "Work, inequality, and responsibility",
        [
            ten_point_line(
                "Income equality versus incentives",
                value_in_range(donor, "E035", 1, 10),
                "incomes should be made more equal",
                "greater incentives for individual effort",
            ),
            ten_point_line(
                "Government responsibility",
                value_in_range(donor, "E037", 1, 10),
                "people should take more responsibility",
                "government should take more responsibility",
            ),
            ten_point_line(
                "Competition",
                value_in_range(donor, "E039", 1, 10),
                "competition is good",
                "competition is harmful",
            ),
            ten_point_line(
                "Hard work brings success",
                value_in_range(donor, "E040", 1, 10),
                "hard work brings a better life",
                "success is luck and connections",
            ),
            agree_5_line("People who do not work turn lazy", value_in_range(donor, "C038", 1, 5)),
        ],
    )

    trust = coded_choice_line(
        "General trust",
        value_in_range(donor, "A165", 1, 2),
        {1: "most people can be trusted", 2: "need to be very careful"},
    )
    add_section(
        lines,
        "Trust and neighbor preferences",
        [
            trust,
            mentioned_line("Would not like neighbor with criminal record", valid_positive_int(donor, "A124_01")),
            mentioned_line("Would not like neighbor of different race", valid_positive_int(donor, "A124_02")),
            mentioned_line("Would not like immigrant/foreign-worker neighbor", valid_positive_int(donor, "A124_06")),
            mentioned_line("Would not like neighbor with AIDS", valid_positive_int(donor, "A124_07")),
            mentioned_line("Would not like drug-addict neighbor", valid_positive_int(donor, "A124_08")),
            mentioned_line("Would not like homosexual neighbor", valid_positive_int(donor, "A124_09")),
            jobs_scarce_3_line(
                "When jobs are scarce, employers should prioritize nationals over immigrants",
                value_in_range(donor, "C002", 1, 3),
            ),
        ],
    )

    add_section(
        lines,
        "Environment and social priorities",
        [
            environment_priority_line(value_in_range(donor, "B008", 1, 3)),
            mentioned_line("Belongs to environmental or animal-rights organization", valid_positive_int(donor, "A071")),
            member_line("Environmental organization membership", value_in_range(donor, "A103", 0, 2)),
            postmaterialist_12_line(value_in_range(donor, "Y001", 0, 5)),
            postmaterialist_4_line(value_in_range(donor, "Y002", 1, 3)),
            numeric_index_line(
                "Welzel emancipative values index (continuous; higher=more emancipative)",
                parse_float(donor.get("Y020")),
            ),
        ],
    )

    add_section(
        lines,
        "Freedom, technology, and science",
        [
            ten_point_line(
                "Freedom of choice and control over life",
                value_in_range(donor, "A173", 1, 10),
                "none at all",
                "a great deal",
            ),
            future_change_line("More emphasis on technology would be", value_in_range(donor, "E016", 1, 3)),
            ten_point_line(
                "Science and technology make life healthier/easier",
                value_in_range(donor, "E217", 1, 10),
                "completely disagree",
                "completely agree",
            ),
            ten_point_line(
                "Science and technology create next-generation opportunities",
                value_in_range(donor, "E218", 1, 10),
                "completely disagree",
                "completely agree",
            ),
        ],
    )

    add_section(
        lines,
        "Health",
        [
            subjective_health_line(value_in_range(donor, "A009", 1, 5)),
        ],
    )

    return "\n".join(lines)


def augment_prompt(original_prompt: str, donor_profile: str) -> str:
    marker = "\n\nA self-driving car's brakes suddenly fail"
    if marker not in original_prompt:
        raise ValueError("Could not find scenario marker in MM prompt.")
    prefix, suffix = original_prompt.split(marker, 1)
    return f"{prefix}\n\n{donor_profile}{marker}{suffix}"


def build_batch_entry(*, custom_id: str, model: str, prompt: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
    }


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"No header in {path}")
        return rows, list(reader.fieldnames)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def load_wvs_donor_pool(path: Path, wave: int) -> tuple[dict[tuple[str, ...], list[dict[str, str]]], dict[str, Any]]:
    donors_by_key: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    stats = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in WVS_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"WVS CSV missing expected columns: {missing}")
        for row in reader:
            if parse_int(row.get("S002VS")) != wave:
                continue
            stats["wave_rows"] += 1
            key = wvs_match_key(row)
            if key is None:
                stats["missing_match_key"] += 1
                continue
            stats["usable_donor_rows"] += 1
            donors_by_key[key].append({field: row.get(field, "") for field in WVS_FIELDS})
    stats["match_cells"] = len(donors_by_key)
    return donors_by_key, dict(stats)


def output_fieldnames(source_fieldnames: list[str]) -> list[str]:
    extra = [
        "source_custom_id",
        "wvs_wave",
        "wvs_wave_years",
        "wvs_donor_s007",
        "wvs_match_key",
        "wvs_donor_pool_size",
        "wvs_profile_text",
        "wvs_values_included",
        "source_run_name",
        "prompt_version",
        "input_char_count",
        "estimated_input_tokens_4_chars_per_token",
        "estimated_input_tokens_3_5_chars_per_token",
    ]
    fields: list[str] = []
    for field in source_fieldnames:
        if field in {"source_custom_id"}:
            continue
        fields.append(field)
        if field == "custom_id":
            fields.append("source_custom_id")
    for field in extra:
        if field not in fields:
            fields.append(field)
    return fields


def main() -> None:
    args = parse_args()
    source_manifest_path = MM_ROOT / "metadata" / args.mm_run_name / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_csv_path = Path(source_manifest["sample_manifest_file"])
    source_rows, source_fieldnames = read_csv(source_csv_path)
    donors_by_key, donor_stats = load_wvs_donor_pool(args.wvs_csv, args.wave)

    batch_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    skip_reasons = Counter()
    donor_pool_sizes: list[int] = []
    token_4: list[int] = []
    token_35: list[int] = []
    input_chars: list[int] = []
    included_value_counts = Counter()

    for row in source_rows:
        key = mm_match_key(row)
        if key is None:
            skip_reasons["missing_or_unmappable_mm_match_key"] += 1
            continue
        donor_pool = donors_by_key.get(key, [])
        if not donor_pool:
            skip_reasons["no_exact_wvs_wave_donor"] += 1
            continue
        donor = donor_pool[stable_index(row["custom_id"], len(donor_pool), args.seed)]
        donor_profile = render_wvs_profile(donor, args.wave)
        included_values = [
            field for field in RELEVANT_VALUE_FIELDS if wvs_value_present(donor, field)
        ]
        source_custom_id = row["custom_id"]
        custom_id = f"{source_custom_id}_wvsw{args.wave}_d1"
        prompt = augment_prompt(row["prompt"], donor_profile)
        input_text = request_input_text(SYSTEM_PROMPT, prompt)
        input_char_count = len(input_text)
        estimated_4 = estimate_tokens(input_text, chars_per_token=4.0)
        estimated_35 = estimate_tokens(input_text, chars_per_token=3.5)
        batch_rows.append(build_batch_entry(custom_id=custom_id, model=args.model, prompt=prompt))

        manifest_row: dict[str, Any] = dict(row)
        manifest_row["custom_id"] = custom_id
        manifest_row["source_custom_id"] = source_custom_id
        manifest_row["prompt"] = prompt
        manifest_row["wvs_wave"] = args.wave
        manifest_row["wvs_wave_years"] = WAVE_LABELS.get(args.wave, "")
        manifest_row["wvs_donor_s007"] = donor.get("S007", "")
        manifest_row["wvs_match_key"] = "|".join(key)
        manifest_row["wvs_donor_pool_size"] = len(donor_pool)
        manifest_row["wvs_profile_text"] = donor_profile
        manifest_row["wvs_values_included"] = "|".join(included_values)
        manifest_row["source_run_name"] = args.mm_run_name
        manifest_row["prompt_version"] = "wvs_wave_one_donor_neutral_compact_coded_profile_no_overlap_exact_v1"
        manifest_row["input_char_count"] = input_char_count
        manifest_row["estimated_input_tokens_4_chars_per_token"] = estimated_4
        manifest_row["estimated_input_tokens_3_5_chars_per_token"] = estimated_35
        manifest_rows.append(manifest_row)

        donor_pool_sizes.append(len(donor_pool))
        token_4.append(estimated_4)
        token_35.append(estimated_35)
        input_chars.append(input_char_count)
        included_value_counts.update(included_values)

    if not batch_rows:
        raise ValueError(f"No WVS donor-augmented rows were generated. Skip reasons: {dict(skip_reasons)}")

    batch_path = MM_ROOT / "batch_input" / f"{args.run_name}.jsonl"
    output_path = MM_ROOT / "batch_output" / f"{args.run_name}.jsonl"
    metadata_dir = MM_ROOT / "metadata" / args.run_name
    manifest_csv_path = metadata_dir / "sample_manifest.csv"
    manifest_json_path = metadata_dir / "manifest.json"
    prompt_preview_path = metadata_dir / "sample_prompt.txt"

    write_jsonl(batch_path, batch_rows)
    write_csv(manifest_csv_path, manifest_rows, output_fieldnames(source_fieldnames))
    metadata_dir.mkdir(parents=True, exist_ok=True)
    prompt_preview_path.write_text(manifest_rows[0]["prompt"] + "\n", encoding="utf-8")

    token_estimate = {
        "method": "Approximate from rendered system+user message text.",
        "input_characters_total": sum(input_chars),
        "estimated_input_tokens_4_chars_per_token_total": sum(token_4),
        "estimated_input_tokens_3_5_chars_per_token_total": sum(token_35),
        "estimated_input_tokens_4_chars_per_token_mean": sum(token_4) / len(token_4),
        "estimated_input_tokens_3_5_chars_per_token_mean": sum(token_35) / len(token_35),
        "min_estimated_input_tokens_4_chars_per_token": min(token_4),
        "max_estimated_input_tokens_4_chars_per_token": max(token_4),
    }
    run_manifest = {
        "run_name": args.run_name,
        "source_run_name": args.mm_run_name,
        "source_manifest_file": str(source_manifest_path),
        "source_sample_manifest_file": str(source_csv_path),
        "model": args.model,
        "metadata_dir": str(metadata_dir),
        "batch_input_file": str(batch_path),
        "expected_batch_output_file": str(output_path),
        "sample_manifest_file": str(manifest_csv_path),
        "sample_prompt_file": str(prompt_preview_path),
        "endpoint": "/v1/chat/completions",
        "logprobs": False,
        "temperature": None,
        "condition": "individual_demographic_wvs_wave_one_donor_neutral_compact_coded_profile_no_overlap_exact",
        "prompt_version": "wvs_wave_one_donor_neutral_compact_coded_profile_no_overlap_exact_v1",
        "system_prompt": SYSTEM_PROMPT,
        "wvs_source_file": str(args.wvs_csv),
        "wvs_wave": args.wave,
        "wvs_wave_years": WAVE_LABELS.get(args.wave),
        "donors_per_mm_participant": 1,
        "matching": {
            "type": "exact",
            "fields": MATCH_FIELDS,
            "politics": "MM review_political and WVS E033 both binned into low/mid/high progressive orientation.",
            "religion": "MM review_religious and WVS A006 importance of religion both binned into low/mid/high religiosity.",
            "unmatched_mm_rows_are_excluded": True,
        },
        "profile_rendering": {
            "prompt_wording": (
                "The rendered prompt calls the matched values 'Additional questionnaire responses' "
                "and does not mention WVS, donor selection, imputation, survey wave, or Moral Machine "
                "factor labels. Repeated scale anchors are compacted into a scale note; categorical "
                "items show both code and label."
            ),
            "displayed_profile_excludes_fields_overlapping_observed_mm_demographics": [
                "COUNTRY_ALPHA",
                "X001",
                "X003",
                "X003R",
                "X025R",
                "X047R_WVS",
                "E033",
                "A006",
                "F028",
                "F034",
                "F050",
                "F063",
                "A040",
            ],
        },
        "profile_field_groups": {
            "justifiability_ratings": ["F114B", "F115", "F116", "F117", "F120", "F122", "F123"],
            "confidence_and_authority": ["E069_06", "E069_17", "E018"],
            "gender_work_and_family_statements": ["D059", "D060", "D078", "C001", "C001_01", "D061", "D054", "A001"],
            "child_qualities_selected_as_important": ["A029", "A030", "A032", "A034", "A038", "A039", "A041", "A042", "A043B"],
            "statements_about_older_people": ["A208", "A209", "A210"],
            "work_inequality_and_responsibility": ["E035", "E037", "E039", "E040", "C038"],
            "trust_and_neighbor_preferences": ["A165", "A124_01", "A124_02", "A124_06", "A124_07", "A124_08", "A124_09", "C002"],
            "environment_and_social_priorities": ["B008", "A071", "A103", "Y001", "Y002", "Y020"],
            "freedom_technology_and_science": ["A173", "E016", "E217", "E218"],
            "health": ["A009"],
        },
        "relevant_wvs_fields": {
            "A001": "importance of family",
            "A009": "subjective health",
            "A029": "child quality: independence",
            "A030": "child quality: hard work",
            "A032": "child quality: responsibility",
            "A034": "child quality: imagination",
            "A038": "child quality: thrift",
            "A039": "child quality: determination and perseverance",
            "A041": "child quality: unselfishness",
            "A042": "child quality: obedience",
            "A043B": "child quality: self-expression",
            "A071": "environmental or animal-rights organization membership mentioned",
            "A103": "active/inactive environmental organization membership",
            "A124_01": "would not like neighbors with criminal record",
            "A124_02": "would not like neighbors of different race",
            "A124_06": "would not like immigrant/foreign-worker neighbors",
            "A124_07": "would not like neighbors with AIDS",
            "A124_08": "would not like drug addict neighbors",
            "A124_09": "would not like homosexual neighbors",
            "A165": "general trust",
            "A173": "freedom of choice and control over life",
            "A208": "older people are not respected much these days",
            "A209": "older people get more than fair share from government",
            "A210": "older people are a burden on society",
            "B008": "environmental protection versus economic growth",
            "C001": "jobs scarce: men more right to jobs, 3-category",
            "C001_01": "jobs scarce: men more right to jobs, 5-category",
            "C002": "jobs scarce: nationals over immigrants",
            "C038": "people who do not work turn lazy",
            "D054": "main goal in life to make parents proud",
            "D059": "men make better political leaders than women",
            "D060": "university more important for boys than girls",
            "D061": "children suffer when mother works for pay",
            "D078": "men make better business executives than women",
            "E016": "future change: more emphasis on technology",
            "E018": "future change: greater respect for authority",
            "E035": "income equality versus individual incentives",
            "E037": "personal versus government responsibility",
            "E039": "competition good versus harmful",
            "E040": "hard work brings success versus luck/connections",
            "E069_06": "confidence in police",
            "E069_17": "confidence in courts",
            "E217": "science and technology make life healthier/easier",
            "E218": "science and technology create next-generation opportunities",
            "F114B": "justifiable: stealing property",
            "F115": "justifiable: avoiding fare on public transport",
            "F116": "justifiable: cheating on taxes",
            "F117": "justifiable: accepting a bribe",
            "F120": "justifiable: abortion",
            "F122": "justifiable: euthanasia",
            "F123": "justifiable: suicide",
            "Y001": "post-materialist index, 12-item",
            "Y002": "post-materialist index, 4-item",
            "Y020": "Welzel emancipative values index",
        },
        "num_source_requests": len(source_rows),
        "num_requests": len(batch_rows),
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "donor_pool_stats": {
            **donor_stats,
            "matched_mm_rows": len(batch_rows),
            "min_donor_pool_size": min(donor_pool_sizes),
            "median_donor_pool_size": sorted(donor_pool_sizes)[len(donor_pool_sizes) // 2],
            "max_donor_pool_size": max(donor_pool_sizes),
        },
        "included_value_counts": dict(sorted(included_value_counts.items())),
        "answer_labels": source_manifest.get("answer_labels"),
        "option_order": source_manifest.get("option_order"),
        "not_submitted": True,
        "token_estimate": token_estimate,
    }
    manifest_json_path.write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_name": args.run_name,
                "batch_input_file": str(batch_path),
                "sample_manifest_file": str(manifest_csv_path),
                "sample_prompt_file": str(prompt_preview_path),
                "manifest_file": str(manifest_json_path),
                "num_source_requests": len(source_rows),
                "num_requests": len(batch_rows),
                "skip_reasons": dict(sorted(skip_reasons.items())),
                "donor_pool_stats": run_manifest["donor_pool_stats"],
                "estimated_input_tokens_4_chars": token_estimate[
                    "estimated_input_tokens_4_chars_per_token_total"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
