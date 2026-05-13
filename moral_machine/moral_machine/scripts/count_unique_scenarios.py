from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator


SCRIPT_DIR = Path(__file__).resolve().parent
MM_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = MM_ROOT / "raw" / "moral_machine_data" / "SharedResponses.csv.tar.gz"
EXTRACTED_INPUT = MM_ROOT / "raw" / "moral_machine_data" / "SharedResponses.csv"
if EXTRACTED_INPUT.exists():
    DEFAULT_INPUT = EXTRACTED_INPUT
DEFAULT_OUTPUT = MM_ROOT / "metadata" / "full_unique_scenario_counts.json"

CHARACTER_COLUMNS = [
    "Man",
    "Woman",
    "Pregnant",
    "Stroller",
    "OldMan",
    "OldWoman",
    "Boy",
    "Girl",
    "Homeless",
    "LargeWoman",
    "LargeMan",
    "Criminal",
    "MaleExecutive",
    "FemaleExecutive",
    "FemaleAthlete",
    "MaleAthlete",
    "FemaleDoctor",
    "MaleDoctor",
    "Dog",
    "Cat",
]

PAIR_METADATA_COLUMNS = [
    "PedPed",
    "AttributeLevel",
    "ScenarioTypeStrict",
    "ScenarioType",
    "DefaultChoice",
    "NonDefaultChoice",
    "DefaultChoiceIsOmission",
    "NumberOfCharacters",
    "DiffNumberOFCharacters",
]

EXCLUDED_FROM_SCENARIO = [
    "ResponseID",
    "ExtendedSessionID",
    "UserID",
    "ScenarioOrder",
    "Saved",
    "Template",
    "DescriptionShown",
    "LeftHand",
    "UserCountry3",
]


class OutcomeStream:
    def __init__(self, path: Path, intervention: str) -> None:
        self.path = path
        self.intervention = intervention
        command = ["cat", str(path)] if path.suffix.lower() == ".csv" else ["tar", "-xOzf", str(path)]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if self.process.stdout is None:
            raise RuntimeError("tar stdout was not captured")
        self.text = io.TextIOWrapper(self.process.stdout, encoding="utf-8", errors="replace", newline="")
        self.reader = csv.reader(self.text)
        self.header = next(self.reader)
        self.index = {name: idx for idx, name in enumerate(self.header)}
        self.previous_response_id: str | None = None
        self.rows_seen = 0
        self.filtered_rows_seen = 0
        self.unsorted_transitions = 0

    def __iter__(self) -> Iterator[tuple[str, list[str]]]:
        response_idx = self.index["ResponseID"]
        intervention_idx = self.index["Intervention"]
        for row in self.reader:
            self.rows_seen += 1
            if len(row) <= intervention_idx or row[intervention_idx] != self.intervention:
                continue
            response_id = row[response_idx]
            if self.previous_response_id is not None and response_id < self.previous_response_id:
                self.unsorted_transitions += 1
            self.previous_response_id = response_id
            self.filtered_rows_seen += 1
            yield response_id, row

    def close(self) -> tuple[int | None, str]:
        if self.process.stdout is not None:
            self.process.stdout.close()
        stderr = ""
        if self.process.stderr is not None:
            stderr = self.process.stderr.read().decode("utf-8", errors="replace")
        code = self.process.wait()
        return code, stderr


def value(row: list[str], index: dict[str, int], column: str) -> str:
    idx = index[column]
    if idx >= len(row):
        return ""
    return row[idx]


def stable_hash(parts: list[str]) -> bytes:
    payload = "\x1f".join(parts).encode("utf-8", errors="replace")
    return hashlib.blake2b(payload, digest_size=16).digest()


def visible_signature(stay_row: list[str], swerve_row: list[str], index: dict[str, int]) -> bytes:
    parts: list[str] = []
    for action, row in [("stay", stay_row), ("swerve", swerve_row)]:
        parts.append(action)
        parts.append(value(row, index, "Barrier"))
        parts.append(value(row, index, "CrossingSignal"))
        for column in CHARACTER_COLUMNS:
            parts.append(value(row, index, column))
    return stable_hash(parts)


def feature_signature(stay_row: list[str], swerve_row: list[str], index: dict[str, int]) -> bytes:
    parts: list[str] = []
    for action, row in [("stay", stay_row), ("swerve", swerve_row)]:
        parts.append(action)
        for column in ["Intervention", *PAIR_METADATA_COLUMNS, "Barrier", "CrossingSignal", *CHARACTER_COLUMNS]:
            parts.append(value(row, index, column))
    return stable_hash(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count unique Moral Machine paired Stay/Swerve scenario definitions in SharedResponses.csv.tar.gz."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    return parser.parse_args()


def next_or_none(iterator: Iterator[tuple[str, list[str]]]) -> tuple[str, list[str]] | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def main() -> None:
    args = parse_args()
    start = time.time()
    stay_stream = OutcomeStream(args.input, "0")
    swerve_stream = OutcomeStream(args.input, "1")
    if stay_stream.header != swerve_stream.header:
        raise ValueError("Stay/Swerve stream headers differ.")
    index = stay_stream.index

    stay_iter = iter(stay_stream)
    swerve_iter = iter(swerve_stream)
    stay_item = next_or_none(stay_iter)
    swerve_item = next_or_none(swerve_iter)

    matched_pairs = 0
    stay_unpaired = 0
    swerve_unpaired = 0
    visible_hashes: set[bytes] = set()
    feature_hashes: set[bytes] = set()

    while stay_item is not None and swerve_item is not None:
        stay_id, stay_row = stay_item
        swerve_id, swerve_row = swerve_item
        if stay_id == swerve_id:
            matched_pairs += 1
            visible_hashes.add(visible_signature(stay_row, swerve_row, index))
            feature_hashes.add(feature_signature(stay_row, swerve_row, index))
            if args.progress_every and matched_pairs % args.progress_every == 0:
                elapsed = time.time() - start
                print(
                    json.dumps(
                        {
                            "matched_pairs": matched_pairs,
                            "unique_visible": len(visible_hashes),
                            "unique_feature": len(feature_hashes),
                            "elapsed_seconds": round(elapsed, 1),
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            stay_item = next_or_none(stay_iter)
            swerve_item = next_or_none(swerve_iter)
        elif stay_id < swerve_id:
            stay_unpaired += 1
            stay_item = next_or_none(stay_iter)
        else:
            swerve_unpaired += 1
            swerve_item = next_or_none(swerve_iter)

    while stay_item is not None:
        stay_unpaired += 1
        stay_item = next_or_none(stay_iter)
    while swerve_item is not None:
        swerve_unpaired += 1
        swerve_item = next_or_none(swerve_iter)

    stay_code, stay_stderr = stay_stream.close()
    swerve_code, swerve_stderr = swerve_stream.close()
    elapsed = time.time() - start

    result = {
        "source_file": str(args.input),
        "definition": {
            "visible_unique_scenario": (
                "A paired Stay/Swerve scenario after excluding respondent/session/order/choice/presentation fields; "
                "uses only the outcome text-relevant fields Barrier, CrossingSignal, and the 20 character counts "
                "for each action."
            ),
            "feature_unique_scenario": (
                "A paired Stay/Swerve scenario using the visible fields plus MM scenario metadata columns "
                f"{PAIR_METADATA_COLUMNS}."
            ),
            "excluded_fields": EXCLUDED_FROM_SCENARIO,
        },
        "matched_response_id_pairs": matched_pairs,
        "stay_rows_without_swerve_pair": stay_unpaired,
        "swerve_rows_without_stay_pair": swerve_unpaired,
        "unique_visible_scenarios": len(visible_hashes),
        "unique_feature_scenarios": len(feature_hashes),
        "stay_filtered_rows_seen": stay_stream.filtered_rows_seen,
        "swerve_filtered_rows_seen": swerve_stream.filtered_rows_seen,
        "stay_stream_unsorted_transitions": stay_stream.unsorted_transitions,
        "swerve_stream_unsorted_transitions": swerve_stream.unsorted_transitions,
        "elapsed_seconds": elapsed,
        "tar_exit_codes": {"stay_stream": stay_code, "swerve_stream": swerve_code},
        "tar_stderr": {
            "stay_stream": stay_stderr.strip(),
            "swerve_stream": swerve_stderr.strip(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
