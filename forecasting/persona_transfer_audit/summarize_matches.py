from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        p = count / total
        value -= p * math.log(p)
    return value


def _effective_n(counts: Counter[str]) -> float:
    return math.exp(_entropy(counts))


def summarize(args: argparse.Namespace) -> None:
    metadata_dir = args.metadata_dir.expanduser().resolve()
    rows = _read_jsonl(metadata_dir / "parsed_matches.jsonl")

    overall = Counter(str(row["most_aligned_player"]) for row in rows)
    least = Counter(str(row["least_aligned_player"]) for row in rows)
    by_treatment: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_treatment[str(row.get("treatment_name", ""))][str(row["most_aligned_player"])] += 1

    total = sum(overall.values())
    player_rows = []
    for player, count in overall.most_common():
        player_rows.append(
            {
                "player": player,
                "most_aligned_count": count,
                "most_aligned_share": count / total if total else 0.0,
                "least_aligned_count": least.get(player, 0),
                "least_aligned_share": least.get(player, 0) / total if total else 0.0,
            }
        )

    treatment_rows = []
    for treatment, counts in sorted(by_treatment.items()):
        treatment_total = sum(counts.values())
        for player, count in counts.most_common():
            treatment_rows.append(
                {
                    "treatment_name": treatment,
                    "player": player,
                    "most_aligned_count": count,
                    "most_aligned_share": count / treatment_total if treatment_total else 0.0,
                    "treatment_total": treatment_total,
                }
            )

    concentration = {
        "n_matches": total,
        "n_unique_most_aligned_players": len(overall),
        "top_player": overall.most_common(1)[0][0] if overall else None,
        "top_player_share": overall.most_common(1)[0][1] / total if total else 0.0,
        "entropy": _entropy(overall),
        "effective_n": _effective_n(overall),
    }

    _write_csv(metadata_dir / "matched_player_distribution.csv", player_rows)
    _write_csv(metadata_dir / "matched_player_distribution_by_treatment.csv", treatment_rows)
    (metadata_dir / "match_concentration_summary.json").write_text(
        json.dumps(concentration, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(concentration, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize persona transfer match outputs.")
    parser.add_argument("--metadata-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    summarize(parse_args())


if __name__ == "__main__":
    main()

