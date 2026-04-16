from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    _minimal_output_target,
    build_generated_game_records_df,
    build_human_game_records_df,
    load_parsed_outputs_df,
    write_csv,
    write_json,
)


TURN_FIELDS = ["sender_id", "buy", "sell", "status", "recipient_id", "responses"]


def _row_metrics(predicted: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"exact_match": int(predicted == gold)}
    turn_exact_matches: list[int] = []
    field_matches: list[int] = []

    predicted_rounds = predicted.get("rounds") or []
    gold_rounds = gold.get("rounds") or []
    for predicted_round, gold_round in zip(predicted_rounds, gold_rounds):
        for predicted_turn, gold_turn in zip(predicted_round.get("turns") or [], gold_round.get("turns") or []):
            turn_exact_matches.append(int(predicted_turn == gold_turn))
            for field_name in TURN_FIELDS:
                field_matches.append(int(predicted_turn.get(field_name) == gold_turn.get(field_name)))

    metrics["turn_exact_match_rate"] = (
        float(sum(turn_exact_matches) / len(turn_exact_matches)) if turn_exact_matches else float("nan")
    )
    metrics["turn_field_match_rate"] = (
        float(sum(field_matches) / len(field_matches)) if field_matches else float("nan")
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate parsed chip-bargain outputs against the sampled human gold targets."
    )
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--parsed-output-jsonl", type=Path, default=None)
    parser.add_argument("--request-manifest-csv", type=Path, default=None)
    parser.add_argument("--gold-targets-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.run_name:
        metadata_dir = args.forecasting_root / "metadata" / args.run_name
        args.parsed_output_jsonl = args.parsed_output_jsonl or (metadata_dir / "parsed_output.jsonl")
        args.request_manifest_csv = args.request_manifest_csv or (metadata_dir / "request_manifest.csv")
        args.gold_targets_jsonl = args.gold_targets_jsonl or (metadata_dir / "gold_targets.jsonl")
        args.output_dir = args.output_dir or (args.forecasting_root / "results" / f"{args.run_name}__gold_eval")

    if (
        args.parsed_output_jsonl is None
        or args.request_manifest_csv is None
        or args.gold_targets_jsonl is None
        or args.output_dir is None
    ):
        raise ValueError(
            "Provide either --run-name or all of --parsed-output-jsonl, --request-manifest-csv, --gold-targets-jsonl, and --output-dir."
        )

    parsed_df = load_parsed_outputs_df(args.parsed_output_jsonl)
    human_games = build_human_game_records_df(
        gold_targets_jsonl=args.gold_targets_jsonl,
        request_manifest_csv=args.request_manifest_csv,
    )
    generated_games = build_generated_game_records_df(
        parsed_output_jsonl=args.parsed_output_jsonl,
        request_manifest_csv=args.request_manifest_csv,
    )

    gold_lookup = {
        str(row["custom_id"]): _minimal_output_target(row["gold_target"])
        for row in pd.read_json(args.gold_targets_jsonl, lines=True).to_dict(orient="records")
    }

    row_eval_rows: list[dict[str, Any]] = []
    for row in parsed_df.to_dict(orient="records"):
        custom_id = str(row["custom_id"])
        parse_success = bool(row.get("parse_success"))
        eval_row = {
            "custom_id": custom_id,
            "parse_success": parse_success,
            "evaluated": False,
        }
        if not parse_success or custom_id not in gold_lookup:
            row_eval_rows.append(eval_row)
            continue
        predicted = _minimal_output_target(row.get("parsed_target") or {})
        gold = gold_lookup[custom_id]
        eval_row.update(_row_metrics(predicted, gold))
        eval_row["evaluated"] = True
        row_eval_rows.append(eval_row)

    row_eval_df = pd.DataFrame(row_eval_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "row_level_evaluation.csv", row_eval_df)

    evaluated = row_eval_df[row_eval_df["evaluated"].astype(bool)].copy()
    overall_summary = {
        "parsed_output_jsonl": str(args.parsed_output_jsonl),
        "request_manifest_csv": str(args.request_manifest_csv),
        "gold_targets_jsonl": str(args.gold_targets_jsonl),
        "note": "Row-level exact-match diagnostics are secondary. Use treatment-level distribution-distance analysis as the primary benchmark.",
        "generated_parse_success_rate": (
            float(parsed_df["parse_success"].astype(bool).mean()) if not parsed_df.empty else float("nan")
        ),
        "generated_parsed_count": int(len(generated_games)),
        "human_gold_count": int(len(human_games)),
        "evaluated_row_count": int(len(evaluated)),
        "exact_match_rate": float(evaluated["exact_match"].mean()) if not evaluated.empty else float("nan"),
        "turn_exact_match_rate": (
            float(evaluated["turn_exact_match_rate"].mean()) if not evaluated.empty else float("nan")
        ),
        "turn_field_match_rate": (
            float(evaluated["turn_field_match_rate"].mean()) if not evaluated.empty else float("nan")
        ),
    }
    write_json(args.output_dir / "overall_summary.json", overall_summary)
    write_json(
        args.output_dir / "manifest.json",
        {
            "parsed_output_jsonl": str(args.parsed_output_jsonl),
            "request_manifest_csv": str(args.request_manifest_csv),
            "gold_targets_jsonl": str(args.gold_targets_jsonl),
            "output_dir": str(args.output_dir),
            "primary_metric_family": "distribution_distance",
            "secondary_metric_family": "row_level_exact_or_match_rate",
        },
    )


if __name__ == "__main__":
    main()
