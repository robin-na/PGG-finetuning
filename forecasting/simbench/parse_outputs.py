from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    extract_json_object_text,
    extract_text_from_response_record,
    load_request_manifest_df,
    validate_batched_prediction_payload,
    write_json,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse SimBench batched JSON outputs into per-question distributions."
    )
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--batch-output-jsonl", type=Path, default=None)
    parser.add_argument("--request-manifest-csv", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    args = parser.parse_args()

    if args.run_name:
        metadata_dir = args.forecasting_root / "metadata" / args.run_name
        args.batch_output_jsonl = args.batch_output_jsonl or (
            args.forecasting_root / "batch_output" / f"{args.run_name}.jsonl"
        )
        args.request_manifest_csv = args.request_manifest_csv or (metadata_dir / "request_manifest.csv")
        args.output_jsonl = args.output_jsonl or (metadata_dir / "parsed_output.jsonl")

    if args.batch_output_jsonl is None or args.request_manifest_csv is None or args.output_jsonl is None:
        raise ValueError(
            "Provide either --run-name or all of --batch-output-jsonl, --request-manifest-csv, and --output-jsonl."
        )

    manifest = load_request_manifest_df(args.request_manifest_csv).set_index("custom_id")
    raw_rows: list[dict[str, Any]] = []
    with args.batch_output_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw_rows.append(json.loads(line))

    parsed_rows: list[dict[str, Any]] = []
    full_valid_request_count = 0
    request_with_any_valid_answers_count = 0
    expected_question_total = 0
    valid_question_total = 0

    for record in raw_rows:
        custom_id = str(record.get("custom_id", ""))
        parse_errors: list[str] = []
        validation_errors: list[str] = []
        question_validation_errors: dict[str, list[str]] = {}
        raw_text = ""
        json_text = ""
        explanation = ""
        parsed_answers: dict[str, dict[str, float]] = {}
        expected_question_count = 0
        valid_question_count = 0

        manifest_row: Any | None = None
        if custom_id not in manifest.index:
            parse_errors.append("custom_id missing from request manifest")
        else:
            manifest_row = manifest.loc[custom_id]
            response_schema = str(manifest_row["response_schema"])
            question_manifest = json.loads(str(manifest_row["question_manifest_json"]))
            expected_question_count = int(len(question_manifest))
            expected_question_total += expected_question_count

            try:
                raw_text = extract_text_from_response_record(record)
                json_text = extract_json_object_text(raw_text)
                payload = json.loads(json_text)
                normalized_payload, validation_errors, question_validation_errors = (
                    validate_batched_prediction_payload(
                        payload,
                        question_manifest=question_manifest,
                        response_schema=response_schema,
                    )
                )
                if normalized_payload is not None:
                    explanation = str(normalized_payload.get("explanation", "")).strip()
                    parsed_answers = {
                        str(question_id): {
                            str(label): float(probability)
                            for label, probability in answer_dist.items()
                        }
                        for question_id, answer_dist in (normalized_payload.get("answers") or {}).items()
                    }
                    valid_question_count = int(normalized_payload.get("valid_question_count", len(parsed_answers)))
            except Exception as exc:
                parse_errors.append(str(exc))

        valid_question_total += valid_question_count
        parse_success = (
            not parse_errors
            and not validation_errors
            and not question_validation_errors
            and expected_question_count > 0
            and valid_question_count == expected_question_count
        )
        if parse_success:
            full_valid_request_count += 1
        if valid_question_count > 0:
            request_with_any_valid_answers_count += 1

        parsed_rows.append(
            {
                "custom_id": custom_id,
                "parse_success": parse_success,
                "parse_errors": parse_errors,
                "validation_errors": validation_errors,
                "question_validation_errors": question_validation_errors,
                "text": raw_text,
                "json_text": json_text,
                "explanation": explanation,
                "parsed_answers": parsed_answers,
                "expected_question_count": expected_question_count,
                "valid_question_count": valid_question_count,
                "context_id": str(manifest_row["context_id"]) if manifest_row is not None else "",
                "simbench_split": str(manifest_row["simbench_split"]) if manifest_row is not None else "",
                "dataset_name": str(manifest_row["dataset_name"]) if manifest_row is not None else "",
                "response_schema": str(manifest_row["response_schema"]) if manifest_row is not None else "",
                "sample_index": int(manifest_row["sample_index"]) if manifest_row is not None else None,
                "twin_pid": str(manifest_row["twin_pid"]) if manifest_row is not None else "",
            }
        )

    write_jsonl(args.output_jsonl, parsed_rows)
    write_json(
        args.output_jsonl.with_name("parse_summary.json"),
        {
            "batch_output_jsonl": str(args.batch_output_jsonl),
            "request_manifest_csv": str(args.request_manifest_csv),
            "parsed_output_jsonl": str(args.output_jsonl),
            "num_raw_rows": len(raw_rows),
            "num_parsed_rows": len(parsed_rows),
            "parse_success_count": full_valid_request_count,
            "parse_success_rate": full_valid_request_count / float(len(parsed_rows)) if parsed_rows else 0.0,
            "request_with_any_valid_answers_count": request_with_any_valid_answers_count,
            "request_with_any_valid_answers_rate": (
                request_with_any_valid_answers_count / float(len(parsed_rows)) if parsed_rows else 0.0
            ),
            "expected_question_total": int(expected_question_total),
            "valid_question_total": int(valid_question_total),
            "valid_question_rate": (
                valid_question_total / float(expected_question_total) if expected_question_total else 0.0
            ),
        },
    )


if __name__ == "__main__":
    main()
