from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    extract_json_object_text,
    extract_text_from_response_record,
    load_request_manifest_df,
    validate_prediction_payload,
    write_json,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse two-stage forecasting batch outputs into validated structured JSON."
    )
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--batch-output-jsonl", type=Path, default=None)
    parser.add_argument("--request-manifest-csv", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    args = parser.parse_args()

    if args.run_name:
        args.batch_output_jsonl = args.batch_output_jsonl or (
            args.forecasting_root / "batch_output" / f"{args.run_name}.jsonl"
        )
        args.request_manifest_csv = args.request_manifest_csv or (
            args.forecasting_root / "metadata" / args.run_name / "request_manifest.csv"
        )
        args.output_jsonl = args.output_jsonl or (
            args.forecasting_root / "metadata" / args.run_name / "parsed_output.jsonl"
        )

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
    parse_success_count = 0
    for record in raw_rows:
        custom_id = str(record.get("custom_id", ""))
        row_errors: list[str] = []
        validation_errors: list[str] = []
        parsed_target: dict[str, Any] | None = None
        raw_text = ""

        if custom_id not in manifest.index:
            row_errors.append("custom_id missing from request manifest")
        else:
            schema_type = str(manifest.loc[custom_id, "schema_type"])
            try:
                raw_text = extract_text_from_response_record(record)
                json_text = extract_json_object_text(raw_text)
                payload = json.loads(json_text)
                parsed_target, validation_errors = validate_prediction_payload(payload, schema_type)
            except Exception as exc:
                row_errors.append(str(exc))

        parse_success = not row_errors and not validation_errors and parsed_target is not None
        if parse_success:
            parse_success_count += 1

        parsed_rows.append(
            {
                "custom_id": custom_id,
                "parse_success": parse_success,
                "parse_errors": row_errors,
                "validation_errors": validation_errors,
                "text": raw_text,
                "parsed_target": parsed_target,
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
            "parse_success_count": parse_success_count,
            "parse_success_rate": parse_success_count / float(len(parsed_rows)) if parsed_rows else 0.0,
        },
    )


if __name__ == "__main__":
    main()
