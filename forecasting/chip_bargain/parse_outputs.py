from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    attempt_repair_prediction_payload,
    extract_json_object_text,
    extract_text_from_response_record,
    load_request_manifest_df,
    write_json,
    write_jsonl,
    validate_prediction_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse chip-bargain forecasting batch outputs into validated structured JSON."
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
        strict_validation_errors: list[str] = []
        repair_notes: list[str] = []
        parsed_target: dict[str, Any] | None = None
        raw_text = ""

        if custom_id not in manifest.index:
            row_errors.append("custom_id missing from request manifest")
        else:
            manifest_row = manifest.loc[custom_id]
            try:
                raw_text = extract_text_from_response_record(record)
                json_text = extract_json_object_text(raw_text)
                payload = json.loads(json_text)
                validation_inputs = {
                    "players": json.loads(str(manifest_row["players_json"])),
                    "turn_order": json.loads(str(manifest_row["turn_order_json"])),
                    "chip_definitions": json.loads(str(manifest_row["chip_definitions_json"])),
                    "initial_chip_holdings": json.loads(str(manifest_row["initial_chip_holdings_json"])),
                    "participant_chip_values": json.loads(str(manifest_row["participant_chip_values_json"])),
                }
                if "round_turn_orders_json" in manifest_row.index:
                    round_turn_orders_raw = manifest_row["round_turn_orders_json"]
                    if isinstance(round_turn_orders_raw, str) and round_turn_orders_raw.strip():
                        validation_inputs["round_turn_orders"] = json.loads(round_turn_orders_raw)
                parsed_target, validation_errors = validate_prediction_payload(
                    payload,
                    **validation_inputs,
                )
                if validation_errors:
                    strict_validation_errors = list(validation_errors)
                    repaired_payload, repair_notes = attempt_repair_prediction_payload(
                        payload,
                        **validation_inputs,
                    )
                    if repaired_payload is not None:
                        repaired_target, repaired_validation_errors = validate_prediction_payload(
                            repaired_payload,
                            **validation_inputs,
                        )
                        if not repaired_validation_errors and repaired_target is not None:
                            parsed_target = repaired_target
                            validation_errors = []
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
                "strict_validation_errors": strict_validation_errors,
                "repair_notes": repair_notes,
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
