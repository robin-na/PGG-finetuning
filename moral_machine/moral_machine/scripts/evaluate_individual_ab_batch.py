from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MM_ROOT = SCRIPT_DIR.parent
DEFAULT_RUN_NAME = (
    "mm_individual_demographic_complete_5_per_scenario_n_gt_10000_actual_order_seed0_"
    "gpt_5_mini_ab_no_logprobs"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate individual-level Moral Machine A/B batch outputs."
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--batch-output-jsonl", type=Path, default=None)
    parser.add_argument("--sample-manifest-csv", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_paths(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path, Path]:
    manifest_path = args.manifest_json or (MM_ROOT / "metadata" / args.run_name / "manifest.json")
    manifest = load_json(manifest_path.expanduser().resolve())
    output_jsonl = (
        args.batch_output_jsonl or Path(manifest["expected_batch_output_file"])
    ).expanduser().resolve()
    sample_manifest_csv = (
        args.sample_manifest_csv or Path(manifest["sample_manifest_file"])
    ).expanduser().resolve()
    results_dir = (
        args.results_dir
        or (MM_ROOT / "results" / str(manifest.get("run_name") or args.run_name))
    ).expanduser().resolve()
    return manifest, output_jsonl, sample_manifest_csv, results_dir


def read_batch_outputs(path: Path) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            custom_id = str(record.get("custom_id") or "")
            if custom_id:
                outputs[custom_id] = record
    return outputs


def extract_content(record: dict[str, Any] | None) -> tuple[str, str]:
    if record is None:
        return "", "custom_id missing from output"
    response = record.get("response") or {}
    if response.get("status_code") != 200:
        return "", f"status_code={response.get('status_code')}"
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return "", "missing choices"
    return str((choices[0].get("message") or {}).get("content") or ""), ""


def parse_label(record: dict[str, Any] | None) -> dict[str, Any]:
    content, error = extract_content(record)
    if error:
        return {
            "parse_success": False,
            "parse_error": error,
            "raw_content": content,
            "predicted_choice": "",
            "exact_format": False,
        }
    stripped = content.strip()
    if stripped in {"A", "B"}:
        return {
            "parse_success": True,
            "parse_error": "",
            "raw_content": content,
            "predicted_choice": stripped,
            "exact_format": True,
        }
    for char in stripped:
        if char in {"A", "B"}:
            return {
                "parse_success": True,
                "parse_error": "non-exact format; parsed first A/B character",
                "raw_content": content,
                "predicted_choice": char,
                "exact_format": False,
            }
    return {
        "parse_success": False,
        "parse_error": "no A/B label found",
        "raw_content": content,
        "predicted_choice": "",
        "exact_format": False,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["parse_success"]]
    failed = [row for row in rows if not row["parse_success"]]
    if not valid:
        return {"num_rows": len(rows), "num_valid": 0, "num_failed": len(failed)}

    correct = [bool(row["correct"]) for row in valid]
    gold_counts = Counter(row["gold_choice"] for row in rows)
    pred_counts = Counter(row["predicted_choice"] for row in valid)
    gold_majority = gold_counts.most_common(1)[0][0] if gold_counts else None
    majority_baseline = (
        gold_counts[gold_majority] / len(rows) if gold_majority is not None and rows else None
    )
    by_left_action: dict[str, dict[str, Any]] = {}
    for action in sorted({str(row["left_action"]) for row in valid}):
        subset = [row for row in valid if row["left_action"] == action]
        by_left_action[action] = {
            "n": len(subset),
            "accuracy": sum(bool(row["correct"]) for row in subset) / len(subset),
        }
    return {
        "num_rows": len(rows),
        "num_valid": len(valid),
        "num_failed": len(failed),
        "accuracy": sum(correct) / len(correct),
        "accuracy_denominator": "valid parsed outputs",
        "num_exact_format": sum(1 for row in valid if bool(row["exact_format"])),
        "gold_choice_counts": dict(sorted(gold_counts.items())),
        "predicted_choice_counts": dict(sorted(pred_counts.items())),
        "gold_majority_baseline_accuracy": majority_baseline,
        "accuracy_by_left_action": by_left_action,
        "failure_examples": failed[:5],
    }


def main() -> None:
    args = parse_args()
    manifest, output_jsonl, sample_manifest_csv, results_dir = resolve_paths(args)
    sample_rows = read_csv(sample_manifest_csv)
    output_by_id = read_batch_outputs(output_jsonl)

    eval_rows: list[dict[str, Any]] = []
    for sample_row in sample_rows:
        custom_id = sample_row["custom_id"]
        extracted = parse_label(output_by_id.get(custom_id))
        predicted = extracted["predicted_choice"]
        gold = sample_row["gold_choice"]
        eval_rows.append(
            {
                "custom_id": custom_id,
                "response_id": sample_row["response_id"],
                "scenario_hash": sample_row["scenario_hash"],
                "scenario_index": sample_row["scenario_index"],
                "sample_within_scenario": sample_row["sample_within_scenario"],
                "left_action": sample_row["left_action"],
                "right_action": sample_row["right_action"],
                "option_a_action": sample_row["option_a_action"],
                "option_b_action": sample_row["option_b_action"],
                "gold_choice": gold,
                "gold_action": sample_row["gold_action"],
                "predicted_choice": predicted,
                "predicted_action": (
                    sample_row["option_a_action"]
                    if predicted == "A"
                    else sample_row["option_b_action"]
                    if predicted == "B"
                    else ""
                ),
                "correct": bool(extracted["parse_success"] and predicted == gold),
                "scenario_type": sample_row["scenario_type"],
                "scenario_type_strict": sample_row["scenario_type_strict"],
                "attribute_level": sample_row["attribute_level"],
                "pedped": sample_row["pedped"],
                **extracted,
            }
        )

    summary = summarize(eval_rows)
    summary.update(
        {
            "run_name": manifest.get("run_name") or args.run_name,
            "model": manifest.get("model"),
            "batch_output_jsonl": str(output_jsonl),
            "sample_manifest_csv": str(sample_manifest_csv),
            "results_dir": str(results_dir),
        }
    )
    rows_path = results_dir / "individual_ab_rows.csv"
    summary_path = results_dir / "individual_ab_summary.json"
    write_csv(rows_path, eval_rows)
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "rows_csv": str(rows_path),
                "summary_json": str(summary_path),
                "num_valid": summary["num_valid"],
                "accuracy": summary.get("accuracy"),
                "gold_majority_baseline_accuracy": summary.get(
                    "gold_majority_baseline_accuracy"
                ),
                "gold_choice_counts": summary.get("gold_choice_counts"),
                "predicted_choice_counts": summary.get("predicted_choice_counts"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
