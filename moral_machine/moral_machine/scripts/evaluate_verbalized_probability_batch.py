from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MM_ROOT = SCRIPT_DIR.parent
DEFAULT_RUN_NAME = "mm_global_distribution_scenario_only_n_gt_10000_gpt_5_mini_verbalized_probs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Moral Machine scenario-level verbalized A/B probability outputs."
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--batch-output-jsonl", type=Path, default=None)
    parser.add_argument("--scenario-manifest-csv", type=Path, default=None)
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
    scenario_manifest_csv = (
        args.scenario_manifest_csv or Path(manifest["scenario_manifest_file"])
    ).expanduser().resolve()
    results_dir = (
        args.results_dir
        or (MM_ROOT / "results" / str(manifest.get("run_name") or args.run_name))
    ).expanduser().resolve()
    return manifest, output_jsonl, scenario_manifest_csv, results_dir


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


def extract_json_object_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    raise ValueError("no JSON object found")


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


def parse_distribution(record: dict[str, Any] | None) -> dict[str, Any]:
    content, error = extract_content(record)
    if error:
        return {
            "parse_success": False,
            "parse_error": error,
            "raw_content": content,
            "raw_a_percent": "",
            "raw_b_percent": "",
            "raw_percent_sum": "",
            "contract_sum_100": False,
            "model_a_prob": "",
            "model_b_prob": "",
        }
    try:
        payload = json.loads(extract_json_object_text(content))
        raw_a = float(payload["A"])
        raw_b = float(payload["B"])
    except Exception as exc:
        return {
            "parse_success": False,
            "parse_error": str(exc),
            "raw_content": content,
            "raw_a_percent": "",
            "raw_b_percent": "",
            "raw_percent_sum": "",
            "contract_sum_100": False,
            "model_a_prob": "",
            "model_b_prob": "",
        }
    raw_sum = raw_a + raw_b
    if not math.isfinite(raw_sum) or raw_sum <= 0:
        return {
            "parse_success": False,
            "parse_error": "A+B is nonpositive or nonfinite",
            "raw_content": content,
            "raw_a_percent": raw_a,
            "raw_b_percent": raw_b,
            "raw_percent_sum": raw_sum,
            "contract_sum_100": False,
            "model_a_prob": "",
            "model_b_prob": "",
        }
    return {
        "parse_success": True,
        "parse_error": "",
        "raw_content": content,
        "raw_a_percent": raw_a,
        "raw_b_percent": raw_b,
        "raw_percent_sum": raw_sum,
        "contract_sum_100": abs(raw_sum - 100.0) <= 1e-9,
        "model_a_prob": raw_a / raw_sum,
        "model_b_prob": raw_b / raw_sum,
    }


def weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return float("nan")
    return sum(value * weight for value, weight in zip(values, weights)) / total


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["parse_success"]]
    failed = [row for row in rows if not row["parse_success"]]
    if not valid:
        return {"num_rows": len(rows), "num_valid": 0, "num_failed": len(failed)}

    tvds = [float(row["tvd"]) for row in valid]
    uniform_tvds = [float(row["uniform_tvd"]) for row in valid]
    weights = [float(row["global_count"]) for row in valid]
    mean_tvd = sum(tvds) / len(tvds)
    mean_uniform_tvd = sum(uniform_tvds) / len(uniform_tvds)
    weighted_tvd = weighted_mean(tvds, weights)
    weighted_uniform_tvd = weighted_mean(uniform_tvds, weights)
    return {
        "num_rows": len(rows),
        "num_valid": len(valid),
        "num_failed": len(failed),
        "num_contract_sum_violations": sum(
            1 for row in valid if not bool(row["contract_sum_100"])
        ),
        "mean_tvd": mean_tvd,
        "mean_uniform_tvd": mean_uniform_tvd,
        "one_minus_tvd_over_uniform_tvd": (
            1.0 - mean_tvd / mean_uniform_tvd if mean_uniform_tvd else None
        ),
        "weighted_mean_tvd": weighted_tvd,
        "weighted_mean_uniform_tvd": weighted_uniform_tvd,
        "weighted_one_minus_tvd_over_uniform_tvd": (
            1.0 - weighted_tvd / weighted_uniform_tvd if weighted_uniform_tvd else None
        ),
        "failure_examples": failed[:5],
    }


def main() -> None:
    args = parse_args()
    manifest, output_jsonl, scenario_manifest_csv, results_dir = resolve_paths(args)
    scenario_rows = read_csv(scenario_manifest_csv)
    output_by_id = read_batch_outputs(output_jsonl)

    eval_rows: list[dict[str, Any]] = []
    for scenario_row in scenario_rows:
        custom_id = scenario_row["custom_id"]
        extracted = parse_distribution(output_by_id.get(custom_id))
        observed_a = float(scenario_row["observed_a_share"])
        row: dict[str, Any] = {
            "custom_id": custom_id,
            "scenario_hash": scenario_row["scenario_hash"],
            "global_count": int(float(scenario_row["global_count"])),
            "option_a_action": scenario_row["option_a_action"],
            "option_b_action": scenario_row["option_b_action"],
            "observed_a_share": observed_a,
            "observed_b_share": float(scenario_row["observed_b_share"]),
            "scenario_type": scenario_row["scenario_type"],
            "scenario_type_strict": scenario_row["scenario_type_strict"],
            "attribute_level": scenario_row["attribute_level"],
            "pedped": scenario_row["pedped"],
            **extracted,
        }
        if extracted["parse_success"]:
            model_a = float(extracted["model_a_prob"])
            row["tvd"] = abs(model_a - observed_a)
            row["uniform_tvd"] = abs(0.5 - observed_a)
        else:
            row["tvd"] = ""
            row["uniform_tvd"] = abs(0.5 - observed_a)
        eval_rows.append(row)

    summary = summarize(eval_rows)
    summary.update(
        {
            "run_name": manifest.get("run_name") or args.run_name,
            "model": manifest.get("model"),
            "batch_output_jsonl": str(output_jsonl),
            "scenario_manifest_csv": str(scenario_manifest_csv),
            "results_dir": str(results_dir),
            "metric_notes": {
                "tvd": "For binary A/B distributions, TVD equals abs(model_a_share - observed_a_share).",
                "uniform_tvd": "abs(0.5 - observed_a_share), the TVD of uniform guessing for that scenario.",
            },
        }
    )
    rows_path = results_dir / "distribution_alignment_rows.csv"
    summary_path = results_dir / "distribution_alignment_summary.json"
    write_csv(rows_path, eval_rows)
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "rows_csv": str(rows_path),
                "summary_json": str(summary_path),
                "num_valid": summary["num_valid"],
                "mean_tvd": summary.get("mean_tvd"),
                "mean_uniform_tvd": summary.get("mean_uniform_tvd"),
                "one_minus_tvd_over_uniform_tvd": summary.get(
                    "one_minus_tvd_over_uniform_tvd"
                ),
                "weighted_mean_tvd": summary.get("weighted_mean_tvd"),
                "weighted_mean_uniform_tvd": summary.get("weighted_mean_uniform_tvd"),
                "weighted_one_minus_tvd_over_uniform_tvd": summary.get(
                    "weighted_one_minus_tvd_over_uniform_tvd"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
