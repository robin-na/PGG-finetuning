from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GROUND_TRUTH = SCRIPT_DIR / "us16_ground_truth.json"


def _load_text(path: Path) -> str:
    return path.read_text()


def _extract_json_block(text: str) -> dict[str, object]:
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = fenced or re.findall(r"(\{.*\})", text, flags=re.DOTALL)
    if not candidates:
        raise ValueError("Could not find a JSON object in the prediction file.")
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("Found JSON-like text but could not parse a valid JSON object.")


def _rank_map(order: list[str]) -> dict[str, int]:
    return {dataset_name: index + 1 for index, dataset_name in enumerate(order)}


def _spearman(order_pred: list[str], order_true: list[str]) -> float:
    pred = _rank_map(order_pred)
    true = _rank_map(order_true)
    n = len(order_true)
    d_sq = sum((pred[name] - true[name]) ** 2 for name in order_true)
    return 1 - (6 * d_sq) / (n * (n**2 - 1))


def _pairwise_accuracy(order_pred: list[str], order_true: list[str]) -> float:
    pred = _rank_map(order_pred)
    true = _rank_map(order_true)
    names = order_true
    total = 0
    correct = 0
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            total += 1
            pred_prefers_left = pred[left] < pred[right]
            true_prefers_left = true[left] < true[right]
            if pred_prefers_left == true_prefers_left:
                correct += 1
    return correct / total if total else float("nan")


def _confusion(truth: dict[str, str], pred: dict[str, str]) -> dict[str, dict[str, int]]:
    labels = ["positive", "insignificant", "negative"]
    matrix = {gold: {guess: 0 for guess in labels} for gold in labels}
    for dataset_name, gold in truth.items():
        guess = pred[dataset_name]
        matrix[gold][guess] += 1
    return matrix


def _f1_metrics(truth: dict[str, str], pred: dict[str, str]) -> dict[str, object]:
    labels = ["positive", "insignificant", "negative"]
    matrix = _confusion(truth, pred)
    per_label = {}
    macro_f1 = 0.0
    correct = 0
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        correct += tp
        macro_f1 += f1
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "accuracy": correct / len(truth) if truth else float("nan"),
        "macro_f1": macro_f1 / len(labels),
        "per_label": per_label,
        "confusion_matrix": matrix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a task-ranking / 3-way transferability forecast against ground truth."
    )
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--ground-truth-file", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output-file", type=Path, default=None)
    args = parser.parse_args()

    prediction_payload = _extract_json_block(_load_text(args.prediction_file))
    ground_truth = json.loads(args.ground_truth_file.read_text())

    true_order = [row["dataset_name"] for row in ground_truth]
    true_labels = {row["dataset_name"]: row["true_label"] for row in ground_truth}
    predicted_order = prediction_payload["ranking_most_helpful_to_most_harmful"]
    if sorted(predicted_order) != sorted(true_order):
        raise ValueError("Predicted ranking must contain exactly the same task names as ground truth.")

    predicted_label_rows = prediction_payload["task_predictions"]
    predicted_labels = {row["dataset_name"]: row["predicted_label"] for row in predicted_label_rows}
    if sorted(predicted_labels) != sorted(true_labels):
        raise ValueError("Predicted task labels must contain exactly the same task names as ground truth.")

    invalid = [
        (dataset_name, label)
        for dataset_name, label in predicted_labels.items()
        if label not in {"positive", "negative", "insignificant"}
    ]
    if invalid:
        raise ValueError(f"Invalid predicted labels: {invalid}")

    results = {
        "n_tasks": len(true_order),
        "ranking_metrics": {
            "spearman_rho": _spearman(predicted_order, true_order),
            "pairwise_accuracy": _pairwise_accuracy(predicted_order, true_order),
        },
        "classification_metrics": _f1_metrics(true_labels, predicted_labels),
    }

    output_text = json.dumps(results, indent=2, ensure_ascii=True) + "\n"
    if args.output_file is not None:
        args.output_file.write_text(output_text)
    else:
        print(output_text, end="")


if __name__ == "__main__":
    main()
