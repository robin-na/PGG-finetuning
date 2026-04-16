from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


TASK_ORDER = ["trust", "ultimatum", "dictator"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_ci95(values: list[float]) -> tuple[float, float, float]:
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return float("nan"), float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, mean, mean
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    se = math.sqrt(variance / len(values))
    margin = 1.96 * se
    return mean, mean - margin, mean + margin


def question_sort_key(qid: str) -> tuple[int, int]:
    task_rank = 99
    qnum = int(qid.replace("QID", ""))
    if 117 <= qnum <= 122:
        task_rank = 0
    elif 224 <= qnum <= 230:
        task_rank = 1
    elif qnum == 231:
        task_rank = 2
    return task_rank, qnum


def build_task_rows(
    retrieval_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    comparison_rows: list[dict[str, str]],
    trait_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    rows_by_task: dict[str, dict[str, object]] = {}

    for row in retrieval_rows:
        task = row["task_name"]
        rows_by_task.setdefault(task, {"task_name": task})
        rows_by_task[task]["retrieval_accuracy"] = float(row["mean_task_accuracy"])
        rows_by_task[task]["retrieval_ci95_low"] = float(row["ci95_low"])
        rows_by_task[task]["retrieval_ci95_high"] = float(row["ci95_high"])

    for row in baseline_rows:
        task = row["task_name"]
        rows_by_task.setdefault(task, {"task_name": task})
        rows_by_task[task]["baseline_accuracy"] = float(row["mean_task_accuracy"])
        rows_by_task[task]["baseline_ci95_low"] = float(row["ci95_low"])
        rows_by_task[task]["baseline_ci95_high"] = float(row["ci95_high"])

    for row in comparison_rows:
        task = row["task_name"]
        if row["baseline_name"] != "random_uniform_expected":
            continue
        rows_by_task.setdefault(task, {"task_name": task})
        key = f"{row['baseline_name']}_accuracy"
        rows_by_task[task][key] = float(row["mean_task_accuracy"])
        rows_by_task[task][f"{row['baseline_name']}_ci95_low"] = float(row["ci95_low"])
        rows_by_task[task][f"{row['baseline_name']}_ci95_high"] = float(row["ci95_high"])

    for row in trait_rows:
        task = row["task_name"]
        rows_by_task.setdefault(task, {"task_name": task})
        rows_by_task[task]["trait_index_heuristic_accuracy"] = float(row["mean_task_accuracy"])
        rows_by_task[task]["trait_index_heuristic_ci95_low"] = float(row["ci95_low"])
        rows_by_task[task]["trait_index_heuristic_ci95_high"] = float(row["ci95_high"])

    merged = []
    for task in TASK_ORDER:
        row = rows_by_task.get(task)
        if row:
            row["retrieval_minus_baseline"] = row["retrieval_accuracy"] - row["baseline_accuracy"]
            merged.append(row)
    return merged


def build_question_rows(
    retrieval_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    comparison_rows: list[dict[str, str]],
    trait_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    rows_by_qid: dict[str, dict[str, object]] = {}
    retrieval_accs: dict[str, list[float]] = {}
    baseline_accs: dict[str, list[float]] = {}
    comparison_accs: dict[tuple[str, str], list[float]] = {}
    trait_accs: dict[str, list[float]] = {}

    for row in retrieval_rows:
        qid = row["question_id"]
        rows_by_qid.setdefault(qid, {"task_name": row["task_name"], "question_id": qid})
        acc = float(row["normalized_accuracy"])
        retrieval_accs.setdefault(qid, []).append(acc)

    for row in baseline_rows:
        qid = row["question_id"]
        rows_by_qid.setdefault(qid, {"task_name": row["task_name"], "question_id": qid})
        acc = float(row["normalized_accuracy"])
        baseline_accs.setdefault(qid, []).append(acc)

    for row in comparison_rows:
        if row["baseline_name"] != "random_uniform_expected":
            continue
        qid = row["question_id"]
        rows_by_qid.setdefault(qid, {"task_name": row["task_name"], "question_id": qid})
        acc = float(row["accuracy"])
        comparison_accs.setdefault((row["baseline_name"], qid), []).append(acc)

    for row in trait_rows:
        qid = row["question_id"]
        rows_by_qid.setdefault(qid, {"task_name": row["task_name"], "question_id": qid})
        acc = float(row["accuracy"])
        trait_accs.setdefault(qid, []).append(acc)

    for qid, row in rows_by_qid.items():
        mean, low, high = mean_ci95(retrieval_accs[qid])
        row["retrieval_accuracy"] = mean
        row["retrieval_ci95_low"] = low
        row["retrieval_ci95_high"] = high

        mean, low, high = mean_ci95(baseline_accs[qid])
        row["baseline_accuracy"] = mean
        row["baseline_ci95_low"] = low
        row["baseline_ci95_high"] = high

        for baseline_name in ["random_uniform_expected"]:
            mean, low, high = mean_ci95(comparison_accs[(baseline_name, qid)])
            row[f"{baseline_name}_accuracy"] = mean
            row[f"{baseline_name}_ci95_low"] = low
            row[f"{baseline_name}_ci95_high"] = high
        mean, low, high = mean_ci95(trait_accs[qid])
        row["trait_index_heuristic_accuracy"] = mean
        row["trait_index_heuristic_ci95_low"] = low
        row["trait_index_heuristic_ci95_high"] = high

    merged = []
    for qid in sorted(rows_by_qid, key=question_sort_key):
        row = rows_by_qid[qid]
        row["retrieval_minus_baseline"] = row["retrieval_accuracy"] - row["baseline_accuracy"]
        merged.append(row)
    return merged


def plot_task_comparison(task_rows: list[dict[str, object]], output_path: Path) -> None:
    labels = [str(row["task_name"]) for row in task_rows]
    x = list(range(len(labels)))
    width = 0.2

    retrieval = [float(row["retrieval_accuracy"]) for row in task_rows]
    baseline = [float(row["baseline_accuracy"]) for row in task_rows]
    random_uniform = [float(row["random_uniform_expected_accuracy"]) for row in task_rows]
    trait_heuristic = [float(row["trait_index_heuristic_accuracy"]) for row in task_rows]
    retrieval_err = [
        [
            float(row["retrieval_accuracy"]) - float(row["retrieval_ci95_low"]),
            float(row["retrieval_ci95_high"]) - float(row["retrieval_accuracy"]),
        ]
        for row in task_rows
    ]
    baseline_err = [
        [
            float(row["baseline_accuracy"]) - float(row["baseline_ci95_low"]),
            float(row["baseline_ci95_high"]) - float(row["baseline_accuracy"]),
        ]
        for row in task_rows
    ]
    random_err = [
        [
            float(row["random_uniform_expected_accuracy"]) - float(row["random_uniform_expected_ci95_low"]),
            float(row["random_uniform_expected_ci95_high"]) - float(row["random_uniform_expected_accuracy"]),
        ]
        for row in task_rows
    ]
    trait_err = [
        [
            float(row["trait_index_heuristic_accuracy"]) - float(row["trait_index_heuristic_ci95_low"]),
            float(row["trait_index_heuristic_ci95_high"]) - float(row["trait_index_heuristic_accuracy"]),
        ]
        for row in task_rows
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        [i - 1.5 * width for i in x],
        baseline,
        width,
        label="No retrieval",
        yerr=list(zip(*baseline_err)),
        capsize=3,
        error_kw={"elinewidth": 1},
    )
    ax.bar(
        [i - 0.5 * width for i in x],
        retrieval,
        width,
        label="Retrieval",
        yerr=list(zip(*retrieval_err)),
        capsize=3,
        error_kw={"elinewidth": 1},
    )
    ax.bar(
        [i + 0.5 * width for i in x],
        random_uniform,
        width,
        label="Random",
        yerr=list(zip(*random_err)),
        capsize=3,
        error_kw={"elinewidth": 1},
    )
    ax.bar(
        [i + 1.5 * width for i in x],
        trait_heuristic,
        width,
        label="Trait heuristic",
        yerr=list(zip(*trait_err)),
        capsize=3,
        error_kw={"elinewidth": 1},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized accuracy")
    ax.set_title("Task-level accuracy: retrieval vs matched baselines")
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_question_comparison(question_rows: list[dict[str, object]], output_path: Path) -> None:
    labels = [str(row["question_id"]) for row in question_rows]
    x = list(range(len(labels)))
    width = 0.2

    retrieval = [float(row["retrieval_accuracy"]) for row in question_rows]
    baseline = [float(row["baseline_accuracy"]) for row in question_rows]
    random_uniform = [float(row["random_uniform_expected_accuracy"]) for row in question_rows]
    trait_heuristic = [float(row["trait_index_heuristic_accuracy"]) for row in question_rows]
    retrieval_err = [
        [
            float(row["retrieval_accuracy"]) - float(row["retrieval_ci95_low"]),
            float(row["retrieval_ci95_high"]) - float(row["retrieval_accuracy"]),
        ]
        for row in question_rows
    ]
    baseline_err = [
        [
            float(row["baseline_accuracy"]) - float(row["baseline_ci95_low"]),
            float(row["baseline_ci95_high"]) - float(row["baseline_accuracy"]),
        ]
        for row in question_rows
    ]
    random_err = [
        [
            float(row["random_uniform_expected_accuracy"]) - float(row["random_uniform_expected_ci95_low"]),
            float(row["random_uniform_expected_ci95_high"]) - float(row["random_uniform_expected_accuracy"]),
        ]
        for row in question_rows
    ]
    trait_err = [
        [
            float(row["trait_index_heuristic_accuracy"]) - float(row["trait_index_heuristic_ci95_low"]),
            float(row["trait_index_heuristic_ci95_high"]) - float(row["trait_index_heuristic_accuracy"]),
        ]
        for row in question_rows
    ]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(
        [i - 1.5 * width for i in x],
        baseline,
        width,
        label="No retrieval",
        yerr=list(zip(*baseline_err)),
        capsize=2,
        error_kw={"elinewidth": 0.8},
    )
    ax.bar(
        [i - 0.5 * width for i in x],
        retrieval,
        width,
        label="Retrieval",
        yerr=list(zip(*retrieval_err)),
        capsize=2,
        error_kw={"elinewidth": 0.8},
    )
    ax.bar(
        [i + 0.5 * width for i in x],
        random_uniform,
        width,
        label="Random",
        yerr=list(zip(*random_err)),
        capsize=2,
        error_kw={"elinewidth": 0.8},
    )
    ax.bar(
        [i + 1.5 * width for i in x],
        trait_heuristic,
        width,
        label="Trait heuristic",
        yerr=list(zip(*trait_err)),
        capsize=2,
        error_kw={"elinewidth": 0.8},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized accuracy")
    ax.set_title("QID-level accuracy: retrieval vs matched baselines")
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot matched joint-social run comparisons.")
    parser.add_argument("--retrieval-eval-dir", type=Path, required=True)
    parser.add_argument("--baseline-eval-dir", type=Path, required=True)
    parser.add_argument("--comparison-baselines-dir", type=Path, required=True)
    parser.add_argument("--trait-baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_task_rows = read_csv(args.retrieval_eval_dir / "task_summary.csv")
    baseline_task_rows = read_csv(args.baseline_eval_dir / "task_summary.csv")
    comparison_task_rows = read_csv(args.comparison_baselines_dir / "comparison_baselines_task_summary.csv")
    trait_task_rows = read_csv(args.trait_baseline_dir / "trait_index_heuristic_task_summary.csv")
    task_rows = build_task_rows(retrieval_task_rows, baseline_task_rows, comparison_task_rows, trait_task_rows)

    retrieval_question_rows = read_jsonl(args.retrieval_eval_dir / "parsed_predictions.jsonl")
    baseline_question_rows = read_jsonl(args.baseline_eval_dir / "parsed_predictions.jsonl")
    comparison_question_rows = read_csv(args.comparison_baselines_dir / "comparison_baselines_raw.csv")
    trait_question_rows = read_csv(args.trait_baseline_dir / "trait_index_heuristic_raw.csv")
    question_rows = build_question_rows(
        retrieval_question_rows,
        baseline_question_rows,
        comparison_question_rows,
        trait_question_rows,
    )

    write_csv(
        args.output_dir / "task_comparison.csv",
        task_rows,
        [
            "task_name",
            "baseline_accuracy",
            "baseline_ci95_low",
            "baseline_ci95_high",
            "retrieval_accuracy",
            "retrieval_ci95_low",
            "retrieval_ci95_high",
            "retrieval_minus_baseline",
            "random_uniform_expected_accuracy",
            "random_uniform_expected_ci95_low",
            "random_uniform_expected_ci95_high",
            "trait_index_heuristic_accuracy",
            "trait_index_heuristic_ci95_low",
            "trait_index_heuristic_ci95_high",
        ],
    )
    write_csv(
        args.output_dir / "qid_comparison.csv",
        question_rows,
        [
            "task_name",
            "question_id",
            "baseline_accuracy",
            "baseline_ci95_low",
            "baseline_ci95_high",
            "retrieval_accuracy",
            "retrieval_ci95_low",
            "retrieval_ci95_high",
            "retrieval_minus_baseline",
            "random_uniform_expected_accuracy",
            "random_uniform_expected_ci95_low",
            "random_uniform_expected_ci95_high",
            "trait_index_heuristic_accuracy",
            "trait_index_heuristic_ci95_low",
            "trait_index_heuristic_ci95_high",
        ],
    )

    plot_task_comparison(task_rows, args.output_dir / "task_level_accuracy_retrieval_vs_baselines.png")
    plot_question_comparison(question_rows, args.output_dir / "qid_level_accuracy_retrieval_vs_baselines.png")

    print(f"Wrote {args.output_dir / 'task_comparison.csv'}")
    print(f"Wrote {args.output_dir / 'qid_comparison.csv'}")
    print(f"Wrote {args.output_dir / 'task_level_accuracy_retrieval_vs_baselines.png'}")
    print(f"Wrote {args.output_dir / 'qid_level_accuracy_retrieval_vs_baselines.png'}")


if __name__ == "__main__":
    main()
