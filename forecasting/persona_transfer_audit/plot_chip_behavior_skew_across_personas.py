"""Plot chip-bargaining behavioral skew across persona sources."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


THIS_DIR = Path(__file__).resolve().parent

RUNS = [
    {
        "label": "No persona",
        "run": "no_persona_to_chip_bargain_stratified_48_top3_gpt_5_mini_seed_2",
        "significance_file": "chip_no_persona_significance_checks.csv",
    },
    {
        "label": "Demographic\nsurveys",
        "run": "argyle_anes2016_backstory_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2",
        "significance_file": "chip_significance_checks.csv",
    },
    {
        "label": "Twin-2K-500",
        "run": "twin_direct_summary_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2",
        "significance_file": "chip_significance_checks.csv",
    },
    {
        "label": "Synthetic\n(Nemotron)",
        "run": "nemotron_raw_fields_adult_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2",
        "significance_file": "chip_significance_checks.csv",
    },
    {
        "label": "Task-adaptive\n(Concordia)",
        "run": "concordia_chip_bargain_game_grounded_alphaevolve_5_compact_to_chip_bargain_stratified_32x48_top3_gpt_5_mini",
        "significance_file": "chip_significance_checks.csv",
    },
]

METRICS = [
    ("final_surplus", "Final\nsurplus"),
    ("final_welfare", "Final\nwelfare"),
    ("proposer_mean_net_surplus", "Proposer\nnet surplus"),
    ("proposer_acceptance_rate", "Proposal\naccepted"),
    ("proposer_mean_trade_ratio", "Trade ratio\noffered"),
    ("response_acceptance_rate", "Responder\nacceptance"),
    ("response_mean_net_surplus_if_accepted", "Responder surplus\nif accepted"),
    ("received_trade_rate", "Received\ntrade"),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _weighted_sd(rows: list[dict[str, str]], metric: str, weight_column: str) -> float:
    values: list[tuple[float, float]] = []
    for row in rows:
        value = float(row.get(metric, "nan"))
        weight = float(row.get(weight_column, 0.0))
        if not math.isnan(value) and weight > 0:
            values.append((value, weight))
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return float("nan")
    mean = sum(value * weight for value, weight in values) / total_weight
    variance = sum(weight * (value - mean) ** 2 for value, weight in values) / total_weight
    return math.sqrt(max(variance, 0.0))


def _collect_rows(metadata_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in RUNS:
        metadata_dir = metadata_root / run["run"]
        significance_rows = {
            row["metric"]: row
            for row in _read_csv(metadata_dir / str(run["significance_file"]))
        }
        candidate_rows = _read_csv(metadata_dir / "chip_candidate_uniform_behavior_long.csv")
        for metric, metric_label in METRICS:
            row = significance_rows[metric]
            sd = _weighted_sd(candidate_rows, metric, "candidate_uniform_weight")
            mean_diff = float(row["mean_diff"])
            record_low = float(row["record_low"])
            record_high = float(row["record_high"])
            rows.append(
                {
                    "persona_source": run["label"].replace("\n", " "),
                    "run": run["run"],
                    "metric": metric,
                    "metric_label": metric_label.replace("\n", " "),
                    "mean_diff": mean_diff,
                    "record_low": record_low,
                    "record_high": record_high,
                    "candidate_uniform_sd": sd,
                    "standardized_diff": mean_diff / sd if sd > 0 else float("nan"),
                    "standardized_record_low": record_low / sd if sd > 0 else float("nan"),
                    "standardized_record_high": record_high / sd if sd > 0 else float("nan"),
                    "record_ci_excludes_zero": (record_low > 0 and record_high > 0)
                    or (record_low < 0 and record_high < 0),
                }
            )
    return rows


def _plot(rows: list[dict[str, object]], output_dir: Path, output_stem: str) -> None:
    label_to_index = {str(run["label"]).replace("\n", " "): index for index, run in enumerate(RUNS)}
    metric_to_index = {metric: index for index, (metric, _) in enumerate(METRICS)}
    matrix = [
        [float("nan") for _ in RUNS]
        for _ in METRICS
    ]
    ci_marker = [
        [False for _ in RUNS]
        for _ in METRICS
    ]
    for row in rows:
        source = str(row["persona_source"])
        y = metric_to_index[str(row["metric"])]
        x = label_to_index[source]
        matrix[y][x] = float(row["standardized_diff"])
        ci_marker[y][x] = str(row["record_ci_excludes_zero"]).lower() == "true"

    finite_values = [abs(value) for series in matrix for value in series if not math.isnan(value)]
    vmax = max(0.75, min(1.25, max(finite_values) * 1.05 if finite_values else 1.0))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    fig, ax = plt.subplots(figsize=(7.8, 5.1))
    image = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(len(RUNS)))
    ax.set_xticklabels([run["label"] for run in RUNS], rotation=0, ha="center")
    ax.set_yticks(range(len(METRICS)))
    ax.set_yticklabels([label.replace("\n", " ") for _, label in METRICS])
    ax.set_title("Bargaining game behavioral skew across persona sources", pad=12)
    ax.set_xlabel("Persona source")
    ax.set_ylabel("Behavioral target")

    for y in range(len(METRICS)):
        for x in range(len(RUNS)):
            value = matrix[y][x]
            if math.isnan(value):
                continue
            text_color = "white" if abs(value) > vmax * 0.55 else "#202020"
            ax.text(x, y, f"{value:+.2f}", ha="center", va="center", fontsize=8, color=text_color)
            if ci_marker[y][x]:
                ax.scatter(
                    x + 0.38,
                    y - 0.34,
                    marker="o",
                    s=16,
                    facecolor="#202020",
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=3,
                )

    ax.set_xticks([i - 0.5 for i in range(1, len(RUNS))], minor=True)
    ax.set_yticks([i - 0.5 for i in range(1, len(METRICS))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Matched-minus-uniform difference (SD units)")

    fig.tight_layout(rect=[0, 0, 0.98, 1])
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"{output_stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    metadata_root = args.metadata_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rows = _collect_rows(metadata_root)
    _write_csv(output_dir / f"{args.output_stem}_source_data.csv", rows)
    _plot(rows, output_dir, args.output_stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=THIS_DIR / "metadata",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_DIR / "figures",
    )
    parser.add_argument("--output-stem", default="figure_chip_behavior_skew_across_personas")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
