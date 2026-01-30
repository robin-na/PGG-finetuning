from __future__ import annotations

from pathlib import Path
from typing import List

from .comparison import AlignmentRow, ConfigMetricSummary


def _simplify_label(config_name: str) -> str:
    if config_name.startswith("VALIDATION_"):
        return config_name.replace("VALIDATION_", "")
    return config_name


def plot_noise_ceiling(output_dir: Path, alignment_rows: List[AlignmentRow]) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: List[Path] = []
    metrics = sorted({row.metric for row in alignment_rows})
    for metric in metrics:
        rows = [row for row in alignment_rows if row.metric == metric]
        if not rows:
            continue
        labels = [_simplify_label(row.config) for row in rows]
        sim_values = [row.sim_mean for row in rows]
        human_values = [row.human_mean for row in rows]
        noise = [row.human_sem for row in rows]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = list(range(len(labels)))
        ax.errorbar(
            x,
            human_values,
            yerr=noise,
            fmt="o",
            linestyle="none",
            capsize=3,
            label="Human mean (±1 SEM)",
        )
        ax.scatter(x, sim_values, marker="x", label="Simulation mean")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"{metric}: simulation vs human with noise ceiling")
        ax.legend()
        ax.set_ylabel(metric)
        fig.tight_layout()

        output_path = output_dir / f"noise_ceiling_{metric}.png"
        fig.savefig(output_path)
        plt.close(fig)
        paths.append(output_path)
    return paths


def plot_config_metric_errors(
    output_dir: Path, rows: List[ConfigMetricSummary]
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: List[Path] = []
    if not rows:
        return paths

    rows = sorted(rows, key=lambda row: row.config)
    labels = [_simplify_label(row.config) for row in rows]
    metrics = [
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("r2", "R²"),
    ]
    x = list(range(len(labels)))
    width = 0.4
    for key, title in metrics:
        values = [getattr(row, key) for row in rows]
        noise = [row.noise_ceiling for row in rows]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([i - width / 2 for i in x], values, width=width, label=title)
        ax.bar([i + width / 2 for i in x], noise, width=width, label="Noise ceiling")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"{title} by configuration")
        ax.legend()
        fig.tight_layout()

        output_path = output_dir / f"config_{key}_noise_ceiling.png"
        fig.savefig(output_path)
        plt.close(fig)
        paths.append(output_path)
    return paths
