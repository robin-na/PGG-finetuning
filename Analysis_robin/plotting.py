from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from .comparison import AlignmentRow, ConfigMetricSummary


def _simplify_label(config_name: str) -> str:
    if config_name.startswith("VALIDATION_"):
        return config_name.replace("VALIDATION_", "")
    return config_name


def _metric_titles() -> Dict[str, str]:
    return {
        "mean_contribution_rate": "Contribution rate",
        "normalized_efficiency": "Normalized efficiency",
        "punishment_rate": "Punishment rate",
        "reward_rate": "Reward rate",
        "mean_payoff": "Payoff",
    }


def plot_config_metric_rmse(
    output_dir: Path, rows: List[ConfigMetricSummary]
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: List[Path] = []
    if not rows:
        return paths

    metric_titles = _metric_titles()
    metrics = [m for m in metric_titles if any(row.metric == m for row in rows)]
    if not metrics:
        return paths

    rows_by_metric: Dict[str, List[ConfigMetricSummary]] = {
        metric: sorted(
            [row for row in rows if row.metric == metric], key=lambda row: row.config
        )
        for metric in metrics
    }

    n_cols = 2
    n_rows = (len(metrics) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), squeeze=False)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // n_cols][idx % n_cols]
        metric_rows = rows_by_metric[metric]
        labels = [_simplify_label(row.config) for row in metric_rows]
        rmse_values = [row.rmse for row in metric_rows]
        noise = [row.noise_ceiling for row in metric_rows]
        x = list(range(len(labels)))
        width = 0.35
        ax.bar(x, rmse_values, width=width, label="RMSE")
        ax.bar(
            [i + width for i in x],
            noise,
            width=width,
            label="Noise ceiling (human std)",
        )
        ax.set_xticks([i + width / 2 for i in x])
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("RMSE")
        ax.set_title(metric_titles.get(metric, metric))
        ax.legend()

    for idx in range(len(metrics), n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols][idx % n_cols])

    fig.tight_layout()
    output_path = output_dir / "config_rmse_by_metric.png"
    fig.savefig(output_path)
    plt.close(fig)
    paths.append(output_path)
    return paths


def _bootstrap_rmse(
    rmse_values: List[float], n_boot: int = 2000, seed: int = 7
) -> Tuple[float, float]:
    if not rmse_values:
        return 0.0, 0.0
    import random

    rng = random.Random(seed)
    n = len(rmse_values)
    squared = [value * value for value in rmse_values]
    bootstrap: List[float] = []
    for _ in range(n_boot):
        sample = [squared[rng.randrange(n)] for _ in range(n)]
        bootstrap.append((sum(sample) / n) ** 0.5)
    mean_rmse = (sum(squared) / n) ** 0.5
    mean_boot = sum(bootstrap) / n_boot
    variance = sum((value - mean_boot) ** 2 for value in bootstrap) / n_boot
    return mean_rmse, variance**0.5


def plot_aggregate_metric_rmse(
    output_dir: Path, rows: List[ConfigMetricSummary]
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: List[Path] = []
    if not rows:
        return paths

    metric_titles = _metric_titles()
    grouped: Dict[str, List[ConfigMetricSummary]] = {}
    for row in rows:
        if row.metric not in metric_titles:
            continue
        grouped.setdefault(row.metric, []).append(row)

    metrics = [m for m in metric_titles if m in grouped]
    if not metrics:
        return paths

    labels = [metric_titles[m] for m in metrics]
    rmse_values: List[float] = []
    rmse_errors: List[float] = []
    noise_values: List[float] = []

    for metric in metrics:
        metric_rows = grouped[metric]
        config_rmse = [row.rmse for row in metric_rows]
        aggregate_rmse, rmse_err = _bootstrap_rmse(config_rmse)
        rmse_values.append(aggregate_rmse)
        rmse_errors.append(rmse_err)
        noise_rms = (
            sum(row.noise_ceiling ** 2 for row in metric_rows) / len(metric_rows)
        ) ** 0.5
        noise_values.append(noise_rms)

    x = list(range(len(metrics)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, rmse_values, width=width, yerr=rmse_errors, capsize=4, label="RMSE")
    ax.bar(
        [i + width for i in x],
        noise_values,
        width=width,
        label="Noise ceiling (human std)",
    )
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Aggregate RMSE across configurations")
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "aggregate_rmse_by_metric.png"
    fig.savefig(output_path)
    plt.close(fig)
    paths.append(output_path)
    return paths


def plot_metric_means_by_config(
    output_dir: Path, alignment_rows: List[AlignmentRow]
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: List[Path] = []
    if not alignment_rows:
        return paths

    metric_titles = _metric_titles()
    metrics = [m for m in metric_titles if any(row.metric == m for row in alignment_rows)]
    if not metrics:
        return paths

    rows_by_metric = {
        metric: sorted(
            [row for row in alignment_rows if row.metric == metric],
            key=lambda row: row.config,
        )
        for metric in metrics
    }

    n_cols = 2
    n_rows = (len(metrics) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), squeeze=False)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // n_cols][idx % n_cols]
        metric_rows = rows_by_metric[metric]
        labels = [_simplify_label(row.config) for row in metric_rows]
        sim_means = [row.sim_mean for row in metric_rows]
        human_means = [row.human_mean for row in metric_rows]
        human_std = [row.human_std for row in metric_rows]
        x = list(range(len(labels)))
        width = 0.35
        ax.bar(x, sim_means, width=width, label="Simulation mean")
        ax.bar(
            [i + width for i in x],
            human_means,
            width=width,
            yerr=human_std,
            capsize=3,
            label="Human mean (±1 std)",
        )
        ax.set_xticks([i + width / 2 for i in x])
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Mean value")
        ax.set_title(metric_titles.get(metric, metric))
        ax.legend()

    for idx in range(len(metrics), n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols][idx % n_cols])

    fig.tight_layout()
    output_path = output_dir / "metric_means_by_config.png"
    fig.savefig(output_path)
    plt.close(fig)
    paths.append(output_path)
    return paths


def plot_aggregate_metric_means(
    output_dir: Path, alignment_rows: List[AlignmentRow]
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: List[Path] = []
    if not alignment_rows:
        return paths

    metric_titles = _metric_titles()
    metrics = [m for m in metric_titles if any(row.metric == m for row in alignment_rows)]
    if not metrics:
        return paths

    grouped: Dict[str, List[AlignmentRow]] = {}
    for row in alignment_rows:
        if row.metric in metric_titles:
            grouped.setdefault(row.metric, []).append(row)

    sim_means: List[float] = []
    human_means: List[float] = []
    sim_std: List[float] = []
    human_std: List[float] = []
    labels: List[str] = []

    for metric in metrics:
        rows = grouped.get(metric, [])
        if not rows:
            continue
        labels.append(metric_titles.get(metric, metric))
        sim_values = [row.sim_mean for row in rows]
        human_values = [row.human_mean for row in rows]
        sim_mean = sum(sim_values) / len(sim_values)
        human_mean = sum(human_values) / len(human_values)
        sim_means.append(sim_mean)
        human_means.append(human_mean)
        sim_var = (
            sum((value - sim_mean) ** 2 for value in sim_values) / len(sim_values)
            if sim_values
            else 0.0
        )
        human_var = (
            sum((value - human_mean) ** 2 for value in human_values) / len(human_values)
            if human_values
            else 0.0
        )
        sim_std.append(sim_var**0.5)
        human_std.append(human_var**0.5)

    x = list(range(len(labels)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, sim_means, width=width, yerr=sim_std, capsize=4, label="Simulation mean")
    ax.bar(
        [i + width for i in x],
        human_means,
        width=width,
        yerr=human_std,
        capsize=4,
        label="Human mean",
    )
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean value")
    ax.set_title("Aggregate metric means across configurations")
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / "metric_means_aggregate.png"
    fig.savefig(output_path)
    plt.close(fig)
    paths.append(output_path)
    return paths


def plot_metric_variance_by_config(
    output_dir: Path,
    variance_rows: List[Tuple[str, str, float, float]],
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: List[Path] = []
    if not variance_rows:
        return paths

    metric_titles = _metric_titles()
    metrics = [m for m in metric_titles if any(row[1] == m for row in variance_rows)]
    if not metrics:
        return paths

    rows_by_metric: Dict[str, List[Tuple[str, str, float, float]]] = {
        metric: sorted(
            [row for row in variance_rows if row[1] == metric], key=lambda row: row[0]
        )
        for metric in metrics
    }

    n_cols = 2
    n_rows = (len(metrics) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), squeeze=False)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // n_cols][idx % n_cols]
        metric_rows = rows_by_metric[metric]
        labels = [_simplify_label(row[0]) for row in metric_rows]
        sim_vars = [row[2] for row in metric_rows]
        human_vars = [row[3] for row in metric_rows]
        x = list(range(len(labels)))
        width = 0.35
        ax.bar(x, sim_vars, width=width, label="Simulation variance")
        ax.bar(
            [i + width for i in x],
            human_vars,
            width=width,
            label="Human variance",
        )
        ax.set_xticks([i + width / 2 for i in x])
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Variance")
        ax.set_title(metric_titles.get(metric, metric))
        ax.legend()

    for idx in range(len(metrics), n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols][idx % n_cols])

    fig.tight_layout()
    output_path = output_dir / "metric_variance_by_config.png"
    fig.savefig(output_path)
    plt.close(fig)
    paths.append(output_path)
    return paths


def plot_metrics_by_binary_config(
    output_dir: Path,
    rows: List[Tuple[str, str, str, str, float, float]],
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: List[Path] = []
    if not rows:
        return paths

    metric_titles = _metric_titles()
    grouped: Dict[str, List[Tuple[str, str, str, str, float, float]]] = {}
    for row in rows:
        grouped.setdefault(row[0], []).append(row)

    for config_key, config_rows in grouped.items():
        config_label = config_rows[0][1]
        metrics = [m for m in metric_titles if any(r[3] == m for r in config_rows)]
        if not metrics:
            continue
        n_cols = 2
        n_rows = (len(metrics) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), squeeze=False)
        for idx, metric in enumerate(metrics):
            ax = axes[idx // n_cols][idx % n_cols]
            metric_rows = [r for r in config_rows if r[3] == metric]
            metric_rows = sorted(metric_rows, key=lambda r: r[2])
            labels = [r[2] for r in metric_rows]
            sim_values = [r[4] for r in metric_rows]
            human_values = [r[5] for r in metric_rows]
            x = list(range(len(labels)))
            width = 0.35
            ax.bar(x, sim_values, width=width, label="Simulation mean")
            ax.bar(
                [i + width for i in x],
                human_values,
                width=width,
                label="Human mean",
            )
            ax.set_xticks([i + width / 2 for i in x])
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("Mean value")
            ax.set_title(metric_titles.get(metric, metric))
            ax.legend()

        for idx in range(len(metrics), n_rows * n_cols):
            fig.delaxes(axes[idx // n_cols][idx % n_cols])

        fig.suptitle(config_label, y=1.02)
        fig.tight_layout()
        output_path = output_dir / f"metric_means_by_{config_key.lower()}.png"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        paths.append(output_path)
    return paths
