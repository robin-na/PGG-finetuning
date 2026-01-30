from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .aggregation import SummaryRow
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
    output_dir: Path,
    rows_by_model: Dict[str, List[ConfigMetricSummary]],
    model_labels: Dict[str, str],
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: List[Path] = []
    if not rows_by_model:
        return paths

    metric_titles = _metric_titles()
    title_to_metric = {title: key for key, title in metric_titles.items()}
    metrics = [
        m
        for m in metric_titles
        if any(row.metric == m for rows in rows_by_model.values() for row in rows)
    ]
    if not metrics:
        return paths

    model_order = [key for key in model_labels if key in rows_by_model]
    rows_by_model_metric: Dict[str, Dict[str, List[ConfigMetricSummary]]] = {}
    for model_key, rows in rows_by_model.items():
        rows_by_model_metric[model_key] = {
            metric: sorted(
                [row for row in rows if row.metric == metric], key=lambda row: row.config
            )
            for metric in metrics
        }

    noise_by_metric: Dict[str, Dict[str, float]] = {}
    for rows in rows_by_model_metric.values():
        for metric, metric_rows in rows.items():
            noise_map = noise_by_metric.setdefault(metric, {})
            for row in metric_rows:
                noise_map.setdefault(row.config, row.noise_ceiling)

    n_cols = 2
    n_rows = (len(metrics) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), squeeze=False)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // n_cols][idx % n_cols]
        configs: List[str] = sorted(
            {
                row.config
                for rows in rows_by_model_metric.values()
                for row in rows.get(metric, [])
            }
        )
        labels = [_simplify_label(config) for config in configs]
        x = list(range(len(labels)))
        width = 0.8 / (len(model_order) + 1)
        offsets = [
            (i - len(model_order) / 2) * width for i in range(len(model_order) + 1)
        ]
        for offset, model_key in zip(offsets, model_order):
            metric_rows = rows_by_model_metric.get(model_key, {}).get(metric, [])
            rmse_map = {row.config: row.rmse for row in metric_rows}
            rmse_values = [rmse_map.get(config, 0.0) for config in configs]
            ax.bar(
                [i + offset for i in x],
                rmse_values,
                width=width,
                label=model_labels.get(model_key, model_key),
            )
        noise_offset = offsets[-1] if offsets else 0.0
        noise_values = [
            noise_by_metric.get(metric, {}).get(config, 0.0) for config in configs
        ]
        ax.bar(
            [i + noise_offset for i in x],
            noise_values,
            width=width,
            label="Noise ceiling (human std)",
        )
        ax.set_xticks(x)
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
    output_dir: Path,
    rows_by_model: Dict[str, List[ConfigMetricSummary]],
    model_labels: Dict[str, str],
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: List[Path] = []
    if not rows_by_model:
        return paths

    metric_titles = _metric_titles()
    grouped: Dict[str, List[ConfigMetricSummary]] = {}
    for rows in rows_by_model.values():
        for row in rows:
            if row.metric not in metric_titles:
                continue
            grouped.setdefault(row.metric, []).append(row)

    metrics = [m for m in metric_titles if m in grouped]
    if not metrics:
        return paths

    model_order = [key for key in model_labels if key in rows_by_model]
    labels = [metric_titles[m] for m in metrics]
    x = list(range(len(metrics)))
    width = 0.8 / (len(model_order) + 1)
    offsets = [
        (i - len(model_order) / 2) * width for i in range(len(model_order) + 1)
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, model_key in zip(offsets, model_order):
        rmse_values: List[float] = []
        rmse_errors: List[float] = []
        rows = rows_by_model.get(model_key, [])
        for metric in metrics:
            metric_rows = [row for row in rows if row.metric == metric]
            config_rmse = [row.rmse for row in metric_rows]
            aggregate_rmse, rmse_err = _bootstrap_rmse(config_rmse)
            rmse_values.append(aggregate_rmse)
            rmse_errors.append(rmse_err)
        ax.bar(
            [i + offset for i in x],
            rmse_values,
            width=width,
            yerr=rmse_errors,
            capsize=4,
            label=model_labels.get(model_key, model_key),
        )
    noise_offset = offsets[-1] if offsets else 0.0
    noise_values: List[float] = []
    for metric in metrics:
        noise_samples: List[float] = []
        for rows in rows_by_model.values():
            for row in rows:
                if row.metric == metric:
                    noise_samples.append(row.noise_ceiling)
            if noise_samples:
                break
        if noise_samples:
            noise_values.append(
                (sum(value ** 2 for value in noise_samples) / len(noise_samples)) ** 0.5
            )
        else:
            noise_values.append(0.0)
    ax.bar(
        [i + noise_offset for i in x],
        noise_values,
        width=width,
        label="Noise ceiling (human std)",
    )
    ax.set_xticks(x)
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
    output_dir: Path,
    alignment_rows_by_model: Dict[str, List[AlignmentRow]],
    model_labels: Dict[str, str],
    human_summaries: Dict[str, List[SummaryRow]],
    metrics: Iterable[str],
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: List[Path] = []
    if not alignment_rows_by_model:
        return paths

    metric_titles = _metric_titles()
    metrics = [m for m in metric_titles if m in metrics]
    if not metrics:
        return paths

    human_means: Dict[str, Dict[str, float]] = {}
    human_std: Dict[str, Dict[str, float]] = {}
    for config_name, rows in human_summaries.items():
        for metric in metrics:
            values = [row.metrics.get(metric) for row in rows if row.metrics.get(metric) is not None]
            if not values:
                continue
            mean_value = sum(values) / len(values)
            variance = sum((value - mean_value) ** 2 for value in values) / len(values)
            human_means.setdefault(metric, {})[config_name] = mean_value
            human_std.setdefault(metric, {})[config_name] = variance**0.5

    model_order = [key for key in model_labels if key in alignment_rows_by_model]
    rows_by_model_metric: Dict[str, Dict[str, Dict[str, AlignmentRow]]] = {}
    for model_key, rows in alignment_rows_by_model.items():
        rows_by_model_metric[model_key] = {}
        for row in rows:
            rows_by_model_metric[model_key].setdefault(row.metric, {})[row.config] = row

    n_cols = 2
    n_rows = (len(metrics) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), squeeze=False)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // n_cols][idx % n_cols]
        configs = sorted(
            set(human_means.get(metric, {}).keys())
            | {
                row.config
                for rows in alignment_rows_by_model.values()
                for row in rows
                if row.metric == metric
            }
        )
        labels = [_simplify_label(config) for config in configs]
        x = list(range(len(labels)))
        width = 0.8 / (len(model_order) + 1)
        offsets = [
            (i - len(model_order) / 2) * width for i in range(len(model_order) + 1)
        ]
        for offset, model_key in zip(offsets, model_order):
            sim_map = rows_by_model_metric.get(model_key, {}).get(metric, {})
            sim_means = [sim_map.get(config).sim_mean if config in sim_map else 0.0 for config in configs]
            ax.bar(
                [i + offset for i in x],
                sim_means,
                width=width,
                label=model_labels.get(model_key, model_key),
            )
        human_offset = offsets[-1] if offsets else 0.0
        human_mean_values = [human_means.get(metric, {}).get(config, 0.0) for config in configs]
        human_std_values = [human_std.get(metric, {}).get(config, 0.0) for config in configs]
        ax.bar(
            [i + human_offset for i in x],
            human_mean_values,
            width=width,
            yerr=human_std_values,
            capsize=3,
            label="Human mean (±1 std)",
        )
        ax.set_xticks(x)
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
    output_dir: Path,
    alignment_rows_by_model: Dict[str, List[AlignmentRow]],
    model_labels: Dict[str, str],
    metrics: Iterable[str],
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: List[Path] = []
    if not alignment_rows_by_model:
        return paths

    metric_titles = _metric_titles()
    title_to_metric = {title: key for key, title in metric_titles.items()}
    metrics = [m for m in metric_titles if m in metrics]
    if not metrics:
        return paths

    model_order = [key for key in model_labels if key in alignment_rows_by_model]
    labels = [metric_titles[m] for m in metrics]
    x = list(range(len(labels)))
    width = 0.8 / (len(model_order) + 1)
    offsets = [
        (i - len(model_order) / 2) * width for i in range(len(model_order) + 1)
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, model_key in zip(offsets, model_order):
        rows = alignment_rows_by_model.get(model_key, [])
        means: List[float] = []
        stds: List[float] = []
        for metric in metrics:
            values = [row.sim_mean for row in rows if row.metric == metric]
            if values:
                mean_value = sum(values) / len(values)
                variance = sum((value - mean_value) ** 2 for value in values) / len(values)
                means.append(mean_value)
                stds.append(variance**0.5)
            else:
                means.append(0.0)
                stds.append(0.0)
        ax.bar(
            [i + offset for i in x],
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=model_labels.get(model_key, model_key),
        )
    human_offset = offsets[-1] if offsets else 0.0
    human_means: List[float] = []
    human_std: List[float] = []
    for metric in metrics:
        values = [
            row.human_mean
            for rows in alignment_rows_by_model.values()
            for row in rows
            if row.metric == metric
        ]
        if values:
            mean_value = sum(values) / len(values)
            variance = sum((value - mean_value) ** 2 for value in values) / len(values)
            human_means.append(mean_value)
            human_std.append(variance**0.5)
        else:
            human_means.append(0.0)
            human_std.append(0.0)
    ax.bar(
        [i + human_offset for i in x],
        human_means,
        width=width,
        yerr=human_std,
        capsize=4,
        label="Human mean",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean value")
    ax.set_title("Aggregate metric means across configurations")
    ax.legend()
    fig.tight_layout()
    output_paths = [
        output_dir / "metric_means_aggregate.png",
        output_dir / "aggregate_metric_means.png",
    ]
    for output_path in output_paths:
        fig.savefig(output_path)
    plt.close(fig)
    paths.extend(output_paths)
    return paths


def plot_metric_variance_by_config(
    output_dir: Path,
    variance_by_model: Dict[str, Dict[str, Dict[str, float]]],
    human_variance: Dict[str, Dict[str, float]],
    model_labels: Dict[str, str],
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: List[Path] = []
    if not variance_by_model:
        return paths

    metric_titles = _metric_titles()
    metrics = [
        m
        for m in metric_titles
        if any(m in config_metrics for config_metrics in human_variance.values())
        or any(
            m in config_metrics
            for model_metrics in variance_by_model.values()
            for config_metrics in model_metrics.values()
        )
    ]
    if not metrics:
        return paths

    model_order = [key for key in model_labels if key in variance_by_model]

    n_cols = 2
    n_rows = (len(metrics) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), squeeze=False)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // n_cols][idx % n_cols]
        configs = sorted(
            {
                config
                for config, metrics_map in human_variance.items()
                if metric in metrics_map
            }
            | {
                config
                for model_metrics in variance_by_model.values()
                for config, metrics_map in model_metrics.items()
                if metric in metrics_map
            }
        )
        labels = [_simplify_label(config) for config in configs]
        x = list(range(len(labels)))
        width = 0.8 / (len(model_order) + 1)
        offsets = [
            (i - len(model_order) / 2) * width for i in range(len(model_order) + 1)
        ]
        for offset, model_key in zip(offsets, model_order):
            sim_values = [
                variance_by_model.get(model_key, {}).get(config, {}).get(metric, 0.0)
                for config in configs
            ]
            ax.bar(
                [i + offset for i in x],
                sim_values,
                width=width,
                label=model_labels.get(model_key, model_key),
            )
        human_offset = offsets[-1] if offsets else 0.0
        human_values = [human_variance.get(config, {}).get(metric, 0.0) for config in configs]
        ax.bar(
            [i + human_offset for i in x],
            human_values,
            width=width,
            label="Human variance",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Variance across players")
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


def plot_aggregate_metric_variance(
    output_dir: Path,
    variance_by_model: Dict[str, Dict[str, Dict[str, float]]],
    human_variance: Dict[str, Dict[str, float]],
    model_labels: Dict[str, str],
    metrics: Iterable[str],
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: List[Path] = []
    if not variance_by_model and not human_variance:
        return paths

    metric_titles = _metric_titles()
    metrics = [m for m in metric_titles if m in metrics]
    if not metrics:
        return paths

    model_order = [key for key in model_labels if key in variance_by_model]
    labels = [metric_titles[m] for m in metrics]
    x = list(range(len(metrics)))
    width = 0.8 / (len(model_order) + 1)
    offsets = [
        (i - len(model_order) / 2) * width for i in range(len(model_order) + 1)
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, model_key in zip(offsets, model_order):
        model_metrics = variance_by_model.get(model_key, {})
        means: List[float] = []
        for metric in metrics:
            values = [
                config_metrics.get(metric)
                for config_metrics in model_metrics.values()
                if config_metrics.get(metric) is not None
            ]
            means.append(sum(values) / len(values) if values else 0.0)
        ax.bar(
            [i + offset for i in x],
            means,
            width=width,
            label=model_labels.get(model_key, model_key),
        )
    human_offset = offsets[-1] if offsets else 0.0
    human_means: List[float] = []
    for metric in metrics:
        values = [
            config_metrics.get(metric)
            for config_metrics in human_variance.values()
            if config_metrics.get(metric) is not None
        ]
        human_means.append(sum(values) / len(values) if values else 0.0)
    ax.bar(
        [i + human_offset for i in x],
        human_means,
        width=width,
        label="Human mean",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Variance across players")
    ax.set_title("Aggregate variance across configurations")
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "aggregate_metric_variance.png"
    fig.savefig(output_path)
    plt.close(fig)
    paths.append(output_path)
    return paths


def plot_metrics_by_binary_config(
    output_dir: Path,
    rows: List[Tuple[str, str, bool, str, str, str, float, float]],
    model_labels: Dict[str, str],
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: List[Path] = []
    if not rows:
        return paths

    metric_titles = _metric_titles()
    title_to_metric = {title: key for key, title in metric_titles.items()}
    grouped: Dict[str, List[Tuple[str, str, bool, str, str, str, float, float]]] = {}
    for row in rows:
        grouped.setdefault(row[0], []).append(row)

    config_keys = sorted(grouped.keys())
    if not config_keys:
        return paths
    n_cols = 2
    n_rows = (len(config_keys) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), squeeze=False)
    model_order = [key for key in model_labels if key in {row[5] for row in rows}]
    for idx, config_key in enumerate(config_keys):
        ax = axes[idx // n_cols][idx % n_cols]
        config_rows = grouped[config_key]
        config_label = config_rows[0][1]
        metrics = [m for m in metric_titles if any(r[4] == m for r in config_rows)]
        value_order = [False, True]
        value_labels = {
            row[2]: row[3] for row in config_rows if row[2] in value_order
        }
        labels: List[str] = []
        positions: List[int] = []
        pos = 0
        for metric in metrics:
            for value in value_order:
                if not any(r[4] == metric and r[2] == value for r in config_rows):
                    continue
                labels.append(
                    f"{metric_titles.get(metric, metric)}\n{value_labels.get(value, str(value))}"
                )
                positions.append(pos)
                pos += 1
        if not labels:
            continue
        x = positions
        width = 0.8 / (len(model_order) + 1)
        offsets = [
            (i - len(model_order) / 2) * width for i in range(len(model_order) + 1)
        ]
        for offset, model_key in zip(offsets, model_order):
            sim_values = []
            for label in labels:
                metric_label, value_label = label.split("\n", 1)
                metric = title_to_metric.get(metric_label)
                if not metric:
                    sim_values.append(0.0)
                    continue
                value = next(
                    (
                        v
                        for v, v_label in value_labels.items()
                        if v_label == value_label
                    ),
                    None,
                )
                match = next(
                    (
                        r
                        for r in config_rows
                        if r[4] == metric and r[2] == value and r[5] == model_key
                    ),
                    None,
                )
                sim_values.append(match[6] if match else 0.0)
            ax.bar(
                [i + offset for i in x],
                sim_values,
                width=width,
                label=model_labels.get(model_key, model_key),
            )
        human_offset = offsets[-1] if offsets else 0.0
        human_values = []
        for label in labels:
            metric_label, value_label = label.split("\n", 1)
            metric = title_to_metric.get(metric_label)
            if not metric:
                human_values.append(0.0)
                continue
            value = next(
                (v for v, v_label in value_labels.items() if v_label == value_label),
                None,
            )
            match = next(
                (r for r in config_rows if r[4] == metric and r[2] == value),
                None,
            )
            human_values.append(match[7] if match else 0.0)
        ax.bar(
            [i + human_offset for i in x],
            human_values,
            width=width,
            label="Human mean",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Mean value")
        ax.set_title(config_label)
        ax.legend()

    for idx in range(len(config_keys), n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols][idx % n_cols])

    fig.tight_layout()
    output_path = output_dir / "metric_means_by_binary_config.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    paths.append(output_path)
    return paths
