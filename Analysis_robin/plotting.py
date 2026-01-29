from __future__ import annotations

from pathlib import Path
from typing import List

from .comparison import AlignmentRow


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
        labels = [row.config_pair for row in rows]
        sim_values = [row.sim_mean for row in rows]
        human_values = [row.human_mean for row in rows]
        noise = [row.human_noise for row in rows]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = list(range(len(labels)))
        ax.plot(x, human_values, marker="o", label="Human mean")
        ax.plot(x, sim_values, marker="x", label="Simulation mean")
        ax.fill_between(
            x,
            [h - n for h, n in zip(human_values, noise)],
            [h + n for h, n in zip(human_values, noise)],
            color="gray",
            alpha=0.2,
            label="Human noise ceiling (±1 SD)",
        )
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
