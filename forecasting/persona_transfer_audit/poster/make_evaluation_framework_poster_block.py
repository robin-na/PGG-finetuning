"""Render a compact poster-ready notation block."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


THIS_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = THIS_DIR / "evaluation_framework_poster_block"

MAROON = "#941d2b"
INK = "#222222"
MUTED = "#5f5f5f"
PANEL_BG = "#fcf8f8"
LINE = "#d7bdc1"


def add_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=1.15,
            edgecolor=LINE,
            facecolor=PANEL_BG,
        )
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(11.2, 3.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.03,
        0.91,
        "Behavioral skewness notation",
        color=INK,
        fontsize=15.5,
        fontweight="bold",
        va="center",
    )
    ax.text(
        0.03,
        0.84,
        "Compare real human trajectories selected by each persona source to a within-game uniform human baseline.",
        color=MUTED,
        fontsize=10.8,
        va="center",
    )

    add_box(ax, (0.03, 0.17), 0.94, 0.60)
    row_y = [0.65, 0.47, 0.28]
    labels = ["Human baseline", "LLM-persona match", "Plotted skew"]
    equations = [
        r"$P_g(i)=1/n_g$",
        r"$Q_{gs}(i)=\mathrm{Avg}_a\,r_{gasi}$",
        r"$\widetilde{\delta}_{s\ell}=\frac{1}{\sigma^P_\ell}\,\mathrm{Avg}_{g}\{E_{Q_{gs}}[x_{\ell}]-E_{P_g}[x_{\ell}]\}$",
    ]
    notes = [
        "uniform over observed players in game g",
        "average top-3 probabilities across personas",
        "SD-standardized feature difference",
    ]

    for y, label, equation, note in zip(row_y, labels, equations, notes):
        ax.text(0.065, y, label, fontsize=11.0, fontweight="bold", color=INK, va="center")
        ax.text(0.285, y, equation, fontsize=14.8 if label != "Plotted skew" else 12.5, color=MAROON, va="center")
        ax.text(0.75, y, note, fontsize=9.5, color=MUTED, va="center")
        if y != row_y[-1]:
            ax.plot([0.06, 0.94], [y - 0.09, y - 0.09], color="#eadde0", lw=1.0)

    ax.text(
        0.065,
        0.08,
        r"$x_{gi\ell}$ = behavior feature $\ell$ for player $i$ in game $g$; $r_{gasi}=0$ for unlisted top-3 players. "
        r"Values near 0 match the human baseline; positive/negative values indicate over-/under-selection.",
        fontsize=9.2,
        color=MUTED,
    )

    fig.savefig(f"{OUTPUT_STEM}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{OUTPUT_STEM}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{OUTPUT_STEM}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
