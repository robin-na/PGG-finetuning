"""Draw a manuscript-style schematic for revealed-behavior matching."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


THIS_DIR = Path(__file__).resolve().parent


BLUE = "#2678d9"
ORANGE = "#ff7f2a"
RED = "#e31a1c"
TEXT = "#222222"
MUTED = "#6d7480"
LINE = "#4a4a4a"
BLUE_FILL = "#f5f9ff"
ORANGE_FILL = "#fff8f2"
RED_FILL = "#fff7f7"


def _panel(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    color: str,
    letter: str,
    title: str,
    subtitle: str,
) -> None:
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        linewidth=0.95,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(panel)
    badge = Circle((x + 0.025, y + h - 0.032), 0.0155, facecolor=color, edgecolor="none")
    ax.add_patch(badge)
    ax.text(x + 0.025, y + h - 0.032, letter, ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    ax.text(x + 0.053, y + h - 0.025, title, ha="left", va="top", fontsize=9.6, fontweight="bold", color=TEXT)
    ax.text(x + 0.053, y + h - 0.055, subtitle, ha="left", va="top", fontsize=6.6, color=MUTED, linespacing=1.10)


def _card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    edge: str,
    face: str = "#ffffff",
    fontsize: float = 7.4,
    weight: str = "normal",
    align: str = "center",
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.009,rounding_size=0.010",
        linewidth=0.65,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha=align,
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=TEXT,
        linespacing=1.12,
    )


def _source_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    note: str,
    *,
    edge: str,
    face: str,
    icon: bool = True,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.009,rounding_size=0.010",
        linewidth=0.65,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    if icon:
        _stack_icon(ax, x + 0.016, y + 0.016, BLUE)
        tx = x + 0.071
    else:
        ax.add_patch(Rectangle((x + 0.025, y + 0.025), 0.030, 0.028, linewidth=0.9, edgecolor=BLUE, facecolor="none"))
        ax.plot([x + 0.031, x + 0.050], [y + 0.039, y + 0.039], color=BLUE, linewidth=0.8, alpha=0.45)
        tx = x + 0.071
    ax.text(tx, y + h * 0.61, title, ha="left", va="center", fontsize=6.8, fontweight="bold", color=TEXT, linespacing=1.02)
    ax.text(tx, y + h * 0.28, note, ha="left", va="center", fontsize=5.35, color=MUTED)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = LINE, rad: float = 0.0) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=1.0,
        color=color,
        shrinkA=2,
        shrinkB=2,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)


def _stack_icon(ax: plt.Axes, x: float, y: float, color: str) -> None:
    for i in range(3):
        ax.add_patch(
            Rectangle(
                (x + i * 0.0038, y + i * 0.0054),
                0.030,
                0.036,
                linewidth=0.65,
                edgecolor=color,
                facecolor="#ffffff",
            )
        )
    for j in range(3):
        ax.plot([x + 0.007, x + 0.025], [y + 0.029 - j * 0.008, y + 0.029 - j * 0.008], color=color, linewidth=0.55)


def _person_icon(ax: plt.Axes, x: float, y: float, color: str) -> None:
    ax.add_patch(Circle((x, y + 0.017), 0.009, facecolor=color, edgecolor="none"))
    ax.add_patch(Rectangle((x - 0.008, y - 0.006), 0.016, 0.020, facecolor=color, edgecolor="none"))


def _contrib_sequence(ax: plt.Axes, x: float, y: float, values: list[int], *, color: str) -> None:
    for i, value in enumerate(values):
        height = 0.008 + 0.024 * (value / 20)
        ax.add_patch(Rectangle((x + i * 0.012, y), 0.008, height, facecolor=color, edgecolor="none", alpha=0.85))
    ax.text(x + 0.034, y - 0.007, "rounds", ha="center", va="top", fontsize=4.9, color=MUTED)


def draw(output_dir: Path, output_stem: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "figure.dpi": 170,
            "savefig.dpi": 320,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(11.6, 5.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _panel(
        ax,
        0.015,
        0.050,
        0.310,
        0.900,
        color=BLUE,
        letter="A",
        title="Persona inputs",
        subtitle="Profile sources used to condition the model",
    )
    _panel(
        ax,
        0.345,
        0.050,
        0.310,
        0.900,
        color=ORANGE,
        letter="B",
        title="Revealed-behavior matching",
        subtitle="The model chooses among real observed trajectories",
    )
    _panel(
        ax,
        0.675,
        0.050,
        0.310,
        0.900,
        color=RED,
        letter="C",
        title="Coverage and evaluation",
        subtitle="Assess whether matched behavior spans the human support",
    )

    # Panel A
    source_cards = [
        ("No profile\nbaseline", "empty prompt"),
        ("Demographic survey\nprofile", "age, gender, education"),
        ("Digital-twin survey\nsummary", "survey-derived person profile"),
        ("Synthetic persona\nlibrary", "generated population profiles"),
        ("Game-grounded persona\ngenerator", "task-relevant axes"),
    ]
    y0 = 0.760
    for i, (title, note) in enumerate(source_cards):
        y = y0 - i * 0.112
        _source_card(
            ax,
            0.064,
            y,
            0.205,
            0.070,
            title,
            note,
            edge="#c7dcf7",
            face=BLUE_FILL,
            icon=i != 0,
        )

    _arrow(ax, (0.166, 0.238), (0.166, 0.197), color=BLUE)
    _card(
        ax,
        0.082,
        0.128,
        0.168,
        0.058,
        "sample one profile\nfor each scenario",
        edge="#c7dcf7",
        face="#ffffff",
        fontsize=6.5,
        weight="bold",
    )

    # Panel B
    _card(
        ax,
        0.385,
        0.742,
        0.106,
        0.080,
        "Sampled\nprofile",
        edge="#ffd4b7",
        face=ORANGE_FILL,
        fontsize=7.0,
        weight="bold",
    )
    ax.text(
        0.439,
        0.713,
        "e.g., dislikes free-riding,\nvalues fairness, speaks rarely",
        ha="center",
        va="top",
        fontsize=5.3,
        color=MUTED,
        linespacing=1.08,
    )

    _card(
        ax,
        0.505,
        0.742,
        0.106,
        0.080,
        "PGG\ntranscript",
        edge="#ffd4b7",
        face=ORANGE_FILL,
        fontsize=7.0,
        weight="bold",
    )
    ax.text(
        0.559,
        0.713,
        "players' contributions,\nchat, rewards, punishments",
        ha="center",
        va="top",
        fontsize=5.3,
        color=MUTED,
        linespacing=1.08,
    )
    _arrow(ax, (0.492, 0.782), (0.506, 0.782), color=ORANGE)

    ax.text(0.500, 0.645, "Observed candidate behaviors", ha="center", va="center", fontsize=7.3, fontweight="bold", color=TEXT)
    behavior_cards = [
        ("Player A", "fully cooperative", [20, 20, 20, 20, 20, 20], "no chat", "#2b8cbe"),
        ("Player B", "conditional cooperator", [20, 20, 0, 20, 0, 20], "no communication", "#7b68ee"),
        ("Player C", "fully defective", [0, 0, 0, 0, 0, 0], "cheap talk: \"all in\"", "#d95f0e"),
    ]
    for i, (player, label, values, note, color) in enumerate(behavior_cards):
        x = 0.372 + i * 0.091
        _card(ax, x, 0.438, 0.082, 0.150, "", edge="#ffd4b7", face="#fffdf9", fontsize=6.2)
        ax.text(x + 0.041, 0.565, player, ha="center", va="center", fontsize=5.9, fontweight="bold", color=TEXT)
        ax.text(x + 0.041, 0.544, label, ha="center", va="center", fontsize=5.25, color=TEXT)
        _contrib_sequence(ax, x + 0.014, 0.495, values, color=color)
        ax.text(x + 0.041, 0.462, note, ha="center", va="center", fontsize=4.95, color=MUTED)

    _arrow(ax, (0.500, 0.435), (0.500, 0.370), color=ORANGE)
    _card(
        ax,
        0.390,
        0.270,
        0.220,
        0.083,
        "LLM returns top-k matches\nwith probabilities that sum to 1",
        edge="#ffd4b7",
        face=ORANGE_FILL,
        fontsize=6.6,
        weight="bold",
    )
    for i, (name, pct, color) in enumerate([("A", 0.55, "#2b8cbe"), ("B", 0.35, "#7b68ee"), ("C", 0.10, "#d95f0e")]):
        y = 0.218 - i * 0.034
        ax.text(0.405, y + 0.008, name, ha="left", va="center", fontsize=5.9, color=TEXT)
        ax.add_patch(Rectangle((0.425, y), 0.128, 0.016, facecolor="#eeeeee", edgecolor="none"))
        ax.add_patch(Rectangle((0.425, y), 0.128 * pct, 0.016, facecolor=color, edgecolor="none", alpha=0.85))
        ax.text(0.563, y + 0.008, f"{pct:.2f}", ha="left", va="center", fontsize=5.7, color=MUTED)

    # Panel C
    _card(
        ax,
        0.714,
        0.735,
        0.222,
        0.092,
        "Human behavioral support\nall observed trajectories in the scenario",
        edge="#ffb7b5",
        face=RED_FILL,
        fontsize=6.7,
        weight="bold",
    )
    _card(
        ax,
        0.714,
        0.604,
        0.222,
        0.092,
        "Matched distribution\nprobability-weighted selected trajectories",
        edge="#ffb7b5",
        face=RED_FILL,
        fontsize=6.7,
        weight="bold",
    )
    _arrow(ax, (0.825, 0.735), (0.825, 0.696), color=RED)

    ax.text(0.745, 0.545, "Behavioral checks", ha="left", va="center", fontsize=7.3, fontweight="bold", color=TEXT)
    eval_items = [
        ("Behavioral skew", "Do selected players over-contribute, over-reward, etc.?"),
        ("Behavior-space coverage", "Does the selected distribution cover observed behavior types?"),
        ("Residual identity concentration", "Does matching collapse after similar behaviors are grouped?"),
        ("Support vs. sampling", "Could reweighting personas recover the human distribution?"),
    ]
    for i, (title, note) in enumerate(eval_items):
        y = 0.450 - i * 0.080
        _card(ax, 0.718, y, 0.214, 0.058, title, edge="#ffcccc", face="#ffffff", fontsize=6.25, weight="bold")
        ax.text(0.825, y + 0.006, note, ha="center", va="bottom", fontsize=4.95, color=MUTED)

    fig.tight_layout(pad=0.4)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"{output_stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "figures")
    parser.add_argument("--output-stem", default="figure_persona_matching_pipeline_schematic")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    draw(args.output_dir.expanduser().resolve(), args.output_stem)
