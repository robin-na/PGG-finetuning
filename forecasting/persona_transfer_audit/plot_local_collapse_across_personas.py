"""Compare local top-1 collapse across persona sources and target games."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent

PERSONA_RUNS = [
    {
        "label": "Demographic\nsurveys",
        "display_label": "Demographic surveys",
        "pgg_run": "argyle_anes2016_backstory_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2",
        "bargaining_run": "argyle_anes2016_backstory_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2",
    },
    {
        "label": "Twin-2K-500",
        "display_label": "Twin-2K-500",
        "pgg_run": "twin_direct_summary_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2",
        "bargaining_run": "twin_direct_summary_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2",
    },
    {
        "label": "Synthetic\n(Nemotron)",
        "display_label": "Synthetic (Nemotron)",
        "pgg_run": "nemotron_raw_fields_adult_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2",
        "bargaining_run": "nemotron_raw_fields_adult_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2",
    },
    {
        "label": "Task-adaptive\n(Concordia)",
        "display_label": "Task-adaptive (Concordia)",
        "pgg_run": "concordia_pgg_game_grounded_alphaevolve_5_compact_to_pgg_stratified_32x40_top3_gpt_5_mini",
        "bargaining_run": "concordia_chip_bargain_game_grounded_alphaevolve_5_compact_to_chip_bargain_stratified_32x48_top3_gpt_5_mini",
    },
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def _median(values: list[float]) -> float:
    return _percentile(values, 0.5)


def _modal_share_null(
    n: int,
    k: int,
    *,
    iterations: int = 10000,
    seed: int = 20260520,
) -> tuple[float, float, float]:
    rng = random.Random(seed + n * 1009 + k * 9173)
    values: list[float] = []
    for _ in range(iterations):
        counts = [0] * k
        for __ in range(n):
            counts[rng.randrange(k)] += 1
        values.append(max(counts) / n)
    return _percentile(values, 0.025), sum(values) / len(values), _percentile(values, 0.975)


def _safe_players(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def _effective_n(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        if probability > 0:
            entropy -= probability * math.log(probability)
    return math.exp(entropy)


def _collect_game_rows(
    metadata_root: Path,
    *,
    game_key: str,
    run_key: str,
    game_label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for persona_index, run in enumerate(PERSONA_RUNS):
        metadata_dir = metadata_root / str(run[run_key])
        parsed_path = metadata_dir / "parsed_matches_long.csv"
        parsed_rows = _read_csv(parsed_path)
        by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in parsed_rows:
            if str(row.get("is_top1", "")).lower() != "true" and row.get("match_rank") != "1":
                continue
            by_game[row[game_key]].append(row)

        for game_id, game_rows in sorted(by_game.items()):
            top1_counts = Counter(row["player"] for row in game_rows)
            if not top1_counts:
                continue
            first = game_rows[0]
            players = _safe_players(first.get("players", "[]"))
            n_players = len(players) or len(set(top1_counts))
            top1_total = sum(top1_counts.values())
            if n_players <= 1 or top1_total <= 1:
                continue
            top_player, top_count = top1_counts.most_common(1)[0]
            observed = top_count / top1_total
            null_low, null_mean, null_high = _modal_share_null(top1_total, n_players)
            effective_n = _effective_n(top1_counts)
            rows.append(
                {
                    "game": game_label,
                    "persona_source": run["display_label"],
                    "persona_order": persona_index,
                    "run": run[run_key],
                    "unit_id": game_id,
                    "treatment_name": first.get("treatment_name", ""),
                    "n_players": n_players,
                    "top1_total": top1_total,
                    "top1_unique_players": len(top1_counts),
                    "top1_effective_n": effective_n,
                    "top1_effective_n_share_of_players": effective_n / n_players if n_players else float("nan"),
                    "top1_top_player": top_player,
                    "top1_top_player_count": top_count,
                    "top1_top_player_share": observed,
                    "uniform_null_low": null_low,
                    "uniform_null_mean": null_mean,
                    "uniform_null_high": null_high,
                    "excess_modal_share": observed - null_mean,
                }
            )
    return rows


def _collect_rows(metadata_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(_collect_game_rows(metadata_root, game_key="game_id", run_key="pgg_run", game_label="Public goods game"))
    rows.extend(
        _collect_game_rows(
            metadata_root,
            game_key="record_id",
            run_key="bargaining_run",
            game_label="Bargaining game",
        )
    )
    return rows


def _bootstrap_ci(values: list[float], *, seed: int, iterations: int = 5000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    medians: list[float] = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for __ in values]
        medians.append(_median(sample))
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def _plot_panel(ax: plt.Axes, rows: list[dict[str, Any]], *, game: str, seed_offset: int) -> None:
    game_rows = [row for row in rows if row["game"] == game]
    colors = ["#5b8cc0", "#7c66c2", "#c78042", "#4f9a74"]
    x_positions = list(range(len(PERSONA_RUNS)))
    rng = random.Random(1234 + seed_offset)

    ax.axhline(0.0, color="#4d4d4d", linewidth=1.0)
    for x, run in zip(x_positions, PERSONA_RUNS):
        label = run["display_label"]
        source_rows = [row for row in game_rows if row["persona_source"] == label]
        values = [float(row["excess_modal_share"]) for row in source_rows]
        jittered_x = [x + rng.uniform(-0.12, 0.12) for _ in values]
        ax.scatter(
            jittered_x,
            values,
            s=16,
            color=colors[x],
            alpha=0.55,
            edgecolor="white",
            linewidth=0.25,
            zorder=2,
        )
        if values:
            median = _median(values)
            low, high = _bootstrap_ci(values, seed=9000 + seed_offset * 100 + x)
            ax.plot([x - 0.23, x + 0.23], [median, median], color="#202020", linewidth=2.1, zorder=4)
            ax.plot([x, x], [low, high], color="#202020", linewidth=1.2, zorder=3)
            ax.text(
                x,
                median + 0.035,
                f"{median:+.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#202020",
                zorder=5,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([run["label"] for run in PERSONA_RUNS], rotation=0, ha="center")
    ax.set_title(game, pad=10)
    ax.set_ylabel("Excess modal top-1 share\nabove uniform null")
    ax.set_ylim(-0.05, 0.65)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)


def _plot(rows: list[dict[str, Any]], output_dir: Path, output_stem: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), sharey=True)
    _plot_panel(axes[0], rows, game="Public goods game", seed_offset=0)
    _plot_panel(axes[1], rows, game="Bargaining game", seed_offset=1)
    axes[1].set_ylabel("")
    fig.suptitle("Local identity collapse across persona sources", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
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
    parser.add_argument("--metadata-root", type=Path, default=THIS_DIR / "metadata")
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "figures")
    parser.add_argument("--output-stem", default="figure_local_collapse_across_personas")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
