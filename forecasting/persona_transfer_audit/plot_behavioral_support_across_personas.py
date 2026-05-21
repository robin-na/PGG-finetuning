"""Plot behavior-space effective support across persona sources."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent

RUNS = [
    {
        "label": "No persona",
        "display_label": "No persona",
        "pgg_run": "no_persona_to_pgg_stratified_40_top3_gpt_5_mini_seed_2",
        "bargaining_run": "no_persona_to_chip_bargain_stratified_48_top3_gpt_5_mini_seed_2",
    },
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

PGG_FEATURES = [
    "mean_contribution_rate",
    "full_contribution_rate",
    "zero_contribution_rate",
    "contribution_sd",
    "messages_per_round",
    "reward_given_round_rate",
    "punish_given_round_rate",
    "punish_received_round_rate",
]

BARGAINING_FEATURES = [
    "final_surplus",
    "final_welfare",
    "proposer_mean_net_surplus",
    "proposer_acceptance_rate",
    "proposer_mean_trade_ratio",
    "response_acceptance_rate",
    "response_mean_net_surplus_if_accepted",
    "received_trade_rate",
]

COLORS = ["#6b6b6b", "#5b8cc0", "#7c66c2", "#c78042", "#4f9a74"]


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


def _as_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _percentile(values: list[float], p: float) -> float:
    values = sorted(value for value in values if not math.isnan(value))
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


def _bootstrap_ci(values: list[float], *, seed: int, iterations: int = 5000) -> tuple[float, float]:
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    medians: list[float] = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for __ in values]
        medians.append(_median(sample))
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def _deduplicate_candidate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        key = row.get("player_key", "")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _feature_stats(rows: list[dict[str, str]], features: list[str]) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for feature in features:
        values = np.array(
            [
                value
                for value in (_as_float(row.get(feature)) for row in rows)
                if not math.isnan(value)
            ],
            dtype=float,
        )
        if values.size == 0:
            continue
        sd = float(np.std(values))
        if sd <= 1e-12:
            continue
        stats[feature] = (float(np.mean(values)), sd)
    return stats


def _standardized_matrix(
    rows: list[dict[str, str]],
    stats: dict[str, tuple[float, float]],
) -> np.ndarray:
    matrix: list[list[float]] = []
    for row in rows:
        values: list[float] = []
        for feature, (mean, sd) in stats.items():
            value = _as_float(row.get(feature))
            if math.isnan(value):
                values.append(0.0)
            else:
                values.append((value - mean) / sd)
        matrix.append(values)
    return np.array(matrix, dtype=float)


def _kernel_and_distances(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if matrix.shape[0] == 0:
        empty = np.zeros((0, 0), dtype=float)
        return empty, empty, float("nan")
    differences = matrix[:, None, :] - matrix[None, :, :]
    distances_sq = np.sum(differences * differences, axis=2)
    distances = np.sqrt(np.maximum(distances_sq, 0.0))
    nonzero_distances = distances[distances > 1e-12]
    sigma = float(np.median(nonzero_distances)) if nonzero_distances.size else 1.0
    if not math.isfinite(sigma) or sigma <= 1e-12:
        sigma = 1.0
    kernel = np.exp(-distances_sq / (2.0 * sigma * sigma))
    return kernel, distances, sigma


def _normalize(weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    if total <= 0 or not math.isfinite(total):
        return np.ones_like(weights, dtype=float) / len(weights)
    return weights / total


def _kernel_effective_n(weights: np.ndarray, kernel: np.ndarray) -> float:
    concentration = float(weights @ kernel @ weights)
    if concentration <= 0 or not math.isfinite(concentration):
        return float("nan")
    return 1.0 / concentration


def _probability_effective_n(weights: np.ndarray) -> float:
    concentration = float(np.sum(weights * weights))
    if concentration <= 0 or not math.isfinite(concentration):
        return float("nan")
    return 1.0 / concentration


def _expected_pairwise_distance(weights: np.ndarray, distances: np.ndarray) -> float:
    return float(weights @ distances @ weights)


def _probability_column(row: dict[str, str]) -> float:
    value = _as_float(row.get("match_probability"))
    if math.isnan(value):
        value = _as_float(row.get("probability"))
    return 0.0 if math.isnan(value) else max(value, 0.0)


def _collect_q_weights(
    matched_rows: list[dict[str, str]],
    *,
    game_id: str,
    game_key: str,
    player_keys: list[str],
) -> tuple[np.ndarray, float, float]:
    key_to_index = {key: index for index, key in enumerate(player_keys)}
    weights = np.zeros(len(player_keys), dtype=float)
    raw_total = 0.0
    unmatched = 0.0
    for row in matched_rows:
        if row.get(game_key) != game_id:
            continue
        probability = _probability_column(row)
        if probability <= 0:
            continue
        raw_total += probability
        player_key = row.get("player_key") or f"{game_id}::{row.get('player', '')}"
        index = key_to_index.get(player_key)
        if index is None:
            unmatched += probability
            continue
        weights[index] += probability
    matched_total = float(np.sum(weights))
    if matched_total <= 0:
        return np.ones(len(player_keys), dtype=float) / len(player_keys), raw_total, unmatched
    return weights / matched_total, raw_total, unmatched


def _candidate_weights(rows: list[dict[str, str]]) -> np.ndarray:
    weights = np.array([_as_float(row.get("candidate_uniform_weight")) for row in rows], dtype=float)
    if np.any(np.isnan(weights)) or float(np.sum(weights)) <= 0:
        weights = np.ones(len(rows), dtype=float)
    return _normalize(weights)


def _collect_game_rows(
    *,
    metadata_root: Path,
    game_label: str,
    run_key: str,
    canonical_run: str,
    candidate_file: str,
    matched_file: str,
    game_key: str,
    features: list[str],
) -> list[dict[str, Any]]:
    canonical_candidates = _deduplicate_candidate_rows(
        _read_csv(metadata_root / canonical_run / candidate_file)
    )
    stats = _feature_stats(canonical_candidates, features)
    by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in canonical_candidates:
        by_game[row[game_key]].append(row)

    rows: list[dict[str, Any]] = []
    for run_index, run in enumerate(RUNS):
        metadata_dir = metadata_root / str(run[run_key])
        matched_rows = _read_csv(metadata_dir / matched_file)
        for game_id, candidate_rows in sorted(by_game.items()):
            player_keys = [row["player_key"] for row in candidate_rows]
            matrix = _standardized_matrix(candidate_rows, stats)
            kernel, distances, sigma = _kernel_and_distances(matrix)
            p_weights = _candidate_weights(candidate_rows)
            q_weights, raw_q_total, unmatched_q = _collect_q_weights(
                matched_rows,
                game_id=game_id,
                game_key=game_key,
                player_keys=player_keys,
            )

            p_effective = _kernel_effective_n(p_weights, kernel)
            q_effective = _kernel_effective_n(q_weights, kernel)
            p_probability_effective = _probability_effective_n(p_weights)
            q_probability_effective = _probability_effective_n(q_weights)
            p_distance = _expected_pairwise_distance(p_weights, distances)
            q_distance = _expected_pairwise_distance(q_weights, distances)
            rows.append(
                {
                    "game": game_label,
                    "persona_source": run["display_label"],
                    "persona_order": run_index,
                    "run": run[run_key],
                    "unit_id": game_id,
                    "treatment_name": candidate_rows[0].get("treatment_name", ""),
                    "n_players": len(candidate_rows),
                    "n_features": len(stats),
                    "features": "|".join(stats.keys()),
                    "kernel_sigma": sigma,
                    "q_raw_probability_total": raw_q_total,
                    "q_unmatched_probability": unmatched_q,
                    "p_kernel_effective_n": p_effective,
                    "q_kernel_effective_n": q_effective,
                    "kernel_effective_support_ratio": q_effective / p_effective
                    if p_effective > 0
                    else float("nan"),
                    "p_probability_effective_n": p_probability_effective,
                    "q_probability_effective_n": q_probability_effective,
                    "probability_effective_support_ratio": q_probability_effective
                    / p_probability_effective
                    if p_probability_effective > 0
                    else float("nan"),
                    "p_expected_pairwise_distance": p_distance,
                    "q_expected_pairwise_distance": q_distance,
                    "pairwise_distance_ratio": q_distance / p_distance
                    if p_distance > 0
                    else float("nan"),
                }
            )
    return rows


def _collect_rows(metadata_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(
        _collect_game_rows(
            metadata_root=metadata_root,
            game_label="Public goods game",
            run_key="pgg_run",
            canonical_run="no_persona_to_pgg_stratified_40_top3_gpt_5_mini_seed_2",
            candidate_file="candidate_uniform_behavior_long.csv",
            matched_file="matched_player_behavior_long.csv",
            game_key="game_id",
            features=PGG_FEATURES,
        )
    )
    rows.extend(
        _collect_game_rows(
            metadata_root=metadata_root,
            game_label="Bargaining game",
            run_key="bargaining_run",
            canonical_run="no_persona_to_chip_bargain_stratified_48_top3_gpt_5_mini_seed_2",
            candidate_file="chip_candidate_uniform_behavior_long.csv",
            matched_file="chip_matched_player_behavior_long.csv",
            game_key="record_id",
            features=BARGAINING_FEATURES,
        )
    )
    return rows


def _plot_panel(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    game: str,
    seed_offset: int,
) -> None:
    game_rows = [row for row in rows if row["game"] == game]
    rng = random.Random(20260520 + seed_offset)
    x_positions = list(range(len(RUNS)))

    ax.axhline(1.0, color="#4d4d4d", linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
    for x, run in zip(x_positions, RUNS):
        source_rows = [row for row in game_rows if row["persona_source"] == run["display_label"]]
        values = [
            float(row["kernel_effective_support_ratio"])
            for row in source_rows
            if not math.isnan(float(row["kernel_effective_support_ratio"]))
        ]
        jittered_x = [x + rng.uniform(-0.13, 0.13) for _ in values]
        ax.scatter(
            jittered_x,
            values,
            s=17,
            color=COLORS[x],
            alpha=0.55,
            edgecolor="white",
            linewidth=0.25,
            zorder=2,
        )
        if values:
            median = _median(values)
            low, high = _bootstrap_ci(values, seed=8100 + seed_offset * 100 + x)
            ax.plot([x - 0.24, x + 0.24], [median, median], color="#202020", linewidth=2.2, zorder=4)
            ax.plot([x, x], [low, high], color="#202020", linewidth=1.2, zorder=3)
            ax.text(
                x,
                min(median + 0.045, 1.05),
                f"{median:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#202020",
                zorder=5,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([run["label"] for run in RUNS], rotation=0, ha="center")
    ax.set_title(game, pad=10)
    ax.set_ylim(0.0, 1.1)
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
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True)
    _plot_panel(axes[0], rows, game="Public goods game", seed_offset=0)
    _plot_panel(axes[1], rows, game="Bargaining game", seed_offset=1)
    axes[0].set_ylabel("Behavioral effective support ratio\n$N_{eff}^K(Q_g) / N_{eff}^K(P_g)$")
    axes[0].text(-0.12, 1.05, "A", transform=axes[0].transAxes, fontsize=13, fontweight="bold")
    axes[1].text(-0.08, 1.05, "B", transform=axes[1].transAxes, fontsize=13, fontweight="bold")
    fig.suptitle("Behavior-space support of matched human trajectories", y=1.02, fontsize=12)
    fig.tight_layout()
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
    parser.add_argument("--output-stem", default="figure_behavioral_support_across_personas")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
