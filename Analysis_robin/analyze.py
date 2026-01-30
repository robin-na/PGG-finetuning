from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .aggregation import (
    summarize_by_game,
    summarize_by_player,
    summarize_by_round,
    write_game_summary,
    write_player_summary,
    write_round_summary,
)
from .comparison import compare_configs, write_alignment, write_metric_summary
from .config import (
    DEFAULT_ANALYSIS_OUTPUT_ROOT,
    DEFAULT_HUMAN_CONFIG_CSV,
    DEFAULT_HUMAN_ROUNDS_CSV,
    DEFAULT_OUTPUT_ROOT,
)
from .io_utils import load_human_rows, load_simulation_runs
from .plotting import plot_noise_ceiling


def _timestamp() -> str:
    return datetime.now().strftime("%y%m%d%H%M")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PGG simulation vs human data")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Path to simulation outputs.",
    )
    parser.add_argument(
        "--analysis-output-root",
        type=Path,
        default=DEFAULT_ANALYSIS_OUTPUT_ROOT,
        help="Where to write analysis outputs.",
    )
    parser.add_argument(
        "--human-rounds",
        type=Path,
        default=DEFAULT_HUMAN_ROUNDS_CSV,
        help="Human player-rounds CSV.",
    )
    parser.add_argument(
        "--human-configs",
        type=Path,
        default=DEFAULT_HUMAN_CONFIG_CSV,
        help="Human config mapping CSV.",
    )
    args = parser.parse_args()

    sim_rows = load_simulation_runs(args.output_root)
    human_rows = load_human_rows(args.human_rounds, args.human_configs)

    timestamp = _timestamp()
    output_dir = args.analysis_output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_game_summary = {
        config: summarize_by_game(rows) for config, rows in sim_rows.items()
    }
    human_game_summary = {
        config: summarize_by_game(rows) for config, rows in human_rows.items()
    }
    sim_player_summary = {
        config: summarize_by_player(rows) for config, rows in sim_rows.items()
    }
    human_player_summary = {
        config: summarize_by_player(rows) for config, rows in human_rows.items()
    }
    sim_round_summary = {
        config: summarize_by_round(rows) for config, rows in sim_rows.items()
    }
    human_round_summary = {
        config: summarize_by_round(rows) for config, rows in human_rows.items()
    }

    for config, summary in sim_game_summary.items():
        write_game_summary(output_dir / f"sim_game_summary_{config}.csv", summary)
    for config, summary in human_game_summary.items():
        write_game_summary(output_dir / f"human_game_summary_{config}.csv", summary)
    for config, summary in sim_player_summary.items():
        write_player_summary(output_dir / f"sim_player_summary_{config}.csv", summary)
    for config, summary in human_player_summary.items():
        write_player_summary(output_dir / f"human_player_summary_{config}.csv", summary)
    for config, summary in sim_round_summary.items():
        write_round_summary(output_dir / f"sim_round_summary_{config}.csv", summary)
    for config, summary in human_round_summary.items():
        write_round_summary(output_dir / f"human_round_summary_{config}.csv", summary)

    metrics = [
        "mean_contribution_rate",
        "mean_payoff",
        "punishment_rate",
        "reward_rate",
        "normalized_efficiency",
    ]
    alignment_rows, metric_summaries = compare_configs(
        sim_game_summary, human_game_summary, metrics
    )

    write_alignment(output_dir / "alignment_game_summary.csv", alignment_rows)
    write_metric_summary(output_dir / "alignment_metric_summary.csv", metric_summaries)

    plot_noise_ceiling(output_dir, alignment_rows)

    manifest = {
        "timestamp": timestamp,
        "output_root": str(args.output_root),
        "analysis_output_root": str(args.analysis_output_root),
        "human_rounds": str(args.human_rounds),
        "human_configs": str(args.human_configs),
        "metrics": metrics,
        "notes": "Noise ceiling plotted as ±1 SD of human config-level means.",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
