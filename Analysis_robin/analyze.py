from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

from .aggregation import (
    summarize_by_game,
    summarize_by_game_with_config,
    summarize_by_player,
    summarize_by_round,
    write_game_summary,
    write_player_summary,
    write_round_summary,
)
from .comparison import (
    compare_by_config,
    compare_configs,
    write_alignment,
    write_config_metric_summary,
    write_metric_summary,
)
from .config import (
    DEFAULT_ANALYSIS_OUTPUT_ROOT,
    DEFAULT_HUMAN_CONFIG_CSV,
    DEFAULT_HUMAN_ROUNDS_CSV,
    DEFAULT_OUTPUT_ROOT,
)
from .io_utils import load_human_rows, load_simulation_runs
from .plotting import (
    plot_aggregate_metric_rmse,
    plot_aggregate_metric_means,
    plot_config_metric_rmse,
    plot_metric_variance_by_config,
    plot_metric_means_by_config,
    plot_metrics_by_binary_config,
)


def _timestamp() -> str:
    return datetime.now().strftime("%y%m%d%H%M")


def _humanize_config_key(config_key: str) -> str:
    label = config_key.replace("CONFIG_", "")
    label = label.replace("_", " ")
    label = re.sub(r"([a-z])([A-Z])", r"\1 \2", label)
    return label.strip().title()


def _binarize_value(config_key: str, value: object) -> Tuple[bool, bool]:
    if config_key == "CONFIG_defaultContribProp":
        try:
            return True, float(value) > 0
        except (TypeError, ValueError):
            return False, False
    if isinstance(value, bool):
        return True, value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False, False
    if numeric in (0.0, 1.0):
        return True, bool(numeric)
    return False, False


def _config_value_label(config_key: str, value: bool) -> str:
    if config_key == "CONFIG_chat":
        return "Communication" if value else "No communication"
    if config_key == "CONFIG_allOrNothing":
        return "Binary contribution" if value else "Continuous contribution"
    if config_key == "CONFIG_defaultContribProp":
        return "Opt-out" if value else "Opt-in"
    if config_key == "CONFIG_showOtherSummaries":
        return "Peer outcomes visible" if value else "Peer outcomes hidden"
    if config_key == "CONFIG_showPunishmentId":
        return "Punisher identity revealed" if value else "Punisher identity hidden"
    if config_key == "CONFIG_showNRounds":
        return "Known horizon" if value else "Unknown horizon"
    if config_key == "CONFIG_rewardExists":
        return "Reward exists" if value else "No rewards"
    if config_key == "CONFIG_punishmentExists":
        return "Punishment exists" if value else "No punishment"
    return "Yes" if value else "No"


def _config_title(config_key: str) -> str:
    if config_key == "CONFIG_chat":
        return "Communication"
    if config_key == "CONFIG_allOrNothing":
        return "Contribution mode"
    if config_key == "CONFIG_defaultContribProp":
        return "Default contribution"
    if config_key == "CONFIG_showOtherSummaries":
        return "Peer outcome visibility"
    if config_key == "CONFIG_showPunishmentId":
        return "Punisher identity revealed"
    if config_key == "CONFIG_showNRounds":
        return "Horizon knowledge"
    if config_key == "CONFIG_rewardExists":
        return "Reward exists"
    if config_key == "CONFIG_punishmentExists":
        return "Punishment exists"
    return _humanize_config_key(config_key)


def _collect_binary_keys(game_details) -> List[str]:
    values = {}
    for item in game_details:
        for key, value in item.config.environment.items():
            if not key.startswith("CONFIG_"):
                continue
            ok, binary = _binarize_value(key, value)
            if not ok:
                continue
            values.setdefault(key, set()).add(binary)
    return [key for key, vals in sorted(values.items()) if len(vals) == 2]


def _build_binary_config_rows(
    sim_games,
    human_games,
    metrics: Iterable[str],
) -> List[Tuple[str, str, str, str, float, float]]:
    rows: List[Tuple[str, str, str, str, float, float]] = []
    all_games = list(sim_games) + list(human_games)
    binary_keys = _collect_binary_keys(all_games)
    for config_key in binary_keys:
        for metric in metrics:
            for value in (False, True):
                sim_values = []
                for item in sim_games:
                    ok, binary = _binarize_value(
                        config_key, item.config.environment.get(config_key)
                    )
                    if ok and binary == value:
                        metric_value = item.metrics.get(metric)
                        if metric_value is not None:
                            sim_values.append(metric_value)
                human_values = []
                for item in human_games:
                    ok, binary = _binarize_value(
                        config_key, item.config.environment.get(config_key)
                    )
                    if ok and binary == value:
                        metric_value = item.metrics.get(metric)
                        if metric_value is not None:
                            human_values.append(metric_value)
                if not sim_values or not human_values:
                    continue
                sim_mean = sum(sim_values) / len(sim_values)
                human_mean = sum(human_values) / len(human_values)
                rows.append(
                    (
                        config_key,
                        _config_title(config_key),
                        _config_value_label(config_key, value),
                        metric,
                        sim_mean,
                        human_mean,
                    )
                )
    return rows


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
    parser.add_argument(
        "--sim-include-all",
        action="store_true",
        help="Include all simulation runs per config instead of only the latest.",
    )
    parser.add_argument(
        "--sim-filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Filter simulation runs by config.json values. Use dotted paths like "
            "'model.base_model=...'. Can be specified multiple times."
        ),
    )
    args = parser.parse_args()

    filter_pairs = dict(
        item.split("=", 1) for item in args.sim_filter if "=" in item
    )
    sim_rows = load_simulation_runs(
        args.output_root,
        include_all_runs=args.sim_include_all,
        config_filters=filter_pairs,
    )
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
        "punishment_rate",
        "reward_rate",
        "normalized_efficiency",
    ]
    alignment_rows, metric_summaries = compare_configs(
        sim_game_summary, human_game_summary, metrics
    )
    config_metric_summaries = compare_by_config(
        sim_game_summary, human_game_summary, metrics
    )

    write_alignment(output_dir / "alignment_game_summary.csv", alignment_rows)
    write_metric_summary(output_dir / "alignment_metric_summary.csv", metric_summaries)
    write_config_metric_summary(
        output_dir / "alignment_config_metric_summary.csv", config_metric_summaries
    )

    plot_config_metric_rmse(output_dir, config_metric_summaries)
    plot_aggregate_metric_rmse(output_dir, config_metric_summaries)
    plot_metric_means_by_config(output_dir, alignment_rows)
    plot_aggregate_metric_means(output_dir, alignment_rows)

    variance_metrics = [
        "mean_contribution_rate",
        "punishment_rate",
        "reward_rate",
        "mean_payoff",
    ]
    variance_rows = []
    for config_name, sim_summary in sim_player_summary.items():
        human_summary = human_player_summary.get(config_name)
        if not human_summary:
            continue
        for metric in variance_metrics:
            sim_values = [row.metrics.get(metric) for row in sim_summary]
            human_values = [row.metrics.get(metric) for row in human_summary]
            sim_values = [value for value in sim_values if value is not None]
            human_values = [value for value in human_values if value is not None]
            if not sim_values or not human_values:
                continue
            sim_mean = sum(sim_values) / len(sim_values)
            human_mean = sum(human_values) / len(human_values)
            sim_var = sum((value - sim_mean) ** 2 for value in sim_values) / len(sim_values)
            human_var = (
                sum((value - human_mean) ** 2 for value in human_values) / len(human_values)
            )
            variance_rows.append((config_name, metric, sim_var, human_var))
    plot_metric_variance_by_config(output_dir, variance_rows)

    sim_game_details = [
        item
        for rows in sim_rows.values()
        for item in summarize_by_game_with_config(rows)
    ]
    human_game_details = [
        item
        for rows in human_rows.values()
        for item in summarize_by_game_with_config(rows)
    ]
    binary_rows = _build_binary_config_rows(
        sim_game_details,
        human_game_details,
        variance_metrics,
    )
    plot_metrics_by_binary_config(output_dir, binary_rows)

    manifest = {
        "timestamp": timestamp,
        "output_root": str(args.output_root),
        "analysis_output_root": str(args.analysis_output_root),
        "human_rounds": str(args.human_rounds),
        "human_configs": str(args.human_configs),
        "sim_include_all": args.sim_include_all,
        "sim_filters": filter_pairs,
        "metrics": metrics,
        "variance_metrics": variance_metrics,
        "notes": (
            "RMSE plotted with human standard deviation as the noise ceiling. "
            "Aggregate RMSE uses bootstrap standard deviation across configs for error bars. "
            "Metric mean plots compare simulation vs human means with human std error bars."
        ),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
