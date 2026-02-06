from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from .aggregation import SummaryRow, summarize_by_game, summarize_by_player
from .comparison import compare_by_config, compare_configs
from .config import (
    DEFAULT_ANALYSIS_OUTPUT_ROOT,
    DEFAULT_HUMAN_CONFIG_CSV,
    DEFAULT_HUMAN_ROUNDS_CSV,
    DEFAULT_OUTPUT_ROOT,
    REPO_ROOT,
)
from .io_utils import (
    GameConfig,
    RowRecord,
    _matches_filters,
    _parse_filter_value,
    _read_csv_rows,
    _safe_parse_dict,
    _sorted_run_dirs,
    load_human_rows,
)
from .plotting import (
    plot_aggregate_metric_means,
    plot_aggregate_metric_rmse,
    plot_aggregate_metric_variance,
)


def _timestamp() -> str:
    return datetime.now().strftime("%y%m%d%H%M")


def _variance_by_config(
    summaries: Dict[str, List[SummaryRow]], metrics: Iterable[str]
) -> Dict[str, Dict[str, float]]:
    variance: Dict[str, Dict[str, float]] = {}
    for config_name, rows in summaries.items():
        for metric in metrics:
            values = [row.metrics.get(metric) for row in rows if row.metrics.get(metric) is not None]
            if not values:
                continue
            mean_value = sum(values) / len(values)
            var = sum((value - mean_value) ** 2 for value in values) / len(values)
            variance.setdefault(config_name, {})[metric] = var
    return variance


def _load_sim_rows(
    output_root: Path,
    include_all_runs: bool,
    config_filters: Dict[str, str],
) -> Dict[str, List[RowRecord]]:
    configs: Dict[str, List[RowRecord]] = {}
    if not output_root.exists():
        return configs
    parsed_filters = {
        key: _parse_filter_value(value) for key, value in (config_filters or {}).items()
    }
    for config_dir in sorted(output_root.iterdir()):
        if not config_dir.is_dir():
            continue
        if not config_dir.name.startswith("VALIDATION_"):
            continue
        run_dirs = _sorted_run_dirs(config_dir)
        if not run_dirs:
            continue
        selected_runs: List[Path] = []
        if include_all_runs:
            for run_dir in run_dirs:
                config_path = run_dir / "config.json"
                participants_path = run_dir / "participant_sim.csv"
                if not config_path.exists() or not participants_path.exists():
                    continue
                config_payload = json.loads(config_path.read_text(encoding="utf-8"))
                if parsed_filters and not _matches_filters(config_payload, parsed_filters):
                    continue
                selected_runs.append(run_dir)
        else:
            for run_dir in reversed(run_dirs):
                config_path = run_dir / "config.json"
                participants_path = run_dir / "participant_sim.csv"
                if not config_path.exists() or not participants_path.exists():
                    continue
                config_payload = json.loads(config_path.read_text(encoding="utf-8"))
                if parsed_filters and not _matches_filters(config_payload, parsed_filters):
                    continue
                selected_runs = [run_dir]
                break
        if not selected_runs:
            continue
        for run_dir in selected_runs:
            config_path = run_dir / "config.json"
            participants_path = run_dir / "participant_sim.csv"
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            environment = config_payload.get("environment", {})
            config = GameConfig(name=config_dir.name, environment=environment)
            rows = configs.setdefault(config_dir.name, [])
            run_prefix = f"{run_dir.name}-" if include_all_runs else ""
            for row in _read_csv_rows(participants_path):
                rows.append(
                    RowRecord(
                        game_id=f"{run_prefix}{row.get('gameId', '')}",
                        player_id=row.get("playerAvatar", ""),
                        round_index=int(float(row.get("roundIndex") or 0)),
                        contribution=float(row.get("data.contribution") or 0.0),
                        punished=_safe_parse_dict(row.get("data.punished", "")),
                        rewarded=_safe_parse_dict(row.get("data.rewarded", "")),
                        config=config,
                    )
                )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple simulation output roots against human data."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Path to baseline simulation outputs.",
    )
    parser.add_argument(
        "--output-persona-root",
        type=Path,
        default=REPO_ROOT / "output_persona",
        help="Path to persona (full transcript) outputs.",
    )
    parser.add_argument(
        "--output-persona-summary-root",
        type=Path,
        default=REPO_ROOT / "output_persona_summary",
        help="Path to persona (LLM summary) outputs.",
    )
    parser.add_argument(
        "--output-persona-summary-finetuned-root",
        type=Path,
        default=REPO_ROOT / "output_persona_summary_finetuned",
        help="Path to persona (finetuned summary) outputs.",
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

    filter_pairs = dict(item.split("=", 1) for item in args.sim_filter if "=" in item)

    model_specs = [
        (
            "no_reasoning",
            "no reasoning, no persona",
            args.output_root,
            False,
        ),
        (
            "reasoning",
            "reasoning, no persona",
            args.output_root,
            True,
        ),
        (
            "persona_full",
            "reasoning, randomly sampled full transcript as persona",
            args.output_persona_root,
            True,
        ),
        (
            "persona_summary",
            "reasoning, randomly sampled LLM summary as persona",
            args.output_persona_summary_root,
            True,
        ),
        (
            "persona_summary_finetuned",
            "reasoning, optimized LLM summary as persona",
            args.output_persona_summary_finetuned_root,
            True,
        ),
    ]

    sim_rows_by_model: Dict[str, Dict[str, List]] = {}
    for model_key, _, output_root, include_reasoning in model_specs:
        model_filters = dict(filter_pairs)
        model_filters["model.include_reasoning"] = include_reasoning
        rows = _load_sim_rows(
            output_root,
            include_all_runs=args.sim_include_all,
            config_filters=model_filters,
        )
        if rows:
            sim_rows_by_model[model_key] = rows

    human_rows = load_human_rows(args.human_rounds, args.human_configs)

    timestamp = _timestamp()
    output_dir = args.analysis_output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_game_summary_by_model = {
        model_key: {config: summarize_by_game(rows) for config, rows in sim_rows.items()}
        for model_key, sim_rows in sim_rows_by_model.items()
    }
    sim_player_summary_by_model = {
        model_key: {config: summarize_by_player(rows) for config, rows in sim_rows.items()}
        for model_key, sim_rows in sim_rows_by_model.items()
    }
    human_game_summary = {
        config: summarize_by_game(rows) for config, rows in human_rows.items()
    }
    human_player_summary = {
        config: summarize_by_player(rows) for config, rows in human_rows.items()
    }

    core_metrics = [
        "mean_contribution_rate",
        "punishment_rate",
        "reward_rate",
        "normalized_efficiency",
    ]
    variance_metrics = [
        "mean_contribution_rate",
        "punishment_rate",
        "reward_rate",
    ]

    alignment_rows_by_model = {}
    config_metric_summaries_by_model = {}
    for model_key, sim_game_summary in sim_game_summary_by_model.items():
        alignment_rows, _ = compare_configs(
            sim_game_summary, human_game_summary, core_metrics
        )
        config_metric_summaries = compare_by_config(
            sim_game_summary, human_game_summary, core_metrics
        )
        if alignment_rows:
            alignment_rows_by_model[model_key] = alignment_rows
        if config_metric_summaries:
            config_metric_summaries_by_model[model_key] = config_metric_summaries

    variance_by_model = {
        model_key: _variance_by_config(sim_summary, variance_metrics)
        for model_key, sim_summary in sim_player_summary_by_model.items()
    }
    human_variance = _variance_by_config(human_player_summary, variance_metrics)

    model_labels = {key: label for key, label, _, _ in model_specs}

    plot_aggregate_metric_rmse(
        output_dir, config_metric_summaries_by_model, model_labels
    )
    plot_aggregate_metric_means(
        output_dir, alignment_rows_by_model, model_labels, core_metrics
    )
    plot_aggregate_metric_variance(
        output_dir, variance_by_model, human_variance, model_labels, variance_metrics
    )

    manifest = {
        "timestamp": timestamp,
        "analysis_output_root": str(args.analysis_output_root),
        "human_rounds": str(args.human_rounds),
        "human_configs": str(args.human_configs),
        "sim_include_all": args.sim_include_all,
        "sim_filters": filter_pairs,
        "models": [
            {
                "key": key,
                "label": label,
                "output_root": str(output_root),
                "include_reasoning": include_reasoning,
                "included": key in sim_rows_by_model,
            }
            for key, label, output_root, include_reasoning in model_specs
        ],
        "metrics": core_metrics,
        "variance_metrics": variance_metrics,
        "outputs": [
            "aggregate_rmse_by_metric.png",
            "aggregate_metric_means.png",
            "aggregate_metric_variance.png",
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
