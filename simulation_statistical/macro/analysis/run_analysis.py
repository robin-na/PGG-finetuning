from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)
for path in (SCRIPT_DIR, PACKAGE_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from Macro_simulation_eval.analysis.run_analysis import (  # noqa: E402
    DEFAULT_BINARY_FACTORS,
    DEFAULT_MEDIAN_FACTORS,
    _extract_analysis_csv_from_config,
    _load_run_config,
    _resolve_existing_path,
    _resolve_run_eval_csv,
    _split_csv_arg,
    _timestamp_id,
    build_directional_rows,
    compute_sim_game_metrics,
    load_human_game_table,
    maybe_plot_directional_effects,
)
from supplemental import (  # noqa: E402
    DIRECT_BASELINE_SPECS,
    LINEAR_CONFIG_FEATURES,
    build_direct_baseline_benchmark_table,
    build_direct_baseline_game_table,
    build_linear_config_baseline_benchmark_table,
    build_linear_config_baseline_game_table,
    build_macro_metric_bundle,
    build_noise_ceiling_tables,
    build_sim_benchmark_game_table,
    build_directional_rows_with_baseline,
    build_directional_sign_summary,
    extract_rounds_csv_from_config,
    plot_aggregate_efficiency_comparison,
    plot_directional_effects_with_baseline,
    plot_game_level_scatter,
    plot_rmse_summary,
    plot_variance_bars,
    update_manifest,
)
from simulation_statistical.paths import BENCHMARK_DATA_ROOT, MACRO_REPORT_ROOT, MACRO_RUN_ROOT  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze statistical macro simulation outputs.")
    parser.add_argument("--eval_csv", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument(
        "--eval_root",
        type=str,
        default=MACRO_RUN_ROOT,
    )
    parser.add_argument("--compare_run_ids", type=str, default=None)
    parser.add_argument("--compare_labels", type=str, default=None)
    parser.add_argument(
        "--analysis_root",
        type=str,
        default=MACRO_REPORT_ROOT,
    )
    parser.add_argument("--analysis_run_id", type=str, default=None)
    parser.add_argument("--human_analysis_csv", type=str, default=None)
    parser.add_argument("--human_rows_csv", type=str, default=None)
    parser.add_argument("--binary_factors", type=str, default=",".join(DEFAULT_BINARY_FACTORS))
    parser.add_argument("--median_factors", type=str, default=",".join(DEFAULT_MEDIAN_FACTORS))
    parser.add_argument("--shared_games_only", action="store_true")
    parser.add_argument("--aggregate_by_benchmark", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    compare_run_ids = _split_csv_arg(args.compare_run_ids)
    compare_labels = _split_csv_arg(args.compare_labels)
    if compare_run_ids:
        run_ids = compare_run_ids
        if compare_labels and len(compare_labels) != len(compare_run_ids):
            raise ValueError("If provided, --compare_labels must match --compare_run_ids length.")
        labels = compare_labels if compare_labels else compare_run_ids
    else:
        if not args.run_id and not args.eval_csv:
            raise ValueError("Provide --eval_csv or --run_id (or --compare_run_ids).")
        if args.eval_csv and not args.run_id:
            run_ids = ["adhoc_eval_csv"]
            labels = ["adhoc_eval_csv"]
        else:
            run_ids = [args.run_id]
            labels = [args.run_id]

    binary_factors = _split_csv_arg(args.binary_factors)
    median_factors = _split_csv_arg(args.median_factors)

    analysis_root = Path(args.analysis_root).resolve()
    analysis_run_id = args.analysis_run_id or _timestamp_id()
    out_dir = analysis_root / analysis_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run_records: List[Dict[str, Any]] = []
    game_tables: Dict[str, pd.DataFrame] = {}
    human_csv_by_run: Dict[str, Path] = {}
    human_rows_csv_by_run: Dict[str, Path] = {}
    eval_csv_by_run: Dict[str, Path] = {}
    unit_key = "benchmark_id" if args.aggregate_by_benchmark else "gameId"

    for index, run_id in enumerate(run_ids):
        if args.eval_csv and run_id == "adhoc_eval_csv":
            rows_csv = Path(args.eval_csv).resolve()
            run_cfg: Dict[str, Any] = {}
        else:
            rows_csv = _resolve_run_eval_csv(run_id, eval_root=args.eval_root)
            run_cfg = _load_run_config(run_id, eval_root=args.eval_root)

        if not rows_csv.exists():
            raise FileNotFoundError(f"Macro eval CSV not found: {rows_csv}")
        eval_csv_by_run[run_id] = rows_csv

        if args.human_analysis_csv:
            human_csv = Path(args.human_analysis_csv).resolve()
        else:
            human_csv = _extract_analysis_csv_from_config(run_cfg, eval_root=args.eval_root, run_id=run_id)
            if human_csv is None:
                raise ValueError(
                    f"Could not infer human analysis CSV for run '{run_id}'. Pass --human_analysis_csv explicitly."
                )
        if not human_csv.exists():
            raise FileNotFoundError(f"Human analysis CSV not found: {human_csv}")

        if args.human_rows_csv:
            human_rows_csv = Path(args.human_rows_csv).resolve()
        else:
            human_rows_csv = extract_rounds_csv_from_config(
                run_id if run_id != "adhoc_eval_csv" else None,
                args.eval_root,
            )
            if human_rows_csv is None and args.eval_csv:
                fallback = _resolve_existing_path(
                    os.path.join(BENCHMARK_DATA_ROOT, "raw_data", "validation_wave", "player-rounds.csv")
                )
                human_rows_csv = fallback if isinstance(fallback, Path) else None
        if human_rows_csv is None or not human_rows_csv.exists():
            raise FileNotFoundError(
                f"Human rounds CSV not found for run '{run_id}'. Pass --human_rows_csv explicitly."
            )

        human_csv_by_run[run_id] = human_csv
        human_rows_csv_by_run[run_id] = human_rows_csv
        if args.aggregate_by_benchmark:
            merged = build_sim_benchmark_game_table(
                sim_rows_csv=rows_csv,
                human_analysis_csv=human_csv,
                human_rows_csv=human_rows_csv,
                binary_factors=binary_factors,
                median_factors=median_factors,
                run_id=run_id,
                label=labels[index],
            )
            n_simulated = int(pd.to_numeric(merged.get("sim_games_in_benchmark"), errors="coerce").sum()) if not merged.empty else 0
        else:
            human = load_human_game_table(human_csv, binary_factors=binary_factors, median_factors=median_factors)
            sim = compute_sim_game_metrics(rows_csv, human_cfg=human)
            merged = human.merge(sim, on="gameId", how="inner")
            merged["run_id"] = run_id
            merged["label"] = labels[index]
            n_simulated = int(sim["gameId"].nunique())
        game_tables[run_id] = merged

        run_records.append(
            {
                "run_id": run_id,
                "label": labels[index],
                "eval_csv": str(rows_csv),
                "human_analysis_csv": str(human_csv),
                "human_rows_csv": str(human_rows_csv),
                "n_games_simulated": n_simulated,
                "n_games_after_merge": int(len(merged)),
            }
        )

    shared_games: Optional[set[str]] = None
    if args.shared_games_only and len(game_tables) > 1:
        for table in game_tables.values():
            game_ids = set(table[unit_key].astype(str).tolist())
            shared_games = game_ids if shared_games is None else (shared_games & game_ids)
        shared_games = shared_games or set()

    direct_baseline_tables: List[Dict[str, Any]] = []
    direct_baseline_summaries: List[pd.DataFrame] = []
    noise_ceiling_summary_df = pd.DataFrame()
    noise_ceiling_detail_df = pd.DataFrame()
    unique_human_csvs = {str(path.resolve()) for path in human_csv_by_run.values()}
    unique_human_rows = {str(path.resolve()) for path in human_rows_csv_by_run.values()}
    if len(run_ids) > 1 and len(unique_human_csvs) == 1 and len(unique_human_rows) == 1:
        baseline_human_csv = next(iter(human_csv_by_run.values()))
        baseline_human_rows = next(iter(human_rows_csv_by_run.values()))
        noise_ceiling_summary_df, noise_ceiling_detail_df = build_noise_ceiling_tables(
            human_analysis_csv=baseline_human_csv,
            human_rows_csv=baseline_human_rows,
        )
        for spec in DIRECT_BASELINE_SPECS:
            if args.aggregate_by_benchmark:
                baseline_df, baseline_summary = build_direct_baseline_benchmark_table(
                    human_analysis_csv=baseline_human_csv,
                    human_rows_csv=baseline_human_rows,
                    feature_source=spec["feature_source"],
                    model_kind=spec["model_kind"],
                    run_id=spec["run_id"],
                    label=spec["label"],
                )
            else:
                baseline_df, baseline_summary = build_direct_baseline_game_table(
                    human_analysis_csv=baseline_human_csv,
                    human_rows_csv=baseline_human_rows,
                    binary_factors=binary_factors,
                    median_factors=median_factors,
                    feature_source=spec["feature_source"],
                    model_kind=spec["model_kind"],
                    run_id=spec["run_id"],
                    label=spec["label"],
                )
            if shared_games is not None and not baseline_df.empty:
                baseline_df = baseline_df[
                    baseline_df[unit_key].astype(str).isin(shared_games)
                ].copy()
            if baseline_df.empty:
                continue
            direct_baseline_tables.append(
                {
                    "run_id": spec["run_id"],
                    "label": spec["label"],
                    "table": baseline_df,
                }
            )
            if not baseline_summary.empty:
                direct_baseline_summaries.append(baseline_summary.copy())
            run_records.append(
                {
                    "run_id": spec["run_id"],
                    "label": spec["label"],
                    "eval_csv": "",
                    "human_analysis_csv": str(baseline_human_csv),
                    "human_rows_csv": str(baseline_human_rows),
                    "n_games_simulated": int(
                        pd.to_numeric(
                            baseline_df.get("sim_games_in_benchmark", baseline_df[unit_key]),
                            errors="coerce",
                        ).sum()
                    )
                    if args.aggregate_by_benchmark
                    else int(baseline_df["gameId"].nunique()),
                    "n_games_after_merge": int(len(baseline_df)),
                }
            )

    game_level_frames: List[pd.DataFrame] = []
    aggregate_rows: List[Dict[str, Any]] = []
    directional_frames: List[pd.DataFrame] = []

    for run_id, label in zip(run_ids, labels):
        merged = game_tables[run_id].copy()
        if shared_games is not None:
            merged = merged[merged[unit_key].astype(str).isin(shared_games)].copy()
        game_level_frames.append(merged)

        abs_err = (merged["sim_normalized_efficiency"] - merged["human_normalized_efficiency"]).abs()
        sq_err = (merged["sim_normalized_efficiency"] - merged["human_normalized_efficiency"]) ** 2
        aggregate_rows.append(
            {
                "run_id": run_id,
                "label": label,
                "n_games": int(len(merged)),
                "mae_normalized_efficiency": float(abs_err.mean()) if len(abs_err) else None,
                "rmse_normalized_efficiency": float((sq_err.mean()) ** 0.5) if len(sq_err) else None,
                "corr_normalized_efficiency": float(
                    merged["sim_normalized_efficiency"].corr(merged["human_normalized_efficiency"])
                )
                if len(merged) >= 2
                else None,
                "mean_human_normalized_efficiency": float(merged["human_normalized_efficiency"].mean())
                if len(merged)
                else None,
                "mean_sim_normalized_efficiency": float(merged["sim_normalized_efficiency"].mean())
                if len(merged)
                else None,
            }
        )

        directional_frames.append(
            build_directional_rows(
                merged=merged,
                run_id=run_id,
                label=label,
                binary_factors=binary_factors,
                median_factors=median_factors,
            )
        )

    for baseline_item in direct_baseline_tables:
        baseline_df = baseline_item["table"].copy()
        run_id = baseline_item["run_id"]
        label = baseline_item["label"]
        game_level_frames.append(baseline_df)
        baseline_abs_err = (
            baseline_df["sim_normalized_efficiency"] - baseline_df["human_normalized_efficiency"]
        ).abs()
        baseline_sq_err = (
            baseline_df["sim_normalized_efficiency"] - baseline_df["human_normalized_efficiency"]
        ) ** 2
        aggregate_rows.append(
            {
                "run_id": run_id,
                "label": label,
                "n_games": int(len(baseline_df)),
                "mae_normalized_efficiency": float(baseline_abs_err.mean()) if len(baseline_abs_err) else None,
                "rmse_normalized_efficiency": float((baseline_sq_err.mean()) ** 0.5)
                if len(baseline_sq_err)
                else None,
                "corr_normalized_efficiency": float(
                    baseline_df["sim_normalized_efficiency"].corr(baseline_df["human_normalized_efficiency"])
                )
                if len(baseline_df) >= 2
                else None,
                "mean_human_normalized_efficiency": float(baseline_df["human_normalized_efficiency"].mean())
                if len(baseline_df)
                else None,
                "mean_sim_normalized_efficiency": float(baseline_df["sim_normalized_efficiency"].mean())
                if len(baseline_df)
                else None,
            }
        )
        directional_frames.append(
            build_directional_rows(
                merged=baseline_df,
                run_id=run_id,
                label=label,
                binary_factors=binary_factors,
                median_factors=median_factors,
            )
        )

    selected_df = pd.DataFrame(run_records)
    aggregate_df = pd.DataFrame(aggregate_rows)
    game_level_df = pd.concat(game_level_frames, ignore_index=True) if game_level_frames else pd.DataFrame()
    directional_df = pd.concat(directional_frames, ignore_index=True) if directional_frames else pd.DataFrame()

    directional_summary_rows: List[Dict[str, Any]] = []
    if not directional_df.empty:
        for (run_id, label), group in directional_df.groupby(["run_id", "label"], sort=False):
            nonzero_mask = group["human_sign"] != 0
            directional_summary_rows.append(
                {
                    "run_id": run_id,
                    "label": label,
                    "n_factors_evaluated": int(len(group)),
                    "sign_match_rate_all": float(group["sign_match"].mean()),
                    "sign_match_rate_nonzero_human": float(group.loc[nonzero_mask, "sign_match"].mean())
                    if nonzero_mask.any()
                    else None,
                }
            )
    directional_summary_df = pd.DataFrame(directional_summary_rows)

    selected_df.to_csv(out_dir / "selected_runs.csv", index=False)
    aggregate_df.to_csv(out_dir / "aggregate_efficiency_metrics.csv", index=False)
    game_level_df.to_csv(out_dir / "game_level_metrics.csv", index=False)
    directional_df.to_csv(out_dir / "directional_effects.csv", index=False)
    directional_summary_df.to_csv(out_dir / "directional_sign_summary.csv", index=False)
    direct_baseline_summary_df = (
        pd.concat(direct_baseline_summaries, ignore_index=True) if direct_baseline_summaries else pd.DataFrame()
    )
    if not direct_baseline_summary_df.empty:
        direct_baseline_summary_df.to_csv(out_dir / "direct_regression_baseline_summaries.csv", index=False)
        linear_subset = direct_baseline_summary_df[
            direct_baseline_summary_df["run_id"] == "linear_config_baseline"
        ].copy()
        if not linear_subset.empty:
            linear_subset.to_csv(out_dir / "linear_config_baseline_summary.csv", index=False)
    if not noise_ceiling_summary_df.empty:
        noise_ceiling_summary_df.to_csv(out_dir / "noise_ceiling_summary.csv", index=False)
    if not noise_ceiling_detail_df.empty:
        noise_ceiling_detail_df.to_csv(out_dir / "noise_ceiling_by_benchmark.csv", index=False)

    plot_paths = None
    if not args.no_plots:
        plot_paths = maybe_plot_directional_effects(
            directional=directional_df,
            plot_dir=out_dir / "figures",
            dpi=args.dpi,
        )
        if len(run_ids) > 1:
            aggregate_fig = plot_aggregate_efficiency_comparison(
                aggregate_df=aggregate_df,
                out_path=out_dir / "figures" / "aggregate_efficiency_comparison.png",
                dpi=args.dpi,
            )
            if aggregate_fig:
                plot_paths = list(plot_paths or [])
                plot_paths.append(aggregate_fig)

    manifest = {
        "analysis_run_id": analysis_run_id,
        "analysis_root": str(analysis_root),
        "eval_root": str(Path(args.eval_root).resolve()),
        "run_ids": run_ids,
        "labels": labels,
        "shared_games_only": bool(args.shared_games_only),
        "shared_games_count": int(len(shared_games)) if shared_games is not None else None,
        "aggregate_by_benchmark": bool(args.aggregate_by_benchmark),
        "human_csv_by_run": {run_id: str(path) for run_id, path in human_csv_by_run.items()},
        "binary_factors": binary_factors,
        "median_factors": median_factors,
        "plots": [str(path) for path in plot_paths] if plot_paths else [],
        "linear_config_baseline_available": any(
            item["run_id"] == "linear_config_baseline" for item in direct_baseline_tables
        ),
        "direct_regression_baselines_available": [item["run_id"] for item in direct_baseline_tables],
        "noise_ceiling_available": bool(not noise_ceiling_summary_df.empty),
    }
    manifest_path = out_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if len(run_ids) == 1:
        run_id = run_ids[0]
        bundle = build_macro_metric_bundle(
            sim_rows_csv=eval_csv_by_run[run_id],
            human_rows_csv=human_rows_csv_by_run[run_id],
            human_analysis_csv=human_csv_by_run[run_id],
            game_level_df=game_tables[run_id].copy(),
        )
        enriched_game_level = bundle["enriched_game_level"]
        rmse_summary = bundle["rmse_summary"]
        variance_summary = bundle["variance_summary"]
        player_variance_by_game = bundle["player_variance_by_game"]
        linear_summary = bundle["linear_summary"]
        noise_ceiling_summary = bundle["noise_ceiling_summary"]
        noise_ceiling_detail = bundle["noise_ceiling_detail"]

        game_level_path = out_dir / "game_level_metrics.csv"
        rmse_path = out_dir / "macro_rmse_summary.csv"
        variance_path = out_dir / "macro_variance_summary.csv"
        player_var_path = out_dir / "macro_player_variance_by_game.csv"
        linear_summary_path = out_dir / "linear_config_baseline_summary.csv"
        linear_meta_path = out_dir / "linear_config_baseline_metadata.json"
        noise_ceiling_summary_path = out_dir / "noise_ceiling_summary.csv"
        noise_ceiling_detail_path = out_dir / "noise_ceiling_by_benchmark.csv"

        enriched_game_level.to_csv(game_level_path, index=False)
        rmse_summary.to_csv(rmse_path, index=False)
        variance_summary.to_csv(variance_path, index=False)
        player_variance_by_game.to_csv(player_var_path, index=False)
        linear_summary.to_csv(linear_summary_path, index=False)
        noise_ceiling_summary.to_csv(noise_ceiling_summary_path, index=False)
        noise_ceiling_detail.to_csv(noise_ceiling_detail_path, index=False)
        linear_meta_path.write_text(
            json.dumps(
                {
                    "feature_cols": LINEAR_CONFIG_FEATURES,
                    "n_feature_cols": len(LINEAR_CONFIG_FEATURES),
                    "learn_analysis_csv": bundle["learn_analysis_csv"],
                    "learn_rows_csv": bundle["learn_rows_csv"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        supplemental_files = [
            str(game_level_path),
            str(rmse_path),
            str(variance_path),
            str(player_var_path),
            str(linear_summary_path),
            str(linear_meta_path),
        ]
        directional_with_linear = pd.DataFrame()
        if "linear_normalized_efficiency" in enriched_game_level.columns:
            directional_with_linear = build_directional_rows_with_baseline(
                merged=enriched_game_level,
                factors=list(dict.fromkeys(binary_factors + median_factors)),
            )
            directional_summary_with_linear = build_directional_sign_summary(directional_with_linear)
            directional_path = out_dir / "directional_effects.csv"
            directional_summary_path = out_dir / "directional_sign_summary.csv"
            directional_with_linear.to_csv(directional_path, index=False)
            directional_summary_with_linear.to_csv(directional_summary_path, index=False)
            supplemental_files.extend([str(directional_path), str(directional_summary_path)])

        if not args.no_plots:
            figures_dir = out_dir / "figures"
            rmse_fig = plot_rmse_summary(rmse_summary, figures_dir / "macro_rmse_by_target.png", dpi=args.dpi)
            player_var_fig = plot_variance_bars(
                variance_summary,
                figures_dir / "macro_variance_across_players.png",
                title="Within-benchmark Player Variance",
                metric_specs=[
                    (
                        "mean_var_players_contrib_rate_sim",
                        "mean_var_players_contrib_rate_linear",
                        "mean_var_players_contrib_rate_human",
                        "contribution",
                    ),
                    (
                        "mean_var_players_punishment_rate_sim",
                        "mean_var_players_punishment_rate_linear",
                        "mean_var_players_punishment_rate_human",
                        "punishment_rate",
                    ),
                    (
                        "mean_var_players_reward_rate_sim",
                        "mean_var_players_reward_rate_linear",
                        "mean_var_players_reward_rate_human",
                        "reward_rate",
                    ),
                ],
                dpi=args.dpi,
            )
            game_var_fig = plot_variance_bars(
                variance_summary,
                figures_dir / "macro_variance_across_games.png",
                title="Across-benchmark Variance of Benchmark-level Means",
                metric_specs=[
                    (
                        "var_across_games_contrib_rate_sim",
                        "var_across_games_contrib_rate_linear",
                        "var_across_games_contrib_rate_human",
                        "contribution",
                    ),
                    (
                        "var_across_games_punishment_rate_sim",
                        "var_across_games_punishment_rate_linear",
                        "var_across_games_punishment_rate_human",
                        "punishment_rate",
                    ),
                    (
                        "var_across_games_reward_rate_sim",
                        "var_across_games_reward_rate_linear",
                        "var_across_games_reward_rate_human",
                        "reward_rate",
                    ),
                    (
                        "var_across_games_normalized_efficiency_sim",
                        "var_across_games_normalized_efficiency_linear",
                        "var_across_games_normalized_efficiency_human",
                        "normalized_efficiency",
                    ),
                ],
                dpi=args.dpi,
            )
            scatter_fig = plot_game_level_scatter(
                enriched_game_level,
                figures_dir / "macro_game_level_scatter_by_target.png",
                dpi=args.dpi,
            )
            directional_fig = None
            if not directional_with_linear.empty:
                safe_run_id = run_id.replace("/", "__")
                directional_fig = plot_directional_effects_with_baseline(
                    directional_with_linear,
                    figures_dir / f"directional_effects_{safe_run_id}.png",
                    label=run_id,
                    dpi=args.dpi,
                )
            for candidate in (rmse_fig, player_var_fig, game_var_fig, scatter_fig, directional_fig):
                if candidate:
                    supplemental_files.append(candidate)

        update_manifest(
            manifest_path,
            {
                "name": "single_run_rmse_variance_report",
                "run_id": run_id,
                "linear_config_features": LINEAR_CONFIG_FEATURES,
                "generated_files": supplemental_files,
            },
        )

    print(f"Wrote statistical macro analysis -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
