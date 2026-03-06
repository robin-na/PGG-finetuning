from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from archetype_augmented_regression.io_utils import json_safe


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%SZ")


def parse_ood_batch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run eval_style_augmented_regression.py over wave-anchored OOD splits."
    )
    parser.add_argument(
        "--ood-root",
        type=Path,
        default=Path("benchmark/data_ood_splits_wave_anchored"),
        help="Root containing <factor>/<direction>/processed_data split folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/archetype_augmented_regression/ood_wave_anchored"),
        help="Root for run outputs (writes into runs/<run_id>/...).",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional explicit run id.")
    parser.add_argument("--target", type=str, default="itt_relative_efficiency")
    parser.add_argument("--style-source", choices=["oracle", "synthetic", "both"], default="both")
    parser.add_argument("--eval-granularity", choices=["game", "config_treatment", "both"], default="both")
    parser.add_argument("--group-col", type=str, default="CONFIG_treatmentName")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--style-ridge-alpha", type=float, default=1.0)
    parser.add_argument("--style-oof-folds", type=int, default=5)
    parser.add_argument(
        "--cluster-counts",
        type=int,
        nargs="+",
        default=[4, 6, 8, 10, 12, 16],
    )
    parser.add_argument(
        "--max-splits",
        type=int,
        default=0,
        help="If > 0, process only the first N splits (useful for smoke tests).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a split if both summary.json and results.csv already exist in run dir.",
    )
    return parser.parse_args()


def find_splits(ood_root: Path) -> list[Path]:
    return sorted([p for p in ood_root.glob("*/*") if p.is_dir()])


def run_one_split(split_dir: Path, out_dir: Path, args: argparse.Namespace) -> subprocess.CompletedProcess[str]:
    learn_csv = split_dir / "processed_data" / "df_analysis_learn.csv"
    val_csv = split_dir / "processed_data" / "df_analysis_val.csv"
    out_csv = out_dir / "results.csv"
    out_json = out_dir / "summary.json"

    cmd = [
        sys.executable,
        "-m",
        "archetype_augmented_regression.eval_style_augmented_regression",
        "--learn-analysis-csv",
        str(learn_csv),
        "--val-analysis-csv",
        str(val_csv),
        "--target",
        args.target,
        "--style-source",
        args.style_source,
        "--eval-granularity",
        args.eval_granularity,
        "--group-col",
        args.group_col,
        "--ridge-alpha",
        str(args.ridge_alpha),
        "--style-ridge-alpha",
        str(args.style_ridge_alpha),
        "--style-oof-folds",
        str(args.style_oof_folds),
        "--cluster-counts",
        *[str(v) for v in args.cluster_counts],
        "--output-csv",
        str(out_csv),
        "--output-json",
        str(out_json),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def extract_best_rows(summary: dict[str, Any], split: str, factor: str, direction: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    best_by_model = summary.get("best_by_model", {})
    for style_source, by_granularity in best_by_model.items():
        for eval_granularity, by_model in by_granularity.items():
            for model, rec in by_model.items():
                out.append(
                    {
                        "factor": factor,
                        "direction": direction,
                        "split": split,
                        "style_source": style_source,
                        "eval_granularity": eval_granularity,
                        "model": model,
                        "k_clusters": rec.get("k_clusters"),
                        "baseline_r2": rec.get("baseline_r2"),
                        "augmented_r2": rec.get("augmented_r2"),
                        "delta_r2": rec.get("delta_r2"),
                        "baseline_r2_oos_train_mean": rec.get("baseline_r2_oos_train_mean"),
                        "augmented_r2_oos_train_mean": rec.get("augmented_r2_oos_train_mean"),
                        "delta_r2_oos_train_mean": rec.get("delta_r2_oos_train_mean"),
                        "baseline_rmse": rec.get("baseline_rmse"),
                        "augmented_rmse": rec.get("augmented_rmse"),
                        "delta_rmse": rec.get("delta_rmse"),
                        "baseline_mae": rec.get("baseline_mae"),
                        "augmented_mae": rec.get("augmented_mae"),
                        "delta_mae": rec.get("delta_mae"),
                        "n_train_eval_rows": rec.get("n_train_eval_rows"),
                        "n_val_eval_rows": rec.get("n_val_eval_rows"),
                        "style_map_val_rmse": rec.get("style_map_val_rmse"),
                        "style_map_val_mae": rec.get("style_map_val_mae"),
                }
            )
    return out


def extract_noise_ceiling_rows(
    summary: dict[str, Any], split: str, factor: str, direction: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_granularity = summary.get("noise_ceiling_by_eval_granularity", {}) or {}
    for eval_granularity, rec in by_granularity.items():
        oracle = rec.get("oracle_test_config_mean", {}) or {}
        mapped = rec.get("train_to_test_config_mean", {}) or {}
        out.append(
            {
                "factor": factor,
                "direction": direction,
                "split": split,
                "eval_granularity": eval_granularity,
                "n_train_rows": rec.get("n_train_rows"),
                "n_test_rows": rec.get("n_test_rows"),
                "n_train_unique_configs": rec.get("n_train_unique_configs"),
                "n_test_unique_configs": rec.get("n_test_unique_configs"),
                "n_overlap_configs": rec.get("n_overlap_configs"),
                "unseen_test_share": rec.get("unseen_test_share"),
                "train_mean_target": rec.get("train_mean_target"),
                "oracle_test_config_mean_r2": oracle.get("r2"),
                "oracle_test_config_mean_r2_oos_train_mean": oracle.get("r2_oos_train_mean"),
                "oracle_test_config_mean_rmse": oracle.get("rmse"),
                "oracle_test_config_mean_mae": oracle.get("mae"),
                "train_to_test_config_mean_r2": mapped.get("r2"),
                "train_to_test_config_mean_r2_oos_train_mean": mapped.get("r2_oos_train_mean"),
                "train_to_test_config_mean_rmse": mapped.get("rmse"),
                "train_to_test_config_mean_mae": mapped.get("mae"),
            }
        )
    return out


def extract_sampling_noise_ceiling_row(
    summary: dict[str, Any], split: str, factor: str, direction: str
) -> dict[str, Any]:
    rec = summary.get("noise_ceiling_sampling_config_treatment", {}) or {}
    return {
        "factor": factor,
        "direction": direction,
        "split": split,
        "group_col": rec.get("group_col"),
        "n_rows": rec.get("n_rows"),
        "n_groups": rec.get("n_groups"),
        "n_groups_with_replicates": rec.get("n_groups_with_replicates"),
        "mean_group_size": rec.get("mean_group_size"),
        "median_group_size": rec.get("median_group_size"),
        "mse_floor": rec.get("mse_floor"),
        "rmse_floor": rec.get("rmse_floor"),
        "r2_ceiling_test_mean": rec.get("r2_ceiling_test_mean"),
        "r2_ceiling_train_mean": rec.get("r2_ceiling_train_mean"),
    }


def build_comparison_tables(
    all_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compare_rows: list[dict[str, Any]] = []
    for (split, granularity, model), sub in all_df.groupby(["split", "eval_granularity", "model"]):
        has_oracle = (sub["style_source"] == "oracle").any()
        has_synthetic = (sub["style_source"] == "synthetic").any()
        if not (has_oracle and has_synthetic):
            continue
        oracle = sub[sub["style_source"] == "oracle"].iloc[0]
        synthetic = sub[sub["style_source"] == "synthetic"].iloc[0]
        compare_rows.append(
            {
                "split": split,
                "factor": oracle["factor"],
                "direction": oracle["direction"],
                "eval_granularity": granularity,
                "model": model,
                "oracle_best_delta_r2": oracle["delta_r2"],
                "synthetic_best_delta_r2": synthetic["delta_r2"],
                "oracle_best_delta_r2_oos_train_mean": oracle["delta_r2_oos_train_mean"],
                "synthetic_best_delta_r2_oos_train_mean": synthetic["delta_r2_oos_train_mean"],
                "oracle_best_delta_rmse": oracle["delta_rmse"],
                "synthetic_best_delta_rmse": synthetic["delta_rmse"],
                "synthetic_beats_oracle_on_r2_gain": bool(synthetic["delta_r2"] > oracle["delta_r2"]),
                "synthetic_beats_oracle_on_r2_oos_gain": bool(
                    synthetic["delta_r2_oos_train_mean"] > oracle["delta_r2_oos_train_mean"]
                ),
                "synthetic_beats_oracle_on_rmse_drop": bool(
                    synthetic["delta_rmse"] < oracle["delta_rmse"]
                ),
            }
        )
    comparison_df = pd.DataFrame(compare_rows)

    if len(comparison_df):
        counts_df = (
            comparison_df.groupby(["eval_granularity", "model"], as_index=False)[
                [
                    "synthetic_beats_oracle_on_r2_gain",
                    "synthetic_beats_oracle_on_r2_oos_gain",
                    "synthetic_beats_oracle_on_rmse_drop",
                ]
            ]
            .mean()
            .merge(
                comparison_df.groupby(["eval_granularity", "model"], as_index=False)
                .size()
                .rename(columns={"size": "n_splits"}),
                on=["eval_granularity", "model"],
                how="left",
            )
        )
    else:
        counts_df = pd.DataFrame(
            columns=[
                "eval_granularity",
                "model",
                "synthetic_beats_oracle_on_r2_gain",
                "synthetic_beats_oracle_on_r2_oos_gain",
                "synthetic_beats_oracle_on_rmse_drop",
                "n_splits",
            ]
        )

    if len(all_df):
        split_level_df = all_df.copy()
        split_level_df["win_r2"] = split_level_df["delta_r2"] > 0
        split_level_df["win_r2_oos_train_mean"] = split_level_df["delta_r2_oos_train_mean"] > 0
        split_level_df["win_rmse"] = split_level_df["delta_rmse"] < 0
        split_level_df["win_mae"] = split_level_df["delta_mae"] < 0
        split_level_df["win_r2_rmse"] = split_level_df["win_r2"] & split_level_df["win_rmse"]
        split_level_df["win_all"] = split_level_df["win_r2"] & split_level_df["win_rmse"] & split_level_df["win_mae"]
        improvement_counts_df = (
            split_level_df.groupby(["style_source", "eval_granularity", "model"], as_index=False)
            .agg(
                n_splits=("split", "nunique"),
                r2_win_rate=("win_r2", "mean"),
                r2_oos_train_mean_win_rate=("win_r2_oos_train_mean", "mean"),
                rmse_win_rate=("win_rmse", "mean"),
                mae_win_rate=("win_mae", "mean"),
                r2_rmse_win_rate=("win_r2_rmse", "mean"),
                all3_win_rate=("win_all", "mean"),
                mean_delta_r2=("delta_r2", "mean"),
                mean_delta_r2_oos_train_mean=("delta_r2_oos_train_mean", "mean"),
                mean_delta_rmse=("delta_rmse", "mean"),
                mean_delta_mae=("delta_mae", "mean"),
                median_delta_r2=("delta_r2", "median"),
                median_delta_r2_oos_train_mean=("delta_r2_oos_train_mean", "median"),
                median_delta_rmse=("delta_rmse", "median"),
                median_delta_mae=("delta_mae", "median"),
            )
        )
    else:
        split_level_df = pd.DataFrame()
        improvement_counts_df = pd.DataFrame()

    return comparison_df, counts_df, split_level_df, improvement_counts_df


def run_ood_batch(args: argparse.Namespace) -> dict[str, Any]:
    start_ts = datetime.now(timezone.utc)
    splits = find_splits(args.ood_root)
    if args.max_splits > 0:
        splits = splits[: args.max_splits]
    if not splits:
        raise SystemExit(f"No splits found under: {args.ood_root}")

    run_id = args.run_id or utc_run_id()
    run_root = args.output_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    sampling_noise_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for idx, split in enumerate(splits, 1):
        rel = split.relative_to(args.ood_root)
        factor, direction = rel.parts
        split_name = f"{factor}/{direction}"
        out_dir = run_root / factor / direction
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "summary.json"
        out_csv = out_dir / "results.csv"

        if args.skip_existing and out_json.exists() and out_csv.exists():
            print(f"[{idx}/{len(splits)}] skip existing {split_name}")
        else:
            print(f"[{idx}/{len(splits)}] run {split_name}")
            proc = run_one_split(split, out_dir, args)
            if proc.returncode != 0:
                failures.append(
                    {
                        "split": split_name,
                        "returncode": proc.returncode,
                        "stderr_tail": proc.stderr[-2000:],
                    }
                )
                print(f"  failed rc={proc.returncode}")
                continue

        try:
            summary = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover
            failures.append({"split": split_name, "returncode": -1, "stderr_tail": f"summary read error: {exc}"})
            print("  failed reading summary.json")
            continue

        rows.extend(extract_best_rows(summary, split_name, factor, direction))
        noise_rows.extend(extract_noise_ceiling_rows(summary, split_name, factor, direction))
        sampling_noise_rows.append(extract_sampling_noise_ceiling_row(summary, split_name, factor, direction))

    all_df = pd.DataFrame(rows)
    all_path = run_root / "all_splits_best_rows.csv"
    all_df.to_csv(all_path, index=False)

    comparison_df, counts_df, split_level_df, improvement_counts_df = build_comparison_tables(all_df)
    noise_df = pd.DataFrame(noise_rows)
    sampling_noise_df = pd.DataFrame(sampling_noise_rows)
    comparison_path = run_root / "comparison_synthetic_vs_oracle.csv"
    comparison_counts_path = run_root / "comparison_counts.csv"
    split_level_path = run_root / "config_only_improvement_split_level.csv"
    improvement_counts_path = run_root / "config_only_improvement_counts.csv"
    noise_split_path = run_root / "noise_ceiling_by_split.csv"
    noise_summary_path = run_root / "noise_ceiling_summary.csv"
    sampling_noise_split_path = run_root / "noise_ceiling_sampling_config_treatment_by_split.csv"
    sampling_noise_summary_path = run_root / "noise_ceiling_sampling_config_treatment_summary.csv"
    comparison_df.to_csv(comparison_path, index=False)
    counts_df.to_csv(comparison_counts_path, index=False)
    split_level_df.to_csv(split_level_path, index=False)
    improvement_counts_df.to_csv(improvement_counts_path, index=False)
    noise_df.to_csv(noise_split_path, index=False)
    sampling_noise_df.to_csv(sampling_noise_split_path, index=False)

    if len(noise_df):
        noise_summary_df = (
            noise_df.groupby(["eval_granularity"], as_index=False)
            .agg(
                n_splits=("split", "nunique"),
                mean_unseen_test_share=("unseen_test_share", "mean"),
                mean_oracle_r2=("oracle_test_config_mean_r2", "mean"),
                mean_oracle_r2_oos_train_mean=("oracle_test_config_mean_r2_oos_train_mean", "mean"),
                mean_oracle_rmse=("oracle_test_config_mean_rmse", "mean"),
                mean_train_to_test_r2=("train_to_test_config_mean_r2", "mean"),
                mean_train_to_test_r2_oos_train_mean=("train_to_test_config_mean_r2_oos_train_mean", "mean"),
                mean_train_to_test_rmse=("train_to_test_config_mean_rmse", "mean"),
            )
        )
    else:
        noise_summary_df = pd.DataFrame()
    noise_summary_df.to_csv(noise_summary_path, index=False)

    if len(sampling_noise_df):
        sampling_noise_summary_df = pd.DataFrame(
            [
                {
                    "n_splits": int(sampling_noise_df["split"].nunique()),
                    "mean_n_groups": float(sampling_noise_df["n_groups"].mean()),
                    "mean_n_groups_with_replicates": float(
                        sampling_noise_df["n_groups_with_replicates"].mean()
                    ),
                    "mean_group_size": float(sampling_noise_df["mean_group_size"].mean()),
                    "median_group_size": float(sampling_noise_df["median_group_size"].mean()),
                    "mean_mse_floor": float(sampling_noise_df["mse_floor"].mean()),
                    "mean_rmse_floor": float(sampling_noise_df["rmse_floor"].mean()),
                    "mean_r2_ceiling_test_mean": float(
                        sampling_noise_df["r2_ceiling_test_mean"].mean()
                    ),
                    "mean_r2_ceiling_train_mean": float(
                        sampling_noise_df["r2_ceiling_train_mean"].mean()
                    ),
                }
            ]
        )
    else:
        sampling_noise_summary_df = pd.DataFrame()
    sampling_noise_summary_df.to_csv(sampling_noise_summary_path, index=False)

    end_ts = datetime.now(timezone.utc)
    manifest = {
        "run_id": run_id,
        "started_at_utc": start_ts.isoformat(),
        "ended_at_utc": end_ts.isoformat(),
        "elapsed_seconds": (end_ts - start_ts).total_seconds(),
        "ood_root": str(args.ood_root),
        "output_root": str(args.output_root),
        "run_root": str(run_root),
        "n_splits_requested": len(splits),
        "n_failures": len(failures),
        "failures": failures,
        "args": json_safe(vars(args)),
        "outputs": {
            "all_splits_best_rows_csv": str(all_path),
            "comparison_csv": str(comparison_path),
            "comparison_counts_csv": str(comparison_counts_path),
            "config_only_split_level_csv": str(split_level_path),
            "config_only_counts_csv": str(improvement_counts_path),
            "noise_ceiling_by_split_csv": str(noise_split_path),
            "noise_ceiling_summary_csv": str(noise_summary_path),
            "noise_ceiling_sampling_config_treatment_by_split_csv": str(sampling_noise_split_path),
            "noise_ceiling_sampling_config_treatment_summary_csv": str(sampling_noise_summary_path),
        },
    }
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "latest_run.txt").write_text(f"{run_id}\n", encoding="utf-8")

    print("\nDONE")
    print(f"run_root: {run_root}")
    print(f"manifest: {manifest_path}")
    print(f"n_failures: {len(failures)}")
    return manifest


def main() -> None:
    args = parse_ood_batch_args()
    run_ood_batch(args)


if __name__ == "__main__":
    main()
