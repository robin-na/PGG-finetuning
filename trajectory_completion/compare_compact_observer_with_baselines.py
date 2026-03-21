from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import load_wave_games
from .evaluate import _rollout_game, _summarize_game, _summarize_overall
from .evaluate_compact_observer_outputs import (
    _build_avatar_mapping,
    _load_request_manifest,
    _parsed_rounds_to_round_records,
    _read_jsonl,
)
from .build_compact_observer_batch_inputs import _load_avatar_map, _load_game_rows
from .parse_compact_observer_outputs import (
    _build_output_row,
    _extract_text_from_response_record,
    _load_manifest as _load_parse_manifest,
    _sanitize_parsed_output,
    _validate_parsed_output,
    parse_compact_observer_text,
)


def _wave_parts(split: str) -> tuple[str, str]:
    if split == "learn":
        return "learning_wave", "learn"
    if split == "val":
        return "validation_wave", "val"
    raise ValueError(f"Unsupported split: {split}")


def _parse_raw_outputs(
    *,
    input_jsonl: Path,
    request_manifest_csv: Path,
) -> list[dict[str, object]]:
    records = _read_jsonl(input_jsonl)
    expectations_by_custom_id = _load_parse_manifest(request_manifest_csv)
    parsed_rows: list[dict[str, object]] = []
    for record in records:
        try:
            text = _extract_text_from_response_record(record)
            parsed = parse_compact_observer_text(text)
        except Exception as exc:
            parsed = {
                "reflection": "",
                "overall_reflection_marker_seen": False,
                "predicted_rounds": [],
                "parse_errors": [f"Top-level extraction failure: {exc}"],
            }
        parsed = _sanitize_parsed_output(parsed, expectations_by_custom_id.get(str(record.get("custom_id", ""))))
        validation_errors = _validate_parsed_output(parsed, expectations_by_custom_id.get(str(record.get("custom_id", ""))))
        parsed_rows.append(_build_output_row(record, parsed, validation_errors))
    return parsed_rows


def _evaluate_parsed_rows(
    *,
    repo_root: Path,
    split: str,
    parsed_rows: list[dict[str, object]],
    request_manifest_csv: Path,
    baseline_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wave_name, processed_suffix = _wave_parts(split)
    games = load_wave_games(
        repo_root=repo_root,
        wave_name=wave_name,
        processed_suffix=processed_suffix,
        min_num_rounds_exclusive=0,
    )
    games_by_id = {game.game_id: game for game in games}

    avatar_map = _load_avatar_map(repo_root / f"data/raw_data/{wave_name}/players.csv")
    game_rows = _load_game_rows(repo_root / f"data/raw_data/{wave_name}/games.csv")
    request_manifest = _load_request_manifest(request_manifest_csv)

    actor_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []
    request_status_rows: list[dict[str, object]] = []

    for parsed_row in parsed_rows:
        custom_id = str(parsed_row.get("custom_id", ""))
        manifest_row = request_manifest.get(custom_id)
        if manifest_row is None:
            request_status_rows.append(
                {
                    "custom_id": custom_id,
                    "evaluated": False,
                    "reason": "missing_manifest_row",
                    "parse_success": bool(parsed_row.get("parse_success")),
                }
            )
            continue

        game_id = str(manifest_row["game_id"])
        game = games_by_id.get(game_id)
        if game is None:
            request_status_rows.append(
                {
                    "custom_id": custom_id,
                    "game_id": game_id,
                    "evaluated": False,
                    "reason": "missing_game",
                    "parse_success": bool(parsed_row.get("parse_success")),
                }
            )
            continue

        if not bool(parsed_row.get("parse_success")):
            request_status_rows.append(
                {
                    "custom_id": custom_id,
                    "game_id": game_id,
                    "evaluated": False,
                    "reason": "parse_failed",
                    "parse_success": False,
                    "parse_errors": json.dumps(parsed_row.get("parse_errors", []), ensure_ascii=False),
                    "validation_errors": json.dumps(parsed_row.get("validation_errors", []), ensure_ascii=False),
                    "k": int(manifest_row["k"]),
                }
            )
            continue

        try:
            raw_player_order, avatar_to_player = _build_avatar_mapping(
                game_id=game_id,
                expected_avatars=list(manifest_row["avatars"]),
                game_rows=game_rows,
                avatar_map=avatar_map,
            )
            predicted_rounds = _parsed_rounds_to_round_records(
                game=game,
                k=int(manifest_row["k"]),
                raw_player_order=raw_player_order,
                avatar_to_player=avatar_to_player,
                parsed_rounds=list(parsed_row.get("predicted_rounds", [])),
            )
            game_actor_rows, game_round_rows = _rollout_rows_for_prediction(
                game=game,
                baseline_name=baseline_name,
                k=int(manifest_row["k"]),
                predicted_rounds=predicted_rounds,
            )
            actor_rows.extend(game_actor_rows)
            round_rows.extend(game_round_rows)
            request_status_rows.append(
                {
                    "custom_id": custom_id,
                    "game_id": game_id,
                    "evaluated": True,
                    "reason": "",
                    "parse_success": True,
                    "k": int(manifest_row["k"]),
                    "num_predicted_rounds": len(predicted_rounds),
                }
            )
        except Exception as exc:
            request_status_rows.append(
                {
                    "custom_id": custom_id,
                    "game_id": game_id,
                    "evaluated": False,
                    "reason": f"conversion_failed: {exc}",
                    "parse_success": bool(parsed_row.get("parse_success")),
                    "k": int(manifest_row["k"]),
                }
            )

    if not actor_rows or not round_rows:
        raise RuntimeError("No parsed outputs could be evaluated.")

    actor_df = pd.DataFrame(actor_rows)
    round_df = pd.DataFrame(round_rows)
    status_df = pd.DataFrame(request_status_rows)
    game_summary_df, overall_df = _summaries_from_predictions(actor_df, round_df)
    return actor_df, round_df, game_summary_df, overall_df, status_df


def _rollout_rows_for_prediction(*, game, baseline_name: str, k: int, predicted_rounds):
    from .evaluate import _evaluate_game_rollout

    return _evaluate_game_rollout(game=game, baseline_name=baseline_name, k=k, predicted_rounds=predicted_rounds)


def _summaries_from_predictions(actor_df: pd.DataFrame, round_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    game_summary_rows = []
    for (baseline, k, game_id), group in actor_df.groupby(["baseline", "k", "game_id"], sort=True):
        game_round_group = round_df[
            (round_df["baseline"] == baseline) & (round_df["k"] == k) & (round_df["game_id"] == game_id)
        ]
        game_summary_rows.append(_summarize_game(group, game_round_group))
    game_summary_df = pd.DataFrame(game_summary_rows).sort_values(["k", "baseline", "game_id"]).reset_index(drop=True)
    overall_df = _summarize_overall(actor_df, round_df)
    future_efficiency_df = (
        game_summary_df.groupby(["baseline", "k"], as_index=False)[
            [
                "actual_future_relative_efficiency",
                "predicted_future_relative_efficiency",
                "future_relative_efficiency_abs_error",
                "actual_future_normalized_efficiency",
                "predicted_future_normalized_efficiency",
                "future_normalized_efficiency_abs_error",
            ]
        ]
        .mean()
    )
    overall_df = overall_df.merge(future_efficiency_df, on=["baseline", "k"], how="left")
    return game_summary_df, overall_df


def _evaluate_baselines_for_games(
    *,
    repo_root: Path,
    split: str,
    selected_game_ids: set[str],
    k_values: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wave_name, processed_suffix = _wave_parts(split)
    games = load_wave_games(
        repo_root=repo_root,
        wave_name=wave_name,
        processed_suffix=processed_suffix,
        min_num_rounds_exclusive=0,
    )
    selected_games = [game for game in games if game.game_id in selected_game_ids]
    if not selected_games:
        raise RuntimeError("No selected games were found in the processed wave data.")

    actor_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []
    for k in k_values:
        eligible_games = [game for game in selected_games if game.num_rounds > k]
        for game in eligible_games:
            game_actor_rows, game_round_rows = _rollout_game(game, k)
            actor_rows.extend(game_actor_rows)
            round_rows.extend(game_round_rows)

    actor_df = pd.DataFrame(actor_rows)
    round_df = pd.DataFrame(round_rows)
    game_summary_df, overall_df = _summaries_from_predictions(actor_df, round_df)
    return actor_df, round_df, game_summary_df, overall_df


def _headline_columns() -> list[str]:
    return [
        "baseline",
        "k",
        "num_games",
        "contribution_rate_mae",
        "total_contribution_rate_mae",
        "round_normalized_efficiency_mae",
        "future_normalized_efficiency_abs_error",
        "punish_target_f1",
        "reward_target_f1",
    ]


def _filter_to_llm_evaluated_subset(
    *,
    baseline_actor_df: pd.DataFrame,
    baseline_round_df: pd.DataFrame,
    llm_game_summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    llm_game_ids_by_k = {
        int(k): set(group["game_id"].astype(str))
        for k, group in llm_game_summary_df.groupby("k", sort=True)
    }
    baseline_actor_parts = []
    baseline_round_parts = []
    for k, game_ids in llm_game_ids_by_k.items():
        baseline_actor_parts.append(
            baseline_actor_df[
                (baseline_actor_df["k"] == k) & (baseline_actor_df["game_id"].astype(str).isin(game_ids))
            ]
        )
        baseline_round_parts.append(
            baseline_round_df[
                (baseline_round_df["k"] == k) & (baseline_round_df["game_id"].astype(str).isin(game_ids))
            ]
        )
    filtered_actor_df = pd.concat(baseline_actor_parts, ignore_index=True)
    filtered_round_df = pd.concat(baseline_round_parts, ignore_index=True)
    filtered_game_summary_df, filtered_overall_df = _summaries_from_predictions(filtered_actor_df, filtered_round_df)
    return filtered_actor_df, filtered_round_df, filtered_game_summary_df, filtered_overall_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare compact observer batch outputs against within-game trajectory completion baselines."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", type=str, choices=["learn", "val"], default="val")
    parser.add_argument("--request-manifest-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--parsed-output-jsonl", type=Path, default=None)
    parser.add_argument("--baseline-name", type=str, default="compact_observer_llm")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.parsed_output_jsonl is not None and args.parsed_output_jsonl.exists():
        parsed_rows = _read_jsonl(args.parsed_output_jsonl)
    else:
        parsed_rows = _parse_raw_outputs(
            input_jsonl=args.output_jsonl,
            request_manifest_csv=args.request_manifest_csv,
        )
        parsed_path = args.parsed_output_jsonl or (args.output_dir / "parsed_output.jsonl")
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        with parsed_path.open("w", encoding="utf-8") as handle:
            for row in parsed_rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

    request_manifest_df = pd.read_csv(args.request_manifest_csv)
    selected_game_ids = set(request_manifest_df["game_id"].astype(str))
    k_values = sorted({int(value) for value in request_manifest_df["k"].tolist()})

    llm_actor_df, llm_round_df, llm_game_summary_df, llm_overall_df, llm_status_df = _evaluate_parsed_rows(
        repo_root=args.repo_root,
        split=args.split,
        parsed_rows=parsed_rows,
        request_manifest_csv=args.request_manifest_csv,
        baseline_name=args.baseline_name,
    )
    baseline_actor_df, baseline_round_df, baseline_game_summary_df, baseline_overall_df = _evaluate_baselines_for_games(
        repo_root=args.repo_root,
        split=args.split,
        selected_game_ids=selected_game_ids,
        k_values=k_values,
    )
    (
        baseline_on_llm_subset_actor_df,
        baseline_on_llm_subset_round_df,
        baseline_on_llm_subset_game_summary_df,
        baseline_on_llm_subset_overall_df,
    ) = _filter_to_llm_evaluated_subset(
        baseline_actor_df=baseline_actor_df,
        baseline_round_df=baseline_round_df,
        llm_game_summary_df=llm_game_summary_df,
    )

    all_actor_df = pd.concat([baseline_actor_df, llm_actor_df], ignore_index=True)
    all_round_df = pd.concat([baseline_round_df, llm_round_df], ignore_index=True)
    all_game_summary_df = pd.concat([baseline_game_summary_df, llm_game_summary_df], ignore_index=True)
    all_overall_df = pd.concat([baseline_overall_df, llm_overall_df], ignore_index=True).sort_values(
        ["k", "baseline"]
    )
    comparison_on_llm_subset_game_summary_df = pd.concat(
        [baseline_on_llm_subset_game_summary_df, llm_game_summary_df],
        ignore_index=True,
    )
    comparison_on_llm_subset_overall_df = pd.concat(
        [baseline_on_llm_subset_overall_df, llm_overall_df],
        ignore_index=True,
    ).sort_values(["k", "baseline"])
    headline_df = all_overall_df[_headline_columns()].reset_index(drop=True)
    headline_on_llm_subset_df = comparison_on_llm_subset_overall_df[_headline_columns()].reset_index(drop=True)
    headline_pivot_df = headline_df.pivot(index="k", columns="baseline")
    headline_pivot_df.columns = [
        f"{baseline}__{metric}" for metric, baseline in headline_pivot_df.columns.to_flat_index()
    ]
    headline_pivot_df = headline_pivot_df.reset_index()
    headline_on_llm_subset_pivot_df = headline_on_llm_subset_df.pivot(index="k", columns="baseline")
    headline_on_llm_subset_pivot_df.columns = [
        f"{baseline}__{metric}" for metric, baseline in headline_on_llm_subset_pivot_df.columns.to_flat_index()
    ]
    headline_on_llm_subset_pivot_df = headline_on_llm_subset_pivot_df.reset_index()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    llm_actor_df.to_csv(args.output_dir / "llm_actor_level_predictions.csv", index=False)
    llm_round_df.to_csv(args.output_dir / "llm_round_level_predictions.csv", index=False)
    llm_game_summary_df.to_csv(args.output_dir / "llm_game_summary.csv", index=False)
    llm_overall_df.to_csv(args.output_dir / "llm_overall_summary.csv", index=False)
    llm_status_df.to_csv(args.output_dir / "llm_request_status.csv", index=False)
    baseline_actor_df.to_csv(args.output_dir / "baseline_actor_level_predictions.csv", index=False)
    baseline_round_df.to_csv(args.output_dir / "baseline_round_level_predictions.csv", index=False)
    baseline_game_summary_df.to_csv(args.output_dir / "baseline_game_summary.csv", index=False)
    baseline_overall_df.to_csv(args.output_dir / "baseline_overall_summary.csv", index=False)
    baseline_on_llm_subset_actor_df.to_csv(args.output_dir / "baseline_actor_level_predictions_on_llm_subset.csv", index=False)
    baseline_on_llm_subset_round_df.to_csv(args.output_dir / "baseline_round_level_predictions_on_llm_subset.csv", index=False)
    baseline_on_llm_subset_game_summary_df.to_csv(args.output_dir / "baseline_game_summary_on_llm_subset.csv", index=False)
    baseline_on_llm_subset_overall_df.to_csv(args.output_dir / "baseline_overall_summary_on_llm_subset.csv", index=False)
    all_game_summary_df.to_csv(args.output_dir / "comparison_game_summary.csv", index=False)
    all_overall_df.to_csv(args.output_dir / "comparison_overall_summary.csv", index=False)
    headline_df.to_csv(args.output_dir / "headline_metrics.csv", index=False)
    headline_pivot_df.to_csv(args.output_dir / "headline_metrics_pivot.csv", index=False)
    comparison_on_llm_subset_game_summary_df.to_csv(
        args.output_dir / "comparison_game_summary_on_llm_subset.csv",
        index=False,
    )
    comparison_on_llm_subset_overall_df.to_csv(
        args.output_dir / "comparison_overall_summary_on_llm_subset.csv",
        index=False,
    )
    headline_on_llm_subset_df.to_csv(args.output_dir / "headline_metrics_on_llm_subset.csv", index=False)
    headline_on_llm_subset_pivot_df.to_csv(
        args.output_dir / "headline_metrics_pivot_on_llm_subset.csv",
        index=False,
    )

    manifest = {
        "repo_root": str(args.repo_root),
        "split": args.split,
        "request_manifest_csv": str(args.request_manifest_csv),
        "output_jsonl": str(args.output_jsonl),
        "parsed_output_jsonl": str(args.parsed_output_jsonl or (args.output_dir / "parsed_output.jsonl")),
        "baseline_name": args.baseline_name,
        "selected_game_count": len(selected_game_ids),
        "k_values": k_values,
        "evaluated_requests": int(llm_status_df["evaluated"].fillna(False).sum()),
        "skipped_requests": int((~llm_status_df["evaluated"].fillna(False)).sum()),
        "llm_evaluated_games_by_k": {
            str(k): int(count) for k, count in llm_game_summary_df.groupby("k")["game_id"].nunique().items()
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote outputs to {args.output_dir}")
    print(all_overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
