#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_ood_splits import (
    SplitSpec,
    as_str_set,
    build_direction_dataset,
    normalize_bool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one-factor OOD splits while preserving original wave roles: "
            "train is sampled only from learning_wave, test only from validation_wave."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("benchmark/data"),
        help="Filtered benchmark dataset root (default: benchmark/data).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmark/data_ood_splits_wave_anchored"),
        help="Output root for generated wave-anchored OOD datasets.",
    )
    return parser.parse_args()


def to_float(value: float) -> float:
    return float(value) if pd.notna(value) else float("nan")


def make_readme(output_root: Path) -> None:
    lines = [
        "# Wave-Anchored OOD Split Datasets",
        "",
        "Each split directory is a drop-in dataset root with the same structure as `benchmark/data`:",
        "- `raw_data`",
        "- `processed_data`",
        "- `demographics`",
        "",
        "Mapping used within each direction folder:",
        "- `learning_wave` = train side sampled from original learning wave only",
        "- `validation_wave` = test side sampled from original validation wave only",
        "",
        "Rules:",
        "- One CONFIG factor at a time (no multi-factor filtering).",
        "- Both directions per factor.",
        "- Numeric factors use per-wave medians (`<= median` vs `> median`).",
        "- Boolean factors use per-wave `False` vs `True`.",
        "",
        "Build command:",
        "```bash",
        "python benchmark/build_ood_splits_wave_anchored.py",
        "```",
        "",
        "See `summary.json` for counts and per-split selection shares.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    source_root = (repo_root / args.source_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    analysis_learn = pd.read_csv(source_root / "processed_data" / "df_analysis_learn.csv")
    analysis_val = pd.read_csv(source_root / "processed_data" / "df_analysis_val.csv")
    analysis_learn["gameId"] = analysis_learn["gameId"].astype(str)
    analysis_val["gameId"] = analysis_val["gameId"].astype(str)

    n_learn_total = int(len(analysis_learn))
    n_val_total = int(len(analysis_val))

    specs = [
        SplitSpec(slug="player_count", column="CONFIG_playerCount", kind="numeric"),
        SplitSpec(slug="num_rounds", column="CONFIG_numRounds", kind="numeric"),
        SplitSpec(slug="all_or_nothing", column="CONFIG_allOrNothing", kind="boolean"),
        SplitSpec(slug="default_contrib_prop", column="CONFIG_defaultContribProp", kind="boolean"),
        SplitSpec(slug="reward_exists", column="CONFIG_rewardExists", kind="boolean"),
        SplitSpec(slug="show_n_rounds", column="CONFIG_showNRounds", kind="boolean"),
        SplitSpec(slug="show_punishment_id", column="CONFIG_showPunishmentId", kind="boolean"),
        SplitSpec(slug="show_other_summaries", column="CONFIG_showOtherSummaries", kind="boolean"),
        SplitSpec(slug="mpcr", column="CONFIG_MPCR", kind="numeric"),
    ]

    summary: dict[str, object] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "strategy": "wave_anchored",
        "n_learning_games_total": n_learn_total,
        "n_validation_games_total": n_val_total,
        "splits": {},
    }

    for spec in specs:
        if spec.column not in analysis_learn.columns or spec.column not in analysis_val.columns:
            raise KeyError(f"Missing required split column in learn/val: {spec.column}")

        split_summary: dict[str, object] = {
            "column": spec.column,
            "kind": spec.kind,
            "directions": {},
        }

        if spec.kind == "numeric":
            learn_values = pd.to_numeric(analysis_learn[spec.column], errors="coerce")
            val_values = pd.to_numeric(analysis_val[spec.column], errors="coerce")
            learn_median = to_float(learn_values.median())
            val_median = to_float(val_values.median())
            split_summary["learning_median"] = learn_median
            split_summary["validation_median"] = val_median
            directions = {
                "low_to_high": (learn_values <= learn_median, val_values > val_median),
                "high_to_low": (learn_values > learn_median, val_values <= val_median),
            }
        elif spec.kind == "boolean":
            learn_values = normalize_bool(analysis_learn[spec.column])
            val_values = normalize_bool(analysis_val[spec.column])
            directions = {
                "false_to_true": (~learn_values, val_values),
                "true_to_false": (learn_values, ~val_values),
            }
        else:
            raise ValueError(f"Unknown split kind: {spec.kind}")

        for direction, (learn_train_mask, val_test_mask) in directions.items():
            train_ids = {
                "learning_wave": as_str_set(analysis_learn.loc[learn_train_mask, "gameId"]),
                "validation_wave": set(),
            }
            test_ids = {
                "learning_wave": set(),
                "validation_wave": as_str_set(analysis_val.loc[val_test_mask, "gameId"]),
            }

            direction_root = output_root / spec.slug / direction
            direction_root.mkdir(parents=True, exist_ok=True)
            dataset_summary = build_direction_dataset(
                source_root=source_root,
                out_root=direction_root,
                train_ids=train_ids,
                test_ids=test_ids,
            )

            n_train = int(len(train_ids["learning_wave"]))
            n_test = int(len(test_ids["validation_wave"]))
            detail = {
                **dataset_summary,
                "selection": {
                    "n_train_learning_selected": n_train,
                    "n_train_learning_total": n_learn_total,
                    "train_learning_share": (n_train / n_learn_total) if n_learn_total else 0.0,
                    "n_test_validation_selected": n_test,
                    "n_test_validation_total": n_val_total,
                    "test_validation_share": (n_test / n_val_total) if n_val_total else 0.0,
                },
                "train_game_ids_by_source": {
                    sw: int(len(ids)) for sw, ids in train_ids.items()
                },
                "test_game_ids_by_source": {
                    sw: int(len(ids)) for sw, ids in test_ids.items()
                },
            }

            (direction_root / "summary.json").write_text(
                json.dumps(detail, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            split_summary["directions"][direction] = detail

        summary["splits"][spec.slug] = split_summary

    make_readme(output_root)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
