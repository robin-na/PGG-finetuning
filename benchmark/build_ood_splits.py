from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

SOURCE_WAVES = ("learning_wave", "validation_wave")


@dataclass(frozen=True)
class SplitSpec:
    slug: str
    column: str
    kind: str  # "numeric" or "boolean"


def as_str_set(values: Iterable[object]) -> set[str]:
    return {str(v) for v in values if pd.notna(v)}


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "t", "yes"})
    )


def parse_id_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x) for x in parsed if str(x)]
    except (ValueError, SyntaxError):
        pass
    if text.startswith("[") and text.endswith("]"):
        body = text[1:-1].strip()
        if not body:
            return []
        parts = [p.strip().strip("'").strip('"') for p in body.split(",")]
        return [p for p in parts if p]
    if "," in text:
        parts = [p.strip().strip("'").strip('"') for p in text.split(",")]
        return [p for p in parts if p]
    return [text]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_raw_csv(source_root: Path, source_wave: str, name: str) -> pd.DataFrame:
    return pd.read_csv(source_root / "raw_data" / source_wave / name)


def filter_by_ids(df: pd.DataFrame, col: str, ids: set[str]) -> pd.DataFrame:
    if col not in df.columns:
        return df.iloc[0:0].copy()
    if not ids:
        return df.iloc[0:0].copy()
    return df[df[col].astype(str).isin(ids)].copy()


def concat_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


def concat_from_sources(
    source_root: Path,
    filename: str,
    game_ids_by_source: dict[str, set[str]],
    *,
    game_col: str | None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for sw in SOURCE_WAVES:
        df = read_raw_csv(source_root, sw, filename)
        if game_col is None:
            parts.append(df)
            continue
        sub = filter_by_ids(df, game_col, game_ids_by_source.get(sw, set()))
        parts.append(sub)
    return concat_parts(parts)


def build_target_raw_wave(
    source_root: Path,
    out_root: Path,
    target_wave: str,
    game_ids_by_source: dict[str, set[str]],
) -> tuple[dict[str, int], set[str]]:
    out_dir = out_root / "raw_data" / target_wave

    games_df = concat_from_sources(
        source_root,
        "games.csv",
        game_ids_by_source,
        game_col="_id",
    )
    selected_treatment_ids = as_str_set(games_df.get("treatmentId", pd.Series([], dtype=object)))
    selected_batch_ids = as_str_set(games_df.get("batchId", pd.Series([], dtype=object)))
    write_csv(games_df, out_dir / "games.csv")

    game_lobbies_df = concat_from_sources(
        source_root,
        "game-lobbies.csv",
        game_ids_by_source,
        game_col="gameId",
    )
    selected_treatment_ids |= as_str_set(
        game_lobbies_df.get("treatmentId", pd.Series([], dtype=object))
    )
    selected_batch_ids |= as_str_set(
        game_lobbies_df.get("batchId", pd.Series([], dtype=object))
    )
    selected_lobby_config_ids = as_str_set(
        game_lobbies_df.get("lobbyConfigId", pd.Series([], dtype=object))
    )
    write_csv(game_lobbies_df, out_dir / "game-lobbies.csv")

    player_rounds_df = concat_from_sources(
        source_root,
        "player-rounds.csv",
        game_ids_by_source,
        game_col="gameId",
    )
    player_ids = as_str_set(player_rounds_df.get("playerId", pd.Series([], dtype=object)))
    write_csv(player_rounds_df, out_dir / "player-rounds.csv")

    rounds_df = concat_from_sources(
        source_root,
        "rounds.csv",
        game_ids_by_source,
        game_col="gameId",
    )
    selected_round_ids = as_str_set(rounds_df.get("_id", pd.Series([], dtype=object)))
    write_csv(rounds_df, out_dir / "rounds.csv")

    stages_df = concat_from_sources(
        source_root,
        "stages.csv",
        game_ids_by_source,
        game_col="gameId",
    )
    if "roundId" in stages_df.columns:
        stages_df = stages_df[stages_df["roundId"].astype(str).isin(selected_round_ids)].copy()
    selected_stage_ids = as_str_set(stages_df.get("_id", pd.Series([], dtype=object)))
    write_csv(stages_df, out_dir / "stages.csv")

    player_stages_df = concat_from_sources(
        source_root,
        "player-stages.csv",
        game_ids_by_source,
        game_col="gameId",
    )
    if "playerId" in player_stages_df.columns:
        player_stages_df = player_stages_df[
            player_stages_df["playerId"].astype(str).isin(player_ids)
        ].copy()
    if "roundId" in player_stages_df.columns:
        player_stages_df = player_stages_df[
            player_stages_df["roundId"].astype(str).isin(selected_round_ids)
        ].copy()
    if "stageId" in player_stages_df.columns:
        player_stages_df = player_stages_df[
            player_stages_df["stageId"].astype(str).isin(selected_stage_ids)
        ].copy()
    write_csv(player_stages_df, out_dir / "player-stages.csv")

    player_inputs_df = concat_from_sources(
        source_root,
        "player-inputs.csv",
        game_ids_by_source,
        game_col="gameId",
    )
    if "playerId" in player_inputs_df.columns:
        player_inputs_df = player_inputs_df[
            player_inputs_df["playerId"].astype(str).isin(player_ids)
        ].copy()
    write_csv(player_inputs_df, out_dir / "player-inputs.csv")

    players_parts: list[pd.DataFrame] = []
    for sw in SOURCE_WAVES:
        df = read_raw_csv(source_root, sw, "players.csv")
        players_parts.append(filter_by_ids(df, "_id", player_ids))
    players_df = concat_parts(players_parts)
    write_csv(players_df, out_dir / "players.csv")

    batches_parts: list[pd.DataFrame] = []
    for sw in SOURCE_WAVES:
        df = read_raw_csv(source_root, sw, "batches.csv")
        batches_parts.append(filter_by_ids(df, "_id", selected_batch_ids))
    batches_df = concat_parts(batches_parts)
    write_csv(batches_df, out_dir / "batches.csv")

    treatments_parts: list[pd.DataFrame] = []
    for sw in SOURCE_WAVES:
        df = read_raw_csv(source_root, sw, "treatments.csv")
        treatments_parts.append(filter_by_ids(df, "_id", selected_treatment_ids))
    treatments_df = concat_parts(treatments_parts)
    write_csv(treatments_df, out_dir / "treatments.csv")

    selected_factor_ids: set[str] = set()
    if "factorIds" in treatments_df.columns:
        for value in treatments_df["factorIds"]:
            selected_factor_ids.update(parse_id_list(value))

    factors_parts: list[pd.DataFrame] = []
    for sw in SOURCE_WAVES:
        df = read_raw_csv(source_root, sw, "factors.csv")
        factors_parts.append(filter_by_ids(df, "_id", selected_factor_ids))
    factors_df = concat_parts(factors_parts)
    write_csv(factors_df, out_dir / "factors.csv")

    selected_factor_type_ids = as_str_set(
        factors_df.get("factorTypeId", pd.Series([], dtype=object))
    )

    factor_types_parts: list[pd.DataFrame] = []
    for sw in SOURCE_WAVES:
        df = read_raw_csv(source_root, sw, "factor-types.csv")
        factor_types_parts.append(filter_by_ids(df, "_id", selected_factor_type_ids))
    factor_types_df = concat_parts(factor_types_parts)
    write_csv(factor_types_df, out_dir / "factor-types.csv")

    lobby_configs_parts: list[pd.DataFrame] = []
    for sw in SOURCE_WAVES:
        df = read_raw_csv(source_root, sw, "lobby-configs.csv")
        lobby_configs_parts.append(filter_by_ids(df, "_id", selected_lobby_config_ids))
    lobby_configs_df = concat_parts(lobby_configs_parts)
    write_csv(lobby_configs_df, out_dir / "lobby-configs.csv")

    return {
        "games": int(len(games_df)),
        "actions": int(len(player_rounds_df)),
        "players": int(len(players_df)),
        "rounds": int(len(rounds_df)),
        "source_game_counts": {
            sw: int(len(game_ids_by_source.get(sw, set()))) for sw in SOURCE_WAVES
        },
    }, player_ids


def load_df_analysis(source_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    learn = pd.read_csv(source_root / "processed_data" / "df_analysis_learn.csv")
    val = pd.read_csv(source_root / "processed_data" / "df_analysis_val.csv")
    learn["gameId"] = learn["gameId"].astype(str)
    val["gameId"] = val["gameId"].astype(str)
    return learn, val


def load_df_rounds(source_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    learn = pd.read_csv(source_root / "processed_data" / "df_rounds_learn.csv")
    val = pd.read_csv(source_root / "processed_data" / "df_rounds_val.csv")
    learn["gameId"] = learn["gameId"].astype(str)
    val["gameId"] = val["gameId"].astype(str)
    return learn, val


def subset_from_sources(
    learn_df: pd.DataFrame,
    val_df: pd.DataFrame,
    ids_by_source: dict[str, set[str]],
    *,
    game_col: str,
) -> pd.DataFrame:
    learn_ids = ids_by_source.get("learning_wave", set())
    val_ids = ids_by_source.get("validation_wave", set())

    out_parts = [
        filter_by_ids(learn_df, game_col, learn_ids),
        filter_by_ids(val_df, game_col, val_ids),
    ]
    return concat_parts(out_parts)


def build_processed_data(
    source_root: Path,
    out_root: Path,
    train_ids_by_source: dict[str, set[str]],
    test_ids_by_source: dict[str, set[str]],
) -> dict[str, int]:
    out_dir = out_root / "processed_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis_learn_src, analysis_val_src = load_df_analysis(source_root)
    rounds_learn_src, rounds_val_src = load_df_rounds(source_root)

    analysis_learn = subset_from_sources(
        analysis_learn_src,
        analysis_val_src,
        train_ids_by_source,
        game_col="gameId",
    )
    analysis_val = subset_from_sources(
        analysis_learn_src,
        analysis_val_src,
        test_ids_by_source,
        game_col="gameId",
    )

    rounds_learn = subset_from_sources(
        rounds_learn_src,
        rounds_val_src,
        train_ids_by_source,
        game_col="gameId",
    )
    rounds_val = subset_from_sources(
        rounds_learn_src,
        rounds_val_src,
        test_ids_by_source,
        game_col="gameId",
    )

    analysis_val_dedup = analysis_val.drop_duplicates(subset=["gameId"], keep="first").copy()

    write_csv(analysis_learn, out_dir / "df_analysis_learn.csv")
    write_csv(analysis_val, out_dir / "df_analysis_val.csv")
    write_csv(analysis_val_dedup, out_dir / "df_analysis_val_dedup.csv")
    write_csv(rounds_learn, out_dir / "df_rounds_learn.csv")
    write_csv(rounds_val, out_dir / "df_rounds_val.csv")

    learn_cfg_ids = as_str_set(analysis_learn.get("CONFIG_configId", pd.Series([], dtype=object)))
    val_cfg_ids = as_str_set(analysis_val.get("CONFIG_configId", pd.Series([], dtype=object)))

    paired_learn_src = pd.read_csv(source_root / "processed_data" / "df_paired_learn.csv")
    paired_val_src = pd.read_csv(source_root / "processed_data" / "df_paired_val.csv")
    paired_all = concat_parts([paired_learn_src, paired_val_src])
    if "CONFIG_configId" in paired_all.columns:
        paired_all["CONFIG_configId"] = paired_all["CONFIG_configId"].astype(str)
        paired_all = paired_all.drop_duplicates(subset=["CONFIG_configId"], keep="first")

    paired_learn = filter_by_ids(paired_all, "CONFIG_configId", learn_cfg_ids)
    paired_val = filter_by_ids(paired_all, "CONFIG_configId", val_cfg_ids)
    write_csv(paired_learn, out_dir / "df_paired_learn.csv")
    write_csv(paired_val, out_dir / "df_paired_val.csv")

    prediction = pd.read_csv(source_root / "processed_data" / "prediction_survey.csv")
    prediction = filter_by_ids(prediction, "CONFIG_configId", learn_cfg_ids | val_cfg_ids)
    write_csv(prediction, out_dir / "prediction_survey.csv")

    return {
        "df_analysis_learn.csv": int(len(analysis_learn)),
        "df_analysis_val.csv": int(len(analysis_val)),
        "df_analysis_val_dedup.csv": int(len(analysis_val_dedup)),
        "df_rounds_learn.csv": int(len(rounds_learn)),
        "df_rounds_val.csv": int(len(rounds_val)),
        "df_paired_learn.csv": int(len(paired_learn)),
        "df_paired_val.csv": int(len(paired_val)),
        "prediction_survey.csv": int(len(prediction)),
    }


def build_demographics(
    source_root: Path,
    out_root: Path,
    train_ids_by_source: dict[str, set[str]],
    test_ids_by_source: dict[str, set[str]],
    train_player_ids: set[str],
    test_player_ids: set[str],
) -> dict[str, int]:
    out_dir = out_root / "demographics"
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_learn_src = pd.read_csv(source_root / "demographics" / "demographics_numeric_learn.csv")
    demo_val_src = pd.read_csv(source_root / "demographics" / "demographics_numeric_val.csv")

    for df in (demo_learn_src, demo_val_src):
        df["gameId"] = df["gameId"].astype(str)
        df["playerId"] = df["playerId"].astype(str)

    demo_train = subset_from_sources(
        demo_learn_src,
        demo_val_src,
        train_ids_by_source,
        game_col="gameId",
    )
    demo_train = filter_by_ids(demo_train, "playerId", train_player_ids)

    demo_test = subset_from_sources(
        demo_learn_src,
        demo_val_src,
        test_ids_by_source,
        game_col="gameId",
    )
    demo_test = filter_by_ids(demo_test, "playerId", test_player_ids)

    write_csv(demo_train, out_dir / "demographics_numeric_learn.csv")
    write_csv(demo_test, out_dir / "demographics_numeric_val.csv")

    return {
        "demographics_numeric_learn.csv": int(len(demo_train)),
        "demographics_numeric_val.csv": int(len(demo_test)),
    }


def make_masks(meta: pd.DataFrame, spec: SplitSpec) -> tuple[dict[str, tuple[pd.Series, pd.Series]], dict[str, object]]:
    if spec.kind == "numeric":
        values = pd.to_numeric(meta[spec.column], errors="coerce")
        median = float(values.median())
        low = values <= median
        high = values > median
        return {
            "low_to_high": (low, high),
            "high_to_low": (high, low),
        }, {"kind": "numeric", "median": median}

    if spec.kind == "boolean":
        values = normalize_bool(meta[spec.column])
        false_mask = ~values
        true_mask = values
        return {
            "false_to_true": (false_mask, true_mask),
            "true_to_false": (true_mask, false_mask),
        }, {"kind": "boolean"}

    raise ValueError(f"Unknown split kind: {spec.kind}")


def ids_by_source(meta: pd.DataFrame, mask: pd.Series) -> dict[str, set[str]]:
    return {
        "learning_wave": as_str_set(
            meta.loc[mask & (meta["source_wave"] == "learning_wave"), "gameId"]
        ),
        "validation_wave": as_str_set(
            meta.loc[mask & (meta["source_wave"] == "validation_wave"), "gameId"]
        ),
    }


def build_direction_dataset(
    source_root: Path,
    out_root: Path,
    train_ids: dict[str, set[str]],
    test_ids: dict[str, set[str]],
) -> dict[str, object]:
    learn_raw_stats, train_player_ids = build_target_raw_wave(
        source_root=source_root,
        out_root=out_root,
        target_wave="learning_wave",
        game_ids_by_source=train_ids,
    )
    val_raw_stats, test_player_ids = build_target_raw_wave(
        source_root=source_root,
        out_root=out_root,
        target_wave="validation_wave",
        game_ids_by_source=test_ids,
    )

    processed_stats = build_processed_data(
        source_root=source_root,
        out_root=out_root,
        train_ids_by_source=train_ids,
        test_ids_by_source=test_ids,
    )
    demographics_stats = build_demographics(
        source_root=source_root,
        out_root=out_root,
        train_ids_by_source=train_ids,
        test_ids_by_source=test_ids,
        train_player_ids=train_player_ids,
        test_player_ids=test_player_ids,
    )

    return {
        "learning_wave": learn_raw_stats,
        "validation_wave": val_raw_stats,
        "combined": {
            "games": int(learn_raw_stats["games"] + val_raw_stats["games"]),
            "actions": int(learn_raw_stats["actions"] + val_raw_stats["actions"]),
            "players": int(learn_raw_stats["players"] + val_raw_stats["players"]),
            "rounds": int(learn_raw_stats["rounds"] + val_raw_stats["rounds"]),
        },
        "processed_data_rows": processed_stats,
        "demographics_rows": demographics_stats,
    }


def make_readme(output_root: Path) -> None:
    lines = [
        "# OOD Split Datasets",
        "",
        "Each split directory is a drop-in dataset root with the same structure as `benchmark/data`:",
        "- `raw_data`",
        "- `processed_data`",
        "- `demographics`",
        "",
        "Mapping used within each direction folder:",
        "- `learning_wave` = train side of the split",
        "- `validation_wave` = test side of the split",
        "",
        "Rules:",
        "- One CONFIG factor at a time (no multi-factor filtering).",
        "- Both directions per factor.",
        "- Numeric factors use `low <= median` and `high > median`.",
        "",
        "Build command:",
        "```bash",
        "python benchmark/build_ood_splits.py",
        "```",
        "",
        "See `summary.json` for counts and per-split details.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_root = repo_root / "benchmark" / "data"
    output_root = repo_root / "benchmark" / "data_ood_splits"
    output_root.mkdir(parents=True, exist_ok=True)

    analysis_learn = pd.read_csv(source_root / "processed_data" / "df_analysis_learn.csv")
    analysis_val = pd.read_csv(source_root / "processed_data" / "df_analysis_val.csv")
    analysis_learn["source_wave"] = "learning_wave"
    analysis_val["source_wave"] = "validation_wave"
    meta = concat_parts([analysis_learn, analysis_val])
    meta["gameId"] = meta["gameId"].astype(str)

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
        "n_total_games": int(len(meta)),
        "splits": {},
    }

    for spec in specs:
        if spec.column not in meta.columns:
            raise KeyError(f"Missing required split column: {spec.column}")

        directions, details = make_masks(meta, spec)
        split_summary: dict[str, object] = {"column": spec.column, **details, "directions": {}}

        for direction, (train_mask, test_mask) in directions.items():
            train_ids = ids_by_source(meta, train_mask)
            test_ids = ids_by_source(meta, test_mask)

            direction_root = output_root / spec.slug / direction
            direction_root.mkdir(parents=True, exist_ok=True)
            dataset_summary = build_direction_dataset(
                source_root=source_root,
                out_root=direction_root,
                train_ids=train_ids,
                test_ids=test_ids,
            )

            detail = {
                **dataset_summary,
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
