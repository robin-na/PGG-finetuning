from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd


@dataclass(frozen=True)
class BenchmarkFilters:
    no_chat: bool = True
    low_player_count_leq: int = 11
    low_num_rounds_leq: int = 15
    show_n_rounds: bool = True
    no_reward: bool = True
    dont_show_punishment_id: bool = True


def _require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def filter_df_analysis_learn(
    df: pd.DataFrame,
    filters: BenchmarkFilters,
) -> pd.DataFrame:
    """
    Returns a filtered copy of df_analysis_learn with the benchmark regime applied.
    Expected shape for current benchmark: 10 rows (games), 48 total participants.
    """
    required = [
        "gameId",
        "CONFIG_chat",
        "CONFIG_playerCount",
        "CONFIG_numRounds",
        "CONFIG_showNRounds",
        "CONFIG_rewardExists",
        "CONFIG_showPunishmentId",
    ]
    _require_cols(df, required)

    mask = pd.Series(True, index=df.index)
    if filters.no_chat:
        mask &= df["CONFIG_chat"].astype(bool) == False
    if filters.low_player_count_leq is not None:
        mask &= df["CONFIG_playerCount"].astype(int) <= int(filters.low_player_count_leq)
    if filters.low_num_rounds_leq is not None:
        mask &= df["CONFIG_numRounds"].astype(int) <= int(filters.low_num_rounds_leq)
    if filters.show_n_rounds:
        mask &= df["CONFIG_showNRounds"].astype(bool) == True
    if filters.no_reward:
        mask &= df["CONFIG_rewardExists"].astype(bool) == False
    if filters.dont_show_punishment_id:
        mask &= df["CONFIG_showPunishmentId"].astype(bool) == False

    out = df.loc[mask].copy()

    # Ensure one row per gameId to avoid accidental duplication in downstream joins.
    out = out.drop_duplicates(subset=["gameId"], keep="first")
    return out


def write_filtered_env_csv(
    input_csv: Path,
    output_csv: Path,
    filters: Optional[BenchmarkFilters] = None,
) -> pd.DataFrame:
    filters = filters or BenchmarkFilters()
    df = pd.read_csv(input_csv)
    filtered = filter_df_analysis_learn(df, filters)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)
    return filtered


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_persona_summary_jsonl_for_game_ids(
    input_jsonl: Path,
    output_jsonl: Path,
    game_ids: Iterable[str],
    *,
    require_finished: bool = True,
) -> int:
    game_id_set = {str(g) for g in game_ids}
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for obj in iter_jsonl(input_jsonl):
            exp = str(obj.get("experiment"))
            if exp not in game_id_set:
                continue
            if require_finished and obj.get("game_finished") is not True:
                continue
            out_f.write(json.dumps(obj, ensure_ascii=True))
            out_f.write("\n")
            kept += 1
    return kept


def build_nochat_lowplayers_lowrounds_shownrounds_noreward_noid_benchmark(
    *,
    df_analysis_learn_csv: Path = Path("data/processed_data/df_analysis_learn.csv"),
    persona_summary_learn_jsonl: Path = Path("Persona/summary_gpt51_learn.jsonl"),
    out_env_csv: Path = Path(
        "benchmark/df_analysis_learn_noChat_lowPlayers_lowRounds_showNRounds_noReward_noId.csv"
    ),
    out_persona_jsonl: Path = Path(
        "benchmark/summary_gpt51_learn_noChat_lowPlayers_lowRounds_showNRounds_noReward_noId.jsonl"
    ),
    filters: Optional[BenchmarkFilters] = None,
) -> dict:
    filters = filters or BenchmarkFilters()
    filtered = write_filtered_env_csv(df_analysis_learn_csv, out_env_csv, filters)
    game_ids = filtered["gameId"].astype(str).tolist()
    kept_personas = write_persona_summary_jsonl_for_game_ids(
        persona_summary_learn_jsonl,
        out_persona_jsonl,
        game_ids,
        require_finished=True,
    )
    return {
        "out_env_csv": str(out_env_csv),
        "out_persona_jsonl": str(out_persona_jsonl),
        "n_games": int(len(filtered)),
        "sum_player_count": int(filtered["CONFIG_playerCount"].astype(int).sum()) if "CONFIG_playerCount" in filtered.columns else None,
        "n_persona_records": int(kept_personas),
    }


if __name__ == "__main__":
    result = build_nochat_lowplayers_lowrounds_shownrounds_noreward_noid_benchmark()
    print(json.dumps(result, indent=2, sort_keys=True))

