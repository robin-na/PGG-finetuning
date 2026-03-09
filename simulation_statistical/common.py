from __future__ import annotations

import ast
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


CONFIG_KEYS = {
    "CONFIG_allOrNothing",
    "CONFIG_chat",
    "CONFIG_defaultContribProp",
    "CONFIG_endowment",
    "CONFIG_multiplier",
    "CONFIG_numRounds",
    "CONFIG_playerCount",
    "CONFIG_punishmentCost",
    "CONFIG_punishmentExists",
    "CONFIG_punishmentMagnitude",
    "CONFIG_rewardCost",
    "CONFIG_rewardExists",
    "CONFIG_rewardMagnitude",
    "CONFIG_showNRounds",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
}


BOOL_CONFIG_KEYS = {
    "CONFIG_allOrNothing",
    "CONFIG_chat",
    "CONFIG_punishmentExists",
    "CONFIG_rewardExists",
    "CONFIG_showNRounds",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
}


AVATAR_POOL: List[str] = sorted(
    {
        "CHICK",
        "CHICKEN",
        "COW",
        "CROCODILE",
        "DOG",
        "DUCK",
        "ELEPHANT",
        "FROG",
        "GORILLA",
        "HORSE",
        "MONKEY",
        "MOOSE",
        "OWL",
        "PARROT",
        "PINGUIN",
        "RABBIT",
        "SLOTH",
        "SNAKE",
        "WALRUS",
        "WHALE",
    }
)


@dataclass
class MicroGameContext:
    game_id: str
    game_name: str
    env: Dict[str, Any]
    player_ids: List[str]
    avatar_by_player: Dict[str, str]
    player_by_avatar: Dict[str, str]
    rounds: List[int]
    round_to_rows: Dict[int, pd.DataFrame]
    demographics_by_player: Dict[str, str]


@dataclass
class MacroGameContext:
    game_id: str
    game_name: str
    env: Dict[str, Any]
    player_ids: List[str]
    avatar_by_player: Dict[str, str]
    player_by_avatar: Dict[str, str]
    demographics_by_player: Dict[str, str]


def log(*args: Any, **kwargs: Any) -> None:
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def timestamp_yymmddhhmm() -> str:
    return datetime.now().strftime("%y%m%d%H%M")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def relocate_output(path: Optional[str], directory: str) -> Optional[str]:
    if not path:
        return None
    base = os.path.basename(path)
    return os.path.join(directory, base) if base else directory


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", "", "nan", "none", "null"}:
        return False
    return bool(text)


def is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except Exception:
        return False


def normalize_avatar(name: Any) -> str:
    if name is None:
        return ""
    return str(name).strip().upper()


def parse_bool_arg(value: str, default: bool) -> bool:
    if value is None:
        return default
    return as_bool(value)


def parse_dict(value: Any) -> Dict[str, int]:
    if isinstance(value, dict):
        out: Dict[str, int] = {}
        for key, raw in value.items():
            try:
                out[str(key)] = int(raw)
            except Exception:
                continue
        return out
    if value is None or is_nan(value):
        return {}
    text = str(value).strip()
    if text in {"", "{}", "None", "nan", "null"}:
        return {}
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
    if not isinstance(parsed, dict):
        return {}
    out: Dict[str, int] = {}
    for key, raw in parsed.items():
        try:
            out[str(key)] = int(raw)
        except Exception:
            continue
    return out


def make_unique_avatar_map(
    player_ids: Sequence[str],
    raw_avatar_by_player: Mapping[str, Any],
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    used: Dict[str, str] = {}
    for index, pid in enumerate(player_ids, start=1):
        base = normalize_avatar(raw_avatar_by_player.get(pid))
        if not base:
            base = f"PLAYER_{index}"
        avatar = base
        suffix = 2
        while avatar in used and used[avatar] != pid:
            avatar = f"{base}_{suffix}"
            suffix += 1
        out[pid] = avatar
        used[avatar] = pid
    return out


def _decode_gender(row: Mapping[str, Any]) -> Optional[str]:
    for col, label in (
        ("gender_man", "man"),
        ("gender_woman", "woman"),
        ("gender_non_binary", "non-binary"),
    ):
        try:
            if int(float(row.get(col, 0) or 0)) == 1:
                return label
        except Exception:
            continue
    return None


def _decode_education(row: Mapping[str, Any]) -> Optional[str]:
    for col, label in (
        ("education_high_school", "high_school"),
        ("education_bachelor", "bachelor"),
        ("education_master", "master"),
    ):
        try:
            if int(float(row.get(col, 0) or 0)) == 1:
                return label
        except Exception:
            continue
    return None


def _article_for(next_word: str) -> str:
    if not next_word:
        return "a"
    return "an" if next_word[0].lower() in {"a", "e", "i", "o", "u"} else "a"


def _education_phrase(education: Optional[str]) -> Optional[str]:
    if education == "high_school":
        return "a high school educational background"
    if education == "bachelor":
        return "a bachelor's level educational background"
    if education == "master":
        return "a master's level educational background"
    return None


def demographics_line(row: Optional[Mapping[str, Any]]) -> str:
    if not row:
        return "Your demographic profile is unavailable."

    age_missing = as_bool(row.get("age_missing"))
    age_raw = row.get("age")
    age: Optional[str] = None
    if not age_missing and age_raw is not None and not is_nan(age_raw):
        try:
            fval = float(age_raw)
            age = str(int(fval)) if fval.is_integer() else str(fval)
        except Exception:
            age = str(age_raw)

    gender = _decode_gender(row)
    education_phrase = _education_phrase(_decode_education(row))

    identity_phrase: Optional[str] = None
    if age and gender in {"man", "woman"}:
        identity_phrase = f"a {age} year old {gender}"
    elif age and gender == "non-binary":
        identity_phrase = f"a {age} year old non-binary person"
    elif age:
        identity_phrase = f"{age} years old"
    elif gender in {"man", "woman"}:
        identity_phrase = f"{_article_for(gender)} {gender}"
    elif gender == "non-binary":
        identity_phrase = "a non-binary person"

    if identity_phrase and education_phrase:
        return f"You are {identity_phrase} with {education_phrase}."
    if identity_phrase:
        return f"You are {identity_phrase}."
    if education_phrase:
        return f"You have {education_phrase}."
    return "Your demographic profile is unavailable."


def load_optional_csv(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _build_env_lookup(df_analysis: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if "gameId" not in df_analysis.columns:
        return out
    analysis = df_analysis.copy()
    analysis["gameId"] = analysis["gameId"].astype(str)
    analysis = analysis.drop_duplicates(subset=["gameId"], keep="first")
    for _, row in analysis.iterrows():
        game_id = str(row.get("gameId"))
        env = {"gameId": game_id, "name": row.get("name", game_id)}
        for key in CONFIG_KEYS:
            if key not in row.index:
                continue
            value = row.get(key)
            if key in BOOL_CONFIG_KEYS:
                value = as_bool(value)
            env[key] = value
        out[game_id] = env
    return out


def _build_avatar_lookup(df_players: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if "_id" not in df_players.columns:
        return out
    for _, row in df_players.iterrows():
        player_id = str(row.get("_id"))
        if not player_id:
            continue
        out[player_id] = normalize_avatar(row.get("data.avatar"))
    return out


def _build_demographics_lookup(
    df_demographics: pd.DataFrame,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    by_game_player: Dict[Tuple[str, str], Dict[str, Any]] = {}
    by_player: Dict[str, Dict[str, Any]] = {}
    if "playerId" not in df_demographics.columns:
        return by_game_player, by_player
    for _, row in df_demographics.iterrows():
        player_id = str(row.get("playerId"))
        if not player_id:
            continue
        record = row.to_dict()
        by_player[player_id] = record
        game_id = row.get("gameId")
        if game_id is not None and not is_nan(game_id):
            by_game_player[(str(game_id), player_id)] = record
    return by_game_player, by_player


def _build_round_index(df_rounds: pd.DataFrame) -> pd.DataFrame:
    rounds = df_rounds.copy()
    rounds["gameId"] = rounds["gameId"].astype(str)
    rounds["roundId"] = rounds["roundId"].astype(str)
    rounds["playerId"] = rounds["playerId"].astype(str)
    rounds["__row_order"] = range(len(rounds))
    if "createdAt" in rounds.columns:
        rounds["__created_at"] = pd.to_datetime(rounds["createdAt"], errors="coerce", utc=True)
    else:
        rounds["__created_at"] = pd.NaT

    round_order = (
        rounds.groupby(["gameId", "roundId"], as_index=False)
        .agg(__created_at=("__created_at", "min"), __row_order=("__row_order", "min"))
        .sort_values(["gameId", "__created_at", "__row_order", "roundId"], na_position="last")
    )
    round_order["roundIndex"] = round_order.groupby("gameId").cumcount() + 1
    merged = rounds.merge(round_order[["gameId", "roundId", "roundIndex"]], on=["gameId", "roundId"], how="left")
    merged["roundIndex"] = pd.to_numeric(merged["roundIndex"], errors="coerce").fillna(0).astype(int)
    return merged


def build_micro_game_contexts(
    df_rounds: pd.DataFrame,
    df_analysis: pd.DataFrame,
    df_demographics: pd.DataFrame,
    df_players: pd.DataFrame,
) -> List[MicroGameContext]:
    rounds = _build_round_index(df_rounds)
    env_lookup = _build_env_lookup(df_analysis)
    if env_lookup:
        rounds = rounds[rounds["gameId"].isin(set(env_lookup.keys()))].copy()
    avatar_lookup = _build_avatar_lookup(df_players)
    demo_by_game_player, demo_by_player = _build_demographics_lookup(df_demographics)

    contexts: List[MicroGameContext] = []
    for game_id, game_df in rounds.groupby("gameId", sort=True):
        env = env_lookup.get(str(game_id))
        if not env:
            log(f"[warn] skipping gameId={game_id}; missing CONFIG row in analysis CSV")
            continue

        round_ids = sorted(int(x) for x in game_df["roundIndex"].dropna().unique().tolist() if int(x) > 0)
        if not round_ids:
            continue

        sorted_rows = game_df.sort_values(["roundIndex", "playerId"])
        player_ids = list(dict.fromkeys(sorted_rows["playerId"].astype(str).tolist()))
        avatar_by_player = make_unique_avatar_map(player_ids, avatar_lookup)
        player_by_avatar = {avatar: player_id for player_id, avatar in avatar_by_player.items()}

        demographics_by_player: Dict[str, str] = {}
        for player_id in player_ids:
            demo_row = demo_by_game_player.get((str(game_id), player_id))
            if demo_row is None:
                demo_row = demo_by_player.get(player_id)
            demographics_by_player[player_id] = demographics_line(demo_row)

        round_to_rows = {
            round_id: (
                game_df[game_df["roundIndex"] == round_id]
                .copy()
                .sort_values(["playerId"])
                .reset_index(drop=True)
            )
            for round_id in round_ids
        }

        contexts.append(
            MicroGameContext(
                game_id=str(game_id),
                game_name=str(env.get("name", game_id)),
                env=dict(env),
                player_ids=player_ids,
                avatar_by_player=avatar_by_player,
                player_by_avatar=player_by_avatar,
                rounds=round_ids,
                round_to_rows=round_to_rows,
                demographics_by_player=demographics_by_player,
            )
        )
    return contexts


def _parse_player_ids_cell(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x).strip()]
        except Exception:
            pass
        return [part.strip() for part in text.split(",") if part.strip()]
    return []


def _build_player_ids_by_game(df_rounds: pd.DataFrame, df_analysis: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}

    if {"gameId", "playerId"}.issubset(df_rounds.columns):
        rounds = df_rounds.copy()
        rounds["gameId"] = rounds["gameId"].astype(str)
        rounds["playerId"] = rounds["playerId"].astype(str)
        rounds["__row_order"] = range(len(rounds))
        if "createdAt" in rounds.columns:
            rounds["__created_at"] = pd.to_datetime(rounds["createdAt"], errors="coerce", utc=True)
        else:
            rounds["__created_at"] = pd.NaT
        rounds = rounds.sort_values(["gameId", "__created_at", "__row_order", "playerId"], na_position="last")
        for game_id, game_df in rounds.groupby("gameId", sort=True):
            ordered: List[str] = []
            seen: set[str] = set()
            for raw_player_id in game_df["playerId"].tolist():
                player_id = str(raw_player_id)
                if not player_id or player_id in seen:
                    continue
                ordered.append(player_id)
                seen.add(player_id)
            if ordered:
                out[str(game_id)] = ordered

    if {"gameId", "playerIds"}.issubset(df_analysis.columns):
        analysis = df_analysis.copy()
        analysis["gameId"] = analysis["gameId"].astype(str)
        analysis = analysis.drop_duplicates(subset=["gameId"], keep="first")
        for _, row in analysis.iterrows():
            game_id = str(row.get("gameId"))
            if game_id in out and out[game_id]:
                continue
            parsed = _parse_player_ids_cell(row.get("playerIds"))
            if parsed:
                out[game_id] = parsed

    return out


def _ensure_configured_player_count(game_id: str, player_ids: List[str], configured_count: int) -> List[str]:
    out = list(player_ids)
    if configured_count <= 0 or len(out) >= configured_count:
        return out
    missing = configured_count - len(out)
    for index in range(1, missing + 1):
        out.append(f"{game_id}__SYNTH_{index}")
    return out


def _build_avatar_seed_map(player_ids: List[str], avatar_lookup: Mapping[str, str]) -> Dict[str, str]:
    seed_map: Dict[str, str] = {}
    used: set[str] = set()
    for player_id in player_ids:
        avatar = normalize_avatar(avatar_lookup.get(player_id))
        if avatar:
            seed_map[player_id] = avatar
            used.add(avatar)

    available = [avatar for avatar in AVATAR_POOL if avatar not in used]
    for player_id in player_ids:
        if player_id in seed_map:
            continue
        seed_map[player_id] = available.pop(0) if available else ""
    return seed_map


def build_macro_game_contexts(
    df_analysis: pd.DataFrame,
    df_rounds: pd.DataFrame,
    df_players: pd.DataFrame,
    df_demographics: pd.DataFrame,
) -> List[MacroGameContext]:
    env_lookup = _build_env_lookup(df_analysis)
    avatar_lookup = _build_avatar_lookup(df_players)
    player_ids_by_game = _build_player_ids_by_game(df_rounds, df_analysis)
    demo_by_game_player, demo_by_player = _build_demographics_lookup(df_demographics)

    contexts: List[MacroGameContext] = []
    for game_id in sorted(env_lookup.keys()):
        env = dict(env_lookup[game_id])
        configured_count = int(env.get("CONFIG_playerCount", 0) or 0)
        player_ids = list(player_ids_by_game.get(game_id, []))
        if configured_count > 0 and len(player_ids) > configured_count:
            player_ids = player_ids[:configured_count]
        if configured_count > 0:
            player_ids = _ensure_configured_player_count(game_id, player_ids, configured_count)
        if not player_ids:
            log(f"[warn] skipping gameId={game_id}; no player IDs found")
            continue

        env["CONFIG_playerCount"] = len(player_ids)
        avatar_seed_map = _build_avatar_seed_map(player_ids, avatar_lookup)
        avatar_by_player = make_unique_avatar_map(player_ids, avatar_seed_map)
        player_by_avatar = {avatar: player_id for player_id, avatar in avatar_by_player.items()}

        demographics_by_player: Dict[str, str] = {}
        for player_id in player_ids:
            demo_row = demo_by_game_player.get((game_id, player_id))
            if demo_row is None:
                demo_row = demo_by_player.get(player_id)
            demographics_by_player[player_id] = demographics_line(demo_row)

        contexts.append(
            MacroGameContext(
                game_id=game_id,
                game_name=str(env.get("name", game_id)),
                env=env,
                player_ids=player_ids,
                avatar_by_player=avatar_by_player,
                player_by_avatar=player_by_avatar,
                demographics_by_player=demographics_by_player,
            )
        )
    return contexts
