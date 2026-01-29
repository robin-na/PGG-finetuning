from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
import ast


@dataclass
class GameConfig:
    name: str
    environment: Dict[str, Any]


@dataclass
class RowRecord:
    game_id: str
    player_id: str
    round_index: int
    contribution: float
    punished: Dict[str, float]
    rewarded: Dict[str, float]
    config: GameConfig


def _safe_parse_dict(value: str) -> Dict[str, float]:
    if value is None:
        return {}
    value = value.strip()
    if value in ("", "{}", "nan", "NaN"):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): float(v) for k, v in parsed.items()}


def _read_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def _latest_run_dir(config_dir: Path) -> Optional[Path]:
    if not config_dir.exists():
        return None
    run_dirs = [p for p in config_dir.iterdir() if p.is_dir()]
    numeric_runs = []
    for run_dir in run_dirs:
        try:
            numeric_runs.append((int(run_dir.name), run_dir))
        except ValueError:
            continue
    if not numeric_runs:
        return None
    return max(numeric_runs, key=lambda item: item[0])[1]


def load_simulation_runs(output_root: Path) -> Dict[str, List[RowRecord]]:
    configs: Dict[str, List[RowRecord]] = {}
    if not output_root.exists():
        return configs
    for config_dir in sorted(output_root.iterdir()):
        if not config_dir.is_dir():
            continue
        if not config_dir.name.startswith("VALIDATION_"):
            continue
        latest = _latest_run_dir(config_dir)
        if not latest:
            continue
        config_path = latest / "config.json"
        participants_path = latest / "participant_sim.csv"
        if not config_path.exists() or not participants_path.exists():
            continue
        with config_path.open(encoding="utf-8") as handle:
            config_payload = json.load(handle)
        environment = config_payload.get("environment", {})
        config = GameConfig(name=config_dir.name, environment=environment)
        rows: List[RowRecord] = []
        for row in _read_csv_rows(participants_path):
            rows.append(
                RowRecord(
                    game_id=row.get("gameId", ""),
                    player_id=row.get("playerAvatar", ""),
                    round_index=int(float(row.get("roundIndex") or 0)),
                    contribution=float(row.get("data.contribution") or 0.0),
                    punished=_safe_parse_dict(row.get("data.punished", "")),
                    rewarded=_safe_parse_dict(row.get("data.rewarded", "")),
                    config=config,
                )
            )
        configs[config_dir.name] = rows
    return configs


def load_human_config_map(config_csv: Path) -> Dict[str, GameConfig]:
    mapping: Dict[str, GameConfig] = {}
    if not config_csv.exists():
        return mapping
    for row in _read_csv_rows(config_csv):
        game_id = row.get("gameId")
        name = row.get("name")
        if not game_id or not name:
            continue
        environment: Dict[str, Any] = {}
        for key, value in row.items():
            if not key.startswith("CONFIG_"):
                continue
            if value is None or value == "":
                continue
            try:
                environment[key] = float(value)
                if environment[key].is_integer():
                    environment[key] = int(environment[key])
            except ValueError:
                environment[key] = value
        mapping[game_id] = GameConfig(name=name, environment=environment)
    return mapping


def load_human_rows(player_rounds_csv: Path, config_csv: Path) -> Dict[str, List[RowRecord]]:
    configs: Dict[str, List[RowRecord]] = {}
    config_map = load_human_config_map(config_csv)
    if not player_rounds_csv.exists():
        return configs
    round_index_map: Dict[str, Dict[str, int]] = {}
    for row in _read_csv_rows(player_rounds_csv):
        game_id = row.get("gameId")
        if not game_id:
            continue
        config = config_map.get(game_id)
        if not config:
            continue
        round_id = row.get("roundId", "")
        game_rounds = round_index_map.setdefault(game_id, {})
        if round_id not in game_rounds:
            game_rounds[round_id] = len(game_rounds) + 1
        rows = configs.setdefault(config.name, [])
        rows.append(
            RowRecord(
                game_id=game_id,
                player_id=row.get("playerId", ""),
                round_index=game_rounds.get(round_id, 0),
                contribution=float(row.get("data.contribution") or 0.0),
                punished=_safe_parse_dict(row.get("data.punished", "")),
                rewarded=_safe_parse_dict(row.get("data.rewarded", "")),
                config=config,
            )
        )
    return configs
