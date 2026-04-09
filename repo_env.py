from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Iterable


DEFAULT_ENV_FILENAMES = (".api_keys.env", ".env")


def find_repo_root(search_from: Path | None = None) -> Path:
    start = (search_from or Path(__file__)).resolve()
    current = start if start.is_dir() else start.parent
    for candidate in (current, *current.parents):
        if (
            (candidate / ".api_keys.env").is_file()
            or (candidate / ".git").exists()
            or (candidate / ".gitignore").is_file()
        ):
            return candidate
    return Path(__file__).resolve().parent


def _parse_env_assignment(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].strip()
    if "=" not in line:
        return None

    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        try:
            parsed = ast.literal_eval(value)
            value = str(parsed)
        except Exception:
            value = value[1:-1]
    return key, value


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = _parse_env_assignment(line)
            if parsed is None:
                continue
            key, value = parsed
            values[key] = value
    return values


def load_repo_env(
    *,
    search_from: Path | None = None,
    filenames: Iterable[str] = DEFAULT_ENV_FILENAMES,
    override: bool = False,
) -> list[Path]:
    repo_root = find_repo_root(search_from=search_from)
    loaded_paths: list[Path] = []
    for filename in filenames:
        path = repo_root / filename
        if not path.is_file():
            continue
        loaded_paths.append(path)
        for key, value in _load_env_file(path).items():
            if override or key not in os.environ:
                os.environ[key] = value
    return loaded_paths


def get_env_var(
    name: str,
    *,
    default: str | None = None,
    search_from: Path | None = None,
) -> str | None:
    load_repo_env(search_from=search_from)
    return os.getenv(name, default)


def require_env_var(name: str, *, search_from: Path | None = None) -> str:
    value = get_env_var(name, search_from=search_from)
    if value:
        return value
    raise KeyError(f"{name} was not found in the environment or repo env files.")
