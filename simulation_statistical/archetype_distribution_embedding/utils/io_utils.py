from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd


def ensure_dir(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_dir(path: Path | str) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    file_path = Path(path)
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]], append: bool = False) -> None:
    file_path = ensure_parent_dir(path)
    mode = "a" if append else "w"
    with file_path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_text(path: Path | str, text: str) -> None:
    file_path = ensure_parent_dir(path)
    file_path.write_text(text, encoding="utf-8")


def save_dataframe(df: pd.DataFrame, path: Path | str) -> None:
    file_path = ensure_parent_dir(path)
    df.to_parquet(file_path, index=False)


def load_dataframe(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def save_pickle(obj: Any, path: Path | str) -> None:
    file_path = ensure_parent_dir(path)
    joblib.dump(obj, file_path)


def load_pickle(path: Path | str) -> Any:
    return joblib.load(Path(path))
