from __future__ import annotations

import datetime
import os
import re
from typing import Optional


def log(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def timestamp_yymmddhhmm() -> str:
    return datetime.datetime.now().strftime("%y%m%d%H%M")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def relocate_output(path: Optional[str], directory: str) -> Optional[str]:
    if not path:
        return None
    base = os.path.basename(path)
    return os.path.join(directory, base) if base else directory
