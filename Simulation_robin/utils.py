from __future__ import annotations

import datetime
import errno
import os
import re
import sys
from typing import Optional


def _redirect_std_streams() -> None:
    try:
        devnull = open(os.devnull, "w")
    except OSError:
        return
    sys.stdout = devnull
    sys.stderr = devnull


def log(*args, **kwargs):
    kwargs.setdefault("flush", True)
    try:
        print(*args, **kwargs)
    except OSError as exc:
        if exc.errno in (errno.ESTALE, errno.EBADF):
            _redirect_std_streams()
            return
        raise


def timestamp_yymmddhhmm() -> str:
    return datetime.datetime.now().strftime("%y%m%d%H%M")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def relocate_output(path: Optional[str], directory: str) -> Optional[str]:
    if not path:
        return None
    base = os.path.basename(path)
    return os.path.join(directory, base) if base else directory
