from __future__ import annotations

from transformers import HfArgumentParser

try:
    from .run_simulation import Args, main
    from .utils import log
except ImportError:
    from run_simulation import Args, main
    from utils import log


if __name__ == "__main__":
    parser = HfArgumentParser(Args)
    parsed, unknown = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    cfg = parsed[0] if isinstance(parsed, (list, tuple)) else parsed
    if unknown:
        log("[note] unknown args (ignored):", unknown)
    main(cfg)
