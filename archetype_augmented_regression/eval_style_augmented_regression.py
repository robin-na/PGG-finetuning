#!/usr/bin/env python3
"""CLI wrapper for single-split archetype-augmented regression eval."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from archetype_augmented_regression.eval_pipeline import parse_eval_args, run_eval


def main() -> None:
    args = parse_eval_args()
    run_eval(args)


if __name__ == "__main__":
    main()
