#!/usr/bin/env python3
"""CLI wrapper for wave-anchored OOD batch runs."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from archetype_augmented_regression.ood_batch_pipeline import parse_ood_batch_args, run_ood_batch


def main() -> None:
    args = parse_ood_batch_args()
    run_ood_batch(args)


if __name__ == "__main__":
    main()
