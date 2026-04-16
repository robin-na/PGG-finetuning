from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecasting.common.runs import ALL_MODELS, ALL_VARIANTS, MODEL_SLUGS, build_dataset_runs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build non-PGG forecasting batch inputs with baseline, demographic-only, and Twin-sampled variants."
    )
    parser.add_argument(
        "--dataset-key",
        required=True,
        choices=[
            "minority_game_bret_njzas",
            "longitudinal_trust_game_ht863",
            "two_stage_trust_punishment_y2hgu",
            "multi_game_llm_fvk2c",
        ],
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--forecasting-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", type=str, default=",".join(ALL_MODELS))
    parser.add_argument("--variants", type=str, default=",".join(ALL_VARIANTS))
    parser.add_argument("--max-records-per-treatment", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown_models = [model for model in models if model not in MODEL_SLUGS]
    if unknown_models:
        raise ValueError(f"Unsupported model names: {unknown_models}")
    for variant in variants:
        if variant not in ALL_VARIANTS:
            raise ValueError(f"Unsupported variant name: {variant}")
    build_dataset_runs(
        dataset_key=args.dataset_key,
        forecasting_root=args.forecasting_root,
        repo_root=args.repo_root,
        models=models,
        variants=variants,
        seed=args.seed,
        max_records_per_treatment=args.max_records_per_treatment,
    )


if __name__ == "__main__":
    main()
