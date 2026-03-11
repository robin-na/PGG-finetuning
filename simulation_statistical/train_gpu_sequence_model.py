from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
for path in (SCRIPT_DIR, PACKAGE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from simulation_statistical.gpu_sequence_policy import (  # noqa: E402
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_GPU_SEQUENCE_POLICY_MODEL_PATH,
    DEFAULT_GPU_SEQUENCE_TRAIN_SUMMARY_PATH,
    DEFAULT_LEARN_ANALYSIS_CSV,
    DEFAULT_LEARN_CLUSTER_WEIGHTS_PATH,
    DEFAULT_LEARN_ROUNDS_CSV,
    train_gpu_sequence_policy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the GPU-backed structured sequence archetype policy on the learning wave."
    )
    parser.add_argument("--artifacts_root", type=str, default=None)
    parser.add_argument("--output_model_path", type=str, default=None)
    parser.add_argument("--summary_output_path", type=str, default=None)
    parser.add_argument("--learn_cluster_weights_path", type=str, default=str(DEFAULT_LEARN_CLUSTER_WEIGHTS_PATH))
    parser.add_argument("--learn_analysis_csv", type=str, default=str(DEFAULT_LEARN_ANALYSIS_CSV))
    parser.add_argument("--learn_rounds_csv", type=str, default=str(DEFAULT_LEARN_ROUNDS_CSV))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--max_batches_per_epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.artifacts_root:
        artifacts_root = Path(args.artifacts_root).resolve()
    else:
        artifacts_root = DEFAULT_ARTIFACTS_ROOT.resolve()

    output_model_path = (
        Path(args.output_model_path).resolve()
        if args.output_model_path
        else (artifacts_root / "models" / DEFAULT_GPU_SEQUENCE_POLICY_MODEL_PATH.name)
    )
    summary_output_path = (
        Path(args.summary_output_path).resolve()
        if args.summary_output_path
        else (artifacts_root / "outputs" / DEFAULT_GPU_SEQUENCE_TRAIN_SUMMARY_PATH.name)
    )
    return output_model_path, summary_output_path


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_model_path, summary_output_path = _resolve_paths(args)
    train_gpu_sequence_policy(
        output_model_path=output_model_path,
        summary_output_path=summary_output_path,
        learn_cluster_weights_path=Path(args.learn_cluster_weights_path).resolve(),
        learn_analysis_csv=Path(args.learn_analysis_csv).resolve(),
        learn_rounds_csv=Path(args.learn_rounds_csv).resolve(),
        device=args.device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        progress_every=int(args.progress_every),
        max_batches_per_epoch=args.max_batches_per_epoch,
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
