from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], cwd: Path) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def run_completed_eval(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.expanduser().resolve()
    metadata_dir = args.metadata_dir.expanduser().resolve()
    output_jsonl = args.output_jsonl.expanduser().resolve()
    if not output_jsonl.is_file():
        raise FileNotFoundError(f"Batch output not found: {output_jsonl}")

    python = sys.executable
    _run(
        [
            python,
            "-B",
            "-m",
            "forecasting.persona_transfer_audit.parse_match_outputs",
            "--metadata-dir",
            str(metadata_dir),
            "--output-jsonl",
            str(output_jsonl),
        ],
        cwd=repo_root,
    )
    _run(
        [
            python,
            "-B",
            "-m",
            "forecasting.persona_transfer_audit.comprehensive_eval",
            "--metadata-dir",
            str(metadata_dir),
        ],
        cwd=repo_root,
    )
    _run(
        [
            python,
            "-B",
            "-m",
            "forecasting.persona_transfer_audit.summarize_matches",
            "--metadata-dir",
            str(metadata_dir),
        ],
        cwd=repo_root,
    )
    _run(
        [
            python,
            "-B",
            "-m",
            "forecasting.persona_transfer_audit.evaluate_matches",
            "--metadata-dir",
            str(metadata_dir),
        ],
        cwd=repo_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run parse, concentration summary, and behavior evaluation for completed match output."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run_completed_eval(parse_args())


if __name__ == "__main__":
    main()
