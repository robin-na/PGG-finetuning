from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from Macro_simulation_eval.concordia_simulator import run_macro_simulation_eval_concordia  # noqa: E402
from Macro_simulation_eval.run_macro_simulation_eval_concordia import ConcordiaArgs  # noqa: E402


def test_concordia_args_override_output_defaults():
    args = ConcordiaArgs()

    assert args.output_root.endswith("macro_simulation_eval_concordia")
    assert args.rows_out_path == "output/macro_simulation_eval_concordia.csv"
    assert args.transcripts_out_path == "output/history_transcripts_concordia.jsonl"
    assert args.concordia_agent_prefab == "rational"
    assert args.concordia_embedder == "hash"
    assert args.concordia_hash_dim == 384


def test_concordia_backend_rejects_resume_before_io():
    args = ConcordiaArgs(resume_from_run="existing-run")

    with pytest.raises(NotImplementedError, match="resume_from_run"):
        run_macro_simulation_eval_concordia(args)


def test_concordia_backend_rejects_parallel_games_before_io():
    args = ConcordiaArgs(max_parallel_games=2)

    with pytest.raises(NotImplementedError, match="max_parallel_games 1"):
        run_macro_simulation_eval_concordia(args)
