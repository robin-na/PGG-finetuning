from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Micro_behavior_eval.analysis.metrics import aggregate_scores, score_rows


def test_aggregation_on_toy_data():
    df = pd.DataFrame(
        [
            {
                "gameId": "g1",
                "roundIndex": 1,
                "playerId": "p1",
                "predicted_contribution": 10,
                "actual_contribution": 8,
                "predicted_contribution_parsed_bool": True,
                "predicted_punished_pid_dict": {},
                "predicted_rewarded_pid_dict": {},
                "actual_punished_pid_dict": {},
                "actual_rewarded_pid_dict": {},
            },
            {
                "gameId": "g1",
                "roundIndex": 1,
                "playerId": "p2",
                "predicted_contribution": 4,
                "actual_contribution": 8,
                "predicted_contribution_parsed_bool": False,
                "predicted_punished_pid_dict": {"p9": 2},
                "predicted_rewarded_pid_dict": {},
                "actual_punished_pid_dict": {"p9": 1},
                "actual_rewarded_pid_dict": {},
            },
        ]
    )
    scored = score_rows(df)
    overall = aggregate_scores(scored).iloc[0]

    assert overall["n_rows"] == 2
    assert overall["predicted_contribution_parsed"] == pytest.approx(0.5)
    assert overall["contrib_mae"] == pytest.approx(3.0)
    assert overall["contrib_rmse"] == pytest.approx(10 ** 0.5)
    assert overall["contrib_bias"] == pytest.approx(-1.0)
    assert overall["action_exact_match"] == pytest.approx(0.5)
    assert overall["target_f1"] == pytest.approx(1.0)
    assert overall["target_hit_any"] == pytest.approx(1.0)
    assert overall["unit_mae_on_overlap"] == pytest.approx(1.0)


def test_cli_smoke_with_real_subset(tmp_path: Path):
    repo_root = REPO_ROOT
    candidates = sorted((repo_root / "Micro_behavior_eval" / "output").glob("*/micro_behavior_eval.csv"))
    if not candidates:
        pytest.skip("No real micro_behavior_eval.csv found under Micro_behavior_eval/output.")

    subset_df = pd.read_csv(candidates[0]).head(20)
    eval_csv = tmp_path / "subset_micro_eval.csv"
    subset_df.to_csv(eval_csv, index=False)

    analysis_root = tmp_path / "analysis_results"
    cmd = [
        sys.executable,
        str(repo_root / "Micro_behavior_eval" / "analysis" / "run_analysis.py"),
        "--eval_csv",
        str(eval_csv),
        "--analysis_root",
        str(analysis_root),
        "--analysis_run_id",
        "smoke_test",
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

    out_dir = analysis_root / "smoke_test"
    assert (out_dir / "metrics_overall.csv").exists()
    assert (out_dir / "metrics_by_round.csv").exists()
    assert (out_dir / "metrics_by_game.csv").exists()
    assert (out_dir / "row_level_scored.csv").exists()
    assert (out_dir / "analysis_manifest.json").exists()
