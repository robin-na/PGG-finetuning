from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Micro_behavior_eval.analysis.metrics import score_rows


def _base_row() -> dict:
    return {
        "gameId": "g1",
        "roundIndex": 1,
        "playerId": "p0",
        "predicted_contribution": 5,
        "actual_contribution": 5,
        "predicted_contribution_parsed_bool": True,
        "predicted_punished_pid_dict": {},
        "predicted_rewarded_pid_dict": {},
        "actual_punished_pid_dict": {},
        "actual_rewarded_pid_dict": {},
    }


def test_action_exact_match():
    row = _base_row()
    row["predicted_punished_pid_dict"] = {"p1": 2}
    row["actual_punished_pid_dict"] = {"p1": 2}
    scored = score_rows(pd.DataFrame([row])).iloc[0]

    assert scored["action_exact_match"] == 1.0
    assert scored["target_set_exact"] == 1.0
    assert scored["target_f1"] == 1.0
    assert scored["target_hit_any"] == 1.0
    assert scored["unit_mae_on_overlap"] == 0.0
    assert scored["unit_exact_on_overlap_rate"] == 1.0


def test_correct_target_wrong_unit_counts_for_target_metrics():
    row = _base_row()
    row["predicted_punished_pid_dict"] = {"p1": 3}
    row["actual_punished_pid_dict"] = {"p1": 1}
    scored = score_rows(pd.DataFrame([row])).iloc[0]

    assert scored["action_exact_match"] == 0.0
    assert scored["target_set_exact"] == 1.0
    assert scored["target_precision"] == 1.0
    assert scored["target_recall"] == 1.0
    assert scored["target_f1"] == 1.0
    assert scored["target_hit_any"] == 1.0
    assert scored["unit_mae_on_overlap"] == 2.0
    assert scored["unit_exact_on_overlap_rate"] == 0.0


def test_typed_mismatch_punish_vs_reward_is_miss():
    row = _base_row()
    row["predicted_rewarded_pid_dict"] = {"p1": 1}
    row["actual_punished_pid_dict"] = {"p1": 1}
    scored = score_rows(pd.DataFrame([row])).iloc[0]

    assert scored["action_exact_match"] == 0.0
    assert scored["target_set_exact"] == 0.0
    assert scored["target_precision"] == 0.0
    assert scored["target_recall"] == 0.0
    assert scored["target_f1"] == 0.0
    assert scored["target_jaccard"] == 0.0
    assert scored["target_hit_any"] == 0.0
