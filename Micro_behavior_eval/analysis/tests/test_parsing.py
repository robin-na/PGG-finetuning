from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Micro_behavior_eval.analysis.io_utils import parse_action_columns, parse_action_dict


def test_parse_action_dict_json_and_literal():
    parsed_json, malformed_json = parse_action_dict('{"p1": 2, "p2": "3"}')
    parsed_lit, malformed_lit = parse_action_dict("{'p3': 4}")

    assert malformed_json is False
    assert malformed_lit is False
    assert parsed_json == {"p1": 2, "p2": 3}
    assert parsed_lit == {"p3": 4}


def test_parse_action_dict_malformed_to_empty():
    parsed, malformed = parse_action_dict("{not_valid")
    assert parsed == {}
    assert malformed is True


def test_parse_action_columns_tracks_malformed_counts():
    df = pd.DataFrame(
        {
            "predicted_punished_pid": ['{"a": 1}', "{bad"],
            "predicted_rewarded_pid": ["{}", "{}"],
            "actual_punished_pid": ['{"a": 1}', '{"b": 1}'],
            "actual_rewarded_pid": ["{}", "{}"],
        }
    )
    out, summary = parse_action_columns(df)

    assert "predicted_punished_pid_dict" in out.columns
    assert summary.malformed_counts["predicted_punished_pid"] == 1
    assert summary.malformed_rows == 1
