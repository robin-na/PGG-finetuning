from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecasting.non_pgg_batch_builder import main as common_main


if __name__ == "__main__":
    common_main(
        [
            "--dataset-key",
            "two_stage_trust_punishment_y2hgu",
            "--forecasting-root",
            str(SCRIPT_DIR),
            "--max-records-per-treatment",
            "50",
        ]
    )
