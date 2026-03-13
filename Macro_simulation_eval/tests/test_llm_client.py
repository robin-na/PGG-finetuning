import importlib
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class DummyTokenizer:
    def __init__(self):
        self.eos_token_id = 9
        self.pad_token_id = 0
        self._pieces = {
            0: "",
            1: "P",
            2: "Q",
            3: "body",
            4: "<<END_JSON>>",
        }

    def decode(self, ids, skip_special_tokens=True):
        pieces = []
        for token_id in ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id in {self.eos_token_id, self.pad_token_id}:
                continue
            pieces.append(self._pieces.get(token_id, f"<{token_id}>"))
        return "".join(pieces)


@pytest.mark.parametrize(
    "module_name",
    [
        "Macro_simulation_eval.llm_client",
        "Simulation_robin.llm_client",
    ],
)
def test_force_eos_after_stop_sequences_only_finishes_matching_rows(module_name):
    module = importlib.import_module(module_name)
    processor = module._ForceEosAfterStopSequences(
        tokenizer=DummyTokenizer(),
        stop_sequences=["<<END_JSON>>"],
        prompt_lens=[2, 2],
        eos_token_id=9,
    )
    input_ids = torch.tensor(
        [
            [1, 2, 4],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )
    scores = torch.zeros((2, 10), dtype=torch.float32)

    updated = processor(input_ids, scores.clone())
    min_score = torch.finfo(updated.dtype).min

    assert updated[0, 9].item() == 0
    assert torch.all(updated[0, :9] == min_score)
    assert torch.all(updated[1] == 0)


@pytest.mark.parametrize(
    "module_name",
    [
        "Macro_simulation_eval.llm_client",
        "Simulation_robin.llm_client",
    ],
)
def test_force_eos_after_stop_sequences_ignores_stop_text_inside_prompt(module_name):
    module = importlib.import_module(module_name)
    processor = module._ForceEosAfterStopSequences(
        tokenizer=DummyTokenizer(),
        stop_sequences=["<<END_JSON>>"],
        prompt_lens=[2],
        eos_token_id=9,
    )
    input_ids = torch.tensor([[4, 4, 3]], dtype=torch.long)
    scores = torch.zeros((1, 10), dtype=torch.float32)

    updated = processor(input_ids, scores.clone())

    assert torch.all(updated == 0)
