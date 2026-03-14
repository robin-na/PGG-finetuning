import importlib
import json
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


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def test_macro_openai_chat_completion_uses_max_completion_tokens(monkeypatch):
    module = importlib.import_module("Macro_simulation_eval.llm_client")
    captured = {}

    def fake_post(url, headers=None, data=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json.loads(data)
        captured["timeout"] = timeout
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(module.requests, "post", fake_post)

    client = module.LLMClient(
        provider="openai",
        openai_model="gpt-5-mini",
        openai_api_key="test-key",
    )

    output = client.generate_batch(
        prompts=None,
        messages_list=[[{"role": "user", "content": "hello"}]],
        stop=None,
        max_new_tokens=17,
        temperature=0.7,
        top_p=0.9,
        seed=0,
    )

    assert output == ["ok"]
    assert captured["payload"]["model"] == "gpt-5-mini"
    assert captured["payload"]["max_completion_tokens"] == 17
    assert "max_tokens" not in captured["payload"]


def test_macro_vllm_chat_completion_keeps_max_tokens(monkeypatch):
    module = importlib.import_module("Macro_simulation_eval.llm_client")
    captured = {}

    def fake_post(url, headers=None, data=None, timeout=None):
        captured["payload"] = json.loads(data)
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(module.requests, "post", fake_post)

    client = module.LLMClient(
        provider="vllm",
        vllm_model="some-served-model",
        vllm_api_key="EMPTY",
    )

    output = client.generate_batch(
        prompts=None,
        messages_list=[[{"role": "user", "content": "hello"}]],
        stop=None,
        max_new_tokens=23,
        temperature=0.7,
        top_p=0.9,
        seed=0,
    )

    assert output == ["ok"]
    assert captured["payload"]["model"] == "some-served-model"
    assert captured["payload"]["max_tokens"] == 23
    assert "max_completion_tokens" not in captured["payload"]


def test_macro_openai_chat_completion_retries_without_unsupported_param(monkeypatch):
    module = importlib.import_module("Macro_simulation_eval.llm_client")
    payloads = []

    def fake_post(url, headers=None, data=None, timeout=None):
        payload = json.loads(data)
        payloads.append(payload)
        if len(payloads) == 1:
            return _FakeResponse(
                {
                    "error": {
                        "message": "Unsupported parameter: 'top_p' is not supported with this model.",
                        "type": "invalid_request_error",
                        "param": "top_p",
                        "code": "unsupported_parameter",
                    }
                },
                status_code=400,
            )
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(module.requests, "post", fake_post)

    client = module.LLMClient(
        provider="openai",
        openai_model="gpt-5-mini",
        openai_api_key="test-key",
    )

    output = client.generate_batch(
        prompts=None,
        messages_list=[[{"role": "user", "content": "hello"}]],
        stop=None,
        max_new_tokens=17,
        temperature=0.7,
        top_p=0.9,
        seed=0,
    )

    assert output == ["ok"]
    assert len(payloads) == 2
    assert payloads[0]["top_p"] == 0.9
    assert "top_p" not in payloads[1]
