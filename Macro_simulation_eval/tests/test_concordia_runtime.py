from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from Macro_simulation_eval.concordia_runtime import (  # noqa: E402
    ConcordiaModules,
    OpenAIEmbeddingClient,
    build_language_model_adapter,
    build_sentence_embedder,
    get_prefab_module,
)


class FakeLanguageModelBase(ABC):
    @abstractmethod
    def sample_text(self, prompt, *, max_tokens=256, terminators=(), temperature=0.0, timeout=-1, seed=None):
        raise NotImplementedError

    @abstractmethod
    def sample_choice(self, prompt, responses, *, seed=None):
        raise NotImplementedError


class FakeClient:
    def __init__(self, outputs, provider="local"):
        self.outputs = list(outputs)
        self.provider = provider
        self.calls = []

    def generate_batch(
        self,
        prompts,
        messages_list,
        stop,
        max_new_tokens,
        temperature,
        top_p,
        seed,
        async_openai=False,
        max_concurrency=1,
    ):
        self.calls.append(
            {
                "prompts": prompts,
                "messages_list": messages_list,
                "stop": stop,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "async_openai": async_openai,
                "max_concurrency": max_concurrency,
            }
        )
        if not self.outputs:
            raise AssertionError("No fake outputs remaining.")
        return [self.outputs.pop(0)]


def test_build_language_model_adapter_subclasses_concordia_base():
    client = FakeClient(outputs=["trim me  ", "dog"])
    adapter = build_language_model_adapter(
        SimpleNamespace(LanguageModel=FakeLanguageModelBase),
        client,
        seed=17,
        default_temperature=0.4,
        default_top_p=0.8,
    )

    assert isinstance(adapter, FakeLanguageModelBase)
    assert (
        adapter.sample_text(
            "hello",
            max_tokens=7,
            terminators=("STOP",),
            temperature=None,
            top_p=None,
            top_k=16,
        )
        == "trim me"
    )

    choice_idx, choice_text, metadata = adapter.sample_choice("Pick one", ["cat", "dog"], seed=23)

    assert choice_idx == 1
    assert choice_text == "dog"
    assert metadata == {"raw_text": "dog"}
    assert client.calls[0]["max_new_tokens"] == 7
    assert client.calls[0]["stop"] == ["STOP"]
    assert client.calls[0]["temperature"] == 0.4
    assert client.calls[0]["top_p"] == 0.8
    assert client.calls[0]["seed"] == 17
    assert client.calls[1]["seed"] == 23
    assert client.calls[1]["temperature"] == 0.0


def test_build_language_model_adapter_omits_default_max_tokens_for_openai():
    client = FakeClient(outputs=["trim me  "], provider="openai")
    adapter = build_language_model_adapter(
        SimpleNamespace(LanguageModel=FakeLanguageModelBase, DEFAULT_MAX_TOKENS=5000),
        client,
        seed=17,
        default_temperature=0.4,
        default_top_p=0.8,
    )

    assert adapter.sample_text("hello") == "trim me"
    assert client.calls[0]["max_new_tokens"] is None


def test_build_language_model_adapter_omits_stop_for_openai_and_trims_locally():
    client = FakeClient(outputs=["answer<END> trailing"], provider="openai")
    adapter = build_language_model_adapter(
        SimpleNamespace(LanguageModel=FakeLanguageModelBase, DEFAULT_MAX_TOKENS=5000),
        client,
        seed=17,
        default_temperature=0.4,
        default_top_p=0.8,
    )

    assert adapter.sample_text("hello", terminators=("<END>",)) == "answer"
    assert client.calls[0]["stop"] is None


def test_get_prefab_module_handles_optional_basic_with_plan():
    modules = ConcordiaModules(
        basic_associative_memory=object(),
        entity_lib=object(),
        language_model_lib=object(),
        prefab_basic="basic-module",
        prefab_basic_with_plan=None,
        prefab_rational="rational-module",
    )

    assert get_prefab_module(modules, "basic") == "basic-module"
    assert get_prefab_module(modules, "rational") == "rational-module"
    with pytest.raises(RuntimeError):
        get_prefab_module(modules, "basic_with_plan")


def test_build_sentence_embedder_uses_explicit_openai_key_even_for_non_openai_provider():
    args = SimpleNamespace(
        concordia_embedder="openai",
        concordia_embedding_model="text-embedding-3-small",
        openai_api_key="test-key",
        openai_api_key_env="OPENAI_API_KEY",
    )
    client = SimpleNamespace(openai_api_key=None, openai_base_url="https://api.openai.com/v1", openai_timeout_sec=60)

    embedder = build_sentence_embedder(args, client)

    assert isinstance(embedder, OpenAIEmbeddingClient)
