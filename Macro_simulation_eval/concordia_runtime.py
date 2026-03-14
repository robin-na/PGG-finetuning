from __future__ import annotations

import difflib
import importlib
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import requests


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class ConcordiaModules:
    basic_associative_memory: Any
    entity_lib: Any
    language_model_lib: Any
    prefab_basic: Any
    prefab_basic_with_plan: Optional[Any]
    prefab_rational: Any


def _import_optional(module_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


def load_concordia_modules(concordia_import_path: Optional[str] = None) -> ConcordiaModules:
    if concordia_import_path:
        repo_root = os.path.abspath(os.path.expanduser(str(concordia_import_path)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
    try:
        return ConcordiaModules(
            basic_associative_memory=importlib.import_module(
                "concordia.associative_memory.basic_associative_memory"
            ),
            entity_lib=importlib.import_module("concordia.typing.entity"),
            language_model_lib=importlib.import_module("concordia.language_model.language_model"),
            prefab_basic=importlib.import_module("concordia.prefabs.entity.basic"),
            prefab_basic_with_plan=_import_optional("concordia.prefabs.entity.basic_with_plan"),
            prefab_rational=importlib.import_module("concordia.prefabs.entity.rational"),
        )
    except ModuleNotFoundError as exc:
        message = (
            "Unable to import `concordia`. Install it into the active environment or pass "
            "`--concordia_import_path /path/to/concordia` pointing at a checkout root."
        )
        raise RuntimeError(message) from exc


class HashingSentenceEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = max(32, int(dim))

    def __call__(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall(str(text or "").lower())
        if not tokens:
            return vector
        for token in tokens:
            idx = hash(token) % self.dim
            sign = 1.0 if (hash(token + "::sign") & 1) == 0 else -1.0
            vector[idx] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector


class OpenAIEmbeddingClient:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        timeout_sec: int = 60,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = int(timeout_sec)

    def __call__(self, text: str) -> np.ndarray:
        payload = {
            "model": self._model,
            "input": str(text or ""),
        }
        response = requests.post(
            f"{self._base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self._timeout_sec,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"OpenAI embeddings error {response.status_code}: {response.text}")
        data = response.json()
        try:
            embedding = data["data"][0]["embedding"]
        except Exception as exc:
            raise RuntimeError(f"Unexpected embeddings response: {data}") from exc
        return np.asarray(embedding, dtype=np.float32)


def _apply_stop_sequences(text: str, stop: Optional[Sequence[str]]) -> str:
    if not stop:
        return text
    earliest_idx = None
    for seq in stop:
        if not seq:
            continue
        idx = text.find(seq)
        if idx != -1 and (earliest_idx is None or idx < earliest_idx):
            earliest_idx = idx
    if earliest_idx is None:
        return text
    return text[:earliest_idx]


def build_sentence_embedder(args: Any, client: Any) -> Any:
    embedder_name = str(getattr(args, "concordia_embedder", "hash") or "hash").strip().lower()
    if embedder_name == "hash":
        return HashingSentenceEmbedder(dim=int(getattr(args, "concordia_hash_dim", 384) or 384))
    if embedder_name == "openai":
        api_key = (
            getattr(args, "openai_api_key", None)
            or getattr(client, "openai_api_key", None)
            or os.getenv(str(getattr(args, "openai_api_key_env", "OPENAI_API_KEY")))
        )
        if not api_key:
            raise ValueError(
                "`--concordia_embedder openai` requires an OpenAI API key via --openai_api_key "
                "or the configured environment variable."
            )
        return OpenAIEmbeddingClient(
            api_key=api_key,
            model=str(
                getattr(args, "concordia_embedding_model", "text-embedding-3-small")
                or "text-embedding-3-small"
            ),
            base_url=str(getattr(client, "openai_base_url", "https://api.openai.com/v1")),
            timeout_sec=int(getattr(client, "openai_timeout_sec", 60) or 60),
        )
    raise ValueError("Unsupported concordia embedder. Use `hash` or `openai`.")


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _match_choice(raw_text: str, responses: Sequence[str]) -> int:
    normalized_raw = _normalize_text(raw_text)
    normalized_responses = [_normalize_text(resp) for resp in responses]
    for idx, normalized in enumerate(normalized_responses):
        if normalized_raw == normalized:
            return idx
    for idx, normalized in enumerate(normalized_responses):
        if normalized and normalized in normalized_raw:
            return idx
    if normalized_raw.isdigit():
        candidate = int(normalized_raw)
        if 0 <= candidate < len(responses):
            return candidate
        if 1 <= candidate <= len(responses):
            return candidate - 1
    scores = [
        difflib.SequenceMatcher(a=normalized_raw, b=normalized).ratio()
        for normalized in normalized_responses
    ]
    if not scores:
        return 0
    return int(max(range(len(scores)), key=lambda idx: scores[idx]))


def build_language_model_adapter(
    language_model_lib: Any,
    client: Any,
    *,
    seed: int = 0,
    default_temperature: float = 0.7,
    default_top_p: float = 1.0,
) -> Any:
    base_cls = language_model_lib.LanguageModel
    concordia_default_max_tokens = int(getattr(language_model_lib, "DEFAULT_MAX_TOKENS", 5000))

    class MacroConcordiaLanguageModel(base_cls):
        def __init__(self):
            self._client = client
            self._seed = int(seed)
            self._default_temperature = float(default_temperature)
            self._default_top_p = float(default_top_p)

        def sample_text(
            self,
            prompt: str,
            *,
            max_tokens: int = concordia_default_max_tokens,
            terminators: Sequence[str] = (),
            temperature: float = 1.0,
            top_p: float = 0.95,
            top_k: int = 64,
            timeout: float = 60,
            seed: Optional[int] = None,
        ) -> str:
            del timeout, top_k
            messages_list = [[{"role": "user", "content": str(prompt)}]]
            provider = str(getattr(self._client, "provider", "") or "").strip().lower()
            terminator_list = list(terminators) if terminators else None
            requested_max_new_tokens = max(1, int(max_tokens))
            if provider == "openai" and requested_max_new_tokens == concordia_default_max_tokens:
                requested_max_new_tokens = None
            outputs = self._client.generate_batch(
                prompts=[str(prompt)],
                messages_list=messages_list,
                stop=None if provider == "openai" else terminator_list,
                max_new_tokens=requested_max_new_tokens,
                temperature=self._default_temperature if temperature is None else float(temperature),
                top_p=self._default_top_p if top_p is None else float(top_p),
                seed=self._seed if seed is None else int(seed),
                async_openai=False,
                max_concurrency=1,
            )
            text = str(outputs[0])
            if provider == "openai":
                text = _apply_stop_sequences(text, terminator_list)
            return text.strip()

        def sample_choice(
            self,
            prompt: str,
            responses: Sequence[str],
            *,
            seed: Optional[int] = None,
        ) -> Tuple[int, str, Dict[str, Any]]:
            choices_text = "\n".join(f"{idx + 1}. {response}" for idx, response in enumerate(responses))
            choice_prompt = (
                f"{prompt.rstrip()}\n\nChoose exactly one option from the list below.\n"
                "Reply with the option text only.\n\n"
                f"{choices_text}"
            )
            raw_text = self.sample_text(
                choice_prompt,
                max_tokens=64,
                terminators=("\n",),
                temperature=0.0,
                seed=seed,
            )
            choice_idx = _match_choice(raw_text, responses)
            return choice_idx, str(responses[choice_idx]), {"raw_text": raw_text}

    return MacroConcordiaLanguageModel()


def get_prefab_module(modules: ConcordiaModules, prefab_name: str) -> Any:
    normalized = str(prefab_name or "rational").strip().lower()
    if normalized == "basic":
        return modules.prefab_basic
    if normalized == "basic_with_plan":
        if modules.prefab_basic_with_plan is None:
            raise RuntimeError(
                "This Concordia installation does not provide `concordia.prefabs.entity.basic_with_plan`."
            )
        return modules.prefab_basic_with_plan
    if normalized == "rational":
        return modules.prefab_rational
    raise ValueError(
        "Unsupported Concordia prefab. Use `basic`, `basic_with_plan`, or `rational`."
    )
