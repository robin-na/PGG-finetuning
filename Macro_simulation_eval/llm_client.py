from __future__ import annotations

import json
import os
import sys
import warnings
from typing import Any, Dict, List, Optional

import asyncio

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from repo_env import get_env_var


def _apply_stop_sequences(text: str, stop: Optional[List[str]]) -> str:
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


def _require_openai_key(env_name: str) -> str:
    api_key = get_env_var(env_name)
    if api_key:
        return api_key
    print(f"ERROR: {env_name} environment variable not set!")
    print()
    print("Please set your API key or add it to .api_keys.env:")
    print(f"  export {env_name}='your-api-key-here'")
    print()
    sys.exit(1)


def _extract_message_content(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts: List[str] = []
        for item in message:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(message)


class _ForceEosAfterStopSequences(LogitsProcessor):
    """Stop completed rows individually without truncating the whole batch."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        stop_sequences: List[str],
        prompt_lens: List[int],
        eos_token_id: Optional[int],
    ):
        self.tokenizer = tokenizer
        self.stop_sequences = [seq for seq in stop_sequences if seq]
        self.prompt_lens = prompt_lens
        self.eos_token_id = eos_token_id
        self.finished = [False] * len(prompt_lens)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.stop_sequences or self.eos_token_id is None:
            return scores

        min_score = torch.finfo(scores.dtype).min
        for idx, seq in enumerate(input_ids):
            if not self.finished[idx]:
                prompt_len = self.prompt_lens[idx]
                decoded = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                if any(stop_seq in decoded for stop_seq in self.stop_sequences):
                    self.finished[idx] = True
            if self.finished[idx]:
                scores[idx, :] = min_score
                scores[idx, self.eos_token_id] = 0
        return scores


@torch.inference_mode()
def _batch_generate_until(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompts: List[str],
    stop: Optional[List[str]] = None,  # e.g., ['</TAG>'] for XML-like outputs
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 0,
) -> List[str]:
    """
    Generate continuations for a batch of prompts and (optionally) cut at the first occurrence of `stop`.
    Returns the raw generated strings (decoded new text only, not including the prompts).
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    max_length = getattr(tok, "model_max_length", None)
    use_truncation = max_length is not None and max_length < 1_000_000
    tokenization_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": use_truncation,
    }
    if use_truncation:
        tokenization_kwargs["max_length"] = max_length
    inputs = tok(prompts, **tokenization_kwargs)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    prompt_lengths = [inputs["input_ids"][i].shape[-1] for i in range(len(prompts))]
    logits_processor = None
    forced_eos_token_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    if stop and forced_eos_token_id is not None:
        logits_processor = LogitsProcessorList(
            [_ForceEosAfterStopSequences(tok, stop, prompt_lengths, forced_eos_token_id)]
        )

    gen_out = model.generate(
        **inputs,
        do_sample=temperature > 0.0,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
        logits_processor=logits_processor,
    )

    out_texts = []
    for i in range(len(prompts)):
        prompt_len = prompt_lengths[i]
        gen_ids = gen_out[i][prompt_len:]
        new_text = tok.decode(gen_ids, skip_special_tokens=True)
        new_text = _apply_stop_sequences(new_text, stop)
        out_texts.append(new_text)
    return out_texts


class LLMClient:
    def __init__(
        self,
        provider: str,
        tok: Optional[AutoTokenizer] = None,
        model: Optional[AutoModelForCausalLM] = None,
        base_model: Optional[str] = None,
        openai_model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_api_key_env: str = "OPENAI_API_KEY",
        openai_base_url: str = "https://api.openai.com/v1",
        openai_timeout_sec: int = 60,
        vllm_model: Optional[str] = None,
        vllm_api_key: Optional[str] = None,
        vllm_api_key_env: str = "VLLM_API_KEY",
        vllm_base_url: str = "http://localhost:8000/v1",
        vllm_timeout_sec: int = 60,
    ):
        self.provider = (provider or "local").lower()
        self.tok = tok
        self.model = model
        self.base_model = base_model
        self.openai_model = openai_model
        self.openai_api_key_env = openai_api_key_env
        self.openai_base_url = openai_base_url.rstrip("/")
        self.openai_timeout_sec = openai_timeout_sec
        self.vllm_model = vllm_model or base_model or openai_model
        self.vllm_api_key_env = vllm_api_key_env
        self.vllm_base_url = vllm_base_url.rstrip("/")
        self.vllm_timeout_sec = vllm_timeout_sec
        if self.provider == "openai":
            self.openai_api_key = openai_api_key or _require_openai_key(openai_api_key_env)
            if not self.openai_model:
                raise ValueError("Missing OpenAI model name: set --openai_model.")
            self.vllm_api_key = None
        elif self.provider == "vllm":
            self.openai_api_key = None
            self.vllm_api_key = vllm_api_key or get_env_var(vllm_api_key_env) or "EMPTY"
            if not self.vllm_model:
                raise ValueError("Missing vLLM model name: set --vllm_model or --base_model.")
        elif self.provider == "local":
            self.openai_api_key = None
            self.vllm_api_key = None
        else:
            raise ValueError(f"Unsupported provider '{provider}'. Use 'local', 'openai', or 'vllm'.")

    def _remote_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int],
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> str:
        if self.provider == "openai":
            model_name = self.openai_model
            api_key = self.openai_api_key
            base_url = self.openai_base_url
            timeout_sec = self.openai_timeout_sec
            backend_name = "OpenAI"
        elif self.provider == "vllm":
            model_name = self.vllm_model
            api_key = self.vllm_api_key
            base_url = self.vllm_base_url
            timeout_sec = self.vllm_timeout_sec
            backend_name = "vLLM"
        else:
            raise ValueError(f"Remote chat completion is unavailable for provider '{self.provider}'.")

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_new_tokens is not None:
            if self.provider == "openai":
                payload["max_completion_tokens"] = int(max_new_tokens)
            else:
                payload["max_tokens"] = int(max_new_tokens)
        if stop:
            payload["stop"] = stop
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        ignored_params = set()
        while True:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout_sec,
            )
            if resp.status_code < 400:
                break
            if self.provider == "openai":
                try:
                    error_payload = resp.json()
                except Exception:
                    error_payload = None
                error_info = error_payload.get("error", {}) if isinstance(error_payload, dict) else {}
                unsupported_param = error_info.get("param")
                if (
                    resp.status_code == 400
                    and error_info.get("code") == "unsupported_parameter"
                    and isinstance(unsupported_param, str)
                    and unsupported_param in payload
                    and unsupported_param not in ignored_params
                ):
                    ignored_params.add(unsupported_param)
                    payload.pop(unsupported_param, None)
                    continue
            raise RuntimeError(f"{backend_name} API error {resp.status_code}: {resp.text}")
        data = resp.json()
        try:
            content = _extract_message_content(data["choices"][0]["message"]["content"])
        except Exception as exc:
            raise RuntimeError(f"Unexpected {backend_name} response: {data}") from exc
        return _apply_stop_sequences(content, stop)

    def generate_batch(
        self,
        prompts: Optional[List[str]],
        messages_list: Optional[List[List[Dict[str, str]]]],
        stop: Optional[List[str]],
        max_new_tokens: Optional[int],
        temperature: float,
        top_p: float,
        seed: int,
        async_openai: bool = False,
        max_concurrency: int = 8,
    ) -> List[str]:
        if self.provider in {"openai", "vllm"}:
            if messages_list is None:
                raise ValueError(f"{self.provider} provider requires messages_list.")
            use_async_requests = bool(async_openai or self.provider == "vllm")
            bounded_concurrency = max(1, int(max_concurrency))
            if not use_async_requests or bounded_concurrency <= 1:
                outputs = []
                for messages in messages_list:
                    outputs.append(
                        self._remote_chat_completion(
                            messages=messages,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            stop=stop,
                        )
                    )
                return outputs

            async def _run_async() -> List[str]:
                semaphore = asyncio.Semaphore(bounded_concurrency)

                async def _run_one(messages: List[Dict[str, str]]) -> str:
                    async with semaphore:
                        return await asyncio.to_thread(
                            self._remote_chat_completion,
                            messages=messages,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            stop=stop,
                        )

                tasks = [_run_one(messages) for messages in messages_list]
                return await asyncio.gather(*tasks)

            return asyncio.run(_run_async())
        if prompts is None or self.tok is None or self.model is None:
            raise ValueError("Local provider requires prompts and a loaded model/tokenizer.")
        if max_new_tokens is None:
            raise ValueError("Local provider requires max_new_tokens.")
        chunk_size = len(prompts)
        env_chunk_size = os.getenv("PGG_LOCAL_GENERATION_CHUNK_SIZE")
        if env_chunk_size:
            try:
                env_value = int(env_chunk_size)
            except ValueError:
                warnings.warn(
                    "Ignoring invalid PGG_LOCAL_GENERATION_CHUNK_SIZE "
                    f"value '{env_chunk_size}'; expected a positive integer."
                )
            else:
                if env_value > 0:
                    chunk_size = min(chunk_size, env_value)
        while True:
            try:
                outputs: List[str] = []
                for start in range(0, len(prompts), chunk_size):
                    chunk_prompts = prompts[start : start + chunk_size]
                    outputs.extend(
                        _batch_generate_until(
                            self.tok,
                            self.model,
                            chunk_prompts,
                            stop=stop,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            seed=seed,
                        )
                    )
                return outputs
            except torch.OutOfMemoryError as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if chunk_size <= 1:
                    raise exc
                reduced_size = max(1, chunk_size // 2)
                warnings.warn(
                    "CUDA OOM during generation with batch size "
                    f"{chunk_size}; retrying with reduced chunk size {reduced_size}."
                )
                chunk_size = reduced_size
