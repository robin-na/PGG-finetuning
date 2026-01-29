from __future__ import annotations

import json
import os
import sys
import warnings
from typing import Any, Dict, List, Optional

import asyncio

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


def _apply_stop_sequences(text: str, stop: Optional[List[str]]) -> str:
    if not stop:
        return text
    earliest_idx = None
    earliest_seq = None
    for seq in stop:
        if not seq:
            continue
        idx = text.find(seq)
        if idx != -1 and (earliest_idx is None or idx < earliest_idx):
            earliest_idx = idx
            earliest_seq = seq
    if earliest_idx is None or earliest_seq is None:
        return text
    return text[: earliest_idx + len(earliest_seq)]


def _require_openai_key(env_name: str) -> str:
    api_key = os.getenv(env_name)
    if api_key:
        return api_key
    print(f"ERROR: {env_name} environment variable not set!")
    print()
    print("Please set your API key:")
    print(f"  export {env_name}='your-api-key-here'")
    print()
    sys.exit(1)


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
    stopping_criteria = None
    if stop:
        class _StopOnSequences(StoppingCriteria):
            def __init__(self, tokenizer: AutoTokenizer, stop_sequences: List[str], prompt_lens: List[int]):
                self.tokenizer = tokenizer
                self.stop_sequences = stop_sequences
                self.prompt_lens = prompt_lens

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
                for idx, seq in enumerate(input_ids):
                    prompt_len = self.prompt_lens[idx]
                    decoded = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                    for stop_seq in self.stop_sequences:
                        if stop_seq and stop_seq in decoded:
                            return True
                return False

        stopping_criteria = StoppingCriteriaList([_StopOnSequences(tok, stop, prompt_lengths)])

    gen_out = model.generate(
        **inputs,
        do_sample=temperature > 0.0,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
        stopping_criteria=stopping_criteria,
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
        openai_model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_api_key_env: str = "OPENAI_API_KEY",
        openai_base_url: str = "https://api.openai.com/v1",
        openai_timeout_sec: int = 60,
    ):
        self.provider = (provider or "local").lower()
        self.tok = tok
        self.model = model
        self.openai_model = openai_model
        self.openai_api_key_env = openai_api_key_env
        self.openai_base_url = openai_base_url.rstrip("/")
        self.openai_timeout_sec = openai_timeout_sec
        if self.provider == "openai":
            self.openai_api_key = openai_api_key or _require_openai_key(openai_api_key_env)
            if not self.openai_model:
                raise ValueError("Missing OpenAI model name: set --openai_model.")
        elif self.provider == "local":
            self.openai_api_key = None
        else:
            raise ValueError(f"Unsupported provider '{provider}'. Use 'local' or 'openai'.")

    def _openai_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.openai_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
        }
        if stop:
            payload["stop"] = stop
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            f"{self.openai_base_url}/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=self.openai_timeout_sec,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text}")
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Unexpected OpenAI response: {data}") from exc
        return _apply_stop_sequences(content, stop)

    def generate_batch(
        self,
        prompts: Optional[List[str]],
        messages_list: Optional[List[List[Dict[str, str]]]],
        stop: Optional[List[str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        async_openai: bool = False,
        max_concurrency: int = 8,
    ) -> List[str]:
        if self.provider == "openai":
            if messages_list is None:
                raise ValueError("OpenAI provider requires messages_list.")
            if not async_openai:
                outputs = []
                for messages in messages_list:
                    outputs.append(
                        self._openai_chat_completion(
                            messages=messages,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            stop=stop,
                        )
                    )
                return outputs

            async def _run_async() -> List[str]:
                semaphore = asyncio.Semaphore(max_concurrency)

                async def _run_one(messages: List[Dict[str, str]]) -> str:
                    async with semaphore:
                        return await asyncio.to_thread(
                            self._openai_chat_completion,
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
