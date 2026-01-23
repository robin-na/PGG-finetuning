from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _apply_stop(text: str, stop: Optional[str]) -> str:
    if stop:
        cut_idx = text.find(stop)
        if cut_idx != -1:
            return text[:cut_idx]
    return text


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
    stop: Optional[str] = None,          # e.g., ']' for arrays
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

    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_out = model.generate(
        **inputs,
        do_sample=temperature > 0.0,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    )

    out_texts = []
    for i in range(len(prompts)):
        prompt_len = inputs["input_ids"][i].shape[-1]
        gen_ids = gen_out[i][prompt_len:]
        new_text = tok.decode(gen_ids, skip_special_tokens=True)
        if stop:
            cut_idx = new_text.find(stop)
            if cut_idx != -1:
                new_text = new_text[:cut_idx]
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
        stop: Optional[str],
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
        return _apply_stop(content, stop)

    def generate_batch(
        self,
        prompts: Optional[List[str]],
        messages_list: Optional[List[List[Dict[str, str]]]],
        stop: Optional[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> List[str]:
        if self.provider == "openai":
            if messages_list is None:
                raise ValueError("OpenAI provider requires messages_list.")
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
        if prompts is None or self.tok is None or self.model is None:
            raise ValueError("Local provider requires prompts and a loaded model/tokenizer.")
        return _batch_generate_until(
            self.tok,
            self.model,
            prompts,
            stop=stop,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
