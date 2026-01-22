"""
LLM client for OpenAI API integration with retry logic.

This module provides a wrapper around the OpenAI API with robust error handling,
exponential backoff for rate limits, and cost tracking.
"""

from dataclasses import dataclass
import os
import time
from typing import Optional

from openai import OpenAI, RateLimitError, APIError

# Generic system prompt for all agents
SYSTEM_PROMPT = """You are a participant in an economic game theory experiment. Your goal is to maximize your total earnings.

You will be given game rules and historical information. Based on this information, make decisions that you believe will maximize your payoff.

Follow these guidelines:
1. Read the rules carefully
2. Consider both short-term and long-term consequences
3. Think about how other players might respond to your actions
4. Output your decisions in the exact format requested

Be strategic but realistic in your choices."""


@dataclass
class BackendResult:
    """Container for backend responses and usage metadata."""

    text: str
    tokens_used: int = 0
    cost_usd: float = 0.0


class BaseLLMBackend:
    """Interface for LLM backends."""

    def call(
        self,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int
    ) -> BackendResult:
        """Execute a request against the backend."""
        raise NotImplementedError


class OpenAIBackend(BaseLLMBackend):
    """OpenAI-backed implementation."""

    def __init__(self, model: str, temperature: float, api_key: Optional[str]):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)

    def call(
        self,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int
    ) -> BackendResult:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=max_tokens
        )

        result_text = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens

        if 'gpt-4o' in self.model.lower():
            cost_usd = tokens_used * 0.006 / 1000
        elif 'gpt-4' in self.model.lower():
            cost_usd = tokens_used * 0.045 / 1000
        else:
            cost_usd = tokens_used * 0.001 / 1000

        return BackendResult(text=result_text, tokens_used=tokens_used, cost_usd=cost_usd)


class VLLMBackend(BaseLLMBackend):
    """vLLM OpenAI-compatible backend."""

    def __init__(
        self,
        model: str,
        temperature: float,
        api_key: Optional[str],
        base_url: Optional[str]
    ):
        # Configuration:
        # - LLM_BACKEND=vllm
        # - VLLM_BASE_URL=http://localhost:8000/v1 (OpenAI-compatible endpoint)
        # - VLLM_API_KEY=... (optional; vLLM commonly ignores this)
        self.model = model
        self.temperature = temperature
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("VLLM_API_KEY") or "EMPTY"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def call(
        self,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int
    ) -> BackendResult:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=max_tokens
        )

        result_text = response.choices[0].message.content.strip()
        tokens_used = getattr(response.usage, "total_tokens", 0) or 0
        return BackendResult(text=result_text, tokens_used=tokens_used, cost_usd=0.0)


class HFBackend(BaseLLMBackend):
    """Hugging Face Inference backend using huggingface_hub."""

    def __init__(
        self,
        model: Optional[str],
        temperature: float,
        token: Optional[str],
        peft_adapter_path: Optional[str] = None
    ):
        # Configuration:
        # - LLM_BACKEND=hf
        # - HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
        # - HF_TOKEN=... (required for gated/private models)
        # - HF_PEFT_PATH=/path/to/adapter (optional; enables local PEFT loading)

        self.model = model or os.getenv("HF_MODEL_ID")
        if not self.model:
            raise ValueError(
                "HF model not found. Please set HF_MODEL_ID environment variable "
                "or pass model parameter."
            )
        self.temperature = temperature
        self.peft_adapter_path = peft_adapter_path or os.getenv("HF_PEFT_PATH")
        if self.peft_adapter_path:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            base_model = AutoModelForCausalLM.from_pretrained(self.model)
            self.model_runner = PeftModel.from_pretrained(base_model, self.peft_adapter_path)
            self.model_runner.eval()
            if torch.cuda.is_available():
                self.model_runner.to("cuda")
        else:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(model=self.model, token=token or os.getenv("HF_TOKEN"))

    def call(
        self,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int
    ) -> BackendResult:
        # Prefer chat-completions when available so system/user roles are preserved.
        if self.peft_adapter_path:
            prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if hasattr(self.model_runner, "device"):
                inputs = {key: value.to(self.model_runner.device) for key, value in inputs.items()}
            outputs = self.model_runner.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature
            )
            result_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result_text = result_text.split("Assistant:", maxsplit=1)[-1].strip()
        elif hasattr(self.client, "chat_completion"):
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=self.temperature
            )
            result_text = response.choices[0].message.content.strip()
        else:
            # Fallback for text-generation models that do not support chat:
            # keep a lightweight, explicit prompt format.
            prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            response = self.client.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0
            )
            result_text = response.strip()

        return BackendResult(text=result_text, tokens_used=0, cost_usd=0.0)


class LLMClient:
    """Client for making LLM API calls with retry logic.

    This class handles:
    - Backend selection (OpenAI, vLLM, Hugging Face)
    - Request execution with exponential backoff on rate limits
    - Error handling and logging
    - Token usage tracking
    """

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 1.0,
        api_key: Optional[str] = None,
        backend: Optional[str] = None,
        vllm_base_url: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_peft_path: Optional[str] = None
    ):
        """Initialize LLM client.

        Args:
            model: OpenAI model name (e.g., "gpt-4", "gpt-4o", "gpt-3.5-turbo")
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random)
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            backend: Backend selector ("openai", "vllm", "hf"). Defaults to LLM_BACKEND env var.
            vllm_base_url: Optional override for VLLM_BASE_URL
            hf_token: Optional override for HF_TOKEN
            hf_peft_path: Optional override for HF_PEFT_PATH (local PEFT adapter)

        Raises:
            ValueError: If required backend configuration is missing
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key

        # Track usage
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0

        backend_name = (backend or os.getenv("LLM_BACKEND", "openai")).lower()
        if backend_name == "openai":
            self.backend = OpenAIBackend(model=self.model, temperature=self.temperature, api_key=self.api_key)
        elif backend_name == "vllm":
            self.backend = VLLMBackend(
                model=self.model,
                temperature=self.temperature,
                api_key=self.api_key,
                base_url=vllm_base_url
            )
        elif backend_name == "hf":
            hf_model = None if self.model == "gpt-4" else self.model
            self.backend = HFBackend(
                model=hf_model,
                temperature=self.temperature,
                token=hf_token,
                peft_adapter_path=hf_peft_path
            )
        else:
            raise ValueError(
                f"Unknown LLM backend '{backend_name}'. Expected 'openai', 'vllm', or 'hf'."
            )

    def call(
        self,
        user_prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        max_retries: int = 3,
        max_tokens: int = 500
    ) -> str:
        """Call LLM backend with retry logic.

        Args:
            user_prompt: The user message to send
            system_prompt: The system message (defaults to generic prompt)
            max_retries: Maximum number of retry attempts
            max_tokens: Maximum tokens in response (increased to 500 for structured output)

        Returns:
            str: The model's response text

        Raises:
            Exception: If all retries are exhausted
        """
        for attempt in range(max_retries):
            try:
                result = self.backend.call(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens
                )

                # Track usage
                self.total_calls += 1
                self.total_tokens += result.tokens_used
                self.total_cost_usd += result.cost_usd

                return result.text

            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"Rate limit error after {max_retries} attempts")
                    raise

            except APIError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"API error: {e}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"API error after {max_retries} attempts: {e}")
                    raise

            except Exception as e:
                print(f"Unexpected error calling LLM backend: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        # Should not reach here, but just in case
        raise Exception(f"Failed to get response after {max_retries} attempts")

    def get_usage_summary(self) -> dict:
        """Get summary of API usage statistics.

        Returns:
            dict: Dictionary with usage statistics
        """
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 2),
            "model": self.model
        }

    def print_usage_summary(self):
        """Print usage summary to console."""
        summary = self.get_usage_summary()
        print("\n" + "="*50)
        print("LLM API Usage Summary")
        print("="*50)
        print(f"Model: {summary['model']}")
        print(f"Total API calls: {summary['total_calls']}")
        print(f"Total tokens used: {summary['total_tokens']}")
        print(f"Estimated cost: ${summary['total_cost_usd']:.2f} USD")
        print("="*50 + "\n")
