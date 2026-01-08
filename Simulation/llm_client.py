"""
LLM client for OpenAI API integration with retry logic.

This module provides a wrapper around the OpenAI API with robust error handling,
exponential backoff for rate limits, and cost tracking.
"""

from openai import OpenAI, RateLimitError, APIError
import os
import time
from typing import Optional

# Generic system prompt for all agents
SYSTEM_PROMPT = """You are a participant in an economic game theory experiment. Your goal is to maximize your total earnings.

You will be given game rules and historical information. Based on this information, make decisions that you believe will maximize your payoff.

Follow these guidelines:
1. Read the rules carefully
2. Consider both short-term and long-term consequences
3. Think about how other players might respond to your actions
4. Output your decisions in the exact format requested

Be strategic but realistic in your choices."""


class LLMClient:
    """Client for making OpenAI API calls with retry logic.

    This class handles:
    - API authentication
    - Request execution with exponential backoff on rate limits
    - Error handling and logging
    - Token usage tracking
    """

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 1.0,
        api_key: Optional[str] = None
    ):
        """Initialize LLM client.

        Args:
            model: OpenAI model name (e.g., "gpt-4", "gpt-4o", "gpt-3.5-turbo")
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random)
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)

        Raises:
            ValueError: If API key is not provided and not found in environment
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize OpenAI client (new style)
        self.client = OpenAI(api_key=self.api_key)

        # Track usage
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0

    def call(
        self,
        user_prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        max_retries: int = 3,
        max_tokens: int = 500
    ) -> str:
        """Call OpenAI API with retry logic.

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
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens
                )

                # Extract response
                result = response.choices[0].message.content.strip()

                # Track usage
                self.total_calls += 1
                tokens_used = response.usage.total_tokens
                self.total_tokens += tokens_used

                # Estimate cost (approximate rates)
                if 'gpt-4o' in self.model.lower():
                    # GPT-4o: ~$0.0025/1K prompt tokens, ~$0.01/1K completion tokens
                    # Rough estimate: average to $0.006/1K tokens
                    self.total_cost_usd += tokens_used * 0.006 / 1000
                elif 'gpt-4' in self.model.lower():
                    # GPT-4: ~$0.03/1K prompt tokens, ~$0.06/1K completion tokens
                    # Rough estimate: average to $0.045/1K tokens
                    self.total_cost_usd += tokens_used * 0.045 / 1000
                else:
                    # GPT-3.5-turbo: ~$0.001/1K tokens
                    self.total_cost_usd += tokens_used * 0.001 / 1000

                return result

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
                print(f"Unexpected error calling OpenAI API: {e}")
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
