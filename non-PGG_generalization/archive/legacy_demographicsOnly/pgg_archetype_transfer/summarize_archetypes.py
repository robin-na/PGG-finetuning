#!/usr/bin/env python3
"""Pre-summarize PGG archetypes into concise trait summaries for prompt injection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from config import OUTPUT_ROOT, SUMMARIZATION_MODEL

SYSTEM_PROMPT = """You are an expert behavioral scientist. Given a detailed PGG (Public Goods Game) behavioral archetype, extract a concise trait summary that captures the person's general behavioral tendencies. Focus on traits that transfer across different economic games:

- Cooperation level (generous/moderate/selfish)
- Fairness sensitivity (high/medium/low)
- Risk tolerance (risk-seeking/neutral/risk-averse)
- Patience / time preference (patient/impatient)
- Punishment willingness (willing to punish unfairness / avoids punishment)
- Reciprocity (reciprocates kindness, retaliates against unfairness, or indifferent)
- Strategic thinking (strategic/reactive/fixed-rule)

Output ONLY the trait summary in 2-4 sentences. No headers or bullet points."""


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def summarize_batch(
    client: OpenAI,
    texts: List[str],
    model: str,
    batch_size: int = 20,
) -> List[str]:
    """Summarize archetype texts using the LLM."""
    summaries = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for text in batch:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text[:4000]},  # truncate very long texts
                ],
                max_tokens=200,
                temperature=0.0,
            )
            summaries.append(resp.choices[0].message.content.strip())
        if i + batch_size < len(texts):
            print(f"  Summarized {i + batch_size}/{len(texts)}...")
            time.sleep(0.5)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PGG archetypes to trait summaries.")
    parser.add_argument(
        "--bank-texts",
        type=Path,
        default=OUTPUT_ROOT / "archetype_bank" / "archetype_bank_texts.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "archetype_bank" / "archetype_trait_summaries.jsonl",
    )
    parser.add_argument("--model", type=str, default=SUMMARIZATION_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of archetypes to summarize (for testing)")
    args = parser.parse_args()

    print(f"Loading archetype texts from {args.bank_texts}...")
    bank_texts = load_jsonl(args.bank_texts)
    if args.limit:
        bank_texts = bank_texts[: args.limit]
    print(f"  {len(bank_texts)} archetypes to summarize.")

    texts = [r["text"] for r in bank_texts]

    print(f"Summarizing with {args.model}...")
    client = OpenAI()
    summaries = summarize_batch(client, texts, model=args.model)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for i, (row, summary) in enumerate(zip(bank_texts, summaries)):
            out = {"idx": row["idx"], "trait_summary": summary}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Saved {len(summaries)} summaries to {args.output}")

    # Show a few examples
    print("\n--- Example summaries ---")
    for i in range(min(3, len(summaries))):
        print(f"\n[Archetype {i}] {summaries[i]}")


if __name__ == "__main__":
    main()
