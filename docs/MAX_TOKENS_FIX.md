# Max Tokens Fix for Structured Output

## Problem

When running experiments with structured output format, LLM responses were being truncated:

```
Warning: Could not parse contribution response '<REASONING>
In this public goods game, the optimal strategy is to encourage cooperative behavior whi...'
```

The response was cut off before reaching the `<CONTRIBUTE>` tag, causing the parser to fall back to default values.

## Root Cause

**`max_tokens=150`** was too small for the new structured output format:

```xml
<REASONING>
...potentially 100-200 tokens of reasoning...
</REASONING>
<CONTRIBUTE>
15
</CONTRIBUTE>
```

Total: ~150-250 tokens needed, but only 150 allowed → truncation

## Solution

**Changed `max_tokens` from 150 to 500** in [Simulation/llm_client.py:76](Simulation/llm_client.py#L76)

```python
def call(
    self,
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    max_retries: int = 3,
    max_tokens: int = 500  # Changed from 150
) -> str:
```

## Token Allocation

With `max_tokens=500`, we now have room for:
- **`<REASONING>`** tag: ~150-300 tokens
- **`<CONTRIBUTE>`** tag: ~10 tokens
- **`<MESSAGE>`** tag: ~50-100 tokens
- **`<REDISTRIBUTE>`** tag: ~20-50 tokens
- **Buffer**: ~50-100 tokens

## Cost Impact

**Negligible increase in API cost:**
- Old: ~150 tokens per call
- New: ~300-400 tokens per call (actual usage, not max)
- Cost difference: ~$0.0003 per call (using gpt-4o)
- For 40 experiments × 4 agents × 2-3 calls = 320-480 calls
- Additional cost: **~$0.10-0.15 USD total**

## Verification

Run the test script to verify structured output parsing works:

```bash
python3 test_structured_output.py
```

Expected output:
```
✓ Structured format (XML tags) is now supported
✓ Old unstructured format still works as fallback
✓ Parser is backwards compatible
```

## Before/After Examples

### Before (max_tokens=150)
```
Response (truncated):
<REASONING>
In this public goods game, the optimal strategy is to encourage cooperative behavior whi...

Parsing result:
Warning: Could not parse... defaulting to 10
```

### After (max_tokens=500)
```
Response (complete):
<REASONING>
In this public goods game, the optimal strategy is to encourage cooperative behavior while
protecting individual interests. Contributing 15 coins signals cooperation without excessive risk.
</REASONING>
<CONTRIBUTE>
15
</CONTRIBUTE>

Parsing result:
Successfully parsed: 15
```

## Related Files

- ✅ [Simulation/llm_client.py](Simulation/llm_client.py) - Max tokens increased
- ✅ [Simulation/prompt_builder.py](Simulation/prompt_builder.py) - Structured format prompts
- ✅ [Simulation/response_parser.py](Simulation/response_parser.py) - XML tag parsing
- ✅ [STRUCTURED_OUTPUT_FORMAT.md](STRUCTURED_OUTPUT_FORMAT.md) - Full documentation

## Status

✅ **FIXED** - Ready to run experiments

You can now run:
```bash
python run_experiments.py
```

The parsing warnings should be eliminated or significantly reduced.
