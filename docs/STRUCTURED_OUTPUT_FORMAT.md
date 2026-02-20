# Structured Output Format for LLM Responses

## Overview

We've implemented a structured XML-tag based output format for LLM agents to improve parsing reliability and allow agents to show their reasoning explicitly.

## Format Specifications

### 1. Contribution Decisions

**Prompt asks for:**
```
**Output format (required):**
<REASONING>
Your strategic thinking and reasoning here...
</REASONING>
<CONTRIBUTE>
A single integer number only (no text)
</CONTRIBUTE>
```

**Example LLM response:**
```xml
<REASONING>
Given that this is a one-shot game, I should balance individual gains with group benefit.
Contributing 15 coins allows me to test cooperation while maintaining some safety.
</REASONING>
<CONTRIBUTE>
15
</CONTRIBUTE>
```

### 2. Chat Messages

**Prompt asks for:**
```
**Output format (required):**
<REASONING>
Your strategic thinking here...
</REASONING>
<MESSAGE>
Your message to other players (or 'nothing' to stay silent)
</MESSAGE>
```

**Example LLM response:**
```xml
<REASONING>
I should encourage cooperation to maximize group payoff in this game.
</REASONING>
<MESSAGE>
Let's all contribute 20 coins for maximum benefit!
</MESSAGE>
```

### 3. Redistribution Decisions (Punishment/Reward)

**Prompt asks for:**
```
**Output format (required):**
<REASONING>
Your strategic thinking and justification here...
</REASONING>
<REDISTRIBUTE>
A JSON array of exactly N integers, e.g., [0, 2, 1, 0]
</REDISTRIBUTE>
```

**Example LLM response:**
```xml
<REASONING>
Player DOG contributed 0 coins (free-rode), so I'll punish with 2 units to enforce norms.
Player CAT contributed 20 coins (cooperated), no punishment needed.
Player BIRD contributed 5 coins (partial cooperation), mild punishment with 1 unit.
</REASONING>
<REDISTRIBUTE>
[2, 0, 1]
</REDISTRIBUTE>
```

## Benefits

### 1. **Explicit Reasoning**
- LLMs can show their strategic thinking
- Researchers can analyze decision-making processes
- Easier to debug unexpected behaviors

### 2. **Reliable Parsing**
- Clear delimiters reduce parsing errors
- Fallback to unstructured parsing for robustness
- Backwards compatible with old format

### 3. **Structured Data**
- Reasoning can be logged separately from actions
- Enables future analysis of LLM strategies
- Machine-readable format for automated analysis

## Implementation Details

### Parser Priority

The `ResponseParser` tries multiple strategies in order:

**For Contributions:**
1. Extract from `<CONTRIBUTE>...</CONTRIBUTE>` tags
2. Try direct integer conversion
3. Extract first number from text
4. Default to `endowment // 2`

**For Redistribution:**
1. Extract from `<REDISTRIBUTE>...</REDISTRIBUTE>` tags
2. Try JSON array parsing
3. Extract all numbers from text
4. Default to `[0, 0, 0, ...]`

**For Chat:**
1. Extract from `<MESSAGE>...</MESSAGE>` tags
2. Remove common prefixes ("I say:", etc.)
3. Remove surrounding quotes
4. Return cleaned message

### Backwards Compatibility

The parser is **fully backwards compatible**:
- Old experiments without structured tags still work
- Mixed responses (some tagged, some not) are handled
- Graceful degradation to fallback strategies

## Files Modified

1. **[Simulation/prompt_builder.py](Simulation/prompt_builder.py)**
   - Updated `build_contribution_prompt()` to add structured format instructions
   - Updated `build_chat_prompt()` to add structured format
   - Updated `build_redistribution_prompt()` to add structured format

2. **[Simulation/response_parser.py](Simulation/response_parser.py)**
   - Enhanced `parse_contribution()` to extract from `<CONTRIBUTE>` tags
   - Enhanced `parse_redistribution()` to extract from `<REDISTRIBUTE>` tags
   - Enhanced `parse_chat_message()` to extract from `<MESSAGE>` tags

3. **[test_structured_output.py](test_structured_output.py)** (new)
   - Comprehensive test suite for structured output parsing
   - Validates both structured and unstructured formats

## Testing

Run the test suite:
```bash
python3 test_structured_output.py
```

Expected output:
```
✓ Structured format (XML tags) is now supported
✓ Old unstructured format still works as fallback
✓ Parser is backwards compatible
```

## Usage in Experiments

No changes needed to run experiments! The new format is automatically used:

```bash
# Existing experiments will use structured format
python run_experiments.py

# Old experiment data can still be analyzed
python analyze_results.py
```

## Future Extensions

Potential enhancements:
1. **Save reasoning separately**: Log `<REASONING>` content to separate files for analysis
2. **Reasoning quality metrics**: Analyze reasoning sophistication
3. **Multi-agent reasoning comparison**: Compare how different agents justify same situations
4. **Reasoning-based clustering**: Group agents by reasoning patterns

## Example: Full Game Flow

**Round 1 Contribution Prompt:**
```
### Contribution Stage: Decide how much to contribute.

You have 20 coins in your private fund. How much do you move to the public fund?
You can contribute any amount from 0 to 20.

**Output format (required):**
<REASONING>
Your strategic thinking and reasoning here...
</REASONING>
<CONTRIBUTE>
A single integer number only (no text)
</CONTRIBUTE>
```

**LLM Response:**
```xml
<REASONING>
This is a one-shot game with MPCR=0.4 and 4 players (multiplier=1.6).
If everyone contributes 20, we each get 20*1.6=32 coins (net +12).
If I defect while others cooperate, I get 20+15*1.6=44 coins (net +24).
However, mutual defection gives 0 payoff. I'll contribute 15 to signal cooperation.
</REASONING>
<CONTRIBUTE>
15
</CONTRIBUTE>
```

**Parsed Result:**
- **Action**: Contribute 15 coins
- **Reasoning**: (logged) "This is a one-shot game with MPCR=0.4..."

## Benefits for Research

1. **Transparency**: See how LLMs reason about game-theoretic scenarios
2. **Interpretability**: Understand why agents choose certain strategies
3. **Validation**: Verify that reasoning aligns with actions
4. **Comparison**: Compare reasoning quality across models (GPT-4, Claude, etc.)

## Notes

- The `<REASONING>` tag is optional - parser works without it
- Tag names are case-insensitive (`<CONTRIBUTE>` = `<contribute>`)
- Extra whitespace inside tags is automatically trimmed
- If LLM doesn't follow format, parser falls back to heuristics

---

**Status**: ✅ Implemented and tested
**Version**: 1.0
**Last Updated**: 2026-01-01
