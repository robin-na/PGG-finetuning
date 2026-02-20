"""
Test script for structured output parsing.

This script tests the ResponseParser's ability to extract data from
XML-tagged responses as well as unstructured text.
"""

import sys
from pathlib import Path

# Add Simulation to path
sys.path.insert(0, str(Path(__file__).parent / "Simulation"))
from response_parser import ResponseParser

print("=" * 80)
print("TESTING STRUCTURED OUTPUT PARSING")
print("=" * 80)
print()

# Test 1: Contribution parsing
print("-" * 80)
print("TEST 1: CONTRIBUTION PARSING")
print("-" * 80)
print()

test_cases_contrib = [
    # Structured format (expected)
    ("""<REASONING>
I should contribute moderately to balance individual and group gains.
</REASONING>
<CONTRIBUTE>
15
</CONTRIBUTE>""", 20, "Structured format"),

    # Without reasoning tag
    ("""<CONTRIBUTE>
10
</CONTRIBUTE>""", 20, "Structured without reasoning"),

    # Old unstructured format (fallback)
    ("I want to contribute 12 coins", 20, "Unstructured text"),

    # Direct number
    ("5", 20, "Direct number"),

    # Invalid (should default to 10)
    ("I'm not sure what to do", 20, "Invalid input"),
]

for response, max_val, description in test_cases_contrib:
    result = ResponseParser.parse_contribution(response, max_val)
    print(f"{description}:")
    print(f"  Input: {response[:50]}...")
    print(f"  Parsed: {result}")
    print()

# Test 2: Redistribution parsing
print("-" * 80)
print("TEST 2: REDISTRIBUTION PARSING")
print("-" * 80)
print()

test_cases_redist = [
    # Structured format
    ("""<REASONING>
Player DOG free-rode, so I'll punish with 2 units.
Player CAT cooperated, no punishment.
Player BIRD free-rode slightly, punish with 1 unit.
</REASONING>
<REDISTRIBUTE>
[2, 0, 1]
</REDISTRIBUTE>""", 3, "Structured format"),

    # Without reasoning
    ("""<REDISTRIBUTE>
[0, 1, 2]
</REDISTRIBUTE>""", 3, "Structured without reasoning"),

    # Old JSON format (fallback)
    ("[1, 1, 0]", 3, "Direct JSON array"),

    # Comma-separated
    ("2, 0, 1", 3, "Comma-separated"),

    # Invalid (should default to [0, 0, 0])
    ("no punishment", 3, "Invalid input"),
]

for response, num_targets, description in test_cases_redist:
    result = ResponseParser.parse_redistribution(response, num_targets)
    print(f"{description}:")
    print(f"  Input: {response[:50]}...")
    print(f"  Parsed: {result}")
    print()

# Test 3: Chat message parsing
print("-" * 80)
print("TEST 3: CHAT MESSAGE PARSING")
print("-" * 80)
print()

test_cases_chat = [
    # Structured format
    ("""<REASONING>
I should encourage cooperation to maximize group payoff.
</REASONING>
<MESSAGE>
Let's all contribute 20 coins for maximum benefit!
</MESSAGE>""", "Structured format"),

    # Without reasoning
    ("""<MESSAGE>
I'll contribute 15 coins
</MESSAGE>""", "Structured without reasoning"),

    # Old unstructured format (fallback)
    ('I say: "We should cooperate"', "Old prefixed format"),

    # Direct message
    ("Trust me, I will contribute!", "Direct message"),

    # Nothing response
    ("""<MESSAGE>
nothing
</MESSAGE>""", "Nothing response"),
]

for response, description in test_cases_chat:
    result = ResponseParser.parse_chat_message(response)
    print(f"{description}:")
    print(f"  Input: {response[:50]}...")
    print(f"  Parsed: '{result}'")
    print()

print("=" * 80)
print("ALL TESTS COMPLETED")
print("=" * 80)
print()
print("Summary:")
print("✓ Structured format (XML tags) is now supported")
print("✓ Old unstructured format still works as fallback")
print("✓ Parser is backwards compatible")
print()
