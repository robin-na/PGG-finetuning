"""
Response parser for extracting structured data from LLM outputs.

This module provides robust parsing functions that handle various LLM response formats
and always return valid default values when parsing fails.
"""

import re
from typing import List


class ResponseParser:
    """Parser for LLM responses with fallback defaults.

    This class provides static methods to parse different types of LLM outputs:
    - Contribution amounts (single integer)
    - Redistribution decisions (array of integers for punishment/reward)
    - Chat messages (text)

    All methods are designed to be robust and always return valid values.
    """

    @staticmethod
    def parse_contribution(response: str, max_value: int) -> int:
        """Extract contribution amount from LLM response.

        Attempts multiple parsing strategies:
        1. Extract from <CONTRIBUTE>...</CONTRIBUTE> tags (structured format)
        2. Direct integer conversion
        3. Extract first number from text
        4. Default to half of max_value if all else fails

        Args:
            response: Raw LLM response text
            max_value: Maximum valid contribution amount (typically endowment)

        Returns:
            int: Contribution amount, clamped to [0, max_value]
        """
        response_clean = response.strip()

        # Strategy 1: Try extracting from <CONTRIBUTE> tags
        contribute_match = re.search(r'<CONTRIBUTE>\s*(\d+)\s*</CONTRIBUTE>', response_clean, re.IGNORECASE | re.DOTALL)
        if contribute_match:
            try:
                value = int(contribute_match.group(1))
                return max(0, min(value, max_value))
            except ValueError:
                pass

        # Strategy 2: Try direct integer conversion
        try:
            value = int(response_clean)
            # Clamp to valid range
            return max(0, min(value, max_value))
        except ValueError:
            pass

        # Strategy 3: Try extracting first number from text
        # Use word boundaries to avoid partial matches
        numbers = re.findall(r'\b\d+\b', response)
        if numbers:
            value = int(numbers[0])
            return max(0, min(value, max_value))

        # Strategy 4: Default fallback
        default = max_value // 2
        print(f"Warning: Could not parse contribution response '{response[:500]}...', defaulting to {default}")
        return default

    @staticmethod
    def parse_redistribution(response: str, num_targets: int) -> List[int]:
        """Parse array of punishment/reward amounts from LLM response.

        Expected format examples:
        - Structured: "<REDISTRIBUTE>[0, 2, 1]</REDISTRIBUTE>"
        - JSON array: "[0, 2, 1]"
        - Comma-separated: "0, 2, 1"
        - Space-separated: "0 2 1"

        Args:
            response: Raw LLM response text
            num_targets: Expected number of targets (length of array)

        Returns:
            List[int]: List of amounts (one per target), all non-negative
        """
        response_clean = response.strip()

        # Strategy 1: Try extracting from <REDISTRIBUTE> tags
        redistribute_match = re.search(r'<REDISTRIBUTE>\s*(.+?)\s*</REDISTRIBUTE>', response_clean, re.IGNORECASE | re.DOTALL)
        if redistribute_match:
            inner_content = redistribute_match.group(1).strip()
            # Try parsing as JSON array
            try:
                import json
                parsed = json.loads(inner_content)
                if isinstance(parsed, list):
                    result = [max(0, int(x)) for x in parsed[:num_targets]]
                    while len(result) < num_targets:
                        result.append(0)
                    return result[:num_targets]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Strategy 2: Try JSON array parsing (entire response)
        try:
            import json
            parsed = json.loads(response_clean)
            if isinstance(parsed, list):
                # Take first num_targets elements, convert to non-negative ints
                result = [max(0, int(x)) for x in parsed[:num_targets]]
                # Pad with zeros if too short
                while len(result) < num_targets:
                    result.append(0)
                return result[:num_targets]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Strategy 3: Try extracting all numbers (ignoring negative signs)
        numbers = re.findall(r'\d+', response)
        if numbers:
            result = [int(x) for x in numbers[:num_targets]]
            # Pad with zeros if too short
            while len(result) < num_targets:
                result.append(0)
            return result[:num_targets]

        # Strategy 4: Default to no punishment/reward
        print(f"Warning: Could not parse redistribution response '{response[:100]}...', defaulting to zeros")
        return [0] * num_targets

    @staticmethod
    def parse_chat_message(response: str, max_length: int = 2000) -> str:
        """Extract chat message from LLM response.

        Args:
            response: Raw LLM response text
            max_length: Maximum allowed message length

        Returns:
            str: Cleaned and truncated message
        """
        message = response.strip()

        # Strategy 1: Try extracting from <MESSAGE> tags
        message_match = re.search(r'<MESSAGE>\s*(.+?)\s*</MESSAGE>', message, re.IGNORECASE | re.DOTALL)
        if message_match:
            message = message_match.group(1).strip()

        # Remove common wrapper phrases if present
        unwanted_prefixes = [
            "I say: ",
            "I would say: ",
            "My message is: ",
            "I respond: ",
            "I send: ",
            "Message: "
        ]
        for prefix in unwanted_prefixes:
            if message.lower().startswith(prefix.lower()):
                message = message[len(prefix):].strip()

        # Remove quotes if the entire message is quoted
        if (message.startswith('"') and message.endswith('"')) or \
           (message.startswith("'") and message.endswith("'")):
            message = message[1:-1].strip()

        # Truncate if too long
        if len(message) > max_length:
            message = message[:max_length] + "..."

        return message

    @staticmethod
    def validate_contribution_type(
        amount: int,
        contribution_type: str,
        endowment: int
    ) -> int:
        """Validate contribution based on contribution type constraint.

        Args:
            amount: Proposed contribution amount
            contribution_type: "variable" or "all_or_nothing"
            endowment: Total endowment amount

        Returns:
            int: Valid contribution amount
        """
        if contribution_type == "all_or_nothing":
            # Force to 0 or endowment (whichever is closer)
            if amount < endowment / 2:
                return 0
            else:
                return endowment
        else:
            # Variable: just clamp to range
            return max(0, min(amount, endowment))


# ===== Testing / Demo =====
if __name__ == "__main__":
    # Test contribution parsing
    print("Testing contribution parsing:")
    test_cases = [
        ("15", 20),
        ("I want to contribute 10 coins", 20),
        ("I'll put 5 into the fund", 20),
        ("twentyfive", 20),  # Should default to 10
        ("0", 20),
        ("100", 20),  # Should clamp to 20
    ]
    for response, max_val in test_cases:
        result = ResponseParser.parse_contribution(response, max_val)
        print(f"  '{response}' → {result}")

    print("\nTesting redistribution parsing:")
    test_cases_redist = [
        ("[0, 2, 1]", 3),
        ("0, 2, 1", 3),
        ("0 2 1", 3),
        ("I punish player 1 with 2 units and player 2 with 1 unit", 3),
        ("no punishment", 3),
    ]
    for response, num_targets in test_cases_redist:
        result = ResponseParser.parse_redistribution(response, num_targets)
        print(f"  '{response}' → {result}")

    print("\nTesting chat message parsing:")
    test_cases_chat = [
        ("Let's all cooperate!",),
        ('I say: "We should contribute 20 each"',),
        ("A" * 250,),  # Should truncate
    ]
    for (response,) in test_cases_chat:
        result = ResponseParser.parse_chat_message(response)
        print(f"  '{response[:50]}...' → '{result[:50]}...'")
