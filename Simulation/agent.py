"""
LLM-based agents for Public Goods Game simulation.

This module defines the LLMAgent class that represents a player in the game.
Agents use an LLM to make decisions about contributions, punishments, rewards, and chat.
"""

from typing import List, Dict
from config import PGGConfig
from llm_client import LLMClient
from response_parser import ResponseParser


# Avatar names for agents (following existing convention from generate_prompts.py)
AVATAR_NAMES = [
    "DOG", "CHICKEN", "CAT", "FISH", "BIRD", "RABBIT",
    "TURTLE", "HAMSTER", "FROG", "MOUSE", "SNAKE", "BEAR",
    "LION", "TIGER", "ELEPHANT", "MONKEY", "PANDA", "KOALA", "WOLF", "FOX"
]


class LLMAgent:
    """An agent that plays the Public Goods Game using an LLM.

    The agent makes decisions by:
    1. Receiving a prompt describing the game state
    2. Calling an LLM API to get a response
    3. Parsing the response to extract the decision
    4. Storing the interaction in memory for debugging
    """

    def __init__(
        self,
        agent_id: str,
        avatar_name: str,
        llm_client: LLMClient,
        config: PGGConfig
    ):
        """Initialize an LLM agent.

        Args:
            agent_id: Unique identifier for this agent (e.g., "agent_0")
            avatar_name: Avatar name visible to other players (e.g., "DOG")
            llm_client: LLM client for making API calls
            config: Game configuration
        """
        self.agent_id = agent_id
        self.avatar_name = avatar_name
        self.llm_client = llm_client
        self.config = config
        self.memory: List[Dict[str, str]] = []  # Store prompts and responses

    def get_contribution_decision(self, prompt: str) -> tuple[int, str]:
        """Get contribution decision from LLM.

        Args:
            prompt: Full prompt describing game state and asking for contribution

        Returns:
            tuple: (contribution_amount, raw_response)
        """
        # Call LLM
        response = self.llm_client.call(prompt, max_tokens=500)

        # Store in memory
        self.memory.append({
            "type": "contribution",
            "prompt": prompt,
            "response": response
        })

        # Parse response
        amount = ResponseParser.parse_contribution(response, self.config.endowment)

        # Validate based on contribution type
        amount = ResponseParser.validate_contribution_type(
            amount,
            self.config.contribution_type,
            self.config.endowment
        )

        return amount, response

    def get_chat_message(self, prompt: str) -> tuple[str, str]:
        """Get chat message from LLM.

        Args:
            prompt: Prompt asking for chat message

        Returns:
            tuple: (chat_message, raw_response)
        """
        # Call LLM
        response = self.llm_client.call(prompt, max_tokens=500)

        # Store in memory
        self.memory.append({
            "type": "chat",
            "prompt": prompt,
            "response": response
        })

        # Parse response
        message = ResponseParser.parse_chat_message(response)

        # Check if agent wants to stay silent
        if message.lower() in ["nothing", "none", "no message", "pass", "skip", ""]:
            return "", response

        return message, response

    def get_redistribution_decision(
        self,
        prompt: str,
        num_targets: int
    ) -> tuple[List[int], str]:
        """Get punishment/reward decisions from LLM.

        Args:
            prompt: Prompt describing other players and asking for redistribution
            num_targets: Expected number of targets (length of output array)

        Returns:
            tuple: (amounts_list, raw_response)
        """
        # Call LLM
        response = self.llm_client.call(prompt, max_tokens=500)

        # Store in memory
        self.memory.append({
            "type": "redistribution",
            "prompt": prompt,
            "response": response,
            "num_targets": num_targets
        })

        # Parse response
        amounts = ResponseParser.parse_redistribution(response, num_targets)

        return amounts, response

    def get_memory_summary(self) -> str:
        """Get a summary of agent's interaction history.

        Returns:
            str: Formatted memory summary
        """
        lines = [f"Agent {self.avatar_name} ({self.agent_id}) Memory:"]
        lines.append("=" * 60)

        for i, interaction in enumerate(self.memory):
            lines.append(f"\nInteraction {i+1} ({interaction['type']})")
            lines.append(f"Response: {interaction['response']}")

        return "\n".join(lines)


def create_agents(config: PGGConfig, llm_client: LLMClient) -> List[LLMAgent]:
    """Create a list of LLM agents for the game.

    Args:
        config: Game configuration (determines group_size)
        llm_client: Shared LLM client instance

    Returns:
        List of LLMAgent instances
    """
    agents = []
    for i in range(config.group_size):
        agent = LLMAgent(
            agent_id=f"agent_{i}",
            avatar_name=AVATAR_NAMES[i % len(AVATAR_NAMES)],
            llm_client=llm_client,
            config=config
        )
        agents.append(agent)

    return agents


# ===== Testing / Demo =====
if __name__ == "__main__":
    from config import PGGConfig
    from llm_client import LLMClient

    print("Testing Agent Creation")
    print("=" * 60)

    # Create config and client
    config = PGGConfig(group_size=4)

    # Note: This will fail without OPENAI_API_KEY, but demonstrates the structure
    try:
        client = LLMClient(model="gpt-4", temperature=1.0)
        agents = create_agents(config, client)

        print(f"Created {len(agents)} agents:")
        for agent in agents:
            print(f"  - {agent.avatar_name} ({agent.agent_id})")

        # Test a simple prompt (requires API key)
        test_prompt = "You have 20 coins. How much do you contribute? Output a single integer:"
        print(f"\nTest prompt: {test_prompt}")
        amount = agents[0].get_contribution_decision(test_prompt)
        print(f"Agent {agents[0].avatar_name} contributed: {amount}")

    except ValueError as e:
        print(f"Note: {e}")
        print("This is expected if OPENAI_API_KEY is not set.")
