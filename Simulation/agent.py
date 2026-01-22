"""
Agent module for Public Goods Game simulation.

This module defines PGG agents that prepare context for EDSL parallel execution,
replacing direct LLM API calls with context preparation for survey-based execution.
"""

from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from config import PGGConfig


class PGGAgent:
    """Agent for PGG using EDSL execution (no direct LLM calls).

    This agent prepares context dictionaries for EDSL survey execution
    instead of making direct LLM API calls. The actual LLM calls are
    handled by EDSLGameClient in parallel.
    """

    def __init__(self, agent_id: str, avatar_name: str, config: 'PGGConfig'):
        """Initialize PGG agent.

        Args:
            agent_id: Unique identifier (e.g., "agent_0")
            avatar_name: Human-readable name (e.g., "Alice")
            config: Game configuration
        """
        self.agent_id = agent_id
        self.avatar_name = avatar_name
        self.config = config
        self.memory: List[Dict] = []

    def prepare_contribution_context(self, prompt: str) -> Dict:
        """Prepare context for contribution survey.

        Args:
            prompt: Full contribution prompt for this agent

        Returns:
            Context dict with agent_id, avatar_name, and prompt for EDSL execution
        """
        return {
            "agent_id": self.agent_id,
            "avatar_name": self.avatar_name,  # For two-stage reasoning
            "prompt": prompt
        }

    def prepare_redistribution_context(
        self,
        prompt: str,
        other_agents: List[Dict],
        current_wallet: float
    ) -> Dict:
        """Prepare context for redistribution survey.

        Calculates budget constraint: max_units = wallet / peer_incentive_cost

        Args:
            prompt: Full redistribution prompt for this agent
            other_agents: List of dicts with "agent_id" and "avatar_name"
            current_wallet: Current wallet balance (after contribution stage)

        Returns:
            Context dict with agent_id, avatar_name, prompt, other_agents, and max_units
        """
        # Calculate maximum units affordable with current wallet
        max_units = int(current_wallet / self.config.peer_incentive_cost)

        return {
            "agent_id": self.agent_id,
            "avatar_name": self.avatar_name,  # For two-stage reasoning
            "prompt": prompt,
            "other_agents": other_agents,
            "max_units": max_units
        }


def create_pgg_agents(config: 'PGGConfig') -> List[PGGAgent]:
    """Create agents for EDSL-based simulation.

    Uses human-friendly avatar names (Alice, Bob, Charlie, etc.)
    instead of animal names.

    Args:
        config: Game configuration

    Returns:
        List of PGGAgent instances
    """
    agents = []

    # Human-friendly avatar names
    AVATAR_NAMES = [
        "Alice", "Bob", "Charlie", "Diana", "Eve",
        "Frank", "Grace", "Hank", "Ivy", "Jack",
        "Kate", "Liam", "Mia", "Noah", "Olivia",
        "Peter", "Quinn", "Rachel", "Sam", "Tara"
    ]

    for i in range(config.group_size):
        agent = PGGAgent(
            agent_id=f"agent_{i}",
            avatar_name=AVATAR_NAMES[i % len(AVATAR_NAMES)],
            config=config
        )
        agents.append(agent)

    return agents


# ===== Testing / Demo =====
if __name__ == "__main__":
    from config import PGGConfig

    print("Testing Agent Creation")
    print("=" * 60)

    # Create config
    config = PGGConfig(
        group_size=4,
        punishment_enabled=True,
        peer_incentive_cost=2
    )

    # Create agents
    agents = create_pgg_agents(config)

    print(f"Created {len(agents)} agents:")
    for agent in agents:
        print(f"  - {agent.avatar_name} ({agent.agent_id})")

    # Test context preparation
    print("\nTest contribution context:")
    test_prompt = "You have 20 coins. How much do you contribute?"
    context = agents[0].prepare_contribution_context(test_prompt)
    print(f"  agent_id: {context['agent_id']}")
    print(f"  prompt: {context['prompt'][:50]}...")

    print("\nTest redistribution context:")
    other_agents = [
        {"agent_id": "agent_1", "avatar_name": "Bob"},
        {"agent_id": "agent_2", "avatar_name": "Charlie"}
    ]
    redist_context = agents[0].prepare_redistribution_context(
        prompt="Select players to punish:",
        other_agents=other_agents,
        current_wallet=15.0
    )
    print(f"  agent_id: {redist_context['agent_id']}")
    print(f"  max_units: {redist_context['max_units']}")
    print(f"  other_agents: {len(redist_context['other_agents'])} players")
