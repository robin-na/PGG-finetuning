"""
Demo: Show Actor Anonymity prompt differences WITHOUT running LLM calls.

This script demonstrates how prompts differ between revealed vs. anonymous identity
by creating mock round states and showing the formatted prompts.
"""

from config import PGGConfig
from prompt_builder import PromptBuilder
from environment import RoundState

print("=" * 80)
print("ACTOR ANONYMITY DEMO")
print("=" * 80)
print()
print("This demo shows how prompts differ based on actor_anonymity setting.")
print("We create a mock scenario where DOG was punished by CAT in Round 1,")
print("then show how this appears in Round 2 prompts.")
print()

# Create mock agent names
agent_names = {
    "agent_0": "DOG",
    "agent_1": "CAT",
    "agent_2": "BIRD"
}

# Create mock Round 1 state where:
# - DOG contributed 20 (cooperated)
# - CAT contributed 20 (cooperated)
# - BIRD contributed 5 (free-rode)
# - CAT punished BIRD with 2 units
mock_round_1 = RoundState(
    round_num=1,
    contributions={"agent_0": 20, "agent_1": 20, "agent_2": 5},
    chat_messages=[],
    public_fund=54.0,  # (20+20+5) * 1.2
    payoffs={
        "agent_0": 18.0,   # (20-20) + 18 = 18
        "agent_1": 14.0,   # (20-20) + 18 - 4 (punishment cost) = 14
        "agent_2": 27.0    # (20-5) + 18 - 6 (punishment impact) = 27
    },
    punishments={("agent_1", "agent_2"): 2},  # CAT punished BIRD
    rewards={},
    wallets={"agent_0": 18.0, "agent_1": 14.0, "agent_2": 27.0}
)

# Test 1: REVEALED IDENTITY (actor_anonymity=False)
print("-" * 80)
print("TEST 1: REVEALED IDENTITY (actor_anonymity=False)")
print("-" * 80)
print()

config_revealed = PGGConfig(
    group_size=3,
    game_length=2,
    mpcr=0.4,
    punishment_enabled=True,
    actor_anonymity=False  # Show WHO punished
)

builder_revealed = PromptBuilder(config_revealed)

# Show how BIRD sees Round 1 history in Round 2
history_revealed = builder_revealed._format_round_summary(
    agent_id="agent_2",
    agent_name="BIRD",
    round_state=mock_round_1,
    agent_names=agent_names
)

print("What BIRD sees in Round 2 contribution prompt:")
print()
print(history_revealed)
print()

# Test 2: ANONYMOUS IDENTITY (actor_anonymity=True)
print("-" * 80)
print("TEST 2: ANONYMOUS IDENTITY (actor_anonymity=True)")
print("-" * 80)
print()

config_anonymous = PGGConfig(
    group_size=3,
    game_length=2,
    mpcr=0.4,
    punishment_enabled=True,
    actor_anonymity=True   # Hide WHO punished
)

builder_anonymous = PromptBuilder(config_anonymous)

# Show how BIRD sees Round 1 history in Round 2
history_anonymous = builder_anonymous._format_round_summary(
    agent_id="agent_2",
    agent_name="BIRD",
    round_state=mock_round_1,
    agent_names=agent_names
)

print("What BIRD sees in Round 2 contribution prompt:")
print()
print(history_anonymous)
print()

# Summary comparison
print("=" * 80)
print("KEY DIFFERENCE:")
print("=" * 80)
print()
print("REVEALED (actor_anonymity=False):")
print("  → 'CAT punished you with 2 units, deducting 6 coins.'")
print("  → BIRD knows it was CAT who punished")
print("  → BIRD might retaliate against CAT in Round 2")
print()
print("ANONYMOUS (actor_anonymity=True):")
print("  → 'You received 2 punishment units, deducting 6 coins.'")
print("  → BIRD only knows they were punished, not by whom")
print("  → BIRD cannot specifically target revenge")
print()
print("=" * 80)
print()
print("This difference affects:")
print("  ✓ Fear of retaliation (revealed) vs. no fear (anonymous)")
print("  ✓ Strategic targeting vs. group-level deterrence")
print("  ✓ Reputation building vs. norm enforcement")
print()
print("=" * 80)
