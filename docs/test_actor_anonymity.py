"""
Test script to demonstrate Actor Anonymity feature.

This script runs two experiments:
1. Revealed identity (actor_anonymity=False) - players know who punished them
2. Anonymous identity (actor_anonymity=True) - players only know they were punished
"""

from config import PGGConfig
from main import run_experiment

# Experiment 1: Revealed Identity
print("=" * 70)
print("TEST 1: REVEALED IDENTITY (actor_anonymity=False)")
print("Players will see WHO punished/rewarded them")
print("=" * 70)

config_revealed = PGGConfig(
    group_size=3,
    game_length=2,  # 2 rounds to show history effects
    endowment=20,
    mpcr=0.4,

    # Enable punishment with revealed identity
    punishment_enabled=True,
    punishment_cost=2,
    punishment_impact=3,
    actor_anonymity=False,  # Show who punished

    # Show peer outcomes so agents can target punishments
    peer_outcome_visibility=True,

    llm_model="gpt-4o",
    llm_temperature=1.0
)

run_experiment("test_revealed_identity", config_revealed, num_games=1, verbose=True)

print("\n" + "=" * 70)
print()

# Experiment 2: Anonymous Identity
print("=" * 70)
print("TEST 2: ANONYMOUS IDENTITY (actor_anonymity=True)")
print("Players will only know THAT they were punished, not by whom")
print("=" * 70)

config_anonymous = PGGConfig(
    group_size=3,
    game_length=2,  # 2 rounds to show history effects
    endowment=20,
    mpcr=0.4,

    # Enable punishment with anonymous identity
    punishment_enabled=True,
    punishment_cost=2,
    punishment_impact=3,
    actor_anonymity=True,  # Hide who punished

    # Show peer outcomes so agents can target punishments
    peer_outcome_visibility=True,

    llm_model="gpt-4o",
    llm_temperature=1.0
)

run_experiment("test_anonymous_identity", config_anonymous, num_games=1, verbose=True)

print("\n" + "=" * 70)
print("COMPARISON:")
print("=" * 70)
print()
print("Check the prompt files in:")
print("  experiments/test_revealed_identity/prompts/")
print("  experiments/test_anonymous_identity/prompts/")
print()
print("In Round 2 contribution prompts, you'll see the difference:")
print()
print("REVEALED (actor_anonymity=False):")
print("  'DOG punished you with 2 units, deducting 6 coins.'")
print()
print("ANONYMOUS (actor_anonymity=True):")
print("  'You received 2 punishment units, deducting 6 coins.'")
print()
print("=" * 70)
