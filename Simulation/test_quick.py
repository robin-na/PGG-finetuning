"""
Quick test script for PGG simulation with minimal settings.
Tests with game_length=1 and small group_size for fast verification.
"""

from config import PGGConfig
from main import run_experiment

# Create a minimal test configuration
test_config = PGGConfig(
    group_size=3,        # Small group for faster testing
    game_length=1,       # Just 1 round for quick test
    endowment=20,
    mpcr=0.4,
    contribution_framing="opt_in",
    punishment_enabled=False,  # Disable for faster testing
    communication_enabled=False,
    llm_model="gpt-4o",
    llm_temperature=1.0
)

print("=" * 70)
print("Quick Test: 3 agents, 1 round, no punishment")
print("=" * 70)

# Run a single game
run_experiment("quick_test", test_config, num_games=1, verbose=True)

print("\n" + "=" * 70)
print("Test complete! Check experiments/quick_test/ for results.")
print("=" * 70)
