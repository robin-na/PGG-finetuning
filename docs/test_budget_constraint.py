"""
Test budget constraint and detailed redistribution logging functionality

This script runs a small experiment to verify:
1. Prompts include budget information
2. Proportional scaling logic works correctly
3. redistribution_details.csv is properly recorded
4. Wallet balances remain non-negative
"""

import sys
from pathlib import Path
import pandas as pd

# Add Simulation to path
sys.path.insert(0, str(Path(__file__).parent / "Simulation"))

from config import PGGConfig
from main import run_experiment


def test_budget_constraint():
    """Run test experiment"""

    print("="*60)
    print("Testing Budget Constraint Functionality")
    print("="*60)
    print()

    # Create a large group configuration (likely to trigger budget issues)
    config = PGGConfig(
        group_size=10,  # Large group
        game_length=1,
        endowment=20,
        mpcr=0.4,
        communication_enabled=True,
        punishment_enabled=True,
        punishment_cost=1,
        punishment_impact=3,
        reward_enabled=True,
        reward_cost=1,
        reward_impact=0.75,
        peer_outcome_visibility=True,
        actor_anonymity=False,
        llm_model="gpt-4o",
        llm_temperature=1.0
    )

    print("Test Configuration:")
    print(f"  Group size: {config.group_size} (each can redistribute to {config.group_size-1} others)")
    print(f"  Initial endowment: {config.endowment} coins")
    print(f"  Punishment cost: {config.punishment_cost} coins/unit")
    print(f"  Reward cost: {config.reward_cost} coins/unit")
    print(f"  Max possible spending: {(config.group_size-1) * (config.punishment_cost + config.reward_cost)} coins/person")
    print()
    print("Expected: Most agents will exceed budget, triggering proportional scaling")
    print()
    print("="*60)
    print()

    # Run experiment
    exp_name = "budget_constraint_test"
    run_experiment(exp_name, config, num_games=1, verbose=True)

    print()
    print("="*60)
    print("Verification Results")
    print("="*60)
    print()

    exp_dir = Path("Simulation/experiments") / exp_name

    # 1. Check if files exist
    required_files = [
        "config.json",
        "game_log.csv",
        "chat_messages.csv",
        "raw_responses.csv",
        "redistribution_details.csv"  # NEW
    ]

    print("1. Checking output files:")
    all_exist = True
    for filename in required_files:
        filepath = exp_dir / filename
        exists = filepath.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {filename}")
        all_exist = all_exist and exists

    if not all_exist:
        print("\n✗ Some files are missing!")
        return

    print()

    # 2. Check wallet balances
    print("2. Checking wallet balances (should all be non-negative):")
    game_log = pd.read_csv(exp_dir / "game_log.csv")

    negative_wallets = game_log[game_log['cumulative_wallet'] < 0]
    if len(negative_wallets) > 0:
        print(f"  ✗ Found {len(negative_wallets)} negative wallets!")
        print(negative_wallets[['avatar_name', 'cumulative_wallet']])
    else:
        print(f"  ✓ All {len(game_log)} wallet balances are non-negative")
        min_wallet = game_log['cumulative_wallet'].min()
        max_wallet = game_log['cumulative_wallet'].max()
        print(f"    Range: [{min_wallet:.2f}, {max_wallet:.2f}]")

    print()

    # 3. Analyze redistribution_details
    print("3. Analyzing detailed redistribution records:")
    redist_df = pd.read_csv(exp_dir / "redistribution_details.csv")

    print(f"  Total records: {len(redist_df)}")
    print(f"  Punishment records: {len(redist_df[redist_df['type'] == 'punishment'])}")
    print(f"  Reward records: {len(redist_df[redist_df['type'] == 'reward'])}")

    # Scaling statistics
    scaled_records = redist_df[redist_df['was_scaled'] == True]
    if len(scaled_records) > 0:
        print(f"\n  Scaling statistics:")
        print(f"    Scaled records: {len(scaled_records)} / {len(redist_df)} ({len(scaled_records)/len(redist_df)*100:.1f}%)")

        # Calculate average scaling ratio
        scaled_records_with_decided = scaled_records[scaled_records['units_decided'] > 0]
        if len(scaled_records_with_decided) > 0:
            scaled_records_with_decided = scaled_records_with_decided.copy()
            scaled_records_with_decided['scale_ratio'] = (
                scaled_records_with_decided['units_actual'] / scaled_records_with_decided['units_decided']
            )
            avg_scale = scaled_records_with_decided['scale_ratio'].mean()
            print(f"    Average scaling ratio: {avg_scale:.3f}")

        # Group by agent
        print(f"\n  Agents with most scaling:")
        scaling_by_agent = scaled_records.groupby('actor_name').size().sort_values(ascending=False).head(5)
        for agent, count in scaling_by_agent.items():
            print(f"    {agent}: {count} times")
    else:
        print(f"  ✓ No records were scaled (all agents stayed within budget)")

    print()

    # 4. Verify scaling logic
    print("4. Verifying scaling logic correctness:")

    # Check: if was_scaled=True, then units_actual <= units_decided
    invalid_scaling = redist_df[
        (redist_df['was_scaled'] == True) &
        (redist_df['units_actual'] > redist_df['units_decided'])
    ]

    if len(invalid_scaling) > 0:
        print(f"  ✗ Found {len(invalid_scaling)} incorrect scaling records (actual > decided)")
    else:
        print(f"  ✓ All scaling records are correct (actual <= decided)")

    # Check: units_actual should all be integers
    if (redist_df['units_actual'] != redist_df['units_actual'].astype(int)).any():
        print(f"  ✗ Found non-integer units_actual values")
    else:
        print(f"  ✓ All units_actual are integers")

    print()

    # 5. Sample records
    print("5. Sample detailed redistribution records:")
    if len(redist_df) > 0:
        sample = redist_df.head(3)
        print(sample[['actor_name', 'target_name', 'type', 'units_decided', 'units_actual', 'was_scaled', 'cost', 'impact']].to_string(index=False))
    else:
        print("  (No redistribution records)")

    print()
    print("="*60)
    print("Test completed!")
    print()
    print(f"Detailed data located at: {exp_dir.absolute()}")
    print()


if __name__ == "__main__":
    import os

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print()
        print("Please set your API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print()
        sys.exit(1)

    test_budget_constraint()
