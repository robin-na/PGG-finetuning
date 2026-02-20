"""
Quick test script to verify the new logging features.

This script runs a minimal experiment to test:
1. Chat message logging to chat_messages.csv
2. Raw response logging to raw_responses.csv
3. All three CSV files are created correctly
"""

import sys
from pathlib import Path

# Add Simulation to path
sys.path.insert(0, str(Path(__file__).parent / "Simulation"))

from config import PGGConfig
from main import run_experiment


def test_logging():
    """Run a small test experiment with all logging features enabled."""

    print("="*60)
    print("TESTING ENHANCED LOGGING FEATURES")
    print("="*60)
    print()

    # Create a test configuration with communication enabled
    config = PGGConfig(
        group_size=3,  # Small group for quick test
        game_length=1,  # Single round
        endowment=20,
        mpcr=0.4,
        communication_enabled=True,  # Test chat logging
        punishment_enabled=True,  # Test redistribution logging
        peer_outcome_visibility=True,
        llm_model="gpt-4o",
        llm_temperature=1.0
    )

    print("Test Configuration:")
    print(f"  Group size: {config.group_size}")
    print(f"  Rounds: {config.game_length}")
    print(f"  Communication: {config.communication_enabled}")
    print(f"  Punishment: {config.punishment_enabled}")
    print()
    print("="*60)
    print()

    # Run the experiment
    exp_name = "logging_test"
    run_experiment(exp_name, config, num_games=1, verbose=True)

    print()
    print("="*60)
    print("VERIFYING OUTPUT FILES")
    print("="*60)
    print()

    # Check output files
    exp_dir = Path("Simulation/experiments") / exp_name

    files_to_check = [
        ("config.json", "Configuration file"),
        ("game_log.csv", "Main game data"),
        ("chat_messages.csv", "Chat messages (NEW)"),
        ("raw_responses.csv", "Raw LLM responses (NEW)")
    ]

    all_exist = True
    for filename, description in files_to_check:
        filepath = exp_dir / filename
        exists = filepath.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {description}: {filename}")

        if exists and filename.endswith('.csv'):
            # Count lines
            with open(filepath) as f:
                lines = f.readlines()
                print(f"   └─ {len(lines)-1} data rows (excluding header)")

        all_exist = all_exist and exists

    print()

    if all_exist:
        print("SUCCESS: All output files created correctly!")
        print()
        print("You can inspect the files at:")
        print(f"  {exp_dir.absolute()}")
    else:
        print("WARNING: Some files are missing!")

    print()
    print("="*60)

    # Show sample of raw responses
    if (exp_dir / "raw_responses.csv").exists():
        print()
        print("SAMPLE RAW RESPONSES")
        print("="*60)
        import csv
        with open(exp_dir / "raw_responses.csv") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 2:  # Show first 2 responses
                    break
                print(f"\nAgent: {row['avatar_name']}")
                print(f"Type: {row['prompt_type']}")
                print(f"Response: {row['raw_response'][:150]}...")
                print(f"Parsed: {row['parsed_result']}")
        print()
        print("="*60)


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

    test_logging()
