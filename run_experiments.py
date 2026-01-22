"""
Experimental Design Script for PGG Simulation

This script:
1. Generates 40 experiment configurations (20 without punishment + 20 with punishment)
2. Uses Latin Hypercube Sampling for parameter space coverage
3. Runs all experiments with game_length=1
4. Analyzes results using Analysis module
5. Generates figures similar to the paper

Usage:
    python run_experiments.py
"""

import numpy as np
import sys
from pathlib import Path
from scipy.stats import qmc
import json
from dataclasses import asdict

# Add Simulation to path
sys.path.insert(0, str(Path(__file__).parent / "Simulation"))
from config import PGGConfig
from main import run_experiment


def generate_experiment_configs(n_samples=20, punishment_enabled=False, random_seed=42):
    """
    Generate experiment configurations using Latin Hypercube Sampling.

    Args:
        n_samples: Number of configurations to generate
        punishment_enabled: Whether to enable punishment
        random_seed: Random seed for reproducibility

    Returns:
        List of PGGConfig objects
    """
    # Set random seed
    np.random.seed(random_seed)

    # Define parameter ranges (14 parameters)
    # We'll use LHS to sample in [0, 1] then map to actual ranges
    sampler = qmc.LatinHypercube(d=14, seed=random_seed)
    samples = sampler.random(n=n_samples)

    configs = []

    for i, sample in enumerate(samples):
        # Map [0,1] samples to parameter ranges

        # Continuous parameters
        group_size = int(2 + sample[0] * 18)  # 2-20
        mpcr = 0.06 + sample[1] * 0.64  # 0.06-0.7

        # Binary parameters (threshold at 0.5)
        contribution_type = "all_or_nothing" if sample[2] > 0.5 else "variable"
        contribution_framing = "opt_out" if sample[3] > 0.5 else "opt_in"
        communication_enabled = False  # Always False - chat not implemented in EDSL version
        peer_outcome_visibility = bool(sample[5] > 0.5)
        actor_anonymity = bool(sample[6] > 0.5)
        horizon_knowledge = "unknown" if sample[7] > 0.5 else "known"

        # Unified cost for both punishment and reward
        peer_incentive_cost = int(1 + sample[8] * 3)  # 1-4
        punishment_impact = int(1 + sample[9] * 3)  # 1-4

        # Reward parameters
        reward_enabled = bool(sample[10] > 0.5)
        # sample[11] no longer used (was reward_cost, now unified)
        reward_impact = 0.5 + sample[12] * 1.0  # 0.5-1.5

        # Enforce mutual exclusivity: punishment XOR reward
        # If both would be enabled, randomly choose one
        if punishment_enabled and reward_enabled:
            if sample[11] > 0.5:  # Use sample[11] as tiebreaker
                punishment_enabled = False
            else:
                reward_enabled = False

        # Reserved for future parameter
        # sample[13] currently unused

        config = PGGConfig(
            # Game structure
            group_size=group_size,
            game_length=1,  # FIXED at 1
            endowment=20,   # FIXED at 20
            horizon_knowledge=horizon_knowledge,

            # Economic parameters
            mpcr=round(mpcr, 2),

            # Contribution mechanism
            contribution_type=contribution_type,
            contribution_framing=contribution_framing,

            # Social information
            communication_enabled=communication_enabled,
            peer_outcome_visibility=peer_outcome_visibility,
            actor_anonymity=actor_anonymity,

            # Punishment
            punishment_enabled=punishment_enabled,
            peer_incentive_cost=peer_incentive_cost,
            punishment_impact=punishment_impact,

            # Reward
            reward_enabled=reward_enabled,
            reward_impact=round(reward_impact, 2),

            # LLM settings
            llm_model="gpt-4o",
            llm_temperature=1.0
        )

        configs.append(config)

    return configs


def run_all_experiments(output_dir="experiments"):
    """
    Generate and run all 40 experiments.

    20 experiments without punishment (control)
    20 experiments with punishment (treatment)
    """
    print("=" * 80)
    print("EXPERIMENTAL DESIGN: 40 PGG Experiments")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  - Game length: FIXED at 1 round")
    print("  - Endowment: FIXED at 20 coins")
    print("  - Control group: 20 experiments WITHOUT punishment")
    print("  - Treatment group: 20 experiments WITH punishment")
    print("  - Sampling method: Latin Hypercube Sampling")
    print()
    print("=" * 80)
    print()

    # Generate configurations
    print("Generating experiment configurations...")
    control_configs = generate_experiment_configs(
        n_samples=10,
        punishment_enabled=False,
        random_seed=42
    )
    treatment_configs = generate_experiment_configs(
        n_samples=10,
        punishment_enabled=True,
        random_seed=42  # Same seed for matched pairs
    )

    print(f"✓ Generated {len(control_configs)} control configurations")
    print(f"✓ Generated {len(treatment_configs)} treatment configurations")
    print()

    # Print statistics about configurations
    print("Configuration Statistics:")
    print(f"  Control group:")
    control_reward = sum(1 for c in control_configs if c.reward_enabled)
    control_punishment = sum(1 for c in control_configs if c.punishment_enabled)
    print(f"    - Communication enabled: 0/{len(control_configs)} (0%) - Always False")
    print(f"    - Reward enabled: {control_reward}/{len(control_configs)} ({control_reward/len(control_configs)*100:.0f}%)")
    print(f"    - Punishment enabled: {control_punishment}/{len(control_configs)} ({control_punishment/len(control_configs)*100:.0f}%)")

    print(f"  Treatment group:")
    treatment_reward = sum(1 for c in treatment_configs if c.reward_enabled)
    treatment_punishment = sum(1 for c in treatment_configs if c.punishment_enabled)
    print(f"    - Communication enabled: 0/{len(treatment_configs)} (0%) - Always False")
    print(f"    - Reward enabled: {treatment_reward}/{len(treatment_configs)} ({treatment_reward/len(treatment_configs)*100:.0f}%)")
    print(f"    - Punishment enabled: {treatment_punishment}/{len(treatment_configs)} ({treatment_punishment/len(treatment_configs)*100:.0f}%)")
    print()

    # Save all configurations
    all_configs = []

    # Run control experiments
    print("=" * 80)
    print("RUNNING CONTROL EXPERIMENTS (without punishment)")
    print("=" * 80)
    print()

    for i, config in enumerate(control_configs, 1):
        exp_name = f"exp_{i:03d}_control"
        print(f"[{i}/20] Running {exp_name}...")
        print(f"  Group size: {config.group_size}, MPCR: {config.mpcr}, "
              f"Communication: {config.communication_enabled}, "
              f"Framing: {config.contribution_framing}")
        print(f"  Punishment: {config.punishment_enabled}, "
              f"Reward: {config.reward_enabled}")

        try:
            run_experiment(exp_name, config, num_games=1, verbose=False)
            all_configs.append({
                'experiment_name': exp_name,
                'config': asdict(config),
                'status': 'completed'
            })
            print(f"  ✓ Completed\n")
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            all_configs.append({
                'experiment_name': exp_name,
                'config': asdict(config),
                'status': 'failed',
                'error': str(e)
            })

    # Run treatment experiments
    print("=" * 80)
    print("RUNNING TREATMENT EXPERIMENTS (with punishment)")
    print("=" * 80)
    print()

    for i, config in enumerate(treatment_configs, 1):
        exp_name = f"exp_{i:03d}_treatment"
        print(f"[{i}/20] Running {exp_name}...")
        print(f"  Group size: {config.group_size}, MPCR: {config.mpcr}, "
              f"Communication: {config.communication_enabled}, "
              f"Framing: {config.contribution_framing}")
        print(f"  Punishment: {config.punishment_enabled}, "
              f"Reward: {config.reward_enabled}, "
              f"Peer incentive cost: {config.peer_incentive_cost}")

        try:
            run_experiment(exp_name, config, num_games=1, verbose=False)
            all_configs.append({
                'experiment_name': exp_name,
                'config': asdict(config),
                'status': 'completed'
            })
            print(f"  ✓ Completed\n")
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            all_configs.append({
                'experiment_name': exp_name,
                'config': asdict(config),
                'status': 'failed',
                'error': str(e)
            })

    # Save experiment manifest
    manifest_path = Path(output_dir) / "experiment_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(all_configs, f, indent=2)

    print("=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)
    print()
    print(f"Total experiments run: {len(all_configs)}")
    print(f"Successful: {sum(1 for c in all_configs if c['status'] == 'completed')}")
    print(f"Failed: {sum(1 for c in all_configs if c['status'] == 'failed')}")
    print()
    print(f"Experiment manifest saved to: {manifest_path}")
    print()


def config_to_key(config_dict):
    """
    Convert a config dict to a hashable key for comparison.
    Excludes punishment_enabled since we're comparing base configurations.
    Handles backward compatibility with old parameter names.
    """
    # Handle backward compatibility for peer_incentive_cost
    if 'peer_incentive_cost' in config_dict:
        peer_incentive_cost = config_dict['peer_incentive_cost']
    elif config_dict.get('punishment_enabled', False):
        # Old experiment with punishment - use punishment_cost
        peer_incentive_cost = config_dict.get('punishment_cost', 1)
    elif config_dict.get('reward_enabled', False):
        # Old experiment with reward - use reward_cost
        peer_incentive_cost = config_dict.get('reward_cost', 1)
    else:
        # No incentive mechanism
        peer_incentive_cost = 1

    key_params = [
        config_dict['group_size'],
        config_dict['mpcr'],
        config_dict['contribution_type'],
        config_dict['contribution_framing'],
        config_dict['communication_enabled'],
        config_dict['peer_outcome_visibility'],
        config_dict['actor_anonymity'],
        config_dict['horizon_knowledge'],
        config_dict['reward_enabled'],
        peer_incentive_cost,
        config_dict['reward_impact']
    ]
    return tuple(key_params)


def generate_additional_configs(n_samples=10, punishment_enabled=False,
                                existing_configs=None, max_group_size=10,
                                random_seed=100, max_attempts=1000):
    """
    Generate additional experiment configurations, avoiding duplicates.

    Args:
        n_samples: Number of new configurations to generate
        punishment_enabled: Whether to enable punishment
        existing_configs: Set of existing config keys to avoid
        max_group_size: Maximum group size (exclusive)
        random_seed: Random seed for reproducibility
        max_attempts: Maximum attempts to find unique configs

    Returns:
        List of new PGGConfig objects
    """
    if existing_configs is None:
        existing_configs = set()

    configs = []
    attempts = 0
    seed_offset = 0

    while len(configs) < n_samples and attempts < max_attempts:
        # Use different seeds for each attempt
        current_seed = random_seed + seed_offset
        np.random.seed(current_seed)

        # Generate one sample
        sample = np.random.rand(14)

        # Map to parameters with group_size constraint
        group_size = int(2 + sample[0] * (max_group_size - 2))  # 2 to max_group_size-1
        mpcr = 0.06 + sample[1] * 0.64  # 0.06-0.7

        contribution_type = "all_or_nothing" if sample[2] > 0.5 else "variable"
        contribution_framing = "opt_out" if sample[3] > 0.5 else "opt_in"
        communication_enabled = bool(sample[4] > 0.5)
        peer_outcome_visibility = bool(sample[5] > 0.5)
        actor_anonymity = bool(sample[6] > 0.5)
        horizon_knowledge = "unknown" if sample[7] > 0.5 else "known"

        # Unified cost for both punishment and reward
        peer_incentive_cost = int(1 + sample[8] * 3)  # 1-4
        punishment_impact = int(1 + sample[9] * 3)  # 1-4

        reward_enabled = bool(sample[10] > 0.5)
        # sample[11] no longer used (was reward_cost, now unified)
        reward_impact = 0.5 + sample[12] * 1.0  # 0.5-1.5

        # Enforce mutual exclusivity: punishment XOR reward
        # If both would be enabled, randomly choose one
        if punishment_enabled and reward_enabled:
            if sample[11] > 0.5:  # Use sample[11] as tiebreaker
                punishment_enabled = False
            else:
                reward_enabled = False

        # Create config
        config = PGGConfig(
            group_size=group_size,
            game_length=1,
            endowment=20,
            horizon_knowledge=horizon_knowledge,
            mpcr=round(mpcr, 2),
            contribution_type=contribution_type,
            contribution_framing=contribution_framing,
            communication_enabled=communication_enabled,
            peer_outcome_visibility=peer_outcome_visibility,
            actor_anonymity=actor_anonymity,
            punishment_enabled=punishment_enabled,
            peer_incentive_cost=peer_incentive_cost,
            punishment_impact=punishment_impact,
            reward_enabled=reward_enabled,
            reward_impact=round(reward_impact, 2),
            llm_model="gpt-4o",
            llm_temperature=1.0
        )

        # Check if this config is unique
        config_key = config_to_key(asdict(config))
        if config_key not in existing_configs:
            configs.append(config)
            existing_configs.add(config_key)

        seed_offset += 1
        attempts += 1

    if len(configs) < n_samples:
        print(f"Warning: Only generated {len(configs)} unique configs out of {n_samples} requested")

    return configs


def run_additional_experiments(output_dir="Simulation/experiments", n_samples=10, max_group_size=10):
    """
    Run additional experiments, avoiding duplicates and constraining group size.

    Args:
        output_dir: Directory containing existing experiments
        n_samples: Number of new experiments per group (control/treatment)
        max_group_size: Maximum group size (exclusive)
    """
    print("=" * 80)
    print("RUNNING ADDITIONAL EXPERIMENTS")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  - New experiments per group: {n_samples}")
    print(f"  - Group size constraint: 2 <= group_size < {max_group_size}")
    print(f"  - Game length: FIXED at 1 round")
    print(f"  - Endowment: FIXED at 20 coins")
    print()

    # Load existing manifest
    manifest_path = Path(output_dir) / "experiment_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            existing_experiments = json.load(f)
        print(f"✓ Loaded {len(existing_experiments)} existing experiments")
    else:
        existing_experiments = []
        print("⚠ No existing manifest found, starting fresh")

    # Extract existing config keys
    existing_keys = set()
    for exp in existing_experiments:
        if exp['status'] == 'completed':
            existing_keys.add(config_to_key(exp['config']))
    print(f"✓ Found {len(existing_keys)} unique completed configurations")
    print()

    # Generate new configurations
    print("Generating new experiment configurations...")
    control_configs = generate_additional_configs(
        n_samples=n_samples,
        punishment_enabled=False,
        existing_configs=existing_keys.copy(),
        max_group_size=max_group_size,
        random_seed=200
    )

    treatment_configs = generate_additional_configs(
        n_samples=n_samples,
        punishment_enabled=True,
        existing_configs=existing_keys.copy(),
        max_group_size=max_group_size,
        random_seed=200  # Same seed for matched pairs
    )

    print(f"✓ Generated {len(control_configs)} new control configurations")
    print(f"✓ Generated {len(treatment_configs)} new treatment configurations")
    print()

    # Determine starting experiment number
    existing_numbers = []
    for exp in existing_experiments:
        exp_name = exp['experiment_name']
        if exp_name.startswith('exp_'):
            num_str = exp_name.split('_')[1]
            existing_numbers.append(int(num_str))

    start_num = max(existing_numbers) + 1 if existing_numbers else 1

    # Run control experiments
    print("=" * 80)
    print("RUNNING ADDITIONAL CONTROL EXPERIMENTS (without punishment)")
    print("=" * 80)
    print()

    for i, config in enumerate(control_configs, start=start_num):
        exp_name = f"exp_{i:03d}_control"
        print(f"[{i-start_num+1}/{len(control_configs)}] Running {exp_name}...")
        print(f"  Group size: {config.group_size}, MPCR: {config.mpcr}, "
              f"Communication: {config.communication_enabled}, "
              f"Framing: {config.contribution_framing}")

        try:
            run_experiment(exp_name, config, num_games=1, verbose=False)
            existing_experiments.append({
                'experiment_name': exp_name,
                'config': asdict(config),
                'status': 'completed'
            })
            print(f"  ✓ Completed\n")
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            existing_experiments.append({
                'experiment_name': exp_name,
                'config': asdict(config),
                'status': 'failed',
                'error': str(e)
            })

    # Run treatment experiments
    print("=" * 80)
    print("RUNNING ADDITIONAL TREATMENT EXPERIMENTS (with punishment)")
    print("=" * 80)
    print()

    for i, config in enumerate(treatment_configs, start=start_num):
        exp_name = f"exp_{i:03d}_treatment"
        print(f"[{i-start_num+1}/{len(treatment_configs)}] Running {exp_name}...")
        print(f"  Group size: {config.group_size}, MPCR: {config.mpcr}, "
              f"Peer incentive cost: {config.peer_incentive_cost}, "
              f"Punishment impact: {config.punishment_impact}")

        try:
            run_experiment(exp_name, config, num_games=1, verbose=False)
            existing_experiments.append({
                'experiment_name': exp_name,
                'config': asdict(config),
                'status': 'completed'
            })
            print(f"  ✓ Completed\n")
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            existing_experiments.append({
                'experiment_name': exp_name,
                'config': asdict(config),
                'status': 'failed',
                'error': str(e)
            })

    # Save updated manifest
    with open(manifest_path, 'w') as f:
        json.dump(existing_experiments, f, indent=2)

    print("=" * 80)
    print("ADDITIONAL EXPERIMENTS COMPLETED")
    print("=" * 80)
    print()
    print(f"Total experiments in manifest: {len(existing_experiments)}")
    print(f"Successful: {sum(1 for e in existing_experiments if e['status'] == 'completed')}")
    print(f"Failed: {sum(1 for e in existing_experiments if e['status'] == 'failed')}")
    print()
    print(f"Updated manifest saved to: {manifest_path}")
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

    # Uncomment the function you want to run:

    # Run initial experiments (already done)
    # run_all_experiments(output_dir="Simulation/experiments")

    # Run additional experiments with group_size < 10
    run_additional_experiments(
        output_dir="Simulation/experiments",
        n_samples=10,  # 10 control + 10 treatment
        max_group_size=15  # group_size will be in range [2, 9]
    )

    print("=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print()
    print("1. Analyze results:")
    print("   python analyze_results.py")
    print()
    print("2. Generate figures:")
    print("   The analysis script will create:")
    print("   - Figure: Punishment Effect (contribution & efficiency)")
    print("   - Figure: Feature Importance (PFI & SHAP)")
    print()
    print("=" * 80)
