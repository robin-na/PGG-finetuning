"""
Contribution analysis module for PGG experiments.

This module implements three core metrics:
1. Average Contribution: Percentage of endowment contributed per round
2. Normalized Efficiency: Actual earnings vs. theoretical baselines
3. Punishment Effect: Difference between treatment and control experiments

All metrics follow the methodology from the paper.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

# Add Simulation directory to path to import config and environment
sys.path.insert(0, str(Path(__file__).parent.parent / "Simulation"))
from config import PGGConfig
from environment import EfficiencyCalculator, RoundState


def calculate_average_contribution(
    game_log_df: pd.DataFrame,
    endowment: int = 20
) -> float:
    """Calculate average contribution as percentage of endowment.

    Args:
        game_log_df: DataFrame from game_log.csv with columns:
            - contribution: Amount contributed
            - round: Round number
            - agent_id: Agent identifier
        endowment: Endowment amount per round (default: 20)

    Returns:
        float: Average contribution percentage (0.0 to 1.0)

    Example:
        >>> df = pd.read_csv("experiments/baseline/game_log.csv")
        >>> avg_contrib = calculate_average_contribution(df)
        >>> print(f"Average contribution: {avg_contrib:.2%}")
    """
    if game_log_df.empty:
        return 0.0

    # Calculate per-round contribution percentage
    game_log_df = game_log_df.copy()
    game_log_df['contribution_pct'] = game_log_df['contribution'] / endowment

    # Return mean across all rounds and agents
    return game_log_df['contribution_pct'].mean()


def calculate_efficiency_from_log(
    game_log_path: str,
    config_path: str = None
) -> float:
    """Calculate normalized efficiency by replaying game from CSV log.

    This function reconstructs the game history from the CSV log and calculates
    efficiency using the EfficiencyCalculator. Handles dropouts properly by
    tracking active players per round.

    Args:
        game_log_path: Path to game_log.csv file
        config_path: Path to config.json file (default: same directory as log)

    Returns:
        float: Normalized efficiency (0.0 to 1.0, can be negative or > 1)

    Example:
        >>> efficiency = calculate_efficiency_from_log(
        ...     "experiments/baseline/game_log.csv"
        ... )
        >>> print(f"Normalized efficiency: {efficiency:.3f}")
    """
    # Load game log
    game_log_path = Path(game_log_path)
    df = pd.read_csv(game_log_path)

    if df.empty:
        return 0.0

    # Load config
    if config_path is None:
        config_path = game_log_path.parent / "config.json"

    with open(config_path, 'r') as f:
        config_data = json.load(f)
        config_dict = config_data.get('config', config_data)

    # Create PGGConfig object
    # Handle backward compatibility for old parameter names
    # Old experiments used 'punishment_cost' and 'reward_cost'
    # New experiments use unified 'peer_incentive_cost'
    punishment_enabled = config_dict.get('punishment_enabled', False)
    reward_enabled = config_dict.get('reward_enabled', False)

    if 'peer_incentive_cost' in config_dict:
        peer_incentive_cost = config_dict['peer_incentive_cost']
    elif punishment_enabled:
        # Old experiment with punishment - use punishment_cost
        peer_incentive_cost = config_dict.get('punishment_cost', 1)
    elif reward_enabled:
        # Old experiment with reward - use reward_cost
        peer_incentive_cost = config_dict.get('reward_cost', 1)
    else:
        # No incentive mechanism
        peer_incentive_cost = 1

    config = PGGConfig(
        group_size=config_dict['group_size'],
        game_length=config_dict['game_length'],
        endowment=config_dict['endowment'],
        mpcr=config_dict['mpcr'],
        horizon_knowledge=config_dict.get('horizon_knowledge', 'known'),
        contribution_type=config_dict.get('contribution_type', 'variable'),
        contribution_framing=config_dict.get('contribution_framing', 'opt_in'),
        communication_enabled=config_dict.get('communication_enabled', False),
        peer_outcome_visibility=config_dict.get('peer_outcome_visibility', True),
        actor_anonymity=config_dict.get('actor_anonymity', False),
        punishment_enabled=punishment_enabled,
        peer_incentive_cost=peer_incentive_cost,
        punishment_impact=config_dict.get('punishment_impact', 3),
        reward_enabled=reward_enabled,
        reward_impact=config_dict.get('reward_impact', 1.0),
        llm_model=config_dict.get('llm_model', 'gpt-4o'),
        llm_temperature=config_dict.get('llm_temperature', 1.0)
    )

    # Initialize efficiency calculator
    eff_calc = EfficiencyCalculator(config)

    # Group by round to reconstruct round states
    for round_num, round_df in df.groupby('round'):
        # Extract payoffs for this round
        payoffs = {}
        for _, row in round_df.iterrows():
            agent_id = row['agent_id']
            # Round payoff is the payoff for this round (already includes redistributions)
            payoffs[agent_id] = row['round_payoff']

        # Create a minimal RoundState for efficiency calculation
        # We only need payoffs for the EfficiencyCalculator
        round_state = type('RoundState', (), {'payoffs': payoffs})()

        # Update efficiency calculator
        active_player_count = len(payoffs)
        eff_calc.update_round(round_state, active_player_count)

    # Return final efficiency
    return eff_calc.get_efficiency()


def calculate_metrics_from_experiment(
    experiment_dir: str
) -> Dict[str, float]:
    """Calculate all metrics for a single experiment.

    Args:
        experiment_dir: Path to experiment directory containing:
            - game_log.csv
            - config.json

    Returns:
        dict with keys:
            - average_contribution: Average contribution percentage
            - normalized_efficiency: Normalized efficiency
            - endowment: Endowment amount used

    Example:
        >>> metrics = calculate_metrics_from_experiment("experiments/baseline")
        >>> print(f"Contribution: {metrics['average_contribution']:.2%}")
        >>> print(f"Efficiency: {metrics['normalized_efficiency']:.3f}")
    """
    experiment_dir = Path(experiment_dir)
    game_log_path = experiment_dir / "game_log.csv"
    config_path = experiment_dir / "config.json"

    # Load config to get endowment
    with open(config_path, 'r') as f:
        config_data = json.load(f)
        config_dict = config_data.get('config', config_data)
        endowment = config_dict.get('endowment', 20)

    # Load game log
    df = pd.read_csv(game_log_path)

    # Calculate metrics
    avg_contrib = calculate_average_contribution(df, endowment)
    efficiency = calculate_efficiency_from_log(game_log_path, config_path)

    return {
        'average_contribution': avg_contrib,
        'normalized_efficiency': efficiency,
        'endowment': endowment
    }


def calculate_punishment_effect(
    treatment_dir: str,
    control_dir: str
) -> Dict[str, float]:
    """Calculate punishment effect by comparing paired experiments.

    The punishment effect measures how enabling punishment changes both
    cooperation levels and efficiency compared to a control condition.

    Args:
        treatment_dir: Path to experiment with punishment enabled
        control_dir: Path to experiment with punishment disabled

    Returns:
        dict with keys:
            - efficiency_treatment: Normalized efficiency with punishment
            - efficiency_control: Normalized efficiency without punishment
            - punishment_effect_efficiency: Difference (treatment - control)
            - contribution_treatment: Average contribution with punishment
            - contribution_control: Average contribution without punishment
            - punishment_effect_contribution: Difference (treatment - control)

    Example:
        >>> effect = calculate_punishment_effect(
        ...     "experiments/baseline_treatment",
        ...     "experiments/baseline_control"
        ... )
        >>> print(f"Punishment effect on efficiency: {effect['punishment_effect_efficiency']:+.3f}")
        >>> print(f"Punishment effect on contribution: {effect['punishment_effect_contribution']:+.2%}")
    """
    # Calculate metrics for both experiments
    treatment_metrics = calculate_metrics_from_experiment(treatment_dir)
    control_metrics = calculate_metrics_from_experiment(control_dir)

    # Calculate effects (differences)
    return {
        'efficiency_treatment': treatment_metrics['normalized_efficiency'],
        'efficiency_control': control_metrics['normalized_efficiency'],
        'punishment_effect_efficiency': (
            treatment_metrics['normalized_efficiency'] -
            control_metrics['normalized_efficiency']
        ),
        'contribution_treatment': treatment_metrics['average_contribution'],
        'contribution_control': control_metrics['average_contribution'],
        'punishment_effect_contribution': (
            treatment_metrics['average_contribution'] -
            control_metrics['average_contribution']
        )
    }


def save_metrics(
    experiment_dir: str,
    metrics: Dict[str, float],
    filename: str = "metrics.json"
):
    """Save calculated metrics to JSON file.

    Args:
        experiment_dir: Directory to save metrics
        metrics: Dictionary of metrics to save
        filename: Output filename (default: metrics.json)

    Example:
        >>> metrics = calculate_metrics_from_experiment("experiments/baseline")
        >>> save_metrics("experiments/baseline", metrics)
    """
    experiment_dir = Path(experiment_dir)
    output_path = experiment_dir / filename

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {output_path}")


# ===== Testing / Demo =====
if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("Contribution Analysis Module - Testing")
    print("=" * 70)

    # Check if experiment directory is provided
    if len(sys.argv) > 1:
        experiment_dir = sys.argv[1]
        print(f"\nAnalyzing experiment: {experiment_dir}")

        try:
            # Calculate and display metrics
            metrics = calculate_metrics_from_experiment(experiment_dir)

            print("\nResults:")
            print(f"  Average Contribution: {metrics['average_contribution']:.2%}")
            print(f"  Normalized Efficiency: {metrics['normalized_efficiency']:.3f}")
            print(f"  Endowment: {metrics['endowment']}")

            # Save metrics
            save_metrics(experiment_dir, metrics)

        except Exception as e:
            print(f"\nError analyzing experiment: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("\nUsage:")
        print("  python contribution.py <experiment_directory>")
        print("\nExample:")
        print("  python contribution.py ../Simulation/experiments/baseline")
        print("\nOr for punishment effect analysis:")
        print("  python -c \"from contribution import calculate_punishment_effect; \\")
        print("             print(calculate_punishment_effect('treatment_dir', 'control_dir'))\"")

        # Check if there are any experiments to analyze
        experiments_dir = Path(__file__).parent.parent / "Simulation" / "experiments"
        if experiments_dir.exists():
            experiment_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
            if experiment_dirs:
                print(f"\nAvailable experiments in {experiments_dir}:")
                for exp_dir in sorted(experiment_dirs)[:5]:  # Show first 5
                    print(f"  - {exp_dir.name}")
                if len(experiment_dirs) > 5:
                    print(f"  ... and {len(experiment_dirs) - 5} more")
