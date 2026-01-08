"""
Analysis Script for PGG Experimental Results

This script:
1. Loads all experiment results
2. Calculates key metrics (contribution, efficiency, punishment effect)
3. Generates figures similar to the paper:
   - Figure 1: Punishment Effect on Contribution and Efficiency
   - Figure 2: Feature Importance Analysis

Usage:
    python analyze_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json

# Add Analysis to path
sys.path.insert(0, str(Path(__file__).parent / "Analysis"))
from contribution import calculate_metrics_from_experiment
from feature import (
    prepare_ml_dataset,
    train_enet_model,
    calculate_pfi,
    plot_pfi
)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11


def load_experiment_results(experiments_dir="Simulation/experiments"):
    """
    Load all experiment results and calculate metrics.

    Returns:
        DataFrame with columns: experiment_id, punishment_enabled, metrics...
    """
    experiments_dir = Path(experiments_dir)
    results = []

    # Find all experiment directories
    exp_dirs = sorted([d for d in experiments_dir.iterdir() if d.is_dir()])

    print(f"Found {len(exp_dirs)} experiment directories")
    print("Calculating metrics...")

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name

        # Determine if punishment was enabled
        punishment_enabled = "treatment" in exp_name

        try:
            # Calculate metrics
            metrics = calculate_metrics_from_experiment(str(exp_dir))

            # Load config for additional info
            with open(exp_dir / "config.json", 'r') as f:
                config_data = json.load(f)
                config = config_data.get('config', config_data)

            results.append({
                'experiment_id': exp_name,
                'punishment_enabled': punishment_enabled,
                'average_contribution': metrics['average_contribution'],
                'normalized_efficiency': metrics['normalized_efficiency'],
                'group_size': config['group_size'],
                'mpcr': config['mpcr'],
                'communication': config.get('communication_enabled', False),
                'contribution_framing': config.get('contribution_framing', 'opt_in'),
                'peer_visibility': config.get('peer_outcome_visibility', True),
                'actor_anonymity': config.get('actor_anonymity', False)
            })

            print(f"  ✓ {exp_name}: contribution={metrics['average_contribution']:.2%}, "
                  f"efficiency={metrics['normalized_efficiency']:.3f}")

        except Exception as e:
            print(f"  ✗ {exp_name}: Error - {e}")
            continue

    df = pd.DataFrame(results)
    print(f"\nSuccessfully analyzed {len(df)} experiments")

    return df


def plot_punishment_effect(df, save_path="analysis_outputs"):
    """
    Generate Figure 1: Punishment Effect on Contribution and Efficiency

    Creates a figure with two panels:
    A) Average Contribution (with vs. without punishment)
    B) Normalized Efficiency (with vs. without punishment)

    Style: Horizontal point plot with error bars
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    control = df[df['punishment_enabled'] == False]
    treatment = df[df['punishment_enabled'] == True]

    # Calculate means and 95% CI
    from scipy import stats

    def calculate_ci(data, confidence=0.95):
        """Calculate 95% confidence interval"""
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        return mean, ci

    # Contribution statistics
    contrib_control_mean, contrib_control_ci = calculate_ci(control['average_contribution'].values)
    contrib_treatment_mean, contrib_treatment_ci = calculate_ci(treatment['average_contribution'].values)

    # Efficiency statistics
    eff_control_mean, eff_control_ci = calculate_ci(control['normalized_efficiency'].values)
    eff_treatment_mean, eff_treatment_ci = calculate_ci(treatment['normalized_efficiency'].values)

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Define colors (single dataset style)
    color_main = '#4A90E2'  # Blue

    # Panel A: Average Contribution
    ax = axes[0]

    # Plot data points with error bars (horizontal)
    y_positions = [1, 0]  # With punishment at top, Without at bottom
    means_contrib = [contrib_treatment_mean, contrib_control_mean]
    cis_contrib = [contrib_treatment_ci, contrib_control_ci]

    # Plot points and error bars
    ax.errorbar(means_contrib, y_positions,
                xerr=cis_contrib,
                fmt='o',
                markersize=10,
                color=color_main,
                ecolor=color_main,
                elinewidth=2,
                capsize=5,
                capthick=2)

    # Styling
    ax.set_yticks(y_positions)
    ax.set_yticklabels(['With\npunishment', 'Without\npunishment'], fontsize=11)
    ax.set_xlabel('Average contribution', fontsize=12)
    ax.set_title('A', fontsize=16, fontweight='bold', loc='left', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))

    # Set x-axis limits with some padding
    x_min = min(means_contrib) - max(cis_contrib) - 0.05
    x_max = max(means_contrib) + max(cis_contrib) + 0.05
    ax.set_xlim(x_min, x_max)

    # Panel B: Normalized Efficiency
    ax = axes[1]

    means_eff = [eff_treatment_mean, eff_control_mean]
    cis_eff = [eff_treatment_ci, eff_control_ci]

    # Plot points and error bars
    ax.errorbar(means_eff, y_positions,
                xerr=cis_eff,
                fmt='o',
                markersize=10,
                color=color_main,
                ecolor=color_main,
                elinewidth=2,
                capsize=5,
                capthick=2)

    # Styling
    ax.set_yticks(y_positions)
    ax.set_yticklabels(['With\npunishment', 'Without\npunishment'], fontsize=11)
    ax.set_xlabel('Normalized efficiency', fontsize=12)
    ax.set_title('B', fontsize=16, fontweight='bold', loc='left', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Set x-axis limits with some padding
    x_min = min(means_eff) - max(cis_eff) - 0.1
    x_max = max(means_eff) + max(cis_eff) + 0.1
    ax.set_xlim(x_min, x_max)

    plt.tight_layout()
    output_path = save_path / "figure_1_punishment_effect.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved Figure 1 to: {output_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 80)
    print("PUNISHMENT EFFECT SUMMARY")
    print("=" * 80)
    print()
    print("Average Contribution:")
    effect_contrib = contrib_treatment_mean - contrib_control_mean
    print(f"  Without punishment: {contrib_control_mean:.2%} ± {contrib_control_ci:.2%} (95% CI)")
    print(f"  With punishment:    {contrib_treatment_mean:.2%} ± {contrib_treatment_ci:.2%} (95% CI)")
    print(f"  Effect:             {effect_contrib:+.2%}")
    print()
    print("Normalized Efficiency:")
    effect_eff = eff_treatment_mean - eff_control_mean
    print(f"  Without punishment: {eff_control_mean:.3f} ± {eff_control_ci:.3f} (95% CI)")
    print(f"  With punishment:    {eff_treatment_mean:.3f} ± {eff_treatment_ci:.3f} (95% CI)")
    print(f"  Effect:             {effect_eff:+.3f}")
    print()


def plot_feature_importance(df, save_path="analysis_outputs"):
    """
    Generate Figure 2: Feature Importance Analysis

    Uses Permutation Feature Importance to identify which design parameters
    most strongly influence cooperation efficiency.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    print()

    # Prepare dataset for ML
    print("Preparing ML dataset...")

    # Create dataset with treatment outcomes
    # For each experiment, we need control efficiency as baseline
    treatment_df = df[df['punishment_enabled'] == True].copy()
    control_df = df[df['punishment_enabled'] == False].copy()

    # Match treatment with control (same index)
    # Need to extract ALL parameters from config files for feature analysis
    ml_data = []

    experiments_dir = Path("experiments")
    for i in range(min(len(treatment_df), len(control_df))):
        t_row = treatment_df.iloc[i]
        c_row = control_df.iloc[i]

        # Load full config for treatment experiment
        t_exp_dir = experiments_dir / t_row['experiment_id']
        with open(t_exp_dir / "config.json", 'r') as f:
            t_config_data = json.load(f)
            t_config = t_config_data.get('config', t_config_data)

        ml_data.append({
            'experiment_id': t_row['experiment_id'],
            # Game structure
            'group_size': t_config['group_size'],
            'game_length': t_config['game_length'],
            'mpcr': t_config['mpcr'],
            'horizon_knowledge': 1 if t_config.get('horizon_knowledge', 'known') == 'unknown' else 0,
            # Contribution mechanism
            'contribution_type': 1 if t_config.get('contribution_type', 'variable') == 'all_or_nothing' else 0,
            'contribution_framing': 1 if t_config.get('contribution_framing', 'opt_in') == 'opt_out' else 0,
            # Social information
            'communication': int(t_config.get('communication_enabled', False)),
            'peer_outcome_visibility': int(t_config.get('peer_outcome_visibility', True)),
            'actor_anonymity': int(t_config.get('actor_anonymity', False)),
            # Punishment parameters
            'punishment_cost': t_config.get('punishment_cost', 1),
            'punishment_impact': t_config.get('punishment_impact', 3),
            # Reward parameters
            'reward_enabled': int(t_config.get('reward_enabled', False)),
            'reward_cost': t_config.get('reward_cost', 1),
            'reward_impact': t_config.get('reward_impact', 1.0),
            # Outcomes
            'efficiency_control': c_row['normalized_efficiency'],
            'efficiency_treatment': t_row['normalized_efficiency']
        })

    ml_df = pd.DataFrame(ml_data)

    # Full feature set (14 parameters + efficiency_control)
    feature_cols = [
        'group_size',
        'game_length',
        'mpcr',
        'horizon_knowledge',
        'contribution_type',
        'contribution_framing',
        'communication',
        'peer_outcome_visibility',
        'actor_anonymity',
        'punishment_cost',
        'punishment_impact',
        'reward_enabled',
        'reward_cost',
        'reward_impact',
        'efficiency_control'
    ]

    # Train model
    print("Training Elastic Net model...")
    if len(ml_df) < 15:
        print(f"WARNING: Only {len(ml_df)} samples. Need ~20+ for reliable results.")
        print("Running simplified analysis...")

    try:
        model, X_train, X_val, y_val, _, baseline_rmse = train_enet_model(ml_df)

        # Calculate PFI
        print("\nCalculating Permutation Feature Importance...")
        df_pfi = calculate_pfi(
            model, X_val, y_val, baseline_rmse, feature_cols,
            n_repeats=10  # Reduced for speed
        )

        # Plot PFI
        plot_pfi(df_pfi, save_path=save_path / "figure_2_feature_importance.png")

        print("\n" + "=" * 80)
        print("TOP 5 MOST IMPORTANT FEATURES")
        print("=" * 80)
        print()
        for idx, row in df_pfi.head(5).iterrows():
            print(f"{idx+1}. {row['display_name']:25s} "
                  f"{row['mean_importance']:6.1%} ± {row['std_importance']:.1%}")
        print()

    except Exception as e:
        print(f"\nFeature importance analysis failed: {e}")
        print("This may be due to insufficient sample size or data variability.")


def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print("PGG EXPERIMENTAL RESULTS ANALYSIS")
    print("=" * 80)
    print()

    # Load results
    df = load_experiment_results("experiments")

    if len(df) == 0:
        print("\nERROR: No experiment results found!")
        print("Please run: python run_experiments.py first")
        return

    # Create output directory
    output_dir = Path("analysis_outputs")
    output_dir.mkdir(exist_ok=True)

    # Save raw data
    df.to_csv(output_dir / "experiment_results.csv", index=False)
    print(f"\n✓ Saved results to: {output_dir / 'experiment_results.csv'}")

    # Generate figures
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    # Figure 1: Punishment Effect
    plot_punishment_effect(df, save_path=output_dir)

    # Figure 2: Feature Importance
    plot_feature_importance(df, save_path=output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print(f"All outputs saved to: {output_dir.absolute()}")
    print()
    print("Generated files:")
    print("  - experiment_results.csv")
    print("  - figure_1_punishment_effect.png")
    print("  - figure_2_feature_importance.png")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
