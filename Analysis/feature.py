"""
Feature analysis and model interpretation module for PGG experiments.

This module implements machine learning model interpretation to replicate Figure 4:
1. Data Preparation: Aggregate experiment results for ML training
2. Model Training: Elastic Net with pairwise interactions
3. PFI: Permutation Feature Importance (Figure 4A)
4. SHAP: Shapley Additive Explanations (Figure 4B)

Methodology follows the paper's approach for identifying which design parameters
influence cooperation efficiency.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Machine learning imports
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt

# SHAP for model interpretation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")

# Add Analysis directory to path
sys.path.insert(0, str(Path(__file__).parent))
from contribution import calculate_metrics_from_experiment


# Feature names for the 14 design parameters + baseline efficiency
FEATURE_NAMES = [
    'group_size',
    'game_length',
    'mpcr',
    'communication',
    'contribution_type',
    'contribution_framing',
    'peer_outcome_visibility',
    'actor_anonymity',
    'horizon_knowledge',
    'punishment_cost',
    'punishment_impact',
    'reward_enabled',
    'reward_cost',
    'reward_impact',
    'efficiency_control'
]

# Feature display names for plots
FEATURE_DISPLAY_NAMES = {
    'group_size': 'Group Size',
    'game_length': 'Game Length',
    'mpcr': 'MPCR',
    'communication': 'Communication',
    'contribution_type': 'All-or-Nothing',
    'contribution_framing': 'Opt-Out Framing',
    'peer_outcome_visibility': 'Peer Visibility',
    'actor_anonymity': 'Actor Anonymity',
    'horizon_knowledge': 'Unknown Horizon',
    'punishment_cost': 'Punishment Cost',
    'punishment_impact': 'Punishment Impact',
    'reward_enabled': 'Reward Enabled',
    'reward_cost': 'Reward Cost',
    'reward_impact': 'Reward Impact',
    'efficiency_control': 'Baseline Efficiency'
}


def prepare_ml_dataset(
    experiments_dir: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Aggregate simulation results for ML model training.

    Scans the experiments directory for paired treatment/control experiments,
    calculates metrics, and creates a dataset suitable for ML training.

    Args:
        experiments_dir: Path to experiments/ directory
        output_path: Optional path to save aggregated CSV

    Returns:
        DataFrame with one row per experiment pair (treatment/control)
        Columns: 14 design parameters + efficiency_control + efficiency_treatment

    Example:
        >>> df = prepare_ml_dataset("../Simulation/experiments")
        >>> print(f"Dataset: {len(df)} experiments, {len(df.columns)} features")
    """
    experiments_dir = Path(experiments_dir)
    records = []

    # Find all treatment experiments (those with _treatment suffix or control pair)
    experiment_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]

    # Group experiments by base name (before _treatment or _control)
    experiment_pairs = {}
    for exp_dir in experiment_dirs:
        name = exp_dir.name
        if name.endswith('_treatment'):
            base_name = name[:-10]  # Remove '_treatment'
            if base_name not in experiment_pairs:
                experiment_pairs[base_name] = {}
            experiment_pairs[base_name]['treatment'] = exp_dir
        elif name.endswith('_control'):
            base_name = name[:-8]  # Remove '_control'
            if base_name not in experiment_pairs:
                experiment_pairs[base_name] = {}
            experiment_pairs[base_name]['control'] = exp_dir
        else:
            # No suffix, check if both treatment and control configs exist
            # For now, skip these
            pass

    print(f"Found {len(experiment_pairs)} experiment pairs")

    # Process each pair
    for base_name, pair in experiment_pairs.items():
        if 'treatment' not in pair or 'control' not in pair:
            print(f"Warning: Incomplete pair for {base_name}, skipping")
            continue

        treatment_dir = pair['treatment']
        control_dir = pair['control']

        try:
            # Load configs
            with open(treatment_dir / "config.json") as f:
                config_t = json.load(f)["config"]
            with open(control_dir / "config.json") as f:
                config_c = json.load(f)["config"]

            # Calculate metrics
            metrics_t = calculate_metrics_from_experiment(str(treatment_dir))
            metrics_c = calculate_metrics_from_experiment(str(control_dir))

            # Create record with features
            record = {
                'experiment_id': base_name,
                # Game structure (continuous)
                'group_size': config_t['group_size'],
                'game_length': config_t['game_length'],
                'mpcr': config_t['mpcr'],
                # Binary features
                'communication': int(config_t.get('communication_enabled', False)),
                'contribution_type': 1 if config_t.get('contribution_type', 'variable') == 'all_or_nothing' else 0,
                'contribution_framing': 1 if config_t.get('contribution_framing', 'opt_in') == 'opt_out' else 0,
                'peer_outcome_visibility': int(config_t.get('peer_outcome_visibility', True)),
                'actor_anonymity': int(config_t.get('actor_anonymity', False)),
                'horizon_knowledge': 1 if config_t.get('horizon_knowledge', 'known') == 'unknown' else 0,
                # Punishment/reward parameters (continuous)
                'punishment_cost': config_t.get('punishment_cost', 1),
                'punishment_impact': config_t.get('punishment_impact', 3),
                'reward_enabled': int(config_t.get('reward_enabled', False)),
                'reward_cost': config_t.get('reward_cost', 1),
                'reward_impact': config_t.get('reward_impact', 1.0),
                # Target and baseline
                'efficiency_control': metrics_c['normalized_efficiency'],
                'efficiency_treatment': metrics_t['normalized_efficiency']
            }
            records.append(record)

        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(records)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved dataset to: {output_path}")

    return df


def train_enet_model(
    df: pd.DataFrame,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, List[str], float]:
    """Train Elastic Net model with pairwise interactions.

    Critical: Must include PolynomialFeatures(degree=2, interaction_only=True)
    as per paper methodology.

    Args:
        df: DataFrame from prepare_ml_dataset()
        test_size: Fraction of data for validation (default: 0.15)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (model, X_train, X_val, y_val, feature_cols, baseline_rmse)

    Example:
        >>> df = prepare_ml_dataset("../Simulation/experiments")
        >>> model, X_train, X_val, y_val, features, rmse = train_enet_model(df)
        >>> print(f"Model RMSE: {rmse:.4f}")
    """
    # Feature columns (14 params + efficiency_control)
    feature_cols = FEATURE_NAMES.copy()

    X = df[feature_cols]
    y = df['efficiency_treatment']

    # Split: train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Build pipeline: Standardize → Interactions → Elastic Net
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('interactions', PolynomialFeatures(
            degree=2,
            interaction_only=True,  # Only X1*X2, not X1^2
            include_bias=False
        )),
        ('enet', ElasticNetCV(
            cv=5,
            random_state=random_state,
            max_iter=10000,
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
            n_jobs=-1
        ))
    ])

    print("\nTraining Elastic Net with pairwise interactions...")
    model.fit(X_train, y_train)

    # Evaluate baseline performance
    y_pred = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"\nValidation Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Best l1_ratio: {model.named_steps['enet'].l1_ratio_:.3f}")
    print(f"  Best alpha: {model.named_steps['enet'].alpha_:.6f}")

    return model, X_train, X_val, y_val, feature_cols, rmse


def calculate_pfi(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    baseline_rmse: float,
    feature_cols: List[str],
    n_repeats: int = 30,
    random_state: int = 42
) -> pd.DataFrame:
    """Calculate Permutation Feature Importance.

    For each feature:
    1. Shuffle its values in validation set
    2. Recalculate RMSE
    3. Compute % increase: (RMSE_shuffled - RMSE_baseline) / RMSE_baseline
    4. Repeat n_repeats times for confidence intervals

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation targets
        baseline_rmse: Baseline RMSE (unshuffled)
        feature_cols: List of feature names
        n_repeats: Number of shuffling repeats (default: 30)
        random_state: Random seed

    Returns:
        DataFrame with columns: feature, mean_importance, std_importance, ci_lower, ci_upper

    Example:
        >>> df_pfi = calculate_pfi(model, X_val, y_val, rmse, features)
        >>> print(df_pfi.head())
    """
    np.random.seed(random_state)
    results = []

    print(f"\nCalculating Permutation Feature Importance ({n_repeats} repeats)...")

    for idx, col in enumerate(feature_cols):
        importance_scores = []

        for repeat in range(n_repeats):
            # Create shuffled copy
            X_val_shuffled = X_val.copy()
            X_val_shuffled[col] = np.random.permutation(X_val[col].values)

            # Predict and calculate RMSE
            y_pred_shuffled = model.predict(X_val_shuffled)
            rmse_shuffled = root_mean_squared_error(y_val, y_pred_shuffled)

            # Calculate % increase
            pct_increase = (rmse_shuffled - baseline_rmse) / baseline_rmse
            importance_scores.append(pct_increase)

        # Aggregate statistics
        mean_imp = np.mean(importance_scores)
        std_imp = np.std(importance_scores)
        ci_lower = np.percentile(importance_scores, 2.5)
        ci_upper = np.percentile(importance_scores, 97.5)

        results.append({
            'feature': col,
            'display_name': FEATURE_DISPLAY_NAMES.get(col, col),
            'mean_importance': mean_imp,
            'std_importance': std_imp,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })

        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx + 1}/{len(feature_cols)} features")

    df_pfi = pd.DataFrame(results).sort_values('mean_importance', ascending=False)
    print("PFI calculation complete.")

    return df_pfi


def plot_pfi(
    df_pfi: pd.DataFrame,
    save_path: str = "pfi_results.png",
    figsize: Tuple[int, int] = (10, 8)
):
    """Generate Figure 4A: PFI bar chart with error bars.

    Args:
        df_pfi: DataFrame from calculate_pfi()
        save_path: Output file path
        figsize: Figure size (width, height)

    Example:
        >>> plot_pfi(df_pfi, "analysis_outputs/figure_4a.png")
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by importance
    df_plot = df_pfi.sort_values('mean_importance')

    # Create horizontal bar chart
    y_pos = np.arange(len(df_plot))
    ax.barh(y_pos, df_plot['mean_importance'],
            xerr=[df_plot['mean_importance'] - df_plot['ci_lower'],
                  df_plot['ci_upper'] - df_plot['mean_importance']],
            capsize=3, alpha=0.7, color='steelblue')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['display_name'])
    ax.set_xlabel('% Increase in RMSE (Permutation Importance)', fontsize=12)
    ax.set_title('Permutation Feature Importance (30 repeats)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved PFI plot to: {save_path}")
    plt.close()


def calculate_shap(
    model,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Calculate SHAP values for feature interpretation.

    Challenge: Pipeline includes PolynomialFeatures which creates interaction terms.
    Solution: Transform data manually and use LinearExplainer on the final estimator.

    Args:
        model: Trained model pipeline
        X_train: Training features
        X_val: Validation features
        feature_cols: List of feature names

    Returns:
        Tuple of (shap_values_original, X_val) where shap_values are aggregated
        back to original 15 features

    Example:
        >>> shap_values, X_val = calculate_shap(model, X_train, X_val, features)
        >>> print(f"SHAP values shape: {shap_values.shape}")
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")

    print("\nCalculating SHAP values...")

    # Extract preprocessing steps
    scaler = model.named_steps['scaler']
    poly = model.named_steps['interactions']
    enet = model.named_steps['enet']

    # Transform data
    X_train_scaled = scaler.transform(X_train)
    X_train_poly = poly.transform(X_train_scaled)
    X_val_scaled = scaler.transform(X_val)
    X_val_poly = poly.transform(X_val_scaled)

    # Get feature names after polynomial transformation
    feature_names_poly = poly.get_feature_names_out(feature_cols)

    print(f"  Original features: {len(feature_cols)}")
    print(f"  Polynomial features (with interactions): {len(feature_names_poly)}")

    # Create LinearExplainer for Elastic Net
    explainer = shap.LinearExplainer(enet, X_train_poly, feature_names=feature_names_poly)
    shap_values = explainer.shap_values(X_val_poly)

    print("  SHAP values calculated")

    # Aggregate SHAP values back to original features
    print("  Aggregating interaction terms back to original features...")
    shap_values_original = aggregate_shap_to_original_features(
        shap_values, feature_names_poly, feature_cols
    )

    print("SHAP calculation complete.")

    return shap_values_original, X_val


def aggregate_shap_to_original_features(
    shap_values: np.ndarray,
    feature_names_poly: List[str],
    original_features: List[str]
) -> np.ndarray:
    """Aggregate SHAP values from interaction terms back to original features.

    Strategy: For each interaction term "x_i x_j", add half of its SHAP value
    to both x_i and x_j. This is a heuristic approach to attribute interaction
    effects back to the constituent features.

    Args:
        shap_values: SHAP values for polynomial features (n_samples, n_poly_features)
        feature_names_poly: Feature names after PolynomialFeatures
        original_features: Original feature names

    Returns:
        np.ndarray: SHAP values aggregated to original features (n_samples, n_original)
    """
    n_samples = shap_values.shape[0]
    n_original = len(original_features)
    shap_aggregated = np.zeros((n_samples, n_original))

    # Map feature names to indices
    feature_to_idx = {feat: idx for idx, feat in enumerate(original_features)}

    for poly_idx, poly_name in enumerate(feature_names_poly):
        # PolynomialFeatures creates names like "x0 x1" for interactions
        # Check if this is an interaction term (contains space)
        if ' ' in poly_name:
            # Split interaction: "x0 x1" -> ["x0", "x1"]
            parts = poly_name.split(' ')
            # Distribute SHAP value equally to both features
            for part in parts:
                if part in feature_to_idx:
                    orig_idx = feature_to_idx[part]
                    shap_aggregated[:, orig_idx] += shap_values[:, poly_idx] / len(parts)
        else:
            # Main effect term (no interaction)
            if poly_name in feature_to_idx:
                orig_idx = feature_to_idx[poly_name]
                shap_aggregated[:, orig_idx] += shap_values[:, poly_idx]

    return shap_aggregated


def plot_shap(
    shap_values: np.ndarray,
    X_val: pd.DataFrame,
    save_path: str = "shap_summary.png",
    figsize: Tuple[int, int] = (10, 8)
):
    """Generate Figure 4B: SHAP summary plot.

    Args:
        shap_values: SHAP values (n_samples, n_features)
        X_val: Validation features
        save_path: Output file path
        figsize: Figure size

    Example:
        >>> plot_shap(shap_values, X_val, "analysis_outputs/figure_4b.png")
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")

    # Rename columns to display names
    X_val_display = X_val.copy()
    X_val_display.columns = [FEATURE_DISPLAY_NAMES.get(col, col) for col in X_val.columns]

    # Create SHAP summary plot
    plt.figure(figsize=figsize)
    shap.summary_plot(
        shap_values,
        X_val_display,
        plot_type="dot",  # Violin plot style
        show=False
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved SHAP plot to: {save_path}")
    plt.close()


def save_analysis_results(
    output_dir: str,
    df_pfi: pd.DataFrame,
    shap_values: Optional[np.ndarray] = None,
    X_val: Optional[pd.DataFrame] = None,
    model_performance: Optional[Dict] = None
):
    """Save all analysis results to files.

    Args:
        output_dir: Directory to save outputs
        df_pfi: PFI results DataFrame
        shap_values: Optional SHAP values array
        X_val: Optional validation features
        model_performance: Optional dict with RMSE, R², etc.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save PFI results
    df_pfi.to_csv(output_dir / "pfi_results.csv", index=False)
    print(f"Saved PFI results to: {output_dir / 'pfi_results.csv'}")

    # Save SHAP values
    if shap_values is not None and X_val is not None:
        shap_df = pd.DataFrame(
            shap_values,
            columns=[FEATURE_DISPLAY_NAMES.get(col, col) for col in X_val.columns]
        )
        shap_df.to_csv(output_dir / "shap_values.csv", index=False)
        print(f"Saved SHAP values to: {output_dir / 'shap_values.csv'}")

    # Save model performance
    if model_performance is not None:
        with open(output_dir / "model_performance.json", 'w') as f:
            json.dump(model_performance, f, indent=2)
        print(f"Saved model performance to: {output_dir / 'model_performance.json'}")


# ===== Main Pipeline =====
def run_full_analysis(
    experiments_dir: str,
    output_dir: str = "analysis_outputs",
    n_pfi_repeats: int = 30
):
    """Run complete analysis pipeline: data prep → training → PFI → SHAP → plots.

    Args:
        experiments_dir: Path to simulation experiments directory
        output_dir: Directory to save all outputs
        n_pfi_repeats: Number of PFI repeats (default: 30)

    Example:
        >>> run_full_analysis("../Simulation/experiments")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Feature Analysis Pipeline - Replicating Figure 4")
    print("=" * 70)

    # Step 1: Prepare dataset
    print("\n[1/6] Preparing ML dataset...")
    df = prepare_ml_dataset(
        experiments_dir,
        output_path=output_dir / "aggregated_experiments.csv"
    )

    if len(df) < 10:
        print(f"\nWarning: Only {len(df)} experiments found. Need ~100+ for reliable results.")
        print("Consider running more simulation experiments with varied parameters.")
        return

    # Step 2: Train model
    print("\n[2/6] Training Elastic Net model...")
    model, X_train, X_val, y_val, feature_cols, rmse = train_enet_model(df)

    model_performance = {
        'rmse': float(rmse),
        'r2': float(r2_score(y_val, model.predict(X_val))),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_features': len(feature_cols)
    }

    # Step 3: Calculate PFI
    print(f"\n[3/6] Calculating Permutation Feature Importance...")
    df_pfi = calculate_pfi(
        model, X_val, y_val, rmse, feature_cols, n_repeats=n_pfi_repeats
    )

    # Step 4: Plot PFI (Figure 4A)
    print("\n[4/6] Generating PFI plot (Figure 4A)...")
    plot_pfi(df_pfi, save_path=output_dir / "figure_4a_pfi.png")

    # Step 5: Calculate SHAP
    if SHAP_AVAILABLE:
        print("\n[5/6] Calculating SHAP values...")
        try:
            shap_values, X_val_shap = calculate_shap(model, X_train, X_val, feature_cols)

            # Step 6: Plot SHAP (Figure 4B)
            print("\n[6/6] Generating SHAP plot (Figure 4B)...")
            plot_shap(shap_values, X_val_shap, save_path=output_dir / "figure_4b_shap.png")

        except Exception as e:
            print(f"\nSHAP calculation failed: {e}")
            shap_values = None
            X_val_shap = None
    else:
        print("\n[5/6] SHAP unavailable (not installed)")
        print("[6/6] Skipping SHAP plot")
        shap_values = None
        X_val_shap = None

    # Save all results
    print("\nSaving results...")
    save_analysis_results(
        output_dir,
        df_pfi,
        shap_values,
        X_val_shap if shap_values is not None else None,
        model_performance
    )

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 70)

    # Print top features
    print("\nTop 5 Most Important Features (PFI):")
    for idx, row in df_pfi.head(5).iterrows():
        print(f"  {idx+1}. {row['display_name']}: {row['mean_importance']:.3f} "
              f"± {row['std_importance']:.3f}")


# ===== Testing / Demo =====
if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("Feature Analysis Module - Testing")
    print("=" * 70)

    if len(sys.argv) > 1:
        experiments_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "analysis_outputs"

        print(f"\nExperiments directory: {experiments_dir}")
        print(f"Output directory: {output_dir}")

        try:
            run_full_analysis(experiments_dir, output_dir)
        except Exception as e:
            print(f"\nError running analysis: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("\nUsage:")
        print("  python feature.py <experiments_directory> [output_directory]")
        print("\nExample:")
        print("  python feature.py ../Simulation/experiments analysis_outputs")
        print("\nThis will:")
        print("  1. Aggregate all experiment results")
        print("  2. Train Elastic Net model with pairwise interactions")
        print("  3. Calculate Permutation Feature Importance")
        print("  4. Generate Figure 4A (PFI plot)")
        print("  5. Calculate SHAP values")
        print("  6. Generate Figure 4B (SHAP plot)")
        print("\nNote: Requires ~100+ experiment pairs for reliable results")
