# Analysis Module for PGG LLM Agent Simulation

This module provides comprehensive analysis tools to calculate key metrics and replicate Figure 4 from the paper using machine learning model interpretation.

## Overview

The Analysis module consists of two main components:

1. **contribution.py**: Calculate three core metrics
   - Average Contribution
   - Normalized Efficiency
   - Punishment Effect

2. **feature.py**: Machine learning model interpretation
   - Elastic Net regression with pairwise interactions
   - Permutation Feature Importance (Figure 4A)
   - SHAP values (Figure 4B)

## Installation

### Dependencies

```bash
# Core dependencies
pip install pandas numpy scikit-learn matplotlib

# Optional (for SHAP analysis)
pip install shap
```

## Usage

### 1. Calculating Metrics (contribution.py)

#### Analyze a Single Experiment

```bash
cd Analysis
python contribution.py ../Simulation/experiments/baseline
```

Output:
```
Results:
  Average Contribution: 62.50%
  Normalized Efficiency: 0.785
  Endowment: 20
```

#### Programmatic Usage

```python
from contribution import (
    calculate_metrics_from_experiment,
    calculate_punishment_effect
)

# Single experiment
metrics = calculate_metrics_from_experiment("../Simulation/experiments/baseline")
print(f"Efficiency: {metrics['normalized_efficiency']:.3f}")

# Compare treatment vs control
effect = calculate_punishment_effect(
    "../Simulation/experiments/baseline_treatment",
    "../Simulation/experiments/baseline_control"
)
print(f"Punishment effect: {effect['punishment_effect_efficiency']:+.3f}")
```

### 2. ML Model Interpretation (feature.py)

#### Run Full Analysis Pipeline

```bash
cd Analysis
python feature.py ../Simulation/experiments analysis_outputs
```

This will:
1. Aggregate all experiment results into a training dataset
2. Train Elastic Net model with pairwise interactions
3. Calculate Permutation Feature Importance (30 repeats)
4. Generate Figure 4A (PFI bar chart)
5. Calculate SHAP values
6. Generate Figure 4B (SHAP summary plot)

#### Programmatic Usage

```python
from feature import run_full_analysis

# Run complete analysis
run_full_analysis(
    experiments_dir="../Simulation/experiments",
    output_dir="analysis_outputs",
    n_pfi_repeats=30
)
```

#### Step-by-Step Usage

```python
from feature import (
    prepare_ml_dataset,
    train_enet_model,
    calculate_pfi,
    plot_pfi,
    calculate_shap,
    plot_shap
)

# 1. Prepare dataset
df = prepare_ml_dataset(
    "../Simulation/experiments",
    output_path="aggregated_experiments.csv"
)

# 2. Train model
model, X_train, X_val, y_val, features, rmse = train_enet_model(df)

# 3. Calculate and plot PFI
df_pfi = calculate_pfi(model, X_val, y_val, rmse, features, n_repeats=30)
plot_pfi(df_pfi, save_path="figure_4a_pfi.png")

# 4. Calculate and plot SHAP
shap_values, X_val_shap = calculate_shap(model, X_train, X_val, features)
plot_shap(shap_values, X_val_shap, save_path="figure_4b_shap.png")
```

## Output Files

### From contribution.py

**metrics.json** (per experiment):
```json
{
  "average_contribution": 0.625,
  "normalized_efficiency": 0.785,
  "endowment": 20
}
```

### From feature.py

```
analysis_outputs/
├── aggregated_experiments.csv     # ML training dataset
├── pfi_results.csv                # Feature importance rankings
├── figure_4a_pfi.png             # PFI bar chart
├── shap_values.csv               # Individual SHAP values
├── figure_4b_shap.png            # SHAP summary plot
└── model_performance.json        # RMSE, R², etc.
```

**model_performance.json**:
```json
{
  "rmse": 0.0842,
  "r2": 0.7634,
  "n_train": 272,
  "n_val": 48,
  "n_features": 15
}
```

**pfi_results.csv**:
```csv
feature,display_name,mean_importance,std_importance,ci_lower,ci_upper
efficiency_control,Baseline Efficiency,0.3245,0.0123,0.3002,0.3488
mpcr,MPCR,0.1834,0.0089,0.1659,0.2009
communication,Communication,0.1623,0.0104,0.1419,0.1827
...
```

## Key Metrics Explained

### 1. Average Contribution

**Formula**: `mean(contribution / endowment)` across all players and rounds

**Interpretation**:
- 0.0 = No cooperation (all defect)
- 1.0 = Full cooperation (contribute maximum)
- Typical range: 0.3 - 0.7 for human experiments

### 2. Normalized Efficiency

**Formula**: `(E_actual - E_defect) / (E_cooperate - E_defect)`

Where:
- `E_actual` = Sum of all payoffs across all rounds
- `E_defect` = Baseline if no one contributes (Nash equilibrium)
- `E_cooperate` = Baseline if everyone contributes max (Pareto optimal)

**Interpretation**:
- 1.0 = Perfect cooperation (Pareto optimal)
- 0.0 = Nash equilibrium (all defect)
- < 0.0 = Worse than defection (excessive punishment costs)
- > 1.0 = Rewards generated surplus

**Critical**: Must be calculated round-by-round to handle player dropouts properly. Cannot be computed accurately post-hoc from final data alone.

### 3. Punishment Effect

**Formula**: `efficiency_treatment - efficiency_control`

**Interpretation**:
- Positive = Punishment improves efficiency
- Negative = Punishment harms efficiency
- Close to zero = No effect

## Data Requirements

### For Metrics Calculation

- Minimum: 1 complete experiment with game_log.csv and config.json
- For punishment effect: 1 treatment + 1 control experiment (paired)

### For ML Model (Figure 4)

**Minimum Dataset**:
- ~100 experiment pairs (treatment/control) for training
- ~20 experiment pairs for validation

**Recommended Dataset** (matches paper scale):
- 320 experiments for training
- 40 experiments for validation
- Full coverage of 14-parameter design space

### Generating Experiment Data

Use the Simulation module to run experiments with varied parameters:

```python
from Simulation.config import PGGConfig
from Simulation.main import run_experiment

# Example: Vary MPCR and framing
mpcr_values = [0.2, 0.4, 0.6]
framing_types = ["opt_in", "opt_out"]

for mpcr in mpcr_values:
    for framing in framing_types:
        # Treatment (with punishment)
        config_t = PGGConfig(
            group_size=4,
            game_length=10,
            mpcr=mpcr,
            contribution_framing=framing,
            punishment_enabled=True
        )
        run_experiment(f"mpcr{mpcr}_{framing}_treatment", config_t)

        # Control (without punishment)
        config_c = PGGConfig(
            group_size=4,
            game_length=10,
            mpcr=mpcr,
            contribution_framing=framing,
            punishment_enabled=False
        )
        run_experiment(f"mpcr{mpcr}_{framing}_control", config_c)
```

## Architecture Details

### EfficiencyCalculator Integration

The `EfficiencyCalculator` is integrated into the `PGGEnvironment` class in `Simulation/environment.py`:

```python
class PGGEnvironment:
    def __init__(self, config: PGGConfig):
        self.efficiency_calculator = EfficiencyCalculator(config)

    def add_round_to_history(self, round_state: RoundState):
        self.efficiency_calculator.update_round(round_state)

    def get_efficiency(self) -> float:
        return self.efficiency_calculator.get_efficiency()
```

This ensures efficiency is calculated correctly during simulation.

### Elastic Net Model Pipeline

```
Input (15 features)
    ↓
StandardScaler (normalize features)
    ↓
PolynomialFeatures (degree=2, interaction_only=True)
    → Creates ~120 features (15 main + 105 interactions)
    ↓
ElasticNetCV (L1 + L2 regularization, 5-fold CV)
    ↓
Output (efficiency_treatment prediction)
```

### SHAP Value Aggregation

Since PolynomialFeatures creates interaction terms (e.g., "x0 x1"), SHAP values must be aggregated back to original features:

**Strategy**: For interaction term "x_i x_j", split SHAP value equally:
- 50% attributed to feature i
- 50% attributed to feature j

This is a heuristic approach to maintain interpretability at the original feature level.

## Validation

### Expected Results

**Metric Ranges** (based on typical PGG experiments):
- Average Contribution: 0.3 - 0.7
- Normalized Efficiency: 0.4 - 0.9
- Punishment Effect: -0.1 to +0.3

**Model Performance**:
- RMSE: < 0.15 (per paper)
- R²: > 0.6

**Top Features** (from paper):
1. Baseline Efficiency (efficiency_control)
2. MPCR
3. Communication
4. Punishment Impact
5. Group Size

### Testing with Sample Data

```bash
# Test with existing experiments
cd Analysis
python contribution.py ../Simulation/experiments/baseline
python feature.py ../Simulation/experiments analysis_outputs
```

## Troubleshooting

### Issue: "Module not found: config"

**Cause**: Python path not set correctly

**Solution**: The modules add the Simulation directory to sys.path automatically. If this fails, set PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:../Simulation"
```

### Issue: "Not enough experiments found"

**Cause**: Need more experiment pairs for ML training

**Solution**: Run more simulation experiments with varied parameters. Aim for 100+ experiment pairs.

### Issue: "SHAP calculation failed"

**Cause**: SHAP not installed or compatibility issue

**Solution**:
```bash
pip install shap
# If issues persist, the analysis continues without SHAP (PFI still works)
```

### Issue: "Negative efficiency values"

**Cause**: Excessive punishment costs exceed cooperation gains

**Solution**: This is valid! Negative efficiency means the game outcome is worse than complete defection due to punishment overhead.

### Issue: "Efficiency > 1.0"

**Cause**: Rewards create surplus beyond the cooperation baseline

**Solution**: This is also valid! Rewards can increase total payoffs beyond what's possible through contributions alone.

## References

- **Baseline GRU model**: `../Simulation/baseline_gru_set.py`
- **Prompt generation**: `../alsobay2025publicGoodsGame/generate_prompts.py`
- **Experimental configurations**: `../data/exp_config_files/validation.yaml`

## Citation

If you use this analysis module, please cite the paper:

```
[Paper citation to be added]
```

## Contact

For questions or issues, please open an issue on GitHub.
