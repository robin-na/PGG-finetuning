# Experimental Workflow Guide

This guide explains how to run the 40-experiment design (game_length=1, punishment on/off) and generate the two analysis figures.

## Prerequisites

1. **Set your OpenAI API Key**:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

2. **Install required packages**:
```bash
pip install openai pandas numpy matplotlib seaborn scikit-learn scipy
```

## Step 1: Generate and Run Experiments

Run the experimental design script:

```bash
python run_experiments.py
```

**What this does**:
- Generates 40 experiment configurations (20 control + 20 treatment)
- Uses Latin Hypercube Sampling for parameter space coverage
- Fixed parameters:
  - `game_length = 1` (single round)
  - `endowment = 20` (fixed)
- Sampled parameters (14 total):
  - `group_size`: 2-20
  - `mpcr`: 0.06-0.7
  - `contribution_type`: variable vs all-or-nothing
  - `contribution_framing`: opt-in vs opt-out
  - `communication_enabled`: True/False
  - `peer_outcome_visibility`: True/False
  - `actor_anonymity`: True/False
  - `horizon_knowledge`: known vs unknown
  - `punishment_cost`: 1-4
  - `punishment_impact`: 1-4
  - `reward_enabled`: True/False
  - `reward_cost`: 1-4
  - `reward_impact`: 0.5-1.5

**Output**:
- Experiment directories: `Simulation/experiments/control_exp_{i}/` and `Simulation/experiments/treatment_exp_{i}/`
- Each contains:
  - `config.json` - Configuration used
  - `game_log.csv` - Round-by-round data
  - `prompts/` - LLM prompts and responses

**Estimated Time**:
- ~40 experiments × 1 round × 4 agents × 2-3 API calls = ~240-360 API calls
- At ~2 seconds per call = ~8-12 minutes total
- Cost: ~$2-4 USD (using gpt-4o)

## Step 2: Analyze Results and Generate Figures

After experiments complete, run the analysis script:

```bash
python analyze_results.py
```

**What this does**:
- Loads all experiment results from `Simulation/experiments/`
- Calculates three core metrics:
  1. **Average Contribution**: % of endowment contributed
  2. **Normalized Efficiency**: (Actual - Defect) / (Cooperate - Defect)
  3. **Punishment Effect**: Difference between treatment and control
- Generates two figures:

### Figure 1: Punishment Effect on Contribution and Efficiency
- **Panel A**: Boxplot of Average Contribution (with vs without punishment)
- **Panel B**: Boxplot of Normalized Efficiency (with vs without punishment)
- Shows individual data points + means
- Displays effect size and p-value

### Figure 2: Feature Importance Analysis
- Uses Elastic Net regression to predict `efficiency_treatment` from parameters
- Calculates Permutation Feature Importance (PFI) with 30 repeats
- Shows which design parameters most influence cooperation
- Includes confidence intervals (95% CI)

**Output**:
- `analysis_outputs/experiment_results.csv` - Aggregated metrics
- `analysis_outputs/figure_1_punishment_effect.png` - Punishment effect visualization
- `analysis_outputs/figure_2_feature_importance.png` - Feature importance analysis

## Interpreting Results

### Expected Patterns

**Figure 1 - Punishment Effect**:
- **Contribution**: Should be higher with punishment (if deterrence works)
- **Efficiency**: May be higher OR lower depending on punishment costs
  - Higher: Punishment deters free-riding, increases cooperation
  - Lower: Punishment costs exceed cooperation gains (wasteful)
  - Near zero: No effect or costs cancel gains

**Figure 2 - Feature Importance**:
- **Top features expected**:
  - `mpcr`: Higher MPCR = stronger incentive to cooperate
  - `communication`: Enables coordination
  - `group_size`: Larger groups = harder to coordinate
  - `efficiency_control`: Strong predictor of treatment efficiency
  - `punishment_cost/impact`: Determines punishment effectiveness

### Game Length = 1 Implications

With single-round games:
- **No reputation effects**: Agents can't build trust over time
- **No learning**: No opportunity to adapt to others' behavior
- **No retaliation**: Can't punish back in future rounds
- **Pure Nash equilibrium**: Rational choice is to free-ride (contribute 0)

This isolates the **immediate deterrence effect** of punishment threat.

## Troubleshooting

### Issue: "OPENAI_API_KEY not found"
**Fix**: Set the environment variable:
```bash
export OPENAI_API_KEY='sk-...'
```

### Issue: "No experiment results found"
**Fix**: Run `python run_experiments.py` first to generate data

### Issue: "Only N samples. Need ~20+ for reliable results"
**Fix**: This is expected with 20 pairs. The analysis will still run but warns about sample size.

### Issue: Rate limit errors
**Fix**: The script has exponential backoff retry logic. If persistent:
```python
# Edit llm_client.py to add delay between calls
import time
time.sleep(1)  # Add 1 second delay between API calls
```

### Issue: High API costs
**Fix**: Reduce the number of experiments:
```python
# Edit run_experiments.py, line 115
N_SAMPLES = 10  # Change from 20 to 10 (20 experiments total instead of 40)
```

## Next Steps

1. **Validate results**: Check if punishment increases cooperation in your LLM agents
2. **Compare to human data**: If available, compare LLM behavior to human baselines
3. **Extend to multi-round**: Change `game_length` to 5-10 to study dynamics
4. **Test different LLMs**: Try gpt-3.5-turbo, claude-3-opus, etc.
5. **Add persona variation**: Use different system prompts (altruistic, selfish, etc.)

## File Structure

```
PGG-finetuning/
├── run_experiments.py          # Generate and run 40 experiments
├── analyze_results.py          # Analyze results and generate figures
├── Simulation/
│   ├── main.py                 # Core game engine
│   ├── config.py              # Configuration dataclass
│   ├── environment.py         # Game mechanics
│   ├── agent.py               # LLM agent wrapper
│   ├── prompt_builder.py      # Context-sensitive prompts
│   ├── llm_client.py          # OpenAI API interface
│   └── experiments/           # [Generated by run_experiments.py]
│       ├── control_exp_0/
│       ├── treatment_exp_0/
│       └── ...
├── Analysis/
│   ├── contribution.py        # Metric calculations
│   └── feature.py             # ML model interpretation
└── analysis_outputs/          # [Generated by analyze_results.py]
    ├── experiment_results.csv
    ├── figure_1_punishment_effect.png
    └── figure_2_feature_importance.png
```

## References

- Paper methodology: Feature importance via Elastic Net + PFI
- Latin Hypercube Sampling: Efficient parameter space coverage
- Normalized Efficiency: Standard metric in experimental economics
