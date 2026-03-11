# Raw Behavior Cluster Ablation

This folder tests whether the useful macro signal in the current archetype-cluster pipeline can be recovered without using LLM-generated archetype summaries at all.

The idea is:

1. build a player-in-game feature vector directly from observed raw behavior
2. cluster those behavioral vectors with a GMM
3. aggregate player weights to game-level cluster mixtures
4. fit the same `CONFIG -> mixture` Dirichlet model used in the text-cluster pipeline
5. evaluate the same direct benchmark-level regressions used for the text-cluster direct baselines

Current runner:

```bash
python simulation_statistical/raw_behavior_cluster/run_raw_behavior_cluster_ablation.py
```

Outputs are written under:

- `simulation_statistical/raw_behavior_cluster/artifacts/outputs`

Main outputs:

- `learn_behavior_features.parquet`
- `val_behavior_features.parquet`
- `raw_behavior_cluster_diagnostics.csv`
- `raw_behavior_cluster_direct_eval.csv`
- `raw_behavior_cluster_report.md`

The feature vector is intentionally simple and fully observable:

- contribution level, zero/full rates, volatility, endgame delta
- punishment/reward giving and receiving rates and units
- round payoff rate
- chat activity, word count, and phase-specific message rates when chat logs are available
