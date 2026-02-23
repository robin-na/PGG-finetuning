# Benchmark Subsets

This folder contains small, tightly filtered subsets of the learning dataset used for strict OOD generalization and controlled simulation checks.

## Build the 10-game training-regime subset

```bash
python benchmark/build_benchmark_subset.py
```

Outputs:
- `benchmark/df_analysis_learn_noChat_lowPlayers_lowRounds_showNRounds_noReward_noId.csv`
- `benchmark/summary_gpt51_learn_noChat_lowPlayers_lowRounds_showNRounds_noReward_noId.jsonl`

