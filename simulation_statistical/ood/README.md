# OOD Direct Benchmarks

This folder contains true out-of-distribution direct macro benchmarks for the statistical project.

Current runner:

- [run_direct_ood_benchmarks.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/ood/run_direct_ood_benchmarks.py)

What it does:

1. starts from `benchmark_statistical/data`
2. builds wave-anchored one-factor train/test splits from:
   - `df_analysis_learn.csv`
   - `df_analysis_val.csv`
   - learning/validation `player-rounds.csv`
3. evaluates direct benchmark-level regressors for:
   - `human_mean_contribution_rate`
   - `human_normalized_efficiency`
4. compares:
   - `ridge_config`
   - predicted text-cluster direct baselines
   - predicted raw-behavior-cluster direct baselines
   - oracle versions of the cluster-mixture baselines

Outputs:

- [wave_anchored_ood_report.md](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/ood/artifacts/outputs/wave_anchored_ood_report.md)
- [wave_anchored_ood_summary.csv](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/ood/artifacts/outputs/wave_anchored_ood_summary.csv)
- [wave_anchored_ood_split_summary.csv](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/ood/artifacts/outputs/wave_anchored_ood_split_summary.csv)
- [wave_anchored_ood_results.csv](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/ood/artifacts/outputs/wave_anchored_ood_results.csv)

Run:

```bash
python simulation_statistical/ood/run_direct_ood_benchmarks.py
```
