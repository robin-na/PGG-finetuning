# Benchmark Macro SLURM Scripts

These scripts run **macro simulation only** on the filtered benchmark dataset:

- dataset root: `benchmark/data`
- wave: `validation_wave`
- game subset: `reports/benchmark/manifests/benchmark__data__validation_wave__strict_1__seed_0.csv`
  - 40 games total (2 per validation config: one punishment, one no punishment)

## Scripts

- `slurm_macro_benchmark_filtered_no_archetype.sh`
- `slurm_macro_benchmark_filtered_random_archetype.sh`
- `slurm_macro_benchmark_filtered_oracle_archetype.sh`
- `slurm_macro_benchmark_filtered_retrieved_archetype.sh`

All scripts run via `benchmark/scripts/run_split_pipeline.py` with `--stages macro` and then refresh `run_index.json`.

## Submit

```bash
sbatch benchmark/run_scripts/slurm_macro_benchmark_filtered_no_archetype.sh
sbatch benchmark/run_scripts/slurm_macro_benchmark_filtered_random_archetype.sh
sbatch benchmark/run_scripts/slurm_macro_benchmark_filtered_oracle_archetype.sh
sbatch benchmark/run_scripts/slurm_macro_benchmark_filtered_retrieved_archetype.sh
```

## Notes

- `random_archetype` samples from `Persona/archetype_oracle_gpt51_learn.jsonl`.
- `oracle_archetype` matches from `Persona/archetype_oracle_gpt51_val.jsonl`.
- `retrieved_archetype` auto-builds split synthetic ridge archetypes if missing.
