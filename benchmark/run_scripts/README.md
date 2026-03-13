# Benchmark Macro SLURM Scripts

These scripts run **macro simulation only** on the filtered benchmark dataset:

- dataset root: `benchmark/data`
- wave: `validation_wave`
- game subset: `reports/benchmark/manifests/benchmark__data__validation_wave__strict_1__seed_0.csv`
  - 40 games total (2 per validation config: one punishment, one no punishment)

## Scripts

- `slurm_macro_benchmark_filtered_no_archetype.sh`
- `slurm_macro_benchmark_filtered_no_archetype_vllm_2xa100.sh`
- `slurm_macro_benchmark_filtered_no_archetype_vllm_2xa100_autosweep.sh`
- `slurm_macro_benchmark_filtered_random_archetype.sh`
- `slurm_macro_benchmark_filtered_random_archetype_local_12b.sh`
- `slurm_macro_benchmark_filtered_oracle_archetype.sh`
- `slurm_macro_benchmark_filtered_retrieved_archetype.sh`
- `slurm_macro_benchmark_filtered_config_bank_archetype_local_12b.sh`

All scripts run via `benchmark/scripts/run_split_pipeline.py` with `--stages macro` and then refresh `run_index.json`.
They accept env overrides for `MANIFEST_CSV`, `PROVIDER`, `BASE_MODEL`, `MAX_PARALLEL_GAMES`, `VLLM_BASE_URL`, `VLLM_MODEL`, and `VLLM_MAX_CONCURRENCY`.

## Submit

```bash
sbatch benchmark/run_scripts/slurm_macro_benchmark_filtered_no_archetype.sh
sbatch benchmark/run_scripts/slurm_macro_benchmark_filtered_no_archetype_vllm_2xa100.sh
sbatch benchmark/run_scripts/slurm_macro_benchmark_filtered_no_archetype_vllm_2xa100_autosweep.sh
sbatch benchmark/run_scripts/slurm_macro_benchmark_filtered_random_archetype.sh
sbatch benchmark/run_scripts/slurm_macro_benchmark_filtered_random_archetype_local_12b.sh
sbatch benchmark/run_scripts/slurm_macro_benchmark_filtered_oracle_archetype.sh
sbatch benchmark/run_scripts/slurm_macro_benchmark_filtered_retrieved_archetype.sh
sbatch benchmark/run_scripts/slurm_macro_benchmark_filtered_config_bank_archetype_local_12b.sh
```

## Notes

- `random_archetype` samples from `Persona/archetype_oracle_gpt51_learn.jsonl`.
- `oracle_archetype` matches from `Persona/archetype_oracle_gpt51_val.jsonl`.
- `retrieved_archetype` auto-builds split synthetic ridge archetypes if missing.
- `config_bank_archetype` uses a fixed precomputed assignment JSONL, by default `Persona/archetype_sampling/outputs/game_assignment_manifests/config_bank_archetype/game_assignments_config_bank_archetype_seed0.jsonl`.
- For `PROVIDER=vllm`, point `VLLM_BASE_URL` at an OpenAI-compatible `vllm serve` endpoint and set `MAX_PARALLEL_GAMES` above `1` to overlap multiple games.
- For 2 GPUs, throughput is usually better if you shard the manifest and run one job per GPU rather than using tensor parallel for a model that already fits on one GPU.
- `slurm_macro_benchmark_filtered_no_archetype_vllm_2xa100.sh` automates the 2-GPU case by starting one vLLM server per GPU, sharding the manifest, and merging the shard runs back into one combined run directory.
- `slurm_macro_benchmark_filtered_no_archetype_vllm_2xa100_autosweep.sh` adds a small calibration sweep over `MAX_PARALLEL_GAMES_PER_SERVER` and `VLLM_MAX_CONCURRENCY`, then launches the full run with the fastest stable setting.

## Multi-GPU

Create two stable shards:

```bash
python benchmark/scripts/shard_game_manifest.py \
  --input-csv reports/benchmark/manifests/benchmark__data__validation_wave__strict_1__seed_0.csv \
  --output-dir reports/benchmark/manifests/shards \
  --num-shards 2
```

Then submit one job per shard, for example:

```bash
sbatch --export=ALL,PROVIDER=vllm,VLLM_BASE_URL=http://127.0.0.1:8000/v1,MAX_PARALLEL_GAMES=4,VLLM_MAX_CONCURRENCY=8,MANIFEST_CSV=reports/benchmark/manifests/shards/manifest_shard_00_of_02.csv \
  benchmark/run_scripts/slurm_macro_benchmark_filtered_no_archetype.sh

sbatch --export=ALL,PROVIDER=vllm,VLLM_BASE_URL=http://127.0.0.1:8001/v1,MAX_PARALLEL_GAMES=4,VLLM_MAX_CONCURRENCY=8,MANIFEST_CSV=reports/benchmark/manifests/shards/manifest_shard_01_of_02.csv \
  benchmark/run_scripts/slurm_macro_benchmark_filtered_no_archetype.sh
```
