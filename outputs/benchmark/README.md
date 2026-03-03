# Benchmark Outputs

This directory stores generated artifacts from benchmark pipeline runs.

## Structure

- `runs/`
  - Per-split generated artifacts.
  - Path pattern: `runs/<split-relative-path>/...`
  - Split path mapping:
    - `benchmark/data` -> `benchmark_filtered`
    - `benchmark/data_ood_splits/<factor>/<direction>` -> `benchmark_ood/<factor>/<direction>`
  - Example:
    - `runs/benchmark_filtered/archetype_retrieval/model_runs/...`
    - `runs/benchmark_filtered/micro_behavior_eval/oracle_archetype/2602270203/...`
    - `runs/benchmark_ood/all_or_nothing/false_to_true/archetype_retrieval/model_runs/...`
- `logs/`
  - Batch logs and run summaries
- `cache/`
  - Reusable cached helper outputs (for example merged archetype pool JSONL)

## Notes

- This folder is for mutable run outputs.
- Do not place source benchmark datasets here.
