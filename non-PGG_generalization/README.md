# non-PGG_generalization

This folder stores non-PGG generalization assets, benchmark specs, and evaluation code for Twin-based transfer experiments.

## Main folders

- `data/`
  - Source-specific local assets and scratch space.
  - `PGG/`: consolidated PGG demographics and oracle archetype JSONL files.
  - `Twin-2k-500/`: local Twin-related assets when needed.
- `task_grounding/`
  - Benchmark definitions and Twin question inventories.
  - Start with:
    - `TWIN_TASK_GROUNDING.md`
    - `PGG_TRANSFER_BENCHMARK.md`
- `pgg_transfer_eval/`
  - Batch builders, evaluators, and comparison baselines for the Twin social-game benchmark.
  - Includes random, human-consistency, kNN allowed-input, and trait-heuristic comparison baselines.
  - See:
    - `pgg_transfer_eval/README.md`
- `legacy_demographicsOnly/`
  - Archived demographics-only PGG-to-Twin prototype. This is preserved for reference and is not the active benchmark.

## PGG profile extraction

The new transfer-oriented PGG profile/card extraction pipeline lives under:

- `Persona/transfer_profiles/README.md`

The request builder is:

```bash
python Persona/misc/build_transfer_profile_requests.py
```

It now builds a combined learn+validation-wave raw profile bank before creating the LLM batch requests.

## Regenerate consolidated PGG assets

From repo root:

```bash
python non-PGG_generalization/build_consolidated_data.py
```
