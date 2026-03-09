# LLM-Only Archetype Generation Pipeline

This folder implements an **LLM-only** pipeline for:

1. Building a global archetype cluster codebook from raw oracle personas.
2. Soft-assigning each oracle archetype row to clusters.
3. Aggregating per-game cluster distributions.

The final artifact is a game-level distribution over global clusters, usable as training targets for `CONFIG_* -> archetype distribution` mapping.

## Design choices

- Clustering is done on **full persona text** (all headers together), not per-tag independent clustering.
- Assignments are **soft** by default (probability distributions).
- No embedding clustering is used in this pipeline.
- OpenAI Batch JSONL is used for LLM stages.

## Files

- `io_utils.py`
  - Shared utilities for JSONL I/O, key normalization, content extraction from batch outputs, and robust JSON parsing from LLM responses.

- `prompt_templates.py`
  - Prompt builders for:
    - map stage (local clustering + row assignments + game distributions)
    - reduce stage (merge local clusters into global clusters)

- `build_map_batch_input.py`
  - Builds map-stage batch input JSONL from:
    - `Persona/archetype_oracle_gpt51_learn.jsonl`
    - `data/processed_data/df_analysis_learn.csv`
  - Performs stratified sharding, shared anchor selection, and token-budget-aware packing.
  - Emits OpenAI batch requests (`/v1/chat/completions`) with default model `gpt-5.1`.

- `parse_map_batch_output.py`
  - Parses map-stage batch output JSONL.
  - Normalizes local clusters, soft assignments, and model-reported game distributions.
  - Validates assignment coverage against request manifest.

- `build_reduce_batch_input.py`
  - Builds reduce-stage batch input from parsed local clusters/assignments.
  - Produces one consolidation request that asks LLM to create global clusters and local->global mapping.

- `parse_reduce_batch_output.py`
  - Parses reduce-stage batch output.
  - Writes:
    - `global_clusters.jsonl`
    - `local_to_global.jsonl`
  - Includes optional fallback mapping for unmapped local clusters.

- `finalize_game_distributions.py`
  - Applies local->global mapping to soft row assignments.
  - Aggregates to per-game global cluster distributions.
  - Writes wide CSV table for downstream modeling.

## End-to-end run

### 1) Build map-stage batch input

```bash
python Persona/archetype_generation/build_map_batch_input.py \
  --archetype-jsonl Persona/archetype_oracle_gpt51_learn.jsonl \
  --config-csv data/processed_data/df_analysis_learn.csv \
  --output-dir Persona/archetype_generation/out/map \
  --model gpt-5.1 \
  --n-shards 5 \
  --target-prompt-tokens 110000
```

Outputs:

- `Persona/archetype_generation/out/map/map_batch_requests.jsonl`
- `Persona/archetype_generation/out/map/map_request_manifest.jsonl`
- `Persona/archetype_generation/out/map/map_row_table.jsonl`
- `Persona/archetype_generation/out/map/map_build_summary.json`

Upload `map_batch_requests.jsonl` to OpenAI Batch and run it.

### 2) Parse map-stage batch output

```bash
python Persona/archetype_generation/parse_map_batch_output.py \
  --batch-output-jsonl <OPENAI_BATCH_OUTPUT_JSONL> \
  --request-manifest-jsonl Persona/archetype_generation/out/map/map_request_manifest.jsonl \
  --output-dir Persona/archetype_generation/out/map_parsed
```

Outputs:

- `map_clusters.jsonl`
- `map_assignments.jsonl`
- `map_game_distributions_llm.jsonl`
- `map_parsed_requests.jsonl`
- `map_parse_errors.jsonl`
- `map_parse_summary.json`

### 3) Build reduce-stage batch input

```bash
python Persona/archetype_generation/build_reduce_batch_input.py \
  --map-clusters-jsonl Persona/archetype_generation/out/map_parsed/map_clusters.jsonl \
  --map-assignments-jsonl Persona/archetype_generation/out/map_parsed/map_assignments.jsonl \
  --map-row-table-jsonl Persona/archetype_generation/out/map/map_row_table.jsonl \
  --output-dir Persona/archetype_generation/out/reduce \
  --model gpt-5.1
```

Outputs:

- `reduce_batch_requests.jsonl`
- `reduce_request_manifest.json`
- `reduce_local_cluster_cards.jsonl`

Upload `reduce_batch_requests.jsonl` to OpenAI Batch and run it.

### 4) Parse reduce-stage batch output

```bash
python Persona/archetype_generation/parse_reduce_batch_output.py \
  --batch-output-jsonl <OPENAI_REDUCE_BATCH_OUTPUT_JSONL> \
  --custom-id reduce_global_001 \
  --local-cluster-cards-jsonl Persona/archetype_generation/out/reduce/reduce_local_cluster_cards.jsonl \
  --output-dir Persona/archetype_generation/out/reduce_parsed
```

Outputs:

- `global_clusters.jsonl`
- `local_to_global.jsonl`
- `redundant_pairs.jsonl`
- `reduce_parse_summary.json`

### 5) Finalize per-game global distributions

```bash
python Persona/archetype_generation/finalize_game_distributions.py \
  --map-row-table-jsonl Persona/archetype_generation/out/map/map_row_table.jsonl \
  --map-assignments-jsonl Persona/archetype_generation/out/map_parsed/map_assignments.jsonl \
  --local-to-global-jsonl Persona/archetype_generation/out/reduce_parsed/local_to_global.jsonl \
  --scope target \
  --output-dir Persona/archetype_generation/out/final
```

Outputs:

- `row_global_assignments.jsonl`
- `game_cluster_distributions.jsonl`
- `global_cluster_prevalence.jsonl`
- `game_cluster_distribution_table.csv`
- `finalize_summary.json`

`game_cluster_distributions.jsonl` is the main artifact for your downstream `CONFIG_* -> distribution` modeling.

## Map-stage request schema (LLM output)

Top-level JSON object expected from each map request:

- `request_id`
- `clusters`: list of local cluster definitions (`local_cluster_id`, `name`, `description`, `representative_persona`, `non_redundant_signal`)
- `assignments`:
  - `target`: list of row assignments with `cluster_probs`
  - `anchors`: same structure for anchor rows
- `game_distributions`: list of per-game local-cluster distributions for target rows

## Reduce-stage request schema (LLM output)

Top-level JSON object expected:

- `reduce_id`
- `global_clusters`: list of global cluster definitions + merged local cluster keys
- `local_to_global`: explicit mapping rows
- `redundant_pairs`: optional audit trail of near-duplicate local clusters

## Notes on scale and token limits

- The learn oracle file is very large; this pipeline avoids a single mega prompt.
- `build_map_batch_input.py` uses:
  - stratified sharding
  - shared anchors
  - token-budget-aware packing (`--target-prompt-tokens`)
- If requests are too large/small, tune:
  - `--target-prompt-tokens`
  - `--anchor-token-budget`
  - `--max-anchors`
  - `--n-shards`

## Defaults

- Default model: `gpt-5.1`
- Batch endpoint in generated JSONL: `/v1/chat/completions`
- Output format requested from model: JSON object (`response_format: json_object`)

## Suggested next step after this pipeline

Use `game_cluster_distribution_table.csv` as supervised targets to train a calibrated predictor from `CONFIG_*` to global-cluster distribution for unseen game designs.
