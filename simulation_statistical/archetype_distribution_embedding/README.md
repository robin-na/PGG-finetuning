# Archetype Distribution Embedding Pipeline

This package implements the embedding-only archetype distribution pipeline inside `simulation_statistical`.
It builds player-in-game tables from archetype text plus CONFIG environments, cleans the tagged archetype text,
exports JSONL for external embeddings, fits a learn-wave GMM soft clustering model, aggregates those weights to the
game level, and fits an environment-to-distribution model over the required `CONFIG_*` columns.

## Inputs

Default learn-wave inputs:

- `Persona/archetype_oracle_gpt51_learn.jsonl`
- `benchmark_statistical/data/processed_data/df_analysis_learn.csv`
- `benchmark_statistical/data/raw_data/learning_wave/player-rounds.csv`

Default validation-wave inputs:

- `Persona/archetype_oracle_gpt51_val.jsonl`
- `data/processed_data/df_analysis_val.csv`
- `data/raw_data/validation_wave/player-rounds.csv`

The validation defaults intentionally use the root `data/` tables instead of `benchmark_statistical/data/`
because the benchmark-statistical validation subset omits 53 games / 460 archetype rows that are present in the
archetype JSONL. Paths are configurable on the CLI if you want a different subset.

## Join Assumptions

- Archetype JSONL `experiment` is treated as `game_id`.
- Archetype JSONL `participant` is treated as `player_id`.
- The canonical row unit is player-in-game, created by joining archetypes to distinct `(gameId, playerId)` pairs
  from player-round data, then attaching one config row per `gameId`.
- The pipeline validates uniqueness and fails loudly if unmatched joins exceed a small threshold.

## Workflow

1. Prepare player-game tables and export embedding inputs:

```bash
python simulation_statistical/archetype_distribution_embedding/run_pipeline.py prepare-inputs
```

2. Run embeddings manually after exporting your API key:

```bash
export OPENAI_API_KEY=YOUR_KEY_HERE
python simulation_statistical/archetype_distribution_embedding/features/embed_openai.py --input simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/embedding_input_learn.jsonl --output simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/embedding_output_learn.jsonl --model text-embedding-3-large --batch-size 128 --api-key-env OPENAI_API_KEY
python simulation_statistical/archetype_distribution_embedding/features/embed_openai.py --input simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/embedding_input_val.jsonl --output simulation_statistical/archetype_distribution_embedding/artifacts/intermediate/embedding_output_val.jsonl --model text-embedding-3-large --batch-size 128 --api-key-env OPENAI_API_KEY
```

3. Resume the post-embedding stages:

```bash
python simulation_statistical/archetype_distribution_embedding/run_pipeline.py all-post-embed
```

You can also run just:

- `fit-clusters`
- `fit-env-model`
- `post-embed`

## Artifacts

Intermediate outputs are written under:

- `simulation_statistical/archetype_distribution_embedding/artifacts/intermediate`

Models are written under:

- `simulation_statistical/archetype_distribution_embedding/artifacts/models`

Final outputs are written under:

- `simulation_statistical/archetype_distribution_embedding/artifacts/outputs`

Key outputs include:

- merged and cleaned player-game tables for learn and val
- embedding input JSONL and embedding output join matrices
- fitted PCA transform and GMM soft cluster model
- player-level and game-level cluster distributions
- fitted Dirichlet environment model
- predicted game-level cluster distributions
- clustering and environment-model evaluation summaries
