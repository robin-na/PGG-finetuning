## Concordia-backed macro simulator

This adds a side-by-side backend for `Macro_simulation_eval` without changing the existing macro simulator or the other simulation pipelines.

The installed package exposes the prefab modules under:

```python
from concordia.prefabs.entity import rational, basic, basic_with_plan
```

The top-level `concordia.prefabs` package itself is effectively just a namespace package.

### What it does

- Reuses the existing macro game loading from `CONFIG_*` fields.
- Reuses the existing archetype assignment pipeline.
- Keeps the existing row schema and transcript tags so downstream analysis still works.
- Persists native Concordia `SimulationLog` artifacts for each game as JSON and HTML.
- Uses Concordia's native simulation stack instead of a manual repo-side stage loop:
  - player entities come from `concordia.prefabs.entity`
  - the simulation is assembled through `concordia.prefabs.simulation.generic.Simulation`
  - stage execution uses `concordia.environment.engines.simultaneous.Simultaneous`
  - a repo-local PGG game-master prefab handles the public-goods mechanics and output compatibility
- Replaces the per-call full transcript prompt loop with Concordia memory updates:
  - each agent gets seeded observations for rules, avatar, goal, and persona
  - each stage directly updates player memory through `observe(...)`
  - round events are stored as episodic observations instead of replaying the whole game history every time

### Entry point

Run the Concordia backend with:

```bash
python -m Macro_simulation_eval.run_macro_simulation_eval_concordia \
  --provider openai \
  --openai_model gpt-5-mini
```

### Key Concordia options

- `--concordia_agent_prefab rational|basic|basic_with_plan`
- `--concordia_embedder hash|openai`
- `--concordia_embedding_model text-embedding-3-small`
- `--concordia_hash_dim 384`
- `--concordia_goal "..."`

### Mapping from the current macro pipeline

- Chat stage:
  - one simultaneous engine step with a free-form action spec
  - resolved chat is observed by all players and written into `<CHAT_LOG>` transcript entries
- Contribution stage:
  - one simultaneous engine step with a Concordia `ActionSpec`
  - `CHOICE` for all-or-nothing games
  - `FLOAT` otherwise
- Punishment / reward stage:
  - one simultaneous engine step with a free-form JSON action spec
  - resolved actions are parsed back into the same sparse JSON columns as the current macro backend
- Reasoning:
  - when `--include_reasoning true`, the backend stores Concordia agent logs if available via `get_last_log()`

### Current limitations

- Native Concordia `SimulationLog` files are written under `<run_dir>/concordia_logs/` as one JSON file and one HTML file per game.
- Concordia's `simultaneous` engine already runs player actions concurrently within a stage.
- `--resume_from_run` is not supported yet because the repo-local PGG game master is not an `EntityWithComponents`, so Concordia's native checkpoint loader cannot restore its internal round/stage state.
- `--max_parallel_games` must stay at `1`. Concordia gives us per-stage parallel player execution, but this backend still runs separate games sequentially at the repo orchestration layer.
- The PGG logic is implemented by a repo-local custom game-master prefab rather than one of Concordia's built-in matrix-game prefabs. That is intentional because the macro pipeline needs multi-round public-goods mechanics, optional chat, and optional punishment/reward stages.
