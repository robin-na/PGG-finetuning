# Simulation Statistical

`simulation_statistical` is the numerical baseline package for PGG behavior. It parallels the existing micro and macro eval folders, but replaces the LLM decision policy with deterministic seeded sampling over the legal action space implied by each game's CONFIG environment.

Its benchmark data and artifact roots now live under `benchmark_statistical/`.

Current strategy set:
- `random_baseline`

Behavior of `random_baseline` in each round:
1. contribution amount:
   - `0` or `CONFIG_endowment` when `CONFIG_allOrNothing=true`
   - integer from `0` to `CONFIG_endowment` when `CONFIG_allOrNothing=false`
2. punishment:
   - if enabled, choose one random target with probability `0.10`
   - magnitude fixed at `1`
3. reward:
   - if enabled, choose one random target with probability `0.10`
   - magnitude fixed at `1`

## Layout

- `simulation_statistical/micro`
  - micro-style prediction vs actual rows
  - data defaults to `benchmark_statistical/data`
  - outputs default to `benchmark_statistical/micro/runs`
  - reports default to `benchmark_statistical/micro/reports`
  - report plots include contribution MAE and a punishment-only target F1 supplement
- `simulation_statistical/macro`
  - full-game numerical rollouts
  - data defaults to `benchmark_statistical/data`
  - validation evaluation uses `benchmark_statistical/data/processed_data/df_analysis_val_averaged.csv`
  - outputs default to `benchmark_statistical/macro/runs`
  - reports default to `benchmark_statistical/macro/reports`
  - report plots include RMSE by target and variance across players/games, with a learning-wave linear CONFIG baseline overlaid alongside the statistical simulation
- shared utilities live in:
  - `simulation_statistical/common.py`
  - `simulation_statistical/policy.py`

## Run

Micro:

```bash
python simulation_statistical/micro/run_micro_simulation.py
```

Macro:

```bash
python simulation_statistical/macro/run_macro_simulation.py
```

## Notes

- The current numerical baseline uses no chat strategy; `data.chat_message` is left blank.
- Validation runs simulate one representative game per `CONFIG_treatmentName`, with macro evaluation scored against the treatment-averaged benchmark table.
- The code is structured so later strategy variants can be added by extending `simulation_statistical/policy.py` and wiring a new policy selector into the runners.
