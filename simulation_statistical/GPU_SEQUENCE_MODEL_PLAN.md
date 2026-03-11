# GPU-Ready Structured Sequence Model

This note describes the next rewrite of the statistical simulator so training and rollout can use GPU efficiently.

## Why the current code does not use GPU well

The current structured policy in `simulation_statistical/structured_sequence_policy.py` is mostly:

- Python loops over games, rounds, players, and peers
- pandas-based feature construction
- scikit-learn SGD models
- CPU-side per-round simulation logic

That means a large GPU does not help much. The expensive part is not matrix multiplication; it is Python orchestration and per-row feature building.

## GPU-friendly rewrite

The rewrite should move almost all hot-path computation into batched tensors.

### Inputs per game

For a game with `T` rounds and at most `N` players, build:

- `env_features`: shape `[E]`
- `contrib_history`: shape `[T, N]`
- `punish_history`: shape `[T, N, N]`
- `reward_history`: shape `[T, N, N]`
- `payoff_history`: shape `[T, N]`
- `alive_mask`: shape `[T, N]`
- `visibility_mask` tensors derived from:
  - `CONFIG_showNRounds`
  - `CONFIG_showOtherSummaries`
  - `CONFIG_showPunishmentId`
  - `CONFIG_showRewardId`

Each sample at round `t` should be built from prefixes `[:t]`, not from text.

### Model

Use a small PyTorch model with three components:

1. `EnvEncoder`
   - MLP over CONFIG features
   - output: `env_emb`

2. `PlayerStateEncoder`
   - one hidden state per player
   - inputs each round:
     - own previous contribution/payoff
     - visible peer contribution summary
     - visible punish/reward summaries
     - optional exact per-peer visible action tensors
   - implementation:
     - GRU or Transformer block
   - output: `player_state[t, i]`

3. Heads
   - Contribution head:
     - all-or-nothing: binary classifier
     - continuous: bounded regressor
   - Action head:
     - for each focal-target pair at round `t`, classify:
       - `none`
       - `punish`
       - `reward`
     - for now, units stay fixed at `1`

### Best practical first version

Use:

- `env_emb = MLP(CONFIG)`
- `player_state = GRU(player_inputs_t, player_state_{t-1})`
- `pair_logits = MLP([player_state_i, player_state_j, current_contrib_i, current_contrib_j, env_emb])`

This is much easier to train than a full Transformer and already GPU-friendly.

## Training

### Teacher-forced training

For learning-wave data:

- feed real histories up to `t-1`
- predict contribution at `t`
- feed real current-round contributions
- predict per-peer action label at `t`

Loss:

- all-or-nothing contribution: cross-entropy
- continuous contribution: MSE or Huber
- per-peer action label: cross-entropy with class weights

### Recommended improvements

After the basic model works:

1. separate contribution heads for:
   - `CONFIG_allOrNothing = 1`
   - `CONFIG_allOrNothing = 0`

2. scheduled sampling for rollout robustness

3. optional online latent update:
   - initialize from archetype/env prior
   - update player hidden state from observed actions instead of keeping cluster fixed forever

## Simulation

At inference time:

1. encode CONFIG once
2. initialize player hidden states
3. for each round:
   - predict all player contributions in one batch
   - predict all focal-target action logits in one batch
   - sample `none/punish/reward`
   - enforce budget constraints
   - update hidden states with realized round outcome

This can still keep game-by-game round order, but the expensive neural inference runs in GPU batches.

## Expected speedup

GPU helps if we actually move to this tensorized PyTorch version.

- Training: large speedup
- Batched micro simulation: moderate to large speedup
- Macro simulation: moderate speedup

The current CPU implementation would not benefit much from A100/H100/H200.

## What to implement next

Suggested new files:

- `simulation_statistical/gpu_sequence_model.py`
- `simulation_statistical/train_gpu_sequence_model.py`
- `simulation_statistical/run_gpu_micro_simulation.py`
- `simulation_statistical/run_gpu_macro_simulation.py`

Suggested artifact path:

- `simulation_statistical/archetype_distribution_embedding/artifacts/models/gpu_sequence_model.pt`

## Running on SSH today

Today, the current structured model is still CPU/scikit-learn based.

Use `simulation_statistical/scripts/run_structured_exact_sequence_pipeline.sh` for the current end-to-end pipeline on a remote machine.

That script runs:

- current structured-model training
- full micro validation simulation
- full macro validation simulation
- micro comparison analysis
- macro comparison analysis

It does not use GPU.

## Running on SSH after the GPU rewrite

Once the PyTorch version exists, the intended commands should look like:

```bash
python simulation_statistical/train_gpu_sequence_model.py \
  --device cuda \
  --output_model_path simulation_statistical/archetype_distribution_embedding/artifacts/models/gpu_sequence_model.pt

python simulation_statistical/run_gpu_micro_simulation.py \
  --model_path simulation_statistical/archetype_distribution_embedding/artifacts/models/gpu_sequence_model.pt \
  --run_id gpu_exact_sequence_micro_val

python simulation_statistical/run_gpu_macro_simulation.py \
  --model_path simulation_statistical/archetype_distribution_embedding/artifacts/models/gpu_sequence_model.pt \
  --run_id gpu_exact_sequence_macro_val
```

Those entry points do not exist yet; they are the next implementation target.
