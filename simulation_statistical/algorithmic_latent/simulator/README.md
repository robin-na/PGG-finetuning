# Simulator Runtime

This folder contains the first runtime simulator for the algorithmic-latent redesign.

Current runtime:

- [runtime.py](/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/simulation_statistical/algorithmic_latent/simulator/runtime.py)

Current responsibilities:

1. sample latent family assignments for players from the fitted environment-level family mixture
2. roll out contributions from family-specific multinomial contribution heads
3. roll out per-target `none / punish / reward` decisions from family-specific action heads
4. enforce information availability and within-round budget constraints
5. record structured round state for later decisions

Current simplifications:

- one sampled family per player at game start
- no posterior family updating during the game yet
- action units fixed to `1`
- no chat policy yet
- optional global punish/reward row-rate shrinkage loaded from `action_rate_calibration.json`

The runtime is explicitly two-stage within each round:

1. all players contribute
2. if applicable, players punish/reward after seeing round contributions

Timing and visibility rules:

- round-`T` contribution does not see round-`T` peer actions
- round-`T` punish/reward does see round-`T` peer contributions
- punish/reward identities only become visible in later rounds when the corresponding `CONFIG_showPunishmentId` / `CONFIG_showRewardId` flag allows it

Run through the existing simulator entrypoints with:

```bash
python simulation_statistical/micro/run_micro_simulation.py --strategy algorithmic_latent_family
python simulation_statistical/macro/run_macro_simulation.py --strategy algorithmic_latent_family
```
