## Local 12B four-variant scorecard

Runs compared:

- `no_archetype/10477689_r0_macro_no_archetype_local_12b`
- `oracle_archetype/10478932_r0_macro_oracle_archetype_local_12b`
- `config_bank_archetype/10491393_r0_macro_config_bank_archetype_local_12b`
- `random_archetype/10492196_r0_macro_random_archetype_local_12b_s0`

Note: the config-bank run is labeled `retrieved archetype` in the comparison outputs for plotting compatibility.

For fair model-to-model comparison, treatment and rollout diagnostics should be read on the shared 31 games.

### 1. Primary criterion: treatment-effect replication on shared 31 games

Target: directional match of normalized-efficiency deltas across 11 CONFIG factors.

| label | sign-match rate | mean abs delta error |
|---|---:|---:|
| random archetype | 0.636 | 0.060 |
| retrieved archetype (config bank) | 0.636 | 0.084 |
| oracle archetype | 0.545 | 0.110 |
| no archetype | 0.545 | 0.127 |
| linear CONFIG baseline | 0.727 | 0.057 |

Interpretation:

- No simulator beats the linear CONFIG baseline on treatment-effect replication.
- Among simulators, `random archetype` is best on this narrow treatment-direction criterion, with `config bank` close behind.

### 2. Full-rollout alignment on shared 31 games

Game-level parity:

- Contribution rate: `oracle` best (`RMSE 0.178`, `corr 0.697`)
- Punishment rate: `oracle` best (`RMSE 0.179`, `corr 0.675`)
- Normalized efficiency: `oracle` best overall among simulators (`RMSE 0.256`, `corr 0.503`)
- OLS CONFIG baseline on the same shared games:
  - contribution rate: `RMSE 0.190`, `corr 0.447`
  - normalized efficiency: `RMSE 0.248`, `corr 0.216`

Interpretation:

- `oracle archetype` is the strongest simulator on direct game-level rollout alignment.
- On efficiency RMSE alone, oracle is close to the CONFIG baseline, but the baseline still wins on treatment-effect sign replication.

### 3. Behavioral realism under rollout

Within-game player-distribution Wasserstein-1 means:

- Contribution rate: oracle `0.152` < config bank `0.192` < random `0.222` ~= no-archetype `0.225`
- Punishment rate: oracle `0.155` < config bank `0.194` < no-archetype `0.261` < random `0.291`
- Reward rate: oracle `0.365` < random `0.441` ~= config bank `0.441` < no-archetype `0.637`

Within-game player-heterogeneity mean absolute gap from human:

- oracle `0.035`
- config bank `0.053`
- no archetype `0.092`
- random `0.102`

Trajectory nuance:

- `random archetype` has low trajectory RMSE on contribution and efficiency levels, but negative trajectory correlation, indicating rough level matching with the wrong shape.
- `oracle archetype` has the cleanest positive trajectory correlation on round-normalized efficiency and the best overall structural realism.

## Bottom line

If "simulation quality" means the most behaviorally representative full rollout, `oracle archetype` is the best of the four runs.

If "simulation quality" means treatment-effect replication across conditions, `random archetype` and `config bank` are better than `oracle` on the shared 31 games, but neither beats the linear CONFIG baseline.

Recommended framing:

- headline metric: treatment-effect replication vs linear CONFIG baseline
- best simulator for behavioral realism: `oracle archetype`
- secondary simulator for treatment sensitivity: `config bank archetype`

## Key artifacts

- `directional_effects_shared31_manual_summary.csv`
- `directional_effects_shared31_manual.csv`
- `aggregate_efficiency_metrics.csv`
- `directional_sign_summary.csv`
- `../local12b_four_variant_rollout/game_level_alignment_summary.csv`
- `../local12b_four_variant_rollout/player_distribution_wasserstein_summary.csv`
- `../local12b_four_variant_rollout/player_heterogeneity_gap_manual.csv`
- `../local12b_four_variant_rollout/trajectory_curve_summary_manual.csv`
