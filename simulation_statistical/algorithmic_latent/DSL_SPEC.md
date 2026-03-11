# DSL Spec

`DSL` means `domain-specific language`: a small controlled language for expressing strategy rules.

In this project, the DSL is not arbitrary code. It is a constrained rule format for contribution, punishment, and reward behavior.

## Why Use a DSL

Without a DSL:

- LLM-generated strategies become too unconstrained
- many programs can fit the same short human trace
- fitting becomes unstable
- interpretation becomes inconsistent

With a restricted DSL:

- strategy families are comparable
- parameters can be fit statistically
- visibility constraints are explicit
- OOD transfer is more plausible

## Visibility Rules

Every DSL rule may only access structured variables that are visible to the player.

Allowed information sources:

- game `CONFIG`
- own prior contributions
- observed peer contributions
- punish/reward received by the player
- visible summaries if `CONFIG_showOtherSummaries=true`
- visible punisher identities only if `CONFIG_showPunishmentId=true`
- visible rewarder identities only if `CONFIG_showRewardId=true`
- remaining rounds only if `CONFIG_showNRounds=true`

Disallowed:

- hidden target identities when not shown
- hidden summaries
- future rounds
- latent information about other players not inferable from visible history

## State Schema

### Static `CONFIG` Features

- `player_count`
- `endowment`
- `multiplier`
- `mpcr`
- `all_or_nothing`
- `punishment_exists`
- `punishment_cost`
- `punishment_magnitude`
- `reward_exists`
- `reward_cost`
- `reward_magnitude`
- `show_other_summaries`
- `show_punishment_id`
- `show_reward_id`
- `show_n_rounds`
- `chat_enabled`

### Dynamic Round State

Contribution-stage variables:

- `round_index`
- `round_phase_visible`
- `rounds_remaining_visible`
- `own_prev_contribution_rate`
- `peer_prev_mean_contribution_rate`
- `peer_prev_std_contribution_rate`
- `punished_prev_any`
- `rewarded_prev_any`
- `punish_received_prev_units`
- `reward_received_prev_units`
- `prev_round_payoff`

Action-stage extras:

- `own_current_contribution_rate`
- `peer_current_mean_contribution_rate`
- `target_current_contribution_rate`
- `target_minus_peer_mean`
- `target_minus_expected_norm`
- `target_punished_me_before_visible`
- `target_rewarded_me_before_visible`

## Rule Types

The DSL should expose three rule types.

### 1. Contribution Rule

Output:

- distribution over legal contribution bins

First implementation:

- all-or-nothing games: `{0, endowment}`
- continuous games: `{0, 5, 10, 15, 20}`

Template:

```text
contrib_logit(bin) =
  alpha_bin
  + beta_peer * peer_prev_mean_contribution_rate
  + beta_peer_var * peer_prev_std_contribution_rate
  + beta_own * own_prev_contribution_rate
  + beta_punished * punished_prev_any
  + beta_rewarded * rewarded_prev_any
  + beta_payoff * prev_round_payoff_norm
  + beta_endgame * endgame_signal
  + beta_group * normalized_group_size
```

### 2. Punish Rule

Output:

- probability of punishing each visible target
- optional separate distribution over units later

First implementation:

- unit size fixed to `1`

Template:

```text
punish_score(target) =
  gamma0
  + gamma_dev * max(expected_norm - target_current_contribution_rate, 0)
  + gamma_ret * retaliated_by_target_before_visible
  + gamma_end * endgame_signal
  + gamma_vis * punishment_id_visible
```

### 3. Reward Rule

Output:

- probability of rewarding each visible target

First implementation:

- unit size fixed to `1`

Template:

```text
reward_score(target) =
  eta0
  + eta_dev * max(target_current_contribution_rate - expected_norm, 0)
  + eta_rec * rewarded_by_target_before_visible
  + eta_end * endgame_signal
  + eta_vis * reward_id_visible
```

## Family Representation

Each algorithm family is a restricted subspace of the DSL:

- same variable vocabulary
- only some coefficients active
- family-specific parameter priors

Example:

- `unconditional_cooperator`
  - high positive intercept on full contribution
  - no peer-response term
- `conditional_cooperator`
  - strong positive `beta_peer`
- `endgame_defector`
  - negative `beta_endgame` on high contributions late in the game

## Why Not Free-Form Programs

Free-form strategy synthesis is tempting but should be deferred.

Main risks:

- overfitting finite traces
- incoherent strategies across players
- no clean likelihood-based fitting
- weak identifiability

So the first implementation should use:

- a fixed DSL
- a small family library
- numeric parameters fit from data

and only later consider richer symbolic search if the constrained version is clearly too weak.
