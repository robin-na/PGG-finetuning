# Family Library

This file defines the first-pass algorithm families for the redesign.

The aim is not to enumerate every plausible human strategy. The aim is to start with a compact set that:

- maps onto known PGG behavior patterns
- is easy to distinguish statistically
- is useful for institutional interpretation

## Family Set

### 1. `unconditional_cooperator`

Interpretation:

- contributes highly regardless of peer history
- may reward others lightly
- low punishment tendency

Active parameters:

- high positive contribution intercept
- weak peer-response terms
- weak endgame term
- weak sanction terms

### 2. `unconditional_defector`

Interpretation:

- contributes little regardless of peer behavior
- rarely punishes or rewards

Active parameters:

- low contribution intercept
- weak peer-response terms
- weak sanction terms

### 3. `conditional_cooperator`

Interpretation:

- raises contribution when peers contributed more last round
- lowers contribution when peers defect

Active parameters:

- moderate baseline contribution
- strong positive peer-mean coefficient
- optional response to punish/reward received

### 4. `generous_conditional_cooperator`

Interpretation:

- similar to conditional cooperation, but with a higher baseline and more forgiveness

Active parameters:

- higher baseline than `conditional_cooperator`
- positive peer-response
- weaker negative response to bad peer behavior
- more likely to reward high contributors

### 5. `endgame_defector`

Interpretation:

- contributes earlier but strategically reduces contribution late when the horizon is salient

Active parameters:

- moderate early baseline
- negative endgame coefficient
- effect only meaningful when `CONFIG_showNRounds=true`

### 6. `retaliatory_punisher`

Interpretation:

- punishes low contributors
- especially likely to punish visible prior aggressors

Active parameters:

- punishment deviation coefficient
- retaliation coefficient
- visibility-sensitive retaliation boost

### 7. `norm_enforcer`

Interpretation:

- punishes low contributors relative to the current group norm
- not mainly retaliatory

Active parameters:

- strong punishment deviation coefficient
- low retaliation coefficient
- moderate own contribution baseline

### 8. `reward_oriented_cooperator`

Interpretation:

- contributes relatively highly
- uses reward to reinforce above-norm contributors

Active parameters:

- positive baseline contribution
- positive reward deviation coefficient
- weak punishment coefficients

## Shared Parameter Blocks

Every family may be parameterized through a common block, but most families will activate only a subset.

Contribution parameters:

- `alpha_contrib_bin[k]`
- `beta_peer_mean`
- `beta_peer_std`
- `beta_own_prev`
- `beta_punished_prev`
- `beta_rewarded_prev`
- `beta_payoff_prev`
- `beta_endgame`
- `beta_group_size`

Punish parameters:

- `gamma0`
- `gamma_negative_deviation`
- `gamma_retaliation`
- `gamma_endgame`
- `gamma_visibility`

Reward parameters:

- `eta0`
- `eta_positive_deviation`
- `eta_reciprocity`
- `eta_endgame`
- `eta_visibility`

## First Inference Regime

Do not start with a fully personalized parameter vector for every player.

First implementation:

- one shared parameter vector per family
- soft player-family assignment probabilities
- optional treatment-level family offsets

Second implementation:

- player-specific posterior shrinkage around family means

## Model Outputs

The first simulator should expose:

- player family posterior
- family mixture by treatment
- contribution distribution by family
- punish/reward rate by family
- treatment shifts in family composition

Those outputs are central to the scientific use case: explaining how institutional design reorganizes behavior, not only predicting one aggregate scalar.
