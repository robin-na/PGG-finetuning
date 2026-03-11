# OOD Evaluation Plan

This redesign should be judged primarily on out-of-distribution transfer, not only random validation.

## Why

Current reduced-form models like `linear_config` are already strong on in-distribution macro prediction.

The algorithmic-latent simulator is only worth the added complexity if it improves one of:

- mechanistic interpretability
- micro-level treatment diagnostics
- out-of-distribution generalization to new design families

## Required Splits

### 1. Group Size OOD

Train:

- small-player games only

Test:

- large-player games only

Why:

- this directly checks whether the latent representation transfers across crowd size

### 2. Sanction Regime OOD

Train:

- no-reward or no-punishment subsets

Test:

- treatments with the missing sanction tool enabled

Why:

- tests whether the simulator can extrapolate from mechanism-level rules instead of memorizing treatment IDs

### 3. Visibility OOD

Train:

- hidden-identity or no-summary settings

Test:

- visible-identity or summary-visible settings

Why:

- checks whether the model correctly uses observability-sensitive latent structure

### 4. Horizon OOD

Train:

- `CONFIG_showNRounds=false`

Test:

- `CONFIG_showNRounds=true`

or the reverse.

Why:

- directly tests endgame-sensitive behavioral rules

### 5. Chat OOD

Train:

- no-chat treatments

Test:

- chat-enabled treatments

Why:

- likely the hardest split, and where an LLM-informed prior may matter most

## Baselines

Every OOD split should compare against:

- `linear_config`
- `archetype_cluster`
- `archetype_cluster_plus`
- algorithmic-latent data-only
- algorithmic-latent with LLM priors

## Metrics

### Macro

- normalized-efficiency MAE
- normalized-efficiency RMSE
- treatment-level correlation
- directional sign agreement

### Behavioral

- contribution distribution by treatment
- punish/reward frequency by treatment
- within-game heterogeneity
- round dynamics
- conditional-cooperation proxy

### Interpretability

Report:

- inferred family mixture by treatment
- family-level contribution patterns
- family-level sanction/reward patterns

The redesign is more valuable if it can say:

- treatment X increases conditional cooperators
- treatment Y suppresses retaliatory punishers
- treatment Z amplifies endgame defection

instead of only “efficiency went up/down.”

## Decision Rule

This redesign should be considered successful if:

- it does not catastrophically lose to simple baselines in-distribution
- it improves at least some strict OOD splits
- it yields interpretable treatment-level behavioral decompositions

If it fails all three, then the added modeling complexity is not justified.
