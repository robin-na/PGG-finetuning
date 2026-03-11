# Prompting Plan

This folder is reserved for prompts used in the algorithmic-latent redesign.

LLM prompting in this branch should follow one rule:

- the LLM proposes structured priors or hypotheses
- the data decides whether those priors survive

## Allowed Prompt Tasks

- summarize a player trajectory into likely family priors
- suggest coefficient sign priors for a family
- reason about how a `CONFIG` change should affect family prevalence or parameter signs
- propose missing family candidates when residuals suggest an uncovered behavior pattern

## Disallowed Prompt Tasks

- directly predict final game outcomes without calibration
- invent unrestricted executable strategy code
- overwrite statistically fit parameters

## Prompt Output Format

Prompt outputs should be structured JSON-like objects whenever possible.

Example:

```json
{
  "candidate_families": [
    {"family": "conditional_cooperator", "weight": 0.65},
    {"family": "reward_oriented_cooperator", "weight": 0.20}
  ],
  "parameter_sign_priors": {
    "beta_peer_mean": "positive",
    "eta_positive_deviation": "positive",
    "beta_endgame": "weak_negative"
  },
  "rationale": [
    "player follows group contribution changes",
    "player rewards above-norm contributors"
  ]
}
```

That structure can then be turned into priors for inference rather than being used directly as the simulator output.
