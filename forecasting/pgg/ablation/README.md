# PGG Twin Card Ablations

This folder owns the PGG-specific ablation tests for Twin-derived profile cards.

The shared Twin profile pipeline remains canonical in:

- `non-PGG_generalization/twin_profiles/`

This folder should contain only PGG forecasting ablation logic:

- PGG-specific ablation card renderers
- PGG-specific ablation variant specs
- PGG-specific batch-building notes
- generated PGG ablation outputs under `output/`

## Placement Decision

Create PGG ablation cards here, not in `non-PGG_generalization/twin_profiles/`, when the card is designed to answer a PGG forecasting question.

Reason:

- `non-PGG_generalization/twin_profiles/` is the shared source of deterministic Twin evidence profiles.
- the ablation changes what PGG prompts expose to the model, not the canonical Twin data representation.
- the same Twin profiles may later be ablated differently for other games.

If an ablation card format becomes a reusable cross-game profile format, then it can be promoted back into the shared Twin profile folder.

## Clean Ablation Rule

For a causal card-content ablation, the sampled Twin profile assignments must be locked.

That means every ablation condition should use the same:

- validation game set
- game order
- seat order
- `twin_pid` assigned to each game seat
- model
- batch settings
- parsing and evaluation pipeline

Only the prompt-facing card content should change.

The default locked assignment for corrected Twin ablations is:

- `forecasting/pgg/profile_sampling/output/twin_to_pgg_validation_persona_sampling/seed_0/game_assignments.jsonl`

The unadjusted assignment can be ablated separately, but corrected and unadjusted should not be mixed inside the same causal card-content comparison.

## Core Question

The current PGG result says that Twin cards help more than demographic-only profiles. This ablation asks which evidence family inside the Twin card is doing the work:

- demographics/background
- direct social-allocation tasks
- social/personality self-report
- non-social economic preferences
- cognitive performance
- miscellaneous lower-priority evidence: heuristics/biases, pricing/consumer choice, and open-text availability

## Active Six-Category Plan

Use exactly six evidence-family categories for the first pass:

- `twin_pgg_background_only`
- `twin_pgg_direct_social_only`
- `twin_pgg_self_report_social_only`
- `twin_pgg_non_social_econ_only`
- `twin_pgg_cognitive_only`
- `twin_pgg_misc_heuristics_pricing_text_only`

These are source-evidence categories, not unsupervised clusters and not downstream diagnostic contrasts.

Each prompt-facing card should expose only the named evidence category. Demographic background appears only in `twin_pgg_background_only`; it is not included as a common baseline in the other five categories. The locked Twin assignment file can still be demographically corrected, but the non-background category prompts should not show demographic fields.

Run the six-category grid on `gpt-5-mini` first. Then rerun the most diagnostic categories on `gpt-5.1`.

## Important Caveat

These are information ablations, not token-count-matched ablations. If a reviewer worries that shorter prompts are easier or harder for the model, add a second-stage length-control condition with neutral filler or non-diagnostic placeholders.

Do not treat leave-one-family-out, scores-only, or anchors-only contrasts as part of this first-pass category plan. Those can be added later as a second-stage diagnostic grid after the six source categories are evaluated.
