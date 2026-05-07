# Twin PGG Card Ablation Variant Spec

## Shared Inputs

All variants read the same structured Twin profile file:

- `non-PGG_generalization/twin_profiles/output/twin_extended_profiles/twin_extended_profiles.jsonl`

All PGG batch runs should reuse the same locked assignment file unless explicitly testing sampling:

- corrected assignment: `forecasting/pgg/profile_sampling/output/twin_to_pgg_validation_persona_sampling/seed_0/game_assignments.jsonl`
- unadjusted assignment: `forecasting/pgg/profile_sampling/output/twin_to_pgg_validation_persona_sampling_unadjusted/seed_0/game_assignments.jsonl`

The main corrected ablation should use the corrected assignment only.

## Category Source

The six categories below come from the explicit Twin catalog/profile mapping structure:

- raw `BlockName`
- raw `QuestionID`
- mapped `family`
- mapped profile subsection in `twin_extended_profile_mapping.csv`

They are not learned clusters.

## Active Six Categories

Each prompt-facing card exposes only the named evidence category. Demographic background is a category, not a shared prompt baseline. The sampling assignment may still be demographically corrected, but hidden demographic fields should not appear in non-background category prompts.

### 1. `twin_pgg_background_only`

Shows only sampled Twin background fields:

- age bracket
- sex assigned at birth
- completed education

Purpose: confirms whether Twin demographics alone reproduce the full Twin-card lift.

### 2. `twin_pgg_direct_social_only`

Shows direct one-shot social-allocation task evidence:

- trust send amount
- trust return share
- ultimatum offer
- ultimatum minimum acceptable amount
- ultimatum rejection rate
- dictator offer

Purpose: tests whether behavior in directly related social-allocation tasks drives PGG prediction.

### 3. `twin_pgg_self_report_social_only`

Shows social/personality questionnaire evidence:

- agreeableness
- prosocial values
- cooperation orientation
- empathy
- social sensitivity
- uncertainty aversion
- behavioral stability proxies from conscientiousness, self-concept clarity, and low neuroticism

Purpose: tests whether broad personality/social self-report is enough without direct game evidence.

### 4. `twin_pgg_non_social_econ_only`

Shows non-social economic preference task summaries:

- patience/later-choice rate
- gain lottery choice rate
- loss lottery choice rate
- mental-accounting endorsement rate

Purpose: tests whether time, risk, and accounting style explain PGG alignment.

### 5. `twin_pgg_cognitive_only`

Shows cognitive task summaries:

- financial literacy
- numeracy
- cognitive reflection
- logical reasoning

Purpose: tests whether strategic/cognitive sophistication drives PGG alignment.

### 6. `twin_pgg_misc_heuristics_pricing_text_only`

Shows the remaining lower-priority Twin evidence families:

- heuristics and biases
- pricing and consumer choice
- open-text availability and coarse open-text metadata

Purpose: tests whether the residual Twin material adds predictive signal beyond the five main categories.

## Deferred Diagnostic Contrasts

The following contrasts are intentionally deferred and are not part of the first six-category grid:

- full-minus-direct-social
- full-minus-self-report
- scores-only
- anchors-only

They answer different questions from the source-category ablation and should only be added after the category result is clear.

## Leakage-Control Standard

An ablation card should not keep full-card `headline`, `summary`, or `behavioral_signature` text if those fields were originally computed from excluded evidence. For example, a `direct_social_only` card should not preserve "behaviorally stable" if the stability signal came from self-report.

The first-pass six-category cards should not render prompt-facing `Headline:` or `Summary:` lines at all. Category-level context belongs only in the once-per-game visible evidence guide; player sections should contain only the visible fields for that category.
