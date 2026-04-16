# Twin Extended Profile Spec

## Purpose

This spec defines a high-coverage participant profile for Twin-2k.

The profile is intentionally:

- descriptive, not predictive
- high-coverage over Twin waves 1-3
- explicit about evidence provenance
- compatible with a later transfer step into PGG or other social-decision tasks

The key design choice is that the profile itself should **not** assert detailed "this person would do X in game Y" forecasts. Instead, it should stop at:

- what was directly observed in Twin
- what latent tendencies are reasonably supported
- what transfer-relevant cues might matter later

Detailed PGG forecasts, if needed, should be a separate downstream step that consumes this profile plus a specific game configuration.

## Files

- Schema: `non-PGG_generalization/twin_profiles/twin_extended_profile_schema.json`
- Per-entry mapping: `non-PGG_generalization/twin_profiles/twin_extended_profile_mapping.csv`
- Mapping builder: `non-PGG_generalization/twin_profiles/build_twin_extended_profile_mapping.py`
- Profile builder: `non-PGG_generalization/twin_profiles/build_twin_extended_profiles.py`
- Card schema: `non-PGG_generalization/twin_profiles/twin_extended_profile_card_schema.json`
- Card renderer: `non-PGG_generalization/twin_profiles/render_twin_extended_profile_cards.py`
- Card-distribution analysis: `non-PGG_generalization/twin_profiles/analyze_twin_profile_card_distributions.py`
- Output JSONL: `non-PGG_generalization/twin_profiles/output/twin_extended_profiles/twin_extended_profiles.jsonl`
- Output preview: `non-PGG_generalization/twin_profiles/output/twin_extended_profiles/preview_twin_extended_profiles.json`
- Compact cards: `non-PGG_generalization/twin_profiles/output/twin_extended_profile_cards/compact/`
- Standard cards: `non-PGG_generalization/twin_profiles/output/twin_extended_profile_cards/standard/`
- Audit cards: `non-PGG_generalization/twin_profiles/output/twin_extended_profile_cards/audit/`
- PGG-prompt cards: `non-PGG_generalization/twin_profiles/output/twin_extended_profile_cards/pgg_prompt/`
- Minimal PGG-prompt cards: `non-PGG_generalization/twin_profiles/output/twin_extended_profile_cards/pgg_prompt_min/`
- Standard-mode analysis: `non-PGG_generalization/twin_profiles/output/twin_extended_profile_cards/standard/analysis/`

## Generation

Regenerate the full deterministic profile set with:

```bash
python non-PGG_generalization/twin_profiles/build_twin_extended_profiles.py
```

Render readable cards from the structured profiles with:

```bash
python non-PGG_generalization/twin_profiles/render_twin_extended_profile_cards.py --all-modes
```

Current full-build status:

- profiles written: `2058`
- validation: each profile is checked against `twin_extended_profile_schema.json` during generation
- cards written per mode: `2058`
- card modes: `compact`, `standard`, `audit`, `pgg_prompt`, `pgg_prompt_min`
- card validation: each card is checked against `twin_extended_profile_card_schema.json`

## Card Layer

The readable card layer is intentionally a rendering step, not a new inference step.

It should:

- restate only what is already supported by the structured profile
- use deterministic templates and score-based phrasing
- explain how high-level trait-like fields are operationalized from Twin blocks and tasks
- keep transfer-relevant cues explicitly marked as cues, not forecasts
- preserve uncertainty notes instead of smoothing them away

It should not:

- add new unsupported latent traits
- convert transfer-relevant cues into exact PGG predictions
- overwrite contradictions between self-report and observed task behavior

In the current card schema, this means:

- `background` is mode-dependent and intentionally shorter than the full demographics object
- `social_style` and `decision_style` focus on what evidence families those sections aggregate
- each transfer cue includes an explicit "constructed from" breakdown so terms like cooperation, norm enforcement, or communication are tied to specific Twin task families rather than treated as free-floating personality labels

The detail modes are intended for downstream experimentation:

- `compact`: shorter background, fewer observed anchors, shorter derivation text
- `standard`: balanced readable card
- `audit`: broader background coverage, more anchors, and fuller construction notes
- `pgg_prompt`: prompt-facing mode for PGG augmentation; shared methodological caveats move to a separate prompt note and per-player cards keep only player-specific limits when needed
- `pgg_prompt_min`: more token-efficient PGG augmentation mode; cue definitions and shared caveats live in one shared prompt file while player cards mostly keep values, anchors, and player-specific limits only

## Inclusion Rule

Default rule:

- keep every Twin catalog entry that has at least one response column
- drop only pure scaffolding / instruction / transition entries with no participant response

In the current catalog:

- total catalog entries: `256`
- retained for the machine-readable evidence profile: `242`
- excluded scaffolding entries: `14`

## Important Keying Rule

Bare `QuestionID` is **not** globally unique in the Twin catalog.

Examples:

- `QID268-270` appear in both `Cognitive tests` and `Personality`
- `QID271-272` appear in both `Cognitive tests` and `Economic preferences`
- `QID275-279` appear in both `Cognitive tests` and `Economic preferences`

So the profile pipeline should key evidence by:

- `question_ref = "{BlockName}::{QuestionID}"`

not by bare `QuestionID`.

## Top-Level Profile Structure

The JSON schema uses these top-level sections:

- `background_context`
- `observed_in_twin`
- `derived_dimensions`
- `behavioral_signature`
- `social_style`
- `decision_style`
- `pgg_relevant_cues`
- `uncertainties`

### Section Intent

`background_context`

- demographics and coarse participant context
- should not be over-interpreted as deep trait evidence

`observed_in_twin`

- direct questionnaire / task evidence grouped into interpretable blocks
- should preserve question references and high-signal diagnostic items

`derived_dimensions`

- cautious trait-like summaries inferred from the observed evidence
- these are still Twin-grounded, not game-specific forecasts

`behavioral_signature`

- 3-8 short lines describing the most distinctive cross-block patterns

`social_style`

- readable synthesis of interpersonal / prosocial / fairness-relevant tendencies

`decision_style`

- readable synthesis of cognition, risk/time style, consumer style, and heuristics

`pgg_relevant_cues`

- transfer-relevant cues only
- these are intentionally **not** scenario-specific PGG behavior predictions

`uncertainties`

- what remains weakly identified or confounded

## Concrete Mapping From Twin Blocks Into Profile Sections

The full per-entry mapping is in `twin_extended_profile_mapping.csv`.

The main grouping is:

### 1. `background_context.demographics`

Include:

- `Demographics::QID11-QID24`

Use for:

- region
- male/female
- age bracket
- education
- race/origin
- citizenship
- marital / family / employment / income / religion / politics context

Retention:

- keep raw values
- also compute harmonized context features for downstream use

### 2. `observed_in_twin.personality_and_self_report`

Include:

- `Personality::QID25-QID35`
- `Personality::QID125-QID148`
- `Personality::QID232-QID239`

Use for:

- broad personality traits
- values
- empathy
- cooperation vs competition orientation
- uncertainty aversion / need for closure
- affect / depressive symptoms
- spending self-control

Retention:

- keep block-level summaries and a few diagnostic items
- do not narrate every matrix row in the readable card

### 3. `observed_in_twin.open_text_responses`

Include:

- `Personality::QID268-QID270`
- `Forward Flow::QID10`

Use for:

- self-narratives about actual / ideal / ought self
- free-association text

Retention:

- preserve raw excerpts
- summarize cautiously
- do not let open text dominate the structured evidence

### 4. `observed_in_twin.social_game_behavior`

Include:

- Trust:
  - `Economic preferences::QID117-QID122`
  - `Economic preferences::QID271-QID272`
- Ultimatum:
  - `Economic preferences::QID224-QID230`
- Dictator:
  - `Economic preferences::QID231`
  - `Economic preferences::QID275`

Use for:

- observed trusting / sending behavior
- reciprocity / return behavior
- proposer fairness
- responder rejection threshold
- unilateral sharing
- decision rationales where text is available

Retention:

- keep raw behavior plus compact summaries
- keep clearly separated from any later transfer step

### 5. `observed_in_twin.economic_preferences_non_social`

Include:

- Mental accounting:
  - `Economic preferences::QID149-QID152`
- Time preference:
  - `Economic preferences::QID84`
  - `Economic preferences::QID244-QID248`
- Risk preference, gains:
  - `Economic preferences::QID250-QID252`
- Risk preference, losses:
  - `Economic preferences::QID276-QID279`

Use for:

- loss/gain asymmetry
- discounting / patience
- certainty equivalents
- mental accounting tendencies

Retention:

- keep raw task outcomes
- summarize into interpretable block features

### 6. `observed_in_twin.cognitive_performance`

Include:

- `Cognitive tests::QID36-QID61`
- `Cognitive tests::QID63-QID83`
- `Cognitive tests::QID123-QID124`
- `Cognitive tests::QID217-QID221`
- `Cognitive tests::QID268-QID279`

Use for:

- financial literacy
- numeracy / probability translation
- CRT-style reflection
- verbal ability
- visual reasoning
- logical reasoning
- metacognitive calibration from self-estimated performance

Retention:

- score into subsummaries where possible
- keep only a few diagnostic examples in the readable card

### 7. `observed_in_twin.heuristics_and_biases`

Include:

- Anchoring:
  - `Anchoring - African countries low::QID163-QID164`
  - `Anchoring - African countries high::QID165-QID166`
  - `Anchoring - redwood low::QID167-QID168`
  - `Anchoring - redwood high::QID169-QID170`
- Base-rate:
  - `Base-rate 30 engineers::QID154`
  - `Base-rate 70 engineers::QID156`
- Framing / outcome / conjunction / search-effort / probability tasks:
  - `Disease - gain::QID157`
  - `Disease-loss::QID158`
  - `Linda -no conjunction::QID159`
  - `Linda-conjunction::QID160`
  - `Outcome bias - success::QID161`
  - `Outcome bias - failure::QID162`
  - `Less is More Gamble A-C::QID171-QID173`
  - `Proportion dominance 1A-2C::QID174-QID179`
  - `Sunk cost - no::QID181`
  - `Sunk cost - yes::QID182`
  - `Absolute vs. relative - calculator::QID183`
  - `Absolute vs. relative - jacket::QID184`
  - `WTA/WTP Thaler problem - WTP certainty::QID189`
  - `WTA/WTP Thaler problem - WTA certainty::QID190`
  - `WTA/WTP Thaler - WTP noncertainty::QID191`
  - `Allais Form 1::QID192`
  - `Allais Form 2::QID193`
  - `Myside Ford::QID194`
  - `Myside German::QID195`
  - `Non-experimental heuristics and biases::QID196`
  - `Probability matching vs. maximizing - Problem 1::QID198`
  - `Probability matching vs. maximizing - Problem 2::QID203`
  - `False consensus::QID287`
  - `Non-experimental heuristics and biases::QID288-QID291`

Use for:

- anchor susceptibility
- framing sensitivity
- base-rate neglect
- conjunction sensitivity
- sunk-cost / WTA-WTP / risk-framing tendencies
- belief projection / false consensus
- simple policy-perception and risk-benefit estimation style

Retention:

- summarize by bias family
- keep a few diagnostic scenarios only

### 8. `observed_in_twin.pricing_and_consumer_choice`

Include:

- `Product Preferences - Pricing::QID9_1-QID9_40`

Use for:

- price sensitivity
- willingness to search for savings
- brand / convenience trade-offs
- reference dependence in purchase decisions

Retention:

- summarize heavily
- do not write 40 item-level product bullets in the readable card

## Explicit Exclusions

The only default exclusions are scaffolding entries with no participant response:

- `Product Preferences - Pricing::QID8`
- `Cognitive tests::QID62`
- `Cognitive tests::QID73`
- `Cognitive tests::QID241`
- `Cognitive tests::QID265`
- `Economic preferences - intro::QID242`
- `Economic preferences::QID243`
- `Economic preferences::QID249`
- `Economic preferences - intro::QID266`
- `Personality::QID127`
- `Personality::QID240`
- `Personality::QID260`
- `Personality::QID262`
- `Personality::QID285`

## Derived Dimensions

The schema fixes the following derived-dimension families:

### `derived_dimensions.social_preferences`

- `trustingness`
- `reciprocity`
- `fairness_enforcement`
- `altruistic_sharing`
- `exploitation_caution`

### `derived_dimensions.decision_style`

- `patience`
- `risk_tolerance_gains`
- `risk_tolerance_losses`
- `mental_accounting_reliance`
- `cognitive_reflection`
- `numeracy`
- `logical_reasoning`
- `anchor_susceptibility`
- `framing_susceptibility`

### `derived_dimensions.consumer_style`

- `price_sensitivity`
- `willingness_to_search`
- `reference_dependence`
- `purchase_inhibition`

### `derived_dimensions.self_regulation_and_affect`

- `empathy`
- `cooperation_orientation`
- `competition_orientation`
- `uncertainty_aversion`
- `depressive_affect`
- `spending_self_control`

## PGG-Relevant Cues

The profile keeps these as cues, not forecasts:

- `cooperation_orientation`
- `conditional_cooperation`
- `norm_enforcement`
- `generosity_without_return`
- `exploitation_caution`
- `communication_coordination`
- `behavioral_stability`

The intent is:

- useful for later transfer
- not overclaiming exact public-goods-game behavior before a specific game context is provided

## Recommended Build Order

1. Compile deterministic block summaries from all retained Twin entries.
2. Select a small set of diagnostic raw items per subsection.
3. Fill the schema sections using only Twin-grounded evidence.
4. Stop at `pgg_relevant_cues`.
5. If needed later, run a separate transfer module that maps profile + game config -> PGG forecast.
