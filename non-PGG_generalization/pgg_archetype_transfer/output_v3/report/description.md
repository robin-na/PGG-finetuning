# PGG Archetype Transfer to Economic Game Prediction
## Experiment Description

We evaluate whether behavioral archetypes derived from a Public Goods Game (PGG)
can improve predictions of individual behavior in other economic games, when only
sparse demographic information (age, sex, education) is available about participants.

**Dataset:** Twin-2K-500 (2,058 US participants, Toubia et al. 2025)
**Target tasks:** 14 question-columns across 9 economic game types
  (Ultimatum Game proposer/responder, Dictator Game, Trust Game sender/receiver,
   Time Preferences, Risk Preferences (gain/loss), Mental Accounting)
**Evaluation:** Pilot of 100 randomly sampled participants (pilot_n=200 from 2,058)
**Prediction model:** GPT-4o, temperature=0

## Methods

Three conditions were tested:

1. **Demographics Only** (baseline): GPT-4o receives only the participant's age
   bracket, sex, and education level, and predicts the game response.

2. **Random Archetype** (control): GPT-4o receives demographics plus a randomly
   sampled PGG behavioral archetype from the learning wave. This tests whether any
   archetype—regardless of fit—adds information beyond demographics alone.

3. **PGG Archetype (D-retrieval)** (main condition): A Ridge regression model maps
   the participant's demographic features to a 3072-dim embedding space. The top-10
   nearest PGG archetypes are retrieved from the learning-wave bank via cosine
   similarity, and provided to GPT-4o alongside the question. The LLM selects the
   most relevant behavioral signals and applies them to the target question.

PGG archetypes are rich prose descriptions of a player's behavior in a Public Goods
Game, covering: contribution patterns, punishment/reward behavior, responses to
others' outcomes, and end-game strategy. These behavioral tendencies are used as
personality signals to predict economic game responses.

## Results

| Mode | Accuracy | Pearson r | MAD |
|------|----------|-----------|-----|
| Demographics Only | 0.610 | 0.395 | 0.651 |
| Random Archetype | 0.567 | 0.300 | 0.726 |
| PGG Archetype (D-retrieval) | 0.573 | 0.312 | 0.737 |

The PGG Archetype condition achieves 0.573 accuracy and 0.312
mean Pearson r with human responses, compared to 0.610 accuracy and
0.395 correlation for the Demographics-Only baseline. This represents
a 0.037 below lift in accuracy and 0.083
below lift in correlation from archetype conditioning.

The Random Archetype control achieves 0.567 accuracy (0.300 r),
below the Demographics-Only baseline.
Comparing Random vs. Retrieved archetypes isolates the value of demographic-based
retrieval over providing any arbitrary PGG behavioral profile.

### Per-Game Results (Archetype mode)

| Game | N cols | Accuracy | Pearson r |
|------|--------|----------|-----------|
| Ultimatum (responder) | 6 | 0.828 | 0.207 |
| Dictator | 1 | 0.710 | 0.498 |
| Ultimatum (proposer) | 1 | 0.660 | 0.454 |
| Trust (receiver) | 5 | 0.296 | 0.285 |
| Trust (sender) | 1 | 0.200 | 0.429 |

## Figures

- **fig1_overall_metrics.png** — Bar chart comparing Accuracy, Pearson r, and MAD
  across three prediction modes.
- **fig2_per_game_accuracy.png** — Grouped bar chart of accuracy by game type and mode.
- **fig3_correlation_distribution.png** — Histogram of per-question Pearson r values
  for each mode; dashed lines show medians.
- **fig4_mad_heatmap.png** — Heatmap of MAD by game type × mode (red = worse, green = better).
- **fig5_archetype_lift.png** — Per-game lift of Archetype over Demographics-Only baseline
  for accuracy and correlation.

## Notes

- Economic game questions (Ultimatum, Dictator, Trust, Time/Risk Preferences) are
  only available in wave 1-3 of Twin-2K-500 (not repeated in wave 4), so we
  evaluate against wave 1-3 ground truth.
- The D-only Ridge model has near-zero hit@1 (expected with only 3 demographic
  dimensions), meaning retrieval is effectively random within demographic strata.
  The main test is therefore whether archetype format/content adds signal over
  pure demographics, not whether retrieval precision matters.
- All predictions are zero-temperature (deterministic).
- Text entry questions (reasoning rationales) are excluded from evaluation.
