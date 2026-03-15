# PGG Transfer Eval

This folder contains the evaluation pipeline for Twin economic-game prediction under the PGG-transfer setup.

## Main pieces

- `build_joint_social_baseline_batch.py`
  - Builds the no-retrieval joint social-game baseline.
  - One request per participant.
  - The prompt contains only `X_allowed`: demographics, personality, cognitive tests, mental accounting, time preference, and risk preference.
  - The target block `Y_target` contains the held-out social-game questions:
    - Trust: `QID117-122`
    - Ultimatum: `QID224-230`
    - Dictator: `QID231`
- `evaluate_batch_results.py`
  - Scores OpenAI Batch outputs against the manifest.
  - Uses the reference-style normalized accuracy metric.
  - Also reports exact-match and MAD as secondary diagnostics.
  - Can recover the `answers` object from partially truncated JSON if the model finished the `answers` block but cut off during `reasoning`.
- `compute_joint_social_comparison_baselines.py`
  - Computes comparison baselines for the joint social benchmark:
    - uniform random baseline
    - human consistency proxy

## Accuracy Metric

The evaluator follows the same accuracy logic used in the Digital-Twin-Simulation reference:

- Binary questions:
  - accuracy is `1` if prediction matches ground truth, else `0`
- Non-binary ordinal questions:
  - accuracy is `1 - abs(pred - truth) / (n_options - 1)`

This keeps accuracy in `[0, 1]`, gives `1` for an exact hit, and gives `0` only when the prediction is maximally far from the ground truth.

We also keep:

- `exact_match`
  - strict equality of predicted option and ground-truth option
- `MAD`
  - absolute difference in option index

## Human Consistency Proxy

There is no true test-retest ceiling for Trust / Ultimatum / Dictator in the public Twin release because the same games are not repeated across waves. Instead, we report a human consistency proxy.

This proxy uses the same participant's other social-game answers to predict a held-out social-game answer. It is not a deployable baseline and should not be called a true ceiling. It is an upper-reference for how internally consistent human behavior is across nearby tasks.

### Generosity questions

These are:

- `QID117` Trust sender
- `QID224` Ultimatum proposer
- `QID231` Dictator split

For each held-out generosity question, predict from the other two generosity questions:

1. Convert each answer to its option index in `1..6`
2. Take the mean of the other two option indices
3. Round to the nearest integer
4. Clamp to `1..6`

So:

- predict `QID117` from `QID224` and `QID231`
- predict `QID224` from `QID117` and `QID231`
- predict `QID231` from `QID117` and `QID224`

### Trust reciprocity questions

These are:

- `QID118` If sent `$5`, now holding `$15`
- `QID119` If sent `$4`, now holding `$12`
- `QID120` If sent `$3`, now holding `$9`
- `QID121` If sent `$2`, now holding `$6`
- `QID122` If sent `$1`, now holding `$3`

For each held-out trust reciprocity question:

1. Convert each observed option into the returned-dollar amount:
   - returned amount = `(amount received + 1) - option_index`
2. For the other trust reciprocity questions, compute the return rate:
   - `return_rate = returned_amount / amount_received`
3. Average those leave-one-out return rates
4. Multiply that average rate by the held-out amount received
5. Round to the nearest feasible returned-dollar amount
6. Convert that returned amount back into the corresponding option index

This is a leave-one-question-out reciprocity-rate predictor inside the trust block.

### Ultimatum receiver questions

These are:

- `QID225` Accept/reject if offered `$5`
- `QID226` Accept/reject if offered `$4`
- `QID227` Accept/reject if offered `$3`
- `QID228` Accept/reject if offered `$2`
- `QID229` Accept/reject if offered `$1`
- `QID230` Accept/reject if offered `$0`

For each held-out ultimatum receiver question:

1. Use the other five accept/reject answers to infer a minimum acceptable offer threshold
2. Search thresholds `t` from `0` to `6`
3. For each threshold, predict:
   - accept if `receiver_amount >= t`
   - reject otherwise
4. Choose the threshold with the fewest classification mistakes on the observed five answers
5. If several thresholds tie, average the tied thresholds and round
6. Apply that inferred threshold to the held-out offer

This is a leave-one-question-out acceptance-threshold predictor inside the ultimatum receiver block.

## Random Baseline

The random baseline is the expected score under uniform random answering over the legal options for each question.

For each question:

1. Enumerate all legal option indices
2. Compute normalized accuracy against the ground truth for each option
3. Average those scores uniformly

Under normalized accuracy, the random baseline can look higher than intuition suggests for multi-option questions because near-miss answers still receive partial credit. For that reason, exact-match and MAD should be reported alongside normalized accuracy.

## Output Files

Typical output structure:

- `output/joint_social_baseline/`
  - request JSONL, manifest, previews, token estimates
- `output/joint_social_baseline/comparison_baselines/`
  - random baseline summaries
  - human consistency proxy summaries

