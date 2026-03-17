# PGG Transfer Eval

This folder contains the evaluation pipeline for Twin economic-game prediction under the PGG-transfer setup.

## Main pieces

- `ORACLE_RETRIEVAL_DESIGN.md`
  - Design for retrieval-augmented Twin prediction using the finished PGG oracle library plus demographics and exact PGG rule summaries.
- `build_oracle_library.py`
  - Builds the retrieval-ready oracle corpus from completed learn+val PGG participants.
  - Joins the raw completion filter, demographics, exact PGG rule summaries, and the existing oracle archetype text.
  - Writes one local markdown file per oracle card plus manifest JSONL files for upload and downstream joins.
- `upload_oracle_library_to_vector_store.py`
  - Uploads the local oracle-card markdown files into an OpenAI vector store.
  - Preserves stable metadata attributes per file so retrieval results can be joined back to local manifests by filename.
- `build_joint_social_oracle_query_batch.py`
  - Builds one OpenAI Batch request per Twin participant to write a retrieval query.
  - Uses the same non-leaky joint-social Twin profile as the reliable baseline, plus Twin demographics.
  - Outputs a compact JSON object with `search_query` and `match_cues`.
- `retrieve_oracle_candidates.py`
  - Takes the query-writer batch outputs and calls `client.vector_stores.search(...)`.
  - Uses OpenAI hosted retrieval only; no reranking in this first version.
  - Dedupes chunk-level hits to unique oracle files, then keeps the top `k` profiles.
- `build_joint_social_oracle_augmented_batch.py`
  - Builds the final oracle-augmented joint-social prediction batch.
  - Uses the corrected joint-social target block and injects the retrieved oracle cards as analogical evidence.
- `build_joint_social_oracle_onestep_batch.py`
  - Builds a one-step Responses API batch that uses hosted `file_search` directly inside the prediction call.
  - Forces one `file_search` call against a supplied vector store, includes `file_search_call.results`, and returns the final prediction JSON in the same response.
  - This is the preferred first-pass hosted-retrieval path because the batch output includes the actual search query and retrieved snippets.
- `build_joint_social_baseline_batch.py`
  - Builds the no-retrieval joint social-game baseline.
  - One request per participant.
  - Supports two prompt variants:
    - `baseline_no_retrieval`: full filtered `X_allowed` profile dump
    - `relevant_structured_summary`: compact structured summary using selected social-value/personality signals plus non-target behavioral-econ evidence
    - `relevant_structured_summary_tuned`: same compact structured summary, plus explicit game-role labels and stricter legal-range instructions for each target question
  - The target block `Y_target` contains the held-out social-game questions:
    - Trust: `QID117-122`
    - Ultimatum: `QID224-230`
    - Dictator: `QID231`
  - Can reuse the exact participant IDs from an existing manifest via `--reuse-manifest`, which is useful for prompt-variant comparisons on the same sample.
  - The output-format example uses placeholder strings for answers rather than anchoring every QID to option `1`.
  - Target questions are rendered with their full text, options, role labels, and legal answer ranges.
- `evaluate_batch_results.py`
  - Scores OpenAI Batch outputs against the manifest.
  - Uses the reference-style normalized accuracy metric.
  - Also reports exact-match and MAD as secondary diagnostics.
  - Can recover the `answers` object from partially truncated JSON if the model finished the `answers` block but cut off during `reasoning`.
  - Supports both `/v1/chat/completions` batch outputs and `/v1/responses` batch outputs.
  - When Responses file-search traces are present, writes them to `retrieval_traces.jsonl`.
  - By default, writes results into an `evals/` folder next to the `outputs_*.jsonl` file being evaluated. You can override this with `--output-dir`.
- `compute_joint_social_comparison_baselines.py`
  - Computes comparison baselines for the joint social benchmark:
    - uniform random baseline
    - human consistency proxy
  - Uses the exact participant subset from the manifest you pass in, so sample runs and full runs get matched comparison baselines.
- `compute_joint_social_knn_baseline.py`
  - Computes a leave-one-out human-neighbor baseline using the same structured Twin inputs as the tuned no-retrieval prompt.
  - Builds one deterministic feature vector per participant from:
    - selected social/value/personality rows
    - mental accounting MC items
    - time-preference switch summaries
    - gain/loss risk-preference switch summaries
  - Finds nearest human neighbors in that allowed-input feature space, then predicts the held-out social-game questions from neighbor behavior.
  - This is a stronger non-LLM comparator than the random baseline, but it is not a deployable prompt baseline; it uses other participants' labeled target answers.
- `compute_joint_social_trait_heuristic.py`
  - Computes a deterministic no-other-labels heuristic from the same structured Twin inputs.
  - Builds hand-coded latent indices for:
    - prosociality
    - fairness / retaliation sensitivity
    - caution / self-interest
    - trustingness
  - Maps those indices directly to the held-out trust, ultimatum, and dictator targets without using any other participants' target labels.

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

## kNN Allowed-Input Baseline

This baseline uses only the same non-leaky structured Twin inputs that the tuned no-retrieval prompt sees, but instead of prompting an LLM it predicts from nearby human participants.

Mechanically:

1. Build a structured feature vector from the allowed Twin profile
   - selected social/value/personality rows
   - mental-accounting answers
   - summarized time/risk switch behavior
2. Standardize those features
3. For each participant, find leave-one-out nearest neighbors in that feature space
4. Predict held-out targets from neighbor behavior
   - generosity items: weighted average of neighbor choices
   - trust-return items: weighted average return rates
   - ultimatum receiver items: weighted average implied acceptance thresholds

This is best interpreted as a human-analog comparator using the same allowed inputs, not as a clean train/test supervised baseline.

## Trait-Index Heuristic

This is the cleanest simple comparator that does not use any other participants' target labels.

Mechanically:

1. Build the same structured Twin input profile used by the tuned no-retrieval prompt
2. Collapse it into a few hand-coded latent indices:
   - prosociality
   - fairness / retaliation sensitivity
   - caution / self-interest
   - trustingness
3. Convert those indices to held-out target predictions with fixed rules
   - trust send: more prosocial / trusting implies more sending
   - trust returns: higher prosociality implies a higher return rate
   - ultimatum offer: more prosociality / fairness implies a fairer offer
   - ultimatum receiver: fairness sensitivity raises the minimum acceptable offer
   - dictator: prosociality lowers the amount kept for self

This baseline is intentionally simple and interpretable. It is useful as a clean no-other-labels comparator, but it should not be expected to compete with stronger human-neighbor baselines.

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
- `output/oracle_library/`
  - retrieval-ready oracle corpus, manifests, and per-card markdown files
- `output/joint_social_oracle_query/`
  - query-writer request JSONL, manifest, previews, token estimates
- retrieval output folder next to the query batch output
  - `oracle_candidates.jsonl`
  - raw search traces
  - retrieval summaries
- `output/joint_social_oracle_augmented/`
  - oracle-augmented prediction request JSONL, manifest, previews, token estimates
- `output/joint_social_oracle_onestep/`
  - one-step hosted-retrieval prediction request JSONL, manifest, previews, token estimates

## One-Step Hosted Retrieval Run Order

1. Build the local oracle corpus
   - `python non-PGG_generalization/pgg_transfer_eval/build_oracle_library.py`
2. Upload the oracle corpus into an OpenAI vector store
   - `python non-PGG_generalization/pgg_transfer_eval/upload_oracle_library_to_vector_store.py`
3. Build the one-step hosted-retrieval prediction batch
   - `python non-PGG_generalization/pgg_transfer_eval/build_joint_social_oracle_onestep_batch.py --vector-store-id <id> --include-reasoning`
4. Run that batch with OpenAI and save the resulting `outputs_*.jsonl`
5. Score it with `evaluate_batch_results.py`
   - retrieval traces will be written to `retrieval_traces.jsonl`

## Two-Stage Retrieval Run Order

This remains available as a more inspectable fallback if you want to separate query writing, retrieval, and prediction.

1. Build the local oracle corpus
2. Upload the oracle corpus into an OpenAI vector store
3. Build the Twin query-writer batch
   - `python non-PGG_generalization/pgg_transfer_eval/build_joint_social_oracle_query_batch.py`
4. Run that batch with OpenAI and save the resulting `outputs_*.jsonl`
5. Retrieve oracle candidates from the vector store
   - `python non-PGG_generalization/pgg_transfer_eval/retrieve_oracle_candidates.py --vector-store-id <id> --responses-jsonl <query_outputs.jsonl>`
6. Build the oracle-augmented prediction batch
   - `python non-PGG_generalization/pgg_transfer_eval/build_joint_social_oracle_augmented_batch.py --candidates-jsonl <oracle_candidates.jsonl> --include-reasoning`
7. Run the final prediction batch, then score it with `evaluate_batch_results.py`

## Archived Unreliable Runs

Pre-fix `joint_social` runs built before March 16, 2026 were archived under:

- `output/archived_unreliable_joint_social_pre_2026-03-16/`

Those runs used a buggy version of the joint-social builder that failed to insert the held-out target question blocks into the prompt text.
