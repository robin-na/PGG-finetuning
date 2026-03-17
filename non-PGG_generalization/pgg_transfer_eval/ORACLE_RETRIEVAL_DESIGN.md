# Oracle Retrieval Design

## Goal

Augment the existing Twin economic-game predictor with retrieved PGG oracle archetypes.

The core claim is:

- Twin gives us partial evidence about a person through `X_allowed`
- the PGG oracle library gives us rich case-based descriptions of social-preference behavior under explicit repeated-game rules
- retrieving a few analogous PGG cases should help the model reason about held-out Twin trust / ultimatum / dictator behavior

This design intentionally starts with the existing oracle library, not the newer transfer-profile cards. The oracle library is the simplest high-coverage starting point.

## Benchmark Target

Use the current reliable Twin baseline setup as the prediction task:

- one request per Twin participant
- joint target block:
  - Trust: `QID117-122`
  - Ultimatum: `QID224-230`
  - Dictator: `QID231`
- same filtered Twin `X_allowed` profile used in the tuned no-retrieval baseline

This keeps the retrieval comparison clean:

- `Twin-only tuned baseline`
- `Twin + random oracle cards`
- `Twin + retrieved oracle cards`

## Library Definition

### Source files

- learn oracle summaries: `Persona/archetype_oracle_gpt51_learn.jsonl`
- val oracle summaries: `Persona/archetype_oracle_gpt51_val.jsonl`
- learn demographics: `demographics/demographics_numeric_learn.csv`
- val demographics: `demographics/demographics_numeric_val.csv`
- raw/processed config source is already accessible through the PGG profile builders

### Inclusion rule

Use both learning and validation waves, but only completed games:

- learn finished oracle rows: `3691`
- val finished oracle rows: `3617`
- total finished oracle rows: `7308`

Demographic join coverage among finished oracle rows:

- learn finished with demographics: `3673`
- val finished with demographics: `3612`
- total finished with demographics: `7285`

Default recommendation:

- keep all `7308` completed oracle rows
- attach demographics when available
- mark missing demographics as unknown rather than dropping those rows

### Library card format

Each library row should be a compact retrieval card with:

- identifiers in metadata only:
  - `wave`
  - `gameId`
  - `playerId`
- demographics:
  - age
  - gender
  - education
- exact PGG rule summary from `CONFIG_*`:
  - group size
  - rounds
  - multiplier
  - all-or-nothing vs continuous contribution
  - punishment available + cost/magnitude
  - reward available + cost/magnitude
  - chat available
  - whether horizon known
- oracle archetype text
- a fixed caveat line:
  - behavior was observed in a repeated public-goods game under the above rules and may be partly specific to that environment

Suggested output file:

- `non-PGG_generalization/data/PGG/oracle_library_finished_with_rules_and_demo.jsonl`

## Query Definition

The query should come from the same Twin profile already used in the tuned baseline. Do not create a separate leakage path for retrieval.

For each Twin participant, build a retrieval query card from:

- demographics
- selected personality / values / empathy signals
- non-target behavioral-econ evidence
- the compact structured Twin summary already used in the tuned baseline prompt

Recommended query text sections:

1. `Twin demographics`
2. `Twin social/value profile`
3. `Twin non-target economic behavior`
4. `Target prediction task`
   - predict trust / ultimatum / dictator responses for this participant

Important:

- retrieval query must not include held-out target answers
- retrieval query should stay compact and use the same participant subset / filtering as the actual prediction prompt

## Retrieval Architecture

Use OpenAI-hosted retrieval rather than a hand-rolled local cosine search.

### Why

The official retrieval stack is more capable than a bare embedding nearest-neighbor baseline:

- Retrieval API supports natural-language search over vector stores
- it supports `rewrite_query=true`
- it supports attribute filtering
- it supports ranking controls such as `ranker`, `score_threshold`, and hybrid semantic/keyword weighting
- the file search tool in the Responses API is a managed retrieval tool over vector stores

Official docs:

- [Retrieval guide](https://developers.openai.com/api/docs/guides/retrieval)
- [File search guide](https://developers.openai.com/api/docs/guides/tools-file-search)

### Recommended architecture

Use a three-stage retrieval system.

### Stage 0: Query writing with an LLM

Do not search with the raw Twin prompt dump.

Instead, first ask a model to convert the Twin participant profile plus target task into a compact retrieval query JSON:

- `query_text`
- `retrieval_focus`
- `optional_attribute_filters`

The query text should describe:

- relevant demographics
- non-target behavioral evidence
- social-preference tendencies implied by the Twin profile
- the fact that the downstream target is trust / ultimatum / dictator prediction

This gives us LLM assistance in retrieval without making the final prediction model search the entire library blindly.

### Stage 1: OpenAI Retrieval API shortlist

Store one oracle card per file in an OpenAI vector store and attach metadata attributes:

- `wave`
- `game_finished`
- `chat_enabled`
- `punishment_enabled`
- `reward_enabled`
- `show_n_rounds`
- `action_space`
- demographic buckets when available:
  - `age_bucket`
  - `gender`
  - `education`

Then call `vector_stores.search(...)` with:

- `query = query_text`
- `rewrite_query = true`
- `max_num_results = 30`
- `attribute_filter = ...` when useful
- `ranking_options` tuned later if needed

Recommended hard filter:

- `game_finished = true`

Recommended default: no demographic hard filter, because demographics should guide retrieval weakly, not exclude plausible matches.

### Stage 2: LLM rerank over shortlist

Take the top `30` results from retrieval and ask a model to rerank the top `10-15` down to `k = 3`.

Rerank prompt should use:

- Twin query summary
- each candidate's demographics
- exact PGG rule summary
- oracle archetype text

Instruction:

- prefer candidates with similar transferable social-preference patterns
- use demographics only as secondary evidence
- avoid returning near-duplicate cards
- return the best `3` analogues

### Stage 3: Prediction with retrieved cards

Inject the final top `3` cards into the joint-social Twin prediction prompt.

## Why Not Let The Prediction Call Search Directly

You can let the Responses API `file_search` tool run inside the final prediction call, but I do not recommend that as the first implementation.

Reasons:

- less reproducible than a separate retrieval manifest
- harder to inspect candidate quality
- harder to compare retrieved vs random controls cleanly
- harder to cache and reuse the same retrieved set across prompt variants

So my recommendation is:

- use OpenAI retrieval for candidate generation
- optionally use an LLM reranker
- keep prediction as a separate, auditable batch step

## Prediction Prompt

Build directly on the current tuned joint-social baseline prompt.

Prompt sections:

1. tuned Twin compact profile
2. retrieved oracle cards
3. held-out joint social target block

Each retrieved card should be formatted as:

### Retrieved PGG analogue `i`

- demographics
- PGG rule summary
- oracle archetype
- caveat:
  - this behavior was observed in a repeated public-goods game under the above rules; use it as analogical evidence rather than as an identity match

Prompt instruction should say:

- use the Twin profile as the primary evidence
- use retrieved PGG cases as analogical evidence
- do not copy them literally
- where retrieved cases disagree, prefer the Twin evidence

## Evaluation Conditions

Minimum first-pass comparison:

1. `Twin-only tuned baseline`
2. `Twin + random oracle cards`
3. `Twin + retrieved oracle cards`

Recommended second-pass ablations:

4. retrieved oracle cards without demographics in the library card
5. retrieved oracle cards without rule summary
6. retrieval API only vs retrieval API + LLM rerank
7. `k = 1` vs `k = 3` vs `k = 5`

Metrics stay the same:

- normalized accuracy
- exact match
- MAD
- task-level and QID-level figures

## Recommended Defaults

These are the defaults I would implement first:

- benchmark unit: one request per Twin participant, joint target block
- library: all completed oracle rows from learn + val
- missing demographics: keep row, mark unknown
- query source: LLM-written retrieval query from the tuned Twin compact structured summary
- transcript on PGG side: no
- retrieval cards: include demographics + rule summary + oracle archetype
- retrieval backend: OpenAI Retrieval API over vector stores
- `rewrite_query = true`
- retrieval top-k after rerank: `3`
- shortlist size before rerank: `30`
- reranking: LLM rerank over top candidates with diversity preference

## Proposed Implementation Files

### Data / library build

- `non-PGG_generalization/pgg_transfer_eval/build_oracle_library.py`
  - union learn + val oracle files
  - filter to completed games
  - join demographics
  - derive rule summary from config
  - write library JSONL

### Retrieval indexing

- `non-PGG_generalization/pgg_transfer_eval/build_oracle_vector_store_files.py`
  - write one retrieval file per oracle card plus metadata attributes

- `non-PGG_generalization/pgg_transfer_eval/retrieve_oracle_candidates.py`
  - build Twin retrieval-query JSON with an LLM
  - call OpenAI Retrieval API for top `N`
  - rerank to top `k`
  - write candidate manifest

### Prediction

- `non-PGG_generalization/pgg_transfer_eval/build_joint_social_oracle_augmented_batch.py`
  - reuse tuned Twin baseline prompt
  - inject retrieved oracle cards
  - emit OpenAI Batch JSONL

### Evaluation

- reuse:
  - `evaluate_batch_results.py`
  - `compute_joint_social_comparison_baselines.py`

## Open Decisions

These are the only decisions I think are worth locking before implementation:

1. Retrieval granularity
   - recommended default: one retrieved set per participant for the whole joint target block
   - alternative: family-specific retrieval for trust / ultimatum / dictator separately

2. Library completeness vs demographic completeness
   - recommended default: keep all `7308` completed rows and mark missing demographics as unknown
   - alternative: drop the `23` completed rows without demographics

3. Top-k
   - recommended default: `k = 3`
   - alternative: `k = 5` if you want more diversity at the cost of a longer prompt

## My Recommendation

Implement the simplest defensible version first:

- one retrieval query per Twin participant
- tuned Twin baseline prompt as-is
- completed learn+val oracle library
- demographics + exact PGG rule summary + oracle archetype in each retrieved card
- OpenAI Retrieval API shortlist
- LLM rerank to top `3`

That gives you an LLM-assisted retrieval stage plus a separate prediction stage, which is cleaner than either:

- naive local cosine retrieval, or
- letting the final prediction call search the whole library implicitly.
