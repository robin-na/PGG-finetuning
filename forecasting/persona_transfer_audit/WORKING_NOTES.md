# Persona Transfer Evaluation Working Notes

Last updated: 2026-05-20

This memo records the current state of the Twin-to-target-game persona transfer project. It is intentionally rough. The goal is to preserve the argument, design decisions, results, caveats, and next steps so that the cleaner manuscript can be revised from a stable record.

## Core Idea

We are evaluating whether a persona library that looks diverse in its source domain remains behaviorally diverse after an LLM maps it into a target social interaction.

The motivating case is Twin -> PGG and Twin -> chip bargaining. A Twin persona summary may contain rich information about a person, but the target game behavior is not directly observed in the Twin source data. When researchers use a persona prompt to simulate PGG or bargaining behavior, the LLM implicitly decides how that source-domain profile maps into target-game behavior.

The evaluation makes this transfer step visible. Instead of asking the model to generate new actions, we give it a persona and a real human game transcript, then ask which observed player trajectory most closely matches the persona. Repeating this over personas and games gives a distribution over real human trajectories selected by the persona-library-plus-model system.

The central claim:

> Persona libraries should be evaluated by their revealed-behavior coverage after LLM-mediated transfer, not only by their source-population diversity or aggregate rollout accuracy.

Or more compactly:

> A persona library can be diverse in source space but collapse in target-action space.

## Working Title And Narrative Direction

Possible title:

> Diverse Personas, Narrow Behaviors: Revealed-Behavior Collapse in LLM Social Simulation

Other candidate titles:

- Persona Prompting Does Not Guarantee Behavioral Diversity
- Which Human Would the Model Have Been?
- When Personas Collapse: Revealed-Behavior Audits of LLM Social Simulation
- Diverse Personas, Narrow Behavioral Support

The punchier title "Persona-Prompted Large Language Models Collapse Into Narrow Behavior" captures the core intuition, but it risks sounding universal before we test more persona libraries, target games, and models. "Diverse Personas, Narrow Behaviors" is safer and still direct.

The broader substantive claim is stronger than "we built a benchmark":

> When LLMs are asked to inhabit a persona, they may identify with a skewed and narrowed subset of real human behavior. Persona prompting can increase apparent diversity in textual descriptions while failing to diversify the real behavioral trajectories the model treats as plausible in a target social environment.

This lets the paper speak to simulation, but also to persona prompting as a general steering method. The central question becomes:

> Does a persona prompt change the model's behavioral policy in the way users expect once the model is placed in a complex social setting?

The unprompted/default model condition is especially important. If a default "helpful assistant" or generic "human participant" prompt disproportionately matches cooperative, articulate, polite, high-welfare, low-conflict, or norm-following players, then persona prompting can be evaluated as an attempted intervention on that default behavioral prior. The key empirical question is whether diverse personas actually diversify selected real human trajectories, or mostly decorate the same underlying assistant-aligned behavioral type.

## Why This Is Different From Simulation Evaluation

The standard simulation evaluation asks whether generated aggregate outcomes match human aggregate outcomes. That is useful, but it mixes multiple error sources:

1. the persona library may not contain the right behavioral types;
2. the LLM may map the persona library to a narrow or skewed subset of target behaviors;
3. the LLM may generate miscalibrated actions even if the persona is informative.

Our matching task removes much of the action-generation burden. The model does not have to invent a contribution path, punishment pattern, reward policy, bargaining offer, acceptance rule, or communication style. It only chooses among human trajectories that actually happened.

This makes the evaluation an optimistic test. If persona prompting fails even when the candidate behaviors are real human behaviors, then full generative simulation is unlikely to recover the target distribution without additional correction.

An important point from discussion: LLM bias in mapping personas to target-game behavior is not a nuisance to isolate away. It is part of the system researchers would actually use when they run persona-prompted simulations. We can vary the matcher model to test robustness, but the LLM-mediated mapping itself is the object of diagnosis.

## Prompt And Design Decisions

Current prompt stance:

- The model behaves as a person with the given profile.
- It identifies which player in the provided social interaction most closely matches its personality.
- The system prompt is simple and does not include evaluation jargon.
- The user prompt starts with "Below is information about yourself."
- Persona IDs, game IDs, validation labels, treatment metadata, and batch metadata are kept outside the prompt.
- The social interaction script describes the target game in ordinary terms before presenting the transcript.

Important formatting decision:

- Player labels are not shuffled. Avatar/player names are arbitrary across games, but they matter within each transcript because real participants refer to one another by those labels in communication.

Response format:

- We use `top_k=3` for both PGG and chip bargaining.
- The model returns a sparse probability distribution over the top matching players.
- Probabilities over the listed top-k players sum to 1.
- Unlisted players are treated as probability 0.

Reason for top-k instead of full ranking:

- Some PGG games have many players, making exhaustive ranking and full probability assignment tedious and error-prone.
- Top-k captures the model's most behaviorally relevant matches without forcing arbitrary distinctions among distant candidates.
- The risk is that uncertainty over many plausible players is compressed into three names. Larger top-k or full distributions remain possible robustness checks.

## Data And Implemented Conditions

Persona source:

- Twin direct persona summaries from the SimBench-style cache.
- We deliberately use the direct Twin summary, not the PGG-specialized persona card.
- The goal is to evaluate transfer from source persona evidence into target-game behavior, not a hand-translated target-domain card.

PGG target:

- Validation-wave public goods games.
- Lean scaled batch: 32 Twin personas x 40 PGG games, top-k 3.
- The 40 games are selected across validation treatments/configurations.
- Main metadata directory: `forecasting/persona_transfer_audit/metadata/twin_direct_summary_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2/`

Chip bargaining target:

- Three-player chip bargaining games.
- Scaled batch: 32 Twin personas x 48 chip games, top-k 3.
- Because chip games have exactly three players, top-k 3 is also a complete probability distribution over all players in each game.
- Main metadata directory: `forecasting/persona_transfer_audit/metadata/twin_direct_summary_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2/`

No-persona controls:

- Built on 2026-05-13.
- These use the same target games as the scaled Twin runs but remove the persona block entirely.
- System prompt: "You are an online participant of an experiment. Identify which player in the provided social interaction matches most closely with how you would behave."
- User prompt starts directly with the social interaction script and asks which observed behavior the model would most plausibly have produced.
- PGG run:
  - run name: `no_persona_to_pgg_stratified_40_top3_gpt_5_mini_seed_2`
  - requests: 40
  - batch input: `forecasting/persona_transfer_audit/batch_input/no_persona_to_pgg_stratified_40_top3_gpt_5_mini_seed_2.jsonl`
  - metadata: `forecasting/persona_transfer_audit/metadata/no_persona_to_pgg_stratified_40_top3_gpt_5_mini_seed_2/`
  - tiktoken input tokens: 95,871 total; mean 2,396.8; min 746; max 7,868.
  - OpenAI batch id: `batch_6a0489bc2838819098def0d7ea85d401`
  - status: completed and downloaded.
  - parse summary: 40 parsed, 0 errors, 0 probability-sum errors, 0 duplicate-player errors.
  - key matched-minus-candidate-uniform behavior differences:
    - mean contribution rate: +0.105;
    - first contribution rate: +0.106;
    - final contribution rate: +0.129;
    - full contribution rate: +0.109;
    - zero contribution rate: -0.073;
    - contribution standard deviation: -1.400;
    - messages per round: +0.065;
    - reward-given round rate: +0.077;
    - punish-given round rate: +0.027;
    - punish-received round rate: -0.036.
  - Interpretation: the no-persona/default model's affinity distribution is at least as skewed toward cooperative, communicative, reward-giving, and low-free-riding PGG trajectories as the Twin-persona condition, and often more so. This supports the idea that persona prompting is operating on top of a strong default behavioral prior.
  - Coverage caveat: because this control has only one request per game, top-1 identity coverage and within-game top-player share are not directly comparable to the 32-persona condition. Behavioral skew and probability-weighted top-k behavior are the cleaner comparisons.
  - Paired same-game comparison against the Twin-persona condition:
    - output: `no_persona_vs_twin_paired_game_behavior_skew.csv`
    - no-persona skews are larger in magnitude than Twin skews for all eight core PGG metrics checked.
    - no-persona minus Twin:
      - mean contribution rate: +0.028, game-bootstrap CI [+0.006, +0.051];
      - full contribution rate: +0.020, CI [-0.020, +0.056];
      - zero contribution rate: -0.021, CI [-0.040, -0.006];
      - contribution standard deviation: -0.258, CI [-0.796, +0.273];
      - messages per round: +0.023, CI [-0.003, +0.056];
      - reward-given round rate: +0.041, CI [+0.008, +0.077];
      - punish-given round rate: +0.015, CI [-0.009, +0.041];
      - punish-received round rate: -0.014, CI [-0.025, -0.004].
    - Interpretation: in PGG, Twin personas appear to attenuate the default model's cooperative/normative affinity, especially for contribution/free-riding and punishment received. However, the Twin condition remains skewed relative to candidate-uniform human behavior, so the defensible claim is attenuation rather than debiasing.
- Chip run:
  - run name: `no_persona_to_chip_bargain_stratified_48_top3_gpt_5_mini_seed_2`
  - requests: 48
  - batch input: `forecasting/persona_transfer_audit/batch_input/no_persona_to_chip_bargain_stratified_48_top3_gpt_5_mini_seed_2.jsonl`
  - metadata: `forecasting/persona_transfer_audit/metadata/no_persona_to_chip_bargain_stratified_48_top3_gpt_5_mini_seed_2/`
  - tiktoken input tokens: 76,231 total; mean 1,588.1; min 1,387; max 1,774.
  - OpenAI batch id: `batch_6a0489bc315c81908e8da977a1f073df`
  - status: completed and downloaded.
  - parse summary: 48 parsed, 0 errors, 0 probability-sum errors, 0 duplicate-player errors. One request returned only two top matches, so there are 143 matched rows rather than 144; this is valid under the top-k schema and the unlisted player receives probability 0.
  - outputs:
    - `chip_no_persona_significance_checks.csv`
    - `no_persona_vs_twin_paired_record_behavior_skew.csv`
  - key record-level matched-minus-candidate-uniform behavior differences:
    - final surplus: +0.481, record-bootstrap CI [+0.248, +0.723];
    - final welfare: +1.053, CI [+0.612, +1.498];
    - proposer mean net surplus: +0.037, CI [-0.067, +0.138];
    - proposer acceptance rate: +0.048, CI [+0.014, +0.083];
    - proposer mean trade ratio: -0.003, CI [-0.046, +0.036];
    - response acceptance rate: -0.026, CI [-0.045, -0.008];
    - responder net surplus if accepted: +0.113, CI [+0.012, +0.209];
    - received trade rate: -0.026, CI [-0.043, -0.009].
  - Paired same-record comparison against the Twin-persona condition:
    - no-persona minus Twin:
      - final surplus: +0.191, CI [-0.007, +0.410];
      - final welfare: +0.551, CI [+0.178, +0.925];
      - proposer mean net surplus: +0.103, CI [+0.004, +0.216];
      - proposer acceptance rate: -0.020, CI [-0.052, +0.013];
      - proposer mean trade ratio: -0.017, CI [-0.069, +0.028];
      - response acceptance rate: -0.017, CI [-0.037, +0.003];
      - responder net surplus if accepted: +0.089, CI [-0.017, +0.186];
      - received trade rate: -0.027, CI [-0.047, -0.008].
  - Interpretation:
    - Chip does not follow the simple PGG pattern where Twin uniformly attenuates a cooperative default prior.
    - The no-persona default is more skewed toward high realized welfare/surplus and higher proposer net surplus than Twin.
    - Twin remains more skewed toward accepted proposals and lower proposer surplus, consistent with an accommodating or mutually acceptable bargaining style.
    - The broader point is that no-persona defaults are already non-neutral, but persona prompting changes the kind of skew rather than simply increasing or decreasing a single bias dimension.
- Interpretation:
  - This is the default model affinity baseline. It tells us which real trajectories the model identifies with when no external persona is provided.
  - Comparing Twin-persona matches against this baseline will show whether persona prompting broadens behavioral coverage beyond the model's default behavioral prior or mostly preserves the same attractor.

Demographic and survey-conditioned baselines:

- The Argyle et al. "silicon sampling" design did not merely prompt the model with abstract demographic categories. It used real survey respondent profiles from public-opinion datasets, then asked the model to answer the same survey items as those respondents. For our purposes, the closest analogue is an external source-library transfer test: sample real survey respondent records from the Argyle replication data and ask whether those survey-conditioned personas map onto the empirical behavior distribution in PGG, chip bargaining, or later CaSiNo.
- We should therefore distinguish three increasingly informative baselines:
  - demographic-only: age, gender/sex, education, nationality/country, employment, and other standard respondent fields;
  - Argyle-style survey-conditioned: demographics plus real public-opinion survey responses or attitudinal variables from an external source respondent, excluding any target-game behavior;
  - rich persona-summary: Twin-style narrative summaries or persona-generator outputs that may encode broader life history, preferences, and self-description.
- The survey-conditioned baseline is especially useful because it asks whether actual human survey traces provide enough behavioral support to improve revealed-behavior matching in strategic games. If survey-conditioned personas still map to skewed PGG or bargaining trajectories, the result speaks directly to the limits of using survey-based or opinion-based personas for revealed-preference social simulation.
- We should not use the actual PGG or chip-bargaining participant profiles as persona inputs for this baseline. The point is to do the same kind of transfer test as Twin, but from a different external persona library. This keeps the design clean: source personas come from Twin, Argyle-style survey records, or another persona generator/library; target behaviors come from PGG, chip bargaining, or another revealed-behavior game.
- We should still be careful about leakage within each external source. Argyle-style prompts should use only source-domain demographics and survey/attitudinal responses, not any behavior from the target game. Twin prompts should use only source-domain persona information unless we are explicitly testing a target-specialized card as a separate condition.

Implemented Argyle ANES 2016 backstory condition:

- Built on 2026-05-13.
- Source data: Argyle et al. replication Dataverse, DOI `10.7910/DVN/JPV20K`.
- Downloaded local source files:
  - `forecasting/persona_transfer_audit/external/argyle_out_of_one_many/Master_ReadMe.txt`
  - `forecasting/persona_transfer_audit/external/argyle_out_of_one_many/Study2Python.py`
  - `forecasting/persona_transfer_audit/external/argyle_out_of_one_many/full_results_2012_2.tab`
  - `forecasting/persona_transfer_audit/external/argyle_out_of_one_many/full_results_2016_2.tab`
  - `forecasting/persona_transfer_audit/external/argyle_out_of_one_many/full_results_2020_2.tab`
- Current batch builder: `forecasting/persona_transfer_audit/build_argyle_to_targets.py`
- Initial implemented source library: ANES 2016 complete-case first-person backstories, using the same fields and templates exposed in the Argyle Study 2 replication code:
  - age, gender, race/ethnicity, state, party identification, ideology, political interest, discussion of politics, church attendance, and affect toward the American flag.
  - The prose now follows the Argyle replication code's original template order: race/ethnicity, politics discussion, ideology, party identification, church attendance, age, gender, political interest, flag affect, and state.
  - GPT-predicted vote probabilities and target election outcomes are not included in prompts.
  - Target-game participant profiles from PGG/chip are not used.
- Eligible source personas with all 10 backstory statements: 2,634.
- Sampled personas: 32, seed 2, same count as the scaled Twin condition.
- Prompt format:
  - system prompt matches the persona condition used for Twin;
  - user prompt starts with "Below is information about yourself.";
  - source respondent IDs stay only in `custom_id` and manifest rows, not in the prompt.
- PGG batch:
  - run name: `argyle_anes2016_backstory_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2`
  - requests: 1,280
  - batch input: `forecasting/persona_transfer_audit/batch_input/argyle_anes2016_backstory_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2.jsonl`
  - metadata: `forecasting/persona_transfer_audit/metadata/argyle_anes2016_backstory_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2/`
  - tiktoken input tokens: 3,178,792 total; mean 2,483.4; median 2,141.5; min 829; max 7,961.
  - OpenAI batch id: `batch_6a0496ee7e7c8190966db7def3b98f2d`
  - submitted: 2026-05-13; status: completed and downloaded.
  - parse summary: 1,280 parsed, 0 errors, 0 probability-sum errors, 0 duplicate-player errors.
  - matched rows: 3,789 rather than 3,840 because some requests returned fewer than three top matches; unlisted players receive probability 0.
  - key matched-minus-candidate-uniform behavior differences:
    - mean contribution rate: +0.114, game-cluster CI [+0.083, +0.146];
    - full contribution rate: +0.127, game-cluster CI [+0.089, +0.165];
    - zero contribution rate: -0.077, game-cluster CI [-0.109, -0.049];
    - contribution standard deviation: -1.577, game-cluster CI [-2.088, -1.080];
    - messages per round: +0.081, game-cluster CI [+0.037, +0.135];
    - reward-given round rate: +0.039, game-cluster CI [-0.001, +0.082];
    - punish-given round rate: +0.015, game-cluster CI [-0.001, +0.033];
    - punish-received round rate: -0.030, game-cluster CI [-0.049, -0.013].
  - coverage/collapse:
    - top-1 selected identities: 152 of 342 candidate player identities (44.4%);
    - top-1 entropy effective identities: 96.6 of 342 (28.2%);
    - probability-weighted top-k effective identities: 151.5 of 342 (44.3%);
    - median within-game modal top-1 share: 59.4%;
    - median within-game top-1 effective N / players: 35.7%.
  - request-conditional global identity-collapse test:
    - selected identity share observed 44.4% vs null mean 93.6%, p < 0.0001;
    - entropy effective identity share observed 0.282 vs null mean 0.751, p < 0.0001;
    - never-selected identity share observed 55.6% vs null mean 6.4%, p < 0.0001;
    - top 5% of identities receive 35.5% of top-1 selections vs null mean 16.1%, p < 0.0001.
  - Interpretation:
    - In PGG, the sparse Argyle/ANES backstory condition is strongly skewed toward cooperative, full-contribution, lower-variance, more communicative, and less punished trajectories.
    - Compared with Twin, Argyle is more cooperatively skewed on every core contribution metric checked. Compared with no-persona, Argyle is also slightly more skewed on mean contribution, full contribution, zero contribution, contribution variance, and messages. This suggests that survey-demographic backstories can amplify the model's cooperative/default affinity in PGG rather than attenuate it.
- Chip-bargaining batch:
  - run name: `argyle_anes2016_backstory_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2`
  - requests: 1,536
  - batch input: `forecasting/persona_transfer_audit/batch_input/argyle_anes2016_backstory_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2.jsonl`
  - metadata: `forecasting/persona_transfer_audit/metadata/argyle_anes2016_backstory_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2/`
  - tiktoken input tokens: 2,572,496 total; mean 1,674.8; median 1,671.0; min 1,470; max 1,867.
  - OpenAI batch id: `batch_6a0496ee6ab88190a3f60a405fb9f419`
  - submitted: 2026-05-13; status: completed and downloaded.
  - parse summary: 1,536 parsed, 0 errors, 0 probability-sum errors, 0 duplicate-player errors.
  - matched rows: 4,605; candidate rows: 4,608. Three requests returned only two top matches, so one unlisted player in each receives probability 0.
  - key matched-minus-candidate-uniform behavior differences:
    - final surplus: +0.253, record-cluster CI [+0.097, +0.429];
    - final welfare: +0.418, record-cluster CI [+0.092, +0.770];
    - proposer mean net surplus: -0.096, record-cluster CI [-0.169, -0.031];
    - proposer acceptance rate: +0.047, record-cluster CI [+0.020, +0.074];
    - proposer mean trade ratio: +0.003, record-cluster CI [-0.017, +0.022];
    - response acceptance rate: -0.0003, record-cluster CI [-0.016, +0.016];
    - responder net surplus if accepted: +0.040, record-cluster CI [-0.029, +0.079];
    - received trade rate: +0.006, record-cluster CI [-0.008, +0.021].
  - coverage/collapse:
    - top-1 selected identities: 132 of 144 candidate player identities (91.7%);
    - top-1 entropy effective identities: 98.8 of 144 (68.6%);
    - probability-weighted top-k effective identities: 129.6 of 144 (90.0%);
    - median within-game modal top-1 share: 65.6%.
  - request-conditional global identity-collapse test:
    - entropy effective identity share observed 0.686 vs null mean 0.968, p < 0.0001;
    - never-selected identity share observed 0.083 vs null mean approximately 0, p < 0.0001;
    - top 5% of identities receive 15.4% of top-1 selections vs null mean 8.4%, p < 0.0001.
  - Interpretation:
    - The sparse Argyle/ANES backstory condition is not neutral in chip bargaining. It selects higher-surplus/higher-welfare trajectories, lower proposer-surplus trajectories, and proposals that are more likely to be accepted.
    - Compared with Twin, Argyle is directionally similar on chip bargaining but somewhat weaker for final surplus/welfare and proposal acceptance; it is more negative on proposer net surplus.

## Evaluation Metrics

We currently analyze three related phenomena.

### 1. Behavioral Skew

Compare matched players against a candidate-uniform baseline within the same shown games.

For each persona-game request, the matched distribution gives weights to selected players. The candidate-uniform baseline gives equal weight to all candidate players shown in that request. The difference estimates what kinds of real human trajectories the persona-library-plus-model system preferentially selects.

PGG behavioral metrics:

- mean contribution rate;
- first contribution rate;
- final contribution rate;
- zero contribution rate;
- full contribution rate;
- contribution variability;
- contribution slope;
- message frequency;
- reward giving and receiving;
- punishment giving and receiving.

Chip behavioral metrics:

- final surplus;
- final welfare;
- proposer mean net surplus;
- proposer acceptance rate;
- proposer mean trade ratio;
- response acceptance rate;
- response surplus conditional on accepted offers;
- received trade rate.

### 2. Local Coverage And Collapse

For each game, ask whether top-1 selections are evenly distributed across the players in that game. This avoids confusing arbitrary avatar labels, because each game's labels are only interpreted within that game.

Useful quantities:

- modal top-1 share within game;
- entropy effective number of selected players;
- goodness-of-fit against a uniform distribution over the players in that game;
- scatter of effective selected share against number of players, especially for PGG where player count varies.

Important clarification:

- Raw modal share is hard to compare across PGG and chip because chip has exactly three players, while PGG has variable and often larger groups.
- The cleaner test asks whether the within-game top-1 distribution is distinguishable from uniform conditional on the number of candidate players.

### 3. Global Identity Collapse

Across all requests, ask whether top-1 matches concentrate on fewer observed player identities than expected under random choice from the same candidate sets.

This is a global identity-collapse test. It does not require defining persona similarity. It asks whether the realized support of selected human trajectories is too small relative to a null in which the same persona-game requests choose uniformly among their available candidates.

Metrics:

- number and share of candidate identities ever selected top-1;
- number and share never selected top-1;
- entropy effective number of selected identities;
- Simpson effective number;
- HHI;
- Gini;
- share of top-1 mass captured by the most selected 1%, 5%, or 10% of observed identities.

## Main Persona Matching Results So Far

### PGG Behavioral Skew

In the 32 x 40 top-k 3 PGG batch, matched players are substantially more cooperative and less variable than candidate-uniform players.

Matched minus candidate-uniform:

- mean contribution rate: +0.077;
- first contribution rate: +0.083;
- final contribution rate: +0.084;
- full contribution rate: +0.089;
- zero contribution rate: -0.052;
- contribution standard deviation: -1.142;
- messages per round: +0.042;
- reward-given round rate: +0.036;
- punish-given round rate: +0.011, weaker and cluster-sensitive;
- punish-received round rate: -0.022.

Significance checks:

- Most behavioral-skew estimates are robust under bootstrap and clustering by game or persona.
- Punish-given round rate is weaker; game-cluster interval includes zero, while persona-cluster interval is slightly positive.

Substantive interpretation:

- The Twin direct summaries, as interpreted by the matcher LLM, are attracted toward high-contribution, low-free-riding, communicative, reward-giving, and less-punished PGG players.
- This is not merely a label artifact: labels cannot be shuffled because communication uses labels, and the behavioral metrics are computed from real human trajectories.

### PGG Coverage And Collapse

Local and global collapse are present.

From the global identity-collapse summary:

- candidate observed player identities: 342;
- top-1 selected identities: 200;
- top-1 selected identity share: 0.585;
- null selected identity share: about 0.936;
- never-selected identity share: 0.415 versus null about 0.064;
- entropy effective selected identities: 131.6;
- null entropy effective identities: about 256.9;
- observed entropy effective share: 0.385 of candidates;
- HHI and Gini are much higher than the null;
- collapse-tail p-values are at the simulation floor for the main collapse metrics.

Interpretation:

- The matching distribution does not merely deviate in behavior. It also covers too small a subset of observed PGG identities relative to a candidate-uniform null.

### PGG Demographic Alignment

We can also evaluate whether the human trajectories selected by the persona-LLM matcher are demographically aligned with the target experiment population. This is useful because persona transfer may fail not only behaviorally but also demographically: the selected behavioral support may overrepresent some demographic groups relative to the real candidate players shown in the game, or relative to the intended persona population.

First-pass PGG analysis:

- Script: `forecasting/persona_transfer_audit/demographic_alignment_pgg.py`
- Target metadata: `forecasting/persona_transfer_audit/metadata/twin_direct_summary_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2/`
- Demographic source: `demographics/merged_demographcs_prolific.csv`
- Coverage: all 342 unique candidate game-player identities in the 40-game PGG evaluation set join to the richer Prolific merge.
- The analysis compares four distributions:
  - candidate-uniform: each shown player receives equal weight within each persona-game request;
  - matched probability: the model's top-k probability mass over selected players;
  - matched top-1: the model's highest-probability selected player;
  - unique human candidate: each unique shown human player receives weight 1, mainly as a descriptive reference.
- One of 1,280 persona-game requests has no parsed match in the current output, so the matched distributions have total usable weight 1,279.

Outputs:

- `pgg_demographic_join_summary.json`
- `pgg_demographic_weighted_rows.csv`
- `pgg_demographic_categorical_distributions.csv`
- `pgg_demographic_numeric_summaries.csv`
- `pgg_demographic_alignment_summary.csv`
- `pgg_demographic_request_level_differences.csv`
- `pgg_demographic_cluster_significance.csv`
- `pgg_demographic_behavior_associations.csv`
- `forecasting/persona_transfer_audit/figures/figure_pgg_demographic_skew.png`
- `forecasting/persona_transfer_audit/figures/figure_pgg_demographic_skew.pdf`

Main descriptive shifts, matched probability minus candidate-uniform:

- self-reported gender: Man +3.6 percentage points, Woman -3.6 percentage points;
- Prolific sex: Man +4.8 points, Woman -5.1 points;
- self-reported education: High school -3.0 points, Master +2.1 points, Other +1.9 points, Bachelor -0.8 points;
- country of residence: United States +2.6 points, United Kingdom -2.5 points;
- nationality: United States +2.5 points, United Kingdom -0.8 points;
- age: essentially flat, about +0.14 years by self-report and +0.05 years by Prolific age;
- Prolific total approvals: +146 approvals.

Top-1 shifts are larger:

- self-reported gender: Man +5.4 points, Woman -4.8 points;
- Prolific sex: Man +6.9 points, Woman -7.2 points;
- education: High school -5.7 points, Master +2.2 points, Other +3.0 points;
- nationality: United States +4.3 points;
- country of residence: United States +3.7 points, United Kingdom -3.8 points;
- age: about +0.7 years;
- Prolific total approvals: +230 approvals.

Request-level bootstrap intervals suggest that the clearest probability-weighted demographic shifts are gender/sex, high-school education, U.S. country of residence/nationality, part-time employment, and total Prolific approvals. Age is not a clear shift in the probability-weighted distribution.

More conservative clustered check:

- Output: `pgg_demographic_cluster_significance.csv`
- Request-level bootstrap treats 1,279 matched persona-game requests as independent. That is too optimistic because the same 32 personas and 40 games are repeatedly used.
- A stricter check bootstraps by game, by persona, and by crossed game/persona cells.
- Under this stricter interpretation, most demographic shifts are directionally consistent but not robustly significant once game-level composition is treated as the uncertainty unit.
- The clearest robust shift is employment status: part-time participants are selected less often than the candidate-uniform baseline.
  - Matched probability: -3.6 percentage points; game-cluster 95% CI [-6.1, -1.3], persona-cluster CI [-4.9, -2.3], crossed CI [-6.5, -0.8].
  - Top-1: -4.7 percentage points; game-cluster CI [-7.7, -1.7], persona-cluster CI [-6.7, -2.8], crossed CI [-8.5, -0.9].
- Gender/sex, education, U.S./U.K. residence, nationality, and Prolific approval-count shifts are significant under request-level or persona-cluster bootstrap but not under game-level or crossed clustering.

Interpretation:

- The demographic skew is smaller than the behavioral skew, but it is not zero.
- The strongest current demographic signal is not "older people" but a modest directional shift toward men, away from high-school education, toward U.S.-based participants, and toward more experienced Prolific workers. These shifts should be reported as exploratory unless they survive larger-sample game-level clustering.
- With the current 40-game PGG sample, only the lower selection of part-time participants is robust across game, persona, and crossed clustering.
- This should be interpreted cautiously. These are demographics of selected real human trajectories, not demographics that the model explicitly inferred from the persona.
- The next version should compare selected-player demographics both to the target experiment population and to the intended persona-library demographics. If the persona library is, for example, demographically balanced but the selected human trajectories are not, that would strengthen the claim that LLM-mediated persona transfer can distort both behavioral and demographic support.

Why demographic skew is weaker than behavioral skew:

- One likely reason is that the measured demographic variables are only weakly associated with the PGG behaviors that the matcher strongly selects.
- A direct within-game check subtracts each game's average behavior from each player's behavior, then estimates demographic-group residuals. Output: `pgg_demographic_behavior_associations.csv`.
- The largest standardized demographic-behavior associations are modest:
  - part-time employment: lower mean contribution (-0.071 raw, -0.36 SD), higher zero contribution (+0.060, +0.35 SD), lower full contribution (-0.068, -0.30 SD);
  - other education: higher reward-giving (+0.079, +0.41 SD), but this category is small;
  - high-school education: lower messages per round (-0.035, -0.16 SD).
- These associations are much smaller than the matched behavioral skew itself and often not stable under game-level clustering.
- A rough one-field decomposition suggests that the observed demographic shifts would mechanically explain only a tiny portion of the behavioral skew. For example, the employment-status shift implies about +0.004 mean contribution-rate difference, compared with the observed matched-minus-baseline mean contribution-rate shift of about +0.077.
- Therefore, the current evidence is more consistent with the matcher selecting behavioral trajectories directly, not merely selecting demographic groups that happen to be behaviorally different.

### Chip Bargaining Behavioral Skew

In the 32 x 48 top-k 3 chip batch, matched players are more successful/cooperative in some bargaining senses, but the pattern is not identical across roles.

Matched minus candidate-uniform:

- final surplus: +0.290;
- final welfare: +0.502;
- proposer mean net surplus: -0.066;
- proposer acceptance rate: +0.068;
- proposer mean trade ratio: +0.015;
- response acceptance rate: -0.009;
- response mean net surplus if accepted: about +0.024 in the latest significance file;
- received trade rate: near zero.

Significance checks:

- final surplus and final welfare are positive and robust;
- proposer acceptance is positive and robust;
- proposer net surplus is negative and robust;
- proposer trade ratio is positive but record-cluster interval includes zero;
- response-side effects are smaller and less central.

Interpretation:

- The matcher tends to select players who produce higher realized welfare/surplus and whose own proposals are more likely to be accepted, but whose proposer surplus is lower. This looks like attraction toward agreeable or mutually beneficial bargaining behavior rather than exploitative proposer behavior.

### Chip Coverage And Collapse

Chip shows global identity collapse, but less severe than PGG because each game has only three players and top-k 3 assigns probability to every player.

From the global identity-collapse summary:

- candidate observed player identities: 144;
- top-1 selected identities: 135;
- top-1 selected identity share: 0.938;
- null selected identity share: about 1.000;
- entropy effective selected identities: 105.4;
- null entropy effective identities: about 139.5;
- observed entropy effective share: 0.732 of candidates;
- HHI and Gini are higher than null.

Interpretation:

- Chip top-k probability coverage is complete by design, but top-1 choices still concentrate more than expected under uniform choice.

## Relationship To Existing Rollout Simulations

We checked whether the existing Twin-sampled rollout simulations align with the behavioral skewness patterns. The answer is mixed.

PGG rollout comparisons:

- Existing directories include `forecasting/pgg/results/twin_sampled_seed_0_gpt_5_mini__vs_human_treatments/`, `twin_sampled_unadjusted_seed_0_gpt_5_mini__vs_human_treatments/`, and corresponding gpt-5.1 variants.
- Signed generated-minus-human comparisons show that first-round contribution is slightly high or near zero, but mean and final contribution are lower than human.
- Across the checked PGG Twin runs:
  - mean total contribution rate: about -0.07 to -0.14;
  - final total contribution rate: about -0.25 to -0.31;
  - first-round total contribution rate: about +0.00 to +0.04;
  - within-round contribution variance is lower, consistent with matching skew;
  - rounds with chat is higher, consistent with matching skew;
  - punishment actor rate is lower or near zero, partly consistent;
  - reward actor rate is lower or near zero, not consistent with matching skew.

Interpretation:

- PGG is not a clean mirror. The matching task selects more cooperative human trajectories, but generative rollouts still under-contribute by the end. That suggests persona-transfer skew and action-generation calibration are separable.

Chip rollout comparisons:

- Existing directories include `forecasting/chip_bargain/results/twin_sampled_unadjusted_seed_0_gpt_5_mini__vs_human_treatments/` and multiple gpt-5.1/gpt-5.4 prompt-card variants.
- Rollouts are higher than human on proposer acceptance and trade ratio, and lower on proposer net surplus.
- This aligns with the chip matching skew on proposer acceptance and lower proposer surplus.
- However, rollout final surplus and welfare are lower than human, while the matching skew selects players with higher final surplus/welfare.

Interpretation:

- Chip aligns on bargaining style but not on realized welfare.
- A plausible mechanism is that the model is too willing to accept or generate inefficient trades: agreeable behavior need not translate into high welfare.

Manuscript implication:

- The evaluation should not claim that matching skew mechanically predicts rollout error.
- The stronger claim is that it diagnoses a separate layer of the problem: the persona-to-revealed-behavior mapping.
- Where matching skew and rollout bias align, it helps explain simulation errors.
- Where they diverge, it shows that simulation error also depends on the model's action-generation and strategic calibration.

## Anticipated Pushback: "Simulation Can Still Be Useful"

Likely objection:

> Even if a persona cannot be mapped cleanly to one real behavioral trajectory, persona-prompted simulation can still be useful or accurate at the aggregate level.

This objection is valid in a narrow sense and should not be dismissed. There are at least three cases where it can be true:

1. Aggregate cancellation: individual-level mismatches may cancel out, producing reasonable means.
2. Task-level sufficiency: a simulation may be useful for stress testing, qualitative exploration, or hypothesis generation even if it is not person-level accurate.
3. Recalibrated simulation: biased persona mappings can sometimes be corrected by reweighting, calibration, or model choice.

Our response should be:

- We are not arguing that persona prompting is never useful.
- We are arguing that aggregate fit is insufficient evidence that the persona library captures the behavioral support of the target population.
- A simulation can be useful for some purposes while still failing as a model of population heterogeneity, subgroup behavior, policy response, or mechanism-level behavior.
- The intended use determines the validation burden.

Recommended framing:

> Persona prompting should be validated at the level required by the scientific claim. If the claim is only about one aggregate outcome in one environment, aggregate accuracy may be informative. But if the claim is that a persona library represents a human population, transports across environments, or supports counterfactual multi-agent simulation, then researchers need to evaluate whether the library covers the revealed behavioral trajectories in the target environment.

Concrete distinction:

- Aggregate prediction question: Does the simulator reproduce the mean contribution rate or final welfare?
- Behavioral-coverage question: Does the simulator represent the range of observed human strategies, including free riders, conditional cooperators, norm enforcers, quiet participants, failed negotiators, and exploitative bargainers?
- Transport question: Does a source-domain persona library map to the right behavioral support in a new target game?

This paper is primarily about the second and third questions.

## Anticipated Pushback: "The Choice Task Is Too Constrained"

Likely objection:

> By forcing the model to choose among already-observed human trajectories, we may be constraining the task too much. If we simply let the model simulate freely, it might generate diverse enough behavior.

Response:

- The constrained-choice task is intentionally an easier and more diagnostic test. The model does not need to invent behavior; it only needs to recognize which real behaviors are plausible for the prompted persona.
- If a persona-library-plus-model system can freely generate diverse human-like behavior, it should usually also be able to distribute affinity across diverse real human trajectories when those trajectories are shown.
- Free generation can appear diverse for reasons that are not reassuring: stochastic variation, prompt noise, invalid behavior, off-support actions, or numerically diverse but behaviorally unrealistic trajectories.
- Matching among real trajectories anchors the evaluation to the empirical support of the target population. It asks whether the model recognizes the diversity that actually occurred, not whether it can invent variety.
- The matching task and rollout task answer different questions. Rollouts test full action generation and strategic dynamics. Matching tests whether persona-conditioned interpretation maps onto the right part of observed human behavior.

Useful framing:

> The question is not whether unconstrained generation can produce variation. The question is whether persona prompting induces variation over the same behavioral support that humans actually occupy in the target setting.

The most informative empirical pattern would compare four quantities:

1. the diversity of the source persona library;
2. the diversity of real human target trajectories selected in the matching task;
3. the diversity of free generated rollouts;
4. the alignment between generated rollouts and the human target distribution.

If free rollouts are diverse but matching collapses, then generation is adding variability that is not clearly grounded in persona-to-human behavioral affinity. If matching is diverse but rollouts collapse, then action-generation calibration or multi-agent dynamics are the problem. If both collapse, the persona-library-plus-model system is strongly failing behavioral transport.

## Anticipated Pushback: "Humans Might Also Choose Nice-Looking Behavior"

Likely objection:

> If we asked a real human before playing a public goods game which transcript most resembled them, they might also choose cooperative, fair, polite, or socially desirable behavior. Once stakes are real, their actual behavior might differ. Therefore, the model's skew toward nice-looking trajectories may not prove that persona prompting fails.

This is a legitimate design risk. The matching task is not a pure revealed-preference task for the matcher. It asks for identification with already-observed trajectories, and identification can be affected by social desirability, moral preference, self-image, narrative coherence, or the salience of articulate/cooperative communication.

Our response should be careful:

- We should not claim that the selected trajectory is the true psychological match for the persona.
- We should not claim that humans would necessarily select their own future behavior under the same task.
- We should not interpret the matching task as replacing incentivized behavioral prediction.

The stronger justification is different:

1. The task evaluates the exact mapping used by persona-prompted simulation.
   - When a persona-prompted LLM simulates PGG behavior, it must implicitly decide what kinds of target-game behavior are compatible with the persona.
   - The matching task makes this implicit compatibility judgment explicit.
   - If the model maps many diverse personas to the same narrow set of real trajectories, that is a failure mode of the deployed persona-plus-model system, even if humans would also show some self-presentation bias.

2. The task is anchored to real behavioral support.
   - Free simulation can produce apparently diverse outputs that are invalid, off-support, strategically incoherent, or artifacts of sampling noise.
   - Matching asks whether the model recognizes the diversity that actually occurred among humans in the target environment.
   - This is especially useful for evaluating coverage, because every candidate trajectory is empirically possible by construction.

3. The test is optimistic relative to generation.
   - The model does not need to invent a contribution path, punishment policy, bargaining strategy, or communication style.
   - It only needs to assign affinity over completed human trajectories.
   - If coverage collapses even under this easier recognition task, full generative simulation has an additional burden.

4. Social-desirability bias is part of the object of study, not only a confound.
   - Persona prompting is widely used as a steering method, not only as a scientific prediction tool.
   - If the model maps diverse personas toward normatively attractive behavior because of helpfulness, politeness, fairness norms, or self-presentation, that is precisely a deployment-relevant limitation.
   - The result says: persona descriptions may not overcome the model's default behavioral prior in complex social settings.

5. The design can include controls that separate mechanisms.
   - Compare first-person wording ("matches your personality") to third-person wording ("which player would this person most likely be?").
   - Compare "most similar" to "least similar" or ask for both, to see whether the model recognizes dispreferred but plausible behavior.
   - Include no-persona, demographic-only, persona-library, and target-domain oracle persona conditions.
   - Run multiple matcher models to test whether the skew is model-specific.
   - Where possible, benchmark against human raters or against real participants' self-identification with transcripts.
   - Ask models to justify matches using behavioral evidence rather than moral preference, and audit whether rationales mention actual actions versus generic niceness.

Recommended manuscript framing:

> The matching task should not be interpreted as a claim that a human participant would truthfully identify their own future behavior from a transcript. Rather, it is an audit of the behavioral affinity distribution induced by a persona-conditioned model when the possible target behaviors are fixed to real human trajectories. This is the mapping that persona-prompted simulation relies on but normally hides inside generated actions.

Useful sentence:

> The question is not whether the model can predict a person's hidden type from a transcript. The question is whether, after receiving a persona, the model treats the empirically observed range of human behavior as plausible in the proportions required for population simulation or behavioral steering.

## Broader Implications Beyond Social Simulation

The evaluation has implications beyond LLM-based human simulation. The broader issue is whether a natural-language persona is sufficient to steer model behavior in complex situations.

This point matters because persona prompting is not only a prediction tool. It is also a steering tool. Researchers and developers use personas to make models simulate people, represent populations, behave like characters, act as professional roles, adapt to users, or produce diverse outputs for evaluation and red-teaming. The same failure mode can matter in all of these settings: a persona may look diverse in demographics, survey answers, Big Five traits, narrative details, or stated preferences, but that diversity may not survive deployment into a complex behavioral task.

Possible application domains:

- Personalized assistants: A user may specify that the assistant should act like a cautious planner, assertive negotiator, empathetic coach, or skeptical analyst. The model may instead map those descriptions onto generic helpful-assistant behavior or a narrow stereotype.
- AI role-play and character agents: A model may preserve surface style while failing to reproduce the character's actual decision tendencies under pressure, conflict, or strategic tradeoffs.
- Customer-service and sales agents: A persona such as "warm but firm" or "customer advocate" may not determine behavior in hard cases involving refunds, complaints, escalation, or policy conflicts.
- Negotiation and mediation systems: A persona prompt may imply a bargaining style, but the model's default helpfulness, fairness norms, or aversion to conflict may dominate when incentives become explicit.
- Education and tutoring agents: "Socratic tutor," "strict examiner," or "supportive coach" personas may not reliably translate into different feedback behavior across student mistakes, frustration, or disengagement.
- Clinical or mental-health support tools: Persona prompts such as "empathetic therapist" or "motivational interviewer" may shape language style while leaving intervention choices insufficiently grounded in expert behavior.
- Synthetic users for product testing: Persona-generated users may sound demographically diverse but still collapse onto similar preferences, politeness norms, complaint patterns, or task strategies.
- Red-teaming and safety evaluation: Personas intended to elicit adversarial, negligent, confused, or malicious behavior may fail if the model's alignment training maps them back toward cooperative or norm-following responses.
- Organizational decision support: Executive, analyst, regulator, activist, or consumer personas may shift rhetoric more than revealed decision policy in multi-step tradeoffs.

This reframes the project from "persona prompting for human simulation" to a more general steering question:

> Does a persona prompt change the model's behavioral policy in the way the user expects, once the model is placed in a complex social environment?

The matching design is one way to audit that question without relying only on generated text. It asks which real behaviors the model treats as compatible with the prompted persona. If many personas map to the same narrow set of behaviors, then persona prompting may be insufficient for behavioral steering even when it changes surface language.

## Core Sufficiency Question

The key scientific question is whether persona + LLM is sufficient to represent what happens in a complex situation.

For simple survey responses, a short persona may move model outputs in the expected direction. But complex social behavior depends on more than stable traits:

- incentives and payoff structure;
- beliefs about others;
- history and adaptation;
- local norms;
- communication style;
- attention and comprehension;
- mistakes, confusion, fatigue, and noncompliance;
- role-specific constraints;
- interaction dynamics and feedback loops.

A persona summary may not contain enough information to determine these behavioral functions. Even if it does, the LLM may not map that information into the target behavior as intended.

Therefore, the paper can frame persona prompting as a transport problem:

> A persona is not itself a behavioral model. It becomes one only after an LLM maps the persona into actions in a target environment. The validity of persona prompting depends on that mapping.

This is why revealed-behavior matching is useful. It directly probes the mapping from persona descriptions to complex real behaviors while holding the candidate behavior set fixed.

## Literature Positioning And References To Track

The exact design appears to be novel so far: a default or persona-conditioned LLM is asked to choose which real human trajectory in a social interaction it most closely matches, and the selected human trajectories are analyzed as the model's revealed behavioral affinity distribution.

There is closely related work, but it tends to stop one step earlier or evaluate a different object.

### Persona Generation And Population Alignment

These papers are highly relevant because they explicitly recognize that persona support and population alignment matter. The gap is that they mostly evaluate diversity or alignment in persona space, survey space, or psychometric space, not in target-domain revealed behavior.

- Paglieri, Cross, Cunningham, Leibo, and Vezhnevets, "Persona Generators: Generating Diverse Synthetic Personas at Scale" (arXiv:2602.03545). This paper frames the problem as support coverage rather than only density matching, and argues that diverse persona generation matters for evaluating AI systems that interact with heterogeneous users. It is especially useful for our framing because it acknowledges long-tail coverage as important for applications beyond simulation, including evaluation and red-teaming. Our extension is to ask whether support coverage in persona space survives the LLM mapping into real human behavior.
- Li, Chen, Namkoong, and Peng, "LLM Generated Persona is a Promise with a Catch" (NeurIPS 2025 Position Paper Track; arXiv:2503.16527). This paper is a direct precedent for the claim that LLM-generated personas can look useful while producing systematic bias in downstream opinion and election-forecast tasks. Our extension is not merely to document another persona limitation. We change the target and measurement: instead of asking generated personas to answer survey/opinion questions or forecast aggregate outcomes, we ask persona-conditioned models to choose among real human behavioral trajectories in strategic social interactions. This tests the LLM-mediated transport from persona description to revealed behavior while holding the candidate action paths fixed.
- Hu et al., "Population-Aligned Persona Generation for LLM-based Social Simulation" (arXiv:2509.10127). This work generates personas from long-term social media data and aligns them to reference psychometric distributions, including Big Five traits. This is exactly the kind of approach our paper can complement: even if a persona set is aligned on Big Five or other population-level descriptors, we still need to test whether those personas map onto the behavioral support of the target setting.
- PersonaGym, "Evaluating Persona Agents and LLMs" (Samuel et al., arXiv:2407.18416; Findings of EMNLP 2025). PersonaGym evaluates whether persona agents behave consistently across persona-relevant environments and emphasizes that persona adherence is hard to evaluate in free-form settings. This supports our broader claim that persona prompting is a steering problem, not just a prompt-format problem. Our design differs by anchoring the evaluation to real human trajectories in social interactions.

Key distinction to emphasize:

> Persona generation papers ask whether the persona set is diverse or population-aligned. We ask whether those personas, after LLM interpretation, cover the real behavioral support of a target environment.

For readers already familiar with the persona-generation limitation literature, the novelty claim should be framed narrowly and empirically:

- Not novel: persona generation can be biased, stereotyped, homogeneous, or misaligned with real populations.
- Novel: evaluating persona libraries by the distribution of real human trajectories they identify with in a target social environment.
- Novel: separating action-generation calibration from persona-to-behavior transport by forcing the model to choose among behavior that actually occurred.
- Novel: measuring collapse in revealed-behavior space, including behavioral skew, local coverage, global identity concentration, and demographic skew within games.
- Novel: applying this to strategic multi-agent economic games with incentives, communication, adaptation, punishment/reward, and bargaining outcomes rather than only survey or psychometric responses.

### Persona Steering And Opinion Representation

This literature shows that persona prompts can move models but often do not fully solve representation or diversity problems.

- Santurkar et al., "Whose Opinions Do Language Models Reflect?" (arXiv:2303.17548). This paper uses public opinion data to evaluate which demographic groups' opinions are reflected by LMs, and finds substantial misalignment that can persist even with demographic steering. This is useful precedent for treating model outputs as reflecting a distribution over human views rather than as neutral responses.
- Liu, Diab, and Fried, "Evaluating Large Language Model Biases in Persona-Steered Generation" (Findings of ACL 2024). This paper finds that LLMs are less steerable toward incongruous personas and that RLHF models can be less diverse in persona-steered open-ended generation. This is one of the closest references for the idea that persona prompting does not guarantee diversity or faithful steering.
- Li et al., "The steerability of large language models toward data-driven personas" (NAACL 2024). This paper develops data-driven personas based on patterns of opinion and shows improved steerability relative to baselines. It is relevant as a constructive contrast: data-driven personas may improve opinion steering, but our question is whether any persona approach transports into complex revealed behavior.

Key distinction to emphasize:

> Opinion-steering work evaluates whether LLM outputs match distributions of stated opinions. We evaluate whether persona-conditioned models identify with the distribution of real behavior in strategic social interactions.

### LLM Social Simulation And Economic Behavior

This literature motivates why behavioral skew matters.

- Work on LLMs in economic games reports systematic behavioral tendencies such as over-altruism, inequality aversion, helpfulness/fairness bias, rationality bias, or sensitivity to framing and persona. These results show that unprompted or lightly prompted LLMs are not neutral samples from human behavioral populations.
- Our contribution is to test whether persona prompting solves this narrowness problem in a target setting with observed human trajectories. Early evidence says not necessarily: Twin personas map to more cooperative PGG players and to more agreeable/high-welfare chip-bargaining players, with substantial identity concentration.

### Design Gap

The gap can be stated as:

> Prior work has asked whether persona libraries are diverse, whether models can be steered toward personas, and whether generated outputs resemble human surveys or game behavior. Much less work asks whether persona-conditioned models, when shown the actual behavioral support of a target population, distribute themselves across that support in the way required for simulation or behavioral steering.

This lets us frame revealed-behavior matching as both a benchmark and a substantive diagnostic:

- Benchmark role: evaluate persona libraries, models, and prompts against real human trajectory support.
- Scientific role: reveal which kinds of human behavior LLMs treat as compatible with their default or persona-conditioned selves.
- Practical role: diagnose whether persona prompting is sufficient to steer models in complex deployments.

## Future Work And Extensions

### More Persona Libraries

Run the same evaluation on additional persona sources:

- no-persona baseline;
- demographic-only personas;
- Twin direct summaries;
- Twin target-domain cards;
- external persona generators or libraries, including persona-generator papers referenced in the recent literature;
- manually constructed behavioral archetypes;
- oracle or near-oracle target-domain summaries when available.

Key comparison:

- Do richer persona libraries improve target-behavior coverage, or do they still collapse into similar revealed trajectories?

Next persona-library recommendation as of 2026-05-13:

- Highest-priority next run: PersonaHub.
  - Rationale: It is a large, public, widely cited synthetic persona corpus designed to create diverse synthetic data from many perspectives. It is not specifically optimized for revealed social behavior, which makes it a useful "popular generic persona library" contrast after Twin and Argyle.
  - Design: sample the same number of personas as Twin/Argyle first (32), then optionally scale to 128. Use the raw persona text with the same matching prompt and the same 40 PGG / 48 chip target sets.
  - Key test: whether web-scale synthetic persona diversity still maps onto the same cooperative/high-welfare real-player trajectories.
  - Initial exploratory sampling/batch build on 2026-05-13:
    - source: `proj-persona/PersonaHub`, `ElitePersonas/elite_personas.part*.jsonl`;
    - builder: `forecasting/persona_transfer_audit/build_personahub_to_targets.py`;
    - sampled 32 personas with seed 2 from the first 1,000 eligible rows in each of 19 elite part files;
    - filter: 120 to 1,600 persona characters;
    - candidates seen after filter: 18,954;
    - sampled persona length: min 282 chars, median 561, mean 576.8, max 942;
    - sample file: `forecasting/persona_transfer_audit/external/personahub/elite_personas_seed_2_n32.jsonl`;
    - PGG batch: `forecasting/persona_transfer_audit/batch_input/personahub_elite_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2.jsonl`, 1,280 requests, 3,206,432 input tokens by tiktoken;
    - chip batch: `forecasting/persona_transfer_audit/batch_input/personahub_elite_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2.jsonl`, 1,536 requests, 2,605,664 input tokens by tiktoken.
    - Status: do not use this as the main PersonaHub condition because the character-length filter was arbitrary. It is retained only as an exploratory artifact.
  - Revised sampling decision on 2026-05-13:
    - Remove all character-length filters from PersonaHub sampling.
    - Distinguish the official 200k-row `persona.jsonl` subset from the much larger `ElitePersonas` shards.
    - The official `persona.jsonl` file is about 21.6 MB and can be sampled exhaustively and uniformly over all 200,000 rows. This is the cleanest non-arbitrary "random PersonaHub" condition, but the personas are short one-line role/perspective descriptions.
    - Each `ElitePersonas` shard is about 16 GB, with 19 shards total, so a genuinely uniform full ElitePersonas sample would require streaming hundreds of GB or implementing a more elaborate remote sampling scheme. For now, any elite sample should be labeled as a preview-frame sample rather than a random sample from the full elite corpus.
  - Unfiltered length profiles:
    - `persona.jsonl` full 200,000-row subset: mean 91.7 chars, SD 26.9, p01 28, p05 46, p50 92, p95 133, p99 161, max 750.
    - ElitePersonas preview frame, first 1,000 rows from each of 19 shard files: mean 597.6 chars, SD 171.4, p01 221, p05 354, p50 583, p95 889, p99 1099, max 2346.
    - The previous 120-1,600 character filter removed only about 0.24% of the inspected elite-preview rows, but it was still arbitrary and is no longer used.
  - Implemented unfiltered full-`persona.jsonl` batch build on 2026-05-13:
    - sample file: `forecasting/persona_transfer_audit/external/personahub/persona_jsonl_unfiltered_seed_2_n32.jsonl`;
    - sample frame: official `persona.jsonl` 200,000-row subset;
    - length filter: none;
    - sampled persona length: min 31 chars, median 90, mean 90.8, max 173;
    - PGG batch: `forecasting/persona_transfer_audit/batch_input/personahub_persona_jsonl_unfiltered_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2.jsonl`, 1,280 requests, 3,099,392 input tokens by tiktoken;
    - chip batch: `forecasting/persona_transfer_audit/batch_input/personahub_persona_jsonl_unfiltered_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2.jsonl`, 1,536 requests, 2,477,216 input tokens by tiktoken.
  - Implemented unfiltered ElitePersonas preview-frame batch build on 2026-05-13:
    - sample file: `forecasting/persona_transfer_audit/external/personahub/elite_personas_preview_unfiltered_seed_2_n32.jsonl`;
    - sample frame: first 1,000 rows from each of 19 `ElitePersonas` shard files;
    - length filter: none;
    - sampled persona length: min 332 chars, median 583.5, mean 589.0, max 1140;
    - PGG batch: `forecasting/persona_transfer_audit/batch_input/personahub_elite_preview_unfiltered_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2.jsonl`, 1,280 requests, 3,213,152 input tokens by tiktoken;
    - chip batch: `forecasting/persona_transfer_audit/batch_input/personahub_elite_preview_unfiltered_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2.jsonl`, 1,536 requests, 2,613,728 input tokens by tiktoken.
  - Caveat: the sampled elite personas are richer than the basic one-line PersonaHub personas, but they are mostly occupational/domain-expertise profiles (researcher, engineer, specialist, analyst). They are not respondent-like demographic/personality profiles. This makes them a good "synthetic role-perspective" condition, not a replacement for a population-representative persona library.
- Next synthetic-population condition: NVIDIA Nemotron-Personas-USA.
  - Rationale: Nemotron-Personas-USA is a synthetic persona dataset explicitly grounded in real-world demographic and geographic distributions. It is a stronger synthetic-population baseline than basic PersonaHub because it includes structured demographic variables and multiple rich narrative persona fields.
  - Source: `nvidia/Nemotron-Personas-USA` on Hugging Face.
  - Dataset shape as of 2026-05-13: 1,000,000 rows in the default train split; 11 Parquet files; about 2.7 GB compressed; 22 fields plus UUID, including persona text fields and demographics.
  - Design positioning: NVIDIA describes Nemotron-Personas-USA as synthetically generated personas grounded in real-world demographic, geographic, and personality-trait distributions to capture population diversity and richness. The dataset card also frames the dataset as useful for improving diversity in synthetic data, mitigating bias, and preventing model collapse. However, this is diversity in persona/context space, not a validation that the personas cover revealed behavior in strategic social settings such as public goods games, punishment/reward decisions, communication, or bargaining. This distinction is exactly why it is a useful test case for our paper.
  - Important design choice: use the full persona profile, not just demographic information. The current renderer includes structured demographics plus `persona`, `professional_persona`, `cultural_background`, `skills_and_expertise`, `skills_and_expertise_list`, `hobbies_and_interests`, `hobbies_and_interests_list`, `sports_persona`, `arts_persona`, `travel_persona`, `culinary_persona`, and `career_goals_and_ambitions`.
  - Caveat from inspection: the full row can contain internal inconsistencies across domain-specific persona fields. For example, a record's structured location may place the person in one state while sports/travel/arts fields refer to activities in another region. This should not be silently cleaned if the goal is to evaluate the source library as provided, but it is worth noting.
  - Initial prose-rendered batch build on 2026-05-13:
    - builder: `forecasting/persona_transfer_audit/build_nemotron_to_targets.py`;
    - sample file: `forecasting/persona_transfer_audit/external/nemotron/nemotron_full_persona_adult_seed_2_n32.jsonl`;
    - sample method: uniform random row offsets from the Hugging Face dataset-server train split, with rejection sampling for `age >= 18`;
    - reason for adult filter: target studies are online adult-participant experiments, so this is a target-eligibility filter rather than a length or content filter;
    - sampled 32 personas with seed 2; 38 random draws were needed to obtain 32 adult profiles;
    - sampled full-profile length: min 4,596 chars, median 5,547.5, mean 5,554.6, max 6,641;
    - PGG batch: `forecasting/persona_transfer_audit/batch_input/nemotron_full_persona_adult_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2.jsonl`, 1,280 requests, 4,368,552 input tokens by tiktoken;
    - chip batch: `forecasting/persona_transfer_audit/batch_input/nemotron_full_persona_adult_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2.jsonl`, 1,536 requests, 4,000,208 input tokens by tiktoken.
    - Status: exploratory only. This version lightly re-rendered demographics into prose, humanized coded fields, and converted list strings, so it is not the cleanest "directly from the persona library" condition.
  - Revised raw-field batch build on 2026-05-13:
    - sample file: `forecasting/persona_transfer_audit/external/nemotron/nemotron_raw_fields_adult_seed_2_n32.jsonl`;
    - prompt format: raw field dump preserving Nemotron field names and raw values, e.g. `age:`, `sex:`, `occupation:`, `persona:`, `professional_persona:`, `sports_persona:`, etc.;
    - source content/style changes: no LLM rewriting, no invented traits, no humanized occupation/category labels, no converted list formatting. The only added text is the field name preceding each raw value;
    - same seed, dataset, adult eligibility filter, and target games as the prose-rendered build;
    - sampled raw-field profile length: min 4,782 chars, median 5,743.5, mean 5,745.8, max 6,831;
    - PGG batch: `forecasting/persona_transfer_audit/batch_input/nemotron_raw_fields_adult_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2.jsonl`, 1,280 requests, 4,465,792 input tokens by tiktoken;
    - chip batch: `forecasting/persona_transfer_audit/batch_input/nemotron_raw_fields_adult_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2.jsonl`, 1,536 requests, 4,116,896 input tokens by tiktoken.
    - Token count change relative to prose-rendered build: PGG +97,240 tokens (+2.23%); chip +116,688 tokens (+2.92%). This is small enough that the raw-field version should be preferred.
    - Submitted on 2026-05-13:
      - PGG OpenAI batch id: `batch_6a04ac99560c8190b66a123789a63ffa`;
      - chip OpenAI batch id: `batch_6a04ac9912248190accf38885d0b208a`.
    - PGG batch completed and evaluated on 2026-05-13:
      - output: `forecasting/persona_transfer_audit/batch_output/nemotron_raw_fields_adult_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2.jsonl`;
      - parse summary: 1,280 parsed, 0 errors, 0 probability-sum errors, 0 duplicate-player errors;
      - matched-minus-candidate-uniform behavior differences:
        - mean contribution rate: +0.113, game-cluster CI [+0.081, +0.146];
        - full contribution rate: +0.125, game-cluster CI [+0.085, +0.165];
        - zero contribution rate: -0.077, game-cluster CI [-0.109, -0.049];
        - contribution standard deviation: -1.507, game-cluster CI [-2.034, -0.994];
        - messages per round: +0.086, game-cluster CI [+0.041, +0.139];
        - reward-given round rate: +0.065, game-cluster CI [+0.024, +0.111];
        - punish-given round rate: +0.011, game-cluster CI [-0.003, +0.028];
        - punish-received round rate: -0.031, game-cluster CI [-0.051, -0.013].
      - Coverage/collapse:
        - top-1 selected 136/342 candidate player identities (39.8%);
        - top-k probability mass covered 236/342 identities (69.0%);
        - top-1 entropy effective number: 88.3/342 (25.8%);
        - probability-mass entropy effective number: 140.6/342 (41.1%);
        - median within-game top-1 effective N share: 31.8%;
        - median within-game modal top-1 share: 62.5%;
        - global selected-identity share was 0.398 versus null mean 0.936, p<0.0001;
        - global entropy effective share was 0.258 versus null mean 0.751, p<0.0001;
        - top 5% of candidate identities received 38.2% of top-1 selections versus null mean 16.1%, p<0.0001.
      - Demographic skew:
        - demographic join had 0 missing participant profiles;
        - probability-weighted matched players were slightly more male by Prolific sex (+0.025) and slightly less female (-0.030), but gender shifts were smaller than behavioral shifts;
        - education shifted away from self-reported high school (-0.068, game-cluster CI [-0.116, -0.019]) and toward master/other education categories, though only the high-school decrease was clearly separated from zero under the clustered checks.
      - Interpretation: Nemotron raw-field personas map to strongly cooperative, high-contribution, low-free-riding, lower-variance, more communicative, more reward-giving, and less-punished PGG trajectories. The magnitude is very close to Argyle/ANES and notably larger than Twin on contribution/message/reward skews.
    - Chip-bargaining batch completed and evaluated on 2026-05-13:
      - output: `forecasting/persona_transfer_audit/batch_output/nemotron_raw_fields_adult_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2.jsonl`;
      - parse summary: 1,536 parsed, 0 errors, 0 probability-sum errors, 0 duplicate-player errors;
      - matched rows: 4,606 rather than 4,608 because two requests returned only two top matches, which is valid under the "rank up to 3" schema;
      - matched-minus-candidate-uniform behavior differences:
        - final surplus: +0.363, record-cluster CI [+0.171, +0.564];
        - final welfare: +0.612, record-cluster CI [+0.239, +1.013];
        - proposer mean net surplus: -0.087, record-cluster CI [-0.173, -0.007];
        - proposer acceptance rate: +0.066, record-cluster CI [+0.039, +0.093];
        - proposer mean trade ratio: +0.005, record-cluster CI [-0.020, +0.031];
        - response acceptance rate: -0.013, record-cluster CI [-0.029, +0.001];
        - responder net surplus if accepted: +0.051, record-cluster CI [-0.015, +0.117];
        - received trade rate: -0.006, record-cluster CI [-0.019, +0.007].
      - Coverage/collapse:
        - top-1 selected 121/144 candidate player identities (84.0%);
        - top-k probability mass covered 144/144 identities (100%);
        - top-1 entropy effective number: 87.8/144 (61.0%);
        - probability-mass entropy effective number: 126.3/144 (87.7%);
        - median within-game top-1 effective N share: 59.8%;
        - median within-game modal top-1 share: 79.7%;
        - global selected-identity share was 0.840 versus null mean 1.000, p<0.0001;
        - global entropy effective share was 0.610 versus null mean 0.969, p<0.0001;
        - top 5% of candidate identities received 16.1% of top-1 selections versus null mean 8.4%, p<0.0001.
      - Interpretation: Nemotron raw-field personas select higher-surplus and higher-welfare chip-bargaining trajectories, with lower proposer surplus and higher proposer acceptance. This is directionally similar to Twin/Argyle on accepted, lower-proposer-surplus bargaining, but the welfare/surplus skew is stronger than Twin/Argyle and weaker than the no-persona default.
  - Interpretation if run: Nemotron is a particularly relevant test of whether population-structured synthetic personas improve revealed-behavior coverage. If even this condition remains skewed toward cooperative/high-welfare/low-conflict real trajectories, the result directly addresses the claim that more detailed and demographically grounded synthetic personas solve persona-collapse concerns.
- Strongest conceptual next run: Persona Generators (Paglieri et al., arXiv:2602.03545).
  - Rationale: This paper explicitly frames persona generation around support coverage and long-tail diversity, which is close to our claim that source-space diversity may not survive target-action matching.
  - Design: if code or generated personas are available, generate personas for generic human/social-interaction contexts and possibly target-specific contexts such as "online public goods game participant" and "online bargaining participant." Compare generic versus target-context generator outputs.
  - Key test: whether a method designed for support coverage in opinions/preferences also covers revealed strategic behavior.
  - Feasibility check on 2026-05-13:
    - The Concordia repository includes a standalone CLI at `concordia/contrib/persona_generators/generate_personas.py`.
    - The CLI supports `--num_personas`, `--output_path`, `--generator`, `--initial_context`, `--diversity_axes`, `--shared_memories`, `--api_type`, and `--model_name`.
    - Available generators in the CLI are `base` plus `alphaevolve_1` through `alphaevolve_5`.
    - The output JSON has `name`, `characteristics`, `memories`, and `shared_memories`.
    - Concordia supports an OpenAI language-model wrapper through `api_type=openai`; use the bundled Python and install `gdm-concordia[openai]` if/when we run it.
    - Technically this is straightforward: generate 32 personas, render the generated JSON fields directly into our matching prompt, and reuse the existing PGG/chip batch/eval pipeline.
  - Design decision:
    - Generic condition: initial context describes adult online economic-experiment participants. This is more comparable to Twin/Nemotron/Argyle as a transfer test.
    - Target-grounded condition: initial context explicitly describes the PGG or chip-bargaining environment and asks for diversity along target-relevant axes. This is a stronger in-domain upper-bound test: if even target-grounded generated personas collapse, the result is especially compelling.
    - Recommended next step: run both small pilot conditions if cost permits. Use 32 generic economic-experiment personas for the main comparison and 32 target-grounded personas for the "best-case support coverage" condition.
  - Game-grounded setup added on 2026-05-15:
    - Setup document: `forecasting/persona_transfer_audit/CONCORDIA_GAME_GROUNDED_SETUP.md`.
    - Generation wrapper: `forecasting/persona_transfer_audit/generate_concordia_game_personas.py`.
    - Batch builder: `forecasting/persona_transfer_audit/build_concordia_to_targets.py`.
    - PGG config: `forecasting/persona_transfer_audit/concordia_configs/pgg_game_grounded_alphaevolve_5.json`.
    - Chip config: `forecasting/persona_transfer_audit/concordia_configs/chip_bargain_game_grounded_alphaevolve_5.json`.
    - Prepared PGG generation artifacts: `forecasting/persona_transfer_audit/external/concordia/concordia_pgg_game_grounded_alphaevolve_5_n32_gpt_5_mini/`.
    - Prepared chip generation artifacts: `forecasting/persona_transfer_audit/external/concordia/concordia_chip_bargain_game_grounded_alphaevolve_5_n32_gpt_5_mini/`.
    - The PGG context is treatment-general but rule-accurate: repeated group contribution decisions, contribution/withdrawal framing, multiplied-and-redistributed public account, variable endowment/multiplier/group size/round count, all-or-nothing variants, chat, punishment, reward, visibility, and rule attention.
    - The chip context follows the actual game structure: three players, three rounds, one proposal turn per player per round, privately valued colored chips, offer/request proposals, private accept/decline decisions by non-proposers, random partner selection when both responders accept, and no trade if nobody accepts.
    - Current axes are target-relevant rather than generic personality axes. PGG axes include contribution tendency, conditional cooperation, free-riding tolerance, norm enforcement, punishment/reward willingness, chat responsiveness, payoff maximization, endgame sensitivity, forgiveness, and rule attention. Chip axes include fairness, surplus maximization, proposer assertiveness, proposal generosity, acceptance threshold, concession willingness, risk tolerance, trust, reciprocity, competitive claiming, strategic patience, and rule attention.
    - PGG is not yet generated separately for each of the 40 treatment configurations. This is deliberate: the first game-grounded condition tests whether one PGG-family persona library covers the behavioral support across treatment variants. A treatment-specific version should be treated as a more permissive upper-bound condition.
    - The generated Concordia persona JSON will be passed into the matching prompt as raw JSON after "Below is information about yourself." We do not rewrite the generated persona into our own prose.
    - Validation so far: generation artifact creation works without API calls; `build_concordia_to_targets.py` passes a smoke test for PGG and chip using a one-persona fake Concordia JSON in `/tmp`.
    - Runtime setup on 2026-05-15: installed `gdm-concordia[openai]` and `scipy` into `/tmp/concordia_deps`, and cloned `google-deepmind/concordia` into `/tmp/concordia_src`. The GitHub source is required because PyPI `gdm-concordia==2.4.0` does not include `concordia.contrib.persona_generators`. Verified `generate_personas.py --help` and generator registry import with `PYTHONPATH=/tmp/concordia_src:/tmp/concordia_deps`.
    - `OPENAI_API_KEY` is loaded from the repo-root `.api_keys.env` through `repo_env.py`, matching the batch-manager credential path.
    - Temporary Concordia runtime patches were needed for `gpt-5-mini`: removed the duplicate `alphaevolve_5` characteristic-generation block in `/tmp/concordia_src/concordia/contrib/persona_generators/persona_generator_five.py`; forced OpenAI chat-completion calls to `temperature=1.0` in `/tmp/concordia_src/concordia/contrib/language_models/openai/base_gpt_model.py` because `gpt-5-mini` rejects non-default temperatures. The first PGG attempt failed at memory generation before this temperature patch, after generating characteristics but before writing `personas.json`.
    - PGG game-grounded personas generated on 2026-05-15:
      - command condition: `pgg_game_grounded_alphaevolve_5`, model `gpt-5-mini`, generator `alphaevolve_5`, `num_personas=32`;
      - output: `forecasting/persona_transfer_audit/external/concordia/concordia_pgg_game_grounded_alphaevolve_5_n32_gpt_5_mini/personas.json`;
      - verified count: 32 personas;
      - file size: 198 KB;
      - raw JSON persona length: median 5,967 chars, min 5,363, max 7,684;
      - each persona has 3 generated memory/context snippets.
    - Chip-bargaining game-grounded personas generated on 2026-05-15:
      - command condition: `chip_bargain_game_grounded_alphaevolve_5`, model `gpt-5-mini`, generator `alphaevolve_5`, `num_personas=32`;
      - output: `forecasting/persona_transfer_audit/external/concordia/concordia_chip_bargain_game_grounded_alphaevolve_5_n32_gpt_5_mini/personas.json`;
      - verified count: 32 personas;
      - file size: 188 KB;
      - raw JSON persona length: median 5,610 chars, min 5,026, max 7,085;
      - each persona has 3 generated memory/context snippets.
    - Persona-matching batches built on 2026-05-15 with matcher model `gpt-5-mini` and `top_k=3`.
      - The builder renders generated Concordia JSON directly, but strips `characteristics.initial_context` from the prompt payload. That field contains the generation context and generation instruction text, not participant profile content. The source `personas.json` files remain unchanged.
      - Prompt hygiene check: sample prompts do not expose `persona_pid`, `game_id`, `record_id`, `custom_id`, validation labels, source keys, or the stripped `initial_context` field.
      - PGG run:
        - run name: `concordia_pgg_game_grounded_alphaevolve_5_to_pgg_stratified_32x40_top3_gpt_5_mini`;
        - batch input: `forecasting/persona_transfer_audit/batch_input/concordia_pgg_game_grounded_alphaevolve_5_to_pgg_stratified_32x40_top3_gpt_5_mini.jsonl`;
        - metadata: `forecasting/persona_transfer_audit/metadata/concordia_pgg_game_grounded_alphaevolve_5_to_pgg_stratified_32x40_top3_gpt_5_mini/`;
        - requests: 1,280;
        - tiktoken input tokens: 4,257,712 total; mean 3,326.3; median 2,977.0; min 1,579; max 9,100.
      - Chip run:
        - run name: `concordia_chip_bargain_game_grounded_alphaevolve_5_to_chip_bargain_stratified_32x48_top3_gpt_5_mini`;
        - batch input: `forecasting/persona_transfer_audit/batch_input/concordia_chip_bargain_game_grounded_alphaevolve_5_to_chip_bargain_stratified_32x48_top3_gpt_5_mini.jsonl`;
        - metadata: `forecasting/persona_transfer_audit/metadata/concordia_chip_bargain_game_grounded_alphaevolve_5_to_chip_bargain_stratified_32x48_top3_gpt_5_mini/`;
        - requests: 1,536;
        - tiktoken input tokens: 3,838,112 total; mean 2,498.8; median 2,491.0; min 2,171; max 2,936.
- Best social-simulation foil: Population-Aligned Persona Generation for LLM-based Social Simulation.
  - Rationale: This is directly in the social-simulation lane and aligns personas to reference psychometric distributions such as Big Five. It would let us ask whether psychometric/population alignment improves revealed-behavior coverage.
  - Practical issue: may require code/data availability or partial reimplementation.
- Essential ablation alongside any next library: demographic-only profiles sampled from the same source library when possible.
  - Rationale: Argyle combines demographics with sparse political/social attitudes. A demographic-only version tells us whether the observed skew comes from demographic labels alone or from the richer survey/backstory fields.

### More Matcher Models

Run the matching task with multiple LLMs.

Reason:

- If collapse is model-specific, it is an LLM interpretation bias.
- If collapse is robust across models, it is stronger evidence of source-library or prompt-level transport limits.

### More Target Games

Extend beyond PGG and chip bargaining to other multi-agent economic games with transcript or trajectory data:

- trust games;
- ultimatum or dictator variants;
- punishment and reward games;
- coordination games;
- communication games;
- repeated bargaining tasks.

The best targets have:

- clear individual trajectories;
- enough players/games to estimate coverage;
- behavioral heterogeneity;
- communication or interaction history when available.

## Candidate Human Behavioral Target Datasets

The next target datasets should satisfy four conditions:

1. real human behavior, not only survey answers;
2. granular individual trajectories with actions and/or communication;
3. clear behavioral metrics that can be computed with high confidence;
4. enough repeated episodes to estimate coverage and collapse.

Multi-agent settings are ideal, but dyadic settings are still useful if they have structured choices, outcomes, and rich communication.

### Highest-Priority Candidates

1. CaSiNo: Campsite Negotiation Dialogues

- Source: Chawla et al., "CaSiNo: A Corpus of Campsite Negotiation Dialogues for Automatic Negotiation Systems" (NAACL 2021).
- Data: 1,030 two-person negotiation dialogues over Food, Water, and Firewood packages.
- Why useful: closed-domain negotiation with rich natural-language dialogue, private preferences, final allocations, points, satisfaction, opponent-likeness ratings, demographics, SVO, and Big Five.
- Clear behavioral metrics:
  - own utility / points;
  - joint utility;
  - fairness of allocation;
  - concession behavior;
  - integrative trade discovery;
  - persuasion strategy labels;
  - prosocial versus self-interested strategies;
  - message length and tone.
- Why it advances the paper: directly tests whether persona descriptions map to negotiation behavior, not only cooperation in public goods or chip trading. The built-in Big Five/SVO metadata also lets us compare persona-space traits to revealed negotiation choices.

Relationship to chip bargaining:

- CaSiNo is likely the stronger negotiation target because it has rich natural-language negotiation, private utilities, final allocations, satisfaction/opponent ratings, demographics, SVO, and Big Five.
- Chip bargaining remains complementary rather than redundant. It is already integrated with our rollout comparison, is a controlled strategic bargaining game, and has three players rather than dyads. Because top-k 3 is a complete distribution in chip, it gives a clean small-game contrast to larger PGG groups and dyadic CaSiNo negotiations.
- The manuscript can use chip as the controlled in-repo bargaining case and CaSiNo as the richer external negotiation case.

2. Persuasion for Good

- Source: Wang et al., "Persuasion for Good: Towards a Personalized Persuasive Dialogue System for Social Good" (ACL 2019).
- Data: 1,017 online persuasion dialogues where one participant tries to persuade another to donate to charity.
- Why useful: real persuasion setting with actual donation outcome, psychological and demographic metadata, and annotated dialogue strategies for a subset.
- Clear behavioral metrics:
  - donation amount;
  - donation intention;
  - persuader strategy use;
  - persuadee resistance/compliance;
  - moral/value/personality correlates;
  - sentiment and dialogue-act patterns.
- Why it advances the paper: connects persona prompting to broader AI steering and persuasion applications. It lets us ask whether LLMs identify with generous, persuadable, high-empathy, or norm-compliant humans rather than the full range of real persuadee behavior.

3. CerealBar

- Source: Suhr et al., CerealBar collaborative game dataset (EMNLP 2019).
- Data: crowdsourced human-human interactions in a two-person collaborative game with a leader and follower, natural-language instructions, movement actions, and task rewards.
- Why useful: contains both communication and fine-grained action in a collaborative environment. The task is structured enough to score behavior, but open enough to reveal coordination style.
- Clear behavioral metrics:
  - instruction length and specificity;
  - follower compliance;
  - task efficiency / points;
  - repair and clarification;
  - exploration versus direct execution;
  - leadership style;
  - coordination success.
- Why it advances the paper: tests persona-to-behavior mapping in cooperative embodied collaboration rather than economic exchange.

4. DeliData and GAP Corpus

- Sources:
  - Karadzhov, Stafford, and Vlachos, "DeliData: A dataset for deliberation in multi-party problem solving" (PACM HCI 2023).
  - Braley and Murray, "The Group Affect and Performance (GAP) Corpus" (GIFT/ICMI 2018).
- Data:
  - DeliData: 500 multi-party problem-solving conversations with message annotations and team performance.
  - GAP: 28 small-group Winter Survival Task meetings with transcripts, decision annotations, speaker metadata, influence, and group performance.
- Why useful: real multi-party deliberation with interpretable group decision outcomes.
- Clear behavioral metrics:
  - leadership / influence;
  - proposal, acceptance, rejection, confirmation;
  - solution quality;
  - contribution volume;
  - probing versus non-probing deliberation;
  - time management and group performance;
  - interruptions or dominance if using derived interruption annotations.
- Why it advances the paper: shifts from economic games to collaborative deliberation, making the broader persona-steering claim more credible.

5. Diplomacy

- Sources:
  - Peskov et al., "It Takes Two to Lie: One to Lie, and One to Listen" (ACL 2020).
  - Niculae et al., "Linguistic Harbingers of Betrayal" (ACL 2015).
  - WebDiplomacy / CICERO-related data where accessible.
- Data: multi-player strategic negotiation game with messages, alliances, deception labels in some corpora, and game actions/orders in some datasets.
- Why useful: probably the strongest target for high-stakes multi-agent social reasoning because it includes alliances, deception, persuasion, and strategic action.
- Clear behavioral metrics:
  - deception labels;
  - perceived deception;
  - betrayal versus sustained cooperation;
  - message volume and tactical language;
  - alliance support / attack actions;
  - survival and score.
- Caveat: high complexity and data-access variation. The public deception corpus has rich messages and labels, but may not contain full order/action logs in the form we would want. Full WebDiplomacy-style data would be ideal but may require licensing or additional processing.
- Why it advances the paper: tests whether LLMs identify with truthful/cooperative/normative players or can cover deceptive, opportunistic, and strategically inconsistent human behavior.

### Additional Useful Candidates

6. DealOrNoDeal and CraigslistBargain

- Sources:
  - Lewis et al., "Deal or No Deal? End-to-End Learning of Negotiation Dialogues" (EMNLP 2017).
  - He et al., "Decoupling Strategy and Generation in Negotiation Dialogues" (EMNLP 2018).
- Data:
  - DealOrNoDeal: human-human multi-issue bargaining dialogues with private reward functions and final deals.
  - CraigslistBargain: more than 6,000 buyer-seller bargaining dialogues over real item listings, with price targets and negotiated outcomes.
- Clear behavioral metrics:
  - utility;
  - agreement rate;
  - opening offer;
  - concession slope;
  - final price relative to listing/target;
  - buyer versus seller surplus;
  - hardline versus accommodating style.
- Why useful: clean and easy to parse. Less rich than CaSiNo for personality/persona validation, but good for scalable bargaining analyses.

7. eBay Best Offer / Sequential Bargaining

- Source: Backus, Blake, Larsen, and Tadelis, "Sequential Bargaining in the Field: Evidence from Millions of Online Bargaining Interactions."
- Data: large-scale real-world bargaining traces from eBay Best Offer.
- Clear behavioral metrics:
  - opening offers;
  - counteroffers;
  - concession paths;
  - acceptance/rejection;
  - final price;
  - bargaining power and split-the-difference behavior.
- Why useful: extremely strong revealed preference and scale, but little or no natural-language communication. Better for offer-sequence matching than full social transcript matching.

8. Avalon / Werewolf / Social Deduction

- Source: Stepputtis et al., "Long-Horizon Dialogue Understanding for Role Identification in the Game of Avalon with Large Language Models" (Findings of EMNLP 2023).
- Data: 20 carefully collected human Avalon games with chat, state, beliefs, persuasion/deception strategies, and hidden roles.
- Clear behavioral metrics:
  - truthful versus deceptive role behavior;
  - accusation and defense patterns;
  - voting behavior;
  - mission choices;
  - persuasion/deception strategy;
  - role identification difficulty.
- Caveat: small sample size and high prompt complexity.
- Why useful: directly links persona prompting to deception, red-teaming, and safety. A good secondary case study even if not the main statistical target.

9. Overcooked-AI Human-Human Data

- Source: Carroll et al., "On the Utility of Learning about Humans for Human-AI Coordination" (NeurIPS 2019) and Overcooked-AI repository.
- Data: human-human and human-AI gameplay trajectories in a cooperative cooking task.
- Clear behavioral metrics:
  - task efficiency;
  - role specialization;
  - waiting/collision/coordination failures;
  - adaptation to partner;
  - path efficiency;
  - soup delivery score.
- Caveat: little or no natural-language communication in the canonical data. Useful for action-level behavioral matching rather than language-rich persona matching.

### Practical Ranking For Next Empirical Expansion

Best immediate next targets:

1. CaSiNo: best mix of tractability, real negotiation, communication, structured outcomes, and persona-relevant metadata.
2. Persuasion for Good: best bridge to broader persona-steering applications because it has psychological traits and a real persuasive outcome.
3. CerealBar: best collaborative action-plus-language target.
4. DeliData or GAP: best multi-party deliberation target with clear team-performance measures.
5. Diplomacy: best high-impact strategic/deception target, but likely more work.

This sequence would let the manuscript show that revealed-behavior collapse is not specific to PGG/chip bargaining. It would cover cooperation, bargaining, persuasion, collaboration, deliberation, and strategic deception.

### Link Matching To Rollout Error More Formally

Instead of only comparing signs informally, build a formal link between matching skew and simulation error:

- use matched-player distributions to predict which aggregate metrics rollouts should over- or under-estimate;
- test across games, persona libraries, and models;
- distinguish errors predicted by behavioral support skew from errors due to action-level calibration.

Current evidence:

- PGG: matching skew predicts high cooperation, but rollouts under-contribute by the end, so calibration/dynamics dominate.
- Chip: matching skew predicts higher acceptance and lower proposer surplus, which rollouts show; welfare diverges.

### Better Coverage Figures

Useful figures for the manuscript:

1. Global identity collapse figure for PGG and chip:
   - observed effective number of selected identities versus null distribution;
   - selected identity share and never-selected share;
   - avoid legends overlapping data points.

2. Behavioral skew figure for PGG:
   - matched minus candidate-uniform for contribution, communication, reward, punishment.
   - include bootstrap/cluster intervals.

3. Behavioral skew figure for chip:
   - matched minus candidate-uniform for surplus, welfare, proposer acceptance, proposer net surplus, trade ratio.
   - keep y-axis labels legible.

4. Local collapse figure:
   - for each game, observed top-1 concentration/effective share versus uniform expectation conditional on number of players.
   - for PGG, use scatter because player count varies.
   - for chip, report against the three-player uniform null.

5. Demographic skew figure for PGG:
   - matched minus within-game candidate-uniform demographic composition;
   - show probability-weighted and top-1 matches separately;
   - use percentage-point units and crossed game/persona cluster intervals.

### Manuscript Claim Discipline

Claims we can make now:

- The persona-library-plus-model system preferentially maps Twin personas to skewed subsets of real target-game behavior.
- PGG matches are skewed toward more cooperative, lower-variance, more communicative, more reward-giving, less-punished trajectories.
- Chip matches are skewed toward more welfare/surplus-generating and more accepted proposer behavior, with lower proposer surplus.
- Global identity coverage is lower than expected under candidate-uniform nulls.
- Rollout errors are only partially explained by matching skew, indicating separable transfer and generation/calibration layers.

Claims to avoid unless expanded:

- Do not claim persona prompting cannot work.
- Do not claim matching skew fully predicts rollout error.
- Do not claim the selected player is the true psychological match for the persona.
- Do not claim avatar-label skew is meaningful across games.
- Do not claim all persona libraries will collapse without testing broader libraries.

## Current Project Artifacts

Working paper files:

- `forecasting/persona_transfer_audit/paper/short_manuscript.tex`
- `forecasting/persona_transfer_audit/paper/short_manuscript.pdf`
- `forecasting/persona_transfer_audit/paper/references.bib`

Main figures:

- `forecasting/persona_transfer_audit/figures/figure_global_identity_collapse.png`
- `forecasting/persona_transfer_audit/figures/figure_pgg_behavior_local_collapse.png`
- `forecasting/persona_transfer_audit/figures/figure_chip_behavior_local_collapse.png`
- `forecasting/persona_transfer_audit/figures/figure_chip_bargain_behavior_skew.png`
- `forecasting/persona_transfer_audit/figures/figure_pgg_demographic_skew.png`

Core scripts:

- `forecasting/persona_transfer_audit/build_twin_to_pgg_pilot.py`
- `forecasting/persona_transfer_audit/build_twin_to_chip_bargain.py`
- `forecasting/persona_transfer_audit/build_no_persona_baselines.py`
- `forecasting/persona_transfer_audit/evaluate_matches.py`
- `forecasting/persona_transfer_audit/evaluate_chip_bargain_matches.py`
- `forecasting/persona_transfer_audit/comprehensive_eval.py`
- `forecasting/persona_transfer_audit/global_identity_collapse.py`
- `forecasting/persona_transfer_audit/significance_checks.py`
- `forecasting/persona_transfer_audit/demographic_alignment_pgg.py`
- `forecasting/persona_transfer_audit/demographic_cluster_significance_pgg.py`
- `forecasting/persona_transfer_audit/plot_pgg_demographic_skew.py`

Main result directories:

- `forecasting/persona_transfer_audit/metadata/twin_direct_summary_to_pgg_stratified_32x40_top3_gpt_5_mini_seed_2/`
- `forecasting/persona_transfer_audit/metadata/twin_direct_summary_to_chip_bargain_stratified_32x48_top3_gpt_5_mini_seed_2/`

Existing rollout comparison directories checked for downstream alignment:

- `forecasting/pgg/results/twin_sampled_seed_0_gpt_5_mini__vs_human_treatments/`
- `forecasting/pgg/results/twin_sampled_unadjusted_seed_0_gpt_5_mini__vs_human_treatments/`
- `forecasting/pgg/results/twin_sampled_seed_0_gpt_5_1__vs_human_treatments/`
- `forecasting/pgg/results/twin_sampled_unadjusted_seed_0_gpt_5_1__vs_human_treatments/`
- `forecasting/chip_bargain/results/twin_sampled_unadjusted_seed_0_gpt_5_mini__vs_human_treatments/`
- `forecasting/chip_bargain/results/twin_sampled_unadjusted_seed_0_gpt_5_1_bargain_card_v1__vs_human_treatments/`
- `forecasting/chip_bargain/results/twin_sampled_unadjusted_seed_0_gpt_5_1_pgg_aligned_v3__vs_human_treatments/`

## Concordia Compact Persona Condition

Built on 2026-05-15 as a separate profile-rendering condition for the game-grounded Concordia personas. This condition is meant to test whether the full Concordia profile has unnecessary internal redundancy for the matching task.

Compact rendering keeps:

- displayed first name;
- `core_motivation`;
- `defining_experience`;
- `description`.

Compact rendering removes:

- `characteristics.initial_context`;
- `axis_position`;
- `specific_attitudes`;
- generated `memories`;
- `shared_memories`;
- title-like suffixes in persona names.

The original Concordia `personas.json` files are unchanged. The compact profile mode is implemented in `forecasting/persona_transfer_audit/build_concordia_to_targets.py` through `--profile-mode compact`.

Compact batch artifacts:

- PGG run: `concordia_pgg_game_grounded_alphaevolve_5_compact_to_pgg_stratified_32x40_top3_gpt_5_mini`
  - requests: 1,280
  - compact profile length: median 744.5 characters; mean 797.9; min 648; max 1,509
  - tiktoken input tokens: 3,283,712 total; mean 2,565.4; median 2,236.5; min 883; max 8,162
  - batch input: `forecasting/persona_transfer_audit/batch_input/concordia_pgg_game_grounded_alphaevolve_5_compact_to_pgg_stratified_32x40_top3_gpt_5_mini.jsonl`
  - metadata: `forecasting/persona_transfer_audit/metadata/concordia_pgg_game_grounded_alphaevolve_5_compact_to_pgg_stratified_32x40_top3_gpt_5_mini/`
- Chip run: `concordia_chip_bargain_game_grounded_alphaevolve_5_compact_to_chip_bargain_stratified_32x48_top3_gpt_5_mini`
  - requests: 1,536
  - compact profile length: median 690.0 characters; mean 732.0; min 583; max 1,386
  - tiktoken input tokens: 2,681,552 total; mean 1,745.8; median 1,739.0; min 1,520; max 2,047
  - batch input: `forecasting/persona_transfer_audit/batch_input/concordia_chip_bargain_game_grounded_alphaevolve_5_compact_to_chip_bargain_stratified_32x48_top3_gpt_5_mini.jsonl`
  - metadata: `forecasting/persona_transfer_audit/metadata/concordia_chip_bargain_game_grounded_alphaevolve_5_compact_to_chip_bargain_stratified_32x48_top3_gpt_5_mini/`

Prompt hygiene check on sampled compact prompts: no persona IDs, game IDs, record IDs, validation labels, source keys, manifest metadata, `initial_context`, axes, attitudes, or memories are exposed in the prompt text.

### Compact Chip-Bargaining Results

Downloaded and evaluated on 2026-05-15:

- run: `concordia_chip_bargain_game_grounded_alphaevolve_5_compact_to_chip_bargain_stratified_32x48_top3_gpt_5_mini`;
- batch id: `batch_6a071fd2d3408190bc6e6d2011059ca0`;
- completed requests: 1,536 / 1,536;
- failed requests: 0;
- parse errors: 0;
- probability errors: 0;
- duplicate-player errors: 0.

Coverage and identity-collapse:

- candidate identities: 144 observed player identities across 48 chip-bargaining games;
- top-1 selected identities: 143 / 144;
- probability-weighted selected identities: 144 / 144;
- top-1 entropy effective identities: 119.76 / 144 = 0.832;
- probability-weighted entropy effective identities: 135.51 / 144 = 0.941;
- median within-game modal top-1 share: 0.578, compared with 1/3 under uniform selection among the three players;
- median within-game effective identity share: 0.845.

Request-conditional global identity-collapse null, 20,000 simulations, one uniform top-1 draw from the three displayed players for each successful request:

- entropy effective identity share: observed 0.8317, null mean 0.9685, 95% null [0.9592, 0.9766], collapse-tail p = 0.00005;
- Simpson effective identity share: observed 0.7352, null mean 0.9411, 95% null [0.9248, 0.9554], p = 0.00005;
- top 5% identity share: observed 0.1204, null mean 0.0839, 95% null [0.0788, 0.0898], p = 0.00005;
- Gini: observed 0.3384, null mean 0.1400, 95% null [0.1207, 0.1601], p = 0.00005.

Behavioral skew relative to within-record candidate-uniform baseline, probability-weighted top-k matches:

- final surplus: +0.017, not distinguishable from zero under record/persona-cluster intervals;
- final welfare: +0.104, iid interval excludes zero but record/persona-cluster intervals include zero;
- proposer mean net surplus: -0.000, not distinguishable from zero;
- proposer acceptance rate: +0.033, record-cluster 95% interval [0.018, 0.048], persona-cluster [0.016, 0.049], sign-test p = 3.17e-25;
- proposer mean trade ratio: +0.024, record-cluster [0.007, 0.044], persona-cluster [0.015, 0.033], sign-test p = 2.44e-10;
- response acceptance rate: -0.0145, record-cluster [-0.0226, -0.0069], persona-cluster [-0.0231, -0.0061], sign-test p = 1.44e-06;
- response net surplus conditional on acceptance: aggregate matched-minus-uniform difference -0.0496; request-level significance mean -0.0263, persona-cluster interval [-0.0498, -0.0027], record-cluster interval includes zero;
- received trade rate: -0.0071, iid and sign-test significant but record/persona-cluster intervals narrowly include zero.

Interpretation: the compact game-grounded Concordia condition has broad nominal coverage, but top-1 choices are still non-uniform within games and globally concentrated relative to a request-conditional uniform null. The behavioral skew is narrower than the stark welfare/surplus skew seen in some earlier chip runs, but the matcher still favors players whose proposals are accepted more often and whose offers are more generous/high-ratio, while it disfavors response-side acceptance.

### Cross-Persona Chip Behavioral Skew Figure

Generated on 2026-05-15:

- figure: `forecasting/persona_transfer_audit/figures/figure_chip_behavior_skew_across_personas.png`
- PDF: `forecasting/persona_transfer_audit/figures/figure_chip_behavior_skew_across_personas.pdf`
- source data: `forecasting/persona_transfer_audit/figures/figure_chip_behavior_skew_across_personas_source_data.csv`
- script: `forecasting/persona_transfer_audit/plot_chip_behavior_skew_across_personas.py`

The figure compares no-persona, Twin, Argyle-style demographic backstories, Nemotron, and compact game-grounded Concordia on the chip-bargaining behavioral-skew metrics. Cells are request-level matched-minus-candidate-uniform differences standardized by candidate-player SD. Black dots mark record-level 95% intervals excluding zero.

Main read:

- proposal acceptance is the most stable positive skew across every condition, including no-persona and compact Concordia;
- Twin, Argyle-style, and Nemotron all show positive final surplus/welfare skew and negative proposer-net-surplus skew;
- compact Concordia attenuates final-surplus/final-welfare skew almost to zero, but preserves positive proposal-acceptance and trade-ratio skews;
- no-persona has the strongest final-welfare/final-surplus skew and negative responder/received-trade skews, consistent with the default model already preferring a relatively legible bargaining role before persona prompting.

### Compact PGG Results

Downloaded and evaluated on 2026-05-15:

- run: `concordia_pgg_game_grounded_alphaevolve_5_compact_to_pgg_stratified_32x40_top3_gpt_5_mini`;
- batch id: `batch_6a071fd223948190912987c9b4657f4a`;
- completed requests: 1,280 / 1,280;
- failed requests: 0;
- parse errors: 0;
- probability errors: 0;
- duplicate-player errors: 0.

Coverage and identity-collapse:

- candidate identities: 342 observed player identities across 40 PGG games;
- top-1 selected identities: 261 / 342 = 0.763;
- probability-weighted selected identities: 333 / 342 = 0.974;
- top-1 entropy effective identities: 198.43 / 342 = 0.580;
- probability-weighted entropy effective identities: 247.61 / 342 = 0.724.

Request-conditional global identity-collapse null, 20,000 simulations, one uniform top-1 draw from the displayed players for each successful request:

- selected identity share: observed 0.7632, null mean 0.9360, 95% null [0.9123, 0.9561], collapse-tail p = 0.00005;
- entropy effective identity share: observed 0.5802, null mean 0.7513, 95% null [0.7356, 0.7662], p = 0.00005;
- Simpson effective identity share: observed 0.4745, null mean 0.6289, 95% null [0.6120, 0.6443], p = 0.00005;
- top 5% identity share: observed 0.2078, null mean 0.1613, 95% null [0.1539, 0.1703], p = 0.00005;
- Gini: observed 0.5501, null mean 0.4077, 95% null [0.3948, 0.4213], p = 0.00005.

Behavioral skew relative to within-game candidate-uniform baseline, probability-weighted top-k matches:

- mean contribution rate: +0.00018, not distinguishable from zero under game/persona-cluster intervals;
- full contribution rate: -0.00014, not distinguishable from zero;
- zero contribution rate: +0.00110, not distinguishable from zero under clustered intervals;
- contribution variability: +0.055, not distinguishable from zero;
- messages per round: +0.0352, game-cluster 95% interval [0.0141, 0.0604], persona-cluster [0.0115, 0.0577];
- reward-given round rate: +0.0283, game-cluster [0.0153, 0.0435], persona-cluster [0.0038, 0.0517];
- punishment-given round rate: +0.0042, not distinguishable from zero under clustered intervals;
- punishment-received round rate: approximately zero.

Interpretation: compact game-grounded Concordia almost eliminates the contribution-level skew that appears in no-persona, Twin, Argyle-style, and Nemotron PGG runs, but it still selects players who communicate more and give rewards more often. It also still shows significant identity concentration relative to the request-conditional uniform null, so reducing profile redundancy and grounding the personas in the target game attenuates behavioral payoff/cooperation skew but does not remove non-uniform matching.

### Cross-Persona PGG Behavioral Skew Figure

Generated on 2026-05-15:

- figure: `forecasting/persona_transfer_audit/figures/figure_pgg_behavior_skew_across_personas.png`
- PDF: `forecasting/persona_transfer_audit/figures/figure_pgg_behavior_skew_across_personas.pdf`
- source data: `forecasting/persona_transfer_audit/figures/figure_pgg_behavior_skew_across_personas_source_data.csv`
- script: `forecasting/persona_transfer_audit/plot_pgg_behavior_skew_across_personas.py`

The figure compares no-persona, Twin, Argyle-style demographic backstories, Nemotron, and compact game-grounded Concordia on PGG behavioral-skew metrics. Cells are request-level matched-minus-candidate-uniform differences standardized by candidate-player SD. Black dots mark game-cluster 95% intervals excluding zero.

Main read:

- no-persona, Twin, Argyle-style, and Nemotron all skew toward more cooperative, less variable contributors who communicate more and receive less punishment;
- compact Concordia nearly removes the contribution, full-contribution, zero-contribution, contribution-variability, and punishment-received skews;
- compact Concordia preserves significant positive skew on communication and reward-giving;
- this supports a more nuanced claim: game-grounded compact personas can attenuate the default cooperation/regularity skew, but they do not make the matching distribution uniform and they still steer toward socially legible cooperative behaviors.

## Next-Step Positioning: Behavioral Axes Rather Than Only Better Prediction

Discussion on 2026-05-16 after compact Concordia results:

The next direction should not only be "find a persona library that predicts better." A stronger contribution is to use the real-trajectory matching design to learn what dimensions of human behavior actually need to be represented for a persona library to cover a target social environment.

Two linked goals:

1. Build behavior-grounded persona libraries.
   - Generic empirical or synthetic persona libraries often preserve surface diversity but fail to cover revealed behavioral heterogeneity in PGG and chip bargaining.
   - Target-grounded Concordia does much better, especially in attenuating PGG over-cooperation, but still shows non-uniform identity coverage and residual skew toward communicative/rewarding or accepted-proposer trajectories.
   - A useful contribution would be a persona library whose support is explicitly calibrated to real human trajectories in a specific behavioral domain, not only to demographics, hobbies, survey responses, or generic personality prose.

2. Learn behavioral dimensions.
   - The more substantive question is: what are the minimal axes needed to distinguish real human behavior in the target setting?
   - For PGG, candidate axes include contribution propensity, conditional cooperation, endgame sensitivity, willingness to reward/punish, punishment sensitivity, communication frequency/style, and rule comprehension.
   - For chip bargaining, candidate axes include proposer generosity, accepted-proposal behavior, surplus claiming, responder acceptance threshold, risk tolerance, reciprocity, and strategic patience.
   - A possible "rank" question: what is the smallest dimension \(d\) such that a \(d\)-dimensional latent/persona representation can reproduce the distribution of matched human trajectories out of sample? This can be evaluated with held-out games, held-out trajectories, and held-out persona conditions.
   - Finding that demographics, hobbies, or broad survey backstories explain little after target-relevant behavioral axes are included would itself be a useful negative result: they may be valid human descriptors without being the right representation for LLM-mediated behavior in this environment.

Relationship to Manning and Horton, "General Social Agents" (`https://benjaminmanning.io/files/optimize.pdf`):

- Their paper optimizes theory-grounded agent prompts or prompt mixtures against human training data, then validates across related but distinct games. The goal is predictive simulation in novel settings.
- Our project becomes too similar if we simply optimize persona prompts/distributions to reduce rollout prediction error.
- The distinction is strongest if our primary object is representation and diagnosis:
  - ask the model to choose among real human trajectories rather than generate actions;
  - measure coverage and collapse directly in the space of revealed human behavior;
  - identify which persona dimensions control the matching distribution;
  - estimate the dimensionality/rank of behavioral heterogeneity;
  - use simulation only as downstream validation, not as the only optimization target.
- Our work can complement General Social Agents: their method needs a theory-grounded prompt space; our method can test whether a proposed prompt/persona space actually spans the relevant revealed-behavior support before using it for simulation.

Concrete analysis plan:

- Construct a human trajectory feature matrix by game and player.
- Define target-relevant behavioral features for each environment before looking at match outcomes.
- Use persona-to-trajectory match probabilities as a matrix over personas and real players.
- Fit low-rank, sparse, or factor models to ask how many latent axes explain held-out match distributions and behavioral moments.
- Run ablations/conjoint-style persona interventions: demographics only, hobbies only, generic personality only, target-relevant axes only, full persona, and shuffled/ablated versions.
- Validate axes by whether they improve coverage, reduce behavioral skew, and transfer to held-out games without overfitting.

## High-Impact Framing For Nature/Science/PNAS/NHB

Discussion on 2026-05-16:

Target framing should be broader than a benchmark of persona prompting. The paper should present a framework for social and behavioral science with LLMs:

- LLM social agents require representational validation before they are used for prediction, simulation, or intervention.
- Persona diversity in prose, demographics, hobbies, or survey backstories is not the same as behavioral support over real human trajectories.
- Revealed-behavior matching gives a calibration-light assay: the model must choose among actual human trajectories instead of generating its own actions.
- This lets us quantify behavioral coverage, identity collapse, skew toward socially legible behavior, and the latent behavioral dimensions needed to span a target environment.
- The scientific payoff is not only "better simulation"; it is a way to learn which dimensions distinguish human behavior in strategic social settings.

Venue positioning:

- Nature Human Behaviour is the most natural aspirational target if the paper makes a substantive claim about understanding individual/collective human behavior, not only LLM engineering. Their scope explicitly covers individual and collective human behavior across psychology, economics, social behavior, communication, personality, judgment, decision-making, norms, and related topics.
- PNAS is plausible if the contribution is framed as a general interdisciplinary method for validating AI social agents, with strong empirical demonstrations across several behavioral domains.
- Nature or Science would require a bigger, cleaner headline: e.g., "LLM persona diversity fails to span real human behavioral diversity, but target-grounded behavioral axes recover it across strategic social environments." This likely needs several domains, external validation, and a result that changes how broad scientific audiences think about AI simulation.
- Nature Computational Science, Nature Machine Intelligence, Science Advances, PNAS Nexus, and similar venues are realistic fallback/parallel targets if the final paper is more method/framework-heavy than behavior-theory-heavy.

What the high-impact version needs:

- At least three behavioral domains, ideally including PGG, chip bargaining, and a richer negotiation setting.
- Multiple persona sources: no-persona, empirical survey/persona summaries, synthetic generic personas, and target-grounded/persona-generator conditions.
- A positive contribution, not only a negative audit: show that behavior-grounded axes or target-grounded generation can reduce skew and improve coverage.
- Dimensionality/rank analysis: estimate the minimal behavioral basis needed to explain matched human trajectories and held-out behavioral moments.
- Ablations that show generic descriptors such as demographics or hobbies add little relative to target-relevant behavioral axes, if that is supported by the data.
- Out-of-sample validation: held-out games, held-out treatments, held-out trajectories, and ideally a downstream simulation check.
- Clear distinction from General Social Agents: they optimize theory-grounded prompts/prompt mixtures for predictive simulation; this project validates whether persona representations span the revealed-behavior support and learns the dimensions of that support. Simulation is downstream validation, not the primary object.

## Identity Coverage Versus Behavioral Support

Discussion on 2026-05-16:

Identity-uniform coverage can be misleading if multiple observed players enacted nearly identical behavior. Example: in a 10-player PGG where nearly everyone contributes faithfully in every round, the matcher is forced to choose a top 3. It may repeatedly choose a few specific identities, producing apparent identity collapse, even though the selected trajectories cover the relevant behavior type.

Implication:

- Identity collapse is a useful diagnostic for label/order/person-specific concentration, but it should not be the main evidence for failure to cover human behavior.
- The primary estimand should be coverage in behavior space, not equality over individual identities.
- Behavioral skew metrics already partially address this, because if the selected identities are behaviorally similar to the unselected identities, matched-minus-candidate-uniform differences shrink.
- However, moment-level behavioral skew is not enough: the selected distribution could match a few moments while missing minority behavioral modes.

Recommended framing:

- For each game \(g\), define the empirical candidate distribution over real trajectories:
  \(P_g = n_g^{-1}\sum_i \delta_{x_{gi}}\), where \(x_{gi}\) is player \(i\)'s behavioral trajectory/features.
- Define the matched distribution:
  \(Q_g = \sum_i q_{gi}\delta_{x_{gi}}\), where \(q_{gi}\) is the LLM-assigned top-k probability, or top-1 mass in the top-1 version.
- Evaluate whether \(Q_g\) approximates \(P_g\) in behavior space, rather than whether \(q_{gi}=1/n_g\) for every identity.

Metrics to add:

- Moment skew: current pre-registered behavioral feature differences \(E_Q[f]-E_P[f]\).
- Distributional distance: MMD/energy distance/Wasserstein distance between \(P_g\) and \(Q_g\) over standardized behavioral features.
- Behavioral cluster coverage: cluster real trajectories into behavior types, then compare selected mass versus candidate-uniform mass over clusters.
- Epsilon-ball coverage: fraction of candidate trajectories whose behavioral neighborhood receives nontrivial matched mass.
- Kernel effective support: entropy/effective-N over behavior clusters or kernel-smoothed trajectories, so duplicate/near-duplicate behaviors do not count as separate missing modes.
- Residual identity concentration: after conditioning on behavior type or behavioral distance, test whether the matcher still collapses onto particular identities. This becomes a secondary diagnostic of arbitrary selection, not the main behavioral-support claim.

Paper language:

- Avoid saying "the model should choose all people equally."
- Say "under candidate-uniform sampling, all real trajectories define the empirical support, but equality over identities is only a strict null. Our central question is whether the persona-conditioned model covers the behavioral support of those trajectories."
- Identity collapse should be reported as an upper-bound or auxiliary warning: it matters most when it remains after behavior-equivalence adjustment.

## Systematic Axis Discovery And Iterative Persona Generation

Discussion on 2026-05-16:

The next step should be behavior-first rather than persona-first. Instead of only changing Concordia axes by intuition, use the revealed human trajectories to define the behavioral support that persona libraries are supposed to cover.

Recommended pipeline:

1. Define behavior space.
   - Build a feature vector \(x_{gi}\) for each player \(i\) in game \(g\).
   - PGG features: contribution level, full/zero contribution rates, contribution slope/endgame shift, conditional cooperation if measurable, reward/punishment given/received, communication frequency/style, rule-attention proxies.
   - Chip features: proposer acceptance, proposal generosity/trade ratio, proposer surplus, final surplus/welfare, responder acceptance threshold, responder surplus conditional on acceptance, received-trade rate.

2. Estimate behavior modes and dimensionality.
   - Use interpretable factor/PCA/NMF/latent-class or clustering models on training games.
   - Select the number of dimensions \(K\) using held-out reconstruction/distributional distance, stability, and interpretability rather than only variance explained.
   - The scientific question is: what is the minimal \(K\) needed to cover the behavioral distribution of real trajectories?

3. Translate behavior modes into candidate persona axes.
   - Start with theory-grounded axes plus data-derived residual axes.
   - For each latent/mode, write a human-interpretable axis description: e.g., "baseline contribution tendency," "conditional cooperation," "norm enforcement," "communication propensity," "proposal generosity," "responder acceptance threshold."
   - Use LLM assistance for wording, but keep human review and a frozen train/test split to avoid prompt-hacking the final validation set.

4. Generate persona libraries systematically.
   - For a fixed axis set \(A_K\), generate \(N\) personas using Concordia compact profiles.
   - Compare several \(K\): e.g., 2, 3, 5, 8, and a richer full-axis condition.
   - Sample axis combinations with an orthogonal design or Latin hypercube so the persona library covers the space without exploding factorially.
   - Include controls: generic axes, demographic-only axes, random/irrelevant axes, and current game-grounded Concordia axes.

5. Evaluate in behavior space.
   - For each game, compare candidate-uniform \(P_g = n_g^{-1}\sum_i\delta_{x_{gi}}\) with matched distribution \(Q_g=\sum_i q_{gi}\delta_{x_{gi}}\).
   - Metrics: moment skew, MMD/energy/Wasserstein distance over standardized behavior features, cluster/mode coverage, epsilon-neighborhood coverage, and residual identity collapse conditional on behavior.

6. Separate support from sampling weights.
   - Let \(M_{pi}\) be the match probability mass assigned by persona \(p\) to trajectory \(i\).
   - Ask whether any mixture \(w\) over personas can make \(w^\top M\) approximate the human behavior distribution.
   - If optimized reweighting works, the persona library has support but the sampling weights are wrong.
   - If no reweighting works, the axis/persona library is missing behavioral support.

7. Iterate only on training/development games.
   - Identify under-covered behavior modes where \(P_g-Q_g\) is large.
   - Summarize their behavioral signatures.
   - Propose new axes or revise axis definitions to target missing modes.
   - Regenerate personas and re-evaluate on dev games.
   - Keep a final held-out test set untouched for the paper.

Objective for automated search:

\[
\mathcal{L}(A) =
\text{BehaviorDistance}(P,Q_A)
+ \lambda_1 \text{MomentSkew}(P,Q_A)
+ \lambda_2 \text{ResidualIdentityCollapse}
+ \lambda_3 \text{PromptLength/Redundancy}
- \lambda_4 \text{Interpretability}.
\]

Use Bayesian optimization, evolutionary search, or coordinate ablation over axis sets, but constrain the search with theory and interpretability. Avoid fully unconstrained prompt optimization, because that would collapse into the General Social Agents style objective and risk producing prompt hacks rather than behavioral insight.

Immediate implementation recommendation:

- First add behavior-space distributional metrics to existing runs.
- Then run support-versus-weight diagnostics on existing persona libraries using optimized persona reweighting.
- Only after that generate new Concordia libraries over systematic axis sets \(K=2,3,5,8\), starting with PGG because compact Concordia already showed the clearest improvement there.

Clarification on General Social Agents, checked 2026-05-16:

- The main "selection method" assumes a finite candidate prompt library and optimizes mixture weights over that library to match training human distributions.
- The paper's strategic-game examples use this selection method: candidate prompts encode theoretically motivated levels of strategic reasoning, then the optimizer finds weights over those prompts.
- The paper also proposes a "construction method" where a hand-specified prompt template contains numeric trait parameters, and derivative-free optimization searches over those parameter values.
- Thus it is not primarily unconstrained prompt generation by an LLM. It is closer to optimizing weights over a theory-grounded prompt library, plus optimizing numeric values inside a theory-grounded template.
- This distinction helps our positioning: optimized persona reweighting should be framed as a diagnostic of whether a library has support, not as our final contribution. Our main contribution remains behavior-space coverage and discovery of the dimensions needed to span real trajectories.

## Pipeline Schematic

Created on 2026-05-19:

- figure: `forecasting/persona_transfer_audit/figures/figure_persona_matching_pipeline_schematic.png`
- PDF: `forecasting/persona_transfer_audit/figures/figure_persona_matching_pipeline_schematic.pdf`
- script: `forecasting/persona_transfer_audit/plot_persona_matching_pipeline_schematic.py`
- TikZ source: `forecasting/persona_transfer_audit/figure_persona_matching_pipeline_schematic_tikz.tex`
- TikZ PDF: `forecasting/persona_transfer_audit/figures/figure_persona_matching_pipeline_schematic_tikz.pdf`

The schematic shows the revealed-behavior matching pipeline in three manuscript-style panels: (A) readable persona-input categories rather than library-specific names, (B) revealed-behavior matching in a concrete PGG example, and (C) coverage/evaluation diagnostics. The PGG example illustrates that the model is not generating actions; it is choosing among real candidate trajectories such as a fully cooperative player, a conditional cooperator with no communication, and a fully defective player using cheap talk. The evaluation panel emphasizes behavioral skew, behavior-space coverage, residual identity concentration, and support-versus-sampling.

The TikZ version was added after the matplotlib/SVG version because it is easier to edit directly in the LaTeX manuscript and gives finer control over vector layout. It was compiled locally with a temporary Tectonic binary into `figure_persona_matching_pipeline_schematic_tikz.pdf` and rendered to PNG for visual QA.

## Cross-Persona Figure Revisions

Updated on 2026-05-20:

- PGG behavioral skew figure:
  - figure: `forecasting/persona_transfer_audit/figures/figure_pgg_behavior_skew_across_personas.png`
  - PDF: `forecasting/persona_transfer_audit/figures/figure_pgg_behavior_skew_across_personas.pdf`
  - source data: `forecasting/persona_transfer_audit/figures/figure_pgg_behavior_skew_across_personas_source_data.csv`
  - script: `forecasting/persona_transfer_audit/plot_pgg_behavior_skew_across_personas.py`
- Bargaining behavioral skew figure:
  - figure: `forecasting/persona_transfer_audit/figures/figure_chip_behavior_skew_across_personas.png`
  - PDF: `forecasting/persona_transfer_audit/figures/figure_chip_behavior_skew_across_personas.pdf`
  - source data: `forecasting/persona_transfer_audit/figures/figure_chip_behavior_skew_across_personas_source_data.csv`
  - script: `forecasting/persona_transfer_audit/plot_chip_behavior_skew_across_personas.py`

Display-label changes:

- `Argyle-style` -> `Demographic surveys`;
- `Twin` -> `Twin-2K-500`;
- `Nemotron` -> `Synthetic (Nemotron)`;
- `Concordia compact` -> `Task-adaptive (Concordia)`;
- display order changed to: No persona, Demographic surveys, Twin-2K-500, Synthetic (Nemotron), Task-adaptive (Concordia);
- the bottom explanatory note was removed from the rendered figures;
- the chip-bargaining title was changed to "Bargaining game behavioral skew across persona sources."

## Cross-Persona Local Collapse

Created on 2026-05-20:

- figure: `forecasting/persona_transfer_audit/figures/figure_local_collapse_across_personas.png`
- PDF: `forecasting/persona_transfer_audit/figures/figure_local_collapse_across_personas.pdf`
- source data: `forecasting/persona_transfer_audit/figures/figure_local_collapse_across_personas_source_data.csv`
- script: `forecasting/persona_transfer_audit/plot_local_collapse_across_personas.py`

The cross-persona local-collapse figure compares PGG and the bargaining game. The no-persona baseline is omitted because it has only one request per game, so "share of personas selecting the modal top-1 player" is degenerate and not comparable to persona libraries. For each persona library and game transcript, the script computes:

\[
\text{excess local collapse}_{gs}
= \max_i \hat q_{gsi}^{\mathrm{top1}}
- \mathbb{E}_{U_g}\left[\max_i \hat q_{gi}^{\mathrm{top1}}\right],
\]

where the first term is the observed modal top-1 player share across profiles from persona source \(s\) in game \(g\), and the second term is the expected modal share under request-conditional uniform top-1 selection over the players shown in the same game. This controls for different numbers of players in PGG games and for the finite number of requests per game.

Median excess local collapse by game and persona source:

- Public goods game:
  - Demographic surveys: +0.372 excess; median observed modal top-1 share 0.594.
  - Twin-2K-500: +0.263 excess; median observed modal top-1 share 0.540.
  - Synthetic (Nemotron): +0.402 excess; median observed modal top-1 share 0.625.
  - Task-adaptive (Concordia): +0.094 excess; median observed modal top-1 share 0.344.
- Bargaining game:
  - Demographic surveys: +0.235 excess; median observed modal top-1 share 0.656.
  - Twin-2K-500: +0.173 excess; median observed modal top-1 share 0.594.
  - Synthetic (Nemotron): +0.376 excess; median observed modal top-1 share 0.797.
  - Task-adaptive (Concordia): +0.157 excess; median observed modal top-1 share 0.578.

Interpretation: task-adaptive Concordia has the lowest local identity collapse in both target games, especially in PGG. Synthetic (Nemotron) has the highest median excess collapse in both games. This figure should be presented as an identity-concentration diagnostic, not as the main behavioral-support estimand, because behaviorally similar players can still make identity-uniformity too strict.

## Behavior-Space Effective Support

Created on 2026-05-20:

- figure: `forecasting/persona_transfer_audit/figures/figure_behavioral_support_across_personas.png`
- PDF: `forecasting/persona_transfer_audit/figures/figure_behavioral_support_across_personas.pdf`
- source data: `forecasting/persona_transfer_audit/figures/figure_behavioral_support_across_personas_source_data.csv`
- script: `forecasting/persona_transfer_audit/plot_behavioral_support_across_personas.py`

This is the first implementation of Metric 1, motivated by the concern that identity-level collapse can be too strict. If several players in the same game behaved almost identically, repeatedly matching one of those players should not be interpreted as missing a distinct behavioral mode. The metric therefore measures how much of the empirical behavior space is covered by the matched probability distribution, not whether all identities are selected equally.

For each game \(g\), each observed player \(i\) is represented by a standardized behavior vector \(z_{gi}\). PGG uses the same core features as the behavioral-skew figure:

- mean contribution rate;
- full contribution rate;
- zero contribution rate;
- contribution variability;
- messages per round;
- reward-given round rate;
- punishment-given round rate;
- punishment-received round rate.

The bargaining game uses:

- final surplus;
- final welfare;
- proposer net surplus;
- proposer acceptance rate;
- offered trade ratio;
- responder acceptance rate;
- responder surplus conditional on acceptance;
- received-trade rate.

Features are standardized using the candidate human trajectories for the corresponding target game. Missing feature values, mainly responder surplus conditional on acceptance in bargaining, are imputed to the target-game mean after standardization. Within each game, define an RBF similarity kernel over human trajectories:

\[
K_{gij} = \exp\left(-\frac{\lVert z_{gi}-z_{gj}\rVert^2}{2\sigma_g^2}\right),
\]

where \(\sigma_g\) is the median nonzero pairwise distance among candidate trajectories in game \(g\). Let \(P_g\) be the candidate-uniform human distribution and \(Q_{gs}\) be the persona-source-specific matched distribution, obtained by aggregating the model's top-k probabilities over profiles and normalizing within the game. The behavior-space effective support is:

\[
N_{\mathrm{eff}}^K(w) = \frac{1}{w^\top K_g w}.
\]

The plotted quantity is:

\[
\frac{N_{\mathrm{eff}}^K(Q_{gs})}{N_{\mathrm{eff}}^K(P_g)}.
\]

A value near 1 means the matched trajectories span about as much behavior-space support as the empirical human trajectories in that game. A value below 1 means matched mass is concentrated in a narrower behavioral region. Values slightly above 1 are possible because the selected subset can be more behaviorally dispersed than the full empirical distribution when the full set contains redundant near-duplicate trajectories.

Median behavior-space effective support ratios:

- Public goods game:
  - No persona: 0.784.
  - Demographic surveys: 0.879.
  - Twin-2K-500: 0.955.
  - Synthetic (Nemotron): 0.873.
  - Task-adaptive (Concordia): 1.011.
- Bargaining game:
  - No persona: 0.921.
  - Demographic surveys: 0.973.
  - Twin-2K-500: 0.978.
  - Synthetic (Nemotron): 0.960.
  - Task-adaptive (Concordia): 0.978.

The source data also includes the unkernelized probability effective-N ratio and the expected pairwise behavioral-distance ratio. The pairwise-distance ratio is useful as a more literal "span" diagnostic. Median pairwise-distance ratios are lower than the kernel support ratios and show the same ordering in PGG: no persona 0.509, demographic surveys 0.738, Twin-2K-500 0.882, Synthetic (Nemotron) 0.730, and Task-adaptive (Concordia) 1.010. In the bargaining game, median pairwise-distance ratios are 0.750, 0.884, 0.932, 0.879, and 0.934, respectively.

Interpretation: this metric partly resolves the fairness concern with local identity collapse. Task-adaptive Concordia no longer merely looks less collapsed by identity; in PGG it spans the empirical behavior space almost exactly under this metric. Twin also performs better than demographic surveys, synthetic Nemotron personas, and the no-persona baseline. In the bargaining game, all persona libraries cover most of the behavioral support once behavior similarity is accounted for, but the no-persona baseline remains narrower. This suggests that bargaining identity collapse is less concerning behaviorally than the raw top-1 concentration figure implies, partly because there are only three players and some trajectories are behaviorally close under the current feature set.

Important caveat: this is only as good as the behavior vector. Metric 2-style distributional tests over the same features would be close to the behavioral-skew analysis we already have, so the immediate value here is not another skew test but an identity-adjusted support diagnostic. Future versions should test feature-set sensitivity, richer communication features, and cluster-based support metrics on held-out games.

## Poster Evaluation Framework Notation

Created on 2026-05-20:

- Word-friendly notation handout: `forecasting/persona_transfer_audit/poster/evaluation_framework_notation.docx`
- copy-ready Markdown source: `forecasting/persona_transfer_audit/poster/evaluation_framework_notation.md`
- generation script: `forecasting/persona_transfer_audit/poster/make_evaluation_framework_notation_docx.py`

The poster notation defines games \(g\), observed players \(I_g\), real behavior trajectories \(b_{gi}\), behavioral features \(x_{gi}\), the candidate-uniform human reference distribution \(P_g\), the LLM top-k response probabilities \(r_{gasi}\), and the persona-source matched distribution \(Q_{gs}\). It also defines the behavioral skewness statistic used in the cross-persona heatmap as a matched-minus-human feature difference, averaged across target games and standardized by the empirical human standard deviation of the feature.

Follow-up on 2026-05-20: the full Word notation was too long for the poster and did not render equations cleanly in Office. A shorter poster insert was created instead:

- compact poster block PNG: `forecasting/persona_transfer_audit/poster/evaluation_framework_poster_block.png`
- compact poster block SVG: `forecasting/persona_transfer_audit/poster/evaluation_framework_poster_block.svg`
- compact poster block PDF: `forecasting/persona_transfer_audit/poster/evaluation_framework_poster_block.pdf`
- compact copy text: `forecasting/persona_transfer_audit/poster/evaluation_framework_poster_block.md`
- rendering script: `forecasting/persona_transfer_audit/poster/make_evaluation_framework_poster_block.py`

The compact version keeps only three definitions: \(P_g(i)=1/n_g\), \(Q_{gs}(i)=\operatorname{Avg}_a r_{gasi}\), and the SD-standardized behavioral skew \(\widetilde{\delta}_{s\ell}\).

## Current Conceptual Framing: Toward Behaviorally Complete Social Agents

Updated on 2026-05-21.

Path convention for collaborators: all file paths in these notes should be written relative to the repository root, i.e. relative to the `PGG-finetuning` folder. Do not use machine-specific absolute paths such as `/Users/...`.

The current broader framing is that this project maps persona libraries into real human behavior in target social environments. A persona source can be:

- no persona / default LLM behavior;
- demographic or survey-based backstories;
- digital-twin survey summaries such as Twin-2K-500;
- general-purpose synthetic persona libraries such as Nemotron;
- task-adaptive persona generation such as Concordia.

For each source, the question is not only whether the persona descriptions look diverse. The question is whether the LLM, when asked to inhabit those personas in a target environment, assigns probability mass to the diverse real behaviors humans actually exhibited.

A concise way to state the motivation:

> Persona prompts are not agents. They are incomplete coordinate systems for human behavior.

Human behavior can depend on many latent dimensions: cooperation, trust, fairness norms, spite, risk tolerance, beliefs about others, strategic sophistication, communication tendency, norm-enforcement preference, sensitivity to incentives, and many others. Generic persona descriptions based on demographics, hobbies, life history, or survey attitudes may omit the dimensions that actually drive behavior in a specific strategic setting. Even rich survey-based persona can therefore be incomplete for the behavior we want to simulate.

The constructive ambition is not literally to build a complete finite representation of humans. A more defensible framing is:

> Toward behaviorally complete social agents: learning the minimal persona dimensions needed to span human behavior in a target environment.

This gives the project three linked contributions:

1. Diagnosis: existing persona libraries can be incomplete.
   - Survey, demographic, digital-twin, and general-purpose synthetic personas can be diverse in description space but still map to a narrower or skewed subset of revealed human behavior in incentivized social interactions.

2. Decomposition: support failure and calibration failure are distinct.
   - Support failure means the persona library cannot produce some human behavior modes at all.
   - Calibration failure means the library has the right behavior modes, but the sampling weights over personas are wrong.
   - This distinction is crucial because aggregate simulation error alone cannot tell us which failure occurred.

3. Method: learn a behaviorally sufficient basis.
   - Use real human trajectories to identify the behavioral axes that distinguish people in a target environment.
   - Generate or select personas that span those axes.
   - Evaluate whether the induced LLM-persona distribution covers held-out human trajectories, not just whether it matches a few aggregate moments.

## Support Versus Calibration

Let \(M_{gpi}\) be the probability that persona \(p\), when evaluated in game \(g\), assigns to human trajectory \(i\). Let \(P_g\) be the empirical human reference distribution in game \(g\). Then a persona library has behavioral support if some mixture over personas can approximate the human distribution:

\[
\min_{w \in \Delta}
D\left(P_g,\sum_p w_p M_{gpi}\right).
\]

Interpretation:

- If this optimized distance remains large, the persona library is missing behavioral support. No reweighting of available personas can recover the human distribution.
- If this optimized distance is small but the library's natural or random sampling distribution performs badly, the issue is calibration or weighting, not support.
- This turns persona evaluation into a sharper diagnostic: first ask whether the library spans the behavior space; then ask how to sample or weight that library.

This also connects to the current behavioral-support figure:

- `forecasting/persona_transfer_audit/figures/figure_behavioral_support_across_personas.png`
- `forecasting/persona_transfer_audit/plot_behavioral_support_across_personas.py`

That figure measures whether the observed matched distribution \(Q_{gs}\) spans behavior-space support similarly to candidate-uniform human trajectories. A next step is to add an optimized-mixture diagnostic over personas to separate support failure from sampling-weight failure.

## Moment Matching And Missing Bases

A useful analogy is moment matching. When researchers calibrate LLM agents or simulated agents, they often choose a set of target moments and tune prompts, agent types, or mixture weights until generated behavior matches those moments. This implicitly assumes the chosen representation is rich enough to express the relevant behavioral variation.

The concern is a basis-completeness problem. If the persona representation omits dimensions that determine behavior, then moment matching can succeed on selected moments while failing on unmeasured dimensions or held-out settings. In a Taylor-expansion analogy, matching coefficients on an incomplete basis cannot recover variation along missing dimensions. Better calibration cannot fix a missing behavioral axis.

For this project, the implication is:

- behavioral skewness shows which moments current persona libraries fail to match;
- behavior-space support asks whether selected trajectories span the empirical behavior space after accounting for similarity between players;
- optimized persona reweighting can test whether failures come from missing support or wrong sampling weights;
- systematic axis discovery should identify which persona dimensions are necessary to make support complete enough for a target class of environments.

This reframes the project from "persona prompting is biased" to a more general methodological claim:

> Moment-matching LLM agents using incomplete persona bases can produce calibrated-looking simulations that fail outside the moments used for calibration.

## Relation To Adjacent Work

Relevant references and how they position the project:

- General Social Agents: Manning and Horton propose building agents from theory-grounded instructions, empirical data, and LLM knowledge, and show improved prediction across novel games. This is close to our constructive direction. Our distinction should be that we first diagnose whether a persona library has behavioral support before optimizing predictive performance or mixture weights. See Manning & Horton (2026), NBER Working Paper 34937: https://ideas.repec.org/p/nbr/nberwo/34937.html

- Quantifying the Persona Effect: Hu and Collier show that persona variables explain limited variance in many subjective NLP datasets and that persona prompting helps most when the persona variables are actually predictive of human disagreement. This supports the basis-incompleteness framing: personas help only when their dimensions align with the behavioral variation to be predicted. See Hu & Collier (2024): https://aclanthology.org/2024.acl-long.554/

- LLMs as Psychological Simulators: Lin argues that LLM simulation should move beyond demographic prompting toward psychologically grounded personas, with stronger validation frameworks and caution around prompt sensitivity, cultural bias, and simulation/substitution boundaries. This aligns with our move from generic personas to task-relevant behavioral dimensions. See Lin (2026): https://journals.sagepub.com/doi/10.1177/25152459251410153

- Generative Agents and Social Simulacra: Park and colleagues show that LLM-based agents can generate believable individual and social behavior in interactive environments. These papers motivate the promise of agent simulation, but their validation emphasizes believability, interaction realism, or design utility rather than whether personas cover the support of observed human behavior in incentivized target settings. See Social Simulacra, Park et al. (2022): https://arxiv.org/abs/2208.04024 and Generative Agents, Park et al. (2023): https://arxiv.org/abs/2304.03442

- Critiques of LLM Social Simulation: Zeng, Brown, and Rounsevell argue that LLM agents may be narratively realistic but methodologically difficult to use as explanatory social models. Our contribution can be framed as making one part of that problem measurable: whether the representation supplied to the LLM spans the revealed behavior it is supposed to simulate. See Zeng et al. (2026): https://www.nature.com/articles/s44260-026-00075-1

## Poster And Paper Messaging

Current poster title:

> Toward LLM Personas That Span Human Behavioral Diversity

This is good for a poster because it is clear and non-overclaiming. For the paper, a stronger title may be:

> Toward Behaviorally Complete Social Agents

or:

> Diverse Personas, Narrow Behaviors: Revealed-Behavior Audits of LLM Social Agents

Suggested poster/paper thesis:

> Existing persona libraries can look diverse in text but remain incomplete for strategic human behavior. We evaluate this by asking persona-conditioned LLMs to choose among real human trajectories, then measuring whether the selected trajectories span the empirical behavior distribution. This reveals whether a persona source has the behavioral support needed for simulation before any downstream calibration.

Suggested future-direction language:

> We propose to learn the behavioral basis of social agents: the smallest set of persona dimensions that spans human behavioral diversity across held-out strategic environments.

This is stronger than "we will vary dimensions." It says the next step is to identify the rank and content of the behavioral representation:

- Which axes matter most?
- How many dimensions are needed before behavior-space support saturates?
- Which common persona dimensions are irrelevant in a target environment?
- Which latent traits explain behavior better than demographics or generic personality descriptions?

This framing also clarifies why the project is not only about LLM simulation accuracy. It is also about using LLMs and revealed human behavior to discover which latent dimensions matter for social and behavioral science.
