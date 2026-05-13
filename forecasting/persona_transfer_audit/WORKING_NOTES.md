# Persona Transfer Evaluation Working Notes

Last updated: 2026-05-13

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
- Hu et al., "Population-Aligned Persona Generation for LLM-based Social Simulation" (arXiv:2509.10127). This work generates personas from long-term social media data and aligns them to reference psychometric distributions, including Big Five traits. This is exactly the kind of approach our paper can complement: even if a persona set is aligned on Big Five or other population-level descriptors, we still need to test whether those personas map onto the behavioral support of the target setting.
- PersonaGym, "Evaluating Persona Agents and LLMs" (Samuel et al., arXiv:2407.18416; Findings of EMNLP 2025). PersonaGym evaluates whether persona agents behave consistently across persona-relevant environments and emphasizes that persona adherence is hard to evaluate in free-form settings. This supports our broader claim that persona prompting is a steering problem, not just a prompt-format problem. Our design differs by anchoring the evaluation to real human trajectories in social interactions.

Key distinction to emphasize:

> Persona generation papers ask whether the persona set is diverse or population-aligned. We ask whether those personas, after LLM interpretation, cover the real behavioral support of a target environment.

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
