# Statistical Simulation Retrospective

Date: 2026-03-10

## Purpose

This report summarizes what we were trying to do in `simulation_statistical`, what we built, how each modeling branch performed, and what the current verdict is.

The original goal was not just to predict aggregate normalized efficiency. The broader goal was:

1. use LLM-derived archetype summaries to recover a latent representation of player behavior,
2. learn how `CONFIG` changes the population distribution of those latent types,
3. use that latent structure to drive a simulator that can say something meaningful about both:
   - macro outcomes under new designs,
   - and micro-level behavioral changes underneath those outcomes.

Over time, two related questions emerged:

1. Can this beat or at least complement a direct `CONFIG -> macro outcome` baseline such as linear regression?
2. Even if it does not beat linear regression on macro prediction, can it still provide a useful behavioral microscope for how institutional design changes player behavior?

## Data and Evaluation Setup

### Training / validation split

- Learning wave:
  - used to train all statistical models and family/cluster assignments
- Validation wave:
  - used for all held-out evaluation

### Evaluation modes

- `micro`
  - one-step prediction against held-out human decisions
  - useful for contribution and action realism
- `macro`
  - full rollout from round 1 onward
  - useful for treatment-level outcome prediction and dynamic behavior

### Two macro evaluation protocols

We used two macro protocols.

1. `40-context treatment average`
   - one representative validation context per treatment
   - faster, but noisy because each treatment gets one stochastic rollout
2. `raw-417 then aggregate`
   - run all 417 raw validation games
   - aggregate simulated outcomes back to the 40 treatment groups
   - this is the fairer treatment-level evaluation

This distinction mattered a lot. Some models looked noticeably better or worse once the evaluation averaged many stochastic rollouts per treatment instead of relying on one representative context.

## What We Built

## 1. Embedding-Based Archetype Pipeline

We first implemented the LLM-summary embedding pipeline in `simulation_statistical/archetype_distribution_embedding`.

Pipeline:

1. clean learning/validation archetype summaries,
2. embed summaries with OpenAI embeddings,
3. reduce dimensionality,
4. fit a GMM over archetype embeddings,
5. assign player-level cluster weights,
6. aggregate those weights to game-level simplex targets,
7. fit a Dirichlet environment model from `CONFIG` to cluster distribution.

Headline diagnostics:

- selected cluster count: `K=6`
- assignment entropy was very low, so the GMM behaved almost like a hard taxonomy
- validation env-model performance:
  - mean cluster MAE `0.1437`
  - average L1 distance `0.8625`
  - JS divergence `0.1757`

Interpretation:

- the clustering step was not degenerate,
- the `CONFIG -> cluster distribution` mapping was learning real signal,
- but the downstream simulator still had to translate that latent structure into realistic behavior.

## 2. `archetype_cluster`

This was the first stable cluster-conditioned simulator.

Design:

- sample one hard cluster per player from the Dirichlet `CONFIG -> cluster mixture` model,
- sample contribution from cluster-conditioned empirical pools,
- sample punishment/reward from cluster-conditioned empirical action models.

This is implemented in `simulation_statistical/trained_policy.py`.

This model turned out to be the strongest learned simulator branch overall.

### Standard 40-treatment macro evaluation

| model | MAE | RMSE | Corr | Mean Sim |
|---|---:|---:|---:|---:|
| `archetype_cluster` | 0.1349 | 0.1643 | 0.4954 | 0.6030 |
| `random_baseline` | 0.2012 | 0.2209 | 0.3897 | 0.5008 |
| `linear_config` | 0.0982 | 0.1230 | 0.4476 | 0.6900 |

### Fair raw-417 treatment-aggregate macro evaluation

| model | MAE | RMSE | Corr | Mean Sim |
|---|---:|---:|---:|---:|
| `archetype_cluster` | 0.1061 | 0.1241 | 0.6912 | 0.6109 |
| `linear_config` | 0.0982 | 0.1230 | 0.4476 | 0.6900 |

Interpretation:

- `archetype_cluster` remained under the human mean contribution / efficiency level,
- but it was the strongest learned model on treatment-level correlation,
- and it clearly beat `random_baseline` on macro fit.

This was the first strong signal that the archetype-distribution idea itself was useful.

## 3. `history_archetype`

This was the first attempt to condition actions on structured history as well as cluster.

Design:

- hard cluster as one input feature,
- compressed summary-history features,
- gradient-boosted contribution and action heads.

Implemented in `simulation_statistical/history_conditioned_policy.py`.

Results:

### Micro

| model | Contrib MAE | Contrib Corr | Action Exact | Target F1 |
|---|---:|---:|---:|---:|
| `history_archetype` | 4.9594 | 0.3360 | 0.6646 | 0.6923 |
| `archetype_cluster` | 7.8772 | 0.0248 | 0.6897 | 0.6958 |

### Macro, 40-treatment

| model | MAE | RMSE | Corr | Mean Sim |
|---|---:|---:|---:|---:|
| `history_archetype` | 0.3134 | 0.4022 | 0.3532 | 0.4033 |
| `archetype_cluster` | 0.1349 | 0.1643 | 0.4954 | 0.6030 |

Interpretation:

- history helped one-step contribution prediction a lot,
- but rollout quality got much worse,
- the model over-rewarded and became too pessimistic at the macro level.

This was the first major sign that better micro imitation was not automatically translating into better macro simulation.

## 4. `exact_sequence_*`

We next tried a much stronger history-conditioned family, eventually implemented as structured exact-sequence state in `simulation_statistical/structured_sequence_policy.py`.

This branch went through several stages:

### 4.1 `exact_sequence_structured`

Design:

- exact ordered history stored as structured state,
- per-target categorical action head,
- continuous contributions initially modeled poorly,
- units fixed to `1`.

Micro result:

- action realism became very strong:
  - action exact match `0.7990`
  - target F1 `0.8060`
- but contribution modeling was weak:
  - contribution MAE `7.8333`

Macro result:

- MAE `0.2870`
- RMSE `0.3340`
- correlation `0.0364`

Failure mode:

- continuous contributions collapsed,
- punish/reward nearly disappeared,
- treatment discrimination collapsed.

### 4.2 `exact_sequence_binned`

To fix the continuous contribution collapse, we replaced the continuous regressor with a 5-bin contribution classifier over `{0, 5, 10, 15, 20}`.

This materially improved micro contribution prediction:

- contribution MAE `4.1072`
- contribution correlation `0.2632`

Macro improved relative to the first structured version but was still poor:

- MAE `0.2293`
- RMSE `0.2698`
- correlation `0.3005`
- mean simulated efficiency `0.8780`

The model became too cooperative and still almost never punished or rewarded.

### 4.3 `exact_sequence_clustercal`

We then added cluster-specific calibration on top of the binned contribution head.

Micro changed only marginally:

- contribution MAE `4.1057`
- contribution correlation `0.2639`

Macro improved modestly:

- MAE `0.2209`
- RMSE `0.2597`
- correlation `0.3493`
- mean simulated efficiency `0.8637`

Still, the model remained too cooperative and largely action-collapsed.

### 4.4 `exact_sequence_history_only`

We also ran an ablation with no cluster at all.

Micro:

- contribution MAE improved slightly further to `4.0572`

Macro:

- MAE worsened to `0.2696`
- correlation worsened to `0.2980`

Interpretation:

- cluster did not help much for one-step micro prediction once rich history was available,
- but it still helped macro treatment discrimination.

### Verdict on the exact-sequence branch

This branch improved one-step micro behavior but did not produce a trustworthy rollout simulator.

The main failure mode was:

- contribution policy drifted toward very high cooperation,
- action policy collapsed toward no punishment / no reward,
- or earlier versions overfit to local history and drifted badly under rollout.

The conclusion was that this branch was not the right place to keep investing for macro simulation.

## 5. `archetype_cluster_plus`

We then returned to the stable `archetype_cluster` engine and added only a small amount of structured visible history.

Design:

- keep the `CONFIG -> cluster distribution -> sampled cluster` structure,
- add last-round own contribution,
- last-round peer mean,
- whether the player was punished/rewarded last round,
- current-round contribution context for sanctioning,
- explicit empirical sanction calibration.

Implemented in `simulation_statistical/cluster_plus_policy.py`.

### Micro

| model | Contrib MAE | Contrib Corr | Action Exact | Target F1 |
|---|---:|---:|---:|---:|
| `archetype_cluster_plus` | 5.1950 | 0.2513 | 0.6910 | 0.6985 |
| `archetype_cluster` | 7.8772 | 0.0248 | 0.6897 | 0.6958 |

### Fair raw-417 treatment-aggregate macro

| model | MAE | RMSE | Corr | Mean Sim |
|---|---:|---:|---:|---:|
| `archetype_cluster_plus` | 0.1452 | 0.1673 | 0.5520 | 0.5674 |
| `archetype_cluster` | 0.1061 | 0.1241 | 0.6912 | 0.6109 |

Interpretation:

- `cluster_plus` was a real micro improvement,
- it also brought punish/reward rates closer to human rates,
- but it weakened the clean treatment-level macro signal already present in `archetype_cluster`.

This reinforced the pattern:

- extra history can improve local realism,
- but it can also wash out the treatment-level latent signal.

## 6. Oracle Cluster Tests

We tested whether the main bottleneck was the `CONFIG -> cluster distribution` mapping itself by replacing predicted cluster distributions with oracle treatment-level distributions derived from validation-wave archetype outputs.

Result:

- for the exact-sequence branch, oracle cluster distributions did not rescue rollout quality,
- for `archetype_cluster`, oracle sometimes improved MAE but hurt treatment discrimination,
- under the fair raw-417 aggregate, the oracle simple cluster model had:
  - MAE `0.0970`
  - RMSE `0.1244`
  - correlation `0.4535`
- regular `archetype_cluster` still had much stronger treatment correlation at `0.6912`.

Interpretation:

- the upstream `CONFIG -> cluster` mapping was not the main bottleneck,
- the downstream behavior model was the larger problem.

## 7. Algorithmic-Latent Redesign

Because anonymous clusters were not interpretable enough, and because the richer history simulators were unstable, we started a separate redesign in `simulation_statistical/algorithmic_latent`.

Goal:

- replace opaque cluster IDs with interpretable algorithmic families,
- fit those families from human traces,
- learn `CONFIG -> family mixture`,
- eventually simulate behavior from those algorithmic latents.

### What was built

1. visibility-aware contribution/action state tables,
2. first family library with 8 families:
   - `unconditional_cooperator`
   - `unconditional_defector`
   - `conditional_cooperator`
   - `generous_conditional_cooperator`
   - `endgame_defector`
   - `retaliatory_punisher`
   - `norm_enforcer`
   - `reward_oriented_cooperator`
3. family-specific contribution and action heads,
4. player-level posterior inference over families,
5. Dirichlet `CONFIG -> family mixture` model,
6. first family-based simulator runtime.

### Environment-family mixture results

Validation performance of `CONFIG -> family mixture`:

| split | mean family MAE | avg L1 | top-family accuracy |
|---|---:|---:|---:|
| validation game | 0.0330 | 0.2637 | 0.5420 |
| validation treatment | 0.0205 | 0.1641 | 0.5750 |

This was clearly better than a mean-family baseline, so the new latent representation was not vacuous.

### Family posterior sharpness

However, player posteriors were still very diffuse:

- mean top-family probability:
  - learning `0.266`
  - validation `0.272`

Interpretation:

- the family library was still too broad or too overlapping,
- the inference problem was underidentified,
- the family representation was not yet sharp enough to support clean simulation.

### First simulator result

The first family-based simulator produced:

- decent micro contribution prediction,
- but action rates were massively too high.

After post-hoc action-rate calibration:

### Micro, calibrated

| model | Contrib MAE | Contrib Corr | Action Exact | Target F1 | Pred Punish | Pred Reward |
|---|---:|---:|---:|---:|---:|---:|
| `algorithmic_latent_calibrated` | 6.2285 | 0.3873 | 0.6599 | 0.6679 | 0.1421 | 0.0934 |
| human actual | - | - | - | - | 0.0671 | 0.1314 |

### Macro, calibrated, 40-treatment

| model | MAE | RMSE | Corr | Mean Sim |
|---|---:|---:|---:|---:|
| `algorithmic_latent_calibrated` | 0.3511 | 0.4255 | 0.1379 | 0.3714 |
| `algorithmic_latent_uncalibrated` | 0.5284 | 0.7240 | 0.2721 | 0.1889 |
| `archetype_cluster` | 0.1349 | 0.1643 | 0.4954 | 0.6030 |

Interpretation:

- the first family runtime was usable as a diagnostic,
- the action-rate calibration fixed a real pathology,
- but the simulator remained far weaker than `archetype_cluster`,
- and still too punitive / low-efficiency.

## What We Learned

## A. The archetype-distribution idea survived

The strongest result in the whole project is that the simple `archetype_cluster` model remained good under the fairer raw-417 aggregate evaluation:

- MAE `0.1061`
- RMSE `0.1241`
- correlation `0.6912`

That means `CONFIG -> latent composition -> rollout` can work at the treatment level.

## B. The downstream behavior model was the real bottleneck

Replacing predicted cluster distributions with oracle ones did not fix the richer simulators.

That means:

- the `CONFIG -> cluster` mapping was not the main issue,
- the main issue was how cluster or family latents were translated into behavior under rollout.

## C. History helped micro, often hurt macro

This was the most repeated pattern across branches:

- more history almost always improved contribution prediction,
- but richer history-conditioned models tended to wash out treatment signal,
- or produce unstable closed-loop dynamics.

## D. Linear regression remained unbeatable for direct macro prediction

If the question is only:

- "what is the normalized efficiency for this treatment?"

then `linear_config` remained the best reduced-form answer on the standard 40-treatment evaluation.

This does not invalidate simulation. It changes what the simulation is good for:

- not necessarily best reduced-form macro prediction,
- but potentially a more interpretable behavioral account of what changed underneath the macro outcome.

## E. The LLM was underused in the first phase

In the original cluster pipeline, the LLM was used only once:

- summary text -> embedding

After that, everything was classical statistical modeling.

That made the latent representation opaque and gave the LLM no explicit role in reasoning about how `CONFIG` changes incentives, observability, coordination, or sanctioning.

The algorithmic-latent redesign is an attempt to fix that by making the latent variables themselves behaviorally meaningful.

## F. The current algorithmic-latent branch is promising, but not ready

It already showed:

- `CONFIG` can predict the new family mixtures better than baseline,
- the redesign is not vacuous.

But it also showed:

- family posteriors are too diffuse,
- the first family-based simulator over-fired actions,
- simple global rate calibration was not enough.

So the branch is conceptually promising, but still at an early stage.

## Current Recommendation

### For production-like macro simulation now

Use:

- `archetype_cluster` as the main learned simulator,
- `linear_config` as the reduced-form macro benchmark,
- `archetype_cluster_plus` as a more behaviorally informative micro-side diagnostic, not as the main macro engine.

### For the next research step

Continue the `algorithmic_latent` branch, but do not keep calibrating the current runtime in small increments.

The next meaningful redesign should be:

1. split latent state into at least:
   - contribution family
   - sanction family
2. replace direct per-target action sampling with hierarchical gating:
   - any punish?
   - any reward?
   - then target choice
3. sharpen or simplify the family library so posterior inference is less diffuse
4. keep the LLM in the role of:
   - proposing family structure,
   - proposing priors over parameter signs or treatment effects,
   - not directly predicting outcomes

## Bottom Line

The project did not fail. It clarified the problem.

What worked:

- embedding-based archetype clustering as a treatment-level latent representation,
- simple cluster-conditioned rollout,
- fair raw-game treatment aggregation for evaluation,
- the idea that latent composition matters.

What did not work:

- the richer history-conditioned exact-sequence family as a macro simulator,
- the assumption that better one-step imitation would automatically produce better rollouts,
- the first algorithmic-latent runtime without stronger family/action redesign.

The current best interpretation is:

- latent composition matters,
- `CONFIG` does shift behavior in ways that can be captured statistically,
- but the simulator needs a more interpretable and better-structured latent behavioral model than either anonymous clusters or the current family runtime.
