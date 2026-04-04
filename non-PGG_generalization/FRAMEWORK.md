# A Credibility Framework for LLM Agent Simulation in Social Science

## 1. The Central Question

Large language models can simulate human participants in economic experiments. But **what exactly do these simulations identify?** Under what conditions can we treat simulated data as credible evidence about human behavior?

This document proposes a framework — analogous to how Imbens (1994) clarified the interpretation of instrumental variables via the Local Average Treatment Effect (LATE) — for understanding **what LLM agent simulations can and cannot tell us**, and how to build credibility for simulation-based results.

---

## 2. The LATE Analogy: What Do LLM Simulations Identify?

### 2.1 The IV/LATE Lesson

Before Imbens and Angrist (1994), researchers used instrumental variables to estimate treatment effects but lacked a clear interpretation of *whose* treatment effect was being estimated. LATE clarified:

- IV does **not** estimate the Average Treatment Effect (ATE) for the full population
- IV estimates the treatment effect for **compliers** — those whose treatment status is changed by the instrument
- This is a **well-defined, interpretable quantity**, even if it is local rather than global

The key contribution was not a new estimator, but a **framework for interpreting what the estimator identifies** and the **assumptions under which it does so** (monotonicity, exclusion restriction, independence).

### 2.2 The LLM Simulation Parallel

Similarly, when we use LLMs to simulate experimental participants:

- LLM simulations do **not** recover the full human data-generating process (DGP)
- Under explicit assumptions, they can recover **specific, well-defined parameters** of the human DGP
- The key is to be precise about **what is identified**, **for whom**, and **under what conditions**

We define the **Simulable Treatment Effect (STE)** as the component of human behavioral variation that an LLM can credibly approximate — analogous to how LATE defines the component of the treatment effect that IV identifies.

### 2.3 What Makes Behavior "Simulable"?

The critical refinement: **simulable behavior is incentive-compatible behavior**. An LLM can credibly predict or augment behavioral data from economic experiments where:

1. **Participants face real monetary incentives** tied to their decisions (not hypothetical scenarios)
2. **The mechanism is transparent** — the mapping from action to payoff is clear enough that observed actions reflect true preferences rather than confusion
3. **Observed actions are revealed preferences** — because of (1) and (2), the data is a clean signal of the individual's underlying type

This is the demarcation line: LLM simulation targets **incentive-compatible experimental behavior**, not survey responses, not hypothetical choices, not self-reports. Survey and interview data are valuable as *inputs* (persona construction), but the *output* we want to predict or augment is incentive-compatible behavior.

### 2.4 Mapping the Assumptions

| IV/LATE Framework | LLM Simulation Framework |
|-------------------|--------------------------|
| **Exclusion restriction**: instrument affects outcome only through treatment | **No training leakage**: LLM predictions are not driven by memorization of experimental stimuli (Ludwig et al., 2025) |
| **Independence**: instrument is as-good-as-randomly assigned | **Moment preservation**: LLM errors do not systematically correlate with covariates of interest (Ludwig et al., 2025; Hullman et al., 2026) |
| **Monotonicity**: instrument shifts treatment in one direction | **Behavioral invariance**: the behavioral theory encoded in prompts is stable across configurations (Manning & Horton, 2026) |
| **Compliers**: the identified subpopulation | **Incentive-compatible tasks**: the identified class of behaviors |
| LATE = E[Y(1) - Y(0) \| complier] | STE(s,s') = E[Y*(s') - Y*(s) \| incentive-compatible task] |

---

## 3. Two Types of Data, One Generative Bridge

### 3.1 The Fundamental Data Requirements

We argue that credible LLM simulation requires **two complementary types of data**:

```
                    ┌──────────────────────────────┐
                    │   GENERATIVE LANGUAGE MODEL   │
                    │   (the bridge / interpolator) │
                    └──────┬───────────────┬────────┘
                           │               │
              ┌────────────▼──┐     ┌──────▼──────────┐
              │  TYPE A DATA  │     │   TYPE B DATA    │
              │   Interview/  │     │   Incentive-     │
              │   Persona     │     │   Compatible     │
              │   (who they   │     │   Behavioral     │
              │    are)       │     │   (what they do) │
              └───────────────┘     └──────────────────┘
```

**Type A: Interview / Persona Data** captures *who the person is* — demographics, psychological traits, values, preferences, life narratives. This data is:
- Rich, high-dimensional, and qualitative
- Captures the *identity* and *context* of the decision-maker
- Example in this repo: **Twin-2K-500** (2,058 US participants with comprehensive survey data across demographics, personality, cognitive tasks, and economic preferences)

**Type B: Incentive-Compatible Behavioral Data** captures *what the person does* under real economic incentives — contributions in public goods games, trust decisions, punishment choices. This data is:
- Precise, quantitative, and causally identified
- Captures *revealed preferences* under specific experimental configurations
- The target of LLM prediction and augmentation
- Example in this repo: **PGG data** (7,354 participants across parameterized Public Goods Game configurations with punishment, reward, and contribution decisions)

### 3.2 Why Both Types Are Necessary

Neither type alone is sufficient:

- **Type A alone** (persona without behavior): Rich descriptions of who someone is, but no ground truth about what they actually do under incentives. LLM predictions conditioned on persona may be plausible but uncalibrated.
- **Type B alone** (behavior without persona): Precise behavioral measurements, but limited to the specific experimental configurations that were run. Cannot generalize to new configurations or new populations without additional structure.

**Together**, they enable a generative approach: the LLM learns the mapping from *persona* (Type A) to *incentive-compatible behavior* (Type B) within the observed configurations, then can **generate credible predictions for new configurations or new personas**.

### 3.3 The Generative Persona Method

Following Paglieri et al. (2026), rather than conditioning on a fixed set of demographics, we can generate diverse synthetic personas that span the full support of human behavioral variation:

- **Density matching** (standard approach): match the modal/average behavior → misses tails
- **Support coverage** (Persona Generators): span the full space of possible behaviors → captures rare but consequential behavioral types

This is critical because economic experiments often care about heterogeneity (e.g., conditional cooperators vs. free riders), not just averages.

---

## 4. Two Operations: Prediction and Augmentation

Drawing from Hullman et al. (2026) and the applied literature, we distinguish two fundamental operations. Both target **incentive-compatible behavior** as their output.

### 4.1 Prediction: Fixed Bases, New Stimuli

**Definition**: Given a population of agents (personas/types) calibrated on one experimental configuration, predict behavior under a *new configuration* where no human data exists.

```
Configuration A (observed)     Configuration B (novel)
   Human data exists    ──►    No human data
   LLM calibrated here  ──►    LLM predicts here
   "Seed games"         ──►    "Target games"
```

**Key idea** (Manning & Horton, 2026): The behavioral *bases* (agent types, strategic reasoning levels, social preference parameters) are held fixed. Only the experimental *stimuli* change — different endowments, different MPCR, different group sizes, different game structures.

**Credibility requirements**:
1. Theory-grounded prompts (not arbitrary prompt engineering)
2. Cross-DGP validation (test on structurally distinct but theoretically related games)
3. Pre-committed population of target settings for valid inference
4. **Both seed and target tasks must be incentive-compatible** — the behavioral data being predicted must be from real-stakes experiments

**Example in this repo**: Calibrate agent types on PGG with punishment (observed), predict behavior in PGG without punishment, or PGG with reward only, or Trust Games (novel configurations).

### 4.2 Augmentation: Expanding the Sample

**Definition**: Given a small sample of human responses in a specific configuration, augment with LLM-predicted responses to obtain more precise parameter estimates.

```
Small human sample (n)  +  Large LLM sample (N >> n)
        ↓                           ↓
  Dshared (gold standard)    DLLM (predicted)
        ↓                           ↓
  ────── Statistical Calibration ──────
        ↓
  θ̂_corrected = θ̂_base + Δ̂
  (unbiased, more precise)
```

**Key methods**:
- **PPI (Prediction-Powered Inference)** (Angelopoulos et al., 2023; Broska et al., 2025): uses LLM predictions as "surrogate" measurements, corrected by a rectifier learned from jointly-labeled data
- **Plug-in bias correction** (Ludwig et al., 2025): learns a conditional bias function b(x) = E[f̂(X) - Y | X = x] from shared data, then debiases LLM predictions before plugging into the estimating equation

**The estimator** (PPI form):

```
θ̂_PPI = (1/n)Σ Y_i  -  λ[(1/n)Σ f̂(X_i) - (1/N)Σ f̂(X̃_j)]
         ↑ human mean     ↑ rectifier (bias correction)
```

Where λ is tuned to minimize variance. The resulting estimator is:
- **Consistent** for the human target parameter (no LLM-induced bias)
- **At least as precise** as the human-only estimator
- **More precise** when the LLM is a reasonably good predictor

### 4.3 The Prediction-Augmentation Duality

These two operations are complementary:

| | Prediction | Augmentation |
|---|---|---|
| **What changes** | Experimental configuration (stimuli) | Sample size (precision) |
| **What's fixed** | Agent population (bases/types) | Experimental configuration |
| **Target output** | Incentive-compatible behavior in novel setting | More precise estimates in observed setting |
| **Validation** | Cross-DGP generalization test | PPI/calibration unbiasedness |
| **When to use** | Exploring new designs before running | Boosting power of existing experiments |
| **Risk** | Wrong theory → wrong predictions | Poor LLM → small precision gains |

---

## 5. The Credibility Hierarchy

We propose a hierarchy of credibility claims, from weakest to strongest:

### Level 0: Heuristic ("it looks right")
- LLM responses qualitatively resemble human responses
- No formal guarantees; vulnerable to memorization, systematic bias, effect size inflation
- **Use case**: Exploratory hypothesis generation, design piloting

### Level 1: Predictive Validation ("it generalizes")
- Theory-grounded agents predict human behavior in held-out settings
- Cross-DGP validation (training and testing on structurally distinct games)
- Pre-committed population of settings for externally valid error estimates
- **Use case**: Forecasting incentive-compatible behavior in novel experimental configurations (Manning & Horton, 2026)

### Level 2: Statistical Calibration ("it's unbiased")
- Formal combination of human and LLM data with explicit bias correction
- Consistent, asymptotically normal estimators with valid confidence intervals
- Does not require LLM to be unbiased — only requires shared labeled data for calibration
- **Use case**: Efficient estimation of treatment effects with mixed human/LLM samples (Broska et al., 2025; Ludwig et al., 2025)

### Level 3: Generative Digital Twin ("it's the person")
- Agent grounded in comprehensive individual-level data (interview + behavioral)
- Predicts individual-level responses with accuracy approaching test-retest reliability
- Validated against held-out behaviors for the same individuals
- **Use case**: Counterfactual analysis, personalized policy evaluation (Toubia et al., 2025; Park et al., 2024)

```
Level 3: Digital Twin ─────────── Individual-level counterfactuals
   ↑ requires Type A + Type B per individual
Level 2: Calibration ─────────── Unbiased aggregate parameters
   ↑ requires shared labeled data (Dshared)
Level 1: Prediction ──────────── Cross-DGP generalization
   ↑ requires theory + validation data
Level 0: Heuristic ───────────── Exploratory only
   ↑ requires nothing beyond LLM access
```

---

## 6. Application: Twin-2K-500 × PGG Configurations

### 6.1 The Setup

This repo contains the ingredients for a Level 1–2 credibility demonstration:

**Type A (Persona Data)**:
- **Twin-2K-500**: 2,058 US participants with comprehensive persona data (demographics, Big Five personality, cognitive tasks, risk preferences, social preferences, time preferences, heuristics and biases experiments)
- Rich enough to construct diverse, theory-grounded agent populations

**Type B (Incentive-Compatible Behavioral Data)**:
- **PGG high-throughput data** (OSF 2d56w): 7,100 participants in parameterized Public Goods Games with punishment/reward/contribution mechanisms
- Demographics available + behavioral archetype oracle descriptions (contribution patterns, punishment behavior, reward behavior, conditional cooperation)
- Multiple configurations: different group sizes, MPCR values, punishment technologies, reward mechanisms
- All core decisions (contribute, punish, reward) are incentive-compatible — real money at stake, transparent mechanisms

**Non-PGG Transfer Targets** (all incentive-compatible):
- Longitudinal Trust Game (ht863): Trust decisions under uncertainty, 10 sessions, token-to-cash conversion
- Two-Stage Trust/Punishment (y2hgu): Deliberation × trustworthiness signaling, 5 experiments, real payoffs
- Minority Game + BRET (njzas): Strategic coordination + risk preferences, 11 rounds, random-round payment
- Multi-Game Battery (fvk2c): UG, TG, PD, SH, Coordination + LLM delegation, ECU-to-GBP conversion

### 6.2 The Concrete Pipeline (Grounded in Existing Repo Infrastructure)

The repo already contains a multi-stage persona pipeline that implements much of this framework:

```
Step 1: CONSTRUCT AGENT POPULATION (existing infrastructure)
  ┌─ Twin-2K-500 (2,058 US participants)
  │   ├── 256 survey questions across 43 blocks
  │   ├── Demographics, Big Five, BDI, Need for Cognition, Empathy
  │   ├── Trust/Ultimatum/Dictator game responses
  │   ├── Risk & time preferences, cognitive tests, heuristics
  │   └── Consumer pricing experiments
  │
  ├── build_twin_extended_profiles.py → Extended Profile Cards
  │   7 PGG-relevant cues per participant:
  │   (1) Cooperation orientation
  │   (2) Conditional cooperation (reciprocity/fairness sensitivity)
  │   (3) Norm enforcement (unfair split resistance)
  │   (4) Generosity without return
  │   (5) Exploitation caution
  │   (6) Communication/coordination style
  │   (7) Behavioral stability
  │
  └── sample_twin_personas_for_pgg_validation.py
      → Demographic-matched sampling (age × education × sex grid)
      → Maps Twin personas to PGG validation population

  ┌─ PGG Behavioral Archetypes (7,954 oracle descriptions)
  │   Rich text describing each participant's:
  │   - Contribution patterns (unconditional, conditional, free-rider)
  │   - Punishment behavior (frequency, targeting, retaliation)
  │   - Reward behavior (reciprocity, alliance signaling)
  │   - Response to others' outcomes (tolerance, strategic adjustment)
  │
  └── demographics_numeric_learn_val_consolidated.csv (7,354 participants)
      age, gender, education across learning + validation waves

Step 2: PREDICTION (within PGG, new configurations)
  Training: Observed PGG configs (from OSF 2d56w, parameterized punishment/reward)
  Target:   Novel PGG configs (different group size, MPCR, mechanism)
  Method:   3-stage LLM pipeline (already implemented):
            (a) Persona inference from observed early-round behavior
            (b) Retrieval query generation → match Twin profiles via behavioral cues
            (c) Prediction conditioned on retrieved persona + game rules
  Metric:   KL divergence, Wasserstein distance to human distributions

Step 3: AUGMENTATION (within PGG, boosting precision)
  Dshared:  Small human sample in target configuration (n ~ 100–500)
  DLLM:     Large LLM sample conditioned on Twin-2K-500 personas (N ~ 5,000–10,000)
  Method:   PPI / plug-in bias correction (Broska et al., 2025; Ludwig et al., 2025)
  Metric:   Effective sample size, CI width reduction

Step 4: TRANSFER (PGG → non-PGG games)
  Test whether agents calibrated on PGG generalize to:
  - Trust games (ht863: longitudinal WTP trust, y2hgu: deliberation + trustworthiness)
  - Punishment/helping games (y2hgu: cost/impact checking × observability)
  - Strategic coordination (njzas: minority game with herding/contrarian dynamics)
  - Risk elicitation (njzas: BRET boxes_collected as risk tolerance measure)
  - Multi-game batteries (fvk2c: UG, TG, PD, SH, Coordination + AI delegation)
  This tests the deepest claim: that the behavioral bases
  (social preferences, risk attitudes, strategic reasoning)
  are truly fixed across game families.
```

### 6.3 Existing MobLab Inference Pipeline

The repo's `pgg_transfer_eval/` already implements a 3-stage LLM-based transfer pipeline:

| Stage | Script | What it does |
|-------|--------|-------------|
| **Persona Inference** | `build_moblab_persona_batch.py` | Infers compact behavioral personas from observed MobLab game behavior |
| **Retrieval Query** | `build_moblab_retrieval_query_batch.py` | Translates inferred personas into retrieval queries matching PGG archetype library |
| **Prediction** | `build_moblab_prediction_batch.py` | Generates behavioral predictions conditioned on persona + retrieved candidates + game rules |

Supported baselines: `direct` (no persona), `persona` (inferred), `meta_persona` (retrieved), `retrieval` (full pipeline).

### 6.4 What This Tests

| Claim | Test | Data |
|-------|------|------|
| Agents generalize across PGG configurations | Predict new PGG params from old | PGG data |
| Personas improve prediction over demographics alone | Compare Twin-2K-500 agents vs. demographic-only agents | Twin-2K-500 + PGG |
| Augmentation provides precision gains | Compare PPI estimates to human-only estimates | PGG Dshared + DLLM |
| Behavioral bases transfer across game families | PGG-calibrated agents predict Trust/Punishment/Coordination | PGG + non-PGG datasets |
| Credibility hierarchy holds | Level 1 > Level 0, Level 2 > Level 1 in practice | All datasets |

---

## 7. Formal Notation (Compact)

Let:
- **w** ∈ W = persona/type (demographics, personality, interview data)
- **s** ∈ S = stimulus/configuration (game rules, endowments, MPCR, group size)
- **Y(w, s)** = human behavioral outcome (contribution, trust, cooperation) — **incentive-compatible**
- **f̂(w, s)** = LLM-predicted outcome
- **b(w, s)** = E[f̂(w, s) - Y(w, s)] = conditional bias of LLM

**The simulable domain**: The set of experimental tasks where observed behavior reveals true preferences because the task is incentive-compatible with real stakes and transparent mechanisms. Within this domain, Y(w, s) ≈ σ*(type(w), s), where σ* is the optimal strategy for type w.

**Prediction task**: Given calibrated f̂(·, s_train), predict Y(w, s_target) for novel s_target ∈ S.

**Augmentation task**: Given D_shared = {(w_i, s, Y_i, f̂_i)}_{i=1}^n and D_LLM = {(w̃_j, s, f̂_j)}_{j=1}^N, estimate θ* = E[Y | s] with:

```
θ̂_corrected = θ̂_LLM - b̂    (plug-in correction)

or

θ̂_PPI = θ̂_H - λ(θ̂_H_LLM - θ̂_LLM)    (PPI correction)
```

Both are consistent for θ* under:
1. LLM is trained independently of D_shared (weaker than No Leakage)
2. The bias b(w, s) can be consistently estimated from D_shared

**The STE**: The Simulable Treatment Effect for two incentive-compatible configurations s, s':

```
STE(s, s') = E[Y*(s') - Y*(s)]
```

This is identified by the LLM when the behavioral bases (social preferences, risk attitudes, strategic reasoning) are stable across configurations — the behavioral invariance assumption (Manning & Horton, 2026).

**The LATE parallel**: Just as θ_LATE = E[Y(1) - Y(0) | complier], the STE identifies:

```
STE(s, s') = E[Y*(s') - Y*(s) | task is incentive-compatible]
```

The STE equals the population treatment effect when the task is incentive-compatible and the LLM preserves moment conditions. The scope of credible simulation — how many tasks, how many configurations — is an empirical question that depends on the LLM, the persona data, and the behavioral domain.

---

## 8. References

### Frameworks
1. Hullman, J., Broska, D., Sun, H., & Shaw, A. (2026). This human study did not involve human subjects: Validating LLM simulations as behavioral evidence. arXiv:2602.15785. **[Credibility framework: heuristic vs. calibration vs. exploratory]**
2. Ludwig, J., Mullainathan, S., & Rambachan, A. (2025). Large Language Models: An Applied Econometric Framework. NBER Working Paper. **[Formal conditions: No Training Leakage + Moment Preservation]**
3. Angelopoulos, A. N. et al. (2023). Prediction-Powered Inference. Science. **[PPI estimator for combining human + LLM data]**
4. Imbens, G. W. & Angrist, J. D. (1994). Identification and Estimation of Local Average Treatment Effects. Econometrica. **[The LATE analogy]**

### Prediction
5. Manning, B. S. & Horton, J. J. (2026). General Social Agents. arXiv:2508.17407. **[Theory-grounded prompts, cross-DGP validation, 883K novel games]**
6. Kazinnik, S. (2024). Bank Run, Interrupted: Modeling Deposit Withdrawals with Generative AI. SSRN 4656722. **[Applied prediction: demographic-conditioned LLM agents simulate depositor panic behavior]**

### Augmentation
7. Broska, D. et al. (2025). Mixed Subjects Design / PPI for Behavioral Science. SSRN 6343598. **[Augmenting human samples with LLM predictions for tighter CIs]**

### Persona Generation
8. Paglieri, D. et al. (2026). Persona Generators: Generating Diverse Synthetic Personas at Scale. arXiv:2602.03545. **[Support coverage > density matching; AlphaEvolve-optimized persona functions]**
9. Park, J. S. et al. (2024). Generative Agents: Interactive Simulacra of Human Behavior. **[Interview-grounded digital twins, 85% of human test-retest accuracy]**
10. Toubia, O. et al. (2025). Twin-2K-500: A Dataset for Building Digital Twins of 2,000 People. arXiv:2505.17479. **[2,058 participants, 4-wave survey, comprehensive persona data]**
