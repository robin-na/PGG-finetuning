# Persona Transfer Audit

This folder is a working home for the persona-to-revealed-behavior matching project.

The core question is whether a persona library that looks diverse in its source domain remains diverse after an LLM maps those personas into behavior in a target environment such as a public goods game (PGG), chip bargaining task, trust/punishment game, or other strategic interaction.

## Motivation

Current forecasting experiments mostly ask whether profile augmentation improves aggregate prediction. That is useful, but it leaves the central transfer step implicit. If we sample a Twin persona and ask an LLM to simulate a PGG participant, the LLM must decide how that source evidence translates into target-game behavior: how much the person contributes, whether they punish or reward, whether they coordinate in chat, whether they reciprocate, and whether they drift over time.

There is no direct observed mapping from a Twin profile to PGG behavior. The LLM-mediated mapping is therefore not a nuisance to isolate away. It is the object we need to diagnose.

The proposed audit makes this mapping visible. Given a persona and a transcript from a target study, we ask a model which observed participant the persona most resembles. Repeating this over personas and games yields a match distribution over actual target-study participants. That distribution tells us which behavioral types the persona-library-plus-model system covers, overrepresents, or omits.

## Working Claim

Persona libraries should be evaluated not only by their source-population diversity or by aggregate simulation accuracy, but by their revealed-behavior coverage after LLM-mediated transfer into a target environment.

In short:

> A persona library can be diverse in source space but collapse in target-action space.

For example, a Twin-derived library may include varied demographic, cognitive, and social-preference profiles. But when an LLM uses those profiles to reason about PGG behavior, many distinct personas may map onto the same observed type: highly cooperative coordinators, cautious contributors, norm enforcers, free riders, or passive low-communication players. If this mapping is skewed, downstream simulations can look plausible on some aggregate metrics while failing to represent target-study heterogeneity.

## Proposed Assay

For each persona-library condition and target game:

1. Sample personas from the library.
2. Sample observed target-study games.
3. Show the model one persona and one full observed game transcript.
4. Ask the model to rank target-game players by behavioral alignment with the persona.
5. Aggregate the selected/ranked players over persona-game pairs.
6. Compare the matched-player distribution with the actual target-study player distribution.

For PGG, the model sees:

- the persona summary,
- game rules and treatment metadata,
- the observed transcript with contribution, punishment/reward, and chat lines,
- the player labels used in that transcript.

The model returns structured JSON with:

- `most_aligned_player`,
- `least_aligned_player`,
- `player_rankings`,
- `alignment_scores`,
- short rationale fields.

## Estimands

The estimand is the behavior of the full persona-library-plus-model transfer system, not a model-free truth about the persona.

If the LLM has a bias toward selecting cooperative, articulate, normatively attractive, or strategically coherent players, that bias is part of the transfer system researchers would use when simulating new environments. We can and should vary the matcher model to test robustness, but the bias itself is substantively meaningful.

## Main Diagnostics

### Match Concentration

- Top-1 matched-player share.
- Entropy of matched players.
- Effective number of matched players.
- Gini or Herfindahl-Hirschman concentration.
- Concentration within treatment/config.

### Behavioral Skew

Compare matched players with the actual observed player population on:

- mean contribution rate,
- first-round contribution,
- final contribution,
- contribution decay,
- punishment/reward actor rate,
- punishment/reward received rate,
- chat messages per round,
- chat coordination language,
- player-level behavioral clusters.

### Source-to-Target Compression

Compare diversity in source persona features with diversity in matched target behavior:

- Do varied Twin summaries map to the same observed players?
- Are some target behavioral regions never selected?
- Are certain persona-library signals ignored or over-weighted?

### Downstream Prediction Link

The strongest empirical result would show that match skew predicts simulation error:

- If matched PGG players are too cooperative, rollout simulations should overpredict contribution or underpredict decay.
- If matched bargaining players are too fairness-oriented, simulations should overpredict rejection or equalizing offers.
- If matched players are too chatty/coordinating, chat-enabled simulations should overproduce coordination messages.

## Pilot: Twin -> PGG

The first pilot uses Twin persona summaries from SimBench:

- source: `forecasting/simbench/cache/twin_persona_summary_cache.jsonl`
- format: released-style direct persona summary beginning with "The following is a description of a person."

This is intentionally not the PGG-specialized transfer card. The PGG-specialized cards already translate Twin evidence into PGG-relevant cues. For this audit, we want to measure the model-mediated transfer from the direct Twin summary into PGG behavior.

The initial pilot builds a small cross-product:

- deterministic sample of Twin persona summaries,
- deterministic sample of observed PGG games,
- one request per persona-game pair.

The default pilot size is deliberately small so prompts can be inspected and API cost controlled before scaling. The default uses the full direct Twin persona summary. Truncation is available through `--max-persona-chars`, but should be treated as an ablation rather than the primary pilot.

Build the default pilot:

```bash
python3 -B -m forecasting.persona_transfer_audit.build_twin_to_pgg_pilot
```

Default output:

- batch input: `forecasting/persona_transfer_audit/batch_input/twin_direct_summary_to_pgg_pilot__n8_x5__gpt_5_mini__seed_0.jsonl`
- manifest: `forecasting/persona_transfer_audit/metadata/twin_direct_summary_to_pgg_pilot__n8_x5__gpt_5_mini__seed_0/manifest.json`
- sample prompt: `forecasting/persona_transfer_audit/metadata/twin_direct_summary_to_pgg_pilot__n8_x5__gpt_5_mini__seed_0/sample_prompt.txt`

Submit/sync the batch using the existing PGG batch manager, pointed at this manifest:

```bash
python3 -B -m forecasting.pgg.manage_openai_batch sync \
  --manifest-json forecasting/persona_transfer_audit/metadata/twin_direct_summary_to_pgg_pilot__n8_x5__gpt_5_mini__seed_0/manifest.json \
  --wait
```

Parse completed batch output:

```bash
python3 -B -m forecasting.persona_transfer_audit.parse_match_outputs \
  --metadata-dir forecasting/persona_transfer_audit/metadata/twin_direct_summary_to_pgg_pilot__n8_x5__gpt_5_mini__seed_0 \
  --output-jsonl forecasting/persona_transfer_audit/batch_output/twin_direct_summary_to_pgg_pilot__n8_x5__gpt_5_mini__seed_0.jsonl
```

Summarize matched-player concentration:

```bash
python3 -B -m forecasting.persona_transfer_audit.summarize_matches \
  --metadata-dir forecasting/persona_transfer_audit/metadata/twin_direct_summary_to_pgg_pilot__n8_x5__gpt_5_mini__seed_0
```

## Candidate Persona Libraries

Future libraries to test:

- Twin direct persona summaries.
- Twin PGG-specialized cards.
- Demographic-only profiles.
- Tianyi-Lab `Personas` from "LLM Generated Persona is a Promise with a Catch."
- NVIDIA `nemotron-personas`.
- Salesforce `SCOPE-Persona`.
- Personas generated by Paglieri et al. (arXiv:2602.03545).
- Any population-aligned persona generator with released artifacts.

## Literature Positioning

This project is adjacent to, but distinct from, several literatures.

### Silicon Samples And Algorithmic Fidelity

Argyle et al. use language models conditioned on demographic backstories to simulate human samples and introduce the idea of algorithmic fidelity: the model should reproduce response patterns conditional on human attributes.

Reference: Argyle et al., "Out of One, Many: Using Language Models to Simulate Human Samples." Political Analysis. https://www.cambridge.org/core/journals/political-analysis/article/out-of-one-many-using-language-models-to-simulate-human-samples/035D7C8A55B237942FB6DBAD7CAA4E49

Difference: that work focuses mainly on survey response distributions. The transfer audit asks whether source persona libraries cover revealed behavioral heterogeneity in a new strategic environment after LLM-mediated transfer.

### Turing Experiments And LLM Replications

Aher, Arriaga, and Kalai use LLMs to simulate human-subject studies, including classic behavioral experiments.

Reference: "Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies." https://www.microsoft.com/en-us/research/publication/using-large-language-models-to-simulate-multiple-humans-and-replicate-human-subject-studies/

Difference: replication checks downstream aggregate outcomes. The transfer audit measures the intermediate mapping from persona evidence to observed target-study participants.

### Generative Agents And Agent-Based Social Simulation

Generative Agents and Concordia develop infrastructure for interactive LLM-based social simulation.

References:

- Park et al., "Generative Agents: Interactive Simulacra of Human Behavior." https://research.google/pubs/generative-agents-interactive-simulacra-of-human-behavior/
- Concordia. https://github.com/google-deepmind/concordia

Difference: those systems enable simulations. The transfer audit evaluates whether the personas used by such systems preserve behavioral coverage in a target task.

### Persona Generators And Persona Libraries

Paglieri et al. argue for persona generation that improves support coverage, especially for long-tail opinions and preferences.

Reference: arXiv:2602.03545. https://arxiv.org/abs/2602.03545

Difference: persona generators are candidate inputs. The transfer audit tests whether their apparent support coverage survives translation into revealed behavior in target games.

### Synthetic Persona Bias

"LLM Generated Persona is a Promise with a Catch" shows that synthetic persona generation can introduce systematic downstream bias in election forecasting and opinion simulation, and releases persona data.

References:

- Paper: https://arxiv.org/abs/2503.16527
- Dataset: https://huggingface.co/datasets/Tianyi-Lab/Personas

Difference: this strongly motivates the audit, but the proposed assay specifically estimates the match distribution over observed target-study participants.

### Socially Grounded Persona Construction

SCOPE argues that demographic personas are a bottleneck and that sociopsychological facets can improve alignment to human responses.

References:

- SCOPE paper: https://arxiv.org/abs/2601.07110
- Dataset: https://huggingface.co/datasets/Salesforce/SCOPE-Persona

Difference: SCOPE improves persona construction for response prediction. The transfer audit tests target-environment behavioral coverage, including collapse and skew.

### Persona Optimization For Behavioral Alignment

Persona evolution and optimization methods refine persona populations to match known target distributions.

Example reference: PEBA / PersonaEvolve, https://arxiv.org/abs/2509.16457

Difference: optimization methods adjust personas to fit a target. The transfer audit first asks whether existing persona libraries transfer cleanly without target-specific fitting.

## Possible Paper Framing

One possible framing:

> Researchers increasingly use LLMs to turn persona libraries into simulated populations. This practice assumes that diversity in the persona library translates into diversity in target-environment behavior. We show that this assumption can fail. We introduce a revealed-behavior coverage audit: given a persona and an observed target-study transcript, an LLM identifies which participant the persona most resembles. Across strategic environments, the resulting match distributions reveal whether persona libraries cover, amplify, or omit target behavioral types. This provides a direct diagnostic for LLM-mediated persona transfer before simulated outcomes are used as evidence.

## Design Decisions To Track

- Whether to use top-1 choice, full ranking, or both. Current default: both.
- Whether to compare players only within the same game or across the full target population. Current pilot: within-game ranking, aggregated across sampled games.
- Whether to show full transcripts or behavior summaries. Current pilot: full transcript plus compact metadata.
- Whether to include player-level numeric summaries in the prompt. Current pilot: no, to force the model to infer from transcript. Parser can add this later.
- Whether to randomize player order/labels. Current pilot: preserve observed labels. Later runs should add shuffled labels as a robustness check.
- Whether to treat the matcher model as a source of bias or a nuisance. Current stance: model-mediated bias is part of the transfer system and should be measured; robustness across matcher models remains useful.
