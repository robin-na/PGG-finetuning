# Transferability Forecast Assets

This directory contains prompt assets for asking an LLM to forecast, before running a transfer experiment, which SimBench tasks will benefit from the Twin persona-summary prior.

The forecast target is the current US-only `SimBenchPop` comparison:
- Baseline: direct group-level prediction
- Transfer arm: Twin `persona_summary` micro-simulation with `n=64` sampled personas

The intended prediction target is not raw score regression. It is:
- a ranking from most helped to most harmed, and
- a 3-way class label per task: `positive`, `negative`, or `insignificant`.

Ground truth is derived from the corrected paper-style SimBench score using the current overlap comparison in:
- `forecasting/simbench/results/simbenchpop__baseline_vs_persona_summary_overlap__gpt_5_mini/`

Files:
- `twin_source_card.md`: detailed source-study card for Twin-2K-500 and the exact representation used here
- `us16_task_cards.json`: factual study cards for the 16 US-only SimBenchPop tasks in the current comparison
- `us16_ground_truth.json`: actual task deltas, ranks, and 3-way labels for evaluation
- `us16_global_ranking_prompt.md`: a fully materialized prompt for ranking + 3-way labeling
- `pairwise_comparison_prompt.md`: a pairwise prompt template for more stable relative judgments
- `evaluate_transferability_forecast.py`: evaluates an LLM forecast against the ground truth

Current task counts:
- `ChaosNLI`: 500 overlap rows
- `Choices13k`: 500 overlap rows
- `ConspiracyCorr`: 1 overlap rows
- `DICES`: 247 overlap rows
- `GlobalOpinionQA`: 23 overlap rows
- `ISSP`: 17 overlap rows
- `Jester`: 136 overlap rows
- `MoralMachine`: 72 overlap rows
- `NumberGame`: 500 overlap rows
- `OSPsychBig5`: 40 overlap rows
- `OSPsychMACH`: 1 overlap rows
- `OSPsychMGKT`: 111 overlap rows
- `OSPsychRWAS`: 22 overlap rows
- `OpinionQA`: 310 overlap rows
- `TISP`: 10 overlap rows
- `WisdomOfCrowds`: 114 overlap rows

Current realized labels:
- `OSPsychRWAS`: rank 1, label `positive`, delta 44.74
- `ChaosNLI`: rank 2, label `positive`, delta 28.92
- `MoralMachine`: rank 3, label `positive`, delta 25.12
- `ConspiracyCorr`: rank 4, label `insignificant`, delta 10.75
- `DICES`: rank 5, label `positive`, delta 3.52
- `WisdomOfCrowds`: rank 6, label `insignificant`, delta 3.18
- `OSPsychMGKT`: rank 7, label `insignificant`, delta 2.86
- `OpinionQA`: rank 8, label `insignificant`, delta 2.41
- `NumberGame`: rank 9, label `insignificant`, delta 2.25
- `GlobalOpinionQA`: rank 10, label `insignificant`, delta -3.46
- `TISP`: rank 11, label `insignificant`, delta -11.16
- `Jester`: rank 12, label `negative`, delta -11.18
- `OSPsychMACH`: rank 13, label `insignificant`, delta -15.53
- `ISSP`: rank 14, label `insignificant`, delta -15.80
- `OSPsychBig5`: rank 15, label `negative`, delta -23.23
- `Choices13k`: rank 16, label `negative`, delta -27.64
