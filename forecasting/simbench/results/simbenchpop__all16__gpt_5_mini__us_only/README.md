# SimBenchPop All-Task US-Only Summary

Merged overlap comparison and per-task diagnosis across the two gpt-5-mini US-only SimBenchPop batches:

- initial 5-task batch
- rest11 batch

## Files

- `overall_overlap_summary.json`: merged overall metrics across all overlap rows
- `dataset_overlap_summary.csv`: task-level overlap metrics with TVD error bars inputs
- `mean_tvd_by_task_overlap.png/pdf`: all-task TVD comparison plot
- `mean_normalized_entropy_by_task_overlap.png/pdf`: all-task aggregate-spread comparison plot
- `task_summary.csv`: per-task diagnosis summary including raw heuristic columns when available
- `diagnosis_plots/`: one diagnosis plot per task
- `raw_heuristics/`: raw-data heuristic eval CSVs for applicable tasks

## Headline

- Total overlap rows: 2604
- Mean TVD: baseline 0.2468, Twin 0.2280
- Weighted mean TVD: baseline 0.1848, Twin 0.1797
- Modal match: baseline 0.6094, Twin 0.6298

## Largest Twin gains (lower TVD is better)

- ChaosNLI: baseline 0.3791, Twin 0.2532
- OSPsychRWAS: baseline 0.3198, Twin 0.2052
- MoralMachine: baseline 0.4191, Twin 0.3704
- DICES: baseline 0.1708, Twin 0.1522
- OSPsychMACH: baseline 0.2493, Twin 0.2343

## Largest baseline gains (lower TVD is better)

- OSPsychBig5: baseline 0.1400, Twin 0.2210
- Jester: baseline 0.4303, Twin 0.4864
- TISP: baseline 0.1462, Twin 0.1964
- Choices13k: baseline 0.1749, Twin 0.2199
- ConspiracyCorr: baseline 0.2735, Twin 0.3010
