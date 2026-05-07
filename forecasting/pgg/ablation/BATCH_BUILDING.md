# Building PGG Ablation Batches

The ablation card runs should reuse the same PGG validation assignment file across variants.

## Generate Ablation Cards

```bash
python3 forecasting/pgg/ablation/build_twin_pgg_ablation_cards.py
```

This writes one folder per active six-category variant under:

```text
forecasting/pgg/ablation/output/twin_pgg_six_category_ablations/
```

Each variant folder contains:

- `twin_pgg_ablation_cards.jsonl`
- `shared_prompt_notes.md`
- `preview_twin_pgg_ablation_cards.json`

Each generated card type exposes only its named evidence category. Demographic background appears only in `twin_pgg_background_only`.

## Build One Batch Input

Example for the corrected direct-social-only ablation:

```bash
python3 forecasting/pgg/build_batch_inputs.py \
  --variant-name twin_pgg_direct_social_only \
  --model gpt-5-mini \
  --selection-mode full \
  --require-valid-starting-players \
  --repeat-count-mode fixed \
  --repeats-per-game 1 \
  --run-name twin_pgg_direct_social_only_seed_0_gpt_5_mini \
  --persona-assignment-file forecasting/pgg/profile_sampling/output/twin_to_pgg_validation_persona_sampling/seed_0/game_assignments.jsonl \
  --persona-cards-file forecasting/pgg/ablation/output/twin_pgg_six_category_ablations/twin_pgg_direct_social_only/twin_pgg_ablation_cards.jsonl \
  --persona-shared-notes-file forecasting/pgg/ablation/output/twin_pgg_six_category_ablations/twin_pgg_direct_social_only/shared_prompt_notes.md
```

The important part is that `--persona-assignment-file` stays fixed across all ablation variants.

## Build All First-Pass `gpt-5-mini` Inputs

Use the same command template for each of the six category variants in `ablation_variants.json`, changing only:

- `--variant-name`
- `--run-name`
- `--persona-cards-file`
- `--persona-shared-notes-file`

Do not submit these to OpenAI until the generated sample prompts have been checked.

The first-pass category grid intentionally excludes leave-one-out, scores-only, and anchors-only variants.

## Generated First-Pass `gpt-5-mini` Inputs

These six batch inputs have been generated with the same locked corrected assignment file:

- `forecasting/pgg/batch_input/twin_pgg_background_only_seed_0_gpt_5_mini.jsonl`
- `forecasting/pgg/batch_input/twin_pgg_direct_social_only_seed_0_gpt_5_mini.jsonl`
- `forecasting/pgg/batch_input/twin_pgg_self_report_social_only_seed_0_gpt_5_mini.jsonl`
- `forecasting/pgg/batch_input/twin_pgg_non_social_econ_only_seed_0_gpt_5_mini.jsonl`
- `forecasting/pgg/batch_input/twin_pgg_cognitive_only_seed_0_gpt_5_mini.jsonl`
- `forecasting/pgg/batch_input/twin_pgg_misc_heuristics_pricing_text_only_seed_0_gpt_5_mini.jsonl`

Each currently contains 417 requests. They are prepared inputs only; this folder does not imply that the requests have been submitted or evaluated.

## Evaluation

After batch outputs are downloaded and parsed, evaluate using the same PGG scripts as the core runs:

- `forecasting/pgg/parse_outputs.py`
- `forecasting/pgg/analyze_vs_human_treatments.py`
- `forecasting/pgg/exploratory/plot_macro_pointwise_alignment.py`
- `forecasting/pgg/exploratory/analyze_micro_distribution_alignment.py`
