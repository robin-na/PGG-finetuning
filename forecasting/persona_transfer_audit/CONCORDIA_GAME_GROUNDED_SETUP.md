# Concordia Game-Grounded Persona Setup

This setup generates target-game-grounded personas with the Concordia Persona Generators pipeline, then feeds the raw generated persona JSON into the existing persona-to-human-trajectory matching batches.

## Conditions

Two game-grounded conditions are currently configured:

- `pgg_game_grounded_alphaevolve_5`: treatment-general public goods game context. The context describes the contribution dilemma, repeated play, variable treatment features, communication, punishment, reward, visibility, all-or-nothing choices, and rule attention.
- `chip_bargain_game_grounded_alphaevolve_5`: exact three-player chip-bargaining structure. The context describes 3 rounds, one proposal turn per player per round, private chip values, offer/request proposals, private accept/decline responses, random selection when both responders accept, and payoff incentives.

The JSON configs live in:

- `forecasting/persona_transfer_audit/concordia_configs/pgg_game_grounded_alphaevolve_5.json`
- `forecasting/persona_transfer_audit/concordia_configs/chip_bargain_game_grounded_alphaevolve_5.json`

## Generate Command Artifacts

These commands do not call an API. They write `generation_config.json`, `generation_command.txt`, `initial_context.txt`, and `diversity_axes.txt`.

```bash
PYTHONPYCACHEPREFIX=/tmp/persona_pycache \
/Users/robinna/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
-m forecasting.persona_transfer_audit.generate_concordia_game_personas \
--condition pgg_game_grounded_alphaevolve_5
```

```bash
PYTHONPYCACHEPREFIX=/tmp/persona_pycache \
/Users/robinna/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
-m forecasting.persona_transfer_audit.generate_concordia_game_personas \
--condition chip_bargain_game_grounded_alphaevolve_5
```

Prepared outputs are under:

- `forecasting/persona_transfer_audit/external/concordia/concordia_pgg_game_grounded_alphaevolve_5_n32_gpt_5_mini/`
- `forecasting/persona_transfer_audit/external/concordia/concordia_chip_bargain_game_grounded_alphaevolve_5_n32_gpt_5_mini/`

## Run Persona Generation

Concordia is installed for this workspace as a temporary runtime split across:

- `/tmp/concordia_deps`: PyPI dependencies, including `gdm-concordia[openai]`, `openai`, and `scipy`.
- `/tmp/concordia_src`: a shallow checkout of `google-deepmind/concordia`.

The GitHub checkout is needed because the PyPI package `gdm-concordia==2.4.0` does not currently include `concordia.contrib.persona_generators`, while the GitHub repository does.

Two temporary runtime patches were applied to `/tmp/concordia_src` for `gpt-5-mini` generation:

- `concordia/contrib/persona_generators/persona_generator_five.py`: removed a duplicate characteristic-generation block in `alphaevolve_5` so `num_personas=32` does not over-generate and then create memories for extra personas.
- `concordia/contrib/language_models/openai/base_gpt_model.py`: forced chat-completion calls to send `temperature=1.0`, because `gpt-5-mini` rejects non-default temperatures.

If these `/tmp` directories are missing in a future session, recreate them:

```bash
/Users/robinna/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
-m pip install "gdm-concordia[openai]" -t /tmp/concordia_deps
```

```bash
/Users/robinna/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
-m pip install scipy -t /tmp/concordia_deps
```

```bash
git clone --depth 1 https://github.com/google-deepmind/concordia.git /tmp/concordia_src
```

Then run either condition. The wrapper calls Concordia's `generate_personas.py` and writes `personas.json`.

```bash
PYTHONPATH=/tmp/concordia_src:/tmp/concordia_deps \
PYTHONPYCACHEPREFIX=/tmp/persona_pycache \
/Users/robinna/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
-m forecasting.persona_transfer_audit.generate_concordia_game_personas \
--condition pgg_game_grounded_alphaevolve_5 \
--execute
```

```bash
PYTHONPATH=/tmp/concordia_src:/tmp/concordia_deps \
PYTHONPYCACHEPREFIX=/tmp/persona_pycache \
/Users/robinna/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
-m forecasting.persona_transfer_audit.generate_concordia_game_personas \
--condition chip_bargain_game_grounded_alphaevolve_5 \
--execute
```

The API provider is set in the config as `api_type=openai` and `model_name=gpt-5-mini`. Override these with `--api-type` and `--model-name` if needed.

## Build Matching Batches

After `personas.json` exists, build the matching batch with the generated JSON profile. The builder does not rewrite the persona into our own prose; it renders the generated JSON directly after `Below is information about yourself.` The only omitted field is `characteristics.initial_context`, because Concordia stores the generator context there and that string includes generation instructions rather than participant profile information.

```bash
PYTHONPYCACHEPREFIX=/tmp/persona_pycache \
/Users/robinna/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
-m forecasting.persona_transfer_audit.build_concordia_to_targets \
--personas-json forecasting/persona_transfer_audit/external/concordia/concordia_pgg_game_grounded_alphaevolve_5_n32_gpt_5_mini/personas.json \
--condition concordia_pgg_game_grounded_alphaevolve_5 \
--target pgg
```

```bash
PYTHONPYCACHEPREFIX=/tmp/persona_pycache \
/Users/robinna/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
-m forecasting.persona_transfer_audit.build_concordia_to_targets \
--personas-json forecasting/persona_transfer_audit/external/concordia/concordia_chip_bargain_game_grounded_alphaevolve_5_n32_gpt_5_mini/personas.json \
--condition concordia_chip_bargain_game_grounded_alphaevolve_5 \
--target chip
```

The batch builder writes the usual `batch_input`, `metadata`, `selected_personas.jsonl`, `sample_prompt.txt`, and token-estimate artifacts.

### Compact Profile Mode

The builder also supports a compact profile condition:

```bash
PYTHONPATH=/tmp/persona_token_deps:. \
PYTHONPYCACHEPREFIX=/tmp/persona_pycache \
/Users/robinna/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
-m forecasting.persona_transfer_audit.build_concordia_to_targets \
--personas-json forecasting/persona_transfer_audit/external/concordia/concordia_pgg_game_grounded_alphaevolve_5_n32_gpt_5_mini/personas.json \
--condition concordia_pgg_game_grounded_alphaevolve_5 \
--target pgg \
--model gpt-5-mini \
--profile-mode compact
```

```bash
PYTHONPATH=/tmp/persona_token_deps:. \
PYTHONPYCACHEPREFIX=/tmp/persona_pycache \
/Users/robinna/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
-m forecasting.persona_transfer_audit.build_concordia_to_targets \
--personas-json forecasting/persona_transfer_audit/external/concordia/concordia_chip_bargain_game_grounded_alphaevolve_5_n32_gpt_5_mini/personas.json \
--condition concordia_chip_bargain_game_grounded_alphaevolve_5 \
--target chip \
--model gpt-5-mini \
--profile-mode compact
```

Compact mode preserves the generated persona identity but removes the longer generation scaffolding from the matching prompt. It keeps only `name`, `core_motivation`, `defining_experience`, and `description`; removes `axis_position`, `specific_attitudes`, `memories`, `shared_memories`, and `initial_context`; and strips title-like suffixes from the displayed name. The source `personas.json` is not modified.

Built on 2026-05-15 with `gpt-5-mini` as the matching model:

- PGG run: `concordia_pgg_game_grounded_alphaevolve_5_to_pgg_stratified_32x40_top3_gpt_5_mini`
  - requests: 1,280
  - tiktoken input tokens: 4,257,712 total; mean 3,326.3; median 2,977.0; min 1,579; max 9,100
  - batch input: `forecasting/persona_transfer_audit/batch_input/concordia_pgg_game_grounded_alphaevolve_5_to_pgg_stratified_32x40_top3_gpt_5_mini.jsonl`
  - metadata: `forecasting/persona_transfer_audit/metadata/concordia_pgg_game_grounded_alphaevolve_5_to_pgg_stratified_32x40_top3_gpt_5_mini/`
- Chip run: `concordia_chip_bargain_game_grounded_alphaevolve_5_to_chip_bargain_stratified_32x48_top3_gpt_5_mini`
  - requests: 1,536
  - tiktoken input tokens: 3,838,112 total; mean 2,498.8; median 2,491.0; min 2,171; max 2,936
  - batch input: `forecasting/persona_transfer_audit/batch_input/concordia_chip_bargain_game_grounded_alphaevolve_5_to_chip_bargain_stratified_32x48_top3_gpt_5_mini.jsonl`
  - metadata: `forecasting/persona_transfer_audit/metadata/concordia_chip_bargain_game_grounded_alphaevolve_5_to_chip_bargain_stratified_32x48_top3_gpt_5_mini/`
- PGG compact run: `concordia_pgg_game_grounded_alphaevolve_5_compact_to_pgg_stratified_32x40_top3_gpt_5_mini`
  - requests: 1,280
  - compact profile length: median 744.5 characters; mean 797.9; min 648; max 1,509
  - tiktoken input tokens: 3,283,712 total; mean 2,565.4; median 2,236.5; min 883; max 8,162
  - batch input: `forecasting/persona_transfer_audit/batch_input/concordia_pgg_game_grounded_alphaevolve_5_compact_to_pgg_stratified_32x40_top3_gpt_5_mini.jsonl`
  - metadata: `forecasting/persona_transfer_audit/metadata/concordia_pgg_game_grounded_alphaevolve_5_compact_to_pgg_stratified_32x40_top3_gpt_5_mini/`
- Chip compact run: `concordia_chip_bargain_game_grounded_alphaevolve_5_compact_to_chip_bargain_stratified_32x48_top3_gpt_5_mini`
  - requests: 1,536
  - compact profile length: median 690.0 characters; mean 732.0; min 583; max 1,386
  - tiktoken input tokens: 2,681,552 total; mean 1,745.8; median 1,739.0; min 1,520; max 2,047
  - batch input: `forecasting/persona_transfer_audit/batch_input/concordia_chip_bargain_game_grounded_alphaevolve_5_compact_to_chip_bargain_stratified_32x48_top3_gpt_5_mini.jsonl`
  - metadata: `forecasting/persona_transfer_audit/metadata/concordia_chip_bargain_game_grounded_alphaevolve_5_compact_to_chip_bargain_stratified_32x48_top3_gpt_5_mini/`

## Generated Libraries

Generated on 2026-05-15 with `gpt-5-mini`, `alphaevolve_5`, and the game-grounded configs above:

- PGG: `forecasting/persona_transfer_audit/external/concordia/concordia_pgg_game_grounded_alphaevolve_5_n32_gpt_5_mini/personas.json`
  - 32 personas
  - file size: 198 KB
  - median raw JSON profile length: 5,967 characters
  - each persona has 3 generated memory/context snippets
- Chip bargaining: `forecasting/persona_transfer_audit/external/concordia/concordia_chip_bargain_game_grounded_alphaevolve_5_n32_gpt_5_mini/personas.json`
  - 32 personas
  - file size: 188 KB
  - median raw JSON profile length: 5,610 characters
  - each persona has 3 generated memory/context snippets

## Design Rationale

This is the target-grounded condition, not the generic transfer condition. The goal is to give Persona Generators the strongest reasonable version of the task without leaking observed target-game participant behavior. PGG is treatment-general rather than treatment-specific because we want one persona library for the PGG family before testing the more permissive treatment-specific upper bound.
