# Archetype Retrieval Pipeline

This folder contains a D+E-to-archetype retrieval pipeline:

- `train_archetype_retrieval.py`:
  - Loads per-tag archetype data from `learning_wave/<TAG>/`.
  - Merges with demographics (D) and game config (E).
  - Applies tag activation rules derived from `Persona/misc/make_batch_persona_type_input.py`.
  - Runs grouped cross-validation by `gameId`.
  - Trains multiple regressors from `D+E -> embedding`.
  - Retrieves nearest archetype in embedding space and reports retrieval metrics.
  - Exports final artifacts per tag/model.

- `retrieve_archetype.py`:
  - Loads a trained model artifact.
  - Predicts an embedding for a provided `D+E` feature row.
  - Returns top-k nearest archetype texts from the tag bank.

- `retrieval_common.py`: shared schema/rule utilities.
- `validation_wave/build_validation_wave.py`:
  - Extracts per-tag sections from `Persona/summary_gpt51_val.jsonl`.
  - Writes per-tag `*_sections_input.jsonl`.
  - Builds and saves per-tag embeddings (`openai` by default).

## Train

```bash
python3 Persona/archetype_retrieval/train_archetype_retrieval.py
```

Output goes to:

- `Persona/archetype_retrieval/model_runs/run_<UTC_TIMESTAMP>/`
- `Persona/archetype_retrieval/model_runs/latest_run.txt` points to the last run.

## Retrieve

Lookup by `gameId` + `playerId`:

```bash
python3 Persona/archetype_retrieval/retrieve_archetype.py \
  --tag CONTRIBUTION \
  --model ridge \
  --game-id <GAME_ID> \
  --player-id <PLAYER_ID> \
  --top-k 5
```

Or pass explicit features:

```bash
python3 Persona/archetype_retrieval/retrieve_archetype.py \
  --tag CONTRIBUTION \
  --model ridge \
  --feature-json /path/to/features.json \
  --top-k 5
```

`features.json` should be a JSON object containing any subset of model features. Missing values are imputed automatically.

## Build Validation Wave

```bash
python3 Persona/archetype_retrieval/validation_wave/build_validation_wave.py
```
