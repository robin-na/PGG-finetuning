# Finetuning persona type generation

This folder contains scripts to build a game-level dataset and fine-tune a HuggingFace model to generate N distinct persona types from a PGG environment.

## What the dataset looks like
Each training example is one game. The input is an instruction with the game config (formatted the same way as `Persona/make_batch_persona_type_input.py`) and the required headers. The output is JSON with `N` type objects (one per player), each shaped like `{"id":"type_k","text":"..."}`.

## Build the dataset
Default behavior keeps only finished players, reduces `N` accordingly, and uses that `N` as the player count in the prompt.

```bash
python Finetuning/build_persona_type_dataset.py \
  --summary Persona/summary_gpt51_learn.jsonl \
  --config data/processed_data/df_analysis_learn.csv \
  --output-dir Finetuning/data
```

Key options:
- `--incomplete-strategy drop-games` drops any game where at least one player did not finish.
- `--incomplete-strategy drop-players` keeps the game, drops unfinished players, and reduces `N`.
- `--incomplete-strategy keep-all` keeps all players regardless of `game_finished`.
- `--player-count-source config` keeps `CONFIG_playerCount` in the prompt even if players are dropped.
- `--player-count-source actual` sets the prompt player count to the number of kept players.
- `--max-players K` drops games with more than `K` players (useful for context limits).

Outputs:
- `Finetuning/data/persona_type_train.jsonl`
- `Finetuning/data/persona_type_val.jsonl`

## Train (HuggingFace)
Example SFT run:

```bash
python Finetuning/train_persona_type_sft.py \
  --model <base-model> \
  --train Finetuning/data/persona_type_train.jsonl \
  --eval Finetuning/data/persona_type_val.jsonl \
  --output-dir Finetuning/out_persona_types \
  --max-seq-len 8192 \
  --batch-size 1 \
  --grad-accum 8
```

No-eval run:

```bash
python Finetuning/train_persona_type_sft.py \
  --model <base-model> \
  --train Finetuning/data/persona_type_train.jsonl \
  --output-dir Finetuning/out_persona_types \
  --max-seq-len 8192 \
  --batch-size 1 \
  --grad-accum 8 \
  --no-eval
```

Optional LoRA:

```bash
python Finetuning/train_persona_type_sft.py \
  --model <base-model> \
  --train Finetuning/data/persona_type_train.jsonl \
  --eval Finetuning/data/persona_type_val.jsonl \
  --output-dir Finetuning/out_persona_types_lora \
  --use-lora
```

## Notes
- Outputs are long. If you see frequent truncation, increase `--max-seq-len` or consider filtering out very large-N games.
- If prompt length consumes the full context, the script drops those examples (see `--min-response-tokens`).
