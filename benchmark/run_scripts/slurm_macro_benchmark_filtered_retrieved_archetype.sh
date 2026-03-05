#!/bin/bash
#SBATCH --job-name=pgg-bmk-m-retr
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$HOME/PGG-finetuning}"

eval "$(/home/software/anaconda3/2023.07/bin/conda shell.bash hook)"
conda activate /home/robinna/.conda/envs/llm_conda

SPLIT_ROOT="benchmark/data"
MANIFEST_CSV="reports/benchmark/manifests/benchmark__data__validation_wave__strict_1__seed_0.csv"
RUN_ROOT="outputs/benchmark/runs/benchmark_filtered"
ARCH_ROOT="${RUN_ROOT}/archetype_retrieval"
RETRIEVED_POOL="${ARCH_ROOT}/validation_wave/synthetic_archetype_ridge_val.jsonl"
VAL_ORACLE_POOL="Persona/archetype_oracle_gpt51_val.jsonl"

PROVIDER="${PROVIDER:-local}"
BASE_MODEL="${BASE_MODEL:-google/gemma-3-27b-it}"
RUN_ID="${SLURM_JOB_ID:-manual}_r${SLURM_RESTART_COUNT:-0}_macro_retrieved_archetype"

test -s "${MANIFEST_CSV}"
test -s "${VAL_ORACLE_POOL}"

if [[ ! -s "${RETRIEVED_POOL}" ]]; then
  python benchmark/scripts/run_split_pipeline.py \
    --split-root "${SPLIT_ROOT}" \
    --stages archetype-train,archetype-validate,archetype-synthetic,index \
    --synthetic-model ridge \
    --train-arg "--models mean linear ridge --allow-tag-errors"
fi

test -s "${RETRIEVED_POOL}"

python benchmark/scripts/run_split_pipeline.py \
  --split-root "${SPLIT_ROOT}" \
  --stages macro \
  --game-ids-csv "${MANIFEST_CSV}" \
  --game-ids-column gameId \
  --archetype-mode matched_summary \
  --archetype-summary-pool "${RETRIEVED_POOL}" \
  --macro-variant retrieved_archetype \
  --macro-arg "--provider ${PROVIDER}" \
  --macro-arg "--base_model ${BASE_MODEL}" \
  --macro-arg "--use_peft False" \
  --macro-arg "--include_reasoning True" \
  --macro-arg "--run_id ${RUN_ID}"

python benchmark/scripts/run_split_pipeline.py \
  --split-root "${SPLIT_ROOT}" \
  --stages index
