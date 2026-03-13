#!/bin/bash
#SBATCH --job-name=pgg-bmk-m-noarch
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

SPLIT_ROOT="${SPLIT_ROOT:-benchmark/data}"
MANIFEST_CSV="${MANIFEST_CSV:-reports/benchmark/manifests/benchmark__data__validation_wave__strict_1__seed_0.csv}"

PROVIDER="${PROVIDER:-local}"
BASE_MODEL="${BASE_MODEL:-google/gemma-3-27b-it}"
VLLM_MODEL="${VLLM_MODEL:-${BASE_MODEL}}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_MAX_CONCURRENCY="${VLLM_MAX_CONCURRENCY:-8}"
OPENAI_MAX_CONCURRENCY="${OPENAI_MAX_CONCURRENCY:-8}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-False}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-False}"
QUANT_COMPUTE_DTYPE="${QUANT_COMPUTE_DTYPE:-auto}"
MAX_PARALLEL_GAMES="${MAX_PARALLEL_GAMES:-1}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual}_r${SLURM_RESTART_COUNT:-0}_macro_no_archetype}"

test -s "${MANIFEST_CSV}"

BACKEND_ARGS=(
  --macro-arg "--provider ${PROVIDER}"
  --macro-arg "--base_model ${BASE_MODEL}"
  --macro-arg "--use_peft False"
  --macro-arg "--load_in_8bit ${LOAD_IN_8BIT}"
  --macro-arg "--load_in_4bit ${LOAD_IN_4BIT}"
  --macro-arg "--quant_compute_dtype ${QUANT_COMPUTE_DTYPE}"
  --macro-arg "--include_reasoning True"
  --macro-arg "--max_parallel_games ${MAX_PARALLEL_GAMES}"
  --macro-arg "--run_id ${RUN_ID}"
)

if [[ "${PROVIDER}" == "vllm" ]]; then
  BACKEND_ARGS+=(
    --macro-arg "--vllm_model ${VLLM_MODEL}"
    --macro-arg "--vllm_base_url ${VLLM_BASE_URL}"
    --macro-arg "--vllm_max_concurrency ${VLLM_MAX_CONCURRENCY}"
  )
elif [[ "${PROVIDER}" == "openai" ]]; then
  BACKEND_ARGS+=(
    --macro-arg "--openai_max_concurrency ${OPENAI_MAX_CONCURRENCY}"
  )
fi

python benchmark/scripts/run_split_pipeline.py \
  --split-root "${SPLIT_ROOT}" \
  --stages macro \
  --game-ids-csv "${MANIFEST_CSV}" \
  --game-ids-column gameId \
  --archetype-mode none \
  --macro-variant no_archetype \
  "${BACKEND_ARGS[@]}"

python benchmark/scripts/run_split_pipeline.py \
  --split-root "${SPLIT_ROOT}" \
  --stages index
