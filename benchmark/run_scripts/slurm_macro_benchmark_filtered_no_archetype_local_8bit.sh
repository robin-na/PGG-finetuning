#!/bin/bash
# Local HF path on 1xA100 with bitsandbytes 8-bit quantization.
#SBATCH --job-name=pgg-bmk-m-noarch-l8
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$HOME/PGG-finetuning}"

CONDA_BIN="${CONDA_BIN:-/home/software/anaconda3/2023.07/bin/conda}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/home/robinna/.conda/envs/llm_conda}"

eval "$(${CONDA_BIN} shell.bash hook)"
conda activate "${CONDA_ENV_PATH}"

echo "[info] nvidia-smi"
nvidia-smi

export PYTHONUNBUFFERED=1

SPLIT_ROOT="${SPLIT_ROOT:-benchmark/data}"
MANIFEST_CSV="${MANIFEST_CSV:-reports/benchmark/manifests/benchmark__data__validation_wave__strict_1__seed_0.csv}"
BASE_MODEL="${BASE_MODEL:-google/gemma-3-4b-it}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-True}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-False}"
QUANT_COMPUTE_DTYPE="${QUANT_COMPUTE_DTYPE:-bf16}"
INCLUDE_REASONING="${INCLUDE_REASONING:-True}"
DEBUG_LEVEL="${DEBUG_LEVEL:-compact}"
MAX_PARALLEL_GAMES="${MAX_PARALLEL_GAMES:-1}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual}_r${SLURM_RESTART_COUNT:-0}_macro_no_archetype_local_8bit}"

if [[ ! -s "${MANIFEST_CSV}" ]]; then
  echo "[error] manifest csv missing or empty: ${MANIFEST_CSV}" >&2
  exit 1
fi

python benchmark/scripts/run_split_pipeline.py \
  --split-root "${SPLIT_ROOT}" \
  --stages macro \
  --game-ids-csv "${MANIFEST_CSV}" \
  --game-ids-column gameId \
  --archetype-mode none \
  --macro-variant no_archetype \
  --macro-arg "--provider local" \
  --macro-arg "--base_model ${BASE_MODEL}" \
  --macro-arg "--use_peft False" \
  --macro-arg "--load_in_8bit ${LOAD_IN_8BIT}" \
  --macro-arg "--load_in_4bit ${LOAD_IN_4BIT}" \
  --macro-arg "--quant_compute_dtype ${QUANT_COMPUTE_DTYPE}" \
  --macro-arg "--include_reasoning ${INCLUDE_REASONING}" \
  --macro-arg "--debug_level ${DEBUG_LEVEL}" \
  --macro-arg "--max_parallel_games ${MAX_PARALLEL_GAMES}" \
  --macro-arg "--run_id ${RUN_ID}"

python benchmark/scripts/run_split_pipeline.py \
  --split-root "${SPLIT_ROOT}" \
  --stages index
