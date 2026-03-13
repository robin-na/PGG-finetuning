#!/bin/bash
# Local HF path on 1xA100 for oracle archetype runs.
# No quantization. Defaults to Gemma 3 12B IT.
#SBATCH --job-name=pgg-bmk-m-oracle-l4
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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
VAL_ORACLE_POOL="${VAL_ORACLE_POOL:-Persona/archetype_oracle_gpt51_val.jsonl}"
BASE_MODEL="${BASE_MODEL:-google/gemma-3-12b-it}"
INCLUDE_REASONING="${INCLUDE_REASONING:-True}"
MAX_PARALLEL_GAMES="${MAX_PARALLEL_GAMES:-1}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual}_r${SLURM_RESTART_COUNT:-0}_macro_oracle_archetype_local_12b}"

if [[ ! -s "${MANIFEST_CSV}" ]]; then
  echo "[error] manifest csv missing or empty: ${MANIFEST_CSV}" >&2
  exit 1
fi
if [[ ! -s "${VAL_ORACLE_POOL}" ]]; then
  echo "[error] oracle pool missing or empty: ${VAL_ORACLE_POOL}" >&2
  exit 1
fi

python benchmark/scripts/run_split_pipeline.py \
  --split-root "${SPLIT_ROOT}" \
  --stages macro \
  --game-ids-csv "${MANIFEST_CSV}" \
  --game-ids-column gameId \
  --archetype-mode matched_summary \
  --archetype-summary-pool "${VAL_ORACLE_POOL}" \
  --macro-variant oracle_archetype \
  --macro-arg "--provider local" \
  --macro-arg "--base_model ${BASE_MODEL}" \
  --macro-arg "--use_peft False" \
  --macro-arg "--include_reasoning ${INCLUDE_REASONING}" \
  --macro-arg "--max_parallel_games ${MAX_PARALLEL_GAMES}" \
  --macro-arg "--run_id ${RUN_ID}"

python benchmark/scripts/run_split_pipeline.py \
  --split-root "${SPLIT_ROOT}" \
  --stages index
