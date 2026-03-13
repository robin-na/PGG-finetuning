#!/bin/bash
# Local HF path on 2xA100 via data parallel sharding.
# One quantized model replica per GPU, no vLLM server.
#SBATCH --job-name=pgg-bmk-m-noarch-l2a8
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
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
RUNS_ROOT="${RUNS_ROOT:-outputs/benchmark/runs/benchmark_filtered}"
TMP_ROOT="${TMP_ROOT:-outputs/benchmark/tmp/${SLURM_JOB_ID:-manual}_macro_no_archetype_local_2xa100_8bit}"
SHARD_DIR="${SHARD_DIR:-${TMP_ROOT}/manifest_shards}"
TMP_OUTPUT_ROOT="${TMP_OUTPUT_ROOT:-${TMP_ROOT}/shard_runs}"
VARIANT_ROOT="${RUNS_ROOT}/macro_simulation_eval/no_archetype"

BASE_MODEL="${BASE_MODEL:-google/gemma-3-27b-it}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-True}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-False}"
QUANT_COMPUTE_DTYPE="${QUANT_COMPUTE_DTYPE:-bf16}"
INCLUDE_REASONING="${INCLUDE_REASONING:-True}"
DEBUG_LEVEL="${DEBUG_LEVEL:-off}"
MAX_PARALLEL_GAMES="${MAX_PARALLEL_GAMES:-1}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual}_r${SLURM_RESTART_COUNT:-0}_macro_no_archetype_local_2xa100_8bit}"
SHARD_RUN_ID_0="${RUN_ID}_shard0"
SHARD_RUN_ID_1="${RUN_ID}_shard1"
FINAL_RUN_DIR="${VARIANT_ROOT}/${RUN_ID}"

mkdir -p "${SHARD_DIR}" "${TMP_OUTPUT_ROOT}" "${VARIANT_ROOT}"
test -s "${MANIFEST_CSV}"

PIPE0_PID=""
PIPE1_PID=""

cleanup() {
  set +e
  if [[ -n "${PIPE0_PID}" ]]; then kill "${PIPE0_PID}" >/dev/null 2>&1 || true; fi
  if [[ -n "${PIPE1_PID}" ]]; then kill "${PIPE1_PID}" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT

run_shard() {
  local gpu_id="$1"
  local shard_csv="$2"
  local shard_output_root="$3"
  local shard_run_id="$4"

  echo "[info] starting shard ${shard_run_id} on GPU ${gpu_id} using ${shard_csv}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  python benchmark/scripts/run_split_pipeline.py \
    --split-root "${SPLIT_ROOT}" \
    --stages macro \
    --game-ids-csv "${shard_csv}" \
    --game-ids-column gameId \
    --archetype-mode none \
    --macro-variant no_archetype \
    --macro-output-root "${shard_output_root}" \
    --macro-arg "--provider local" \
    --macro-arg "--base_model ${BASE_MODEL}" \
    --macro-arg "--use_peft False" \
    --macro-arg "--load_in_8bit ${LOAD_IN_8BIT}" \
    --macro-arg "--load_in_4bit ${LOAD_IN_4BIT}" \
    --macro-arg "--quant_compute_dtype ${QUANT_COMPUTE_DTYPE}" \
    --macro-arg "--include_reasoning ${INCLUDE_REASONING}" \
    --macro-arg "--debug_level ${DEBUG_LEVEL}" \
    --macro-arg "--max_parallel_games ${MAX_PARALLEL_GAMES}" \
    --macro-arg "--run_id ${shard_run_id}"
}

python benchmark/scripts/shard_game_manifest.py \
  --input-csv "${MANIFEST_CSV}" \
  --output-dir "${SHARD_DIR}" \
  --num-shards 2 \
  --prefix "${RUN_ID}"

SHARD0_CSV="${SHARD_DIR}/${RUN_ID}_shard_00_of_02.csv"
SHARD1_CSV="${SHARD_DIR}/${RUN_ID}_shard_01_of_02.csv"

test -s "${SHARD0_CSV}"
test -s "${SHARD1_CSV}"

run_shard 0 "${SHARD0_CSV}" "${TMP_OUTPUT_ROOT}/shard0" "${SHARD_RUN_ID_0}" &
PIPE0_PID="$!"
run_shard 1 "${SHARD1_CSV}" "${TMP_OUTPUT_ROOT}/shard1" "${SHARD_RUN_ID_1}" &
PIPE1_PID="$!"

wait "${PIPE0_PID}" || {
  status="$?"
  echo "[error] local shard0 failed with exit=${status}" >&2
  exit "${status}"
}
PIPE0_PID=""

wait "${PIPE1_PID}" || {
  status="$?"
  echo "[error] local shard1 failed with exit=${status}" >&2
  exit "${status}"
}
PIPE1_PID=""

python benchmark/scripts/merge_macro_shard_runs.py \
  --output-run-dir "${FINAL_RUN_DIR}" \
  --source-manifest "${MANIFEST_CSV}" \
  --shard-run-dir "${TMP_OUTPUT_ROOT}/shard0/${SHARD_RUN_ID_0}" \
  --shard-run-dir "${TMP_OUTPUT_ROOT}/shard1/${SHARD_RUN_ID_1}"

python benchmark/scripts/run_split_pipeline.py \
  --split-root "${SPLIT_ROOT}" \
  --stages index

echo "Final merged run: ${FINAL_RUN_DIR}"
