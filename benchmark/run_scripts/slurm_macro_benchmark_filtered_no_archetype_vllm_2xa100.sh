#!/bin/bash
# Best when the model fits on one A100: one vLLM server per GPU, manifest sharding,
# and moderate game-level concurrency on each server.
#SBATCH --job-name=pgg-bmk-m-noarch-v2a
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
TMP_ROOT="${TMP_ROOT:-outputs/benchmark/tmp/${SLURM_JOB_ID:-manual}_macro_no_archetype_vllm_2xa100}"
SHARD_DIR="${SHARD_DIR:-${TMP_ROOT}/manifest_shards}"
TMP_OUTPUT_ROOT="${TMP_OUTPUT_ROOT:-${TMP_ROOT}/shard_runs}"
VARIANT_ROOT="${RUNS_ROOT}/macro_simulation_eval/no_archetype"

BASE_MODEL="${BASE_MODEL:-google/gemma-3-27b-it}"
VLLM_MODEL="${VLLM_MODEL:-${BASE_MODEL}}"
VLLM_LAUNCHER="${VLLM_LAUNCHER:-auto}"
VLLM_SINGULARITY_MODULE="${VLLM_SINGULARITY_MODULE:-singularity/3.7.0}"
VLLM_SIF_PATH="${VLLM_SIF_PATH:-$HOME/containers/vllm-openai_latest.sif}"
VLLM_HOST_HF_HOME="${VLLM_HOST_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}"
VLLM_CONTAINER_HF_HOME="${VLLM_CONTAINER_HF_HOME:-/root/.cache/huggingface}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_CONCURRENCY="${VLLM_MAX_CONCURRENCY:-4}"
MAX_PARALLEL_GAMES_PER_SERVER="${MAX_PARALLEL_GAMES_PER_SERVER:-3}"
INCLUDE_REASONING="${INCLUDE_REASONING:-False}"
DEBUG_LEVEL="${DEBUG_LEVEL:-off}"
PORT0="${PORT0:-8000}"
PORT1="${PORT1:-8001}"

RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual}_r${SLURM_RESTART_COUNT:-0}_macro_no_archetype_vllm_2xa100}"
SHARD_RUN_ID_0="${RUN_ID}_shard0"
SHARD_RUN_ID_1="${RUN_ID}_shard1"
FINAL_RUN_DIR="${VARIANT_ROOT}/${RUN_ID}"

mkdir -p "${SHARD_DIR}" "${TMP_OUTPUT_ROOT}" "${VARIANT_ROOT}"
test -s "${MANIFEST_CSV}"

source benchmark/run_scripts/vllm_server_launcher.sh

SERVER0_PID=""
SERVER1_PID=""
PIPE0_PID=""
PIPE1_PID=""

cleanup() {
  set +e
  if [[ -n "${PIPE0_PID}" ]]; then kill "${PIPE0_PID}" >/dev/null 2>&1 || true; fi
  if [[ -n "${PIPE1_PID}" ]]; then kill "${PIPE1_PID}" >/dev/null 2>&1 || true; fi
  if [[ -n "${SERVER0_PID}" ]]; then kill "${SERVER0_PID}" >/dev/null 2>&1 || true; fi
  if [[ -n "${SERVER1_PID}" ]]; then kill "${SERVER1_PID}" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT

run_shard() {
  local shard_csv="$1"
  local port="$2"
  local shard_output_root="$3"
  local shard_run_id="$4"

  python benchmark/scripts/run_split_pipeline.py \
    --split-root "${SPLIT_ROOT}" \
    --stages macro \
    --game-ids-csv "${shard_csv}" \
    --game-ids-column gameId \
    --archetype-mode none \
    --macro-variant no_archetype \
    --macro-output-root "${shard_output_root}" \
    --macro-arg "--provider vllm" \
    --macro-arg "--base_model ${BASE_MODEL}" \
    --macro-arg "--vllm_model ${VLLM_MODEL}" \
    --macro-arg "--vllm_base_url http://127.0.0.1:${port}/v1" \
    --macro-arg "--vllm_max_concurrency ${VLLM_MAX_CONCURRENCY}" \
    --macro-arg "--max_parallel_games ${MAX_PARALLEL_GAMES_PER_SERVER}" \
    --macro-arg "--include_reasoning ${INCLUDE_REASONING}" \
    --macro-arg "--debug_level ${DEBUG_LEVEL}" \
    --macro-arg "--use_peft False" \
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

SERVER0_PID="$(start_vllm_server 0 "${PORT0}" "${TMP_ROOT}/vllm_gpu0.log")"
SERVER1_PID="$(start_vllm_server 1 "${PORT1}" "${TMP_ROOT}/vllm_gpu1.log")"

if ! wait_for_server "${PORT0}"; then
  echo "[error] vLLM server on GPU0 failed to become ready on port ${PORT0}" >&2
  print_vllm_logs
  exit 1
fi
if ! wait_for_server "${PORT1}"; then
  echo "[error] vLLM server on GPU1 failed to become ready on port ${PORT1}" >&2
  print_vllm_logs
  exit 1
fi

run_shard "${SHARD0_CSV}" "${PORT0}" "${TMP_OUTPUT_ROOT}/shard0" "${SHARD_RUN_ID_0}" &
PIPE0_PID="$!"
run_shard "${SHARD1_CSV}" "${PORT1}" "${TMP_OUTPUT_ROOT}/shard1" "${SHARD_RUN_ID_1}" &
PIPE1_PID="$!"

wait "${PIPE0_PID}" || {
  status="$?"
  echo "[error] full-run shard0 failed with exit=${status}" >&2
  print_vllm_logs
  exit "${status}"
}
PIPE0_PID=""
wait "${PIPE1_PID}" || {
  status="$?"
  echo "[error] full-run shard1 failed with exit=${status}" >&2
  print_vllm_logs
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
