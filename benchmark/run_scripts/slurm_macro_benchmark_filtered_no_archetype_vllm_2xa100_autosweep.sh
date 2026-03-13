#!/bin/bash
# Auto-tune client-side vLLM concurrency on a small stable manifest sample,
# then run the full 2xA100 no-archetype benchmark with the winning setting.
#SBATCH --job-name=pgg-bmk-m-noarch-v2a-auto
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
TMP_ROOT="${TMP_ROOT:-outputs/benchmark/tmp/${SLURM_JOB_ID:-manual}_macro_no_archetype_vllm_2xa100_autosweep}"
SWEEP_ROOT="${SWEEP_ROOT:-${TMP_ROOT}/autosweep}"
FULL_SHARD_DIR="${FULL_SHARD_DIR:-${TMP_ROOT}/full_manifest_shards}"
SAMPLE_SHARD_DIR="${SAMPLE_SHARD_DIR:-${SWEEP_ROOT}/calibration_shards}"
SWEEP_RUNS_ROOT="${SWEEP_RUNS_ROOT:-${SWEEP_ROOT}/runs_index}"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-${SWEEP_ROOT}/trial_runs}"
SWEEP_RESULTS_TSV="${SWEEP_RESULTS_TSV:-${SWEEP_ROOT}/tuning_results.tsv}"
SWEEP_CHOICE_JSON="${SWEEP_CHOICE_JSON:-${SWEEP_ROOT}/chosen_setting.json}"
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
INCLUDE_REASONING="${INCLUDE_REASONING:-False}"
DEBUG_LEVEL="${DEBUG_LEVEL:-off}"
PORT0="${PORT0:-8000}"
PORT1="${PORT1:-8001}"

SWEEP_SAMPLE_GAMES="${SWEEP_SAMPLE_GAMES:-8}"
SWEEP_SAMPLE_COLUMN="${SWEEP_SAMPLE_COLUMN:-gameId}"
SWEEP_SAMPLE_SALT="${SWEEP_SAMPLE_SALT:-macro_vllm_autosweep}"
SWEEP_MAX_PARALLEL_GAMES_CANDIDATES="${SWEEP_MAX_PARALLEL_GAMES_CANDIDATES:-2,3,4}"
SWEEP_VLLM_MAX_CONCURRENCY_CANDIDATES="${SWEEP_VLLM_MAX_CONCURRENCY_CANDIDATES:-4,6}"

RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual}_r${SLURM_RESTART_COUNT:-0}_macro_no_archetype_vllm_2xa100_autosweep}"
FINAL_RUN_DIR="${VARIANT_ROOT}/${RUN_ID}"

mkdir -p "${SWEEP_ROOT}" "${FULL_SHARD_DIR}" "${TMP_OUTPUT_ROOT}" "${VARIANT_ROOT}" "${SWEEP_OUTPUT_ROOT}"
test -s "${MANIFEST_CSV}"
if (( SWEEP_SAMPLE_GAMES < 2 )); then
  echo "SWEEP_SAMPLE_GAMES must be at least 2." >&2
  exit 1
fi

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
  local max_parallel_games="$5"
  local vllm_max_concurrency="$6"
  local runs_root_override="$7"

  python benchmark/scripts/run_split_pipeline.py \
    --split-root "${SPLIT_ROOT}" \
    --runs-root "${runs_root_override}" \
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
    --macro-arg "--vllm_max_concurrency ${vllm_max_concurrency}" \
    --macro-arg "--max_parallel_games ${max_parallel_games}" \
    --macro-arg "--include_reasoning ${INCLUDE_REASONING}" \
    --macro-arg "--debug_level ${DEBUG_LEVEL}" \
    --macro-arg "--use_peft False" \
    --macro-arg "--run_id ${shard_run_id}"
}

run_trial() {
  local max_parallel_games="$1"
  local vllm_max_concurrency="$2"
  local trial_tag="g${max_parallel_games}_c${vllm_max_concurrency}"
  local trial_root="${SWEEP_ROOT}/trial_${trial_tag}"
  local trial_output_root="${SWEEP_OUTPUT_ROOT}/${trial_tag}"
  local trial_runs_root="${SWEEP_RUNS_ROOT}/${trial_tag}"
  local shard0_csv="${SAMPLE_SHARD_DIR}/calibration_shard_00_of_02.csv"
  local shard1_csv="${SAMPLE_SHARD_DIR}/calibration_shard_01_of_02.csv"
  local trial_run_id_0="${RUN_ID}_${trial_tag}_shard0"
  local trial_run_id_1="${RUN_ID}_${trial_tag}_shard1"
  local start_ts
  local end_ts
  local elapsed_sec
  local status0
  local status1
  local overall_status

  mkdir -p "${trial_root}" "${trial_output_root}" "${trial_runs_root}"

  start_ts="$(date +%s)"
  set +e
  run_shard "${shard0_csv}" "${PORT0}" "${trial_output_root}/shard0" "${trial_run_id_0}" "${max_parallel_games}" "${vllm_max_concurrency}" "${trial_runs_root}" &
  local pid0="$!"
  run_shard "${shard1_csv}" "${PORT1}" "${trial_output_root}/shard1" "${trial_run_id_1}" "${max_parallel_games}" "${vllm_max_concurrency}" "${trial_runs_root}" &
  local pid1="$!"
  wait "${pid0}"
  status0="$?"
  wait "${pid1}"
  status1="$?"
  set -e
  end_ts="$(date +%s)"
  elapsed_sec="$((end_ts - start_ts))"

  if [[ "${status0}" -eq 0 && "${status1}" -eq 0 ]]; then
    overall_status="ok"
  else
    overall_status="fail"
    if [[ "${status0}" -ne 0 ]]; then
      echo "[autosweep] ${trial_tag} shard0 failed with exit=${status0}" >&2
    fi
    if [[ "${status1}" -ne 0 ]]; then
      echo "[autosweep] ${trial_tag} shard1 failed with exit=${status1}" >&2
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${trial_tag}" \
    "${max_parallel_games}" \
    "${vllm_max_concurrency}" \
    "${elapsed_sec}" \
    "${overall_status}" >> "${SWEEP_RESULTS_TSV}"

  echo "[autosweep] ${trial_tag} status=${overall_status} elapsed_sec=${elapsed_sec}"
}

python benchmark/scripts/shard_game_manifest.py \
  --input-csv "${MANIFEST_CSV}" \
  --output-dir "${FULL_SHARD_DIR}" \
  --num-shards 2 \
  --prefix "${RUN_ID}"

FULL_SHARD0_CSV="${FULL_SHARD_DIR}/${RUN_ID}_shard_00_of_02.csv"
FULL_SHARD1_CSV="${FULL_SHARD_DIR}/${RUN_ID}_shard_01_of_02.csv"
SAMPLE_SHARD0_CSV="${SAMPLE_SHARD_DIR}/calibration_shard_00_of_02.csv"
SAMPLE_SHARD1_CSV="${SAMPLE_SHARD_DIR}/calibration_shard_01_of_02.csv"
SAMPLE_GAMES_SHARD0="$(((SWEEP_SAMPLE_GAMES + 1) / 2))"
SAMPLE_GAMES_SHARD1="$((SWEEP_SAMPLE_GAMES / 2))"

test -s "${FULL_SHARD0_CSV}"
test -s "${FULL_SHARD1_CSV}"

python benchmark/scripts/sample_game_manifest.py \
  --input-csv "${FULL_SHARD0_CSV}" \
  --output-csv "${SAMPLE_SHARD0_CSV}" \
  --max-games "${SAMPLE_GAMES_SHARD0}" \
  --column "${SWEEP_SAMPLE_COLUMN}" \
  --salt "${SWEEP_SAMPLE_SALT}_shard0"

python benchmark/scripts/sample_game_manifest.py \
  --input-csv "${FULL_SHARD1_CSV}" \
  --output-csv "${SAMPLE_SHARD1_CSV}" \
  --max-games "${SAMPLE_GAMES_SHARD1}" \
  --column "${SWEEP_SAMPLE_COLUMN}" \
  --salt "${SWEEP_SAMPLE_SALT}_shard1"

test -s "${SAMPLE_SHARD0_CSV}"
test -s "${SAMPLE_SHARD1_CSV}"

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

printf 'trial_tag\tmax_parallel_games\tvllm_max_concurrency\telapsed_sec\tstatus\n' > "${SWEEP_RESULTS_TSV}"

IFS=',' read -r -a MAX_GAME_CANDIDATES <<< "${SWEEP_MAX_PARALLEL_GAMES_CANDIDATES}"
IFS=',' read -r -a CONCURRENCY_CANDIDATES <<< "${SWEEP_VLLM_MAX_CONCURRENCY_CANDIDATES}"

for max_games in "${MAX_GAME_CANDIDATES[@]}"; do
  max_games="${max_games//[[:space:]]/}"
  [[ -n "${max_games}" ]] || continue
  for concurrency in "${CONCURRENCY_CANDIDATES[@]}"; do
    concurrency="${concurrency//[[:space:]]/}"
    [[ -n "${concurrency}" ]] || continue
    run_trial "${max_games}" "${concurrency}"
  done
done

read -r BEST_MAX_PARALLEL_GAMES_PER_SERVER BEST_VLLM_MAX_CONCURRENCY BEST_ELAPSED_SEC < <(
  python - "${SWEEP_RESULTS_TSV}" "${SWEEP_CHOICE_JSON}" <<'PY'
import csv
import json
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
choice_path = Path(sys.argv[2])
rows = []
with results_path.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        if row.get("status") != "ok":
            continue
        rows.append(
            {
                "trial_tag": row["trial_tag"],
                "max_parallel_games": int(row["max_parallel_games"]),
                "vllm_max_concurrency": int(row["vllm_max_concurrency"]),
                "elapsed_sec": int(row["elapsed_sec"]),
            }
        )
if not rows:
    raise SystemExit("no successful autosweep trials")
rows.sort(
    key=lambda rec: (
        rec["elapsed_sec"],
        rec["max_parallel_games"] * rec["vllm_max_concurrency"],
        rec["max_parallel_games"],
        rec["vllm_max_concurrency"],
    )
)
best = rows[0]
choice_path.parent.mkdir(parents=True, exist_ok=True)
choice_path.write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(best["max_parallel_games"], best["vllm_max_concurrency"], best["elapsed_sec"])
PY
)

echo "[autosweep] selected max_parallel_games=${BEST_MAX_PARALLEL_GAMES_PER_SERVER} vllm_max_concurrency=${BEST_VLLM_MAX_CONCURRENCY} elapsed_sec=${BEST_ELAPSED_SEC}"

FULL_RUN_ID_0="${RUN_ID}_shard0"
FULL_RUN_ID_1="${RUN_ID}_shard1"

run_shard "${FULL_SHARD0_CSV}" "${PORT0}" "${TMP_OUTPUT_ROOT}/shard0" "${FULL_RUN_ID_0}" "${BEST_MAX_PARALLEL_GAMES_PER_SERVER}" "${BEST_VLLM_MAX_CONCURRENCY}" "${TMP_ROOT}/full_runs_index" &
PIPE0_PID="$!"
run_shard "${FULL_SHARD1_CSV}" "${PORT1}" "${TMP_OUTPUT_ROOT}/shard1" "${FULL_RUN_ID_1}" "${BEST_MAX_PARALLEL_GAMES_PER_SERVER}" "${BEST_VLLM_MAX_CONCURRENCY}" "${TMP_ROOT}/full_runs_index" &
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
  --shard-run-dir "${TMP_OUTPUT_ROOT}/shard0/${FULL_RUN_ID_0}" \
  --shard-run-dir "${TMP_OUTPUT_ROOT}/shard1/${FULL_RUN_ID_1}"

python benchmark/scripts/run_split_pipeline.py \
  --split-root "${SPLIT_ROOT}" \
  --stages index

echo "Autosweep results: ${SWEEP_RESULTS_TSV}"
echo "Autosweep choice: ${SWEEP_CHOICE_JSON}"
echo "Final merged run: ${FINAL_RUN_DIR}"
