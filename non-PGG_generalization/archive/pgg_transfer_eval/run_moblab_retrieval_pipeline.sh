#!/bin/zsh

set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required" >&2
  exit 1
fi

ROOT="/Users/kehangzh/Desktop/PGG-finetuning/non-PGG_generalization/pgg_transfer_eval"
OUT="${ROOT}/output/moblab_llm"
LIB_OUT="${ROOT}/output/oracle_library"
TASKS_DIR="${OUT}/tasks"
RESP_DIR="${OUT}/responses"
RETQ_DIR="${OUT}/retrieval_query_batch"
PRED_DIR="${OUT}/prediction_batch"
RETR_DIR="${OUT}/retrieval"
EVAL_DIR="${OUT}/evals"
VECTOR_STORE_ID="vs_69c0c916b5b88191b42eeeff6b07509a"
MODEL="gpt-4.1-mini"

mkdir -p "${RETR_DIR}" "${EVAL_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

submit_batch() {
  local requests_jsonl="$1"
  local submit_json="$2"
  python3 "${ROOT}/submit_openai_batch.py" --requests-jsonl "${requests_jsonl}" > "${submit_json}"
}

download_output() {
  local batch_id="$1"
  local output_path="$2"
  python3 "${ROOT}/download_openai_batch_output.py" --batch-id "${batch_id}" --output-path "${output_path}"
}

log "Resuming oracle library upload"
python3 "${ROOT}/upload_oracle_library_to_vector_store.py" \
  --vector-store-id "${VECTOR_STORE_ID}" \
  --mode batch \
  --batch-size 200 \
  --max-concurrency 32 \
  --resume

declare -A TASK_JSONL=(
  [task1]="${TASKS_DIR}/task1_sample_100.jsonl"
  [task2_future_mean]="${TASKS_DIR}/task2_future_mean_sample_100.jsonl"
  [task2_trajectory]="${TASKS_DIR}/task2_trajectory_sample_100.jsonl"
)

declare -A RETR_QUERY_RESP=(
  [task1]="${RESP_DIR}/task1_retrieval_query_outputs.jsonl"
  [task2_future_mean]="${RESP_DIR}/task2_future_mean_retrieval_query_outputs.jsonl"
  [task2_trajectory]="${RESP_DIR}/task2_trajectory_retrieval_query_outputs.jsonl"
)

declare -A RETR_QUERY_MANIFEST=(
  [task1]="${RETQ_DIR}/manifest_task1_sample_100_retrieval_query.jsonl"
  [task2_future_mean]="${RETQ_DIR}/manifest_task2_future_mean_sample_100_retrieval_query.jsonl"
  [task2_trajectory]="${RETQ_DIR}/manifest_task2_trajectory_sample_100_retrieval_query.jsonl"
)

declare -A RETR_OUT_DIR=(
  [task1]="${RETR_DIR}/task1"
  [task2_future_mean]="${RETR_DIR}/task2_future_mean"
  [task2_trajectory]="${RETR_DIR}/task2_trajectory"
)

for task_name in task1 task2_future_mean task2_trajectory; do
  log "Retrieving PGG candidates for ${task_name}"
  mkdir -p "${RETR_OUT_DIR[$task_name]}"
  python3 "${ROOT}/retrieve_moblab_pgg_candidates.py" \
    --vector-store-id "${VECTOR_STORE_ID}" \
    --responses-jsonl "${RETR_QUERY_RESP[$task_name]}" \
    --manifest-jsonl "${RETR_QUERY_MANIFEST[$task_name]}" \
    --output-dir "${RETR_OUT_DIR[$task_name]}"

  log "Evaluating retrieval diversity for ${task_name}"
  python3 "${ROOT}/evaluate_moblab_retrieval_diversity.py" \
    --candidates-jsonl "${RETR_OUT_DIR[$task_name]}/moblab_pgg_candidates.jsonl" \
    --output-dir "${RETR_OUT_DIR[$task_name]}/diversity"

  log "Building retrieval prediction batch for ${task_name}"
  python3 "${ROOT}/build_moblab_prediction_batch.py" \
    --tasks-jsonl "${TASK_JSONL[$task_name]}" \
    --baseline retrieval \
    --retrieval-candidates-jsonl "${RETR_OUT_DIR[$task_name]}/moblab_pgg_candidates.jsonl" \
    --output-dir "${PRED_DIR}"
done

declare -A RETR_REQUESTS=(
  [task1]="${PRED_DIR}/requests_task1_sample_100_retrieval_${MODEL}.jsonl"
  [task2_future_mean]="${PRED_DIR}/requests_task2_future_mean_sample_100_retrieval_${MODEL}.jsonl"
  [task2_trajectory]="${PRED_DIR}/requests_task2_trajectory_sample_100_retrieval_${MODEL}.jsonl"
)

declare -A RETR_MANIFEST=(
  [task1]="${PRED_DIR}/manifest_task1_sample_100_retrieval.jsonl"
  [task2_future_mean]="${PRED_DIR}/manifest_task2_future_mean_sample_100_retrieval.jsonl"
  [task2_trajectory]="${PRED_DIR}/manifest_task2_trajectory_sample_100_retrieval.jsonl"
)

declare -A RETR_RESP=(
  [task1]="${RESP_DIR}/task1_retrieval_outputs.jsonl"
  [task2_future_mean]="${RESP_DIR}/task2_future_mean_retrieval_outputs.jsonl"
  [task2_trajectory]="${RESP_DIR}/task2_trajectory_retrieval_outputs.jsonl"
)

declare -A RETR_EVAL=(
  [task1]="${EVAL_DIR}/task1_retrieval"
  [task2_future_mean]="${EVAL_DIR}/task2_future_mean_retrieval"
  [task2_trajectory]="${EVAL_DIR}/task2_trajectory_retrieval"
)

declare -A SUBMIT_JSON=(
  [task1]="${OUT}/retrieval_submit_task1.json"
  [task2_future_mean]="${OUT}/retrieval_submit_task2_future_mean.json"
  [task2_trajectory]="${OUT}/retrieval_submit_task2_trajectory.json"
)

for task_name in task1 task2_future_mean task2_trajectory; do
  log "Submitting retrieval prediction batch for ${task_name}"
  submit_batch "${RETR_REQUESTS[$task_name]}" "${SUBMIT_JSON[$task_name]}"
done

log "Polling retrieval prediction batches"
python3 - <<'PY'
import json
import time
from pathlib import Path
from openai import OpenAI

root = Path("/Users/kehangzh/Desktop/PGG-finetuning/non-PGG_generalization/pgg_transfer_eval/output/moblab_llm")
submit_paths = {
    "task1": root / "retrieval_submit_task1.json",
    "task2_future_mean": root / "retrieval_submit_task2_future_mean.json",
    "task2_trajectory": root / "retrieval_submit_task2_trajectory.json",
}
client = OpenAI()
terminal = {"completed", "failed", "expired", "cancelled"}

while True:
    statuses = {}
    all_done = True
    for task_name, path in submit_paths.items():
        payload = json.loads(path.read_text())
        batch_id = payload["batch"]["id"]
        batch = client.batches.retrieve(batch_id)
        dump = batch.model_dump() if hasattr(batch, "model_dump") else {}
        statuses[task_name] = {
            "batch_id": batch_id,
            "status": dump.get("status"),
            "request_counts": dump.get("request_counts"),
            "output_file_id": dump.get("output_file_id"),
            "error_file_id": dump.get("error_file_id"),
        }
        if dump.get("status") not in terminal:
            all_done = False
    print(json.dumps(statuses, ensure_ascii=False, indent=2), flush=True)
    if all_done:
        break
    time.sleep(30)
PY

for task_name in task1 task2_future_mean task2_trajectory; do
  log "Downloading retrieval prediction output for ${task_name}"
  batch_id="$(python3 - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${SUBMIT_JSON[$task_name]}").read_text())
print(payload["batch"]["id"])
PY
)"
  download_output "${batch_id}" "${RETR_RESP[$task_name]}"
done

for task_name in task1 task2_future_mean task2_trajectory; do
  log "Evaluating retrieval predictions for ${task_name}"
  python3 "${ROOT}/evaluate_moblab_llm_outputs.py" \
    --responses-jsonl "${RETR_RESP[$task_name]}" \
    --manifest-jsonl "${RETR_MANIFEST[$task_name]}" \
    --tasks-jsonl "${TASK_JSONL[$task_name]}" \
    --output-dir "${RETR_EVAL[$task_name]}"
done

log "Refreshing comparison plots"
python3 "${ROOT}/plot_moblab_llm_comparison.py"

log "MobLab retrieval pipeline finished"
