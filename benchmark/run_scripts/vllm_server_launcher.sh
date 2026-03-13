#!/bin/bash

python_has_vllm() {
  python - <<'PY' >/dev/null 2>&1
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("vllm") else 1)
PY
}

ensure_singularity_available() {
  if command -v singularity >/dev/null 2>&1; then
    return 0
  fi

  if command -v module >/dev/null 2>&1; then
    module load "${VLLM_SINGULARITY_MODULE}" >/dev/null 2>&1 || true
  fi

  command -v singularity >/dev/null 2>&1
}

resolve_vllm_launcher() {
  local launcher="${VLLM_LAUNCHER:-auto}"

  case "${launcher}" in
    host)
      if command -v vllm >/dev/null 2>&1 || python_has_vllm; then
        printf 'host\n'
        return 0
      fi
      echo "[error] VLLM_LAUNCHER=host but vllm is not installed in ${CONDA_ENV_PATH}." >&2
      return 1
      ;;
    singularity)
      if ! ensure_singularity_available; then
        echo "[error] VLLM_LAUNCHER=singularity but singularity is unavailable. Try module load ${VLLM_SINGULARITY_MODULE}." >&2
        return 1
      fi
      if [[ ! -f "${VLLM_SIF_PATH}" ]]; then
        echo "[error] VLLM_LAUNCHER=singularity but image is missing: ${VLLM_SIF_PATH}" >&2
        return 1
      fi
      printf 'singularity\n'
      return 0
      ;;
    auto)
      if command -v vllm >/dev/null 2>&1 || python_has_vllm; then
        printf 'host\n'
        return 0
      fi
      if ensure_singularity_available && [[ -f "${VLLM_SIF_PATH}" ]]; then
        printf 'singularity\n'
        return 0
      fi
      echo "[error] Could not find a usable vLLM launcher. Install vllm in ${CONDA_ENV_PATH} or provide ${VLLM_SIF_PATH} for singularity." >&2
      return 1
      ;;
    *)
      echo "[error] Unsupported VLLM_LAUNCHER=${launcher}. Expected auto, host, or singularity." >&2
      return 1
      ;;
  esac
}

print_vllm_logs() {
  if [[ -f "${TMP_ROOT}/vllm_gpu0.log" ]]; then
    echo "[error] tail ${TMP_ROOT}/vllm_gpu0.log" >&2
    tail -n 40 "${TMP_ROOT}/vllm_gpu0.log" >&2 || true
  fi
  if [[ -f "${TMP_ROOT}/vllm_gpu1.log" ]]; then
    echo "[error] tail ${TMP_ROOT}/vllm_gpu1.log" >&2
    tail -n 40 "${TMP_ROOT}/vllm_gpu1.log" >&2 || true
  fi
}

start_vllm_server() {
  local gpu_id="$1"
  local port="$2"
  local log_path="$3"
  local launcher
  local hf_token
  local cmd=()

  launcher="$(resolve_vllm_launcher)"

  if [[ -n "${VLLM_HOST_HF_HOME:-}" ]]; then
    mkdir -p "${VLLM_HOST_HF_HOME}"
  fi

  hf_token="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

  case "${launcher}" in
    host)
      if command -v vllm >/dev/null 2>&1; then
        cmd=(
          env
          "CUDA_VISIBLE_DEVICES=${gpu_id}"
          vllm serve "${VLLM_MODEL}"
          --host 127.0.0.1
          --port "${port}"
          --dtype "${VLLM_DTYPE}"
          --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
          --disable-log-requests
        )
      else
        cmd=(
          env
          "CUDA_VISIBLE_DEVICES=${gpu_id}"
          python -m vllm.entrypoints.openai.api_server
          --model "${VLLM_MODEL}"
          --host 127.0.0.1
          --port "${port}"
          --dtype "${VLLM_DTYPE}"
          --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
          --disable-log-requests
        )
      fi
      ;;
    singularity)
      cmd=(
        env
        "CUDA_VISIBLE_DEVICES=${gpu_id}"
        "SINGULARITYENV_CUDA_VISIBLE_DEVICES=${gpu_id}"
        "SINGULARITYENV_HF_HOME=${VLLM_CONTAINER_HF_HOME}"
        "SINGULARITYENV_HF_TOKEN=${hf_token}"
        "SINGULARITYENV_HUGGING_FACE_HUB_TOKEN=${hf_token}"
        singularity exec --nv
        -B "${VLLM_HOST_HF_HOME}:${VLLM_CONTAINER_HF_HOME}"
        "${VLLM_SIF_PATH}"
        vllm serve "${VLLM_MODEL}"
        --host 127.0.0.1
        --port "${port}"
        --dtype "${VLLM_DTYPE}"
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
        --disable-log-requests
      )
      ;;
  esac

  if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
    cmd+=(--max-model-len "${VLLM_MAX_MODEL_LEN}")
  fi

  echo "[info] launching vLLM on GPU ${gpu_id} via ${launcher} at http://127.0.0.1:${port}/v1" >&2
  if [[ "${launcher}" == "singularity" ]]; then
    echo "[info] singularity image: ${VLLM_SIF_PATH}" >&2
  fi

  "${cmd[@]}" >"${log_path}" 2>&1 &
  echo $!
}

wait_for_server() {
  local port="$1"
  python - "${port}" <<'PY'
import sys
import time
import urllib.request

port = int(sys.argv[1])
url = f"http://127.0.0.1:{port}/v1/models"

for _ in range(180):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if 200 <= response.status < 300:
                sys.exit(0)
    except Exception:
        time.sleep(2)

sys.exit(1)
PY
}
