#!/bin/bash
#SBATCH --job-name=pgg-stat-gpu-seq
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

eval "$(/home/software/anaconda3/2023.07/bin/conda shell.bash hook)"
conda activate /home/robinna/.conda/envs/llm_conda

export PYTHONUNBUFFERED=1
export PGG_GPU_DEVICE="${GPU_DEVICE:-cuda}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-512}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
DROPOUT="${DROPOUT:-0.10}"
LR="${LR:-0.0005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
MAX_BATCHES_PER_EPOCH="${MAX_BATCHES_PER_EPOCH:-}"
SEED="${SEED:-0}"

MICRO_RUN_ID="${MICRO_RUN_ID:-${SLURM_JOB_ID:-manual}_val40_dedup_gpu_sequence_archetype}"
MACRO_RUN_ID="${MACRO_RUN_ID:-${SLURM_JOB_ID:-manual}_val40_treatment_avg_gpu_sequence_archetype}"
MICRO_REPORT_ID="${MICRO_REPORT_ID:-${SLURM_JOB_ID:-manual}_val40_dedup_gpu_sequence_vs_history_vs_archetype_vs_random_report}"
MACRO_REPORT_ID="${MACRO_REPORT_ID:-${SLURM_JOB_ID:-manual}_val40_treatment_avg_gpu_sequence_vs_history_vs_archetype_vs_random_report}"

MICRO_BASELINE_RUNS="${MICRO_BASELINE_RUNS:-260310_val40_dedup_history_archetype,260310_val40_dedup_archetype_cluster,260309_val40_dedup_random_baseline}"
MICRO_BASELINE_LABELS="${MICRO_BASELINE_LABELS:-history_archetype,archetype_cluster,random_baseline}"
MACRO_BASELINE_RUNS="${MACRO_BASELINE_RUNS:-260310_val40_treatment_avg_history_archetype,260310_val40_treatment_avg_archetype_cluster,260309_val40_treatment_avg_random_baseline}"
MACRO_BASELINE_LABELS="${MACRO_BASELINE_LABELS:-history_archetype,archetype_cluster,random_baseline}"

ARTIFACT_ARGS=()
if [[ -n "$ARTIFACTS_ROOT" ]]; then
  ARTIFACT_ARGS=(--artifacts_root "$ARTIFACTS_ROOT")
fi

SIM_ARTIFACT_ARGS=()
if [[ -n "$ARTIFACTS_ROOT" ]]; then
  SIM_ARTIFACT_ARGS=(--archetype_artifacts_root "$ARTIFACTS_ROOT")
fi

TRAIN_EXTRA_ARGS=()
if [[ -n "$MAX_BATCHES_PER_EPOCH" ]]; then
  TRAIN_EXTRA_ARGS+=(--max_batches_per_epoch "$MAX_BATCHES_PER_EPOCH")
fi

function stage() {
  local name="$1"
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== ${name} ====="
}

stage "environment"
hostname
pwd
which python
python -u -c "import torch; print({'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count()})"
nvidia-smi || true

stage "train gpu sequence policy"
python -u simulation_statistical/train_gpu_sequence_model.py \
  "${ARTIFACT_ARGS[@]}" \
  --device "$PGG_GPU_DEVICE" \
  --epochs "$TRAIN_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --hidden_dim "$HIDDEN_DIM" \
  --dropout "$DROPOUT" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --progress_every "$PROGRESS_EVERY" \
  --seed "$SEED" \
  "${TRAIN_EXTRA_ARGS[@]}"

stage "run micro validation simulation"
python -u simulation_statistical/micro/run_micro_simulation.py \
  --strategy gpu_sequence_archetype \
  --run_id "$MICRO_RUN_ID" \
  --seed "$SEED" \
  --debug_print \
  "${SIM_ARTIFACT_ARGS[@]}"

stage "run macro validation simulation"
python -u simulation_statistical/macro/run_macro_simulation.py \
  --strategy gpu_sequence_archetype \
  --run_id "$MACRO_RUN_ID" \
  --seed "$SEED" \
  --debug_print \
  "${SIM_ARTIFACT_ARGS[@]}"

stage "build micro comparison report"
python -u simulation_statistical/micro/analysis/run_analysis.py \
  --compare_run_ids "${MICRO_RUN_ID},${MICRO_BASELINE_RUNS}" \
  --compare_labels "gpu_sequence_archetype,${MICRO_BASELINE_LABELS}" \
  --analysis_run_id "$MICRO_REPORT_ID"

stage "build macro comparison report"
python -u simulation_statistical/macro/analysis/run_analysis.py \
  --compare_run_ids "${MACRO_RUN_ID},${MACRO_BASELINE_RUNS}" \
  --compare_labels "gpu_sequence_archetype,${MACRO_BASELINE_LABELS}" \
  --analysis_run_id "$MACRO_REPORT_ID" \
  --shared_games_only

stage "done"
echo "micro run: benchmark_statistical/micro/runs/${MICRO_RUN_ID}"
echo "macro run: benchmark_statistical/macro/runs/${MACRO_RUN_ID}"
echo "micro report: benchmark_statistical/micro/reports/${MICRO_REPORT_ID}"
echo "macro report: benchmark_statistical/macro/reports/${MACRO_REPORT_ID}"
