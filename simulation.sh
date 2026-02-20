#!/bin/bash
#SBATCH --job-name=pgg-simulation
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL

mkdir -p logs
eval "$(/home/software/anaconda3/2023.07/bin/conda shell.bash hook)"
conda activate /home/kehangzh/.conda/envs/pgg_env

export HF_TOKEN="${HF_TOKEN}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
which python
echo "Running on $(hostname)"
nvidia-smi

python - <<'PY'
import os
print("HF_TOKEN set:", bool(os.getenv("HF_TOKEN")))
print("HUGGING_FACE_HUB_TOKEN set:", bool(os.getenv("HUGGING_FACE_HUB_TOKEN")))
PY

python Simulation_robin/run_simulation.py \
  --provider local \
  --base_model google/gemma-3-27b-it \
  --use_peft False \
  --include_reasoning True \
  --env_csv Simulation_robin/df_analysis_val_19th.csv \
  --persona random_full_transcript \
  --output_root output_test
