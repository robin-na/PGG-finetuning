#!/bin/bash
#SBATCH --job-name=pgg-simulation
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1          # one task per GPU
#SBATCH --cpus-per-task=8            # 8 CPU threads per rank/GPU
#SBATCH --mem=64G                    # 64G is usually fine for 8B; 96G gives headroom
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

mkdir -p logs

# 1) load conda shell functions from the cluster base
eval "$(/home/software/anaconda3/2023.07/bin/conda shell.bash hook)"

# 2) activate your env by absolute path (not by name)
conda activate /home/robinna/.conda/envs/llm_conda

# 3) sanity print to prove which Python you have
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
which python

# Hugging Face token: either export before sbatch,
#   export HUGGINGFACE_HUB_TOKEN=hf_xxx
# or store in ~/.hf_token and read here:

echo "Running on $(hostname)"
echo "Running on $(hostname)"
nvidia-smi

python Simulation_robin/run_simulation.py --provider local --base_model google/gemma-3-27b-it --use_peft False --include_reasoning True \
    --persona random_summary --output_root output_persona_summary --env_csv Simulation_robin/df_analysis_val_dedup_from14T.csv \


