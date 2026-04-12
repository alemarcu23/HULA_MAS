#!/bin/bash
#SBATCH --job-name=sar-sim
#SBATCH --partition=a100-small
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --output=/scratch/%u/MAS/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/%u/MAS/slurm_logs/%x-%j.err

# --- Configuration (edit these) ---------------------------------------------
MODEL_PATH="/scratch/ammarcu/MAS/models/qwen3-8b"
PROJECT_DIR="/scratch/ammarcu/MAS/HULA_MAS"
ENV_PREFIX="/scratch/ammarcu/.conda/envs/MAS-SAR"
# ----------------------------------------------------------------------------

module purge
module load 2025
module load miniconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Headless display (MATRX imports may reference display; GUI is disabled in hpc_mode)
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export DISPLAY=
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3

# HuggingFace — no internet on compute nodes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/scratch/$USER/.cache/huggingface"

# Model path
export SAR_MODEL_PATH="$MODEL_PATH"
export LLM_ENABLE_THINKING=0

# Base log directory (same for all runs)
export SAR_LOG_DIR="/scratch/$USER/MAS/logs"

# Experiment name: unique per run by default (job ID), overridable via RUN_NAME.
# Usage: sbatch --export=ALL,RUN_NAME=experiment_A run_sar.sh
RUN_NAME="${RUN_NAME:-job_${SLURM_JOB_ID}}"
export SAR_EXPERIMENT_NAME="${RUN_NAME}"

mkdir -p "$SAR_LOG_DIR"
mkdir -p /scratch/$USER/MAS/slurm_logs

cd "$PROJECT_DIR"
nvidia-smi   # confirm GPU allocation

echo "[run] RUN_NAME=${RUN_NAME}  OUTPUT=${SAR_LOG_DIR}/${RUN_NAME}"
python main.py
echo "[done] Simulation completed."
