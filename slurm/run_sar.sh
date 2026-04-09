#!/bin/bash
#SBATCH --job-name=sar-sim
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-eemcs-msc-cs

# --- Configuration (edit these) ---------------------------------------------
MODEL_PATH="/scratch/ammarcu/MAS/models/qwen3.5-9B"
PROJECT_DIR="/scratch/ammarcu/MAS/SAR-env"
# ----------------------------------------------------------------------------

# Load modules
module load 2024r1
module load python/3.10
module load cuda/12.1
module load openmpi
module load py-pip
module load py-numpy
module load py-pyyaml
module load py-tqdm
module load ffmpeg

# Activate virtualenv
source ~/sar-venv/bin/activate

cd "$PROJECT_DIR"

# Point model loading to scratch (no internet on compute nodes)
export SAR_MODEL_PATH="$MODEL_PATH"
export HF_HOME="/scratch/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="/scratch/$USER/.cache/huggingface"
export HF_HUB_OFFLINE=1


# --- Run SAR simulation -----------------------------------------------------
# main.py reads SAR_MODEL_PATH and SAR_LOG_DIR automatically when hpc_mode=True
echo "[run] Starting SAR simulation..."
python main.py


echo "[done] Simulation completed."
