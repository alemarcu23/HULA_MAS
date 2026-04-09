#!/bin/bash
#SBATCH --job-name=sar-sim
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# ============================================================================
# SAR Simulation on Delft Blue — In-process Transformers Backend
#
# Prerequisites (run ONCE on login node):
#   module load 2024r1 python/3.10 cuda/12.1
#   python -m venv ~/sar-venv
#   source ~/sar-venv/bin/activate
#   pip install -r requirements.txt
#   pip install torch transformers accelerate
#
# Download model on login node (has internet):
#   huggingface-cli download Qwen/Qwen3-8B \
#       --local-dir /scratch/$USER/MAS/models/Qwen3-8B
#
# Submit:  sbatch slurm/run_sar.sh
# Monitor: tail -f slurm-<jobid>.out
# ============================================================================

set -euo pipefail

# --- Configuration (edit these) ---------------------------------------------
MODEL_PATH="/scratch/$USER/MAS/models/Qwen3-8B"
PROJECT_DIR="/scratch/$USER/MAS/SAR-env"
# ----------------------------------------------------------------------------

# Load modules
module load 2024r1
module load python/3.10
module load cuda/12.1

# Activate virtualenv
source ~/sar-venv/bin/activate

cd "$PROJECT_DIR"

# Job-specific log directory on scratch
export SAR_LOG_DIR="/scratch/$USER/MAS/logs/sar-${SLURM_JOB_ID}"
mkdir -p "$SAR_LOG_DIR"

# Point model loading to scratch (no internet on compute nodes)
export SAR_MODEL_PATH="$MODEL_PATH"
export HF_HOME="/scratch/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="/scratch/$USER/.cache/huggingface"
export HF_HUB_OFFLINE=1

# Real-time log output (no Python buffering)
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Model:     $MODEL_PATH"
echo "Log dir:   $SAR_LOG_DIR"
echo "Backend:   transformers (in-process)"
echo "=========================================="

# Verify model files exist
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "[ERROR] Model not found at $MODEL_PATH"
    echo "        Download with: huggingface-cli download Qwen/Qwen3-8B --local-dir $MODEL_PATH"
    exit 1
fi

# --- Run SAR simulation -----------------------------------------------------
# main.py reads SAR_MODEL_PATH and SAR_LOG_DIR automatically when hpc_mode=True
echo "[run] Starting SAR simulation..."
python main.py 2>&1 | tee "$SAR_LOG_DIR/simulation.log"
SIM_EXIT=$?

echo "[done] Simulation exited with code $SIM_EXIT"
echo "[done] Logs at: $SAR_LOG_DIR"
echo "[done] Metrics: $SAR_LOG_DIR/simulation_metrics.json"
