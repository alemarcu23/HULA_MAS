#!/bin/bash
#SBATCH --job-name=simulation_3agent_qwen8B_difficult_env
#SBATCH --partition=gpu-a100
#SBATCH --time=07:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --output=/scratch/%u/MAS/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/%u/MAS/slurm_logs/%x-%j.err

# --- Infrastructure (edit these) --------------------------------------------
MODEL_PATH="/scratch/ammarcu/MAS/models/qwen3-8b"
PROJECT_DIR="/scratch/ammarcu/MAS/HULA_MAS"
ENV_PREFIX="/scratch/ammarcu/.conda/envs/MAS-SAR"
# ----------------------------------------------------------------------------

# --- World config -----------------------------------------------------------
WORLD_PRESET="${WORLD_PRESET:-static}"
NUM_AGENTS="${NUM_AGENTS:-3}"
# ----------------------------------------------------------------------------

# --- Per-agent experiment config (override via --export or env vars) --------
# Comma-separated lists; cycles if shorter than NUM_AGENTS.
#
# agent_presets:   capability profile  — scout | medic | heavy_lifter | generalist
# agent_roles:     behavioural role    — scout | medic | heavy_lifter | rescuer | generalist
# comm_strategies: always_respond | busy_aware
# reasoning_strategies: io | cot | react | reflexion | self_refine | self_reflective_tot
# planning_strategies:  io | deps | td | voyager
# replanning_policies:  every_turn | critic_gated
# capability_knowledge: informed | discovery
AGENT_PRESETS="${AGENT_PRESETS:-generalist,generalist,generalist}"
AGENT_ROLES="${AGENT_ROLES:-generalist,generalist,generalist}"
COMM_STRATEGIES="${COMM_STRATEGIES:-always_respond,always_respond,always_respond}"
REASONING_STRATEGIES="${REASONING_STRATEGIES:-react,react,react}"
PLANNING_STRATEGIES="${PLANNING_STRATEGIES:-io,io,io}"
REPLANNING_POLICIES="${REPLANNING_POLICIES:-every_turn,every_turn,every_turn}"
CAPABILITY_KNOWLEDGE="${CAPABILITY_KNOWLEDGE:-informed}"
# ----------------------------------------------------------------------------

module purge
module load 2025
module load miniconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# HuggingFace — no internet on compute nodes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/scratch/$USER/.cache/huggingface"

export SAR_MODEL_PATH="$MODEL_PATH"
export LLM_ENABLE_THINKING=0

export SAR_LOG_DIR="/scratch/$USER/MAS/logs"

# Experiment name: unique per run by default (job ID), overridable via RUN_NAME.
# Usage: sbatch --export=ALL,RUN_NAME=experiment_A run_sar.sh
RUN_NAME="${RUN_NAME:-job_${SLURM_JOB_ID}}"
export SAR_EXPERIMENT_NAME="${RUN_NAME}"

mkdir -p "$SAR_LOG_DIR"
mkdir -p /scratch/$USER/MAS/slurm_logs

cd "$PROJECT_DIR"
nvidia-smi   # confirm GPU allocation

echo "[run] RUN_NAME=${RUN_NAME}  PRESET=${WORLD_PRESET}  AGENTS=${NUM_AGENTS}"
echo "[run] PRESETS=${AGENT_PRESETS}  ROLES=${AGENT_ROLES}"
echo "[run] OUTPUT=${SAR_LOG_DIR}/${RUN_NAME}"

python main.py \
  --preset              "$WORLD_PRESET" \
  --num_agents          "$NUM_AGENTS" \
  --agent_presets       "$AGENT_PRESETS" \
  --agent_roles         "$AGENT_ROLES" \
  --comm_strategies     "$COMM_STRATEGIES" \
  --reasoning_strategies "$REASONING_STRATEGIES" \
  --planning_strategies  "$PLANNING_STRATEGIES" \
  --replanning_policies  "$REPLANNING_POLICIES" \
  --capability_knowledge "$CAPABILITY_KNOWLEDGE"

echo "[done] Simulation completed."
