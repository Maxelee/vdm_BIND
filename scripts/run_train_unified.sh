#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=160G
#SBATCH --constraint=a100-40gb
#SBATCH -t 01-00:00:00
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-6
#SBATCH -o /mnt/home/mlee1/ceph/logs/unified_%a.o%j
#SBATCH -e /mnt/home/mlee1/ceph/logs/unified_%a.e%j

# ==============================================================================
# Run All Model Types with Unified Training Script
# ==============================================================================
# This script uses SLURM array jobs to train all model types in parallel.
# Each array task trains a different model type.
#
# Usage:
#   sbatch scripts/run_train_unified.sh
#
# Model types (array index -> model):
#   0 -> vdm
#   1 -> triple
#   2 -> ddpm
#   3 -> dsm
#   4 -> interpolant
#   5 -> ot_flow
#   6 -> consistency
# ==============================================================================

module restore test_torch
source /mnt/home/mlee1/venvs/torch3/bin/activate

# Unset SLURM environment variables that confuse PyTorch Lightning
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

# Make sure all 4 GPUs are visible
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Change to project directory
cd /mnt/home/mlee1/vdm_BIND

# Define model types and configs
MODELS=(
    "vdm"
    "triple"
    "ddpm"
    "dsm"
    "interpolant"
    "ot_flow"
    "consistency"
)

CONFIGS=(
    "configs/clean_vdm_aggressive_stellar.ini"
    "configs/clean_vdm_triple.ini"
    "configs/ddpm.ini"
    "configs/dsm.ini"
    "configs/interpolant.ini"
    "configs/ot_flow.ini"
    "configs/consistency.ini"
)

# Get model and config for this array task
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "=============================================="
echo "Running model: $MODEL"
echo "Config: $CONFIG"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "=============================================="

# Run training
srun python train_unified.py --model $MODEL --config $CONFIG
