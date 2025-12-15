#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=180G
#SBATCH --constraint=a100-40gb
#SBATCH -t 03-00:00:00
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=2
#SBATCH --array=0-7
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
#   7 -> stochastic_interpolant (interpolant with stochastic=True)
# ==============================================================================

module restore test_torch
source /mnt/home/mlee1/venvs/torch3/bin/activate

# For PyTorch Lightning DDP with SLURM, use srun with ntasks=num_gpus
# Each task will handle one GPU via Lightning's SLURM auto-detection

# Make sure both GPUs are visible
export CUDA_VISIBLE_DEVICES=0,1

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
    "interpolant"
)

CONFIGS=(
    "configs/clean_vdm_aggressive_stellar.ini"
    "configs/clean_vdm_triple.ini"
    "configs/ddpm.ini"
    "configs/dsm.ini"
    "configs/interpolant.ini"
    "configs/ot_flow.ini"
    "configs/consistency.ini"
    "configs/stochastic_interpolant.ini"
)

# Get model and config for this array task
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "=============================================="
echo "Running model: $MODEL"
echo "Config: $CONFIG"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "=============================================="

# Run training with srun (Lightning auto-detects SLURM and uses DDP)
srun python train_unified.py --model $MODEL --config $CONFIG
