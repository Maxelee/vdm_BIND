#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=160G
#SBATCH --constraint=a100-40gb
#SBATCH -t 03-3:15:00
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40        # 40 CPUs for dataloader workers (10 per GPU)
#SBATCH --ntasks-per-node=1       # ONE task - PyTorch Lightning handles DDP internally
#SBATCH -o /mnt/home/mlee1/ceph/logs/stochastic_interpolant_OUTPUT.o%j
#SBATCH -e /mnt/home/mlee1/ceph/logs/stochastic_interpolant_OUTPUT.e%j

# ============================================================================
# Stochastic Interpolant / Flow Matching Training Script
# ============================================================================
# 
# This script trains flow matching models using STOCHASTIC interpolants.
# The interpolant learns a velocity field that transports x_0 -> x_1,
# with added noise during the interpolation path.
#
# Key differences from deterministic interpolant:
#   - Adds noise sigma during interpolation: x_t = (1-t)x_0 + t*x_1 + sigma*sqrt(t(1-t))*noise
#   - Can improve sample diversity at cost of some convergence speed
#   - sigma=0.1 is conservative; can increase to 0.2-0.5 for more diversity
#
# Usage:
#   sbatch scripts/run_train_stochastic_interpolant.sh
# ============================================================================

module restore test_torch
source /mnt/home/mlee1/venvs/torch3/bin/activate

# Unset SLURM environment variables that confuse PyTorch Lightning
# This prevents Lightning from trying to match ntasks with devices
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

# Make sure all 4 GPUs are visible
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Print some debugging info
echo "============================================"
echo "Stochastic Interpolant Training Job Starting"
echo "============================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "============================================"

# PyTorch Lightning DDP: Run Python directly, Lightning spawns GPU workers
python train_interpolant.py --config configs/stochastic_interpolant.ini
