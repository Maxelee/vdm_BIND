#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=160G
#SBATCH --constraint=a100-40gb
#SBATCH -t 03-3:15:00
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40        # 40 CPUs for dataloader workers (10 per GPU)
#SBATCH --ntasks-per-node=1       # ONE task - PyTorch Lightning handles DDP internally
#SBATCH -o /mnt/home/mlee1/ceph/logs/interpolant_OUTPUT.o%j
#SBATCH -e /mnt/home/mlee1/ceph/logs/interpolant_OUTPUT.e%j

# ============================================================================
# Interpolant / Flow Matching Training Script
# ============================================================================
# 
# This script trains flow matching models using stochastic interpolants.
# The interpolant learns a velocity field that transports x_0 -> x_1.
#
# Key features:
#   - Flow matching loss (MSE on velocity prediction)
#   - No noise schedule to tune
#   - Deterministic ODE sampling
#   - Often faster convergence than diffusion
#   - Typically needs fewer sampling steps (20-50 vs 250-1000)
#
# Usage:
#   sbatch scripts/run_train_interpolant.sh
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
echo "Interpolant/Flow Matching Training Job Starting"
echo "============================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "============================================"

# PyTorch Lightning DDP: ONE process, it spawns GPU workers internally
srun python train_interpolant.py --config configs/interpolant.ini
