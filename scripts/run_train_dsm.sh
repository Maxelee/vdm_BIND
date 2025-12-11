#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=160G
#SBATCH --constraint=a100-40gb
#SBATCH -t 03-3:15:00
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40        # 40 CPUs for dataloader workers (10 per GPU)
#SBATCH --ntasks-per-node=1       # ONE task - PyTorch Lightning handles DDP internally
#SBATCH -o /mnt/home/mlee1/ceph/logs/dsm_OUTPUT.o%j
#SBATCH -e /mnt/home/mlee1/ceph/logs/dsm_OUTPUT.e%j

# ============================================================================
# DSM (Denoising Score Matching) Training Script
# ============================================================================
# 
# This script trains DSM models using the same UNet architecture as VDM/Interpolant,
# but with Denoising Score Matching loss instead of VDM ELBO or flow matching.
#
# Key features:
#   - Same architecture as VDM (Fourier features, cross-attention, FiLM)
#   - VP-SDE noise schedule
#   - DSM loss: || epsilon_hat - epsilon ||^2
#   - Fair comparison with VDM/Interpolant
#
# Usage:
#   sbatch scripts/run_train_dsm.sh
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
echo "DSM (Custom UNet) Training Job Starting"
echo "============================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "============================================"

# PyTorch Lightning DDP: Run Python directly, Lightning spawns GPU workers
python train_dsm.py --config configs/dsm.ini
