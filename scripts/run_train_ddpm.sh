#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=160G
#SBATCH --constraint=a100-40gb
#SBATCH -t 03-3:15:00
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40        # 40 CPUs for dataloader workers (10 per GPU)
#SBATCH --ntasks-per-node=1       # ONE task - PyTorch Lightning handles DDP internally
#SBATCH -o /mnt/home/mlee1/ceph/logs/ddpm_OUTPUT.o%j
#SBATCH -e /mnt/home/mlee1/ceph/logs/ddpm_OUTPUT.e%j

# ============================================================================
# DDPM/Score Model Training Script
# ============================================================================
# 
# This script trains score-based diffusion models using the score_models package.
# It uses the NCSNpp architecture with Denoising Score Matching (DSM) loss.
#
# Key features:
#   - VP-SDE noise schedule (like DDPM)
#   - NCSNpp architecture from Yang Song's NCSN++
#   - Input conditioning (DM + large-scale context)
#   - Vector conditioning (35 astrophysical parameters)
#   - EMA for stable sampling
#
# Usage:
#   sbatch scripts/run_train_ddpm.sh
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
echo "DDPM Training Job Starting"
echo "============================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "============================================"

# PyTorch Lightning DDP: ONE process, it spawns GPU workers internally
srun python train_ddpm.py --config configs/ddpm.ini
