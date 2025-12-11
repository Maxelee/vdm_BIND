#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=160G
#SBATCH --constraint=a100-40gb
#SBATCH -t 03-3:15:00
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40        # 40 CPUs for dataloader workers (10 per GPU)
#SBATCH --ntasks-per-node=1       # ONE task - PyTorch Lightning handles DDP internally
#SBATCH -o /mnt/home/mlee1/ceph/logs/consistency_OUTPUT.o%j
#SBATCH -e /mnt/home/mlee1/ceph/logs/consistency_OUTPUT.e%j

# ============================================================================
# Consistency Model Training Script
# ============================================================================
# 
# This script trains Consistency Models (Song et al., 2023) for DMO -> Hydro.
#
# Key features:
#   - Single-step or few-step sampling (vs 250-1000 for diffusion)
#   - Maintains diffusion-quality results
#   - Self-consistency training (CT) loss
#   - Optional denoising pre-training warm-up
#
# Usage:
#   sbatch scripts/run_train_consistency.sh
#
# Reference:
#   Song et al. (2023) "Consistency Models" https://arxiv.org/abs/2303.01469
# ============================================================================

module restore test_torch
source /mnt/home/mlee1/venvs/torch3/bin/activate

# Unset SLURM environment variables that confuse PyTorch Lightning
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

# Make sure all 4 GPUs are visible
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Print debugging info
echo "============================================"
echo "Consistency Model Training Job Starting"
echo "============================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "============================================"

# PyTorch Lightning DDP: Run Python directly, Lightning spawns GPU workers
python train_consistency.py --config configs/consistency.ini
