#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=160G
#SBATCH --constraint=a100-40gb
#SBATCH -t 03-3:15:00
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40        # 40 CPUs for dataloader workers (10 per GPU)
#SBATCH --ntasks-per-node=1       # ONE task - PyTorch Lightning handles DDP internally
#SBATCH -o /mnt/home/mlee1/ceph/logs/ot_flow_OUTPUT.o%j
#SBATCH -e /mnt/home/mlee1/ceph/logs/ot_flow_OUTPUT.e%j

# ============================================================================
# Optimal Transport Flow Matching Training Script
# ============================================================================
# 
# This script trains OT Flow Matching models (Lipman et al., 2022).
#
# Key features:
#   - Optimal transport coupling for straighter interpolation paths
#   - Better sample quality for structured astronomical data
#   - Uses POT library for OT computation
#
# Requirements:
#   pip install POT  # Python Optimal Transport
#
# Usage:
#   sbatch scripts/run_train_ot_flow.sh
#
# Reference:
#   Lipman et al. (2022) "Flow Matching for Generative Modeling"
#   https://arxiv.org/abs/2210.02747
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
echo "OT Flow Matching Training Job Starting"
echo "============================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""
echo "Checking POT installation..."
python -c "import ot; print(f'POT version: {ot.__version__}')" || echo "POT not installed! Run: pip install POT"
echo "============================================"

# PyTorch Lightning DDP: ONE process, it spawns GPU workers internally
srun python train_ot_flow.py --config configs/ot_flow.ini
