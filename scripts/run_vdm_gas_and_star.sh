#!/bin/bash
#SBATCH --job-name=vdm_gas_star
#SBATCH --output=/mnt/home/mlee1/ceph/logs/vdm_gas_and_star_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/vdm_gas_and_star_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100-40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40
#SBATCH --mem=180G

# ==============================================================================
# VDM Gas & Star Improved Training
# ==============================================================================
# 
# Changes from clean_vdm_regularized:
#   1. param_prediction_weight: 0.01 → 0.1 (10x stronger)
#   2. Added baryon_fraction_loss with weight 0.1
#   3. FiLM conditioning (already enabled by default)
#
# Purpose: Improve cosmology → gas/stellar relationship learning
# ==============================================================================

# Prevent Lightning from getting confused by SLURM
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

# Activate environment
source /mnt/home/mlee1/venvs/torch3/bin/activate

# Change to project directory
cd /mnt/home/mlee1/vdm_BIND

# Print info
echo "=============================================="
echo "VDM Gas & Star Improved Training"
echo "=============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPUs: 4x A100-40GB"
echo "Config: configs/clean_vdm_gas_and_star.ini"
echo ""
echo "Key Changes:"
echo "  - param_prediction_weight: 0.01 → 0.1"
echo "  - baryon_fraction_loss: enabled (weight=0.1)"
echo "  - FiLM conditioning: already enabled by default"
echo "=============================================="
nvidia-smi

# Run training with 4 GPUs (Lightning auto-detects available GPUs)
python train_unified.py \
    --model vdm \
    --config configs/clean_vdm_gas_and_star.ini

echo "Training completed at $(date)"
