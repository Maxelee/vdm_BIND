#!/bin/bash
#SBATCH --job-name=vdm_cosmo
#SBATCH --output=/mnt/home/mlee1/ceph/logs/vdm_cosmo_norm_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/vdm_cosmo_norm_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100-40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40
#SBATCH --mem=180G

# ==============================================================================
# VDM Cosmo-Norm Training
# ==============================================================================
# Uses cosmological normalization:
#   - DM fields divided by Omega_m before log transform
#   - Baryonic fields divided by Omega_b before log transform
# This removes first-order cosmology dependence for better generalization.
# ==============================================================================

# Prevent Lightning from getting confused by SLURM
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

# Activate environment
source /mnt/home/mlee1/venvs/torch3/bin/activate

# Change to project directory
cd /mnt/home/mlee1/vdm_BIND

# Create logs directory
mkdir -p /mnt/home/mlee1/ceph/logs

# Print info
echo "=============================================="
echo "VDM Cosmo-Norm Training"
echo "=============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPUs: 4x A100-40GB"
echo "Config: configs/clean_vdm_cosmo_norm.ini"
echo ""
echo "Cosmological Normalization:"
echo "  - DM fields / Omega_m"
echo "  - Baryonic fields / Omega_b"
echo "=============================================="
nvidia-smi

# Run training with 4 GPUs (Lightning auto-detects available GPUs)
python train_unified.py \
    --model vdm \
    --config configs/clean_vdm_cosmo_norm.ini

echo "Training completed at $(date)"
