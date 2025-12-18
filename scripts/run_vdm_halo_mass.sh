#!/bin/bash
#SBATCH --job-name=vdm_halo_mass
#SBATCH --output=/mnt/home/mlee1/ceph/logs/vdm_halo_mass_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/vdm_halo_mass_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100-40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40
#SBATCH --mem=180G

# ==============================================================================
# VDM with Halo Mass Conditioning
# ==============================================================================
# Based on regularized config, but adds log10(halo_mass) as 36th conditioning
# parameter to help model learn mass-dependent baryon profiles.

# Prevent Lightning from getting confused by SLURM
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

# Activate environment
source /mnt/home/mlee1/venvs/torch3/bin/activate

# Change to project directory
cd /mnt/home/mlee1/vdm_BIND

# Print info
echo "=============================================="
echo "VDM with Halo Mass Conditioning"
echo "=============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPUs: 4x A100-40GB"
echo "Config: configs/clean_vdm_halo_mass.ini"
echo "Key feature: log10(halo_mass) added as 36th parameter"
echo "=============================================="
nvidia-smi

# Run training with 4 GPUs (Lightning auto-detects available GPUs)
python train_unified.py \
    --model vdm \
    --config configs/clean_vdm_halo_mass.ini

echo "Training completed at $(date)"
