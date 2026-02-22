#!/bin/bash
#SBATCH --job-name=vdm_strong_film
#SBATCH --output=/mnt/home/mlee1/ceph/logs/vdm_strong_film_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/vdm_strong_film_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100-40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40
#SBATCH --mem=180G

# ==============================================================================
# VDM Strong FiLM Conditioning Experiment
# ==============================================================================
# 
# Hypothesis: Default FiLM init std=0.01 keeps γ≈1 (near-identity), so cosmology
# conditioning is too weak. Increasing to std=0.1 allows stronger modulation.
#
# Key change: film_init_std = 0.1 (10x increase from default 0.01)
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
echo "VDM Strong FiLM Conditioning Experiment"
echo "=============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPUs: 4x A100-40GB"
echo "Config: configs/clean_vdm_strong_film.ini"
echo "Key change: film_init_std = 0.1 (10x stronger)"
echo "=============================================="
nvidia-smi

# Run training with 4 GPUs (Lightning auto-detects available GPUs)
python train_unified.py \
    --model vdm \
    --config configs/clean_vdm_strong_film.ini

echo "Training completed at $(date)"
