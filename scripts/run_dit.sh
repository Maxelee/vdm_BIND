#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=180G
#SBATCH --constraint=a100-40gb
#SBATCH -t 03-00:00:00
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#SBATCH -o /mnt/home/mlee1/ceph/logs/dit.o%j
#SBATCH -e /mnt/home/mlee1/ceph/logs/dit.e%j

# ==============================================================================
# Train DiT (Diffusion Transformer) Model - 4 GPUs, 32-bit precision
# ==============================================================================

# Prevent Lightning from getting confused by SLURM
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

source /mnt/home/mlee1/venvs/torch3/bin/activate

cd /mnt/home/mlee1/vdm_BIND

echo "=============================================="
echo "Running DiT training with 4 GPUs (32-bit precision)"
echo "Config: configs/dit.ini (version 1)"
echo "=============================================="
nvidia-smi

python train_unified.py --model dit --config configs/dit.ini
