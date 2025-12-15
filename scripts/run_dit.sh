#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=180G
#SBATCH --constraint=a100-40gb
#SBATCH -t 03-00:00:00
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=2
#SBATCH -o /mnt/home/mlee1/ceph/logs/dit.o%j
#SBATCH -e /mnt/home/mlee1/ceph/logs/dit.e%j

# ==============================================================================
# Train DiT (Diffusion Transformer) Model
# ==============================================================================

module restore test_torch
source /mnt/home/mlee1/venvs/torch3/bin/activate

export CUDA_VISIBLE_DEVICES=0,1

cd /mnt/home/mlee1/vdm_BIND

echo "=============================================="
echo "Running DiT training with 2 GPUs"
echo "Config: configs/dit.ini"
echo "=============================================="

srun python train_unified.py --model dit --config configs/dit.ini
