#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=160G
#SBATCH --constraint=a100-40gb
#SBATCH -t 03-3:15:00
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40        # 40 CPUs for dataloader workers (10 per GPU)
#SBATCH --ntasks-per-node=1       # ONE task - PyTorch Lightning handles DDP internally
#SBATCH -o /mnt/home/mlee1/ceph/logs/OUTPUT.o%j
#SBATCH -e /mnt/home/mlee1/ceph/logs/OUTPUT.e%j

module restore test_torch
source /mnt/home/mlee1/venvs/torch3/bin/activate

# Unset SLURM environment variables that confuse PyTorch Lightning
# This prevents Lightning from trying to match ntasks with devices
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

# Make sure all 4 GPUs are visible
export CUDA_VISIBLE_DEVICES=0,1,2,3

# PyTorch Lightning DDP: ONE process, it spawns GPU workers internally
srun python train_model_clean.py --config configs/clean_vdm_aggressive_stellar.ini
