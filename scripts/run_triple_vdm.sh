#!/bin/bash
#SBATCH --job-name=triple_vdm
#SBATCH --output=/mnt/home/mlee1/ceph/tb_logs3/triple_vdm_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/tb_logs3/triple_vdm_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --constraint=a100-40gb
#SBATCH --cpus-per-task=40
#SBATCH --mem=180G

# Triple VDM (3 separate single-channel models) training on 4 GPUs

echo "=============================================="
echo "Triple VDM Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4 x A100-40GB"
echo "Start time: $(date)"
echo "=============================================="

# Activate environment
source /mnt/home/mlee1/venvs/torch3/bin/activate

# Navigate to project root
cd /mnt/home/mlee1/vdm_BIND

# Unset SLURM_NTASKS to let Lightning handle DDP
unset SLURM_NTASKS

# Run training with train_unified.py
python train_unified.py --model triple --config configs/clean_vdm_triple_regularized.ini

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
