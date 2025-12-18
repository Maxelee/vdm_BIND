#!/bin/bash
#SBATCH --job-name=stoch_interp
#SBATCH --output=/mnt/home/mlee1/ceph/tb_logs3/stoch_interp_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/tb_logs3/stoch_interp_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100-40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-task=4
#SBATCH --mem=180G

# Unset SLURM_NTASKS to prevent PyTorch Lightning conflicts
unset SLURM_NTASKS

# Activate environment
source /mnt/home/mlee1/venvs/torch3/bin/activate
cd /mnt/home/mlee1/vdm_BIND

echo "Starting Stochastic Interpolant training (BaryonBridge formulation)"
echo "Using 4 GPUs"
echo "Config: configs/stochastic_interpolant.ini"
echo "Version: 2"

# Run training
python train_unified.py \
    --model interpolant \
    --config configs/stochastic_interpolant.ini

echo "Training complete"
