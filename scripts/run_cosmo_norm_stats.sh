#!/bin/bash
#SBATCH --job-name=cosmo_norm_stats
#SBATCH --output=logs/cosmo_norm_stats_%j.out
#SBATCH --error=logs/cosmo_norm_stats_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --partition=cca

# Compute normalization statistics for cosmo_norm mode
# Uses 64 parallel workers for fast file loading
# Uses ALL pixels from each file for quantile transformer (no subsampling)

echo "=================================================="
echo "Computing Cosmological Normalization Statistics"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Started: $(date)"
echo ""

# Activate environment
source /mnt/home/mlee1/venvs/torch3/bin/activate

# Change to project directory
cd /mnt/home/mlee1/vdm_BIND

# Create logs directory if needed
mkdir -p logs

# Run the stats computation
# - 100k file samples for z-score stats
# - ALL pixels used for quantile transformer (100k files × 16384 pixels = 1.6B samples)
# - 64 parallel workers
python scripts/compute_cosmo_norm_stats.py \
    --n_samples 100000 \
    --n_workers 64 \
    --output_dir /mnt/home/mlee1/vdm_BIND/data/

echo ""
echo "=================================================="
echo "Finished: $(date)"
echo "=================================================="
