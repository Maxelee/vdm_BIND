#!/bin/bash
#SBATCH -p cca
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH -J lowmass_datagen
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -o logs/lowmass_datagen_%j.out
#SBATCH -e logs/lowmass_datagen_%j.err
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH -t 7-00:00:00
#SBATCH --array=0-102

# ============================================================
# Generate training data for LOW-MASS halos (10^12 - 10^13 M_sun/h)
# - Mass range: 1e12 to 1e13 (excludes halos already in original dataset)
# - 1 projection per halo (no rotations)
# - Output goes to: train_data_rotated2_128_cpu_lowmass/
# ============================================================

# Load modules
# module load python
# module load hdf5
module restore profiles_new
# Activate virtual environment
source /mnt/home/mlee1/venvs/profiles_new/bin/activate

cd /mnt/home/mlee1/vdm_BIND

# Each array task processes ~103 simulations (1024 total / 10 tasks)
SIMS_PER_TASK=10
START_SIM=$((SLURM_ARRAY_TASK_ID * SIMS_PER_TASK))
END_SIM=$(( (SLURM_ARRAY_TASK_ID + 1) * SIMS_PER_TASK ))

# Last task handles remainder
if [ $SLURM_ARRAY_TASK_ID -eq 9 ]; then
    END_SIM=1024
fi

echo "Array task $SLURM_ARRAY_TASK_ID: Processing sims $START_SIM to $END_SIM"

srun python3 -u data_generation/process_simulations.py \
    --resolution 128 \
    --total_sims 1024 \
    --start_sim $START_SIM \
    --end_sim $END_SIM \
    --num_rotations 1 \
    --mass_threshold 1e12 \
    --mass_upper 1e13 \
    --output_suffix "_lowmass" \
    --hydro_base /mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35 \
    --nbody_base /mnt/home/mlee1/Sims/IllustrisTNG_DM/L50n512/SB35 \
    --fof_nbody_base /mnt/ceph/users/camels/FOF_Subfind/IllustrisTNG_DM/L50n512/SB35 \
    --output_base_root /mnt/home/mlee1/ceph
