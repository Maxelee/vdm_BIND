#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=rome
#SBATCH -J lowmass_halos
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --exclusive
#SBATCH --array=0-102
#SBATCH -o OUTPUT_LOWMASS.o%A_%a
#SBATCH -e OUTPUT_LOWMASS.e%A_%a
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH -t 1-00:00:00

module load openmpi
module load python
module load python-mpi
module load hdf5

source /mnt/home/mlee1/venvs/gen_train_data/bin/activate

# Array job: each task handles 10 simulations
# Task 0 → sims 0-9, Task 1 → sims 10-19, ..., Task 102 → sims 1020-1023
SIMS_PER_TASK=10
START_SIM=$((SLURM_ARRAY_TASK_ID * SIMS_PER_TASK))
END_SIM=$((START_SIM + SIMS_PER_TASK))

# Cap at 1024
if [ $END_SIM -gt 1024 ]; then
    END_SIM=1024
fi

echo "Array task ${SLURM_ARRAY_TASK_ID}: processing sims ${START_SIM} to $((END_SIM - 1))"

# Process halos with 1e12 < M <= 1e13, 1 rotation each
# 16 MPI ranks per node to split the 10 sims
srun -n 16 python3 -u process_simulations2_cpu_lowmass.py --start_sim $START_SIM --end_sim $END_SIM
