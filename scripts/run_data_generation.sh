#!/bin/bash
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH -J cpu_job1_0-341
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -o OUTPUT_CPU_JOB1.o%j
#SBATCH -e OUTPUT_CPU_JOB1.e%j
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH -t 7-00:00:00

# Load modules
module load python
module load hdf5

# Activate virtual environment
source /mnt/home/mlee1/venvs/torch3/bin/activate

# Process simulations 0-341
srun python3 -u process_simulations2_cpu.py --start_sim 0 --end_sim 100
