#!/bin/bash
#SBATCH -p cca
#SBATCH --constraint=rome
#SBATCH -J make_subimages
#SBATCH -N 8
#SBATCH -n 128
#SBATCH --exclusive
#SBATCH -o OUTPUT.o%j
#SBATCH -e OUTPUT.e%j
#SBATCH --mail-user=mel2260@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH -t 6-23:15:00

module load openmpi
module load python
module load python-mpi
module load hdf5

source /mnt/home/mlee1/venvs/gen_train_data/bin/activate

# Install mpi4py if not available
# Run with MPI
srun -n 128 python3 -u process_simulations2_cpu.py --start_sim 0 --end_sim 1024
