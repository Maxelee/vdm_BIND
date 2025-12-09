#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --constraint=a100-40gb
#SBATCH -t 03-3:15:00
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --job-name=bind_array
#SBATCH --output=logs/bind_array_%A_%a.out
#SBATCH --error=logs/bind_array_%A_%a.err
# ============================================================================
# SLURM Array Job for Parallel BIND Processing
# ============================================================================
#
# This script runs multiple simulations in parallel using SLURM array jobs.
# Each array task processes ONE simulation on its own GPU.
#
# Usage Examples:
#
#   # CV suite (25 simulations, indices 0-24)
#   sbatch --array=0-24 run_bind_parallel_array.sh
#
#   # CV suite with max 10 concurrent jobs
#   sbatch --array=0-24%10 run_bind_parallel_array.sh
#
#   # SB35 suite (102 test simulations, indices 0-101)
#   sbatch --array=0-101%10 --export=ALL,SUITE="sb35" run_bind_parallel_array.sh
#
#   # 1P suite (use sim list file approach - see below)
#   sbatch --array=0-60%10 --export=ALL,SUITE="1p" run_bind_parallel_array.sh
#
#   # Specific simulations only
#   sbatch --array=0,5,10,15,20 run_bind_parallel_array.sh
#
# ============================================================================

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
module load test_torch
source /mnt/home/mlee1/venvs/torch3/bin/activate

# ============================================================================
# Configuration Parameters
# ============================================================================

# Suite and Model Configuration
SUITE=${SUITE:-"sb35"}  # cv, sb35, or 1p
MODEL_NAME=${MODEL_NAME:-"clean_vdm_aggressive_stellar_nofocus"}
CONFIG_PATH=${CONFIG_PATH:-"/mnt/home/mlee1/variational-diffusion-cdm/configs/clean_vdm_aggressive_stellar.ini"}

# Output Configuration
BASE_OUTPATH=${BASE_OUTPATH:-"/mnt/home/mlee1/ceph/BIND2d_new"}

# Simulation Parameters
MASS_THRESHOLD=${MASS_THRESHOLD:-1e13}
GRIDSIZE=${GRIDSIZE:-1024}
USE_ENHANCED=${USE_ENHANCED:-"--use_enhanced"}
REALIZATIONS=${REALIZATIONS:-10}

# Control flags
REGENERATE=${REGENERATE:-"--regenerate"}
REGENERATE_ALL=${REGENERATE_ALL:-"--regenerate_all"}
REPASTE=${REPASTE:-"--repaste"}
DO_HYDRO_REPLACE=${DO_HYDRO_REPLACE:-""}

# Analysis flags
RUN_ANALYSES=${RUN_ANALYSES:-"--run_analyses"}
RUN_1P_EXTREMES=${RUN_1P_EXTREMES:-""}
PREP_ONLY=${PREP_ONLY:-""}
BATCH_SIZE=${BATCH_SIZE:-10}

# ============================================================================
# Determine which simulation to process based on array task ID
# ============================================================================

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "============================================================================"
echo "BIND Array Job - Task ${TASK_ID}"
echo "Suite: ${SUITE}"
echo "============================================================================"

# For 1P suite, we need to map array indices to simulation names
if [ "$SUITE" = "1p" ]; then
    # Generate 1P simulation names list (or use a pre-generated file)
    SIM_LIST_FILE="/mnt/home/mlee1/BIND3d/1p_sim_list.txt"
    
    # Create the sim list file if it doesn't exist
    if [ ! -f "$SIM_LIST_FILE" ]; then
        echo "Creating 1P simulation list file..."
        python -c "
import pandas as pd
param_file = '/mnt/home/mlee1/Sims/IllustrisTNG/L50n512/1P/CosmoAstroSeed_IllustrisTNG_L50n512_1P.txt'
oneP_params = pd.read_csv(param_file, delim_whitespace=True)
names = oneP_params['#Name'].to_list()
with open('$SIM_LIST_FILE', 'w') as f:
    for name in names:
        f.write(name + '\n')
"
    fi
    
    # Get the simulation name for this task ID
    SIM_NAME=$(sed -n "$((TASK_ID + 1))p" "$SIM_LIST_FILE")
    SIM_NUMS_ARG="--sim_nums $SIM_NAME"

elif [ "$SUITE" = "sb35" ]; then
    # For SB35, use only test set simulations
    # Test set indices are determined by sim_* directories in the test data folder
    SB35_TEST_LIST_FILE="/mnt/home/mlee1/BIND3d/sb35_test_sim_list.txt"
    
    # Create the SB35 test sim list file if it doesn't exist
    if [ ! -f "$SB35_TEST_LIST_FILE" ]; then
        echo "Creating SB35 test simulation list file..."
        ls /mnt/home/mlee1/ceph/train_data_rotated2_128_cpu/test | sed 's/sim_//' | sort -n > "$SB35_TEST_LIST_FILE"
    fi
    
    # Get the simulation number for this task ID
    SIM_NUM=$(sed -n "$((TASK_ID + 1))p" "$SB35_TEST_LIST_FILE")
    SIM_NUMS_ARG="--sim_nums $SIM_NUM"

else
    # For CV, use the array task ID directly as simulation number
    SIM_NUMS_ARG="--sim_nums $TASK_ID"
fi

echo "Processing simulation: ${SIM_NUMS_ARG}"

# Run the unified pipeline for this single simulation
srun python run_bind_unified.py \
    --suite $SUITE \
    --model_name $MODEL_NAME \
    --config_path $CONFIG_PATH \
    --base_outpath $BASE_OUTPATH \
    --mass_threshold $MASS_THRESHOLD \
    --gridsize $GRIDSIZE \
    --batch_size $BATCH_SIZE \
    $USE_ENHANCED \
    --realizations $REALIZATIONS \
    $REGENERATE \
    $REGENERATE_ALL \
    $REPASTE \
    $DO_HYDRO_REPLACE \
    $RUN_ANALYSES \
    $PREP_ONLY \
    $SIM_NUMS_ARG

EXIT_CODE=$?

echo ""
echo "============================================================================"
echo "Task ${TASK_ID} complete!"
echo "Exit code: $EXIT_CODE"
echo "============================================================================"

exit $EXIT_CODE
