#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --constraint=a100-40gb
#SBATCH -t 03-3:15:00
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#SBATCH --job-name=bind_unified
#SBATCH --output=logs/bind_unified_%A_%a.out
#SBATCH --error=logs/bind_unified_%A_%a.err

# ============================================================================
# Unified BIND Pipeline - SLURM Job Script
# ============================================================================
# 
# Description:
#   Runs the unified BIND pipeline for processing DMO simulations with
#   diffusion models to generate baryonic physics.
#
# Default Behavior:
#   - Suite: CV (all 25 simulations)
#   - Regenerate: Model outputs only (reuses grids/cutouts)
#   - Analyses: ENABLED (profiles, mass comparisons)
#
# New Features (Analysis):
#   - Profile analysis: Radial density profiles (DMO, hydro, generated)
#   - Mass comparison: Integrated mass validation and component breakdown
#   - Parameter extremes: 1P suite parameter sensitivity visualization
#
# Usage Examples:
#   
#   # Basic run (CV suite, regenerate models, with analyses)
#   sbatch run_bind_unified.sh
#   
#   # Disable regeneration (use existing generated halos)
#   sbatch --export=ALL,REGENERATE="" run_bind_unified.sh
#   
#   # Process all suites
#   sbatch --export=ALL,SUITE="all" run_bind_unified.sh
#   
#   # 1P suite with extremes plot
#   sbatch --export=ALL,SUITE="1p",RUN_1P_EXTREMES="--run_1p_extremes" run_bind_unified.sh
#   
#   # Regenerate everything from scratch
#   sbatch --export=ALL,REGENERATE="",REGENERATE_ALL="--regenerate_all" run_bind_unified.sh
#   
#   # Single simulation test
#   sbatch --export=ALL,SIM_NUMS="0",REALIZATIONS="3" run_bind_unified.sh
#
#   # If experiencing OOM (Out Of Memory) errors:
#   sbatch --export=ALL,BATCH_SIZE="5",RUN_1P_EXTREMES="" run_bind_unified.sh
#
#   # PREP ONLY MODE: Generate full maps and cutouts without BIND (useful for 1P)
#   sbatch --export=ALL,SUITE="1p",PREP_ONLY="--prep_only",REGENERATE="",RUN_ANALYSES="" run_bind_unified.sh
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
# All parameters can be overridden via environment variables when submitting
# the job. Example: sbatch --export=ALL,SUITE="cv",MODEL_NAME="nofusion" run_bind_unified.sh
# ============================================================================

# Suite and Model Configuration
# Suite and Model Configuration
SUITE=${SUITE:-"sb35"}  # cv, sb35, 1p, or all (DEFAULT: cv)
MODEL_NAME=${MODEL_NAME:-"clean_vdm_aggressive_stellar"}  # Name for output directories and plots
CONFIG_PATH=${CONFIG_PATH:-"/mnt/home/mlee1/vdm_BIND/configs/clean_vdm_aggressive_stellar.ini"}

# Output Configuration
BASE_OUTPATH=${BASE_OUTPATH:-"/mnt/home/mlee1/ceph/BIND2d_new"}

# Simulation Parameters
MASS_THRESHOLD=${MASS_THRESHOLD:-1e13}
GRIDSIZE=${GRIDSIZE:-1024}
USE_ENHANCED=${USE_ENHANCED:-"--use_enhanced"}  # Use enhanced pasting (recommended)
REALIZATIONS=${REALIZATIONS:-10}

# Optional Features
DO_HYDRO_REPLACE=${DO_HYDRO_REPLACE:-""}  # Set to "--do_hydro_replace" to compute hydro replacement baseline

# Control flags
REGENERATE=${REGENERATE:-"--regenerate"}  # Set to "--regenerate" to regenerate model outputs only (keeps grids/cutouts) - DEFAULT: ENABLED
REGENERATE_ALL=${REGENERATE_ALL:-""}  # Set to "--regenerate_all" to regenerate everything from scratch
REPASTE=${REPASTE:-"--repaste"}  # Set to "--repaste" to force repasting
SIM_NUMS=${SIM_NUMS:-""}  # Comma-separated list of sim numbers (e.g., "0,1,2")

# Analysis flags (NEW)
RUN_ANALYSES=${RUN_ANALYSES:-"--run_analyses"}  # Set to "--run_analyses" to enable profile and mass analyses (~20-30s per sim) - DEFAULT: ENABLED
RUN_1P_EXTREMES=${RUN_1P_EXTREMES:-""}  # Set to "--run_1p_extremes" for parameter extremes plot (1P suite only, ~60-90s). DISABLED by default to save memory.

# Data preparation mode (no BIND generation)
PREP_ONLY=${PREP_ONLY:-""}  # Set to "--prep_only" to only generate full maps and cutouts without running BIND

# Memory optimization (if experiencing OOM errors)
BATCH_SIZE=${BATCH_SIZE:-10}  # Reduce to 5 or lower if running out of memory during generation

# Run unified pipeline
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
    $RUN_1P_EXTREMES \
    $PREP_ONLY \
    ${SIM_NUMS:+--sim_nums $SIM_NUMS}

EXIT_CODE=$?

echo ""
echo "============================================================================"
echo "Unified BIND pipeline complete!"
echo "============================================================================"
echo "Exit code: $EXIT_CODE"
echo "Output directory: $BASE_OUTPATH"
echo "Images directory: /mnt/home/mlee1/BIND3d/imgs/$MODEL_NAME"

if [ "$RUN_ANALYSES" = "--run_analyses" ]; then
    echo "Analysis plots: /mnt/home/mlee1/BIND3d/imgs/$MODEL_NAME/*/analyses/"
fi

if [ "$RUN_1P_EXTREMES" = "--run_1p_extremes" ]; then
    echo "Parameter extremes plot: /mnt/home/mlee1/BIND3d/imgs/$MODEL_NAME/1P/analyses/"
fi

echo "============================================================================"
echo ""

exit $EXIT_CODE
