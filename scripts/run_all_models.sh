#!/bin/bash
# ==============================================================================
# Submit training jobs for all model types
# ==============================================================================
#
# This script submits SLURM jobs to train all supported model types.
#
# Usage:
#   # Submit all models (default: 2 GPUs each)
#   ./scripts/run_all_models.sh
#
#   # Submit specific models
#   ./scripts/run_all_models.sh interpolant ot_flow
#
#   # Override GPU count
#   GPUS=4 ./scripts/run_all_models.sh
#
# ==============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GPUS=${GPUS:-2}
TIME=${TIME:-"48:00:00"}

# All supported model types and their configs
declare -A MODEL_CONFIGS=(
    ["clean"]="configs/yaml/clean_vdm.yaml"
    ["triple"]="configs/yaml/triple_vdm.yaml"
    ["interpolant"]="configs/yaml/interpolant.yaml"
    ["stochastic_interpolant"]="configs/yaml/stochastic_interpolant.yaml"
    ["ot_flow"]="configs/yaml/ot_flow.yaml"
    ["dsm"]="configs/yaml/dsm.yaml"
    ["ddpm"]="configs/yaml/ddpm.yaml"
    ["consistency"]="configs/yaml/consistency.yaml"
)

# Parse arguments
if [ $# -eq 0 ]; then
    MODELS=("clean" "triple" "interpolant" "stochastic_interpolant" "ot_flow" "dsm" "ddpm" "consistency")
else
    MODELS=("$@")
fi

echo "======================================"
echo "VDM-BIND: Submit All Model Training"
echo "======================================"
echo "Project root: $PROJECT_ROOT"
echo "GPUs per job: $GPUS"
echo "Time limit: $TIME"
echo "Models: ${MODELS[*]}"
echo "======================================"

# Create log directory
mkdir -p /mnt/home/mlee1/ceph/slurm_logs

# Submit jobs
for model in "${MODELS[@]}"; do
    CONFIG="${MODEL_CONFIGS[$model]}"
    
    if [ -z "$CONFIG" ]; then
        echo "ERROR: Unknown model type: $model"
        echo "Supported models: ${!MODEL_CONFIGS[*]}"
        exit 1
    fi
    
    CONFIG_PATH="$PROJECT_ROOT/$CONFIG"
    
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "ERROR: Config file not found: $CONFIG_PATH"
        exit 1
    fi
    
    echo ""
    echo "Submitting $model model..."
    echo "  Config: $CONFIG"
    
    JOB_ID=$(sbatch \
        --job-name="vdm_${model}" \
        --output="/mnt/home/mlee1/ceph/slurm_logs/vdm_${model}_%j.out" \
        --error="/mnt/home/mlee1/ceph/slurm_logs/vdm_${model}_%j.err" \
        --time="$TIME" \
        --partition=gpu \
        --constraint=a100-40gb \
        --nodes=1 \
        --ntasks=1 \
        --gpus-per-task="$GPUS" \
        --cpus-per-task=40 \
        --mem=180G \
        --wrap="source /mnt/home/mlee1/venvs/torch3/bin/activate && cd $PROJECT_ROOT && python train_unified.py --config $CONFIG" \
        | awk '{print $4}')
    
    echo "  Job ID: $JOB_ID"
done

echo ""
echo "======================================"
echo "All jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "======================================"
