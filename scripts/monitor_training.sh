#!/bin/bash
# ==============================================================================
# Monitor Training Jobs
# ==============================================================================
# Shows status of all running training jobs including:
# - Current epoch, step, and loss
# - Most recent checkpoint
# - GPU memory usage
#
# Usage:
#   ./scripts/monitor_training.sh          # One-time check
#   watch -n 30 ./scripts/monitor_training.sh  # Auto-refresh every 30s
# ==============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration
LOG_DIR="/mnt/home/mlee1/ceph/logs"
TB_LOGS="/mnt/home/mlee1/ceph/tb_logs3"

echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}         VDM-BIND Training Monitor - $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""

# Get running jobs
RUNNING_JOBS=$(squeue -u $USER -h -o "%i %j %N %M" 2>/dev/null | grep -E "run_trai|unified")

if [ -z "$RUNNING_JOBS" ]; then
    echo -e "${YELLOW}No training jobs currently running.${NC}"
    exit 0
fi

# Model name mapping
declare -A MODEL_NAMES=(
    [0]="VDM (clean)"
    [1]="Triple VDM"
    [2]="DDPM"
    [3]="DSM"
    [4]="Interpolant"
    [5]="OT Flow"
    [6]="Consistency"
    [7]="Stochastic Interp"
)

declare -A MODEL_DIRS=(
    [0]="clean_vdm_aggressive_stellar"
    [1]="triple_vdm_separate_models"
    [2]="ddpm_ncsnpp_vp"
    [3]="dsm_3ch"
    [4]="interpolant_3ch"
    [5]="ot_flow_3ch"
    [6]="consistency_3ch"
    [7]="stochastic_interpolant_3ch"
)

# Process each running job
echo "$RUNNING_JOBS" | while read -r JOB_ID JOB_NAME NODE ELAPSED; do
    # Extract array task ID
    if [[ "$JOB_ID" =~ _([0-9]+)$ ]]; then
        TASK_ID="${BASH_REMATCH[1]}"
    else
        TASK_ID="?"
    fi
    
    # Extract base job ID
    BASE_JOB_ID=$(echo "$JOB_ID" | cut -d'_' -f1)
    
    MODEL_NAME="${MODEL_NAMES[$TASK_ID]:-Unknown}"
    MODEL_DIR="${MODEL_DIRS[$TASK_ID]:-unknown}"
    
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${GREEN}[$TASK_ID] $MODEL_NAME${NC}"
    echo -e "    Job: ${JOB_ID} | Node: ${NODE} | Elapsed: ${ELAPSED}"
    
    # Find the most recent log file for this task (any job ID)
    LOG_FILE=$(ls -t ${LOG_DIR}/unified_${TASK_ID}.o* 2>/dev/null | head -1)
    
    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        # Extract latest epoch/step/loss from log
        LATEST_PROGRESS=$(grep -E "Epoch [0-9]+:" "$LOG_FILE" 2>/dev/null | tail -1)
        if [ -n "$LATEST_PROGRESS" ]; then
            # Parse epoch number
            EPOCH=$(echo "$LATEST_PROGRESS" | grep -oP "Epoch \K[0-9]+" | head -1)
            # Parse iteration speed
            SPEED=$(echo "$LATEST_PROGRESS" | grep -oP "[0-9.]+it/s" | tail -1)
            echo -e "    ${BLUE}Progress:${NC} Epoch ${EPOCH:-?} | Speed: ${SPEED:-?}"
        fi
        
        # Get latest loss values
        LATEST_LOSS=$(grep -E "val/loss|val/elbo|val_loss" "$LOG_FILE" 2>/dev/null | tail -1)
        if [ -n "$LATEST_LOSS" ]; then
            # Extract loss value
            LOSS_VAL=$(echo "$LATEST_LOSS" | grep -oP "(val/loss|val/elbo|val_loss)=[\d.]+" | tail -1)
            if [ -n "$LOSS_VAL" ]; then
                echo -e "    ${BLUE}Loss:${NC} $LOSS_VAL"
            fi
        fi
        
        # Alternative: get from tqdm progress bar
        TQDM_LOSS=$(tail -5 "$LOG_FILE" 2>/dev/null | grep -oP "(train/elbo|train/loss|val/elbo|val/loss)[^,]*" | head -2 | tr '\n' ' ')
        if [ -n "$TQDM_LOSS" ]; then
            echo -e "    ${BLUE}Latest:${NC} $TQDM_LOSS"
        fi
    else
        echo -e "    ${YELLOW}Log file not found${NC}"
    fi
    
    # Find most recent checkpoint
    CKPT_DIR="${TB_LOGS}/${MODEL_DIR}"
    if [ -d "$CKPT_DIR" ]; then
        # Find latest version directory
        LATEST_VERSION=$(ls -td ${CKPT_DIR}/version_* 2>/dev/null | head -1)
        if [ -n "$LATEST_VERSION" ]; then
            LATEST_CKPT=$(ls -t ${LATEST_VERSION}/checkpoints/*.ckpt 2>/dev/null | head -1)
            if [ -n "$LATEST_CKPT" ]; then
                CKPT_NAME=$(basename "$LATEST_CKPT")
                CKPT_SIZE=$(du -h "$LATEST_CKPT" 2>/dev/null | cut -f1)
                CKPT_TIME=$(stat -c %y "$LATEST_CKPT" 2>/dev/null | cut -d'.' -f1)
                echo -e "    ${BLUE}Checkpoint:${NC} $CKPT_NAME ($CKPT_SIZE)"
                echo -e "    ${BLUE}Saved at:${NC} $CKPT_TIME"
            fi
        fi
    fi
    
    echo ""
done

# GPU Usage Summary
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}GPU Usage by Node:${NC}"
echo ""

# Get unique nodes
NODES=$(echo "$RUNNING_JOBS" | awk '{print $3}' | sort -u)

for NODE in $NODES; do
    # Try to get GPU info from the node
    GPU_INFO=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$NODE" "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits" 2>/dev/null)
    
    if [ -n "$GPU_INFO" ]; then
        echo -e "  ${GREEN}$NODE:${NC}"
        echo "$GPU_INFO" | while IFS=',' read -r GPU_ID MEM_USED MEM_TOTAL GPU_UTIL; do
            MEM_USED=$(echo "$MEM_USED" | tr -d ' ')
            MEM_TOTAL=$(echo "$MEM_TOTAL" | tr -d ' ')
            GPU_UTIL=$(echo "$GPU_UTIL" | tr -d ' ')
            MEM_PCT=$((MEM_USED * 100 / MEM_TOTAL))
            
            # Color based on memory usage
            if [ "$MEM_PCT" -gt 90 ]; then
                MEM_COLOR=$RED
            elif [ "$MEM_PCT" -gt 70 ]; then
                MEM_COLOR=$YELLOW
            else
                MEM_COLOR=$GREEN
            fi
            
            echo -e "    GPU $GPU_ID: ${MEM_COLOR}${MEM_USED}/${MEM_TOTAL} MiB (${MEM_PCT}%)${NC} | Util: ${GPU_UTIL}%"
        done
    else
        echo -e "  ${YELLOW}$NODE: Unable to query GPU (SSH timeout or not accessible)${NC}"
    fi
done

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "Tip: Run ${CYAN}watch -n 30 ./scripts/monitor_training.sh${NC} for auto-refresh"
echo -e "${BOLD}============================================================${NC}"
