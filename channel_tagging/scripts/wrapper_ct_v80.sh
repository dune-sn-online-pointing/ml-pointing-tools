#!/bin/bash
# HTCondor wrapper for CT v80 volume training (burst-sample volumes)

set -e

JSON_CONFIG=""
TRAIN_SCRIPT="channel_tagging/models/train_ct_volume_v80.py"

while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--json)
            JSON_CONFIG="$2"
            shift 2
            ;;
        -s|--script)
            TRAIN_SCRIPT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$JSON_CONFIG" ]]; then
    echo "Error: JSON config required (-j/--json)"
    exit 1
fi

echo "========================================="
echo "CHANNEL TAGGING v80 TRAINING - HTCondor Job"
echo "========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "JSON config: $JSON_CONFIG"
echo "========================================="

if [[ -f "$JSON_CONFIG" ]]; then
    PROJECT_DIR="$(cd "$(dirname "$JSON_CONFIG")/../.." && pwd)"
elif [[ -n "${_CONDOR_JOB_IWD:-}" ]] && [[ -f "${_CONDOR_JOB_IWD}/scripts/init.sh" ]]; then
    PROJECT_DIR="$_CONDOR_JOB_IWD"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$PROJECT_DIR"

echo "Setting up environment using init.sh..."
source "$PROJECT_DIR/scripts/init.sh"
echo ""

echo "GPU Information:"
nvidia-smi 2>/dev/null || echo "No GPU available (CPU-only training)"
echo ""

echo "Working directory: $(pwd)"
echo "Starting training..."
echo "Trainer: $TRAIN_SCRIPT"
python3 "$TRAIN_SCRIPT" -j "$JSON_CONFIG"
EXIT_CODE=$?

echo ""
echo "========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE
