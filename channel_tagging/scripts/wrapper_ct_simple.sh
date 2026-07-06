#!/bin/bash
# HTCondor wrapper for CT volume training

set -e

# Parse arguments
JSON_CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--json)
            JSON_CONFIG="$2"
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
echo "CHANNEL TAGGING TRAINING - HTCondor Job"
echo "========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "JSON config: $JSON_CONFIG"
echo "========================================="

# Navigate to project directory.
# Prefer deriving from the JSON config absolute path: <repo>/channel_tagging/json/*.json
if [[ -f "$JSON_CONFIG" ]]; then
    PROJECT_DIR="$(cd "$(dirname "$JSON_CONFIG")/../.." && pwd)"
elif [[ -n "${_CONDOR_JOB_IWD:-}" ]] && [[ -f "${_CONDOR_JOB_IWD}/scripts/init.sh" ]]; then
    PROJECT_DIR="$_CONDOR_JOB_IWD"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$PROJECT_DIR"

# Setup environment using init.sh
echo "Setting up environment using init.sh..."
source "$PROJECT_DIR/scripts/init.sh"
echo ""

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi 2>/dev/null || echo "No GPU available (CPU-only training)"
echo ""

# Run training
cd "$PROJECT_DIR"

echo "Working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

echo "Starting training..."
echo "Command: python3 channel_tagging/models/train_ct_volume_batch_reload.py -j $JSON_CONFIG"
echo ""

python3 channel_tagging/models/train_ct_volume_batch_reload.py -j "$JSON_CONFIG"

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE
