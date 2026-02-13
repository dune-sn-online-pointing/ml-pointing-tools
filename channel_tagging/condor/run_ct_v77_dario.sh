#!/bin/bash
#
# Wrapper script for HTCondor to run v77_dario CT training
#

set -e

echo "========================================================================"
echo "CHANNEL TAGGING V77_DARIO TRAINING"
echo "========================================================================"
echo "Started at: $(date)"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Source environment
echo "Setting up environment..."
cd /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools
source scripts/init.sh
echo ""

# Go to channel_tagging directory
cd channel_tagging

# Check GPU availability
echo "Checking GPU..."
nvidia-smi || echo "No GPU detected (will use CPU)"
echo ""

# Parse arguments
JSON_FILE="$1"
PLANE="${2:-X}"
RELOAD_EPOCHS="${3:-5}"
MAX_SAMPLES="${4:-10000}"

if [ -z "$JSON_FILE" ]; then
    echo "ERROR: No JSON config file provided!"
    echo "Usage: $0 <json_file> [plane] [reload_epochs] [max_samples]"
    exit 1
fi

echo "Configuration:"
echo "  JSON file: $JSON_FILE"
echo "  Plane: $PLANE"
echo "  Reload every: $RELOAD_EPOCHS epochs"
echo "  Max samples per class: $MAX_SAMPLES"
echo ""

# Run training
echo "Starting training..."
python3 models/train_ct_v77_dario.py \
    --json "$JSON_FILE" \
    --plane "$PLANE" \
    --reload-epochs "$RELOAD_EPOCHS" \
    --max-samples "$MAX_SAMPLES"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "Finished at: $(date)"
echo "========================================================================"

exit $EXIT_CODE
