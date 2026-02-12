#!/bin/bash

############################################################
# HTCondor Wrapper Script for Channel Tagging v77_dario batch reload
# Usage: ./run_ct_v77_dario_batch.sh <config_json> <plane> <reload_epochs> <max_samples>
############################################################

# Get arguments
CONFIG_JSON=$1
PLANE=$2
RELOAD_EPOCHS=$3
MAX_SAMPLES=$4

# Source the environment setup
source scripts/init.sh

# Print environment info
echo "=== Environment Info ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python3)"
echo "TensorFlow version: $(python3 -c 'import tensorflow as tf; print(tf.__version__)')"

# Check for GPU
echo ""
echo "=== GPU Info ==="
nvidia-smi

# Change to the project directory
cd /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/channel_tagging

# Run the training
echo ""
echo "=== Starting Training ==="
echo "Config: $CONFIG_JSON"
echo "Plane: $PLANE"
echo "Reload every: $RELOAD_EPOCHS epochs"
echo "Max samples per reload: $MAX_SAMPLES"
echo ""

python3 models/train_ct_v77_dario.py \
    --json "$CONFIG_JSON" \
    --plane "$PLANE" \
    --reload-epochs "$RELOAD_EPOCHS" \
    --max-samples "$MAX_SAMPLES"

exit_code=$?

echo ""
echo "=== Training Complete ==="
echo "Exit code: $exit_code"
echo "Date: $(date)"

exit $exit_code
