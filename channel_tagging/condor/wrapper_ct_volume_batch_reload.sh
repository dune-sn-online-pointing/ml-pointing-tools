#!/bin/bash
# HTCondor wrapper for CT volume batch reload training

# Print environment info
echo "==========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "==========================================="

# Setup DUNE environment
source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup dunesw v09_91_02d00 -q e26:prof

# Python environment with TensorFlow
source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh

# Parse arguments
JSON_FILE=""
PLANE="X"
RELOAD_EPOCHS=5

while getopts "j:p:r:" opt; do
    case $opt in
        j) JSON_FILE="$OPTARG";;
        p) PLANE="$OPTARG";;
        r) RELOAD_EPOCHS="$OPTARG";;
        *) echo "Usage: $0 -j <json_config> [-p <plane>] [-r <reload_epochs>]"; exit 1;;
    esac
done

if [ -z "$JSON_FILE" ]; then
    echo "Error: JSON config file required (-j)"
    exit 1
fi

echo "Configuration:"
echo "  JSON: $JSON_FILE"
echo "  Plane: $PLANE"
echo "  Reload every: $RELOAD_EPOCHS epochs"
echo "==========================================="

# Run training
cd /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools
python channel_tagging/models/train_ct_volume_batch_reload.py \
    --json "$JSON_FILE" \
    --plane "$PLANE" \
    --reload-epochs "$RELOAD_EPOCHS"

EXIT_CODE=$?

echo "==========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "==========================================="

exit $EXIT_CODE
