#!/bin/bash

echo "=========================================="
echo "Starting Three-Plane CT Training v70"
echo "=========================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo ""

# Source LCG environment
echo "Sourcing source-py11.sh..."
source /afs/cern.ch/work/e/evilla/private/dune/source-py11.sh

echo ""
echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"
echo ""

# Run training
echo "Starting training..."
cd /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/channel_tagging/models

python3 train_ct_three_plane_batch_reload.py \
    --json /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/channel_tagging/json/v70_three_plane_v2_10k.json

exit_code=$?

echo ""
echo "=========================================="
echo "Training finished with exit code: $exit_code"
echo "Date: $(date)"
echo "=========================================="

exit $exit_code
