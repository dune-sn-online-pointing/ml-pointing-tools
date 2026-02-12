#!/bin/bash

echo "=========================================="
echo "CT v73 - Cluster Images Training"
echo "=========================================="
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Setup environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh

echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"
echo ""

# Navigate to working directory
cd /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools

# Run training
echo "Starting training..."
python3 channel_tagging/models/train_ct_cluster_batch_reload.py \
    --json channel_tagging/json/v73_cluster_batch_reload_50k.json

echo ""
echo "Training finished: $(date)"
echo "=========================================="
