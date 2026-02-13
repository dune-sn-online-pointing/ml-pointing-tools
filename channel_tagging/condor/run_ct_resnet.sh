#!/bin/bash
# Execution script for ResNet CT training

CONFIG_FILE=$1

# Setup environment
cd /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools
source scripts/init.sh

# Run training
cd channel_tagging/models
python train_ct_resnet.py --config ../json/${CONFIG_FILE}

echo "Training completed!"
