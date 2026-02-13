#!/bin/bash

# Navigate to project root
cd /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools

# Run ED volume batch reload training
python3 electron_direction/models/train_ed_volume_batch_reload.py \
    -j electron_direction/json/ed_volumes_v02_100k.json \
    --plane X \
    --reload-epochs 5
