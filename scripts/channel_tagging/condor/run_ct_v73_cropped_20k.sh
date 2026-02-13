#!/bin/bash

# Navigate to project root
cd /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools

# Run CT training with cropped volumes using ED architecture
python3 channel_tagging/models/train_ct_cropped_volumes_ed_arch.py \
    -j channel_tagging/json/ct_v73_cropped_20k.json
