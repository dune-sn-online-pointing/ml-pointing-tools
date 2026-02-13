#!/bin/bash
cd /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/electron_direction
source ../scripts/init.sh --quiet
python3 models/train_three_plane_simple.py -j json/v54_200k_simple.json
