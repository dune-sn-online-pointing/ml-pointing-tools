#!/bin/bash
source /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/scripts/init.sh
cd /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools
python3 channel_tagging/models/train_ct_volume_hyperopt.py "$@"
