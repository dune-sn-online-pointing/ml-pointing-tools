#!/bin/bash

echo "=========================================="
echo "CT v71 Deep Architecture Training"
echo "=========================================="
echo "Date: $(date)"
echo "Host: $(hostname)"

source /afs/cern.ch/work/e/evilla/private/dune/source-py11.sh

echo "Python: $(which python3)"
cd /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools

python3 channel_tagging/models/train_ct_deep_architecture.py \
    -j channel_tagging/json/v71_deep_arch_20k.json

exit_code=$?
echo "=========================================="
echo "Finished with exit code: $exit_code"
echo "=========================================="
exit $exit_code
