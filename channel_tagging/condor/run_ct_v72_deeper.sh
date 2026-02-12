#!/bin/bash

echo "=========================================="
echo "CT v72 Deeper Architecture Training"
echo "=========================================="
echo "Date: $(date)"
echo "Host: $(hostname)"

source /afs/cern.ch/work/e/evilla/private/dune/source-py11.sh

echo "Python: $(which python3)"
cd /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools

python3 channel_tagging/models/train_ct_volume_batch_reload.py \
    -j channel_tagging/json/v72_deeper_100k.json

exit_code=$?
echo "=========================================="
echo "Finished with exit code: $exit_code"
echo "=========================================="
exit $exit_code
