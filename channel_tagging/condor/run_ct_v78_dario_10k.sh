#!/bin/bash

echo "=========================================="
echo "CT v78 Dario Architecture Training - 10k batches"
echo "=========================================="
echo "Date: $(date)"
echo "Host: $(hostname)"

source /afs/cern.ch/work/e/evilla/private/dune/source-py11.sh

echo "Python: $(which python3)"
echo "TensorFlow: $(python3 -c 'import tensorflow as tf; print(tf.__version__)')"

# Check for GPU
echo ""
echo "=== GPU Info ==="
nvidia-smi

cd /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/channel_tagging

python3 models/train_ct_volume_batch_reload.py \
    -j json/v78_dario_10k.json

exit_code=$?
echo "=========================================="
echo "Finished with exit code: $exit_code"
echo "=========================================="
exit $exit_code
