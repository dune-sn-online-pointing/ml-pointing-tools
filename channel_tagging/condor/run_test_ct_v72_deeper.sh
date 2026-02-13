#!/bin/bash

cd /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools

echo "=========================================="
echo "Testing CT v72 deeper model"
echo "Time: $(date)"
echo "=========================================="

# Run test
python3 channel_tagging/test_ct_v72_deeper.py

echo ""
echo "=========================================="
echo "Test completed: $(date)"
echo "=========================================="
