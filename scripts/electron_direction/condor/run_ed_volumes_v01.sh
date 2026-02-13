#!/bin/bash
#
# Run script for ED Volume Training v01
# Sets up environment and executes training
#

echo "=========================================="
echo "ED Volume Training v01"
echo "Job started at: $(date)"
echo "Running on: $(hostname)"
echo "=========================================="

# Setup LCG environment with TensorFlow
echo "Setting up LCG environment..."
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

# Set working directory
WORK_DIR="/afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/electron_direction"
cd ${WORK_DIR}/models || exit 1

echo "Working directory: $(pwd)"
echo ""

# Configuration file
CONFIG_FILE="${WORK_DIR}/json/ed_volumes_v01.json"

echo "=========================================="
echo "Starting training with config:"
echo "  ${CONFIG_FILE}"
echo "=========================================="
echo ""

# Run training
python3 train_volumes.py -j ${CONFIG_FILE}

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=========================================="

exit ${EXIT_CODE}
