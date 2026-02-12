#!/bin/bash
# HTCondor submission wrapper for ED v52 three-plane with hyperopt

# Print start info
echo "==========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "==========================================="

# Setup DUNE environment
source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup dunesw v09_91_02d00 -q e26:prof

# Setup Python with TensorFlow
source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh

# Parse arguments
JSON_FILE=""

while getopts "j:" opt; do
    case $opt in
        j) JSON_FILE="$OPTARG";;
        *) echo "Usage: $0 -j <json_config>"; exit 1;;
    esac
done

if [ -z "$JSON_FILE" ]; then
    echo "Error: JSON config file required (-j)"
    exit 1
fi

echo "Configuration:"
echo "  JSON: $JSON_FILE"
echo "==========================================="

# Run training
cd /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools
python electron_direction/models/ed_training.py --input_json "$JSON_FILE"

EXIT_CODE=$?

echo "==========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "==========================================="

exit $EXIT_CODE
