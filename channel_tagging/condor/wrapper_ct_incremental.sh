#!/bin/bash
# Condor wrapper for CT incremental training

echo "========================================="
echo "CT INCREMENTAL TRAINING WRAPPER"
echo "========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "Arguments: $@"
echo "========================================="

# Parse arguments
JSON_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--json)
            JSON_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            shift
            ;;
    esac
done

if [ -z "$JSON_FILE" ]; then
    echo "Error: No JSON config file specified"
    exit 1
fi

echo "JSON config: $JSON_FILE"

# Setup environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh

# Run training
cd /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools
python3 channel_tagging/models/train_ct_volume_incremental.py --json "$JSON_FILE"

EXIT_CODE=$?
echo "========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE
