#!/bin/bash
# HTCondor wrapper for enhanced ED training

set -e

# Parse arguments
JSON_CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--json)
            JSON_CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$JSON_CONFIG" ]]; then
    echo "Error: JSON config required (-j/--json)"
    exit 1
fi

echo "========================================="
echo "ELECTRON DIRECTION TRAINING - Enhanced"
echo "========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "JSON config: $JSON_CONFIG"
echo "========================================="

# Navigate to project directory
PROJECT_DIR="/afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools"
cd "$PROJECT_DIR"

# Setup environment using init.sh
echo "Setting up environment using init.sh..."
source "$PROJECT_DIR/scripts/init.sh"
echo ""

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi 2>/dev/null || echo "No GPU available (CPU-only training)"
echo ""

# Run training
cd "$PROJECT_DIR"

echo "Working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

echo "Starting enhanced ED training..."
echo "Command: python3 electron_direction/models/train_three_plane_enhanced.py -j $JSON_CONFIG"
echo ""

python3 electron_direction/models/train_three_plane_enhanced.py -j "$JSON_CONFIG"
EXIT_CODE=$?

echo ""
echo "========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE
