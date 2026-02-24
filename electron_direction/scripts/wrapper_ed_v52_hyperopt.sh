#!/bin/bash
set -e

JSON_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--json) JSON_FILE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1; ;;
    esac
done

[[ -z "$JSON_FILE" ]] && { echo "Error: JSON config required (-j)"; exit 1; }

PROJECT_DIR="/afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/scripts/init.sh"

echo "==========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "JSON config: $JSON_FILE"
echo "==========================================="

echo "Starting ED v52 three-plane training with hyperopt..."
python3 electron_direction/models/train_three_plane_hyperopt.py --input_json "$JSON_FILE"
EXIT_CODE=$?

echo "==========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "==========================================="

exit $EXIT_CODE
