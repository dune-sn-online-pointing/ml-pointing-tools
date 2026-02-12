#!/bin/bash

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--json)
            JSON_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Source environment
source /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/scripts/init.sh

# Run training
cd /afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/electron_direction
python models/train_three_plane_hyperopt.py --input_json "$JSON_PATH"
