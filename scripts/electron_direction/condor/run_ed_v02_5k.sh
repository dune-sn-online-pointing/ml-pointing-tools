#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_DIR/scripts/init.sh"

# Run ED volume batch reload training with all three planes
python3 "$REPO_DIR/electron_direction/models/train_ed_volume_batch_reload.py" \
    -j "$REPO_DIR/electron_direction/json/ed_volumes_v02_5k.json" \
    --plane all \
    --reload-epochs 5
