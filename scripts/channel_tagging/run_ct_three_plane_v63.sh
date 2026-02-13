#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_DIR/scripts/init.sh"

python3 "$REPO_DIR/channel_tagging/models/train_ct_three_plane_batch_reload.py" \
	--json "$REPO_DIR/channel_tagging/json/volume_v63_three_plane_batch_reload.json"
