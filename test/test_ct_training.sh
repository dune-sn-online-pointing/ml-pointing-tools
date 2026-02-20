#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load standard environment and paths
source "$REPO_DIR/scripts/init.sh"

export REPO_DIR

python3 - <<'PY'
import json
import os
from pathlib import Path

repo = Path(os.environ["REPO_DIR"])
config_path = repo / "channel_tagging" / "json" / "volume_v42_corrected_100k.json"
trainer_path = repo / "channel_tagging" / "models" / "ct_training.py"

missing = [p for p in [config_path, trainer_path] if not p.exists()]
if missing:
    print("Missing expected file(s):")
    for p in missing:
        print(f"- {p}")
    raise SystemExit(1)

with config_path.open("r", encoding="utf-8") as f:
    json.load(f)

print("OK: CT config loads and trainer exists")
PY
