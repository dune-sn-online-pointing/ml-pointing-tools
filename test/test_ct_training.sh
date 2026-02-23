#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load standard environment and paths
source "$REPO_DIR/scripts/init.sh"

export REPO_DIR

python3 "$REPO_DIR/test/make_tiny_fixtures.py"

CT_JSON="$REPO_DIR/test/configs/ct_volume_tiny.json"
CT_BASE_OUT="$REPO_DIR/test/artifacts/test_outputs/channel_tagging"

echo "Running CT tiny training..."
python3 "$REPO_DIR/channel_tagging/models/train_ct_volume_simple.py" \
    --json "$CT_JSON" \
    --plane X \
    --max-samples 10

# Training writes output to: <base_dir>/<version>/<timestamp>/
CT_VERSION="ct_tiny"
CT_VERSION_DIR="$CT_BASE_OUT/$CT_VERSION"
CT_OUT_DIR="$(ls -1dt "$CT_VERSION_DIR"/* 2>/dev/null | head -1 || true)"
if [[ -z "$CT_OUT_DIR" ]]; then
    echo "ERROR: CT output directory not found under: $CT_VERSION_DIR" >&2
    exit 1
fi

python3 - <<PY
import json
from pathlib import Path

out_dir = Path(r"$CT_OUT_DIR")
for required in ["results.json", "test_predictions.npz", "training_history.csv"]:
        p = out_dir / required
        if not p.exists():
                raise SystemExit(f"Missing CT output: {p}")

with (out_dir / "results.json").open("r", encoding="utf-8") as f:
        json.load(f)

print("OK: CT training produced expected outputs")
PY

echo "Running CT analysis app (comprehensive_ct_analysis.py)..."
CT_PDF="$CT_OUT_DIR/comprehensive_ct_analysis.pdf"
python3 "$REPO_DIR/channel_tagging/ana/comprehensive_ct_analysis.py" \
    "$CT_OUT_DIR" \
    -o "$CT_PDF"

if [[ ! -s "$CT_PDF" ]]; then
    echo "ERROR: CT analysis did not produce a PDF: $CT_PDF" >&2
    exit 1
fi

echo "OK: CT tiny train+analysis smoke test passed"
