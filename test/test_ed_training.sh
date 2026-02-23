#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load standard environment and paths
source "$REPO_DIR/scripts/init.sh"

export REPO_DIR

python3 "$REPO_DIR/test/make_tiny_fixtures.py"

ED_JSON="$REPO_DIR/test/configs/ed_three_plane_tiny.json"
ED_BASE_OUT="$REPO_DIR/test/artifacts/test_outputs/electron_direction"

echo "Running ED tiny training..."
python3 "$REPO_DIR/electron_direction/models/train_three_plane_simple.py" \
    --json "$ED_JSON"

# Training writes output to: <base_dir>/three_plane_<name>_<timestamp>/
ED_OUT_DIR="$(ls -1dt "$ED_BASE_OUT"/three_plane_tiny_* 2>/dev/null | head -1 || true)"
if [[ -z "$ED_OUT_DIR" ]]; then
    echo "ERROR: ED output directory not found under: $ED_BASE_OUT" >&2
    exit 1
fi

python3 - <<PY
import json
from pathlib import Path

out_dir = Path(r"$ED_OUT_DIR")
for required in ["results.json", "val_predictions.npz"]:
        p = out_dir / required
        if not p.exists():
                raise SystemExit(f"Missing ED output: {p}")

with (out_dir / "results.json").open("r", encoding="utf-8") as f:
        json.load(f)

print("OK: ED training produced expected outputs")
PY

echo "Running ED analysis app (comprehensive_ed_analysis.py)..."
ED_PDF="$ED_OUT_DIR/comprehensive_ed_analysis.pdf"
python3 "$REPO_DIR/electron_direction/ana/comprehensive_ed_analysis.py" \
    "$ED_OUT_DIR" \
    -o "$ED_PDF"

if [[ ! -s "$ED_PDF" ]]; then
    echo "ERROR: ED analysis did not produce a PDF: $ED_PDF" >&2
    exit 1
fi

echo "Running ED ana app (cnn_feature_interpretation.py)..."
ED_INT_PDF="$ED_OUT_DIR/cnn_interpretation_report.pdf"
python3 "$REPO_DIR/electron_direction/ana/cnn_feature_interpretation.py" \
    "$ED_OUT_DIR" \
    -o "$ED_INT_PDF"

if [[ ! -s "$ED_INT_PDF" ]]; then
    echo "ERROR: ED feature interpretation did not produce a PDF: $ED_INT_PDF" >&2
    exit 1
fi

echo "OK: ED tiny train+analysis smoke test passed"
