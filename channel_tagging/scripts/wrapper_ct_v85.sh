#!/bin/bash
# HTCondor wrapper for CT v85: train on radmask volumes, then run the 2x2
# {v83, v85} x {unmasked, masked} cross-evaluation in the SAME job (one GPU job).

set -e

JSON_CONFIG=""
TRAIN_SCRIPT="channel_tagging/models/train_ct_volume_v85.py"
V83_DIR="/eos/project-e/ep-nu/evilla/sn-online-pointing/neural-networks/channel_tagging/ct_volume_v83_20260906_194825"

while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--json) JSON_CONFIG="$2"; shift 2 ;;
        -s|--script) TRAIN_SCRIPT="$2"; shift 2 ;;
        --v83) V83_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$JSON_CONFIG" ]]; then
    echo "Error: JSON config required (-j/--json)"; exit 1
fi

echo "========================================="
echo "CHANNEL TAGGING v85 (radmask) - HTCondor Job"
echo "========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "JSON config: $JSON_CONFIG"
echo "v83 model dir: $V83_DIR"
echo "========================================="

PROJECT_DIR="$(cd "$(dirname "$JSON_CONFIG")/../.." && pwd)"
cd "$PROJECT_DIR"

echo "Setting up environment using init.sh..."
source "$PROJECT_DIR/scripts/init.sh"
echo ""
nvidia-smi 2>/dev/null || echo "No GPU available (CPU-only)"
echo ""

TRAIN_LOG="$(mktemp)"
echo "Starting training: $TRAIN_SCRIPT"
python3 "$TRAIN_SCRIPT" -j "$JSON_CONFIG" 2>&1 | tee "$TRAIN_LOG"
TRAIN_RC="${PIPESTATUS[0]}"
if [[ "$TRAIN_RC" -ne 0 ]]; then
    echo "Training failed with exit code $TRAIN_RC"; exit "$TRAIN_RC"
fi

V85_DIR="$(grep -m1 '^Output directory: ' "$TRAIN_LOG" | sed 's/^Output directory: //')"
rm -f "$TRAIN_LOG"
echo ""
echo "========================================="
echo "v85 model dir: $V85_DIR"
echo "Starting 2x2 cross-evaluation..."
echo "========================================="
python3 channel_tagging/ana/ct_v85_cross_eval.py \
        --v83 "$V83_DIR" --v85 "$V85_DIR" \
        --out "$V85_DIR/cross_eval_2x2.npz"
EVAL_RC=$?

echo ""
echo "========================================="
echo "Job finished at: $(date)"
echo "Train exit: $TRAIN_RC   Cross-eval exit: $EVAL_RC"
echo "========================================="
exit $EVAL_RC
