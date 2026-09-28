#!/bin/bash
# HTCondor wrapper for ED v62/v63 training on the matchfix ES production pool.
# Mirrors wrapper_ed_v61_burstcats.sh (derives the repo dir from the JSON path).
set -e

JSON_CONFIG=""
TRAIN_SCRIPT="electron_direction/models/train_three_plane_prodes_v62.py"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--json) JSON_CONFIG="$2"; shift 2 ;;
        -s|--script) TRAIN_SCRIPT="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$JSON_CONFIG" ]]; then
    echo "Error: JSON config required (-j/--json)"; exit 1
fi

echo "========================================="
echo "ED v62/v63 PRODUCTION-POOL TRAINING - HTCondor Job"
echo "Job started at: $(date)   host: $(hostname)"
echo "JSON config: $JSON_CONFIG"
echo "========================================="

if [[ -f "$JSON_CONFIG" ]]; then
    PROJECT_DIR="$(cd "$(dirname "$JSON_CONFIG")/../.." && pwd)"
elif [[ -n "${_CONDOR_JOB_IWD:-}" ]] && [[ -f "${_CONDOR_JOB_IWD}/scripts/init.sh" ]]; then
    PROJECT_DIR="$_CONDOR_JOB_IWD"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$PROJECT_DIR"

echo "Setting up environment using init.sh..."
source "$PROJECT_DIR/scripts/init.sh"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 || echo "No GPU available"
echo ""
echo "Working directory: $(pwd)"
echo "Trainer: $TRAIN_SCRIPT ${EXTRA_ARGS[*]}"
set +e
python3 "$TRAIN_SCRIPT" -j "$JSON_CONFIG" "${EXTRA_ARGS[@]}"
EXIT_CODE=$?
echo ""
echo "Job finished at: $(date)  exit code: $EXIT_CODE"
exit $EXIT_CODE
