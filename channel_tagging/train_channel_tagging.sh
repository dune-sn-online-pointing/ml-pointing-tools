#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_DIR/scripts/init.sh"

# Parse arguments
json_file=""
plane=""
max_samples=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--json) json_file="$2"; shift 2;;
    --plane) plane="$2"; shift 2;;
    --max-samples) max_samples="$2"; shift 2;;
    *) shift;;
  esac
done

if [[ -z "$json_file" ]]; then
  echo "Error: JSON config file is required (-j|--json)"
  exit 1
fi

check_file "$json_file"

# Build command
CMD=("python3" "$REPO_DIR/channel_tagging/models/ct_training.py" "--input_json" "$json_file")
[[ -n "$plane" ]] && CMD+=("--plane" "$plane")
[[ -n "$max_samples" ]] && CMD+=("--max_samples" "$max_samples")

echo "Running: ${CMD[*]}"
"${CMD[@]}"
