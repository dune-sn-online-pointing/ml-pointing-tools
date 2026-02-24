#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$REPO_DIR/test/test_ct_training.sh"
"$REPO_DIR/test/test_ed_training.sh"

echo "OK: all app smoke tests passed"
