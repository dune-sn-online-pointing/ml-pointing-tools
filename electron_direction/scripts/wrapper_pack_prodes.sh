#!/bin/bash
# HTCondor wrapper: pack the matchfix ES production pool into npy shards.
set -e
PROJECT_DIR="${MLPT_REPO_DIR:-/afs/cern.ch/work/e/evilla/private/dune/refactor-ml-for-pointing}"
cd "$PROJECT_DIR"
echo "Job started at: $(date)  host: $(hostname)"
source "$PROJECT_DIR/scripts/init.sh"
set +e
python3 electron_direction/scripts/pack_prodes_pool.py "$@"
EXIT_CODE=$?
echo "Job finished at: $(date)  exit code: $EXIT_CODE"
exit $EXIT_CODE
