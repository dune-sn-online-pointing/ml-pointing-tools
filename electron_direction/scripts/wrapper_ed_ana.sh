#!/bin/bash
# HTCondor wrapper for ED analysis scripts in electron_direction/ana/ and scripts/.
# NOTE: condor copies the executable to the scratch dir, so the repo path cannot be
# derived from BASH_SOURCE; it is taken from MLPT_REPO_DIR (set in the .sub file).
set -e
PROJECT_DIR="${MLPT_REPO_DIR:-/afs/cern.ch/work/e/evilla/private/dune/refactor-ml-for-pointing}"
SCRIPT="$1"; shift
cd "$PROJECT_DIR"
echo "Job started at: $(date)  host: $(hostname)"
echo "Repo: $PROJECT_DIR"
echo "Script: $SCRIPT  args: $*"
source "$PROJECT_DIR/scripts/init.sh"
export CUDA_VISIBLE_DEVICES=""
set +e
python3 "$SCRIPT" "$@"
EXIT_CODE=$?
echo "Job finished at: $(date)  exit code: $EXIT_CODE"
exit $EXIT_CODE
