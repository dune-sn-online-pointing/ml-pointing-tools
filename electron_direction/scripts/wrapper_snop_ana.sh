#!/bin/bash
# HTCondor wrapper: run an analysis script that lives in the snop-pipeline repo.
# The repo path is taken from SNOP_REPO_DIR (set in the .sub file) because condor
# copies the executable to the scratch dir.
set -e
REPO="${SNOP_REPO_DIR:-/afs/cern.ch/work/e/evilla/private/dune/refactor-snop-pipeline}"
cd "$REPO"
echo "Job started at: $(date)  host: $(hostname)"
echo "Repo: $REPO"
echo "Args: $*"
source "$REPO/scripts/init.sh"
set +e
python3 -u "$@"
EXIT_CODE=$?
echo "Job finished at: $(date)  exit code: $EXIT_CODE"
exit $EXIT_CODE
