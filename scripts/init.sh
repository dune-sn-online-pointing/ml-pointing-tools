#!/bin/bash

# Initialization script for the repo
# Sets up environment, paths, and helper functions
# Source this script from other scripts: source $SCRIPTS_DIR/init.sh
# This file is intended to be *sourced* (often into an interactive shell).
# All scripts source it automatically

_MLPT_IS_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    _MLPT_IS_SOURCED=1
fi

# Idempotency: avoid re-initializing the environment multiple times.
# To force a re-init, run: `MLPT_FORCE_INIT=1 source scripts/init.sh`
if [[ -n "${INIT_DONE:-}" && "${INIT_DONE}" == "1" && -z "${MLPT_FORCE_INIT:-}" ]]; then
    return 0
fi

# Get absolute path to the repository root
export SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"

# Python modules directory
export PYTHON_DIR="$REPO_DIR/python"
export CT_LIB_DIR="$REPO_DIR/channel_tagging/lib"
export ED_LIB_DIR="$REPO_DIR/electron_direction/lib"

# Local packages directory (for healpy, etc.)
export LOCAL_PACKAGES_DIR="$REPO_DIR/local_packages"

# JSON configurations directory
export JSON_DIR="$REPO_DIR/json"

# Source LCG environment with CUDA support
LCG_RELEASE="LCG_106_cuda/x86_64-el9-gcc11-opt"
LCG_VIEW="/cvmfs/sft.cern.ch/lcg/views/$LCG_RELEASE"

if [ -f "$LCG_VIEW/setup.sh" ]; then
    # Some external setup scripts are not safe under nounset (-u).
    # Temporarily disable it when sourcing, then restore caller behavior.
    case $- in
        *u*) HAD_NOUNSET=1 ;;
        *) HAD_NOUNSET=0 ;;
    esac
    if [ "$HAD_NOUNSET" -eq 1 ]; then
        set +u
    fi
    source "$LCG_VIEW/setup.sh"
    if [ "$HAD_NOUNSET" -eq 1 ]; then
        set -u
    fi
    echo "✓ Sourced LCG environment: $LCG_RELEASE"
else
    echo "⚠ Warning: LCG environment not found at $LCG_VIEW"
    echo "  Using system Python instead"
fi

# Add task-specific libs and shared Python modules to PYTHONPATH
export PYTHONPATH="$CT_LIB_DIR:$ED_LIB_DIR:$PYTHON_DIR:${PYTHONPATH:-}"

# Add local packages to PYTHONPATH (for healpy, hyperopt if needed)
if [ -d "$LOCAL_PACKAGES_DIR/lib/python3.11/site-packages" ]; then
    export PYTHONPATH="$LOCAL_PACKAGES_DIR/lib/python3.11/site-packages:${PYTHONPATH:-}"
    echo "✓ Added local packages to PYTHONPATH"
fi

# Helper function to print colored messages
print_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[0;32m[SUCCESS]\033[0m $1"
}

print_warning() {
    echo -e "\033[0;33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

_mlpt_die() {
    print_error "$1"
    if [[ "${_MLPT_IS_SOURCED}" -eq 1 ]]; then
        return 1
    fi
    exit 1
}

# Helper function to check if a file exists
check_file() {
    if [ ! -f "$1" ]; then
        _mlpt_die "File not found: $1"
    fi
}

# Helper function to check if a directory exists
check_dir() {
    if [ ! -d "$1" ]; then
        _mlpt_die "Directory not found: $1"
    fi
}

# Helper function to create output directory if it doesn't exist
ensure_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        print_info "Created directory: $1"
    fi
}

# Print environment information
print_info "ML for Pointing Environment"
print_info "Repository: $REPO_DIR"
print_info "CT libs: $CT_LIB_DIR"
print_info "ED libs: $ED_LIB_DIR"
print_info "Python modules: $PYTHON_DIR"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_info "Python version: $PYTHON_VERSION"

# Check if TensorFlow is available
if python3 -c "import tensorflow" 2>/dev/null; then
    TF_VERSION=$(python3 -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null)
    print_success "TensorFlow $TF_VERSION is available"
else
    print_warning "TensorFlow is not available"
fi

export INIT_DONE=1
