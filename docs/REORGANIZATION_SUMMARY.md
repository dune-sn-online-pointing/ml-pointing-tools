# Repository Reorganization Summary

**Date**: November 10, 2025
**Repository**: `/afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools`

## Overview

Cleaned and reorganized the ML repository for better structure, clarity, and maintainability.

---

## Changes Made

### 1. File Renaming
- **Electron Direction**: `models/main.py` → `models/train_ed_hyperopt.py`
- **Channel Tagging**: `models/main.py` → `models/train_ct_hyperopt.py`

**Rationale**: Descriptive names make it immediately clear what each script does.

---

### 2. Reorganized Submission Files (.sub)

**Moved FROM `scripts/`**:
- All `submit_*_ed_*.sub`, `submit_single_*.sub`, `submit_three_*.sub`, `submit_bootstrap_*.sub` → `electron_direction/condor/`
- All `submit_ct_*.sub`, `submit_condor_channel_tagging*.sub` → `channel_tagging/condor/`

**Result**: Each task now has its own `condor/` folder with all HTCondor submission files.

---

### 3. Reorganized Scripts

**Moved FROM `scripts/` TO task folders**:
- `run_single_*.sh`, `run_three_*.sh`, `run_bootstrap_*.sh`, `run_electron_direction.sh` → `electron_direction/scripts/`
- `run_channel_tagging.sh` → `channel_tagging/scripts/`

**Moved TO `python/` (common tools)**:
- `create_training_plots.py`
- `regenerate_plots.py`

**Moved TO `temp/` (to be reorganized later)**:
- `condor_wrapper*.sh`
- `submit_*.sh`

**Kept in `scripts/` (general only)**:
- `init.sh`
- `manage-submodules.sh`
- `monitor_jobs.sh`
- `monitor_jobs_auto.sh`

**Result**: `scripts/` now contains only general-purpose scripts. Task-specific scripts are in their respective folders.

---

### 4. Documentation Reorganization

**Active Documentation** (kept in `docs/`):
- `Networks.md` - **NEW**: Comprehensive training registry with all ED/CT experiments

**Moved TO `temp/`** (outdated/refactoring-specific):
- `BUG_FIX_DIRECTION_COLUMNS.txt`
- `JOBS_SUMMARY.md`
- `REFACTORING_CHECKLIST.md`
- `REFACTORING_SUMMARY.md`
- `RESUBMISSION_READY.md`
- `TRAINING_SUBMISSIONS_2025-11-04.md`
- `README_REFACTORED.md` (consolidated into main README)
- `PRODUCTION_TRAINING_GUIDE.md`
- `TRAINING_NOTES.md`

**Result**: `docs/` now has only 2 active files - clean and focused.

---

### 5. Networks.md - NEW Training Registry

Created comprehensive training registry documenting:

**Electron Direction (ED)**:
- 5 training attempts tracked
- Bootstrap ensemble (v1): 74.23° - FAILED
- Single-plane X with angular/focal/hybrid losses (v5): RUNNING
- Three-plane attention (v5): OOM failure even with 24GB
- Data quality investigations
- Custom loss functions documented
- Key insights and next steps

**Channel Tagging (CT)**:
- 3 training attempts tracked
- Volume images (simple, streaming, balanced) - all IDLE
- Data migration from cluster to volume images
- Scaling plan documented

**Best Practices section** added for:
- Pre-submission checklist
- Resource estimation guidelines
- Loss function selection
- Post-completion documentation

---

### 6. Outputs Migration to EOS

**Action**: Moved all outputs from local `outputs/` to permanent storage

```bash
rsync -av outputs/ /eos/user/e/evilla/dune/sn-tps/neural_networks/
rm -rf outputs/
```

**Migrated**:
- `outputs/channel_tagging/` → EOS (~43K)
- `outputs/electron_direction/` → EOS (~6.1M)

**Result**: No more `outputs/` directory in repo. All results now in permanent EOS storage.

---

### 7. Python Common Tools Cleanup

**Removed** (moved to `temp/`):
- `data_loader.py.backup`
- `data_loader.py.backup_3plane`
- `regression_libs.py.backup`
- `regression_libs.py.backup_3plane`

**Kept** (actively used):
- `classification_libs.py`
- `regression_libs.py`
- `data_loader.py`
- `matched_three_plane_data_loader.py`
- `three_plane_data_loader.py`
- `streaming_data_loader.py`
- `volume_directory_scanner.py`
- `general_purpose_libs.py`
- `merge_npz_groups.py`
- `prepare_three_plane_matched_data.py`
- `create_training_plots.py`
- `regenerate_plots.py`

**Result**: Clean `python/` directory with only actively used common tools.

---

## Final Structure

```
ml-pointing-tools/
├── channel_tagging/
│   ├── ana/             # Analysis scripts
│   ├── condor/          # HTCondor submission files
│   ├── json/            # Configuration files
│   ├── logs/            # Job logs
│   ├── models/          # Training scripts
│   ├── scripts/run_channel_tagging.sh
│   └── scripts/train_channel_tagging.sh
│
├── electron_direction/
│   ├── ana/             # Analysis scripts
│   ├── condor/          # HTCondor submission files
│   ├── json/            # Configuration files
│   ├── logs/            # Job logs
│   ├── models/          # Training scripts (train_ed_hyperopt.py, etc.)
│   ├── scripts/         # Task-specific scripts
│   └── ...
│
├── docs/
│   └── Networks.md      # Training registry (NEW)
│
├── python/              # Common shared libraries
│   ├── classification_libs.py
│   ├── regression_libs.py
│   ├── data_loader.py
│   ├── *_data_loader.py
│   └── *.py
│
├── scripts/             # General-purpose scripts only
│   ├── init.sh
│   ├── manage-submodules.sh
│   ├── monitor_jobs.sh
│   └── monitor_jobs_auto.sh
│
├── temp/                # Temporary/outdated files
│   ├── condor_wrapper*.sh (to be reorganized)
│   ├── outdated docs
│   └── backup files
│
├── local_packages/      # Local Python packages
├── README.md            # Main repository documentation
├── .gitignore
└── .gitmodules
```

---

## Key Improvements

1. **Clear Separation**: Task-specific files in task folders, general files in `scripts/`
2. **No More main.py**: All scripts have descriptive names
3. **Centralized Documentation**: `Networks.md` tracks all training attempts
4. **Permanent Storage**: Outputs on EOS, not in repo
5. **Clean Python**: No backup files, only active tools
6. **Organized Docs**: Only 2 active docs files, outdated moved to temp
7. **Proper .sub Location**: All submission files in respective `condor/` folders

---

## Impact on Existing Jobs

**NONE** - This is a new clone of the repository. Running jobs in the old repo (`ml_for_pointing`) are completely unaffected.

---

## Next Steps for Future Work

1. **Condor Wrappers**: Review `temp/condor_wrapper*.sh` files and place in appropriate task folders
2. **Update Paths**: When using this repo, update paths in configs to point to new structure
3. **Maintain Networks.md**: Always register new trainings BEFORE submission
4. **Clean temp/**: Periodically review and delete truly obsolete files

---

## Benefits

- ✅ Easier to find files (clear naming, logical structure)
- ✅ Each task self-contained (ana, models, condor, json, logs)
- ✅ Training history tracked (Networks.md)
- ✅ No redundant files (backups removed)
- ✅ Scalable structure (easy to add new tasks)
- ✅ Clear separation of concerns (task vs general)

