# Framework Description

This document describes the code structure, data formats, and the main training/analysis workflows for **ml-pointing-tools**.

## High-level structure

- `channel_tagging/`: Channel Tagging (CT) training + analysis + configs + condor
- `electron_direction/`: Electron Direction (ED) training + analysis + configs + condor
- `python/`: shared utilities (data loading, metrics, plotting helpers)
- `channel_tagging/lib/`, `electron_direction/lib/`: task-specific Python modules
- `scripts/init.sh`: environment bootstrap (LCG + `PYTHONPATH`)
- `docs/`: human-readable documentation (this folder)
- `test/`: end-to-end smoke tests (tiny local fixtures + run training + run ana apps)

## Environment bootstrap

Most workflows start with:

```bash
source scripts/init.sh
```

This sets up the external environment (LCG view when available) and makes the repo’s Python modules importable.

## Data formats

The project largely uses `.npz` files (NumPy archives). There are two common patterns:

1. **Batch/cluster datasets**: contain `images` and `metadata` arrays.
2. **Volume-image datasets**: contain `images` arrays (metadata may be absent depending on producer).

### Common keys

- `images`: image arrays used as network inputs.
- `metadata`: per-sample float arrays (column-based schema).

### Metadata schema (shared)

The shared loader in `python/data_loader.py` supports metadata vectors of length 11/12/13/14.
The most important fields used across CT/ED are:

- `is_main_track` (binary): used for CT main-track identification
- `true_particle_mom` (`px,py,pz`): used for ED direction labels (normalized)
- `true_particle_energy` (MeV): used by ED analysis and likelihood construction
- `plane_id`: optional plane identifier (`U/V/X`)
- `match_id` (only when length=14): used to match U/V/X clusters across planes

### CT: cluster batch files

Typical filename pattern:

`clusters_plane{plane}_batch*.npz`

Expected content:

- `images`: shape `(N, H, W)` (then expanded to `(N, H, W, 1)` for CNNs)
- `metadata`: shape `(N, 11/12/13/14)`

These are consumed by CT helpers such as `python/classification_libs.py` through `python/data_loader.py`.

### CT: volume-image files

Some CT trainers use a simpler “volume image” format, with filenames matching:

`*plane{plane}.npz`

Expected content:

- `images`: an array of images, each of shape `(208, 1242)`.

This is the format used by `channel_tagging/models/train_ct_volume_simple.py`.

### ED: 3-plane matched files

ED matched training uses “triplets” of files with a shared basename:

- `<prefix>_planeU.npz`
- `<prefix>_planeV.npz`
- `<prefix>_planeX.npz`

They can live in a flat folder or in subfolders `U/`, `V/`, `X/`.

Expected content in each file:

- `images`: shape `(N, 128, 16)` (or `(N, 128, 32)` depending on production)
- `metadata`: shape `(N, 14)` where:
  - column `0` is `event_id`
  - columns `7:10` are momentum `(px,py,pz)` used to build direction labels
  - column `10` is `true_particle_energy` (MeV)
  - column `13` is `match_id`

Matching logic:

- the loader matches `(event_id, match_id)` across U/V/X
- samples with `match_id == -1` are treated as “unmatched” and are skipped

The canonical implementation is in `python/data_loader.load_three_plane_matched`.

## Training outputs (conventions)

Most trainers write an output directory containing some subset of:

- `results.json`: training metrics + config snapshot
- `test_predictions.npz` (CT) or `val_predictions.npz` (ED): prediction arrays and truth labels
- `training_history.csv` / `training_history.json`: training curves
- model checkpoints (e.g. `best_model.keras`, `checkpoints/`)

## Analysis apps (“ana”)

The primary “analysis apps” are:

- CT: `channel_tagging/ana/comprehensive_ct_analysis.py`
  - consumes a CT output directory containing `results.json` + `test_predictions.npz`
  - produces a multi-page PDF report
- ED: `electron_direction/ana/comprehensive_ed_analysis.py`
  - consumes an ED output directory containing `results.json` + `val_predictions.npz`
  - produces a multi-page PDF report
  - also writes `cosine_energy_pdf.npz` and a visualization PNG for likelihood work
- ED: `electron_direction/ana/cnn_feature_interpretation.py`
  - consumes the same ED outputs and produces a diagnostic/interpretation PDF

## HTCondor

Each task has:

- `condor/`: `.sub` submit files
- `scripts/`: wrapper scripts referenced by the submit files

Policy:

- Keep only maintained `.sub` templates under `channel_tagging/condor/` and `electron_direction/condor/`.
- Generate any temporary/derived submit files outside version control.
- Keep runtime condor logs/errors/outputs untracked (except `.gitkeep`).

## Tests

The `test/` folder provides end-to-end smoke tests that:

1. generate tiny local `.npz` fixtures
2. run a tiny CT and ED training
3. run the main ana apps against those outputs

Run:

```bash
./test/testAllApps.sh
```
