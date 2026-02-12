# ED Volume Training v01 - Setup Documentation

## Overview
Training electron direction (ED) model using full 1m×1m volume clusters instead of pentagon crops.

**Objective**: Determine if larger spatial context improves direction prediction compared to pentagon-based models (best: 35.3° median error in v58).

## Data Source
- **Location**: `/eos/user/e/evilla/dune/sn-tps/prod_es/es_production_volume_images_tick3_ch2_min2_tot3_e2p0/`
- **Structure**: Three subdirectories (U/, V/, X/) with matched NPZ files
- **Dataset**: 3989 files per plane, ~46 volumes per file (total ~183k volumes)
- **Volume dimensions**: (208, 1242) pixels representing ~1m × 1m physical space
  - 208 channels × 0.479 cm = 99.6 cm
  - 1242 time bins × 0.0805 cm = 100.0 cm

## Architecture Changes from Pentagon Model

### Pentagon Model (train_three_plane_simple.py)
- Input: (128, 16, 1) per plane
- Conv layers: 2-3 blocks
- Batch size: 32
- Filters: 32, 64, 128
- Parameters: ~500k

### Volume Model (train_volumes.py)
- Input: (208, 1242, 1) per plane
- Conv layers: 5 blocks with progressive downsampling
- Batch size: 8 (reduced for memory)
- Filters: 32 → 64 → 128 → 256 → 512
- Parameters: 5.2M (10x larger)
- Downsampling path:
  - (208, 1242) → (104, 621) → (52, 310) → (26, 155) → (13, 77) → (6, 38)
  - GlobalAveragePooling2D: (6, 38, 512) → 512 features per plane
  - Concatenate 3 planes: 1536 features
  - Dense: 256 → 256 → 3 (normalized)

## Implementation Files

### Data Loader
**File**: `/afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/python/volume_ed_data_loader.py`

**Key Features**:
- Loads three-plane matched volumes from U/V/X subdirectories
- Matches by (event, main_cluster_match_id) tuple
- Extracts particle momentum (x, y, z) from metadata
- Normalizes momentum to unit direction vector
- Filters samples with zero momentum
- Per-volume max normalization
- Adds channel dimension for CNN input
- Returns: (images_u, images_v, images_x, directions, energies, metadata)

### Training Script
**File**: `/afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/electron_direction/models/train_volumes.py`

**Architecture Function**: `build_three_plane_volume_cnn(input_shape=(208, 1242, 1))`
- Three parallel CNN branches (U, V, X)
- Each branch: 5 × [Conv2D → MaxPool2D → BatchNorm]
- GlobalAveragePooling2D per branch
- Concatenation → 2 dense layers (256 units) → Normalized 3D output
- Loss: cosine_similarity_loss
- Optimizer: Adam with configurable LR
- Callbacks: EarlyStopping (patience=15), ReduceLROnPlateau, ModelCheckpoint

**Outputs**:
- `best_model.keras` - Best validation model
- `final_model.keras` - Final epoch model
- `val_predictions.npz` - Predictions with energies for stratification
- `results.json` - Metrics (median, mean, 68%, 95% angular errors)

## Configurations

### Test Config (Local Validation)
**File**: `electron_direction/json/ed_volumes_local_test.json`
- Data: max_files=2, max_samples=200
- Training: epochs=3, batch_size=4
- Output: `test_output/volumes_local/`
- **Status**: ✅ PASSED - Verified code works correctly

### Production Config
**File**: `electron_direction/json/ed_volumes_v01.json`
- Data: All files (3989), all volumes (~183k samples)
- Training: epochs=100, batch_size=8, lr=0.001
- Output: `output/volumes_v01/`
- **Status**: 🚀 SUBMITTED - Condor job 13789427

## Condor Submission

### Submit File
**File**: `electron_direction/condor/submit_ed_volumes_v01.sub`
- GPU: request_gpus=1, CUDACapability >= 3.5
- Memory: 16GB (large volumes need more memory)
- CPUs: 4
- Disk: 5GB
- Runtime: 8 hours max (28800s)

### Run Script
**File**: `electron_direction/condor/run_ed_volumes_v01.sh`
- Environment: LCG_104 (TensorFlow, Keras)
- Working dir: `electron_direction/models/`
- Command: `python3 train_volumes.py -j ../json/ed_volumes_v01.json`
- Logs: `condor/logs/ed_volumes_v01_<ClusterId>.<ProcId>.{out,err,log}`

### Job Status
- **Job ID**: 13789427
- **Submitted**: 2025-11-19 07:18 UTC
- **Status**: IDLE (waiting for GPU)
- **Monitor**: `condor_q 13789427` or check logs in `condor/logs/`

## Expected Results

### Performance Expectations
- **Training time**: Several hours (183k volumes, 100 epochs, early stopping)
- **Convergence**: Loss should decrease steadily, watch for plateau
- **Baseline comparison**: Pentagon v58 achieved 35.3° median error on ES data
- **Success criteria**: 
  - Volume model converges (loss < 0.5)
  - Validation metrics computed successfully
  - Compare angular errors with pentagon baseline

### Physics Insights
- **Larger context**: Volumes provide surrounding clusters, energy deposition patterns
- **Trade-offs**: 
  - ✅ More spatial context, multiple clusters visible
  - ⚠️ Larger memory footprint, slower training
  - ⚠️ May include irrelevant noise from distant deposits

### Output Analysis
After job completes, check:
1. `output/volumes_v01/results.json` - Angular error metrics
2. `output/volumes_v01/val_predictions.npz` - Predictions by energy
3. Training curves in logs
4. Compare with `docs/BestModels.dat` pentagon performance

## Next Steps (After Training)
1. Monitor job completion (check logs in `condor/logs/`)
2. Analyze results.json metrics
3. Energy-stratified performance analysis (low vs high energy)
4. Compare with pentagon baseline (v58: 35.3°)
5. If improved: Update BestModels.dat
6. If not improved: Investigate why (overfitting? noise? architecture?)
7. Consider intermediate volumes (e.g., 0.5m × 0.5m) if full volumes underperform

## Files Summary
```
ml-pointing-tools/
├── python/
│   └── volume_ed_data_loader.py          # Custom ED volume data loader
├── electron_direction/
│   ├── models/
│   │   └── train_volumes.py              # Training script with 5-layer CNN
│   ├── json/
│   │   ├── ed_volumes_local_test.json    # Test config (✅ passed)
│   │   └── ed_volumes_v01.json           # Production config (🚀 running)
│   ├── condor/
│   │   ├── submit_ed_volumes_v01.sub     # Condor submit file
│   │   ├── run_ed_volumes_v01.sh         # Run script
│   │   └── logs/                         # Job logs (out, err, log)
│   ├── test_output/
│   │   └── volumes_local/                # Local test results
│   └── output/
│       └── volumes_v01/                  # Production output (pending)
```

## Contact / Issues
- Data location: `/eos/user/e/evilla/dune/sn-tps/prod_es/es_production_volume_images_tick3_ch2_min2_tot3_e2p0/`
- Job logs: `/afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/electron_direction/condor/logs/`
- Check job: `condor_q 13789427` or `condor_history 13789427`
