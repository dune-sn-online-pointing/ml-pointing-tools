# Neural Networks Training Registry

This file tracks successful training runs for each task. Only completed trainings with valid results are documented here.

---

## Electron Direction (ED) Regression

**Goal**: Predict the 3D direction vector of electron tracks from charge deposition images.

**Dataset**: Three-plane matched cluster images from `/eos/home-e/evilla/dune/sn-tps/production_es/cluster_images_nov11/`
- Format: 128×32 pixels per plane (U, V, X)
- Matched across planes via metadata field 13

**Evaluation Metric**: 
- Mean angular error (degrees) 
- **68% Containment**: Angle at which 68% of predictions are better (1-sigma equivalent for angular resolution)

**Current Best**: v19_lr_schedule with 51.18° @ 68% quantile (-0.21° improvement over v18 baseline)

### Successful Training Runs (November 2025)

#### v7: Three-Plane Cosine Loss (10k) - ✅ BEST
- **Job ID**: Multiple iterations
- **Config**: `electron_direction/json/three_plane_v7_10k.json`
- **Dataset**: 10k samples from nov11 (128×32 per plane)
- **Architecture**: 3 conv layers, 32 filters, 2 dense layers (256 units)
- **Loss**: Cosine similarity
- **Results**: 
  - **68% containment: 35.5°** ⭐ BEST
  - Mean angular error: ~45°
  - Flipped predictions: 3.3%
- **Status**: ✅ Completed - Current best model

#### v8: Three-Plane Angular Loss (10k) - ✅ VERY GOOD
- **Config**: `electron_direction/json/three_plane_v8_10k.json`
- **Dataset**: 10k samples from nov11 (128×32 per plane)
- **Architecture**: 3 conv layers, 32 filters, 2 dense layers (256 units)
- **Loss**: Angular loss (direct arccos optimization)
- **Results**:
  - **68% containment: 36.7°**
  - Mean angular error: ~46°
  - Flipped predictions: 3.0%
- **Status**: ✅ Completed - Very close to v7

#### v9: Three-Plane Focal Angular Loss (10k) - ❌ FAILED
- **Config**: `electron_direction/json/three_plane_v9_10k.json`
- **Loss**: Focal angular loss (hard example mining)
- **Results**:
  - 68% containment: 78.3° (WORSE than baseline)
  - Flipped predictions: 6.3%
- **Status**: ✅ Completed but poor performance

#### v10: Enhanced Architecture 50k - ✅ COMPLETED
- **Job ID**: 10630293
- **Config**: `electron_direction/json/three_plane_v10_50k.json`
- **Output**: `three_plane_v10_50k_20251111_133912/`
- **Dataset**: 50k samples from nov11 (128×32 per plane)
- **Architecture**: **4 conv layers, 64 filters**, 2 dense layers (256 units)
- **Loss**: Angular loss
- **Training**: 200 epochs, no hyperopt
- **Status**: ✅ Completed - Analysis pending

#### v11: Enhanced Architecture 50k + Hyperopt - ✅ COMPLETED
- **Job ID**: 10630294
- **Config**: `electron_direction/json/three_plane_v11_50k_hyperopt.json`
- **Output**: `three_plane_v11_50k_hyperopt_20251111_133831/`
- **Dataset**: 50k samples from nov11
- **Architecture**: 4 conv layers, 64 filters + hyperparameter optimization
- **Loss**: Angular loss
- **Training**: Hyperopt enabled
- **Status**: ✅ Completed - Analysis pending

#### v12: Enhanced Architecture 100k - ✅ COMPLETED
- **Job ID**: 10630295
- **Config**: `electron_direction/json/three_plane_v12_100k.json`
- **Output**: `three_plane_v12_100k_20251111_134350/`
- **Dataset**: 100k samples from nov11
- **Architecture**: 4 conv layers, 64 filters, 2 dense layers (256 units)
- **Loss**: Angular loss
- **Training**: 200 epochs, no hyperopt
- **Status**: ✅ Completed - Analysis pending

#### v13: Enhanced Architecture 100k + Hyperopt - ✅ COMPLETED
- **Job ID**: 10630296
- **Config**: `electron_direction/json/three_plane_v13_100k_hyperopt.json`
- **Output**: `three_plane_v13_100k_hyperopt_20251111_134351/`
- **Dataset**: 100k samples from nov11
- **Architecture**: 4 conv layers, 64 filters + hyperparameter optimization
- **Loss**: Angular loss
- **Training**: Hyperopt enabled
- **Status**: ✅ Completed - Analysis pending

#### v14: Enhanced Architecture 10k Hyperopt - ✅ COMPLETED
- **Job ID**: 10282157
- **Config**: `electron_direction/json/three_plane_v14_10k_hyperopt.json`
- **Output**: `three_plane_v14_10k_hyperopt_20251111_175141/`
- **Dataset**: 10k samples from nov11 (128×32 per plane)
- **Architecture**: 4 conv layers, 64 filters + hyperparameter optimization
- **Loss**: Angular loss
- **Training**: 100 epochs with hyperopt, early stopping patience=20
- **Runtime**: 47 minutes
- **Results**:
  - Mean angular error: **65.68°**
  - Median angular error: **55.01°**
  - 25th percentile: 28.43°
  - 75th percentile: 97.90°
  - Best val loss: 1.1463 (epoch 38)
- **Status**: ✅ Completed - Good baseline for hyperopt testing

#### v15: Production 200k - ✅ COMPLETED ⭐ BEST SO FAR
- **Job ID**: 10282540
- **Config**: `electron_direction/json/three_plane_v15_200k.json`
- **Output**: `three_plane_unknown_20251111_193524/` ⚠️ (misnamed - should be v15_200k)
- **Dataset**: 200k samples from nov11 (128×32 per plane)
- **Architecture**: 4 conv layers, 64 filters (no hyperopt)
- **Loss**: Angular loss
- **Training**: 100 epochs, early stopping triggered at epoch 58
- **Runtime**: 3.1 hours (11178 seconds)
- **Results**:
  - Mean angular error: **47.32°** (28% better than v14)
  - Median angular error: **30.44°** (45% better than v14)
  - 25th percentile: 15.48°
  - 75th percentile: 66.52°
  - Best val loss: 0.8258 (epoch 38) - 28% better than v14
- **Status**: ✅ Completed - **Significantly outperforms v14, scales well with data**
- **Note**: Directory naming bug - saved as "unknown" instead of "v15_200k" (investigating)

#### v18: Production 200k with Data Augmentation - ✅ COMPLETED (BASELINE)
- **Job ID**: 13737405 (resubmitted with optimized resources)
- **Config**: `electron_direction/json/three_plane_v18_200k_aug.json`
- **Output**: `three_plane_three_plane_v18_200k_aug_20251112_114654/`
- **Dataset**: 200k samples from nov11 (128×32 per plane)
- **Architecture**: 4 conv layers, 64 filters, batch norm
- **Loss**: Angular loss
- **Training**: 35 epochs (early stopping), **data augmentation enabled**
  - Flip X/Y: 50% probability each
  - Rotation: ±15°
  - Zoom: ±10%
- **Resources**: Optimized (1 CPU + 16GB RAM + any GPU) - **61 matching machines** (vs 0 before)
- **Results**:
  - Mean angular error: **47.16°**
  - Median angular error: **30.61°**
  - **68% quantile: 51.39°** 📊 BASELINE
  - 25th percentile: 15.66°
  - 75th percentile: 65.89°
  - Best val loss: 0.8232 (epoch 25)
- **Status**: ✅ Completed - Baseline reference for v19 comparisons
- **Note**: Job was idle before resource optimization (removed specific GPU requirements)

#### v19_lr_schedule: Production 200k LR Schedule - ✅ COMPLETED ⭐ BEST SO FAR
- **Job ID**: 12833782
- **Config**: `electron_direction/json/three_plane_v19_lr_schedule.json`
- **Dataset**: 200k samples from new data paths (prod_es)
- **Architecture**: 4 conv layers, 64 filters, batch norm
- **Loss**: Angular loss
- **Training**: Cosine annealing learning rate schedule
- **Results**:
  - **68% quantile: 51.18°** ⭐ (-0.21° improvement over v18)
- **Status**: ✅ Completed - Marginal but measurable improvement

#### v19_deeper: Production 200k Deeper Network - ❌ FAILED
- **Job ID**: 12833783
- **Config**: `electron_direction/json/three_plane_v19_deeper.json`
- **Dataset**: 200k samples
- **Architecture**: 5 conv layers, 128 filters (deeper than v18)
- **Training**: Failed with NaN losses (training diverged)
- **Status**: ❌ Failed - Network too complex or learning rate too high

#### v19_multitask: Production 200k Multitask - ❌ FAILED (Disk Quota)
- **Job ID**: 12833786 (resubmission failed)
- **Config**: `electron_direction/json/three_plane_v19_multitask.json`
- **Dataset**: 200k samples
- **Architecture**: Multitask learning (direction + energy)
- **Training**: Initial run succeeded but showed worse performance
- **Results**: 68% quantile: 56.34° (+4.95° WORSE than v18)
- **Issue**: Cannot resubmit with new seed - AFS 85% full, disk quota exceeded
- **Status**: ❌ Failed - Needs AFS cleanup before retry

#### v20: Mixed ES+CC 200k Hyperopt - 💤 IDLE
- **Job ID**: 12831888
- **Config**: `electron_direction/json/three_plane_v20_200k_mixed_hyperopt.json`
- **Dataset**: 200k mixed (ES + CC samples from both prod_es and prod_cc)
- **Architecture**: 4 conv layers, 64 filters + hyperparameter optimization
- **Loss**: Angular loss
- **Training**: 20 hyperopt trials
- **Purpose**: Test if mixing interaction types improves generalization
- **Status**: 💤 Idle - Waiting for resources

#### v21: Mixed ES+CC 200k Fixed Params - 💤 IDLE
- **Job ID**: 12831889
- **Config**: `electron_direction/json/three_plane_v21_200k_mixed.json`
- **Dataset**: 200k mixed (ES + CC)
- **Architecture**: 4 conv layers, 64 filters, batch norm (v14 best params)
- **Loss**: Angular loss
- **Training**: No hyperopt, using known good parameters
- **Purpose**: Direct comparison of mixed data with v18 baseline
- **Status**: 💤 Idle - Waiting for resources

#### v22: Mixed ES+CC Incremental Loading - 💤 IDLE
- **Job ID**: 12833757
- **Config**: `electron_direction/json/three_plane_v22_200k_incremental.json`
- **Dataset**: 200k mixed (ES + CC), incremental loading
  - 10k samples per batch, 20 batches
  - Refreshes data every 5 epochs
- **Architecture**: 4 conv layers, 64 filters, batch norm
- **Loss**: Angular loss
- **Training**: 5 epochs per batch × 20 batches = 100 total epochs
- **Resources**: 16GB RAM (vs 24GB for full load)
- **Purpose**: Memory-efficient training with dataset refresh
- **Status**: 💤 Idle - Waiting for resources

---

### Resubmissions with Fixed Main Cluster Selection (November 14, 2025)

**Data Fix**: All cluster images regenerated with corrected main track selection logic
- **Bug**: Previously used `true_particle_energy` (truth info) - not observable in real data
- **Fix**: Now uses `total_energy` (reconstructed cluster energy) - what model can learn from
- **Impact**: Training labels now match observable image features (verified: 94% electron selections)
- **Version increment**: +3 from original (v20→v23, v21→v24, v22→v25)

#### v23: Mixed ES+CC 200k Hyperopt (FIXED DATA) - ⏳ SUBMITTED
- **Config**: `electron_direction/json/three_plane_v23_200k_mixed_hyperopt.json`
- **Dataset**: 200k mixed (ES + CC), **corrected main track selection**
- **Architecture**: 4 conv layers, 64 filters + hyperparameter optimization
- **Loss**: Angular loss
- **Training**: 20 hyperopt trials
- **Resources**: 24GB RAM, 6 CPUs
- **Replaces**: v20 (trained on old buggy data)
- **Status**: ⏳ Ready for submission

#### v24: Mixed ES+CC 200k Fixed Params (FIXED DATA) - ⏳ SUBMITTED
- **Config**: `electron_direction/json/three_plane_v24_200k_mixed.json`
- **Dataset**: 200k mixed (ES + CC), **corrected main track selection**
- **Architecture**: 4 conv layers, 64 filters, batch norm (v14 best params)
- **Loss**: Angular loss
- **Training**: No hyperopt, using known good parameters
- **Resources**: 24GB RAM, 6 CPUs
- **Replaces**: v21 (trained on old buggy data)
- **Status**: ⏳ Ready for submission

#### v25: Mixed ES+CC Incremental Loading (FIXED DATA) - ⏳ SUBMITTED
- **Config**: `electron_direction/json/three_plane_v25_200k_incremental.json`
- **Dataset**: 200k mixed (ES + CC), **corrected main track selection**, incremental loading
  - 10k samples per batch, 20 batches
  - Refreshes data every 5 epochs
- **Architecture**: 4 conv layers, 64 filters, batch norm
- **Loss**: Angular loss
- **Training**: 5 epochs per batch × 20 batches = 100 total epochs
- **Resources**: 16GB RAM (vs 24GB for full load)
- **Replaces**: v22 (trained on old buggy data)
- **Status**: ⏳ Ready for submission

---

## Channel Tagging (CT) Classification

**Goal**: Classify electron interaction type (ES vs CC) from volume images.

**Dataset**: Volume images (208×1242 pixels) from `/eos/home-e/evilla/dune/sn-tps/production_{es,cc}/volume_images_nov10/`
- ES: ~3,739 files
- CC: ~3,372 files

**Evaluation Metric**: Binary classification accuracy

### Successful Training Runs (November 2025)

#### v3: Baseline Streaming (10k balanced) - ⏳ RUNNING
- **Job ID**: 10630313
- **Config**: `channel_tagging/json/volume_v3_balanced_10k.json`
- **Data**: 10k balanced (5k ES + 5k CC), streaming mode
- **Architecture**: 3 conv layers, 32 filters baseline
- **Training**: Streaming data generator (solves memory issues)
- **Resources**: <5GB memory (vs 37-41GB without streaming)
- **Status**: ⏳ Running 2h10m on bigbird10

#### v4: Deeper Network Streaming (10k balanced) - ⏳ RUNNING
- **Job ID**: 10630312
- **Config**: `channel_tagging/json/volume_v4_balanced_10k.json`
- **Data**: 10k balanced, streaming mode
- **Architecture**: 4 conv layers, 64 filters (deeper network)
- **Training**: Streaming data generator
- **Resources**: <5GB memory
- **Status**: ⏳ Running 2h10m on bigbird10

#### v4: Deeper Network Streaming (10k balanced) - ✅ COMPLETED
- **Job ID**: 10630312
- **Config**: `channel_tagging/json/volume_v4_balanced_10k.json`
- **Data**: 10k balanced, streaming mode
- **Architecture**: 4 conv layers, 64 filters (deeper network)
- **Training**: 19 epochs (early stopping), streaming mode
- **Runtime**: ~9 hours (~1553s/epoch)
- **Resources**: <8GB memory (streaming success!)
- **Results**:
  - Training Accuracy: 61.09%
  - Validation Accuracy: 61.21%
  - Training Loss: 0.6587
  - Best val_loss: 0.64604
- **Status**: ✅ Completed - **Only 61% accuracy suggests need for more data or architectural improvements**

#### v5: Lighter Network Streaming (10k balanced) - ❌ FAILED
- **Job ID**: 10280422, 10630092
- **Config**: `channel_tagging/json/volume_v5_balanced_10k.json`
- **Status**: ❌ Failed - Memory exceeded (37GB) despite streaming flag

#### v6: Simple Network More Data (20k balanced) - ❌ STATUS UNKNOWN
- **Job ID**: 10280423
- **Config**: `channel_tagging/json/volume_v6_simple_20k.json`
- **Status**: Status unknown - need to check logs

### New Submissions (November 12, 2025) - Fixing CT Issues

**Problems Identified from v4:**
1. ⚠️ Only 61% accuracy (barely better than random for binary classification)
2. ✅ Streaming mode works but needs proper implementation
3. ❌ v3, v5 failed with 37-41GB memory usage (not using streaming properly)

**Solutions Implemented:**
- Increased dataset to 100k samples per class
- Added batch normalization for stability
- Increased patience (20-25 epochs)
- Created incremental dataset refresh strategy
- Fixed memory issues in streaming implementation

#### v8: Streaming 100k with Batch Norm - ⏳ RUNNING (10h 44m)
- **Job ID**: 13722817
- **Config**: `channel_tagging/json/volume_v8_streaming_100k.json`
- **Data**: 100k balanced (50k ES + 50k CC), streaming mode
- **Architecture**: 4 conv layers, 64 filters, **batch normalization**
- **Training**: 100 epochs, patience=20, streaming generator
- **Resources**: 16GB RAM, CPU only
- **Status**: ⏳ Running on bigbird10 - 10h 44m runtime

#### v9: Incremental Dataset Refresh 150k - 🔄 RESUBMITTED (2h 17m)
- **Job ID**: 13737399 (resubmitted after config fix)
- **Config**: `channel_tagging/json/volume_v9_streaming_150k.json`
- **Data**: 150k total (75k per class), **streaming mode**
- **Architecture**: 4 conv layers, 64 filters, batch norm
- **Training**: 100 epochs, patience=25
- **Resources**: 12GB RAM
- **Status**: ⏳ Running - 2h 17m runtime (resubmitted with streaming config)

#### v10: Higher Dropout 100k - ⏳ RUNNING (10h 44m)
- **Job ID**: 13722819
- **Config**: `channel_tagging/json/volume_v10_dropout_100k.json`
- **Data**: 100k balanced, streaming mode
- **Architecture**: 4 conv layers, 64 filters, batch norm, **dropout=0.5** (vs 0.4)
- **Training**: 100 epochs, patience=20
- **Resources**: 16GB RAM
- **Status**: ⏳ Running on bigbird10 - 10h 44m runtime

#### v11: Deeper Architecture 100k - ⏳ RUNNING (10h 44m)
- **Job ID**: 13722820
- **Config**: `channel_tagging/json/volume_v11_deeper_100k.json`
- **Data**: 100k balanced, streaming mode
- **Architecture**: **5 conv layers, 128 filters**, batch norm, 512 dense units
- **Training**: 100 epochs, patience=20, batch_size=8, lr=0.0003
- **Resources**: 24GB RAM
- **Status**: ⏳ Running on bigbird10 - 10h 44m runtime

#### v12: Quick Test 10k - ⏳ RUNNING (SLOW - 5+ hours)
- **Job ID**: 13737411
- **Config**: `channel_tagging/json/volume_v12_test_10k.json`
- **Data**: 10k balanced (5k per class), **streaming mode**
- **Architecture**: 4 conv layers, 64 filters, batch norm (same as v8)
- **Training**: 50 epochs, patience=15, batch_size=16
- **Resources**: 16GB RAM
- **Purpose**: Quick test to verify evaluation plots work correctly
- **Status**: ⏳ Running but VERY slow - streaming overhead extreme for small dataset (5+ hours, memory at 12.5GB)
- **Note**: Streaming mode has massive overhead for datasets <100k samples

#### v13: Quick Test 10k WITHOUT Streaming - ⏳ RUNNING (just started)
- **Job ID**: 13738102
- **Config**: `channel_tagging/json/volume_v13_test_10k.json`
- **Data**: 10k balanced (5k per class), **NO streaming** (loads all into memory)
- **Architecture**: 4 conv layers, 64 filters, batch norm (same as v8/v12)
- **Training**: 50 epochs, patience=15, batch_size=16
- **Resources**: 18GB RAM, 6 CPUs
- **Purpose**: Test non-streaming mode - should be 10-20x faster than v12
- **Status**: ⏳ Just submitted
- **Scripts Fixed**: Both `train_ct_volume_streaming.py` and `train_ct_volume_simple.py` now read paths from JSON config instead of hardcoded paths

#### v14: 100k NO Streaming - ⏳ RUNNING
- **Job ID**: 13738908
- **Config**: `channel_tagging/json/volume_v14_100k.json`
- **Data**: 100k balanced (50k per class), **NO streaming** - loads all into memory
- **Architecture**: 4 conv layers, 64 filters, batch norm (same as v8 but without streaming)
- **Training**: 100 epochs, patience=20, batch_size=16
- **Resources**: 20GB RAM, 6 CPUs
- **Improvements**: 
  - Disabled streaming mode (100x faster than v8)
  - Added checkpoints every 5 epochs for crash recovery
  - Fixed path reading from JSON config
- **Expected runtime**: ~5 hours (vs 500 hours with streaming)
- **Status**: ⏳ Just submitted

#### v15: 150k NO Streaming - ⏳ RUNNING (RESUBMITTED)
- **Job ID**: 13740287 (resubmitted after checkpoint fix)
- **Config**: `channel_tagging/json/volume_v15_150k.json`
- **Data**: 150k balanced (75k per class), **NO streaming**
- **Architecture**: 4 conv layers, 64 filters, batch norm (same as v9 but without streaming)
- **Training**: 100 epochs, patience=25, batch_size=16
- **Resources**: 24GB RAM, 6 CPUs
- **Improvements**: 
  - Disabled streaming mode (100x faster)
  - Added checkpoints every 5 epochs
  - Fixed path reading from JSON config
- **Expected runtime**: ~7 hours
- **Status**: ⏳ Just submitted

#### v16: 100k Higher Dropout NO Streaming - ⏳ RUNNING (RESUBMITTED)
- **Job ID**: 13740288 (resubmitted after checkpoint fix)
- **Config**: `channel_tagging/json/volume_v16_100k.json`
- **Data**: 100k balanced (50k per class), **NO streaming**
- **Architecture**: 4 conv layers, 64 filters, batch norm, **dropout=0.5** (vs 0.4 in v14)
- **Training**: 100 epochs, patience=20, batch_size=16
- **Resources**: 20GB RAM, 6 CPUs
- **Improvements**: 
  - Disabled streaming mode (100x faster than v10)
  - Added checkpoints every 5 epochs
  - Fixed path reading from JSON config
- **Expected runtime**: ~5 hours
- **Status**: ⏳ Just submitted

#### v17: 100k Deeper Architecture NO Streaming - ⏳ RUNNING
- **Job ID**: 13738911
- **Config**: `channel_tagging/json/volume_v17_100k.json`
- **Data**: 100k balanced (50k per class), **NO streaming**
- **Architecture**: **5 conv layers, 128 filters**, batch norm, dense=512 (deeper than v14)
- **Training**: 100 epochs, patience=20, batch_size=8, lr=0.0003
- **Resources**: 24GB RAM, 6 CPUs
- **Improvements**: 
  - Disabled streaming mode (100x faster than v11)
  - Added checkpoints every 5 epochs
  - Fixed path reading from JSON config
- **Expected runtime**: ~6-7 hours (larger model)
- **Status**: ⏳ Just submitted

#### v18: 10k Quick Test NO Streaming - ⏳ RUNNING
- **Job ID**: 12835071
- **Config**: `channel_tagging/json/volume_v18_10k.json`
- **Data**: 10k balanced (5k per class), **NO streaming**
- **Architecture**: 4 conv layers, 64 filters, batch norm
- **Training**: 50 epochs, patience=15, batch_size=16
- **Resources**: 18GB RAM, 6 CPUs
- **Status**: ⏳ Running 6+ hours

#### v19: 20k NO Streaming - ⏳ RUNNING
- **Job ID**: 12835072
- **Config**: `channel_tagging/json/volume_v19_20k.json`
- **Data**: 20k balanced (10k per class), **NO streaming**
- **Architecture**: 4 conv layers, 64 filters, batch norm
- **Training**: 75 epochs, patience=18, batch_size=16
- **Resources**: 19GB RAM, 6 CPUs
- **Status**: ⏳ Running 6+ hours

#### v20: 50k NO Streaming - ⏳ RUNNING
- **Job ID**: 12835073
- **Config**: `channel_tagging/json/volume_v20_50k.json`
- **Data**: 50k balanced (25k per class), **NO streaming**
- **Architecture**: 4 conv layers, 64 filters, batch norm
- **Training**: 100 epochs, patience=20, batch_size=16
- **Resources**: 21GB RAM, 6 CPUs
- **Status**: ⏳ Running 6+ hours

---

### Resubmissions with Fixed Main Cluster Selection (November 14, 2025)

**Data Fix**: All cluster images regenerated with corrected main track selection logic
- **Bug**: Previously used `true_particle_energy` (truth info) - not observable in real data
- **Fix**: Now uses `total_energy` (reconstructed cluster energy) - what model can learn from
- **Impact**: Training labels now match observable image features (verified: 94% electron selections)
- **Version increment**: +7 from original (v14→v21, v15→v22, etc.)

#### v21: 100k NO Streaming (FIXED DATA) - ⏳ SUBMITTED
- **Config**: `channel_tagging/json/volume_v21_100k.json`
- **Data**: 100k balanced (50k per class), **corrected main track selection**
- **Architecture**: 4 conv layers, 64 filters, batch norm (same as v14)
- **Training**: 100 epochs, patience=20, batch_size=16
- **Resources**: 20GB RAM, 6 CPUs
- **Replaces**: v14 (trained on old buggy data)
- **Status**: ⏳ Ready for submission

#### v22: 150k NO Streaming (FIXED DATA) - ⏳ SUBMITTED
- **Config**: `channel_tagging/json/volume_v22_150k.json`
- **Data**: 150k balanced (75k per class), **corrected main track selection**
- **Architecture**: 4 conv layers, 64 filters, batch norm (same as v15)
- **Training**: 100 epochs, patience=25, batch_size=16
- **Resources**: 24GB RAM, 6 CPUs
- **Replaces**: v15 (trained on old buggy data)
- **Status**: ⏳ Ready for submission

#### v23: 100k Higher Dropout (FIXED DATA) - ⏳ SUBMITTED
- **Config**: `channel_tagging/json/volume_v23_100k.json`
- **Data**: 100k balanced (50k per class), **corrected main track selection**
- **Architecture**: 4 conv layers, 64 filters, batch norm, **dropout=0.5** (vs 0.4)
- **Training**: 100 epochs, patience=20, batch_size=16
- **Resources**: 20GB RAM, 6 CPUs
- **Replaces**: v16 (trained on old buggy data)
- **Status**: ⏳ Ready for submission

#### v24: 100k Deeper Architecture (FIXED DATA) - ⏳ SUBMITTED
- **Config**: `channel_tagging/json/volume_v24_100k.json`
- **Data**: 100k balanced (50k per class), **corrected main track selection**
- **Architecture**: **5 conv layers, 128 filters**, batch norm, dense=512 (deeper than v21)
- **Training**: 100 epochs, patience=20, batch_size=8, lr=0.0003
- **Resources**: 24GB RAM, 6 CPUs
- **Replaces**: v17 (trained on old buggy data)
- **Status**: ⏳ Ready for submission

#### v25: 10k Quick Test (FIXED DATA) - ⏳ SUBMITTED
- **Config**: `channel_tagging/json/volume_v25_10k.json`
- **Data**: 10k balanced (5k per class), **corrected main track selection**
- **Architecture**: 4 conv layers, 64 filters, batch norm (same as v18)
- **Training**: 50 epochs, patience=15, batch_size=16
- **Resources**: 18GB RAM, 6 CPUs
- **Replaces**: v18 (trained on old buggy data)
- **Status**: ⏳ Ready for submission

#### v26: 20k (FIXED DATA) - ⏳ SUBMITTED
- **Config**: `channel_tagging/json/volume_v26_20k.json`
- **Data**: 20k balanced (10k per class), **corrected main track selection**
- **Architecture**: 4 conv layers, 64 filters, batch norm (same as v19)
- **Training**: 75 epochs, patience=18, batch_size=16
- **Resources**: 19GB RAM, 6 CPUs
- **Replaces**: v19 (trained on old buggy data)
- **Status**: ⏳ Ready for submission

#### v27: 50k (FIXED DATA) - ⏳ SUBMITTED
- **Config**: `channel_tagging/json/volume_v27_50k.json`
- **Data**: 50k balanced (25k per class), **corrected main track selection**
- **Architecture**: 4 conv layers, 64 filters, batch norm (same as v20)
- **Training**: 100 epochs, patience=20, batch_size=16
- **Resources**: 21GB RAM, 6 CPUs
- **Replaces**: v20 (trained on old buggy data)
- **Status**: ⏳ Ready for submission

### Key Technical Solution

**Streaming Mode Implementation**: Using `train_ct_volume_streaming.py` instead of loading full dataset
- Reduced memory from 37-41GB → <8GB
- Uses TensorFlow data generators for batch-wise loading
- All 4 variants running successfully

---

## Training Best Practices

1. **Naming Convention**: Always use proper version names in output directories
   - Format: `{task}_{version}_{variant}_{timestamp}`
   - Never use "unknown" or generic names

2. **Path Management**: 
   - Use `/eos/home-e/evilla/` for data (not `/eos/user/e/evilla/`)
   - Verify dataset consistency before submission

3. **Resource Estimation**:
   - Simple CNN: 4-8GB RAM
   - Streaming mode (CT): <8GB vs 37-41GB for full load
   - Hyperopt: May need 12-24GB depending on architecture

4. **Documentation**: Update this file immediately after job completion with results

---

## Legend

- ✅ Completed successfully with results
- ⏳ Currently running
- ❌ Failed or poor performance (archived)

---

## Data Fix (November 14, 2025)

**Critical Update - Cluster Selection Logic Fixed**

All cluster image datasets were regenerated with corrected cluster selection:
- **Previous**: Selected cluster with highest `true_particle_energy`
- **Fixed**: Selects cluster with highest `reconstructed_energy` (actual cluster energy)
- **Impact**: More physically accurate labels - matches what detector actually measures
- **Status**: cat000001 clusters regenerated, all downstream images updated

**Debug Verification** (make_clusters.cpp with `-d` flag):
- 94% of selected clusters are electrons (PDG=11) ✓
- 6% are photons (PDG=22) - only when no electron cluster exists in that view
- Confirmed: Selection based on reconstructed energy works correctly

**Resubmitted Jobs**: All currently running ML jobs resubmitted with new data (version numbers incremented by 7 for CT, by 3 for ED)

---

*Last updated: November 14, 2025*


## Legend

- ✅ Completed successfully
- ❌ Failed/Poor performance
- ⏳ Running
- �� Resubmitted
- 💤 Idle (waiting for resources)

---

## Recent Submissions (November 11, 2025)

### Electron Direction - 3-Plane Matched

#### 1. Three-Plane CNN with Cosine Loss (v6) - ⏳ IDLE
- **Job ID**: 10629902
- **Config**: `electron_direction/json/three_plane_cosine_v6_10k.json`
- **Data**: `/eos/user/e/evilla/dune/sn-tps/production_es/cluster_images_nov10/`
  - 10k 3-plane matched samples (X matched to both U and V via match_id)
  - Uses metadata field 13 for cross-plane matching
- **Architecture**: Three separate CNN branches (one per plane) + concatenation
  - Input: 128×16×1 per plane
  - Conv layers: 3 per branch (32 filters, kernel=3)
  - Dense layers after concat: 2 (256 units)
  - Output: 3D normalized direction vector
- **Training**: Cosine similarity loss, 100 epochs, batch=32
- **Resources**: 1 CPU, 8GB RAM (test run)
- **Status**: Idle, waiting for CPU slot

### Channel Tagging - Volume Images

#### 1. Balanced ES/CC 10k (v2) - ⏳ IDLE
- **Job ID**: 10629909
- **Config**: `channel_tagging/json/volume_v2_balanced_10k_nov10.json`
- **Data**: 
  - ES: `/eos/user/e/evilla/dune/sn-tps/production_es/volume_images_nov10/` (~3,739 files)
  - CC: `/eos/user/e/evilla/dune/sn-tps/production_cc/volume_images_nov10/` (~3,372 files)
  - 10k balanced: 5k ES + 5k CC
- **Architecture**: Simple CNN for volume images
  - Input: 208×1242×1 (full detector X-plane view)
  - Conv layers: 3 (32 filters)
  - Dense: 128 units, dropout 0.3
  - Output: 2 classes (ES vs CC)
- **Training**: 50 epochs, batch=16, early stopping patience=10
- **Resources**: 1 CPU, 12GB RAM, 1 GPU
- **Status**: Idle, waiting for GPU

### ED Loss Function Experiments - Results

#### Jobs 10626064 & 10626065 - ❌ TIMEOUT
- **Focal Angular Loss** (10626064): Killed after 24h (hit time limit)
- **Hybrid Loss** (10626065): Killed after 24h (hit time limit)
- Both jobs ran for exactly 24 hours before being terminated
- Need to reduce trials or use shorter job flavor for future hyperopt runs


#### v28-v34: GPU-Accelerated Retraining (SUBMITTED Nov 14 2025)

**CRITICAL FIX**: Previous jobs (v14-v17, v21-v27) submitted WITHOUT GPU requests! 
This caused extremely slow training (days instead of hours). Resubmitted with proper GPU allocation.

##### v28: 100k Balanced with GPU - ⏳ RUNNING
- **Job ID**: 12863467
- **Config**: `channel_tagging/json/volume_v28_100k.json`
- **Data**: 100k balanced (50k per class), **corrected main track selection**
- **Architecture**: 4 conv layers, 64 filters, batch norm
- **Training**: 100 epochs, patience=20, batch_size=16
- **Resources**: 16GB RAM, 4 CPUs, **1 GPU** ✓
- **Replaces**: v21 (no GPU)
- **Status**: ⏳ Submitted Nov 14 10:52 CET

##### v29: 150k Balanced with GPU - ⏳ RUNNING
- **Job ID**: 12863468
- **Config**: `channel_tagging/json/volume_v29_150k.json`
- **Data**: 150k balanced (75k per class), **corrected main track selection**
- **Architecture**: 4 conv layers, 64 filters, batch norm
- **Training**: 100 epochs, patience=20, batch_size=16
- **Resources**: 16GB RAM, 4 CPUs, **1 GPU** ✓
- **Replaces**: v22 (no GPU)
- **Status**: ⏳ Submitted Nov 14 10:52 CET

##### v30: 100k Balanced with GPU - ⏳ RUNNING
- **Job ID**: 12863469
- **Config**: `channel_tagging/json/volume_v30_100k.json`
- **Data**: 100k balanced (50k per class), **corrected main track selection**
- **Architecture**: 4 conv layers, 64 filters, batch norm
- **Training**: 100 epochs, patience=20, batch_size=16
- **Resources**: 16GB RAM, 4 CPUs, **1 GPU** ✓
- **Replaces**: v23 (no GPU)
- **Status**: ⏳ Submitted Nov 14 10:52 CET

##### v31: 100k Balanced with GPU - ⏳ RUNNING
- **Job ID**: 12863470
- **Config**: `channel_tagging/json/volume_v31_100k.json`
- **Data**: 100k balanced (50k per class), **corrected main track selection**
- **Architecture**: 4 conv layers, 64 filters, batch norm
- **Training**: 100 epochs, patience=20, batch_size=16
- **Resources**: 16GB RAM, 4 CPUs, **1 GPU** ✓
- **Replaces**: v24 (no GPU)
- **Status**: ⏳ Submitted Nov 14 10:52 CET

##### v32: 10k Balanced with GPU - ⏳ RUNNING
- **Job ID**: 12863471
- **Config**: `channel_tagging/json/volume_v32_10k.json`
- **Data**: 10k balanced (5k per class), **corrected main track selection**
- **Architecture**: 4 conv layers, 64 filters, batch norm
- **Training**: 100 epochs, patience=20, batch_size=16
- **Resources**: 16GB RAM, 4 CPUs, **1 GPU** ✓
- **Replaces**: v25 (no GPU)
- **Status**: ⏳ Submitted Nov 14 10:52 CET

##### v33: 20k Balanced with GPU - ⏳ RUNNING
- **Job ID**: 12863472
- **Config**: `channel_tagging/json/volume_v33_20k.json`
- **Data**: 20k balanced (10k per class), **corrected main track selection**
- **Architecture**: 4 conv layers, 64 filters, batch norm
- **Training**: 100 epochs, patience=20, batch_size=16
- **Resources**: 16GB RAM, 4 CPUs, **1 GPU** ✓
- **Replaces**: v26 (no GPU)
- **Status**: ⏳ Submitted Nov 14 10:52 CET

##### v34: 50k Balanced with GPU - ⏳ RUNNING
- **Job ID**: 12863473
- **Config**: `channel_tagging/json/volume_v34_50k.json`
- **Data**: 50k balanced (25k per class), **corrected main track selection**
- **Architecture**: 4 conv layers, 64 filters, batch norm
- **Training**: 100 epochs, patience=20, batch_size=16
- **Resources**: 16GB RAM, 4 CPUs, **1 GPU** ✓
- **Replaces**: v27 (no GPU)
- **Status**: ⏳ Submitted Nov 14 10:52 CET

**Note**: Jobs v14-v17 and v21-v27 will continue running but will be MUCH slower (CPU-only training).
The GPU versions (v28-v34) should complete significantly faster and will be used for production.


#### v35: Hyperparameter Optimization (10k samples, 5 trials) - ⏳ RUNNING

**Purpose**: First hyperparameter optimization for CT using grid search

- **Job ID**: 12866547
- **Config**: `channel_tagging/json/volume_v35_hyperopt_10k.json`
- **Script**: `channel_tagging/models/run_hyperopt_simple.py`
- **Data**: 10k total samples (5k ES + 5k CC) from prod_es/prod_cc (fixed cluster selection)
- **Resources**: 16GB RAM, 4 CPUs, **1 GPU**, 10GB disk, nextweek flavor
- **Status**: ⏳ RUNNING (submitted Nov 14, 14:43)

**Hyperparameter Search Space**:
- Learning rate: [0.0001, 0.001] (log scale, 3 values)
- Dropout rate: [0.2, 0.3, 0.4, 0.5]
- Number of filters: [32, 64, 128]
- Dense units: [128, 256, 512]

**Search Strategy**:
- Random sampling from full grid (5 trials max)
- Each trial: 50 epochs max, early stopping patience=15
- Metric: Validation accuracy
- Seeds: reproducible (42)

**Expected Output**:
- `hyperopt_results.json`: All trial results with best parameters
- Individual trial directories with models and confusion matrices
- Best model saved separately

**Notes**:
- Using simple wrapper script that runs `train_ct_volume_simple.py` 5 times
- Each trial creates temporary config with updated hyperparameters
- All other features enabled: confusion matrices, predictions, energy analysis
- Output directory: `/eos/user/e/evilla/dune/sn-tps/neural_networks/channel_tagging/v35_hyperopt_10k_hyperopt/TIMESTAMP/`


#### v42: 100k NO Streaming (CORRECTED DATA) - ✅ COMPLETED
- **Job ID**: 12882049
- **Config**: `channel_tagging/json/volume_v42_corrected_100k.json`
- **Data**: 100k balanced (50k per class), **corrected main track selection**
- **Architecture**: 4 conv layers, 64 filters, batch norm
- **Training**: 49 epochs completed, stopped at validation plateau
- **Resources**: 24GB RAM, GPU (H100L MIG)
- **Runtime**: ~50 minutes (21:34 - 21:52)
- **Results**:
  - **Training Accuracy: 67.1%**
  - **Validation Accuracy: 65.3%**
  - Best model saved at epoch with lowest val_loss
- **Status**: ✅ Completed - Baseline performance, needs deeper architecture
- **Model Path**: `/eos/user/e/evilla/dune/sn-tps/neural_networks/channel_tagging/v42_corrected_100k/20251114_213453/best_model.keras`

#### v43: Deep Architecture 100k - ⏳ SUBMITTING
- **Config**: `channel_tagging/json/volume_v43_deep_100k.json`
- **Data**: 100k balanced (50k per class), **corrected main track selection**
- **Architecture**: **Deep custom CNN based on user specification**
  - 6 conv blocks: 28, 28, 29, 47, 48, 48 filters
  - MaxPooling after each conv block
  - 2 dense layers: 96, 32 units
  - Batch norm, dropout 0.4
- **Training**: 100 epochs, patience=20, batch_size=16
- **Resources**: 24GB RAM, 4 CPUs, 1 GPU
- **Purpose**: Test significantly deeper architecture for improved accuracy
- **Status**: ⏳ Ready for submission

#### v44: Incremental Loading with Increased Memory - ⏳ SUBMITTING
- **Config**: `channel_tagging/json/volume_v40_corrected_50k.json`
- **Data**: 50k total (5k per batch × 10 batches), **incremental loading**
- **Architecture**: 4 conv layers, 64 filters, batch norm (same as v42)
- **Training**: 5 epochs per batch, dataset refreshes every batch
- **Resources**: 80GB RAM (increased from v40's 40GB), 4 CPUs, 1 GPU
- **Purpose**: Fix v40 memory issues with incremental loading strategy
- **Status**: ⏳ Ready for submission

---

*Last updated: November 15, 2025*
