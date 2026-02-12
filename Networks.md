# SN-TPS Neural Networks Documentation

**Last Updated:** November 16, 2025

## Overview

This document tracks all neural network training runs for the DUNE SN-TPS project across two main tasks:
1. **Electron Direction (ED)**: Regression for electron direction angles
2. **Channel Tagging (CT)**: Multi-class classification (ES/CC/NC)

---

## 1. Electron Direction (ED)

### Production Models

| Version | Samples | Mean Error | Median Error | Architecture | Hyperopt | Date | Status |
|---------|---------|------------|--------------|--------------|----------|------|--------|
| v14 | ? | **25° @ 68%** | ? | Three-plane | ✅ Yes | Nov 13 | ✅ BEST |
| v26 | 50k | 54.8° | 38.9° | Three-plane | ❌ No | Nov 14 | ⚠️ NaN crash |
| v50 | 100k | 49.9° | 33.5° | Three-plane | ❌ No | Nov 16 | ⚠️ Underperforming |
| v27 | 100k | ? | ? | Three-plane | ❌ No | Nov 14 | Complete |

### Current Work

| Version | Purpose | Status | Job ID |
|---------|---------|--------|--------|
| v52 | 100k with hyperopt | 🔄 Running | Varies |

### Key Issues

**CRITICAL: Hyperopt Required for Good Performance**
- v14 with hyperopt: **25° @ 68% quantile** ✅
- v50 without hyperopt: 49.9° mean ❌ (MUCH WORSE)
- v26/v27/v50 all trained without hyperopt and underperformed
- **Action:** Always enable hyperopt for ED training

**Training Stability:**
- v26 crashed with NaN loss at epoch 73
- Need terminate_on_nan callbacks
- Best checkpoints often before final epoch

---

## 2. Channel Tagging (CT)

### Successful Models (Nov 2025)

| Model Name | Samples | Architecture | Test Acc | Notes | Location |
|------------|---------|--------------|----------|-------|----------|
| **ct_volume_v52_batch_reload** | 100k | Volume CNN (X-plane) | ~75% | Batch reload, good baseline | 20251116_101125 |
| **ct_volume_v72_deeper_10k** | 10k | Deeper Volume CNN | ~78% | Better architecture | 20251124_005847 |
| **ct_volume_v78_dario_10k** | 10k | Dario architecture | ~80% | Best single-plane | 20251123_153908 |
| **three_plane_v70** | 5k | Three-plane CNN | ~82% | Multi-plane fusion | 20251119_174031 |
| **v77_dario_batch_5k** | 5k | Dario + batch reload | ~79% | Production-ready | 20251121_021945/073847 |

### Key Metrics Summary

**Best Performance:**
- **Overall Accuracy:** ~80-82% (three-plane models)
- **ES Classification:** ~85% recall with confidence thresholds ≥0.7
- **CC Classification:** ~80% recall with confidence thresholds ≥0.7
- **Confidence Threshold Trade-off:** 
  - No threshold: ~75% accuracy, 100% retention
  - Threshold ≥0.8: ~85% accuracy, ~30% retention
  - Threshold ≥0.9: ~93% accuracy, ~5% retention

### Architecture Evolution

**Volume-based approach (1m × 1m, 208×1242 pixels):**
- Single X-plane baseline: ~75% accuracy
- Deeper architectures: +3-5% improvement
- Three-plane fusion: +5-7% improvement over single plane

**Training strategies:**
- Batch reload training handles memory constraints for large datasets
- 5k-10k samples sufficient for good convergence
- Early stopping prevents overfitting

### Failed Approaches (Archived)

The following approaches were tested but did not yield better results:
- **Cluster-based images:** Less information than volume approach
- **Cropped volumes:** Lost important spatial context
- **Shallow architectures:** Insufficient capacity for task complexity
- **Very deep networks without regularization:** Overfitting issues
- **Single-batch full-memory loading:** Memory constraints on large datasets
- **Hyperparameter search without proper validation:** Unstable results

### Data Pipeline

**Current Structure:**
- Separate U/V/X plane subfolders with matched `main_cluster_match_id`
- Volume images: 1m × 1m detector region, 208×1242 pixels
- ADC values normalized per plane
- Enables efficient three-plane matched loading

**Comprehensive Analysis:**
- All models have 9-page PDF reports in their directories
- Includes confusion matrices at thresholds 0.6, 0.7, 0.8, 0.9
- ES probability distributions showing class separation
- Training history and ROC curves
- Energy-dependent performance analysis

---

## Network Organization

### Directory Structure

```
/eos/user/e/evilla/dune/sn-tps/neural_networks/
├── electron_direction/     # Electron Direction models (42 versions)
└── channel_tagging/        # Channel Tagging models (14 versions)
```

### Cleanup Strategy

**Keep:**
- Best performing models (v14, v42)
- Latest models (v50, v60)
- Currently training (v52)
- Failed runs with debugging value (v26 NaN crash)

**Archive/Remove:**
- Old development versions (v1-v5 range)
- Superseded small test runs
- Duplicate experiments without unique insights
- Intermediate hyperopt attempts that failed

---

## Current Jobs (Nov 16, 2025 05:10)

| Task | Version | Job ID | Runtime | Status |
|------|---------|--------|---------|--------|
| CT | v52 batch reload | 12910323 | 3h+ | Running |
| CT | v60 three-plane | 12913042 | 2min | Just started |

---

## Best Practices Learned

### General
1. **Always save training history** - Critical for debugging and analysis
2. **Use hyperopt for ED** - Required for good performance (25° vs 50°)
3. **Checkpoint frequently** - NaN crashes can lose best models
4. **Balance classes for CT** - Critical for classification tasks

### Data Management
5. **Matched three-plane data** - Use main_cluster_match_id for cross-plane matching
6. **Volume images for CT** - Better than cluster images (65% vs lower)
7. **Proper data structure** - Separate plane folders enable efficient loading

### Training
8. **Terminate on NaN** - Prevent wasted GPU time on crashed runs
9. **Batch reload for memory** - Handle large datasets that don't fit in RAM
10. **Early stopping** - Prevent overfitting, often best model before final epoch

### Infrastructure
11. **Source init.sh in wrappers** - Critical for PYTHONPATH and environment
12. **Test locally first** - Avoid wasted HTCondor submissions (v60 example)
13. **Use proper command flags** - Silence output properly, not just /dev/null

---

## Next Steps

1. **ED v52:** Monitor hyperopt training - should recover v14 performance
2. **CT v60:** Monitor three-plane training - expected to beat v42's 65.3%
3. **CT v52:** Check batch reload completion
4. **Cleanup:** Archive old network directories (detailed plan needed)
5. **Analysis:** Run comprehensive analysis on v60 when complete

---

## Performance Targets

| Task | Current Best | Target | Gap |
|------|--------------|--------|-----|
| ED | 25° @ 68% (v14) | 20° @ 68% | Hyperopt tuning |
| CT | 65.3% (v42) | 75%+ | Three-plane approach (v60) |

