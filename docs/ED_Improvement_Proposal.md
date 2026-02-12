# Electron Direction (ED) Improvement Proposal

**Date**: November 12, 2025  
**Current Best**: v14 with 24.94° @ 68% containment (from cosine distribution)  
**Latest Production**: v15 with 30.44° median on 200k samples

---

## Current Status Summary

### Performance Achieved
- **v14 (10k, hyperopt)**: 24.94° @ 68% containment, 12.70° median
- **v15 (200k, no hyperopt)**: 30.44° median, 47.32° mean
- **v7 (previous best)**: 35.5° @ 68% containment

### Key Findings
1. **v14 significantly outperforms v15** despite having 20x less data
2. This suggests v14 benefited from **hyperparameter optimization**
3. v15 used fixed hyperparameters without optimization
4. Both use 4 conv layers, 64 filters architecture

---

## Proposed Improvements

### 1. **Immediate: Run v15 with Hyperopt** ⭐ HIGH PRIORITY
**Goal**: Achieve v14 performance level on 200k dataset

**Action**:
- Take v15 config (200k samples, 4 conv/64 filters)
- Enable hyperparameter optimization
- Expected: 20-30% improvement over current v15

**Rationale**: v14 shows hyperopt is crucial. Scaling to 200k with proper hyperopt should give best results.

**Config**: `three_plane_v17_200k_hyperopt.json`
```json
{
  "model": {
    "name": "three_plane_v17_200k_hyperopt",
    "n_conv_layers": 4,
    "n_filters": 64,
    ...
  },
  "data": {
    "max_samples": 200000
  },
  "hyperopt": {
    "enabled": true,
    "n_trials": 50,
    "n_epochs_per_trial": 20
  }
}
```

**Expected Runtime**: ~12-24 hours  
**Expected Result**: < 22° @ 68% containment

---

### 2. **Architecture Exploration**

#### 2a. Deeper Network (v18)
- **Current**: 4 conv layers
- **Proposed**: 6 conv layers with residual connections
- **Rationale**: Three-plane fusion might benefit from deeper feature extraction
- **Risk**: Overfitting on 200k samples

#### 2b. Attention Mechanism (v19)
- Add cross-plane attention layers after CNN branches
- Learn which plane provides most information for each sample
- **Papers**: "Attention is All You Need", multi-modal fusion literature

#### 2c. Multi-Scale Features (v20)
- Use different kernel sizes: 3×3, 5×5, 7×7
- Inception-style modules for each plane
- Capture both fine (charge blobs) and coarse (track shape) features

---

### 3. **Data Augmentation**

#### 3a. Physics-Informed Augmentation
- **Rotation**: Rotate all 3 planes together (preserves 3D geometry)
- **Flipping**: Mirror along wire direction (valid for detector)
- **Noise addition**: Add Gaussian noise to simulate electronics noise
- **Energy scaling**: Scale charge deposition (simulates energy variations)

**Expected Benefit**: Effective 4-8x data increase, better generalization

#### 3b. Mix-Up Regularization
- Blend pairs of samples: `x = λx₁ + (1-λ)x₂`
- Interpolate directions using spherical interpolation (SLERP)
- Proven effective for angular regression tasks

---

### 4. **Loss Function Engineering**

#### 4a. Uncertainty-Aware Loss (v21) ⭐ IMPORTANT FOR LIKELIHOOD
- Predict both direction **and** uncertainty: `(dx, dy, dz, σ)`
- Loss penalizes both error and miscalibrated uncertainty
- **Benefit**: Enables proper likelihood construction
- **Implementation**: Add sigma output head, use negative log-likelihood loss

```python
# Predicted: (direction, log_sigma)
# Loss: angular_error / sigma² + log(sigma²)
```

**This is crucial for likelihood-based analysis!**

#### 4b. Focal Loss for Hard Examples
- Previous attempt (v9) failed with wrong hyperparameters
- Retry with proper tuning: γ=1.0-2.0, α balanced by energy bins
- Focus learning on difficult/high-energy events

#### 4c. Multi-Task Learning
- Jointly predict direction + energy + event type
- Shared CNN features, separate heads
- Energy/type prediction as auxiliary tasks improves direction

---

### 5. **Energy-Dependent Performance**

#### 5a. Energy Binning Analysis
**Action**: Analyze current v14/v15 predictions by energy bins:
- Low energy: < 20 MeV
- Medium: 20-40 MeV  
- High: > 40 MeV

**Script**: Create `analyze_by_energy.py` using saved energy metadata

**Expected Finding**: Performance degrades at low energy (harder to resolve)

#### 5b. Energy-Conditioned Network (v22)
- Add energy as input feature (normalized log-energy)
- Let network learn energy-dependent features
- Alternatively: Train separate models per energy bin

---

### 6. **Ensemble Methods**

#### 6a. Bootstrap Ensemble
- Train 5-10 models with different random seeds
- Average predictions using spherical mean
- Uncertainty from ensemble spread
- **Benefit**: Better calibration for likelihood

#### 6b. Cross-Validation Ensemble
- 5-fold cross-validation on full dataset
- Each fold becomes a production model
- Ensemble all folds for final prediction

---

### 7. **Likelihood Construction** 🎯

**Goal**: Build P(cosθ | E, topology) for supernova pointing

#### Requirements (Now Satisfied ✅)
1. ✅ Predictions saved with energy metadata
2. ✅ True directions for calibration
3. ✅ 68% containment from cosine distribution
4. ⏳ Uncertainty estimates (need v21)

#### Proposed Method
1. Bin by energy: [5-10, 10-15, 15-20, 20-30, 30-50, 50+] MeV
2. For each bin, fit distribution: P(cosθ | E)
   - Options: Gaussian on cosine, Von Mises-Fisher, Kernel Density
3. Smooth across bins (avoid sharp transitions)
4. Validate on held-out test set

**Script to Create**: `build_likelihood.py`
- Input: val_predictions.npz with energies
- Output: likelihood_tables.npz (P(cosθ|E) for each bin)

---

## Recommended Execution Order

### Phase 1: Quick Wins (This Week)
1. ✅ Add energy to predictions
3. ✅ Fix cosine plot with 68% line
4. **v17**: 200k with hyperopt (submit today)
5. Energy binning analysis on v14 results

### Phase 2: Architecture (Next Week)
6. **v21**: Uncertainty-aware network (for likelihood)
7. Data augmentation on best architecture
8. Build likelihood tables

### Phase 3: Advanced (Following Week)
9. Attention mechanism (v19)
10. Energy-conditioned network (v22)
11. Ensemble methods
12. Final production model selection

---

## Success Metrics

### Immediate (v17)
- **Target**: < 22° @ 68% containment
- **Minimum**: < 25° @ 68% containment
- If achieved: Move to Phase 2

### Medium Term (v21)
- **Target**: Uncertainty estimates within 20% of true error
- **Metric**: Calibration plot (predicted σ vs actual error)

### Final Production
- **Target**: < 20° @ 68% containment
- **Target**: Well-calibrated uncertainties for likelihood
- **Target**: Energy-dependent performance documented

---

## Resource Requirements

### Computational
- v17 (200k hyperopt): ~24 hours, 1 GPU, 32GB RAM
- v21 (uncertainty): ~6 hours, 1 GPU, 32GB RAM
- Ensemble (5 models): ~30 hours total

### Storage
- Each model: ~130 MB (checkpoints)
- Predictions: ~5 MB per run
- Likelihood tables: < 1 MB

### Timeline
- Phase 1: 3-5 days
- Phase 2: 5-7 days
- Phase 3: 7-10 days
- **Total**: 3-4 weeks to production-ready model

---

## Risk Assessment

### High Priority Risks
1. **Overfitting on 200k**: Monitor val loss carefully, use dropout
2. **Hyperopt computational cost**: Limit to 50 trials max
3. **Likelihood miscalibration**: Need uncertainty estimates (v21)

### Medium Risks
4. **Attention mechanism complexity**: May not improve, keep simple baseline
5. **Energy conditioning**: May not help if energy info already in images

### Low Risks
6. **Data augmentation**: Physics-informed, low risk
7. **Ensemble**: Proven method, computationally expensive but safe

---

## Open Questions

1. **Q**: Should we explore single-plane vs three-plane tradeoff?  
   **A**: Three-plane consistently better, focus on improving fusion

2. **Q**: What about topology information (ES vs CC)?  
   **A**: Could add as auxiliary task in multi-task learning (v21+)

3. **Q**: Do we need test set or just validation?  
   **A**: Keep test set held-out for final production validation

4. **Q**: Likelihood: Parametric or non-parametric?  
   **A**: Start with Von Mises-Fisher (parametric), fall back to KDE if needed

---

## Conclusion

**Immediate Action**: Submit v17 (200k + hyperopt) tonight.  
**Expected**: This alone should match or beat v14 performance.  
**Next Critical Step**: Implement uncertainty-aware network (v21) for likelihood.  

The path to < 20° @ 68% containment is clear:
1. Scale up hyperopt (v17)
2. Add uncertainty (v21)
3. Optimize with augmentation
4. Build calibrated likelihood tables

**Estimated Timeline to Production**: 3-4 weeks
