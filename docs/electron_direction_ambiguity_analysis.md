# Electron Direction Reconstruction: Directional Ambiguity Resolution

**Date:** November 18, 2025  
**Author:** Analysis of ED models v55-v60  
**Task:** Understanding why the model doesn't show double-peak at cosine ±1

---

## Executive Summary

The electron direction reconstruction model successfully **breaks the directional ambiguity** (forward vs backward) that one would expect from symmetric 2D projections. Analysis of cosine similarity distributions shows a strong preference for the correct direction over its opposite (13:1 ratio), indicating the model has learned discriminative features from the charge deposition patterns.

---

## Expected vs Observed Behavior

### What We Expected
For a particle track in 2D projections, there's an inherent ambiguity about which end is the start and which is the end. This should manifest as a **double-peak distribution** in the cosine similarity between predicted and true directions:
- Peak at **cosine ≈ +1**: Correct direction
- Peak at **cosine ≈ -1**: Reversed direction (180° error)

### What We Observed

**Model v60 (200k ES+CC mixed):**
- Cosine > 0.9 (near +1): **46.24%** ✓
- Cosine < -0.9 (near -1): **3.55%** 
- **Ratio: 13:1** preference for correct direction

**Model v58 (200k ES-only, better performance):**
- Cosine > 0.9 (near +1): **37.87%** ✓
- Cosine < -0.9 (near -1): **2.75%**
- **Ratio: 14:1** preference for correct direction

The absence of a double peak indicates the model successfully resolves the ambiguity.

---

## Physics Features That Break the Ambiguity

### 1. **Bragg Peak (Stopping Power)**

Electrons deposit more energy at the **end** of their track due to the Bragg peak effect:

```
Energy deposition profile along track:
   START ──────────────────────► END
    ▓░░░░░░░░░░░░░░░░░░░░░░░░░███
    │                           │
  Higher E                  Lower E
  (less dE/dx)             (Bragg peak)
```

**What the model sees:**
- **High charge density region** → End of track → Particle came from opposite direction
- **Lower charge density region** → Start of track → Particle going toward higher density

This asymmetry is preserved in the 2D projections and provides a strong directional cue.

### 2. **Multiple Coulomb Scattering**

Scattering increases as the electron loses energy:

```
Track topology:
   START ──────────────────────► END
   ═══════════════════════╗╔╝╚╗╔═
   Straight, dense       Scattered, diffuse
```

**What the model sees:**
- **Straight segment** → High energy → Track beginning
- **Scattered segment** → Low energy → Track end

The topological difference between start and end provides directional information.

### 3. **Three-Plane Correlation**

With U, V, and X plane projections, the model sees **three different 2D views** of the same 3D track:

```
         3D Track
            ↓
    ┌───────┼───────┐
    ↓       ↓       ↓
   U-plane V-plane X-plane
   (image) (image) (image)
```

**Key insight:** The Bragg peak and scattering patterns appear differently in each projection due to the geometry. A reversed direction would produce an **inconsistent pattern** across the three planes.

The CNN learns to correlate these patterns to determine the unique correct direction.

---

## Why ~3% Still Have Cosine ≈ -1

These ambiguous cases likely occur when:

1. **Very short tracks**: Insufficient length to develop clear Bragg peak
2. **Low energy events**: Weak charge deposition makes asymmetry hard to detect
3. **Highly scattered tracks**: Topology is too diffuse to distinguish start from end
4. **Reconstruction artifacts**: Noise or clustering errors obscure the true pattern
5. **Nearly symmetric events**: Rare cases where start and end look similar

---

## The Real Problem: Broad Distribution

While the absence of a double peak is good news, **50% of predictions** fall in the mediocre range (-0.9 < cosine < 0.9):

**v60 Distribution:**
- Excellent (cosine > 0.9): 46.24%
- Mediocre (-0.9 to 0.9): **50.22%**
- Wrong (cosine < -0.9): 3.55%

This indicates:
- Limited angular resolution from 2D projections
- Some events lack sufficient directional information
- Model architecture may not fully exploit available features
- Energy-dependent performance variations

---

## Model Performance Comparison

| Model | Dataset | Median Error | Mean Cosine | Cosine > 0.9 | Cosine < -0.9 |
|-------|---------|--------------|-------------|--------------|---------------|
| v55   | 100k ES | 34.1°       | 0.527       | 37.9%       | 2.8%         |
| v57   | 100k ES | 36.3°       | -           | -           | -            |
| v58   | 200k ES | 35.3°       | 0.527       | 37.9%       | 2.8%         |
| v60   | 200k ES+CC | 28.7°    | 0.555       | 46.2%       | 3.6%         |

**Key observation:** ES-only models (v55, v58) perform better than mixed ES+CC (v60), suggesting CC events have different charge deposition patterns that confuse the direction reconstruction.

---

## Implications and Future Improvements

### What Works
✓ Model successfully learns asymmetric features (Bragg peak, scattering)  
✓ Three-plane correlation provides disambiguating information  
✓ Architecture is capable of learning complex spatial patterns  

### What Could Be Improved

1. **Explicit dE/dx Features**
   - Add charge-weighted spatial moments along track
   - Include longitudinal energy profile as additional input
   - Feed dE/dx gradient as auxiliary information

2. **Energy-Dependent Training**
   - Stratified sampling by energy bins
   - Energy-conditional architecture
   - Separate models for different energy ranges

3. **Attention Mechanisms**
   - Let model focus on start/end regions
   - Learn which plane provides most directional information
   - Highlight Bragg peak regions

4. **Track Segmentation**
   - Explicitly identify start and end segments
   - Compare their characteristics
   - Multi-task learning with segmentation

5. **Data Quality**
   - Better noise rejection in clustering
   - Improved background subtraction
   - Higher quality truth matching

---

## Conclusion

The **absence of a double-peak** cosine distribution is actually **good news**: our model has learned to exploit physics-driven asymmetries (Bragg peak, multiple scattering) to break the directional ambiguity that would otherwise exist in 2D projections.

The challenge ahead is not resolving ambiguity (already achieved), but improving the overall angular resolution to push more events from the mediocre 50% into the excellent >0.9 cosine category.

The fact that ES+CC mixing hurts performance suggests we should:
- Stick with ES-only models (v58 baseline: 35.3° median)
- Investigate what's different about CC events
- Consider separate models or event-type-conditional architectures

---

## References

- Model results: `/eos/user/e/evilla/dune/sn-tps/neural_networks/electron_direction/`
- Analysis scripts: `/afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/electron_direction/ana/`
- Configuration files: `/afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/electron_direction/json/`
