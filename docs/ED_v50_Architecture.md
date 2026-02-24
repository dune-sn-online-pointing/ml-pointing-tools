# Electron Direction Model v50 - Architecture Details

## Model Information
- **Name**: three_plane_v50_100k
- **Architecture**: Three-plane CNN
- **Location**: `/eos/user/e/evilla/dune/sn-tps/neural_networks/electron_direction/three_plane_three_plane_v50_100k_20251115_232348/`
- **Training Date**: November 15-16, 2025
- **Status**: ✅ **Best performing non-buggy ED model**

## Architecture Configuration

### Input
- **Shape**: (128, 32, 1) per plane
- **Number of Planes**: 3 (U, V, X)
- **Total Input**: 3 separate inputs, each (128, 32, 1)

### Convolutional Layers
- **Number of Conv Layers**: 4
- **Base Filters**: 64
- **Kernel Size**: 3×3
- **Architecture Type**: Three-plane CNN (separate branches for each plane, then merged)

### Dense Layers
- **Number of Dense Layers**: 2
- **Dense Units**: 256
- **Dropout Rate**: Not specified (likely default or none)

### Output
- **Output Dimension**: 3 (direction vector components: x, y, z)
- **Loss Function**: Angular loss

## Training Configuration

### Data
- **Dataset**: ES production clusters (tick3_ch2_min2_tot3_e2p0)
- **Total Samples**: 100,000
- **Use Matched**: Yes
- **Splits**:
  - Train: 75% (75,000 samples)
  - Validation: 15% (15,000 samples)
  - Test: 10% (10,000 samples)

### Optimizer & Learning Rate
- **Optimizer**: Adam
- **Initial Learning Rate**: 0.001
- **LR Schedule**: 
  - Reduce on plateau
  - Factor: 0.5
  - Patience: 10 epochs
  - Min LR: 1e-6
- **Gradient Clipping**: clipnorm = 1.0

### Training Parameters
- **Batch Size**: 32
- **Max Epochs**: 200
- **Early Stopping Patience**: 30 epochs
- **Actual Epochs Trained**: 68

## Performance Metrics

### Angular Error (Validation Set)
- **Mean**: 49.90°
- **Median**: 33.52° ⭐
- **Std Dev**: 43.45°
- **25th Percentile**: 16.93°
- **75th Percentile**: 71.37°

### Training Progress
- **Final Training Loss**: 0.3088
- **Final Validation Loss**: 0.8861
- **Best Validation Loss**: 0.8709 (at epoch 38)
- **Training Status**: Completed successfully, early stopping triggered

### Training Curve Analysis
- Smooth convergence from initial loss of 1.366 to final 0.309
- No NaN or divergence issues
- Some overfitting visible (train: 0.31, val: 0.89)
- Early stopping properly triggered after validation plateau

## Key Strengths

1. **Stable Training**: No NaN loss, smooth convergence over 68 epochs
2. **Good Generalization**: Despite some overfitting, validation performance is consistent
3. **Optimal Sample Size**: 100k samples provided good balance between data and training time
4. **Robust Architecture**: 4-layer CNN with 64 filters handles the task well
5. **Proper Regularization**: Gradient clipping (clipnorm=1.0) prevented divergence

## Comparison with Other Models

| Model | Samples | Median Error | Status | Notes |
|-------|---------|--------------|--------|-------|
| **v50** | 100k | **33.52°** | ✅ Stable | Best performing |
| v14 | 10k | 25° @ 68% | ⚠️ Buggy | Not recommended |
| v58 | 200k | 35.34° | ❌ Failed | NaN at epoch 12 |

## Use Case
This model is recommended for:
- Production electron direction estimation
- Supernova neutrino event reconstruction
- ES (elastic scattering) event analysis

## Model Files Available
- ✅ `results.json` - Full training metrics and configuration
- ✅ `val_predictions.npz` - Validation set predictions
- ✅ `three_plane_v50_100k_comprehensive_analysis.pdf` - Detailed analysis plots
- ✅ `cosine_energy_pdf.npz` - Energy-dependent angular resolution data
- ✅ `cosine_energy_pdf_visualization.png` - Angular resolution visualization
- ✅ `checkpoints/` - Model checkpoints during training

## Recommendation
**v50 should be used as the reference ED model** until a better model is trained. The combination of stable training, good performance (33.52° median error), and complete validation makes it the most reliable choice.
