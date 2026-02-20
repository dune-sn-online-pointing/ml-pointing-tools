# Electron Direction Model v58 - Architecture Details

## Model Information
- **Name**: three_plane_v58_200k_lr_schedule
- **Architecture**: Three-plane CNN
- **Location**: `/eos/user/e/evilla/dune/sn-tps/neural_networks/electron_direction/three_plane_three_plane_v58_200k_lr_schedule_20251118_110931/`
- **Training Date**: November 18, 2025

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
- **Dropout Rate**: 0.3

### Output
- **Output Dimension**: 3 (direction vector components: x, y, z)
- **Loss Function**: Angular loss

## Training Configuration

### Data
- **Dataset**: ES production clusters (tick3_ch2_min2_tot3_e2p0)
- **Total Samples**: 200,000
- **Use Matched**: Yes
- **Splits**:
  - Train: 70% (140,000 samples)
  - Validation: 15% (30,000 samples)
  - Test: 15% (30,000 samples)

### Optimizer & Learning Rate
- **Optimizer**: Adam
- **Initial Learning Rate**: 0.001
- **LR Schedule**: 
  - Reduce on plateau
  - Factor: 0.5
  - Patience: 10 epochs
  - Min LR: 1e-6
- **Gradient Clipping**: clipnorm = 0.4

### Training Parameters
- **Batch Size**: 64
- **Max Epochs**: 200
- **Early Stopping Patience**: 30 epochs
- **Actual Epochs Trained**: 12 (stopped early, likely due to NaN loss)

## Performance Metrics

### Angular Error (Validation Set)
- **Mean**: 50.98°
- **Median**: 35.34°
- **Std Dev**: 42.87°
- **25th Percentile**: 18.11°
- **75th Percentile**: 73.54°

### Training Progress
- **Final Training Loss**: 0.8395
- **Final Validation Loss**: 0.8898
- **Best Validation Loss**: 0.8898 (at epoch 12)

## Issues
- Training stopped at epoch 12 due to NaN loss (history shows NaN from epoch 13 onwards)
- This suggests potential training instability, possibly from:
  - Learning rate too high
  - Gradient explosion (despite clipnorm)
  - Batch size or data issues

## Comparison with v50
| Metric | v50 (100k) | v58 (200k) |
|--------|------------|------------|
| Median Error | 33.52° | 35.34° |
| 25th Percentile | 16.93° | 18.11° |
| 75th Percentile | 71.37° | 73.54° |
| Samples | 100,000 | 200,000 |
| Training Status | Complete | Failed (NaN at epoch 12) |

**Conclusion**: v58 performed slightly worse than v50 despite 2× more data, and training was unstable. v50 remains the better model.
