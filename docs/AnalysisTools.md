# Analysis Tools

This repository currently provides the following analysis entrypoints.

All analysis scripts take a *training results directory* (i.e. the output folder created by a training run) and generate PDF summaries and/or auxiliary artifacts in that same directory.

## Channel Tagging (CT)

### Comprehensive CT report
```bash
python3 channel_tagging/ana/comprehensive_ct_analysis.py <results_directory> -o <output_pdf>
```

Produces a PDF summary of classification performance (confusion matrix, per-class metrics, ROC curves, training history, and basic energy-dependent plots when available).

## Electron Direction (ED)

### Comprehensive ED report
```bash
python3 electron_direction/ana/comprehensive_ed_analysis.py <results_directory> -o <output_pdf>
```

Produces a PDF summary of regression performance (angular error, cosine similarity, training history, correlations, and energy-dependent performance).

Also writes:

- `cosine_energy_pdf.npz`
- `cosine_energy_pdf_visualization.png`

### ED interpretation report
```bash
python3 electron_direction/ana/cnn_feature_interpretation.py <results_directory> -o <output_pdf>
```

Produces a PDF report with additional diagnostic plots and summaries derived from the stored validation predictions.
