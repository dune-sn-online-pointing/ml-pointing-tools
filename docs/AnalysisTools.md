# Analysis Tools - Active Scripts

## Current Analysis Workflow

### Comprehensive Analysis (Primary Tools)

These scripts generate complete PDF reports with all metrics, plots, and insights:

#### 1. Electron Direction Analysis
```bash
python electron_direction/ana/comprehensive_ed_analysis.py <results_directory>
```
**Generates**: `<model_name>_comprehensive_analysis.pdf` with:
- Angular error analysis (mean, median, quantiles)
- Training history
- Component correlations (x, y, z predictions)
- Energy-dependent performance
- Cosine similarity analysis
- Best/worst predictions
- Error vs true angle scatter plots

**Also generates**:
- `cosine_energy_pdf.npz`: 2D histogram (energy bins × cosine bins)
- `cosine_energy_pdf_visualization.png`: Heatmap visualization

#### 2. Channel Tagging Analysis
```bash
python channel_tagging/ana/comprehensive_ct_analysis.py <results_directory>
```
**Generates**: `<model_name>_comprehensive_analysis.pdf` with:
- Classification metrics per class
- Confusion matrices
- Training history
- Prediction distributions per class
- Energy-dependent performance
- Volume integration analysis (if applicable)

## Obsolete Scripts

All previous analysis scripts have been moved to `temp/obsolete_analysis/` and should not be used for new work.

---

## Best Practices

1. **Run comprehensive analysis after every training run** to generate the full PDF report
2. **Open PDFs in VS Code** for easy viewing: `code <path_to_pdf>`
3. **Include PDF reports in any results discussion** - they contain all necessary metrics
4. **Use cosine_energy_pdf.npz** for physics-level ED performance analysis
5. **Check `results.json`** for programmatic access to all metrics and history

---

## Active Analysis Files by Task

```
channel_tagging/ana/
└── comprehensive_ct_analysis.py          # ✅ Active

electron_direction/ana/
└── comprehensive_ed_analysis.py          # ✅ Active
```

Total: **2 active analysis scripts** (down from 22 scripts)
