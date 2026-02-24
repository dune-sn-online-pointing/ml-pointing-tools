# Machine Learning for Fast Online Supernova Pointing

This repository contains neural-network tooling for two DUNE SN Online Pointing tasks:

- **Channel Tagging (CT)**: interaction-type classification (ES vs CC)
- **Electron Direction (ED)**: electron direction regression

## Core Concepts

- **Config-driven runs**: training entrypoints take a JSON config describing data locations, model parameters, and output settings.
- **Standardized outputs**: training produces a *results directory* containing metrics (e.g. `results.json`), predictions (`*.npz`), and training history (`*.csv`).
- **PDF reports**: analysis entrypoints consume a results directory and generate a compact PDF summary of performance.

## Setup

If you are using the CERN LCG environment, the standard setup is:

```bash
    source scripts/init.sh
```

For a pip-based setup, install the Python dependencies from:

```bash
    pip install -r python/requirements.txt --target ./local_packages
```

Installing under local_packages will allow to include these libraries in the PYTHONPATH through `scripts/init.sh`.

## Channel Tagging (CT)

**Training** (example entrypoint):

```bash
python3 channel_tagging/models/train_ct_volume_simple.py --json <ct_config.json> --plane X
```

**Analysis** (PDF report):

```bash
python3 channel_tagging/ana/comprehensive_ct_analysis.py <results_directory> -o <output_pdf>
```

## Electron Direction (ED)

**Training** (example entrypoint):

```bash
python3 electron_direction/models/train_three_plane_simple.py --json <ed_config.json>
```

**Analysis** (PDF report):

```bash
python3 electron_direction/ana/comprehensive_ed_analysis.py <results_directory> -o <output_pdf>
python3 electron_direction/ana/cnn_feature_interpretation.py <results_directory> -o <output_pdf>
```

## Tests

End-to-end smoke tests generate tiny local fixtures, run tiny CT/ED trainings, and then execute the main analysis apps:

```bash
bash test/testAllApps.sh
```

Additional documentation lives under [docs/](docs/).
