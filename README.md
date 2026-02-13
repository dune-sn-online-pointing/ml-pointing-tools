# Machine Learning for Fast Online Supernova Pointing

This repository hosts neural network tooling for two tasks in DUNE SN-TPS:

- Channel Tagging (CT): classify interaction type (ES/CC/NC)
- Electron Direction (ED): regress electron direction

## Repository Structure

- channel_tagging/     CT training, configs, analysis, condor
- electron_direction/  ED training, configs, analysis, condor
- python/              Shared data loaders and utilities
- scripts/             Environment setup and helpers
- json/                Shared/legacy configs
- docs/                Documentation and summaries
- test/                Lightweight monitoring checks

## Environment Setup

Most scripts expect the standard environment from scripts/init.sh:

```bash
source scripts/init.sh
```

This sets PYTHONPATH and points to the LCG CUDA stack when available.

## Quick Start

Channel Tagging (local run):

```bash
./scripts/channel_tagging/run_channel_tagging.sh -j channel_tagging/json/production_training.json
```

Electron Direction (local run):

```bash
python3 electron_direction/models/train_three_plane_simple.py \
  -j electron_direction/json/three_plane_v50_10k.json
```

For job submission and model tracking, see docs/Networks.md and task-specific
condor wrappers under channel_tagging/condor and electron_direction/condor.
