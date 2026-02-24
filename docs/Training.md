# Training & Analysis How-To

This repo contains tooling for two ML tasks:

- **Channel Tagging (CT)**: classify interaction type (ES vs CC, and variants)
- **Electron Direction (ED)**: regress the electron direction vector from 3-plane images

## Environment

Most scripts assume you have sourced the standard setup:

```bash
source scripts/init.sh
```

This sets up the external LCG environment (when available) and the local `PYTHONPATH`.

## Channel Tagging (CT)

### Local training (volume images, simple)

The simplest end-to-end trainer is:

```bash
python3 channel_tagging/models/train_ct_volume_simple.py \
  --json channel_tagging/json/volume_v42_corrected_100k.json \
  --plane X \
  --max-samples 1000
```

Notes:

- The script loads ES and CC `.npz` files from the directories in the JSON.
- Outputs are written below `output.base_dir` (as configured in the JSON), in a timestamped subfolder.

### Run the CT analysis app

Given a CT training output folder containing `results.json` and `test_predictions.npz`:

```bash
python3 channel_tagging/ana/comprehensive_ct_analysis.py /path/to/ct/output_dir
```

This writes a multi-page PDF report into the output directory (or use `-o` for a custom filename).

## Electron Direction (ED)

### Local training (3-plane matched, simple)

The simplest ED trainer is:

```bash
python3 electron_direction/models/train_three_plane_simple.py \
  --json electron_direction/json/three_plane_v50_10k.json
```

Outputs are written below `output.base_dir`, in a timestamped directory like:

`three_plane_<name>_<timestamp>/`

### Run the ED analysis apps

Comprehensive analysis (multi-page PDF):

```bash
python3 electron_direction/ana/comprehensive_ed_analysis.py /path/to/ed/output_dir
```

Feature/diagnostic report from predictions:

```bash
python3 electron_direction/ana/cnn_feature_interpretation.py /path/to/ed/output_dir
```

## HTCondor runs

Both tasks provide wrapper scripts under their `scripts/` folder, and `.sub` submit files under `condor/`.
Those are intended for grid/batch training; for a quick sanity check, prefer the local commands above.
