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

### Recommended workflow

1. Validate JSON config before submission:

```bash
python3 scripts/validate_job_config.py channel_tagging/json/<your_config>.json
```

2. Run a tiny local smoke test first (recommended):

```bash
python3 channel_tagging/models/train_ct_volume_batch_reload.py \
  -j channel_tagging/json/<your_config>.json \
  --test-local
```

3. Submit to Condor:

```bash
condor_submit channel_tagging/condor/<your_submit_file>.sub
```

4. Monitor job status and logs:

```bash
condor_q <cluster_id>
tail -f channel_tagging/condor/logs/<job_name>_<cluster_id>.out
tail -f channel_tagging/condor/logs/<job_name>_<cluster_id>.err
```

### Memory and resource notes

- **CT volume trainings**: request **120GB** memory by default to avoid OOM holds.
- For current CT volume production samples, using **10k samples/class** is a safe baseline for stability.
- `request_gpus = 1` is typically required for practical training time.
- Keep `requirements = (OpSysAndVer =?= "AlmaLinux9")` unless you have a specific reason to change it.
- If a job is held for memory (`cgroup memory limit`), increase `request_memory` and resubmit.

### Minimal Condor submit template

```text
universe                = vanilla
executable              = /absolute/path/to/repo/channel_tagging/scripts/wrapper_ct_simple.sh
arguments               = -j /absolute/path/to/repo/channel_tagging/json/<config>.json

request_cpus            = 4
request_memory          = 120GB
request_gpus            = 1
request_disk            = 15GB

+JobFlavour             = "nextweek"
requirements            = (OpSysAndVer =?= "AlmaLinux9")

initialdir              = /absolute/path/to/repo
getenv                  = True

log                     = /absolute/path/to/repo/channel_tagging/condor/logs/<job_name>_$(Cluster).log
output                  = /absolute/path/to/repo/channel_tagging/condor/logs/<job_name>_$(Cluster).out
error                   = /absolute/path/to/repo/channel_tagging/condor/logs/<job_name>_$(Cluster).err

notification            = Error
queue 1
```

### Cleanup and resubmission (if needed)

```bash
condor_rm <cluster_id>
rm -f channel_tagging/condor/logs/<job_name>_<cluster_id>.*
```
