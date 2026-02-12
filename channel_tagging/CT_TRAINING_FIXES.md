# CT Training Configuration Fixes

## Issue Identified
The v72 models trained successfully but were saved to local `training_output/channel_tagging` instead of `/eos/`. This directory was not preserved after training completion, so the models were lost.

## Root Cause
- v72 JSON configs had: `"output_folder": "training_output/channel_tagging"`
- v77 successful jobs had: `"output_folder": "/eos/user/e/evilla/dune/sn-tps/neural_networks/channel_tagging/"`
- v72 submit files requested 100GB memory (excessive) vs v77 used 60GB

## Fixes Applied

### 1. Updated v72 JSON Configs
Fixed all v72 JSON files to save to /eos:
- `../channel_tagging/json/v72_deeper_10k.json`
- `../channel_tagging/json/v72_deeper_20k.json`
- `../channel_tagging/json/v72_deeper_50k.json`
- `../channel_tagging/json/v72_deeper_100k.json`

Changed: `training_output/channel_tagging` → `/eos/user/e/evilla/dune/sn-tps/neural_networks/channel_tagging`

### 2. Created Fixed v72 Submit File
- `condor/submit_ct_v72_deeper_20k_fixed.sub`
- Reduced memory from 100GB → 60GB (matching successful v77 config)
- Uses proper resource requests matching v77

### 3. Created v78 Configuration
New model for 10k batch training with 7 epoch reloads:

**Config:** `json/v78_dario_10k.json`
- Model: `ct_volume_v78_dario_10k`
- Max samples: 10,000 per class
- Batch samples: 10,000 (full reload each time)
- Reload every: 7 epochs (was 5 in v72)
- Saves to: `/eos/user/e/evilla/dune/sn-tps/neural_networks/channel_tagging/`

**Submit file:** `condor/submit_ct_v78_dario_10k.sub`
**Run script:** `condor/run_ct_v78_dario_10k.sh`

### 4. Submitted v78 Training
- Job ID: 9144815
- Status: Queued (waiting for GPU)
- Expected to address v77_dario_20k GPU memory errors by using 10k batches

## v77_dario_20k Failure Analysis
All v77_dario_20k jobs failed with:
```
tensorflow.python.framework.errors_impl.InternalError: 
Failed copying input tensor from CPU:0 to GPU:0: Dst tensor is not initialized
```

This indicates GPU memory/initialization issues. The v78 configuration with 10k batches should avoid this issue.

## Summary
- ✅ v72 configs fixed to save to /eos
- ✅ v72 submit files optimized (60GB memory like v77)
- ✅ v78 created with 10k batches + 7 epoch reload
- ✅ v78 job submitted (cluster 9144815)
- 📋 Can resubmit v72 jobs now that configs are fixed
