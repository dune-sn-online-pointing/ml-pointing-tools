# CT v80: Burst-Sample Volume Retraining

## Motivation

The previous CT volume models (v52–v79) had three issues discovered in July 2026:

1. **Training data gone + condition mismatch**: v78/v79 trained on
   `prod_es`/`prod_cc` neighbourAPA volumes at clustering conditions
   `tick3_ch2_min2_tot3_e2p0`. Those datasets were removed from EOS. More
   importantly, the snop-pipeline runs CT inference on burst-sample volumes
   produced at `tick3_ch2_min2_tot3_e3p0` — a different energy threshold, so
   training and inference saw systematically different images.
2. **Preprocessing mismatch**: v79 trained with per-image max normalization,
   while `snop-pipeline/python/lib/channel_tagger.py` feeds **raw ADC** to the
   model. Per-image max normalization also throws away the absolute amplitude
   scale, which correlates with deposited energy (a real ES/CC discriminator).
3. **Split leakage**: v79 split train/val/test randomly across samples, so
   correlated volumes from the same burst/event could appear on both sides.

## What v80 does

- **Data**: `sn-burst-samples/cat*/cat*_volume_images_tick3_ch2_min2_tot3_e3p0/X`
  (EOS project space) — the same data product the pipeline sees at inference.
- **Cat-level splits**: train = cats 400–571, val = 572–596, test = 597–621.
  Cats 1–399 are deliberately left untouched so pipeline/scenario evaluations
  on them remain unbiased. If scenario evaluation uses all cats, restrict it
  to cats 1–399 when using a v80 model.
- **Preprocessing**: `log1p(image)`, no per-image normalization. Recorded in
  `results.json` under `preprocessing` so deployment code can reproduce it.
  **Deployment note**: `channel_tagger._preprocess_ct_images` in
  refactor-snop-pipeline must apply `np.log1p` for v80 models (read
  `results.json` next to the model, or hard-switch when deploying).
- **Aux variant** (`volume_v80_aux.json`): adds a second input branch with
  truth-free features available online: `n_clusters_in_volume`,
  `log1p(total ADC)`, `log1p(n nonzero pixels)`. CC events at SN energies
  produce de-excitation gammas → displaced blip clusters in the volume, so
  cluster multiplicity is physics-motivated. Standardization constants are
  saved in `results.json`.
- **Model**: 6 conv blocks with BatchNorm ([32,32,48,48,64,64]), GAP, dense
  [96,32], dropout 0.3, channel-axis flip augmentation, early stopping on
  val_accuracy, ReduceLROnPlateau.

## Labels

ES = 0, CC = 1 (unchanged from v52/v79; pipeline takes P(ES) = softmax[:, 0]).

## Running

```bash
# smoke test
python3 channel_tagging/models/train_ct_volume_v80.py -j channel_tagging/json/volume_v80.json --test-local

# condor
condor_submit channel_tagging/condor/submit_volume_v80.sub
condor_submit channel_tagging/condor/submit_volume_v80_aux.sub
```

Outputs go to `/eos/user/e/evilla/dune/sn-tps/neural_networks/channel_tagging/ct_volume_v80*_<timestamp>/`.

## Baseline to beat

v79 (2026-03-03, e2p0 volumes, random split): test accuracy 0.683,
ES recall 0.61, CC recall 0.75.
Note v80 test numbers are not directly comparable (different dataset and
stricter split); the honest comparison is running both models through the
snop-pipeline scenario analysis on held-out cats.
