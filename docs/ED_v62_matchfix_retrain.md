# ED v62 / v63: retraining the three-plane electron-direction network on matcher-fixed products

*21 September 2026. Additive work: no existing model, product or report was modified or
deleted. Everything new lives under
`/eos/project-e/ep-nu/evilla/sn-online-pointing/prod_es_matchfix/` and
`.../neural-networks/electron_direction/three_plane_v6{2,3}_*`. The pointing pipeline
repository was not touched, and the pipeline was not run. Cats 1-399, 623-1224 and the 50
dev cats were never used as training data; the 50 dev cats appear here only as a held-out
evaluation sample.*

## 0. Answer in one paragraph

The ED deficit on the events the 3-plane matcher fix creates was **"never seen in
training"**, not "the information is not there". Retrained on matcher-fixed products, the
network gains **+0.12 in ⟨cos(reco,true)⟩ on exactly that population** (0.47 → 0.59 on the
production test split; 0.41 → 0.56 on the 50 dev burst cats) while the events that were
already there lose at most 0.01 (v62) or gain 0.05 (v63). On the independent burst cats the
ordering is *reversed* with respect to the diagnosis: the newly matched events, which ED
v58 reconstructed at ⟨cos⟩ = 0.409 against 0.516 for the old ones, are now reconstructed at
**0.564 against 0.539** — i.e. the split-induction population is no longer the bad
population, it is the *good* one, because it is high energy. The matcher fix should
therefore stop costing a degree of pointing and start paying: that has to be confirmed by
the pipeline, which is the next step and is deliberately not done here.

## 1. Why

`refactor-online-utils` now resolves the 3-plane partner ambiguity by keeping the most
energetic induction candidate instead of the first one in time
(`src/app/match_clusters.cpp`; the old rule survives behind `--first-in-time-partner`).
That raises the 3-plane match fraction of ES main tracks from 74.8% to 83.8% and adds ~12%
ES events per burst, but the deployed ED model **v58** was much worse on the events the fix
adds: ⟨cos⟩ 0.41 against 0.52, worse in every energy bin, and perfect-CT pointing on the 50
dev bursts went from 12.7 to 13.8 deg
(`/eos/project-e/ep-nu/evilla/sn-online-pointing/pipeline-dev/v80_allfix_matchfix_dev/ed_on_matchfix_diagnosis.md`).
The diagnosis could not separate "never seen in training" from "the information is not
there". Only a retrain can. This document is that measurement.

## 2. What had to be rebuilt, and why the rebuild is not a copy of the v58 pool

v58 (`three_plane_three_plane_v58_200k_lr_schedule_20251118_110931`) was trained on
`/eos/user/e/evilla/dune/sn-tps/prod_es/es_production_cluster_images_tick3_ch2_min2_tot3_e2p0/`
with `use_matched: true` (its `results.json`). **That directory no longer exists**: the
whole `prod_es` subtree of the user EOS area (96% full) was cleared, and with it the pool's
`tps_bg`, `clusters`, `matched_clusters` and `cluster_images`. What survives, in the
project area:

| product | path | files | size |
|---|---|---|---|
| tpstreams | `/eos/project-e/ep-nu/evilla/sn-online-pointing/prod_es/tpstreams` | 4326 | 5.5 GB |
| backtracked TPs `*_tps.root` (margin 10) | `.../prod_es/tps` | 4326 | 1.0 GB |
| `*_tps_bktr0.root` (margin 0, not used) | `.../prod_es/tps` | 1255 | - |
| background TPs | `.../bkgs/tps` | 340 | 141 MB |

So the chain had to restart at `add_backgrounds`, not at `match_clusters`. Two consequences,
both unavoidable and both to be kept in mind:

1. **The background realisation is new.** `add_backgrounds` seeds its background-file
   cursor from `std::random_device`, so `tps_bg` cannot be reproduced bit for bit. The
   rebuilt pool is statistically the same sample, not the same events with the same noise.
2. **v58's split is not recoverable.** Its trainer (`models/train_three_plane_simple.py`)
   shuffles with an unseeded global numpy RNG, stores no file provenance, and uses
   `train_split` 0.7 with the remaining 30% as "val" — there was no held-out test set at
   all. v62/v63 use a fixed-seed (42) **file-level** 0.70/0.15/0.15 split, reproducible and
   leak-free.

**Caveat that follows**: the v62 test split is not unseen data for v58 — v58 trained on the
same 4326 production files (different background overlay, different partner rule). The v58
numbers on that test set are therefore *optimistic*, which biases the comparison in v58's
favour. Section 7 (the 50 dev burst cats, a different, locally produced MC that none of the
three models ever trained on) removes that caveat.

Two of the 4326 source `_tps.root` files are 63-byte empty files in the original production
sample (`..._gen_001906_...2025-10-21T_095140Z`, `..._gen_000493_...2025-11-12T_133721Z`),
so the rebuild covers **4324/4326 files**. This is an upstream defect, not a job failure.

## 3. The rebuild

Configs (new files): `refactor-online-utils/json/prod_es_matchfix/es_production_matchfix.json`
and `es_production_legacyrule.json`. Clustering parameters copied verbatim from the recovered
`json/es_production.json` (`git show fbe622b^:json/es_production.json`): tick 3, channel 2,
min_tps 2, tot 3, **energy_cut 2.0** (the pool's "e2p0", not the burst cats' 3.0),
backtracker margin 10, time tolerance 10 ticks, spatial tolerance 10 cm.

```
sequence.sh -j json/prod_es_matchfix/es_production_matchfix.json --no-compile \
            -ab -mc -mm -gi --skip-files <s> --max-files <m>
```

40 chunks of 110 files, HTCondor cluster 15954673 (all 40 exited 0, ~50 min each), logs on
AFS in `refactor-online-utils/logs/prod_es_matchfix/`. The binaries were **not** rebuilt
(the 2026-09-06 build already contains the fix) and `--first-in-time-partner` was not
passed, so the default (fixed) rule is used. Volume images (`-gv`) were deliberately
skipped: the ED reads cluster images, and volumes would have cost ~13k inodes for nothing.

A second pass (cluster 15954855, 10 jobs) re-matches the **same clusters** with
`--first-in-time-partner` into `..._legacyrule/`. It exists only to label every sample
SAME / CHANGED / NEW / LOST; no images are generated from it.
`electron_direction/ana/make_matchclass_table.py` joins the two on `(event, cluster_id)` of
the X tree and writes `matchclass_table.npz`.

Completeness was checked against the tpstream basenames with
`refactor-online-utils/scripts/check_prodes_matchfix_complete.sh`:
tps_bg 4326/4326, clusters 4326/4326, matched (both rules) 4326/4326, images U/V/X
4324/4326 (the two empty sources).

### What the fix actually does to this pool

Over all 175 361 3-plane-matched main X clusters:

| class (training selection) | N | fraction | partners main under fix | under legacy |
|---|---|---|---|---|
| SAME (same U and V partner) | 137 303 | 78.30% | 0.954 | 0.954 |
| CHANGED (different partner) | 37 209 | 21.22% | 0.906 | 0.001 |
| NEW (matched only under the fix) | 696 | 0.40% | 0.911 | 0.000 |
| LOST (matched only under legacy) | 153 | 0.09% | 0.000 | 0.667 |

This is the key structural point, and it differs from the burst-cat picture of the
diagnosis. v58's training selection (section 4) does **not** require the U/V partners to be
main-track clusters, so a split-induction event was already in v58's training set — paired
with the *first fragment in time*. The fix does not add it, it **re-pairs** it with the
largest fragment. In the *deployed* pipeline selection, which does require main partners,
the same event is genuinely new:

| deployed (pipeline) selection | N |
|---|---|
| in the pipeline under the fixed rule | 165 396 |
| in the pipeline under the legacy rule | 131 174 (**the fix adds 26.1%**) |
| dep SAME / dep NEW / dep LOST | 131 050 / 34 346 / 124 |

So "CHANGED" here and "NEW" in the pipeline are the same physical population, and both
tables are reported below.

### Output tree (all new)

```
/eos/project-e/ep-nu/evilla/sn-online-pointing/prod_es_matchfix/
  tps_bg/                                                  4326 files   6.7 GB
  es_production_clusters_tick3_ch2_min2_tot3_e2p0/         4326        6.1 GB
  es_production_matched_clusters_..._e2p0_matchfix/        4326        1.5 GB
  es_production_matched_clusters_..._e2p0_legacyrule/      4326        1.5 GB
  es_production_cluster_images_..._e2p0_matchfix/{U,V,X}/ 12972       39   GB
  packed_v58sel_seed42/                                      13        8.1 GB
  matchclass_table.npz, burstcats_dev50_cache.npz             2       635  MB
  ed_v62_eval/                                                5        0.9 MB
                                                        ------------------------
                                                        30 296 files    64 GB
```

## 4. The selection, reproduced from what v58 actually used

v58's loader is `python/data_loader.py::load_three_plane_matched`: an X row is taken
whenever `match_id != -1` and the pair `(event, match_id)` also exists in U and V with
`match_id != -1`. `match_clusters` only ever assigns a match id to a **main** X cluster
(`if (!clusters_x[i].get_is_main_cluster()) continue;`), so the X side is implicitly the
main track — confirmed on v58's own `val_predictions.npz`, where 100% of rows have
`is_main_track == 1` and `is_es_interaction == 1`. The partners are **not** required to be
main tracks; the deployed pipeline loader
(`refactor-snop-pipeline/python/lib/sample_loader.py`) does require them, and
`partners_main` is stored per sample so both populations can be selected.

Preprocessing is v58's and the pipeline's: raw float32 ADC, channel dimension appended, no
normalisation, input order `[U, V, X]` (`ed_inference.py`).

**Validation of the reimplementation.** Recomputing v58 on cat000623's matchfix products
with this code reproduces the pipeline's stored `reco_directions.npz` exactly (286/286
events, dot product 1.0000), and over the 50 dev cats it reproduces the diagnosis to the
third digit: N = 13 197 (13 220 quoted), ⟨cos⟩ = 0.5045 (0.504), θ68 = 61.51 (61.5), with
per-class values SAME 0.516 / 60.43 and NEW 0.409 / 72.59 against the diagnosis's
0.516 / 60.4 and 0.409 / 72.6.

Pool size after selection: **175 208 samples from 4324 files**, split by source file into
train 122 566 (3026 files) / val 26 207 (648) / test 26 435 (650). v58's pool gave 162 750
samples, i.e. the rebuild is 7.7% larger.

## 5. Training configuration

| | v58 (deployed) | v62 | v63 |
|---|---|---|---|
| data | old-matcher production pool | **matchfix pool** | matchfix pool |
| init | scratch | scratch | **v58 weights** |
| architecture | three_plane_cnn, 4 conv x 64 (doubling), k3, 2 dense x 256, dropout 0.3, input (128,32,1) | same | same |
| loss | `angular_loss` | `safe_angular_loss` (eps 1e-6) | same |
| optimizer | Adam lr 1e-3, clipnorm 0.5 | same | Adam **lr 1e-4**, clipnorm 0.5 |
| batch / epochs | 64 / 200 | same | same |
| schedule | ES(30, restore best), RLROP(0.5, 10, min 1e-6) | same | same, min_lr 1e-7 |
| split | 0.7 / 0.3 "val", unseeded, no test | 0.70/0.15/0.15 by source file, seed 42 | same |
| outcome | NaN at epoch 13, deployed = epoch 12, val_loss 0.8898 rad | 61 epochs, best epoch 31, val_loss 0.8421 rad (48.2 deg), 40 min | 51 epochs, best epoch 21, val_loss 0.7893 rad (45.2 deg), 33 min |

Deliberate deviations, both forced:

* **`safe_angular_loss`** clips cos to [-1+eps, 1-eps] before `arccos`. v58's own run went
  NaN at epoch 13 (`results.json`: `loss` is NaN from index 12 on) and the deployed weights
  are the epoch-12 checkpoint; `d/dx arccos` diverges at |x| = 1 and the hard [-1, 1] clip
  does not prevent it. Away from the singularity the two losses are identical. Neither v62
  nor v63 hit a NaN.
* **a real test split** (section 2).

Note also that v58's trainer hard-coded `clipnorm=0.5` in `model.compile`, overriding the
0.4 in its json; v62/v63 use 0.5, i.e. what v58 actually ran.

**Models**

* v62 `…/neural-networks/electron_direction/three_plane_v62_matchfix_20260921_131047/best_model.keras`
* v63 `…/neural-networks/electron_direction/three_plane_v63_matchfix_ft58_20260921_132520/best_model.keras`

Each directory carries `config.json` (a copy of the input json), `results.json` (full
history, per-energy tables, environment), `training_history.csv`, `checkpoints/`,
`best_model.keras`, `val_predictions.npz`, `test_predictions.npz`,
`baseline_v58_test_predictions.npz` and the `*_classed.npz` variants, which add per event:
reco dir, true dir, true energy, per-plane cluster energies, `r = min(E_U,E_V)/E_X`,
`partners_main`, `match_class`, `matched_under_legacy_rule` and `in_pipeline_under_legacy`.

Code: `electron_direction/lib/prodes_matchfix_loader.py`,
`electron_direction/scripts/pack_prodes_pool.py`,
`electron_direction/models/train_three_plane_prodes_v62.py`,
`electron_direction/ana/{make_matchclass_table,eval_ed_matchfix,eval_ed_burstcats_matchfix,build_cosine_energy_pdf}.py`,
jsons `electron_direction/json/three_plane_v6{2,3}_prodes_matchfix*.json`,
condor `electron_direction/condor/submit_{pack_prodes,ed_v62_prodes,ed_v63_prodes,matchclass_table,burstcats_eval3}.sub`.

## 6. Results on the held-out test split of the rebuilt pool (26 435 events)

**All test events**

| | N | ⟨cos⟩ | median | Q68 |
|---|---|---|---|---|
| v58 | 26435 | 0.536 | 34.3 | 56.4 |
| v62 | 26435 | 0.553 | 30.4 | 52.8 |
| v63 | 26435 | 0.600 | 28.6 | 47.5 |

**Per true energy [MeV]** (⟨cos⟩ | median | Q68)

| E bin | N | v58 | v62 | v63 |
|---|---|---|---|---|
| 3-5 | 2519 | 0.349 \| 59.4 \| 81.2 | 0.346 \| 57.2 \| 80.6 | 0.408 \| 52.6 \| 73.5 |
| 5-10 | 6131 | 0.518 \| 40.6 \| 59.7 | 0.516 \| 39.3 \| 60.2 | 0.575 \| 35.6 \| 53.4 |
| 10-20 | 9248 | 0.581 \| 30.6 \| 49.2 | 0.590 \| 27.3 \| 44.9 | 0.633 \| 26.1 \| 41.1 |
| 20-30 | 4720 | 0.596 \| 25.3 \| 43.5 | 0.649 \| 20.1 \| 33.6 | 0.686 \| 20.0 \| 32.0 |
| 30+ | 2932 | 0.631 \| 22.9 \| 37.2 | 0.678 \| 17.9 \| 28.1 | 0.712 \| 17.8 \| 27.3 |

**By match class** — the measurement this retrain was for

| class | N | v58 | v62 | v63 |
|---|---|---|---|---|
| SAME | 20811 | 0.555 \| 34.0 \| 54.9 | 0.545 \| 32.1 \| 54.9 | 0.605 \| 29.3 \| 48.4 |
| CHANGED | 5503 | 0.466 \| 35.5 \| 63.6 | **0.585** \| 25.0 \| 42.9 | 0.584 \| 26.0 \| 43.5 |
| NEW | 121 | 0.405 \| 35.7 \| 81.7 | 0.480 \| 30.9 \| 58.1 | 0.512 \| 29.9 \| 56.1 |

**Deployed (pipeline) class** — CHANGED+NEW of the training selection is dep NEW here

| class | N | v58 | v62 | v63 |
|---|---|---|---|---|
| dep SAME | 19871 | 0.557 \| 34.1 \| 54.8 | 0.546 \| 32.3 \| 54.9 | 0.607 \| 29.5 \| 48.5 |
| dep NEW | 5129 | 0.473 \| 35.3 \| 62.8 | **0.591** \| 24.9 \| 42.3 | **0.590** \| 26.0 \| 43.3 |
| not in pipeline (partner non-main) | 1435 | 0.469 \| 33.7 \| 64.0 | 0.515 \| 27.9 \| 55.0 | 0.546 \| 26.2 \| 47.9 |

**dep NEW per true energy [MeV]**

| E bin | N | v58 | v62 | v63 |
|---|---|---|---|---|
| 3-5 | 19 | 0.419 \| 51.4 \| 76.8 | 0.384 \| 46.5 \| 72.1 | 0.514 \| 43.6 \| 65.0 |
| 5-10 | 293 | 0.368 \| 50.0 \| 80.2 | 0.472 \| 41.7 \| 67.6 | 0.474 \| 40.7 \| 68.0 |
| 10-20 | 1982 | 0.446 \| 41.0 \| 67.1 | 0.540 \| 30.0 \| 51.8 | 0.541 \| 30.9 \| 51.9 |
| 20-30 | 1529 | 0.484 \| 33.0 \| 59.0 | 0.627 \| 21.8 \| 35.9 | 0.625 \| 23.3 \| 37.5 |
| 30+ | 1300 | 0.530 \| 28.4 \| 48.6 | 0.660 \| 19.3 \| 29.5 | 0.653 \| 20.2 \| 32.2 |

**Paired Δ⟨cos⟩ against v58** (bootstrap 68% interval, 400 resamples)

| selection | N | v62 | v63 |
|---|---|---|---|
| all | 26435 | +0.017 [+0.014, +0.020] | +0.064 [+0.062, +0.067] |
| SAME | 20811 | **-0.010** [-0.014, -0.007] | +0.050 [+0.047, +0.052] |
| CHANGED | 5503 | **+0.119** [+0.110, +0.127] | +0.118 [+0.111, +0.125] |
| NEW | 121 | +0.075 [+0.012, +0.129] | +0.107 [+0.062, +0.149] |
| dep SAME | 19871 | -0.011 [-0.015, -0.008] | +0.050 [+0.047, +0.053] |
| dep NEW | 5129 | **+0.118** [+0.110, +0.125] | +0.117 [+0.109, +0.122] |

**Versus the truth-free quality variable r = min(E_U,E_V)/E_X**

| r | N | v58 | v62 | v63 |
|---|---|---|---|---|
| <0.3 | 3111 | 0.564 \| 26.7 \| 44.7 | 0.613 \| 22.6 \| 36.2 | 0.647 \| 21.9 \| 33.9 |
| 0.3-0.5 | 5130 | 0.573 \| 30.5 \| 51.2 | 0.607 \| 26.5 \| 44.0 | 0.648 \| 25.1 \| 40.9 |
| 0.5-0.7 | 5259 | 0.571 \| 32.3 \| 50.7 | 0.596 \| 27.9 \| 47.3 | 0.634 \| 27.0 \| 44.0 |
| 0.7-0.9 | 6254 | 0.544 \| 35.6 \| 57.4 | 0.546 \| 33.3 \| 54.6 | 0.603 \| 30.5 \| 49.3 |
| >=0.9 | 6681 | 0.460 \| 42.2 \| 67.6 | 0.457 \| 39.8 \| 67.6 | 0.513 \| 36.4 \| 60.7 |

r runs *backwards* here compared with the burst-fit study, because in this pool r is mostly
an energy proxy: CHANGED events have ⟨E_true⟩ = 23.7 MeV and ⟨r⟩ = 0.49, SAME events
13.8 MeV and 0.73. At fixed true energy 10-30 MeV the r dependence is weak for every model
(v58 0.571-0.614, v62 0.584-0.626, v63 0.637-0.676 across the five r bins), and the retrain
helps most at low r, i.e. exactly where the split-induction events sit. **After the retrain
r carries essentially no extra information about ED quality** — whether it still helps the
*burst likelihood* is a separate question for the pipeline step.

## 7. Independent cross-check: the 50 dev burst cats (623-672), 13 197 true-ES main tracks

Different MC (locally produced `cc_*`/`es_*` samples, not the `prodmarley` production),
e3p0 conditions, deployed pipeline selection, never training data for any of the three
models, and both product sets on disk so the diagnosis's SAME/NEW labels are reproduced
exactly. Table gives ⟨cos⟩ | Q68 [deg].

| selection | N | v58 | v62 | v63 |
|---|---|---|---|---|
| ALL | 13197 | 0.504 \| 61.5 | 0.541 \| 56.0 | **0.559 \| 53.8** |
| SAME | 11774 | 0.516 \| 60.4 | 0.539 \| 56.7 | **0.562 \| 53.8** |
| NEW | 1423 | 0.409 \| 72.6 | **0.564 \| 47.2** | 0.535 \| 53.6 |

per true energy [MeV]:

| bin | N | v58 SAME | v62 SAME | v63 SAME | v58 NEW | v62 NEW | v63 NEW |
|---|---|---|---|---|---|---|---|
| 3-5 | 1522 / 0 | 0.286 | 0.291 | 0.332 | - | - | - |
| 5-10 | 4477 / 107 | 0.503 | 0.519 | 0.550 | 0.353 | 0.433 | 0.469 |
| 10-20 | 4639 / 798 | 0.571 | 0.600 | 0.612 | 0.403 | 0.536 | 0.513 |
| 20-30 | 989 / 398 | 0.644 | 0.695 | 0.707 | 0.413 | 0.626 | 0.576 |
| 30+ | 147 / 120 | 0.701 | 0.699 | 0.743 | 0.488 | 0.658 | 0.599 |

**The NEW deficit is gone and inverted.** With v58, NEW events were 0.107 worse than SAME
(0.409 vs 0.516). With v62 they are 0.025 **better** (0.564 vs 0.539); with v63 0.027 worse
(0.535 vs 0.562) but still +0.126 above v58. Both retrained models are better than v58 on
SAME too, on this independent sample.

## 8. Verdict

1. **"Never seen in training" wins.** The information is there: the same architecture, the
   same preprocessing and the same number of parameters, trained on the matcher-fixed
   population, reconstruct those events at ⟨cos⟩ 0.56-0.59 instead of 0.41-0.47 — as well
   as, or better than, the events the old matcher already handled. Nothing about a split
   induction view is intrinsically unreconstructable; v58 had simply been taught the wrong
   pairing convention (the *first* induction fragment in time, not the biggest).
2. **The cost on the old population is small or negative.** v62 loses 0.010 ± 0.003 in
   ⟨cos⟩ on SAME events on the production test split and *gains* 0.023 on the independent
   burst cats; v63 gains 0.050 and 0.046 respectively. There is no trade-off to argue about.
3. **v63 (fine-tune from v58, lr 1e-4) is the better model overall**, v62 (from scratch) is
   the better model on the newly matched population specifically. On the independent burst
   cats: v63 ALL 0.559 vs v62 0.541; v62 NEW 0.564 vs v63 0.535. Part of v63's advantage is
   not the data at all — v58 only ever completed 12 epochs before going NaN, so v63 is also
   "v58 trained to convergence". The class-split tables separate the two effects: the
   population effect is the +0.118 on CHANGED/dep NEW, which v62 and v63 share to within
   0.001; the extra +0.05 that v63 has on SAME is the training effect.
4. **Do not deploy either model before the pipeline test.** ⟨cos⟩ per event is not θ68 per
   burst; the burst likelihood is energy-weighted and the pdf table below has to be rebuilt
   first.

## 9. Footprint, and what is prepared for the next step

* **Inodes**: 30 296 in `prod_es_matchfix/` + 51 in the two model directories = **30 347**
  new files. `/eos/project-e/ep-nu` went from 136 556 to 105 133 free inodes (the
  difference beyond 30 347 is another agent's); the 40 000 floor was never approached.
* **Disk**: 64 GB (products) + 4.3 GB (two model directories) = **68 GB**;
  3.22 TB still available.
* **Nothing in `/eos/user/e/evilla` was written.** Nothing existing was modified or deleted.
* **Kerberos** ticket valid to 2026-09-22 15:06, i.e. >24 h at every checkpoint.

**Prepared but deliberately not run** (the pipeline's sample loader is being changed by
another agent):

* `electron_direction/ana/build_cosine_energy_pdf.py` rebuilds
  `refactor-snop-pipeline/data/cosine_energy_pdf.npz` for a retrained model. It reproduces
  the v58 recipe exactly (18 reco-cluster-energy bins from 2 to 70 MeV, 100 cosine bins on
  [-1, 1], raw histogram, no smoothing — verified against the deployed table, which was
  built from v58's 48 814 held-out events). Run it on the **validation** split of whichever
  model is chosen, never on the test split used above:

  ```
  python3 electron_direction/ana/build_cosine_energy_pdf.py \
      --predictions <model_dir>/val_predictions.npz \
      --out <model_dir>/cosine_energy_pdf.npz
  ```

  The val split holds 26 207 events, comparable with the 48 814 behind the v58 table; if
  the low-statistics bins above 40 MeV turn out to be too thin, the train split can be
  predicted on as well (it is also disjoint from the test split).
* A **legacy-rule control training** (same recipe, images built with the old partner rule)
  was costed and skipped: it needs ~13 000 more inodes and ~1 h of image generation, and it
  is not needed for the verdict, since the v62-vs-v63 comparison already separates the
  population effect from the training effect. It remains the cleanest way to bound the
  residual if anyone wants it.

---

# Pipeline evaluation (added 22 September 2026)

*50 dev bursts, cats 623-672, samples `/eos/user/e/evilla/dune/sn-tps/mixture_dev_samples`
(read-only), six scenarios `json/six_scenarios_v80.json`, CT v80, uniform prior, and the
NEW default burst composition (`event_budget_mode="generated"`: first 330 generated ES +
3300 generated CC events). Outputs under
`/eos/project-e/ep-nu/evilla/sn-online-pointing/pipeline-dev/ed_retrain/`.*

## 10. Setup

Two minimal additive edits to the pipeline repo, both default-preserving:

* `test/run_small_sample_pipeline.sh`: the scenario analysis config no longer hardcodes the
  likelihood table; it takes `PDF_PATH`, defaulting to `${REPO_DIR}/data/cosine_energy_pdf.npz`.
* `condor/submit_cat_range.sh`: `PDF_PATH` is passed through the job `environment` line
  next to `PRODUCT_SUFFIX` and the rest.

Two new base configs (copies of `json/base_config_prior_uniform.json` with the ED model and
table swapped): `json/base_config_prior_uniform_ed_v62.json`,
`json/base_config_prior_uniform_ed_v63.json`. `burst_direction.py`, `channel_tagger.py`,
`sample_loader.py` and `scenario_cos_theta_report.py` were not touched.

### The likelihood tables

One table per model, built with `electron_direction/ana/build_cosine_energy_pdf.py` from the
**validation** split of the matchfix production pool (26 207 events, disjoint from the test
split of section 6), restricted to the **deployed selection** (main-track U and V partners,
24 755 events) because that is the population the pipeline actually fits. The v58 recipe is
reproduced exactly: 18 reco-cluster-energy bins 2-70 MeV, 100 cosine bins on [-1, 1], raw
histogram. A **reference table for v58 on the same events** was built too, so that "new pdf"
and "new model" can be separated.

The train split was deliberately **not** added despite the thin top bin: at the best epoch
v62's train loss is 0.573 rad against 0.842 on val (v63: 0.649 vs 0.789), so train-split
cosines describe a resolution the models do not have on new data. The cost is that the
50-70 MeV bin holds 127 events instead of the deployed table's 239; it carries 0.5% of a
burst, and the pipeline floors and renormalises the table anyway.

pdf-weighted ⟨cos⟩ per energy bin (a compact summary of the tables):

| E [MeV] | deployed v58 | v58 on matchfix val | v62 | v63 |
|---|---|---|---|---|
| 2-4 | 0.166 | 0.177 | 0.192 | 0.250 |
| 8-10 | 0.529 | 0.549 | 0.553 | 0.606 |
| 18-20 | 0.580 | 0.589 | 0.613 | 0.652 |
| 28-30 | 0.626 | 0.606 | 0.663 | 0.669 |
| 40-50 | 0.655 | 0.598 | 0.715 | 0.706 |

The deployed table and the v58-on-matchfix-val table agree to ~0.02 over most of the range,
i.e. the recipe itself introduces no bias; the v62/v63 tables are genuinely narrower.

### The six runs

| run | ED model | pdf table | products |
|---|---|---|---|
| R1 | v58 (deployed) | deployed | original |
| R2 | v58 (deployed) | deployed | **matchfix** |
| R3 | **v62** | v62 own | matchfix |
| R4 | **v63** | v63 own | matchfix |
| R5 | **v63** | v63 own | original |
| R6 | **v63** | deployed | matchfix |

Submitted two at a time (clusters 15954916/17, 15955105/06, 15955108/09), 25 jobs each,
2 cats/job, `REQUIRE_PRODUCTS=1`, `PRUNE_SCENARIO_OUTPUTS=2`, 10 GB, logs on AFS under
`condor/logs/ed_retrain/<run>/`, one `SUBMITTED_STATE` file per run. All six runs:
**50/50 reports, 0 jobs with a non-zero "unresolved" CT volume lookup (600 lookups in
total), 0 tracebacks**, and the products/model/table actually used were verified from the
written `config.json` and `scenario_analysis_config.json` of every cat.

## 11. Results

**theta68 [deg]** (arccos of the 0.32 quantile of the 50 per-burst cos to truth)

| scenario | R1 | R2 | R3 | R4 | R5 | R6 |
|---|---|---|---|---|---|---|
| 1 best case | 0.79 | 0.67 | 0.66 | 0.88 | 0.97 | 0.68 |
| 2 perfect CT | 12.18 | 13.78 | **9.58** | 10.27 | 9.75 | 11.13 |
| 3 full pipeline | 18.79 | 19.05 | 21.23 | 17.27 | 17.35 | **15.20** |
| 4 weighted CT | 22.55 | 24.08 | **18.36** | 19.54 | 19.00 | 19.41 |
| 5 perfect CT E>10 | 13.32 | 14.20 | **8.04** | 9.39 | 9.10 | 9.31 |
| 6 perfect CT E>5 | 12.10 | 13.54 | 9.59 | 10.08 | **9.01** | 10.39 |

**paired difference against R1 [68% bootstrap]**

| scenario | R2-R1 | R3-R1 | R4-R1 | R5-R1 | R6-R1 |
|---|---|---|---|---|---|
| 1 best case | -0.11 [-0.18,-0.06] | -0.13 [-0.20,-0.07] | +0.10 [+0.04,+0.20] | +0.18 [+0.11,+0.33] | -0.10 [-0.18,-0.06] |
| 2 perfect CT | **+1.60** [+0.71,+2.01] | **-2.60** [-4.16,-1.44] | -1.91 [-3.42,-0.68] | -2.43 [-3.78,-1.03] | -1.06 [-3.04,+0.08] |
| 3 full pipeline | +0.27 [-0.35,+1.17] | +2.44 [+0.37,+4.89] | -1.52 [-3.73,+0.63] | -1.44 [-3.22,+1.14] | **-3.59** [-5.31,-1.05] |
| 4 weighted CT | +1.53 [+0.06,+3.43] | -4.19 [-5.46,+0.77] | -3.01 [-4.58,-0.95] | -3.55 [-5.98,-1.95] | -3.14 [-4.78,-1.33] |
| 5 perfect CT E>10 | +0.88 [+0.22,+1.76] | **-5.29** [-6.40,-3.93] | -3.94 [-4.97,-2.51] | -4.22 [-5.12,-3.01] | -4.01 [-5.34,-2.62] |
| 6 perfect CT E>5 | +1.45 [+0.70,+1.96] | -2.51 [-4.16,-1.51] | -2.02 [-3.25,-0.66] | **-3.09** [-3.91,-1.19] | -1.70 [-3.19,-0.31] |

**mean n_selected per burst** — the matcher fix delivers the expected statistics

| scenario | R1/R5 (original) | R2/R3/R4/R6 (matchfix) | gain |
|---|---|---|---|
| 2 perfect CT | 196 | 219 | **+11.7%** |
| 3 full pipeline | 144 | 145 | +0.7% |
| 4 weighted CT | 2253 | 2916 | +29.4% |
| 5 perfect CT E>10 | 97 | 118 | +21.6% |
| 6 perfect CT E>5 | 171 | 194 | +13.5% |

### The three contrasts that actually separate the effects

Comparing R*x* against R1 confounds model, table and products. The clean one-variable
contrasts, all on the same 50 cats:

**(a) model, at fixed products and table (R6 - R2: v63 vs v58, both matchfix + deployed table)**

| scenario | v58 | v63 | difference |
|---|---|---|---|
| 2 perfect CT | 13.78 | 11.13 | **-2.65** [-4.31,-1.56] |
| 3 full pipeline | 19.05 | 15.20 | **-3.86** [-5.79,-1.22] |
| 4 weighted CT | 24.08 | 19.41 | -4.67 [-7.29,-2.37] |
| 5 perfect CT E>10 | 14.20 | 9.31 | **-4.89** [-6.63,-3.40] |
| 6 perfect CT E>5 | 13.54 | 10.39 | **-3.15** [-4.37,-1.71] |

The retrain is a large, unambiguous gain in every scenario, with intervals well clear of zero.

**(b) matcher fix, at fixed model and table (R4 - R5: matchfix vs original, both v63 + v63 table)**

| scenario | original | matchfix | difference |
|---|---|---|---|
| 2 perfect CT | 9.75 | 10.27 | +0.52 [-0.04,+0.95] |
| 3 full pipeline | 17.35 | 17.27 | -0.09 [-0.96,+0.16] |
| 5 perfect CT E>10 | 9.10 | 9.39 | +0.29 [-0.25,+0.91] |
| 6 perfect CT E>5 | 9.01 | 10.08 | +1.07 [+0.18,+1.20] |

**The matcher fix still does not pay in pointing, even with a model trained on its products.**
It is neutral to slightly negative. What the retrain did was remove most of the penalty: with
v58 the fix cost +1.60 / +1.45 deg (scenarios 2 / 6), with v63 it costs +0.52 / +1.07 and is
consistent with zero in scenarios 3 and 5. The +12% ES events are real (table above) but do
not convert into theta68 on 50 bursts.

**(c) the likelihood table, at fixed model and products (R4 - R6: v63's own table vs the deployed one)**

| scenario | deployed table | v63 table | difference |
|---|---|---|---|
| 2 perfect CT | 11.13 | 10.27 | -0.85 [-1.27,+0.21] |
| 3 full pipeline | 15.20 | 17.27 | **+2.07** [+0.50,+2.78] |
| 5 perfect CT E>10 | 9.31 | 9.39 | +0.07 [-0.20,+0.81] |
| 6 perfect CT E>5 | 10.39 | 10.08 | -0.31 [-0.89,+0.39] |

The model's own, narrower table helps slightly on pure-ES selections and **hurts the deployed
full-pipeline selection by 2 deg**. That selection is 59% CC: a table built on pure ES tells
the fit to trust each event more than it should when more than half of them are contaminants,
so the contaminants pull harder. The right fix is a table built on a CT-selected sample, not
v58's table with a v63 network.

### Coverage sanity check

theta68 across bursts divided by the median per-burst posterior q68 (1.0 = calibrated,
>1 = posterior too narrow):

| scenario | R1 | R2 | R3 (v62 pdf) | R4 (v63 pdf) | R5 (v63 pdf) | R6 (deployed pdf) |
|---|---|---|---|---|---|---|
| 2 perfect CT | 1.20 | 1.22 | **1.53** | 1.15 | 1.06 | 1.27 |
| 3 full pipeline | 1.07 | 1.07 | 1.28 | 1.14 | 1.14 | 1.08 |
| 5 perfect CT E>10 | 1.24 | 1.27 | 1.16 | 1.05 | 1.00 | 1.10 |
| 6 perfect CT E>5 | 1.21 | 1.25 | **1.52** | 1.17 | 1.01 | 1.22 |

The deployed configuration was already ~20% over-confident. **v63's table is the best
calibrated of all** (1.00-1.17 on pure-ES selections). **v62's table is clearly
mis-calibrated** (1.52-1.53): part of R3's attractive theta68 comes with a posterior that is
half as wide as the actual burst-to-burst scatter, and R3 is also the only run whose full
pipeline gets *worse* (+2.44, and the fraction of bursts beyond 30 deg goes 0.04 -> 0.10).
R3's numbers should not be quoted without this caveat.

## 12. Verdict on deployment

1. **Deploy v63.** At fixed products and table it improves every scenario by 2.7-4.9 deg,
   including the deployed full pipeline (19.05 -> 15.20). v62 is not the choice: its
   per-event numbers are competitive but its likelihood table is mis-calibrated and its full
   pipeline degrades.
2. **The matcher fix does not (yet) pay in pointing.** With v58 it cost +1.6 deg; with v63 it
   is neutral (-0.1 to +1.1 deg, three of four decisive scenarios consistent with zero) while
   delivering +12% ES events per burst. Keep it — it is the physically correct rule and it no
   longer costs anything real — but it should be justified by the statistics it adds, not by a
   pointing gain. The honest summary is: *the retrain pays, the matcher fix is now free.*
3. **Rebuild the likelihood table on a CT-selected sample before deploying v63.** Today the
   best full-pipeline number (15.20) comes from pairing v63 with v58's wider table, which is
   conceptually wrong even though it measures better; a table built on the contaminated
   population the pipeline actually fits should recover that gain honestly. Until then, the
   defensible configuration is v63 + the deployed table (R6).

## 13. Caveats

* **50 bursts.** theta68 is the 0.32 quantile of 50 numbers, i.e. a tail statistic. Bootstrap
  intervals are 1-2 deg wide and several rows show a large theta68 shift with a much smaller
  median paired shift (R3 scenario 3: +2.44 deg in theta68, +0.24 deg median paired). Anything
  below ~0.5 deg is noise; the emcee chain itself moves theta68 by ~0.2-0.3 deg between walker
  seeds.
* Scenario 1 (true directions) differs between runs by up to 0.2 deg purely through the
  likelihood table and the event composition; that is the floor of this comparison.
* The retrained models were trained on the *production* ES pool at e2p0 and are evaluated here
  on locally produced burst cats at e3p0 — the same mismatch v58 has always had, but the
  likelihood tables are built on the production pool and are applied to burst cats, which is
  the most likely source of the residual mis-calibration in section 11.
* The 50-70 MeV bin of the new tables holds 127 events (deployed table: 239).
* These are the 50 dev cats only. Nothing here was measured on cats 1-399 or 673-1224.

## 14. Footprint of the pipeline step

* Six runs x 50 cats, per-cat scenario directories tarred after each comparison:
  **1216 files, 526 MB** total (before packing it was ~17 400 files).
  Each cat keeps `catXXXXXX_scenarios.tar`, `scenario_cos_theta_report.json`,
  `scenario_cos_theta_report.pdf` and `scenario_analysis_config.json` plain.
* Comparison markdown/json in `ed_retrain/comparisons/` (9 comparisons).
* `/eos/project-e/ep-nu` free inodes after packing: **101 961**; 3.22 TB free.
* Nothing written to `/eos/user/e/evilla`; the burst samples there were read only.

---

# The likelihood table for the contaminated selection (added 22 September 2026)

*Offline only: no new pipeline runs, no code changes in the pipeline repo beyond three new
analysis scripts under `python/ana/` and new table files. The 50 dev bursts are re-fitted
from the per-event outputs the runs already kept.*

## 15. Why a mixture table

Section 11(c) found that the deployed full-pipeline selection prefers v58's *wider* table
to v63's own (+2.07 deg). That is an accident, not a result: the selection at
CT v80 >= 0.80, E > 5 MeV is only ~41% ES, so the density a single ES table describes is
not the density the fit is sampling. The correct per-event density is the mixture

    p(cos | E) = f(E) * pdf_ES,v63(cos | E) + (1 - f(E)) * pdf_CC(cos | E)

and because the pipeline reads any table with the deployed layout through `PDF_PATH`, a
mixture **table** deploys with zero code change.

## 16. f(E): the ES purity of the CT-selected sample

`python/ana/ed_mixture_purity.py` measures f(E) on the `v80_fixed_1000` campaign, whose
per-event CT scores, truth and energies survive in the per-cat slim tars, with the
generated-event budget (330 ES + 3300 CC) replayed offline by `budget_replay` — the
campaign itself was produced by the old all-events loader. **Cats 673-1224 only** (400
cats, 58 249 selected events); the dev bursts 623-672 are never used to build a table.

| E [MeV] | n_selected | f(E) = ES purity | error |
|---|---|---|---|
| 4-6 | 11693 | 0.480 | 0.005 |
| 6-8 | 22377 | 0.438 | 0.003 |
| 8-10 | 15971 | 0.371 | 0.004 |
| 10-12 | 6294 | 0.322 | 0.006 |
| 12-14 | 1557 | 0.274 | 0.011 |
| 14-16 | 296 | 0.230 | 0.024 |
| 16-18 | 50 | 0.200 | 0.057 |
| >18 | 11 | - | - |

**Overall purity 0.4099**, 145.6 selected events per burst — the runs of section 11 select
145, so the replay reproduces the deployed selection. Above ~18 MeV the CT v80 keeps
essentially nothing (the effect of `docs/CT_v80_energy_topology_study.md`), so bins with
fewer than 20 events take the nearest measured value; no fitted event lands there.

### Closure test on the dev bursts

Recomputed directly from the R6 per-event outputs (true-ES flag, reco direction, true
burst direction), for the same CT >= 0.80, E > 5 selection:

| E [MeV] | N_ES | ⟨cos⟩ of selected ES | v63 ES table says | N_CC | ⟨cos⟩ of selected CC |
|---|---|---|---|---|---|
| 4-6 | 710 | 0.484 | 0.481 | 747 | -0.021 |
| 6-8 | 1206 | 0.531 | 0.574 | 1594 | -0.015 |
| 8-10 | 790 | 0.556 | 0.606 | 1198 | -0.031 |
| 10-12 | 258 | 0.496 | 0.607 | 522 | +0.031 |
| 12-14 | 56 | 0.623 | 0.619 | 135 | +0.076 |

Three things follow, and they matter for the verdict:

1. **The flat-CC assumption is right.** Over all 4231 selected CC events ⟨cos to the true
   burst direction⟩ = **-0.012**; per bin it never exceeds 0.08. Reconstructed CC
   directions are anisotropic in the detector frame, not versus the neutrino.
2. **f(E) transfers.** The ES purity of the dev-burst selection is **0.4175** against the
   0.4099 measured on cats 673-1224 — 0.8% apart.
3. **The ES component is the weak part.** The v63 table overstates ⟨cos⟩ of the
   *CT-selected* ES events by 0.04-0.11 at 6-12 MeV. The table was built on the production
   pool's ES main tracks; the ES events that survive CT v80 are a harder, biased subset.

Tables built by `python/ana/ed_mixture_tables.py`, stored next to the v63 model as
`cosine_energy_pdf_mixture_{fE_flatcc,fE_ccmeas,global_flatcc,fE075_flatcc,fE125_flatcc}.npz`
(~19 kB each, deployed layout, each row already a proper density).

## 17. Variants, re-fitted on the 50 dev bursts

`python/ana/ed_mixture_eval.py` reads the rows out of the R6 per-cat tars, writes them into
a scratch run directory and calls the pipeline's own
`select_electrons_from_run` / `reconstruct_burst_direction` (clipped lookup, seeded,
grid-seeded init, uniform prior), so **only the table changes**. R4 and R6 were verified
bit-identical in metadata, CT scores and ED directions, so both gates apply to the same rows.

**Validation gates: T0 = 15.18 against R6's 15.20; T1 = 17.34 against R4's 17.27.** Both
inside the ~0.2-0.3 deg emcee noise floor. The offline chain is the production chain.

### Scenario 3 — the deployed selection (50 cats, 145 events/burst)

| table | theta68 | vs T0 [68% boot] | vs T1 [68% boot] | median | frac>30 | coverage |
|---|---|---|---|---|---|---|
| T0 deployed v58 table | **15.18** | - | -2.16 [-2.76,-0.64] | 10.61 | 0.02 | 1.08 |
| T1 v63 ES table | 17.34 | +2.16 [+0.66,+2.79] | - | 12.54 | 0.04 | 1.14 |
| T2 mixture f(E) + flat CC | 16.10 | +0.92 [+0.10,+3.53] | -1.24 [-1.58,+1.73] | 11.55 | 0.08 | **0.99** |
| T3 mixture f(E) + measured CC | 16.42 | +1.24 [-0.01,+2.73] | -0.92 [-1.58,+1.26] | 11.63 | 0.06 | 0.95 |
| T4 mixture global f=0.41 + flat CC | **15.53** | +0.34 [-0.29,+3.40] | -1.81 [-2.15,+1.80] | 12.12 | 0.06 | 0.96 |
| T5a mixture f(E) x 0.75 | 16.57 | +1.39 [+0.16,+3.91] | -0.77 [-1.52,+2.26] | 11.76 | 0.08 | 0.95 |
| T5b mixture f(E) x 1.25 | 16.39 | +1.21 [+0.17,+3.35] | -0.95 [-1.51,+1.44] | 12.08 | 0.08 | 1.01 |

* The mixture **removes most of the ES-only pathology**: +2.16 deg (T1) becomes +0.92 (T2)
  or +0.34 (T4).
* The mixture gives **the best-calibrated posterior of every table tried**: coverage
  0.95-1.01 against 1.08 for the deployed table and 1.14 for the ES-only one.
* T3 (measured CC) and T2 (flat CC) agree to 0.3 deg, as they must: the measured CC table
  is flat plus Poisson noise.
* **Getting the purity wrong by +-25% costs less than 0.5 deg** (T5a/T5b vs T2), so the
  construction is not fragile in f.
* But on 50 bursts the mixture does **not** beat the deployed-table accident: T4 is
  statistically indistinguishable from T0 (+0.34, interval spanning zero), T2 is +0.92 with
  a very wide interval. Only T1 is cleanly worse than T0.

The residual is explained by the closure test: with pdf_ES too narrow for the CT-selected
ES subset, a *lower* effective f partly compensates, which is why the single global
f = 0.41 beats the (correct, higher) f(E) = 0.48 in the 4-8 MeV bins where the bulk of the
sample sits. The honest reading is that f(E) is solid and **the ES component is what still
has to be rebuilt**.

### Scenario 2 — what the mixture does where it does not belong

| table | theta68 | vs T1 [68% boot] | coverage |
|---|---|---|---|
| T1 v63 ES table (correct here) | **10.27** | - | 1.15 |
| T0 deployed v58 table | 11.13 | +0.85 [-0.21,+1.27] | 1.27 |
| T2 mixture f(E) + flat CC | 11.87 | **+1.59** [+0.37,+1.69] | 1.21 |

On the pure-ES perfect-CT sample the mixture is **1.6 deg worse** than the ES table it
contains. A mixture table assumes 59% of the events are noise; on a clean sample that
throws away real information. **One table per selection** — never one table for everything.

### Scenario 4 — weighted CT (2916 events/burst)

| table | theta68 | vs T0 | coverage |
|---|---|---|---|
| T0 | 19.41 | - | 1.12 |
| T1 | 19.47 | +0.05 [-1.04,+1.60] | 1.15 |
| T2 | 19.78 | +0.36 [-1.04,+1.17] | 1.11 |

Table-insensitive: the weighted-CT scenario already down-weights each event by P(ES), so
it does its own contamination handling and the table choice is noise.

## 18. Recommendation for the deployable full-pipeline configuration

**ED v63 + `cosine_energy_pdf_mixture_global_flatcc.npz` for scenario-3-like selections,
and v63's plain ES table `cosine_energy_pdf.npz` for pure-ES selections.**

* It is the physically correct construction for a 41%-ES sample rather than a wider table
  that happens to fit, it is the best-calibrated option measured (coverage 0.96 against
  1.08), it matches the best central value within noise (15.53 vs 15.18), and it is
  insensitive to a +-25% error in the purity.
* `cosine_energy_pdf_mixture_fE_flatcc.npz` (the measured f(E)) is the more principled
  variant and is 0.6 deg behind on these 50 bursts; the two cannot be separated with this
  statistics. Prefer it over the global f once the ES component below is fixed, since then
  the compensation the global f is providing will no longer be needed.
* Do **not** put a mixture table on scenario 2/5/6 (+1.6 deg), and do not bother changing
  the table for scenario 4.

### The one thing still to do

Rebuild pdf_ES on the **CT-selected ES subset** rather than on the production pool: the
closure test shows this is the whole residual (0.04-0.11 in ⟨cos⟩ at 6-12 MeV). It needs
v63 ED predictions on CT-selected events from bursts that are not the dev bursts, i.e. one
pipeline pass over some of cats 673+ with the v63 config — the only step in this study that
cannot be done offline. Tuning the ES width on the dev bursts themselves would be fitting
the evaluation set and was deliberately not done.

## 19. Footprint of this step

* 5 mixture tables + f(E) npz + provenance next to the v63 model: **7 files, ~130 kB**.
* 25 condor jobs (cluster 15955114), 50 result npz + summaries in
  `ed_retrain/mixture_eval/`: **50 files, 247 kB**.
* Three new analysis scripts in `python/ana/`: `ed_mixture_purity.py`,
  `ed_mixture_tables.py`, `ed_mixture_eval.py`. No other pipeline file touched.
* `/eos/project-e/ep-nu` free inodes: **103 847**.

---

# The tables were measuring the wrong thing (added 27 September 2026)

*Offline only, on the r3 campaign's per-event outputs. Three new analysis scripts under
`python/ana/` (`ed_r3_tables.py`, `ed_r3_eval.py`, `ed_r3_report.py`) and new table files;
no pipeline runs, no other pipeline file touched. Tables are built on the training slice
**cats 673-900** and evaluated on cats the tables never saw. Cats 400-621 are absent from the
r3 campaign, so nothing under them was read.*

## 20. The defect: resolution where the likelihood wants resolution ⊗ kinematics

Every table in use so far — `data/cosine_energy_pdf.npz`, v62's, v63's — was built from
`cos(reco electron direction, TRUE ELECTRON direction)`, i.e. the ED **angular resolution**
(`build_cosine_energy_pdf.py` reads `cos_reco_true`). But the burst likelihood evaluates

    cos_i = selected_dirs @ n        (`_pdf_likelihood`, n = trial BURST direction)

so the density it needs is the resolution **convolved with the ES kinematic spread**
cos(true electron, nu). A resolution-only table therefore tells the fit that every event is
more informative than it is. Two symptoms, both previously unexplained:

* the section-16 closure test, where the v63 table's ⟨cos⟩ sat 0.04-0.11 above what the
  CT-selected ES events actually deliver;
* scenario 1, which is fed **true** directions — there the resolution is a delta function and
  only kinematics remain, so a resolution table is maximally wrong. That is the +0.2 deg the
  r3 campaign saw in its best case.

Both are fixed by measuring cos against the true burst direction.

## 21. New tables (training slice, cats 673-900, 227 cats)

| table | file | content |
|---|---|---|
| pdf_ES,sel | `cosine_energy_pdf_es_ctselected_r3.npz` | cos(v63 reco dir, TRUE BURST dir) of **true-ES events passing CT v80 >= 0.80, E > 5 MeV** (13 619 events) |
| mixture, global f | `cosine_energy_pdf_mixture_ctsel_global_flatcc.npz` | f·pdf_ES,sel + (1-f)·flat, f = 0.4083 |
| mixture, f(E) | `cosine_energy_pdf_mixture_ctsel_fE_flatcc.npz` | f(E)·pdf_ES,sel + (1-f(E))·flat |
| kinematic | `cosine_energy_pdf_kinematic_r3.npz` | cos(TRUE electron dir, TRUE nu dir), true-ES, E > 3 MeV (49 905 events) |
| purity | `purity_fE_ct080_e5_r3.npz` | f(E) re-measured at r3 |

All in the deployed layout (18 reco-cluster-energy bins 2-70 MeV, 100 cos bins, raw
histogram). The energy axis is the **reco cluster energy** (metadata col 10) for every table,
because that is what `select_electrons_from_run` passes to the lookup; the kinematic table
also carries a true-energy-binned copy as `pdf_2d_true_energy_axis` for reference. Rows with
fewer than 50 events are filled from the nearest measured row — an empty raw row would be
floored and renormalised to *flat*, i.e. "this energy carries no information", which is wrong;
there are simply no CT-selected events above ~18 MeV to measure.

### Purity at r3 is unchanged by the +12% ES events

| | cats | selected/burst | overall f |
|---|---|---|---|
| earlier (v80 campaign, budget replayed) | 673-1224, 400 | 145.6 | **0.4099** |
| r3 (new matcher, native 330+3300) | 673-900, 227 | 146.9 | **0.4083** |

f(E) at r3: 0.487 (4-6), 0.432 (6-8), 0.369 (8-10), 0.322 (10-12), 0.281 (12-14), 0.267
(14-16), 0.194 (16-18) — within errors of the earlier measurement. The matcher fix adds ES
*and* CC events to the CT-selected sample in the same proportion, so the mixture weight does
not move.

Flat CC confirmed a third time: ⟨cos to burst⟩ of the 19 733 selected CC events = **-0.019**.

### ES component: table vs reality (as in section 16)

| E [MeV] | N_ES,sel | v63 table (resolution) | pdf_ES,sel | actual |
|---|---|---|---|---|
| 4-6 | 3230 | 0.481 | **0.473** | 0.473 |
| 6-8 | 5491 | 0.574 | **0.527** | 0.528 |
| 8-10 | 3402 | 0.606 | **0.552** | 0.552 |
| 10-12 | 1186 | 0.607 | **0.522** | 0.522 |
| 12-14 | 256 | 0.619 | **0.532** | 0.533 |
| 14-16 | 46 | 0.649 | 0.532 (filled) | 0.514 |

pdf_ES,sel reproduces the truth by construction; the old table was over-confident by
0.05-0.14 and increasingly so with energy.

### The kinematic density

⟨cos(true e, true nu)⟩ rises from 0.914 (2-4 MeV) through 0.964 (6-8) to 0.990 (>16 MeV):
ES is forward-peaked but not a delta, and that residual spread is the whole information
content of scenario 1.

## 22. Validation gates

Re-fitting with the table the campaign itself used must reproduce the campaign's numbers.

| scenario | gate table | offline theta68 | r3 reports | cats bit-exact | n_selected match |
|---|---|---|---|---|---|
| 3 full pipeline | TG = `mixture_global_flatcc` | 15.77 | 15.83 | 566/722 | 722/722 |
| 1 best case | Tv63 = `cosine_energy_pdf.npz` | 0.99 | 0.99 | 857/1000 | 1000/1000 |

Median per-cat |cos - report| = 0 in both; the non-exact tail is the known emcee chaos
(max 0.58 deg on one scenario-3 cat). **Both gates pass.**

Note for anyone auditing the r3 campaign: the per-cat `scenario_analysis_config.json` on disk
names the v63 table for *all* scenarios, but the fits prove scenario 3 was run with the
mixture (per-cat cos reproduced to 1e-6 only by the mixture). The report stage was evidently
run per scenario and that file holds only the last write.

## 23. Scenario 3 — the full pipeline, on 722 evaluation cats (1-399 and 901-1224)

theta68 [deg]:

| selection | Ncats | T0 deployed v58 | TG r3 mixture (v63-ES) | **TSg mixture (pdf_ES,sel, global f)** | TSfE mixture (pdf_ES,sel, f(E)) |
|---|---|---|---|---|---|
| ALL | 722 | 14.97 | 15.77 | **14.51** | 15.23 |
| 1-399 (old era) | 398 | 16.06 | 16.50 | **15.05** | 15.69 |
| 901-1224 (new era) | 324 | 13.94 | 14.74 | 14.08 | 14.24 |

paired difference against TG (what r3 deployed):

| selection | T0 | **TSg** | TSfE |
|---|---|---|---|
| ALL | -0.80 [-1.24,-0.35] | **-1.26 [-1.46,-0.82]** | -0.54 [-0.95,-0.32] |
| 1-399 | -0.43 [-1.25,-0.14] | **-1.45 [-1.89,-0.96]** | -0.80 [-1.28,-0.33] |
| 901-1224 | -0.81 [-1.46,-0.39] | -0.66 [-1.16,-0.32] | -0.50 [-0.89,-0.00] |

| table | theta68 | median | frac>30 | coverage |
|---|---|---|---|---|
| T0 deployed v58 | 14.97 | 11.69 | 0.03 | **0.98** |
| TG r3 mixture (v63-ES) | 15.77 | 12.41 | 0.06 | 0.94 |
| TSg mixture (pdf_ES,sel, global f) | **14.51** | 11.71 | 0.04 | 0.86 |
| TSfE mixture (pdf_ES,sel, f(E)) | 15.23 | 11.87 | 0.04 | 0.89 |

* **The corrected ES component wins**: -1.26 deg against what r3 deployed, and it also beats
  the v58-table accident (14.51 vs 14.97). For the first time the principled table is also the
  best-performing one.
* The gain is carried by the old era (-1.45); in the new era TSg and T0 are equal within noise.
* **Global f still beats f(E)** (14.51 vs 15.23), as in section 17, and now the compensation
  story cannot be the explanation since the ES component is correct. With the ES component
  fixed, f(E) rising to 0.49 at 4-6 MeV makes the low-energy bulk *more* confident than a flat
  0.41 does, and on this sample that is the wrong direction. I have no deeper explanation and
  the interval separating them ([-1.46,-0.82] vs [-0.95,-0.32]) is not wide enough to call it
  noise; it should be revisited if the fit or the selection changes.
* Coverage moves the other way: T0 is nearly perfect (0.98) while TSg is 14% **over**-covering
  (0.86). A conservative posterior is the safer failure mode, but it means TSg's per-burst
  error bars are now slightly too wide.

## 24. Scenario 1 — best case, TRUE directions, all 1000 cats

| selection | Ncats | T0 deployed v58 | Tv63 (r3 default) | **Tkin kinematic** |
|---|---|---|---|---|
| ALL | 1000 | 0.75 | 0.99 | **0.52** |
| 1-399 | 399 | 0.78 | 1.00 | 0.54 |
| 623-672 (dev) | 50 | 0.65 | 0.91 | 0.47 |
| 673-900 (slice) | 227 | 0.73 | 0.94 | 0.51 |
| 901-1224 | 324 | 0.75 | 1.02 | 0.52 |

paired vs Tv63: T0 **-0.24** [-0.25,-0.22]; Tkin **-0.47** [-0.49,-0.45]. Per era identical
to within 0.05 deg.

| table | theta68 | median | coverage |
|---|---|---|---|
| T0 deployed v58 | 0.75 | 0.56 | 0.37 |
| Tv63 (r3 default) | 0.99 | 0.77 | 0.49 |
| Tkin kinematic | **0.52** | **0.39** | **0.60** |

The kinematic table nearly **halves** scenario 1 (0.99 -> 0.52 deg) and has the least bad
coverage of the three. This removes the r3 campaign's "+0.2 deg worse best case" and then
some: against the b330 reference's table (T0 = deployed v58) the kinematic table is 0.75 ->
0.52, i.e. the best case *improves* by 0.23 deg instead of degrading. All scenario-1
coverages are far below 1 because the per-burst posterior width is set by the likelihood
shape while the burst-to-burst scatter is sub-degree; the kinematic table is the closest to
honest.

## 25. Recommended final table set

| scenario | table | measured gain |
|---|---|---|
| 1 best case (true dirs) | `cosine_energy_pdf_kinematic_r3.npz` | **-0.47 deg** vs r3 (0.99 -> 0.52) |
| 3 full pipeline (CT-selected) | `cosine_energy_pdf_mixture_ctsel_global_flatcc.npz` | **-1.26 deg** vs r3 (15.77 -> 14.51) |
| 2, 5, 6 (true ES, reco dirs) | left as v63 `cosine_energy_pdf.npz` | not evaluated |
| 4 weighted CT | v63 `cosine_energy_pdf.npz` | table-insensitive (section 17) |

Both changes are table swaps through `PDF_PATH`; no code change.

### Flag: scenarios 2/5/6 are very probably improvable too, and were not tested

They were left alone on the grounds that they fit true-ES events not filtered by CT, so the
v63 table is right for them. That reasoning covers the *selection* but not the *observable*:
the v63 table is still cos(reco, **true electron**) while those fits also query
cos(reco, **burst**). The correct table for them is pdf built as cos(reco dir, true burst dir)
on true-ES events with the matching energy cut and no CT filter — the same correction that
bought -1.26 deg in scenario 3 and -0.47 deg in scenario 1. Scenarios 2/5/6 are the
perfect-CT headline numbers, so this is the highest-value remaining item. It is cheap: the
table comes from the same scenario_2/5/6 per-event outputs on cats 673-900, and the
evaluation is one more table in `ed_r3_eval.py`. I did not run it because the instruction was
explicit; say the word.

## 26. Footprint

* 6 new table/purity/provenance files next to the v63 model: **124 kB**.
* 64 condor jobs (cluster 15968773) -> 129 result files, **591 kB** in
  `pipeline-dev/ed_retrain/r3_tables/`.
* `/eos/project-e/ep-nu` free inodes: **98 692**. Nothing written to the user area.

---

# Burst-axis tables for the perfect-CT scenarios (added 28 September 2026)

*Offline, r3 per-event outputs. One new script, `python/ana/ed_r3_burstaxis.py`
(+ scenario 5 added to `ed_r3_eval.py` and the scenario blocks to `ed_r3_report.py`, both mine).
Tables built on the training slice cats 673-900; evaluated on 2-399 and 901-1224 with the
50 dev cats reported separately.*

## 27. One table, not three

Scenarios 2, 5 and 6 differ **only** in the energy cut (E > 3 / 10 / 5 MeV): their per-event
outputs are bit-identical (verified on cat000700 and cat000905 — metadata and reco directions
both `array_equal`). So one pass over the scenario-2 outputs with three cuts gives all three
tables, and the question is whether the cut changes the table at all.

Measured row by row (⟨cos⟩ per energy row of the three tables):

| E [MeV] | e3 | e5 | e10 | max spread |
|---|---|---|---|---|
| 2-4 | 0.252 (N=2291) | 0.479 (filled) | 0.592 (filled) | 0.340 |
| 4-6 | 0.436 (N=7010) | 0.479 (N=3630) | 0.592 (filled) | 0.156 |
| 6-8 | 0.534 (N=7148) | 0.534 (N=7148) | 0.592 (filled) | 0.059 |
| 8-10 | 0.568 (N=6795) | 0.568 (N=6795) | 0.592 (filled) | 0.025 |
| 10-12 and above | identical | identical | identical | **0.000** |

Every row at or above the cut is cut-independent; only the row the cut straddles differs
(4-6 for E > 5), plus rows below it which are extrapolated fills. Crucially the bilinear
energy lookup **reaches one bin below** the event's bin — an event at 10.5 MeV interpolates
between the 8-10 and 10-12 row centres — so the rows just below the cut *are* queried, and the
loosest table is the only one that has real data there.

The fits confirm it: **the single E > 3 table is as good as or better than the cut-matched one
in every scenario**. So the recommendation is one file,
`cosine_energy_pdf_burstaxis_r3_e3.npz`, for scenarios 2, 5 and 6. (The e5 and e10 variants
are kept next to it for the record.)

## 28. Results (772 cats: 398 old era, 324 new era, 50 dev reported separately)

Gate: with v63's own table the offline fit reproduces the r3 reports —
max|cos − report| = 3.7e-04 (sc 2), 2.5e-04 (sc 6), 7.2e-04 (sc 5), n_selected matching for
every cat. theta68 [deg]:

**Scenario 2 (perfect CT, E > 3)**

| selection | Ncats | Tv63 (r3 default) | BA burst-axis | d vs Tv63 [68% boot] |
|---|---|---|---|---|
| ALL | 772 | 9.14 | **8.68** | **-0.46 [-0.62, -0.28]** |
| 1-399 old era | 398 | 9.08 | 8.69 | -0.39 [-0.65, -0.22] |
| 901-1224 new era | 324 | 9.08 | 8.39 | -0.69 [-0.93, -0.25] |
| 623-672 dev | 50 | 10.83 | 10.39 | -0.44 [-1.07, +0.67] |

median 6.87 -> 6.60; frac>30 0.00 both; coverage 1.16 -> 1.13.

**Scenario 6 (perfect CT, E > 5)**

| selection | Ncats | Tv63 | BA (E>5) | BAe3 (E>3) | d BAe3 vs Tv63 |
|---|---|---|---|---|---|
| ALL | 772 | 8.93 | 8.53 | **8.43** | **-0.50 [-0.66, -0.37]** |
| 1-399 | 398 | 8.88 | 8.42 | 8.33 | -0.55 [-0.76, -0.38] |
| 901-1224 | 324 | 8.80 | 8.37 | 8.22 | -0.58 [-0.76, -0.18] |
| 623-672 dev | 50 | 10.30 | 10.04 | 10.06 | -0.24 [-1.20, +0.34] |

median 6.72 -> 6.38; frac>30 0.00; coverage 1.17 -> 1.14.

**Scenario 5 (perfect CT, E > 10)**

| selection | Ncats | Tv63 | BA (E>10) | BAe3 (E>3) | d BAe3 vs Tv63 |
|---|---|---|---|---|---|
| ALL | 772 | 8.36 | 8.20 | **8.09** | **-0.28 [-0.44, -0.07]** |
| 1-399 | 398 | 8.41 | 8.15 | 8.02 | -0.40 [-0.62, -0.22] |
| 901-1224 | 324 | 8.08 | 7.76 | 7.76 | -0.32 [-0.44, +0.16] |
| 623-672 dev | 50 | 9.82 | 9.65 | 9.59 | -0.23 [-0.68, +0.19] |

median 6.39 -> 6.25; frac>30 0.00; coverage 1.12 -> 1.08.

Every scenario improves, coverage improves slightly everywhere (the tables were
over-confident, now marginally less so), and the gains are smaller than in scenario 3
(-1.26) or scenario 1 (-0.47) because these selections are pure ES and the convolution is the
only thing that was missing. The 50 dev cats are systematically ~1.3 deg worse than either
era in all three scenarios — they are a slightly unlucky subset, which is worth remembering
when reading earlier dev-burst-only numbers.

## 29. Final recommended table set, all six scenarios

| scenario | table | measured gain vs r3 |
|---|---|---|
| 1 best case (true dirs) | `cosine_energy_pdf_kinematic_r3.npz` | **-0.47** [-0.49,-0.45] |
| 2 perfect CT (E>3) | `cosine_energy_pdf_burstaxis_r3_e3.npz` | **-0.46** [-0.62,-0.28] |
| 3 full pipeline | `cosine_energy_pdf_mixture_ctsel_global_flatcc.npz` | **-1.26** [-1.46,-0.82] |
| 4 weighted CT | v63 `cosine_energy_pdf.npz` (unchanged) | table-insensitive (section 17) |
| 5 perfect CT (E>10) | `cosine_energy_pdf_burstaxis_r3_e3.npz` | **-0.28** [-0.44,-0.07] |
| 6 perfect CT (E>5) | `cosine_energy_pdf_burstaxis_r3_e3.npz` | **-0.50** [-0.66,-0.37] |

Three files cover all six scenarios; every change is a `PDF_PATH` swap, no code change.
Scenario 4 is left alone because it measured table-insensitive, not because it is right: its
correct density would be a P(ES)-weighted mixture, which was never tested — the honest
statement is "no measured reason to change it".

## 30. What the deployed table should have been, and what it cost historically

`data/cosine_energy_pdf.npz` should have been the density of **cos(reco electron, BURST
direction)** for the population each scenario fits, not cos(reco electron, true electron).
Rebuilding it the right way from the r2-era campaign's own per-event outputs
(`v80_fixed_1000`, ED v58, cats 673-900, 54 114 true-ES events) shows the historical damage
was small, and why:

| E [MeV] | deployed (v58 resolution) | v58 burst-axis (correct) | v63 resolution | v63 burst-axis |
|---|---|---|---|---|
| 4-6 | 0.410 | 0.385 | 0.481 | 0.436 |
| 6-8 | 0.502 | 0.491 | 0.574 | 0.534 |
| 8-10 | 0.529 | 0.528 | 0.606 | 0.568 |
| 10-12 | 0.545 | 0.565 | 0.607 | 0.592 |
| 16-18 | 0.569 | 0.618 | 0.674 | 0.627 |
| 24-26 | 0.610 | 0.673 | 0.708 | 0.680 |

For **v58 the two agree to -0.05…+0.03** in ⟨cos⟩, and where they differ the deployed table
is if anything slightly *under*-confident at high energy. The reason is that v58's angular
resolution (⟨cos⟩ ≈ 0.5-0.6) is far broader than the ES kinematic kernel
(⟨cos(true e, nu)⟩ = 0.91 at 2-4 MeV rising to 0.99 above 16 MeV): convolving a broad kernel
with a narrow one barely changes it. The convention error was therefore numerically almost
invisible in the r2/b330 era and only became material once the ED improved — for v63 the same
comparison differs by 0.04-0.14. **That is why nobody caught it, and it is a warning: every
future ED improvement makes this table convention matter more.**

How much the historical numbers were affected, as far as the kept outputs allow:

* **Scenario 1 is the exception and can be stated exactly.** It is fed true directions, so the
  fit depends on the table *only* — no ED model enters. Measured on r3 rows, the deployed v58
  table gives theta68 = 0.75 deg where the kinematic table gives 0.52. So the historical
  best-case numbers were inflated by **≈0.23 deg**, independently of era or ED model.
* **Scenarios 2/3/5/6**: the v58-era convention error is 2-4x smaller in ⟨cos⟩ than the
  v63-era one, and the v63-era correction is worth -0.28…-0.50 deg in the pure-ES scenarios,
  so the historical pure-ES numbers were plausibly inflated by **≲0.2 deg** — a bias, not a
  reordering. For the full pipeline no useful bound can be set this way, because the r2-era
  fit also had no mixture at all, which is a larger and separate error.
* **What cannot be done without new fits**: `v80_fixed_1000_budget330` keeps only
  `scenario_cos_theta_report.json` per cat, no per-event tars, so b330 itself cannot be
  re-fitted. A direct measurement would mean re-fitting the `v80_fixed_1000` per-event
  outputs under the budget replay with corrected v58 tables (~2 x 1000 fits, a few condor
  hours). Cheap enough if the exact historical bias is ever wanted; not done here.

## 31. Footprint (this step)

* 3 burst-axis tables + provenance next to the v63 model: **108 kB**.
* 56 condor jobs (cluster 15969062); `r3_tables/` now holds 241 files, **1.1 MB**.
* `/eos/project-e/ep-nu` free inodes: **98 963**. Nothing written to the user area.
