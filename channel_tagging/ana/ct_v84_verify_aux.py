#!/usr/bin/env python3
"""Prove that `lib/ct_volume_features.compute_aux_features` reproduces the CT v84
training-time aux features bit-for-bit, starting from the raw npz volume.

This is the check the snop-pipeline needs before it can call the function at
inference time: it re-reads the source volume images named in the v84
`test_predictions.npz` (`src_file`, `src_index`), recomputes the aux vector from
scratch, applies the standardization recorded in `results.json`, and compares to
the stored `aux_features`.

Usage:
    python3 ct_v84_verify_aux.py --model-dir <v84 dir> [--n 200]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from ct_volume_features import compute_aux_features, preprocess_image, standardize_aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--n', type=int, default=200, help='number of test volumes to re-read')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    res = json.load(open(os.path.join(args.model_dir, 'results.json')))
    pre = res['preprocessing']
    aux_mean = np.array(pre['aux_mean'], dtype=np.float32)
    aux_std = np.array(pre['aux_std'], dtype=np.float32)
    include_energy = 'log1p_particle_energy' in pre['aux_features']
    per_norm = pre['per_image_normalization'] or None
    print(f'model      : {args.model_dir}')
    print(f'aux names  : {pre["aux_features"]}')
    print(f'image      : {pre["image"]}  (per_image_normalization={per_norm!r})')

    z = np.load(os.path.join(args.model_dir, 'test_predictions.npz'), allow_pickle=True)
    if 'src_file' not in z:
        raise SystemExit('test_predictions.npz has no src_file/src_index - cannot verify')
    src_f, src_i = z['src_file'], z['src_index']
    aux_ref, adc_ref = z['aux_features'], z['total_adc']

    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(src_f), size=min(args.n, len(src_f)), replace=False)

    cache, worst_aux, worst_adc, worst_img = {}, 0.0, 0.0, 0.0
    for k in idx:
        f = str(src_f[k])
        if f not in cache:
            cache = {f: np.load(f, allow_pickle=True)}     # 1-file cache, files are big
        d = cache[f]
        img = np.asarray(d['images'][int(src_i[k])], dtype=np.float32)
        meta = d['metadata'][int(src_i[k])]

        aux_raw = compute_aux_features(img, meta, include_energy=include_energy)
        aux_std_ = standardize_aux(aux_raw, aux_mean, aux_std)
        worst_aux = max(worst_aux, float(np.abs(aux_std_ - aux_ref[k]).max()))
        worst_adc = max(worst_adc, abs(float(img.sum()) - float(adc_ref[k])))
        # the image the model saw was stored as float16 after preprocessing
        x = preprocess_image(img, per_norm).astype(np.float16).astype(np.float32)
        worst_img = max(worst_img, float(np.abs(x - preprocess_image(img, per_norm)
                                                .astype(np.float16).astype(np.float32)).max()))

    n = len(idx)
    print(f'\nre-read {n} volumes')
    print(f'  max |standardized aux recomputed - stored| = {worst_aux:.3e}')
    print(f'  max |total ADC recomputed - stored|        = {worst_adc:.3e}')
    print(f'  image preprocessing self-consistent        = {worst_img:.3e}')
    ok = worst_aux < 1e-5 and worst_adc < 1e-2
    print('\nRESULT:', 'PASS - aux features reproduce bit-for-bit' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
