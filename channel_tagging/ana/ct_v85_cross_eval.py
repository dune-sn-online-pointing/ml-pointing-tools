#!/usr/bin/env python3
"""2x2 cross-evaluation: {CT v83, CT v85} x {unmasked, radmask} test images.

The test set is defined ONCE - it is the v80/v83/v84 test set, taken straight from
the `src_file` / `src_index` columns of v83's `test_predictions.npz` (whose ordering
was already validated against v80 bit-for-bit).  For every one of those 8000
volumes both the unmasked and the radiological-masked image are read from disk, so
the four score sets are perfectly paired: same events, same order, only the pixels
and the model change.

Runs on the GPU job alongside the v85 training.  Output: one npz with the four
score vectors plus the per-event quantities needed for the tables.

Usage:
    python3 ct_v85_cross_eval.py --v83 <dir> --v85 <dir> --out <file.npz>
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
from ct_volume_features import IMAGE_SHAPE, preprocess_image

MASK_SUFFIX = '_radmask'


def masked_path(unmasked_file):
    """.../<cat>_volume_images_..._e3p0/X/f.npz -> .../<cat>_volume_images_..._e3p0_radmask/X/f.npz"""
    head, tail = unmasked_file.split('/X/')
    return head + MASK_SUFFIX + '/X/' + tail


def load_paired_images(src_file, src_index):
    """Return (X_unmasked, X_masked, adc_unmasked, adc_masked), log1p float16."""
    n = len(src_file)
    Xu = np.zeros((n,) + IMAGE_SHAPE, dtype=np.float16)
    Xm = np.zeros((n,) + IMAGE_SHAPE, dtype=np.float16)
    adcu = np.zeros(n, dtype=np.float32)
    adcm = np.zeros(n, dtype=np.float32)
    Eu = np.full(n, np.nan, dtype=np.float32)
    Em = np.full(n, np.nan, dtype=np.float32)

    by_file = defaultdict(list)
    for i, f in enumerate(src_file):
        by_file[str(f)].append(i)

    for k, (f, rows) in enumerate(by_file.items()):
        mf = masked_path(f)
        du = np.load(f, allow_pickle=True)
        dm = np.load(mf, allow_pickle=True)
        iu, im = du['images'], dm['images']
        mu, mm = du['metadata'], dm['metadata']
        if len(iu) != len(im):
            raise SystemExit(f'FATAL: {f} has {len(iu)} volumes, masked has {len(im)}')
        for i in rows:
            j = int(src_index[i])
            a = np.asarray(iu[j], dtype=np.float32)
            b = np.asarray(im[j], dtype=np.float32)
            Xu[i] = preprocess_image(a, None).astype(np.float16)
            Xm[i] = preprocess_image(b, None).astype(np.float16)
            adcu[i] = a.sum(); adcm[i] = b.sum()
            Eu[i] = float(mu[j].get('particle_energy', np.nan))
            Em[i] = float(mm[j].get('particle_energy', np.nan))
        if (k + 1) % 200 == 0:
            print(f'  {k+1}/{len(by_file)} files...', flush=True)
    return Xu, Xm, adcu, adcm, Eu, Em


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v83', required=True)
    ap.add_argument('--v85', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()

    z = np.load(os.path.join(args.v83, 'test_predictions.npz'), allow_pickle=True)
    src_file, src_index = z['src_file'], z['src_index']
    label = z['true_labels'].astype(int)
    energy_ref = z['energies'].astype(np.float32)
    cat = z['cat']
    print(f'test set: {len(label)} volumes from v83 test_predictions.npz')

    print('reading paired unmasked/masked images...', flush=True)
    Xu, Xm, adcu, adcm, Eu, Em = load_paired_images(src_file, src_index)

    # pairing checks
    if not np.array_equal(Eu, energy_ref):
        raise SystemExit('FATAL: re-read unmasked energies differ from v83 test_predictions')
    if not np.array_equal(Eu, Em):
        n = int((Eu != Em).sum())
        raise SystemExit(f'FATAL: masked/unmasked particle_energy differ in {n} volumes')
    print('pairing validated: same events, identical metadata energies in both products')
    print(f'mask effect: median total ADC {np.median(adcu):.0f} -> {np.median(adcm):.0f} '
          f'(ratio {np.median(adcm/np.maximum(adcu,1)):.4f}); '
          f'{np.mean(adcm < adcu):.3f} of volumes changed')

    import tensorflow as tf
    out = dict(label=label, energy=energy_ref, cat=cat,
               total_adc_unmasked=adcu, total_adc_masked=adcm,
               src_file=src_file, src_index=src_index)

    for mname, mdir in (('v83', args.v83), ('v85', args.v85)):
        model = tf.keras.models.load_model(os.path.join(mdir, 'best_model.keras'), compile=False)
        for iname, X in (('unmasked', Xu), ('masked', Xm)):
            # predict in chunks: a full float32 copy of 8000 volumes is ~8 GB
            preds = []
            for lo in range(0, len(X), 512):
                chunk = X[lo:lo + 512].astype(np.float32)[..., np.newaxis]
                preds.append(np.asarray(model.predict(chunk, batch_size=args.batch_size,
                                                      verbose=0)))
            out[f'p_es_{mname}_on_{iname}'] = np.concatenate(preds)[:, 0].astype(np.float32)
            print(f'  {mname} on {iname}: mean P(ES) = '
                  f'{out[f"p_es_{mname}_on_{iname}"].mean():.4f}', flush=True)
        del model

    # sanity: v83-on-unmasked must reproduce the stored v83 test predictions
    d = np.abs(out['p_es_v83_on_unmasked'] - z['predictions'][:, 0].astype(np.float32)).max()
    print(f'\ncross-check v83-on-unmasked vs stored v83 predictions: max |diff| = {d:.2e}')
    out['selfcheck_v83_maxdiff'] = np.array([d])

    np.savez_compressed(args.out, **out)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
