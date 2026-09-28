#!/usr/bin/env python3
"""
Run one ED model over one packed split of the matchfix ES production pool and write a
predictions npz in the same format as the trainer's val/test_predictions.npz.

Used to get ED v58's predictions on the VALIDATION split, so that a reference
cosine-vs-energy pdf table can be built for v58 on exactly the same events (and with
exactly the same recipe) as the tables of the retrained models — which is what separates
a "new pdf" effect from a "new model" effect in the pipeline comparison.

Usage:
  python3 predict_packed_split.py --model <keras> --packed-dir <dir> --split val \
                                  --out <predictions.npz> [--batch-size 256]
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'lib'))

import prodes_matchfix_loader as pl  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--packed-dir', required=True)
    p.add_argument('--split', default='val', choices=list(pl.SPLITS))
    p.add_argument('--out', required=True)
    p.add_argument('--batch-size', type=int, default=256)
    args = p.parse_args()

    d = pl.load_packed(args.packed_dir, args.split, mmap=True)
    n = len(d['e_x'])
    print(f'{args.split}: {n} samples', flush=True)

    from tensorflow import keras
    model = keras.models.load_model(args.model, compile=False)
    preds = model.predict([np.asarray(d['images_u']), np.asarray(d['images_v']),
                           np.asarray(d['images_x'])],
                          batch_size=args.batch_size, verbose=0)
    preds = np.asarray(preds, dtype=np.float32)
    if preds.ndim == 3:
        preds = preds[:, 0, :]
    preds_n = pl.normalize_rows(preds).astype(np.float32)
    err = pl.angular_errors_deg(preds_n, d['directions'])

    np.savez(args.out,
             predictions=preds_n, predictions_raw=preds,
             true_directions=d['directions'],
             angular_errors=err.astype(np.float32),
             cos_reco_true=pl.cos_from_err(err).astype(np.float32),
             true_energy=d['true_energy'], energies=d['e_x'],
             e_x=d['e_x'], e_u=d['e_u'], e_v=d['e_v'], r=d['r'],
             partners_main=d['partners_main'], file_idx=d['file_idx'],
             event=d['event'], match_id=d['match_id'], x_row=d['x_row'],
             metadata=d['metadata'])
    print(pl.format_overall(pl.overall_stats(err), f'{os.path.basename(args.model)} {args.split}'))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
