#!/usr/bin/env python3
"""
Evaluate any three-plane ED Keras model on sn-burst-samples ES clusters
(pipeline three-plane selection, raw ADC input, order [U, V, X]) for a cat range,
and write predictions + per-energy resolution tables.

Example (v58 baseline on the v61 test cats):
  python3 electron_direction/models/evaluate_ed_on_burstcats.py \
      --model /eos/.../three_plane_three_plane_v58_.../checkpoints/model_epoch_12_val_loss_0.8898.keras \
      --json electron_direction/json/three_plane_v61_burstcats_es.json --split test \
      --out /eos/.../<v61 dir>/eval_v58_test --tag v58
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'lib'))
import burstcats_es_loader as bl  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='.keras model path')
    p.add_argument('--json', required=True, help='config JSON with data.* and *_cats')
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    p.add_argument('--cats', nargs=2, type=int, default=None, help='override cat range lo hi')
    p.add_argument('--out', required=True, help='output directory')
    p.add_argument('--tag', default='model')
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()

    with open(args.json) as f:
        dcfg = json.load(f)['data']
    lo, hi = args.cats or dcfg[f'{args.split}_cats']
    if lo < 400 or hi > 622:
        raise SystemExit('Refusing cats outside 400-622 (held-out evaluation set)')
    dirs = bl.cat_dirs_for_range(dcfg['burst_samples_dir'], lo, hi,
                                 dcfg.get('cluster_subdir_fmt', bl.DEFAULT_SUBDIR_FMT))
    d = bl.load_three_plane_es_from_cats(dirs, x_glob=dcfg.get('x_file_glob', bl.DEFAULT_X_GLOB),
                                         verbose=False, split_name=args.split)

    from tensorflow import keras
    model = keras.models.load_model(args.model, compile=False)
    preds = np.asarray(model.predict([d['images_u'], d['images_v'], d['images_x']],
                                     batch_size=args.batch_size, verbose=0), dtype=np.float32)
    if preds.ndim == 3 and preds.shape[1] == 1:
        preds = preds[:, 0, :]
    preds_n = bl.normalize_rows(preds)
    err = bl.angular_errors_deg(preds_n, d['directions'])

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, f'{args.tag}_{args.split}_predictions.npz'),
                        predictions=preds_n.astype(np.float32), true_directions=d['directions'],
                        angular_errors=err.astype(np.float32), energies=d['cluster_energy'],
                        cluster_energy=d['cluster_energy'], true_energy=d['true_energy'],
                        cat_ids=d['cat_ids'], metadata=d['metadata'])
    res = {
        'model': args.model, 'tag': args.tag, 'split': args.split, 'cats': [lo, hi],
        'cats_used': [c for c, _ in dirs], 'n_samples': int(len(err)),
        'overall': bl.overall_stats(err),
        'per_true_energy': bl.per_energy_table(err, d['true_energy']),
        'per_reco_energy': bl.per_energy_table(err, d['cluster_energy']),
    }
    txt = '\n'.join([bl.format_overall(res['overall'], f'{args.tag} {args.split}'),
                     bl.format_table(res['per_true_energy'], 'per TRUE energy'), '',
                     bl.format_table(res['per_reco_energy'], 'per RECO cluster energy')])
    print(txt)
    with open(os.path.join(args.out, f'{args.tag}_{args.split}_results.json'), 'w') as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(args.out, f'{args.tag}_{args.split}_tables.txt'), 'w') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()
