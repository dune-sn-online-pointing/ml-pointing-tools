#!/usr/bin/env python3
"""Join CT v80 test-set predictions to the full volume metadata.

`test_predictions.npz` written by `channel_tagging/models/train_ct_volume_v80.py`
stores predictions / true_labels / energies / (standardized) aux_features, but no
pointer back to the source volume.  The selection is however fully deterministic:
`train_ct_volume_v80.py` drives every shuffle from a single
`np.random.RandomState(seed)` and consumes it in a fixed order

    train: shuffle(es files), shuffle(cc files), permutation(n_train)
    val  : shuffle(es files), shuffle(cc files), permutation(n_val)
    test : shuffle(es files), shuffle(cc files), permutation(n_test)

and `RandomState.shuffle` consumes an amount of randomness that depends only on
the list length.  So the exact (file, index) of every test sample can be
recovered by replaying the stream with the same file counts.  The replay is
validated by requiring that the reconstructed truth energies and labels match
`test_predictions.npz` bit-for-bit.

Output: a .npz with one row per test volume holding the CT scores plus every
scalar metadata field of the source volume, plus the source file and index.

Usage:
    python3 ct_v80_build_test_join.py --out-dir <dir>
"""

import argparse
import glob
import json
import os

import numpy as np

BURST_DIR = '/eos/project-e/ep-nu/evilla/sn-online-pointing/sn-burst-samples'
VOL_FMT = '{cat}_volume_images_tick3_ch2_min2_tot3_e3p0/X'
MODEL_DIR = ('/eos/project-e/ep-nu/evilla/sn-online-pointing/neural-networks/'
             'channel_tagging/ct_volume_v80_20260706_224935')

# scalar metadata fields carried through to the joined table
META_FLOAT = [
    'particle_energy', 'cluster_energy',
    'main_track_momentum', 'main_track_momentum_x', 'main_track_momentum_y',
    'main_track_momentum_z', 'main_track_neutrino_momentum',
    'main_track_neutrino_momentum_x', 'main_track_neutrino_momentum_y',
    'main_track_neutrino_momentum_z',
    'n_clusters_in_volume', 'n_marley_clusters', 'n_non_marley_clusters',
    'avg_marley_cluster_distance_cm', 'max_marley_cluster_distance_cm',
    'center_channel', 'center_time_tpc', 'volume_size_cm',
    'main_cluster_id', 'volume_index', 'event',
]


def cat_dirs(lo, hi):
    out = []
    for i in range(lo, hi + 1):
        cat = f'cat{i:06d}'
        d = os.path.join(BURST_DIR, cat, VOL_FMT.format(cat=cat))
        if os.path.isdir(d):
            out.append(d)
    return out


def file_lists(lo, hi):
    es, cc = [], []
    for d in cat_dirs(lo, hi):
        es.extend(sorted(glob.glob(os.path.join(d, 'es_*.npz'))))
        cc.extend(sorted(glob.glob(os.path.join(d, 'cc_*.npz'))))
    return es, cc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default=MODEL_DIR)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    res = json.load(open(os.path.join(args.model_dir, 'results.json')))
    dcfg = res['config']['data']
    tcfg = res['config']['training']
    seed = tcfg.get('seed', 42)

    splits = {}
    for key in ('train_cats', 'val_cats', 'test_cats'):
        lo, hi = dcfg[key]
        splits[key] = file_lists(lo, hi)
        print(f'{key} {lo}-{hi}: {len(splits[key][0])} es files, '
              f'{len(splits[key][1])} cc files', flush=True)

    n_tr = res['data_summary']['n_train']
    n_va = res['data_summary']['n_val']
    n_te = res['data_summary']['n_test']

    rng = np.random.RandomState(seed)
    # replay train + val consumption (only list lengths matter)
    for key, n_tot in (('train_cats', n_tr), ('val_cats', n_va)):
        es, cc = splits[key]
        rng.shuffle(list(range(len(es))))
        rng.shuffle(list(range(len(cc))))
        rng.permutation(n_tot)

    es, cc = (list(x) for x in splits['test_cats'])
    rng.shuffle(es)
    rng.shuffle(cc)

    cap = dcfg['max_test_per_class']
    rows = []          # (label, file, idx, metadata dict)
    for label, files in ((0, es), (1, cc)):
        n = 0
        for f in files:
            if n >= cap:
                break
            d = np.load(f, allow_pickle=True)
            md = d['metadata']
            for idx in range(len(md)):
                if n >= cap:
                    break
                rows.append((label, f, idx, md[idx]))
                n += 1
        print(f'  reconstructed class {label}: {n}', flush=True)
    perm = rng.permutation(len(rows))
    rows = [rows[i] for i in perm]

    pred = np.load(os.path.join(args.model_dir, 'test_predictions.npz'), allow_pickle=True)
    labels_ref = pred['true_labels']
    energies_ref = pred['energies']

    labels = np.array([r[0] for r in rows], dtype=np.int32)
    energies = np.array([float(r[3].get('particle_energy', np.nan)) for r in rows],
                        dtype=np.float32)
    if not (labels == labels_ref).all():
        raise SystemExit('FATAL: reconstructed labels do not match test_predictions.npz')
    if not (energies == energies_ref).all():
        n_bad = int((energies != energies_ref).sum())
        raise SystemExit(f'FATAL: {n_bad}/{len(rows)} reconstructed energies differ')
    print('Replay validated: labels and energies match test_predictions.npz exactly')

    # undo the aux standardization to recover the raw aux features
    aux_mean = np.array(res['preprocessing']['aux_mean'], dtype=np.float32)
    aux_std = np.array(res['preprocessing']['aux_std'], dtype=np.float32)
    aux_raw = pred['aux_features'] * aux_std + aux_mean   # [n_clusters, log1p(adc), log1p(npix)]

    out = {
        'p_es': pred['predictions'][:, 0].astype(np.float32),
        'p_cc': pred['predictions'][:, 1].astype(np.float32),
        'label': labels,
        'n_clusters_aux': aux_raw[:, 0],
        'total_adc': np.expm1(aux_raw[:, 1]).astype(np.float32),
        'n_nonzero': np.expm1(aux_raw[:, 2]).astype(np.float32),
        'src_file': np.array([r[1] for r in rows]),
        'src_index': np.array([r[2] for r in rows], dtype=np.int32),
        'cat': np.array([os.path.basename(r[1].split('/X/')[0]).split('_')[0] for r in rows]),
    }
    for f in META_FLOAT:
        out[f] = np.array([float(r[3].get(f, np.nan)) if r[3].get(f, None) is not None
                           else np.nan for r in rows], dtype=np.float32)

    # cross-check: aux n_clusters (from training) vs metadata n_clusters_in_volume
    d_ncl = np.abs(out['n_clusters_aux'] - out['n_clusters_in_volume'])
    print(f'n_clusters cross-check: max |aux - metadata| = {d_ncl.max():.4f}')

    path = os.path.join(args.out_dir, 'ct_v80_test_join.npz')
    np.savez_compressed(path, **out)
    print(f'Wrote {path}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
