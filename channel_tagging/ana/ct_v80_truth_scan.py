#!/usr/bin/env python3
"""Truth-level topology scan of burst-sample volume images (metadata only).

Reads only the 'metadata' array of every volume .npz in a cat range, so it is
cheap (no image decompression) and can cover every volume in the range rather
than the balanced 4000/class CT test subsample.

Usage:
    python3 ct_v80_truth_scan.py --cat-lo 597 --cat-hi 621 --out <file.npz>
"""

import argparse
import glob
import os
import time

import numpy as np

BURST_DIR = '/eos/project-e/ep-nu/evilla/sn-online-pointing/sn-burst-samples'
VOL_FMT = '{cat}_volume_images_tick3_ch2_min2_tot3_e3p0/X'

FIELDS = ['particle_energy', 'cluster_energy', 'n_clusters_in_volume',
          'n_marley_clusters', 'n_non_marley_clusters',
          'avg_marley_cluster_distance_cm', 'max_marley_cluster_distance_cm',
          'volume_size_cm', 'event', 'volume_index']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cat-lo', type=int, required=True)
    ap.add_argument('--cat-hi', type=int, required=True)
    ap.add_argument('--burst-dir', default=BURST_DIR)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rows = {k: [] for k in FIELDS}
    rows['label'] = []
    rows['cat'] = []
    t0 = time.time()
    for i in range(args.cat_lo, args.cat_hi + 1):
        cat = f'cat{i:06d}'
        d = os.path.join(args.burst_dir, cat, VOL_FMT.format(cat=cat))
        if not os.path.isdir(d):
            continue
        for label, pat in ((0, 'es_*.npz'), (1, 'cc_*.npz')):
            for f in sorted(glob.glob(os.path.join(d, pat))):
                md = np.load(f, allow_pickle=True)['metadata']
                for m in md:
                    for k in FIELDS:
                        v = m.get(k, None)
                        rows[k].append(float(v) if v is not None else np.nan)
                    rows['label'].append(label)
                    rows['cat'].append(i)
        print(f'{cat}: {len(rows["label"])} volumes  ({time.time()-t0:.0f}s)', flush=True)

    out = {k: np.array(v, dtype=np.float32) for k, v in rows.items()}
    np.savez_compressed(args.out, **out)
    n = len(out['label'])
    print(f'Wrote {args.out}: {n} volumes '
          f'(ES {int((out["label"]==0).sum())}, CC {int((out["label"]==1).sum())})')


if __name__ == '__main__':
    main()
