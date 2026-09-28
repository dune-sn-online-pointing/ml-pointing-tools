#!/usr/bin/env python3
"""
Pack the rebuilt ES production three-plane ED pool into memmap-able .npy shards.

Two passes over the cluster-image npz files:
  1. metadata only (cheap: the npz are uncompressed, so only the small
     metadata member is read) -> per-split sample counts and bookkeeping
  2. images -> fills pre-allocated .npy memmaps

Output (one directory, 13 files, ~8-9 GB):
  split_manifest.json
  {train,val,test}_{u,v,x}.npy     float32 (N, 128, 32, 1) raw ADC
  {train,val,test}_meta.npz        directions, energies, r, provenance

Usage:
  python3 pack_prodes_pool.py --images-dir <...cluster_images..._matchfix> \
                              --out-dir <...packed...> [--seed 42] [--max-files N]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'lib'))

import prodes_matchfix_loader as pl  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--images-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-files', type=int, default=None,
                   help='debug: only use the first N X files')
    p.add_argument('--x-glob', default='*_planeX.npz')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    x_files = pl.list_x_files(args.images_dir, args.x_glob)
    if args.max_files:
        x_files = x_files[:args.max_files]
    print(f'X files with U and V present: {len(x_files)}', flush=True)
    if not x_files:
        raise SystemExit('no input files')

    splits = pl.split_files(x_files, seed=args.seed)
    for s in pl.SPLITS:
        print(f'  {s:5s}: {len(splits[s])} files', flush=True)

    # ---------------------------------------------------------------- pass 1
    print('\nPass 1: metadata only', flush=True)
    per_split = {}
    for s in pl.SPLITS:
        acc = {k: [] for k in pl.SCALARS}
        acc['file_idx'] = []
        n_empty = 0
        t1 = time.time()
        for fi, fx in enumerate(splits[s]):
            fu, fv = pl.uv_paths(fx)
            try:
                d = pl.select_file(fx, fu, fv, with_images=False)
            except Exception as e:  # noqa: BLE001
                print(f'  WARNING unreadable {os.path.basename(fx)}: {e!r}', flush=True)
                n_empty += 1
                continue
            if d['n'] == 0:
                n_empty += 1
                continue
            for k in pl.SCALARS:
                acc[k].append(d[k])
            acc['file_idx'].append(np.full(d['n'], fi, dtype=np.float32))
            if (fi + 1) % 250 == 0:
                print(f'  [{s}] {fi + 1}/{len(splits[s])} files, '
                      f'{sum(len(a) for a in acc["e_x"])} samples, '
                      f'{time.time() - t1:.0f}s', flush=True)
        packed = {k: np.concatenate(v, axis=0) for k, v in acc.items() if v}
        per_split[s] = packed
        n = len(packed['e_x'])
        print(f'  [{s}] {n} samples from {len(splits[s])} files '
              f'({n_empty} with no selected cluster), {time.time() - t1:.0f}s', flush=True)

    # ---------------------------------------------------------------- pass 2
    # Arrays are built in RAM and written with np.save: np.lib.format.open_memmap
    # on an EOS fuse mount is not reliable.  Peak RAM = one split (~6 GB for train).
    print('\nPass 2: images', flush=True)
    for s in pl.SPLITS:
        n = len(per_split[s]['e_x'])
        mm = {p: np.empty((n,) + pl.IMG_SHAPE, dtype=np.float32) for p in ('u', 'v', 'x')}
        pos = 0
        t1 = time.time()
        for fi, fx in enumerate(splits[s]):
            fu, fv = pl.uv_paths(fx)
            try:
                d = pl.select_file(fx, fu, fv, with_images=True)
            except Exception as e:  # noqa: BLE001
                print(f'  WARNING unreadable {os.path.basename(fx)}: {e!r}', flush=True)
                continue
            k = d['n']
            if k == 0:
                continue
            mm['u'][pos:pos + k] = d['images_u']
            mm['v'][pos:pos + k] = d['images_v']
            mm['x'][pos:pos + k] = d['images_x']
            pos += k
            if (fi + 1) % 250 == 0:
                print(f'  [{s}] {fi + 1}/{len(splits[s])} files, {pos}/{n} rows, '
                      f'{time.time() - t1:.0f}s', flush=True)
        if pos != n:
            raise RuntimeError(f'{s}: filled {pos} rows, expected {n} '
                               '(pass 1 and pass 2 disagree)')
        for p in ('u', 'v', 'x'):
            np.save(os.path.join(args.out_dir, f'{s}_{p}.npy'), mm[p])
        mm.clear()
        np.savez(os.path.join(args.out_dir, f'{s}_meta.npz'), **per_split[s])
        print(f'  [{s}] wrote {n} rows, {time.time() - t1:.0f}s', flush=True)

    manifest = {
        'images_dir': args.images_dir,
        'seed': args.seed,
        'split_fractions': [0.70, 0.15, 0.15],
        'split_level': 'source cluster-image file (X npz)',
        'selection': ('v58 training selection: X match_id != -1 (match_clusters only '
                      'assigns match ids to MAIN X clusters) and the (event, match_id) '
                      'pair present in U and V with match_id != -1; no main-track '
                      'requirement on the U/V partners'),
        'preprocessing': 'raw float32 ADC, channel dim appended, no normalization',
        'image_shape': list(pl.IMG_SHAPE),
        'n_x_files': len(x_files),
        'files': {s: [os.path.basename(f) for f in splits[s]] for s in pl.SPLITS},
        'n_samples': {s: int(len(per_split[s]['e_x'])) for s in pl.SPLITS},
        'n_files': {s: len(splits[s]) for s in pl.SPLITS},
        'built_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'pack_seconds': time.time() - t0,
    }
    with open(os.path.join(args.out_dir, 'split_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print('\n' + json.dumps({k: v for k, v in manifest.items() if k != 'files'}, indent=2))
    print(f'\nDone in {(time.time() - t0) / 60:.1f} min -> {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
