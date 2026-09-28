#!/usr/bin/env python3
"""
Build the SAME / CHANGED / NEW / LOST table for the matcher-fixed ES production pool.

Compares, per X cluster, the matched-cluster products built from the SAME clusters
with the two partner-ambiguity rules:

  fix    : partner = most energetic induction candidate  (default, deployed)
  legacy : partner = first induction candidate in time   (--first-in-time-partner)

Classes (for X clusters with a 3-plane match, match_type == 3):
  SAME    : 3-plane matched under both rules with the same U and V partner ids
  CHANGED : 3-plane matched under both rules, but at least one partner differs
  NEW     : 3-plane matched only under the fixed rule
  LOST    : 3-plane matched only under the legacy rule

Join key for the ED predictions: (root basename, event, match_id of the fixed run),
which is exactly what the cluster-image metadata of the fixed products carries.

Usage:
  python3 make_matchclass_table.py --fix-dir <..._matchfix> --legacy-dir <..._legacyrule> \
                                   --out <table.npz> [--jobs 1]
"""

import argparse
import os
import sys
import time

import numpy as np
import uproot

BRANCHES = ['event', 'cluster_id', 'match_id', 'match_type',
            'matching_clusterId_U', 'matching_clusterId_V',
            'is_main_cluster', 'total_energy']
UV_BRANCHES = ['event', 'cluster_id', 'is_main_cluster', 'total_energy']

CLASS_CODE = {'SAME': 0, 'CHANGED': 1, 'NEW': 2, 'LOST': 3}


def read_file(path):
    """X-plane arrays plus {(event, cluster_id): (is_main, energy)} maps for U and V."""
    with uproot.open(path) as f:
        tx = f['clusters/clusters_tree_X']
        x = (tx.arrays(BRANCHES, library='np') if tx.num_entries
             else {b: np.zeros(0) for b in BRANCHES})
        uv = {}
        for pln in ('U', 'V'):
            t = f[f'clusters/clusters_tree_{pln}']
            if t.num_entries == 0:
                uv[pln] = {}
                continue
            a = t.arrays(UV_BRANCHES, library='np')
            uv[pln] = {(int(e), int(c)): (int(m), float(en))
                       for e, c, m, en in zip(a['event'], a['cluster_id'],
                                              a['is_main_cluster'], a['total_energy'])}
    return x, uv


def partner_info(uv, pln, ev, cid):
    """(is_main, energy) of a partner cluster, or (0, nan) if absent."""
    if cid is None or cid < 0:
        return 0, float('nan')
    return uv[pln].get((int(ev), int(cid)), (0, float('nan')))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fix-dir', required=True)
    p.add_argument('--legacy-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-files', type=int, default=None)
    args = p.parse_args()

    fix_files = sorted(f for f in os.listdir(args.fix_dir) if f.endswith('.root'))
    if args.max_files:
        fix_files = fix_files[:args.max_files]
    print(f'{len(fix_files)} matchfix files', flush=True)

    names, events, cids = [], [], []
    mid_fix, mid_leg = [], []
    typ_fix, typ_leg = [], []
    u_fix, v_fix, u_leg, v_leg = [], [], [], []
    pmain_fix, pmain_leg = [], []
    e_fix = []
    klass = []
    file_names = []
    n_missing_leg = 0
    t0 = time.time()

    for i, fn in enumerate(fix_files):
        pl = os.path.join(args.legacy_dir, fn)
        if not os.path.exists(pl):
            n_missing_leg += 1
            continue
        try:
            a, uva = read_file(os.path.join(args.fix_dir, fn))
            b, uvb = read_file(pl)
        except Exception as e:  # noqa: BLE001
            print(f'  WARNING {fn}: {e!r}', flush=True)
            continue
        fi = len(file_names)
        file_names.append(fn)

        legmap = {(int(ev), int(cid)): k
                  for k, (ev, cid) in enumerate(zip(b['event'], b['cluster_id']))}

        # every X cluster of the fixed run, plus legacy-only 3-plane matches
        seen = set()
        for k in range(len(a['event'])):
            ev, cid = int(a['event'][k]), int(a['cluster_id'][k])
            seen.add((ev, cid))
            tf = int(a['match_type'][k])
            kb = legmap.get((ev, cid))
            tl = int(b['match_type'][kb]) if kb is not None else -1
            if tf != 3 and tl != 3:
                continue
            if tf == 3 and tl == 3:
                same = (int(a['matching_clusterId_U'][k]) == int(b['matching_clusterId_U'][kb])
                        and int(a['matching_clusterId_V'][k]) == int(b['matching_clusterId_V'][kb]))
                cl = 'SAME' if same else 'CHANGED'
            elif tf == 3:
                cl = 'NEW'
            else:
                cl = 'LOST'
            uf = int(a['matching_clusterId_U'][k]) if tf == 3 else -1
            vf = int(a['matching_clusterId_V'][k]) if tf == 3 else -1
            ul = int(b['matching_clusterId_U'][kb]) if (kb is not None and tl == 3) else -1
            vl = int(b['matching_clusterId_V'][kb]) if (kb is not None and tl == 3) else -1
            names.append(fi)
            events.append(ev)
            cids.append(cid)
            mid_fix.append(int(a['match_id'][k]))
            mid_leg.append(int(b['match_id'][kb]) if kb is not None else -1)
            typ_fix.append(tf)
            typ_leg.append(tl)
            u_fix.append(uf)
            v_fix.append(vf)
            u_leg.append(ul)
            v_leg.append(vl)
            pmain_fix.append(int(partner_info(uva, 'U', ev, uf)[0] == 1
                                 and partner_info(uva, 'V', ev, vf)[0] == 1) if tf == 3 else 0)
            pmain_leg.append(int(partner_info(uvb, 'U', ev, ul)[0] == 1
                                 and partner_info(uvb, 'V', ev, vl)[0] == 1) if tl == 3 else 0)
            e_fix.append(float(a['total_energy'][k]))
            klass.append(CLASS_CODE[cl])

        # legacy clusters absent from the fixed file (should not happen: same clusters)
        for kb in range(len(b['event'])):
            key = (int(b['event'][kb]), int(b['cluster_id'][kb]))
            if key in seen or int(b['match_type'][kb]) != 3:
                continue
            ul = int(b['matching_clusterId_U'][kb])
            vl = int(b['matching_clusterId_V'][kb])
            names.append(fi); events.append(key[0]); cids.append(key[1])
            mid_fix.append(-1); mid_leg.append(int(b['match_id'][kb]))
            typ_fix.append(-1); typ_leg.append(3)
            u_fix.append(-1); v_fix.append(-1)
            u_leg.append(ul)
            v_leg.append(vl)
            pmain_fix.append(0)
            pmain_leg.append(int(partner_info(uvb, 'U', key[0], ul)[0] == 1
                                 and partner_info(uvb, 'V', key[0], vl)[0] == 1))
            e_fix.append(float(b['total_energy'][kb]))
            klass.append(CLASS_CODE['LOST'])

        if (i + 1) % 250 == 0:
            print(f'  {i + 1}/{len(fix_files)} files, {len(klass)} rows, '
                  f'{time.time() - t0:.0f}s', flush=True)

    klass = np.asarray(klass, dtype=np.int8)
    out = dict(
        file_idx=np.asarray(names, dtype=np.int32),
        file_names=np.asarray(file_names),
        event=np.asarray(events, dtype=np.int64),
        cluster_id=np.asarray(cids, dtype=np.int64),
        match_id_fix=np.asarray(mid_fix, dtype=np.int64),
        match_id_legacy=np.asarray(mid_leg, dtype=np.int64),
        match_type_fix=np.asarray(typ_fix, dtype=np.int8),
        match_type_legacy=np.asarray(typ_leg, dtype=np.int8),
        u_partner_fix=np.asarray(u_fix, dtype=np.int64),
        v_partner_fix=np.asarray(v_fix, dtype=np.int64),
        u_partner_legacy=np.asarray(u_leg, dtype=np.int64),
        v_partner_legacy=np.asarray(v_leg, dtype=np.int64),
        partners_main_fix=np.asarray(pmain_fix, dtype=np.int8),
        partners_main_legacy=np.asarray(pmain_leg, dtype=np.int8),
        x_total_energy=np.asarray(e_fix, dtype=np.float32),
        klass=klass,
        class_names=np.asarray(['SAME', 'CHANGED', 'NEW', 'LOST']),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    print(f'\nfiles paired: {len(file_names)} (legacy missing for {n_missing_leg})')
    for name, code in CLASS_CODE.items():
        n = int((klass == code).sum())
        print(f'  {name:8s}: {n:8d}  ({100.0 * n / max(len(klass), 1):.2f}%)')
    print(f'wrote {args.out}  ({len(klass)} rows, {time.time() - t0:.0f}s)')


if __name__ == '__main__':
    sys.exit(main())
