#!/usr/bin/env python3
"""
Independent cross-check of ED models on the 50 mixture-dev burst cats (623-672).

These cats are EVALUATION bursts: they were never training data for v58 nor for
v62/v63 (which train on the ES production pool), and they are a different, locally
produced MC from the production sample.  They also have BOTH product sets on disk
-- the original (legacy first-in-time partner) and the `_matchfix` one -- so the
SAME / NEW classification of `ed_on_matchfix_diagnosis.md` can be reproduced
exactly: a true-ES main track selected by the pipeline loader in the matchfix
products is SAME if the same X cluster is also selected in the original products,
NEW otherwise.

Selection = the DEPLOYED pipeline selection (main-track X, main-track U and V
partners), restricted to true-ES main tracks, as in scenario_2_perfect_ct.

Usage:
  python3 eval_ed_burstcats_matchfix.py \
      --model v58=/path/model.keras --model v62=/path/best_model.keras \
      --cats 623-672 --out-dir <...>
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'lib'))
import prodes_matchfix_loader as pl  # noqa: E402

DEFAULT_BASE = '/eos/user/e/evilla/dune/sn-tps/mixture_dev_samples'
FIX_FMT = '{cat}_cluster_images_tick3_ch2_min2_tot3_e3p0_matchfix'
ORIG_FMT = '{cat}_cluster_images_tick3_ch2_min2_tot3_e3p0'
E_BINS = [3, 5, 10, 20, 30, np.inf]
E_LABELS = ['3-5', '5-10', '10-20', '20-30', '30+']


def load_cats(base, cats, verbose=True):
    """Load matchfix true-ES main tracks with the pipeline selection + SAME/NEW flag."""
    acc = {k: [] for k in ('images_u', 'images_v', 'images_x', 'directions', 'e_x',
                           'e_u', 'e_v', 'r', 'true_energy', 'event', 'match_id',
                           'x_row', 'cat')}
    is_new = []
    n_files = 0
    for c in cats:
        cat = f'cat{c:06d}'
        dfix = os.path.join(base, cat, FIX_FMT.format(cat=cat))
        dorig = os.path.join(base, cat, ORIG_FMT.format(cat=cat))
        if not (os.path.isdir(os.path.join(dfix, 'X')) and os.path.isdir(os.path.join(dorig, 'X'))):
            print(f'  {cat}: missing products, skipped')
            continue
        for fx in sorted(glob.glob(os.path.join(dfix, 'X', 'es_*_planeX.npz'))):
            name = os.path.basename(fx)
            fu, fv = pl.uv_paths(fx)
            ox = os.path.join(dorig, 'X', name)
            ou, ov = pl.uv_paths(ox)
            if not all(os.path.exists(p) for p in (fu, fv, ox, ou, ov)):
                continue
            s = pl.select_file(fx, fu, fv, with_images=True,
                               require_main_partners=True, es_main_only=True)
            if s['n'] == 0:
                continue
            o = pl.select_file(ox, ou, ov, with_images=False,
                               require_main_partners=True, es_main_only=True)
            orig_rows = set(o['x_row'].astype(int).tolist())
            for k in acc:
                if k == 'cat':
                    acc['cat'].append(np.full(s['n'], c, dtype=np.int32))
                else:
                    acc[k].append(s[k])
            is_new.append(np.array([int(x) not in orig_rows for x in s['x_row'].astype(int)]))
            n_files += 1
        if verbose:
            print(f'  {cat}: running total {sum(len(a) for a in acc["e_x"])}', flush=True)
    out = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    out['is_new'] = np.concatenate(is_new)
    out['n_files'] = n_files
    return out


def stats(err):
    err = np.asarray(err, dtype=np.float64)
    if err.size == 0:
        return {'N': 0}
    return {'N': int(err.size), 'mean_cos': float(np.mean(np.cos(np.radians(err)))),
            'median_deg': float(np.median(err)), 'q68_deg': float(np.percentile(err, 68))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', action='append', required=True, metavar='TAG=PATH')
    p.add_argument('--base', default=DEFAULT_BASE)
    p.add_argument('--cats', default='623-672')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--cache', default=None,
                   help='npz cache of the loaded cats (written if missing, reused if present)')
    args = p.parse_args()

    lo, hi = (int(x) for x in args.cats.split('-'))
    cats = list(range(lo, hi + 1))
    assert lo >= 623, 'burst cats below 623 are training cats; use 623+ for evaluation'
    os.makedirs(args.out_dir, exist_ok=True)

    if args.cache and os.path.exists(args.cache):
        print(f'Loading cached cats from {args.cache}', flush=True)
        z = np.load(args.cache, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        z.close()
        d['n_files'] = int(d.get('n_files', -1))
    else:
        print(f'Loading cats {lo}-{hi} from {args.base}', flush=True)
        d = load_cats(args.base, cats)
        if args.cache:
            os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
            np.savez(args.cache, **{k: v for k, v in d.items() if k != 'n_files'},
                     n_files=np.int32(d['n_files']))
            print(f'cached to {args.cache}', flush=True)
    n = len(d['e_x'])
    print(f'\n{n} true-ES main tracks (pipeline selection) from {d["n_files"]} ES files; '
          f'NEW: {int(d["is_new"].sum())} ({100 * d["is_new"].mean():.1f}%)', flush=True)

    from tensorflow import keras
    lines, report = [], {'cats': [lo, hi], 'n': int(n),
                         'n_new': int(d['is_new'].sum()), 'models': {}}
    errs = {}
    for spec in args.model:
        tag, path = spec.split('=', 1)
        m = keras.models.load_model(path, compile=False)
        pr = m.predict([d['images_u'], d['images_v'], d['images_x']],
                       batch_size=args.batch_size, verbose=0)
        pr = np.asarray(pr, dtype=np.float32)
        if pr.ndim == 3:
            pr = pr[:, 0, :]
        errs[tag] = pl.angular_errors_deg(pl.normalize_rows(pr), d['directions'])
        np.savez(os.path.join(args.out_dir, f'burstcats_{tag}_predictions.npz'),
                 predictions=pl.normalize_rows(pr).astype(np.float32),
                 true_directions=d['directions'],
                 angular_errors=errs[tag].astype(np.float32),
                 is_new=d['is_new'], cat=d['cat'], e_x=d['e_x'], e_u=d['e_u'],
                 e_v=d['e_v'], r=d['r'], true_energy=d['true_energy'],
                 event=d['event'], match_id=d['match_id'], x_row=d['x_row'])
        del m
        print(f'{tag}: done', flush=True)

    tags = list(errs)
    for sel_name, m in (('ALL', np.ones(n, bool)),
                        ('SAME', ~d['is_new']), ('NEW', d['is_new'])):
        lines.append('')
        lines.append(f'### {sel_name}  (N={int(m.sum())})')
        lines.append(f"{'E bin':>10} {'N':>7} " + ' '.join(f'{t + " <cos>":>13}' for t in tags)
                     + '   ' + ' '.join(f'{t + " Q68":>11}' for t in tags))
        rows = []
        for lab, elo, ehi in [('all', 0, np.inf)] + list(zip(E_LABELS, E_BINS[:-1], E_BINS[1:])):
            mm = m & (d['true_energy'] >= elo) & (d['true_energy'] < ehi)
            cells = [stats(errs[t][mm]) for t in tags]
            rows.append({'bin': lab, 'N': int(mm.sum()), **{t: c for t, c in zip(tags, cells)}})
            if mm.sum() == 0:
                lines.append(f'{lab:>10} {0:>7}')
                continue
            lines.append(f'{lab:>10} {int(mm.sum()):>7} '
                         + ' '.join(f'{c["mean_cos"]:>13.3f}' for c in cells) + '   '
                         + ' '.join(f'{c["q68_deg"]:>11.2f}' for c in cells))
        report['models'][sel_name] = rows

    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(args.out_dir, 'burstcats_eval.txt'), 'w') as f:
        f.write(txt + '\n')
    with open(os.path.join(args.out_dir, 'burstcats_eval.json'), 'w') as f:
        json.dump(report, f, indent=2, default=float)
    print(f'\nwrote {args.out_dir}/burstcats_eval.txt')


if __name__ == '__main__':
    main()
