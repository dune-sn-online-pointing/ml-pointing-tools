#!/usr/bin/env python3
"""
Evaluate ED models on the held-out test split of the matcher-fixed ES production pool.

For each model: overall, per true-energy bin, per reco-cluster-energy bin, split by
match class (SAME = 3-plane matched under both partner rules with the same partners,
NEW = matched only under the fixed rule, CHANGED = different partner) and versus the
truth-free quality variable r = min(E_U, E_V) / E_X.

The question this answers: does training on the new population fix the NEW events
without hurting the SAME ones -- i.e. was the v58 deficit "never seen in training"
or "the information is not there"?

Usage:
  python3 eval_ed_matchfix.py \
      --pred v58=/.../baseline_v58_test_predictions.npz \
      --pred v62=/.../test_predictions.npz \
      --manifest /.../packed.../split_manifest.json \
      --classes /.../matchclass_table.npz \
      --out-dir /.../eval
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'lib'))
import prodes_matchfix_loader as pl  # noqa: E402

CLASS_NAMES = ['SAME', 'CHANGED', 'NEW', 'LOST']
E_BINS = [3, 5, 10, 20, 30, np.inf]
E_LABELS = ['3-5', '5-10', '10-20', '20-30', '30+']
R_BINS = [-np.inf, 0.3, 0.5, 0.7, 0.9, np.inf]
R_LABELS = ['<0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '>=0.9']


def stats(err):
    err = np.asarray(err, dtype=np.float64)
    if err.size == 0:
        return {'N': 0, 'mean_cos': None, 'median_deg': None, 'q68_deg': None}
    return {'N': int(err.size),
            'mean_cos': float(np.mean(np.cos(np.radians(err)))),
            'median_deg': float(np.median(err)),
            'q68_deg': float(np.percentile(err, 68)),
            'mean_deg': float(np.mean(err))}


def bootstrap_dcos(err_a, err_b, n=400, seed=0):
    """68% interval on <cos>_a - <cos>_b for paired samples."""
    if err_a.size == 0:
        return None
    rng = np.random.RandomState(seed)
    ca = np.cos(np.radians(err_a))
    cb = np.cos(np.radians(err_b))
    d = np.array([np.mean((ca - cb)[rng.randint(0, ca.size, ca.size)]) for _ in range(n)])
    return [float(np.percentile(d, 16)), float(np.percentile(d, 84))]


def load_classes(path, manifest_files):
    z = np.load(path, allow_pickle=True)
    # materialise every column ONCE: indexing an NpzFile re-reads the member each time
    fnames = [str(f) for f in z['file_names']]
    fidx = np.asarray(z['file_idx'])
    klass_a = np.asarray(z['klass'])
    mtf = np.asarray(z['match_type_fix'])
    ev_a = np.asarray(z['event'])
    mid_a = np.asarray(z['match_id_fix'])
    pml = (np.asarray(z['partners_main_legacy']) if 'partners_main_legacy' in z.files
           else np.full(len(klass_a), -1, dtype=np.int8))
    sel = np.where(mtf == 3)[0]      # LOST rows have no fixed-run match id
    key = {(fnames[fidx[i]], int(ev_a[i]), int(mid_a[i])): (int(klass_a[i]), int(pml[i]))
           for i in sel}
    # basename of the X npz -> basename of the matched root
    root_of = {f: f.replace('_planeX.npz', '.root') for f in manifest_files}
    z.close()
    return key, root_of


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred', action='append', required=True, metavar='TAG=PATH')
    p.add_argument('--manifest', required=True)
    p.add_argument('--classes', default=None)
    p.add_argument('--split', default='test')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--reference', default=None, help='tag used as the delta reference')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.manifest) as f:
        manifest = json.load(f)
    mfiles = manifest['files'][args.split]

    preds = {}
    for spec in args.pred:
        tag, path = spec.split('=', 1)
        z = np.load(path, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        z.close()
        preds[tag] = d
        print(f'{tag}: {len(d["angular_errors"])} test samples from {path}')
    tags = list(preds)
    ref = args.reference or tags[0]

    # consistency: all prediction files must describe the same events, in the same order
    base = preds[tags[0]]
    for t in tags[1:]:
        for k in ('event', 'match_id', 'file_idx'):
            if not np.array_equal(base[k], preds[t][k]):
                raise SystemExit(f'{t} is not row-aligned with {tags[0]} on {k}')

    n = len(base['angular_errors'])
    klass = np.full(n, -1, dtype=np.int8)
    pmain_leg = np.full(n, -1, dtype=np.int8)
    if args.classes:
        key, root_of = load_classes(args.classes, mfiles)
        miss = 0
        for i in range(n):
            fn = root_of.get(mfiles[int(base['file_idx'][i])])
            c, pl_ = key.get((fn, int(base['event'][i]), int(base['match_id'][i])), (-1, -1))
            if c < 0:
                miss += 1
            klass[i] = c
            pmain_leg[i] = pl_
        print(f'match class joined for {n - miss}/{n} test samples ({miss} unmatched)')

    true_e = base['true_energy']
    reco_e = base['e_x']
    r = base['r']
    pmain = base['partners_main'] > 0.5

    # write, next to each prediction file, the same predictions plus the legacy-rule
    # flags, so the model directory itself carries "is this event also matched under
    # the legacy rule" per event.
    if args.classes:
        for spec in args.pred:
            tag, path = spec.split('=', 1)
            out = path[:-4] + '_classed.npz' if path.endswith('.npz') else path + '_classed.npz'
            z = np.load(path, allow_pickle=False)
            payload = {k: z[k] for k in z.files}
            z.close()
            payload['match_class'] = klass                       # 0 SAME 1 CHANGED 2 NEW 3 LOST -1 unjoined
            payload['match_class_names'] = np.asarray(CLASS_NAMES)
            payload['matched_under_legacy_rule'] = (klass != 2).astype(np.int8)
            payload['partners_main_legacy'] = pmain_leg
            payload['in_pipeline_under_legacy'] = (pmain_leg == 1).astype(np.int8)
            try:
                np.savez(out, **payload)
                print(f'wrote {out}')
            except Exception as e:  # noqa: BLE001
                print(f'could not write {out}: {e!r}')

    report = {'manifest': args.manifest, 'classes': args.classes, 'split': args.split,
              'n_test': int(n), 'reference': ref, 'models': {t: {} for t in tags},
              'class_counts': {CLASS_NAMES[c]: int((klass == c).sum())
                               for c in range(4)} | {'unjoined': int((klass < 0).sum())}}

    lines = []

    def block(title, masks, labels):
        lines.append('')
        lines.append(f'### {title}')
        lines.append(f"{'cell':>12} {'N':>7} " +
                     ' '.join(f'{t + " <cos>":>13}' for t in tags) + '   ' +
                     ' '.join(f'{t + " med":>11}' for t in tags))
        rows = []
        for lab, m in zip(labels, masks):
            nn = int(m.sum())
            cells = [stats(preds[t]['angular_errors'][m]) for t in tags]
            rows.append({'cell': lab, 'N': nn,
                         **{t: c for t, c in zip(tags, cells)}})
            if nn == 0:
                lines.append(f'{lab:>12} {0:>7}')
                continue
            lines.append(f'{lab:>12} {nn:>7} ' +
                         ' '.join(f'{c["mean_cos"]:>13.3f}' for c in cells) + '   ' +
                         ' '.join(f'{c["median_deg"]:>11.2f}' for c in cells))
        return rows

    def masks_energy(sel, energy):
        return [(sel & (energy >= lo) & (energy < hi)) for lo, hi in zip(E_BINS[:-1], E_BINS[1:])]

    all_sel = np.ones(n, bool)
    report['tables'] = {}

    report['tables']['overall'] = block('ALL test events', [all_sel], ['all'])
    report['tables']['true_energy'] = block(
        'ALL: per TRUE energy [MeV]', masks_energy(all_sel, true_e), E_LABELS)
    report['tables']['reco_energy'] = block(
        'ALL: per RECO cluster energy [MeV]', masks_energy(all_sel, reco_e), E_LABELS)

    if args.classes:
        cls_masks, cls_labels = [], []
        for c, cname in enumerate(CLASS_NAMES):
            m = klass == c
            if m.sum():
                cls_masks.append(m)
                cls_labels.append(cname)
        report['tables']['by_class'] = block('By match class', cls_masks, cls_labels)
        for cname in ('SAME', 'NEW', 'CHANGED'):
            c = CLASS_NAMES.index(cname)
            m = klass == c
            if m.sum() == 0:
                continue
            report['tables'][f'{cname}_true_energy'] = block(
                f'{cname}: per TRUE energy [MeV]', masks_energy(m, true_e), E_LABELS)

    # deployed (pipeline) classes: the pipeline only takes matches whose U and V
    # partners are main-track, so an event whose partner became main only under the
    # fixed rule is NEW *for the pipeline* even if it was already 3-plane matched
    # under the looser v58 training selection.
    if args.classes and (pmain_leg >= 0).any():
        dep_same = pmain & (pmain_leg == 1)
        dep_new = pmain & (pmain_leg == 0)
        report['tables']['deployed_class'] = block(
            'Pipeline (deployed) class: partners main under fix / under legacy',
            [dep_same, dep_new, ~pmain], ['dep SAME', 'dep NEW', 'not in pipeline'])
        for lab, m in (('dep SAME', dep_same), ('dep NEW', dep_new)):
            if m.sum():
                report['tables'][f'{lab.replace(" ", "_")}_true_energy'] = block(
                    f'{lab}: per TRUE energy [MeV]', masks_energy(m, true_e), E_LABELS)

    rm = [(r >= lo) & (r < hi) for lo, hi in zip(R_BINS[:-1], R_BINS[1:])]
    report['tables']['by_r'] = block('By r = min(E_U,E_V)/E_X', rm, R_LABELS)
    report['tables']['partners_main'] = block(
        'Deployed-like subset (both partners main-track)', [pmain, ~pmain],
        ['partners main', 'partner non-main'])

    # deltas vs reference, overall and per class
    lines.append('')
    lines.append(f'### delta <cos> vs {ref} (paired bootstrap 68%)')
    deltas = {}
    sels = {'all': all_sel}
    if args.classes:
        for c, cname in enumerate(CLASS_NAMES):
            if (klass == c).sum():
                sels[cname] = klass == c
        if (pmain_leg >= 0).any():
            sels['dep SAME'] = pmain & (pmain_leg == 1)
            sels['dep NEW'] = pmain & (pmain_leg == 0)
    for sname, m in sels.items():
        deltas[sname] = {}
        for t in tags:
            if t == ref:
                continue
            ea = preds[t]['angular_errors'][m]
            eb = preds[ref]['angular_errors'][m]
            dc = float(np.mean(np.cos(np.radians(ea))) - np.mean(np.cos(np.radians(eb))))
            ci = bootstrap_dcos(ea, eb)
            deltas[sname][t] = {'d_mean_cos': dc, 'ci68': ci, 'N': int(m.sum())}
            lines.append(f'{sname:>12} {t:>10}: d<cos> = {dc:+.4f}  '
                         f'[{ci[0]:+.4f}, {ci[1]:+.4f}]  N={int(m.sum())}')
    report['deltas_vs_reference'] = deltas

    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(args.out_dir, 'eval_tables.txt'), 'w') as f:
        f.write(txt + '\n')
    with open(os.path.join(args.out_dir, 'eval_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=float)
    print(f'\nwrote {args.out_dir}/eval_tables.txt and eval_report.json')


if __name__ == '__main__':
    main()
