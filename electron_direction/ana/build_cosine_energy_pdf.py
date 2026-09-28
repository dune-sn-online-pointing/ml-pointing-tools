#!/usr/bin/env python3
"""
Rebuild the pipeline's cosine-vs-energy likelihood table for a new ED model.

The deployed `refactor-snop-pipeline/data/cosine_energy_pdf.npz` is a raw
(no smoothing) histogram of cos(reco, true) in 18 RECO-cluster-energy bins,
built from ED v58's own held-out predictions (48,814 events, see
.../three_plane_v58.../cosine_energy_pdf.npz).  A retrained ED needs its own
table, because the table IS the model's angular resolution.

This script reproduces that recipe exactly (same energy bins, same 100 cosine
bins from -1 to 1, same raw normalisation) from any *_predictions.npz written by
train_three_plane_prodes_v62.py.

IMPORTANT: feed it the VALIDATION split, never the test split used for the
v58-vs-v62 comparison, so the likelihood is not built on the events the
comparison is measured on.

Usage:
  python3 build_cosine_energy_pdf.py \
      --predictions <model_dir>/val_predictions.npz \
      --out <model_dir>/cosine_energy_pdf.npz [--energy true|reco]
"""

import argparse
import json
import os

import numpy as np

V58_ENERGY_BINS = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 16),
                   (16, 18), (18, 20), (20, 22), (22, 24), (24, 26), (26, 28),
                   (28, 30), (30, 35), (35, 40), (40, 50), (50, 70)]
N_COSINE_BINS = 100


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--energy', choices=['reco', 'true'], default='reco',
                   help="which energy the table is binned in; v58's table uses the "
                        "RECO cluster energy (metadata col 10), which is also what "
                        "the pipeline looks the pdf up with")
    p.add_argument('--select', default=None,
                   help='optional numpy expression over the npz arrays, e.g. "r>=0.5"')
    args = p.parse_args()

    z = np.load(args.predictions, allow_pickle=False)
    cos = z['cos_reco_true'] if 'cos_reco_true' in z.files else \
        np.cos(np.radians(z['angular_errors']))
    energy = z['e_x'] if args.energy == 'reco' else z['true_energy']

    mask = (energy > 0) & (energy < 1000) & np.isfinite(cos)
    if args.select:
        mask &= eval(args.select, {'np': np}, {k: z[k] for k in z.files})  # noqa: S307
    cos, energy = cos[mask], energy[mask]

    edges = np.linspace(-1, 1, N_COSINE_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    pdf = np.zeros((len(V58_ENERGY_BINS), N_COSINE_BINS))
    n_per_bin = np.zeros(len(V58_ENERGY_BINS), dtype=np.int64)
    for i, (lo, hi) in enumerate(V58_ENERGY_BINS):
        m = (energy >= lo) & (energy < hi)
        n_per_bin[i] = int(m.sum())
        if n_per_bin[i] > 0:
            counts, _ = np.histogram(cos[m], bins=edges)
            pdf[i] = counts / (counts.sum() * width)
        else:
            print(f'WARNING: empty energy bin [{lo}, {hi}) MeV')

    np.savez(args.out, pdf_2d=pdf, cosine_bin_edges=edges, cosine_bin_centers=centers,
             energy_bins=np.asarray(V58_ENERGY_BINS, dtype=np.int64),
             n_events_per_bin=n_per_bin,
             smoothing_method=np.asarray('Raw histogram (no smoothing)'))
    meta = {'source_predictions': os.path.abspath(args.predictions),
            'energy_axis': args.energy, 'selection': args.select,
            'n_events': int(mask.sum()), 'n_events_per_bin': n_per_bin.tolist()}
    with open(args.out.replace('.npz', '_provenance.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
