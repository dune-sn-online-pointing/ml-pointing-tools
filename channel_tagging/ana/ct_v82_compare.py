#!/usr/bin/env python3
"""Compare CT models on the v80 test cats with the metrics of the v80 study.

Reproduces, for any number of models side by side:
  Table 2  - median score per class and efficiency/rejection vs reco energy
  Table 3  - conditional AUC in bins of reco energy (the metric that matters
             for "is this a topology classifier or an energy cut?")
  Table 8  - working points at MATCHED CC rejection, incl. the fraction of ES
             pointing information retained.

Inputs are model directories containing test_predictions.npz (and results.json).
Both the v80 layout (predictions/true_labels/energies/aux_features, total ADC
recoverable from the standardized aux features via results.json) and the v82
layout (explicit p_es/total_adc/cat_ids/...) are understood.

Read-only: writes only the markdown table file and the figures.
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

# --- palette (dataviz reference instance, light mode; validated) --------------
C_A = '#2a78d6'      # slot 1 blue
C_B = '#eb6834'      # slot 2 orange
C_C = '#1baf7a'      # slot 3 aqua
C_D = '#eda100'      # slot 4 yellow
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#dedcd6'
MODEL_COLORS = [C_A, C_B, C_C, C_D]

E_EDGES = [0, 3, 5, 7, 10, 15, 25, 100]
E_LABELS = ['0-3', '3-5', '5-7', '7-10', '10-15', '15-25', '>25']
E_EDGES_FINE = [0, 2, 4, 6, 8, 10, 12, 15, 18, 22, 30, 100]

# 68th percentile of the truth angle(e-, nu) per energy bin, from Table 0 of
# docs/CT_v80_energy_topology_study.md. Fixed here so that the "pointing
# information kept" column is the SAME yardstick for every model compared.
T68_STUDY = [36.5, 24.5, 19.2, 14.9, 10.9, 7.1, 4.6]

# reweighting binning used by the v82 trainer (for the energy-matched AUC)
RW_EDGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 60, 1e9]

_TABLES = []


def emit(text=''):
    print(text)
    _TABLES.append(text)


# --------------------------------------------------------------------------
def load_model_dir(path, name):
    """Return a dict with p_es, label, E, total_adc, cat (may be None)."""
    npz = np.load(os.path.join(path, 'test_predictions.npz'), allow_pickle=True)
    keys = set(npz.files)
    pred = npz['predictions']
    lab = npz['true_labels'].astype(int)
    E = np.asarray(npz['energies'], dtype=np.float64)
    p_es = np.asarray(npz['p_es'], dtype=np.float64) if 'p_es' in keys else pred[:, 0]

    if 'total_adc' in keys:
        adc = np.asarray(npz['total_adc'], dtype=np.float64)
    elif 'aux_features' in keys:
        # v80: aux_features are standardized [n_clusters, log1p(adc), log1p(npix)]
        with open(os.path.join(path, 'results.json')) as f:
            res = json.load(f)
        mu = np.array(res['preprocessing']['aux_mean'])
        sd = np.array(res['preprocessing']['aux_std'])
        adc = np.expm1(npz['aux_features'][:, 1] * sd[1] + mu[1]).astype(np.float64)
    else:
        adc = np.full(len(lab), np.nan)

    cat = npz['cat_ids'].astype(int) if 'cat_ids' in keys else None
    theta = (np.asarray(npz['theta_true_deg'], dtype=np.float64)
             if 'theta_true_deg' in keys else None)
    return dict(name=name, path=path, p_es=p_es, label=lab, E=E,
                total_adc=adc, cat=cat, theta=theta)


def ebin(E):
    """Index into E_EDGES bins; -1 if outside."""
    b = np.digitize(E, E_EDGES) - 1
    b[(E < E_EDGES[0]) | (E >= E_EDGES[-1])] = -1
    b[~np.isfinite(E)] = -1
    return b


def pointing_weights(E, t68=T68_STUDY):
    """Per-event 1/theta68(E)^2 (Fisher information of the ES scattering angle)."""
    b = ebin(E)
    w = np.zeros(len(E))
    ok = b >= 0
    w[ok] = 1.0 / np.asarray(t68, dtype=np.float64)[b[ok]] ** 2
    return w


def energy_match_weights(E, lab, edges=RW_EDGES, clip=10.0):
    """Same scheme as the v82 trainer: reweight both classes to the mean spectrum."""
    edges = np.asarray(edges, dtype=np.float64)
    nb = len(edges) - 1
    good = np.isfinite(E)
    b = np.clip(np.digitize(np.where(good, E, 0.0), edges) - 1, 0, nb - 1)
    h = np.zeros((2, nb))
    for c in (0, 1):
        m = good & (lab == c)
        h[c] = np.bincount(b[m], minlength=nb).astype(float)
        if h[c].sum():
            h[c] /= h[c].sum()
    target = 0.5 * (h[0] + h[1])
    w = np.ones(len(E))
    for c in (0, 1):
        m = good & (lab == c)
        den = h[c][b[m]]
        w[m] = np.where(den > 0, target[b[m]] / np.maximum(den, 1e-12), 0.0)
    w = np.minimum(w, clip)
    for c in (0, 1):
        m = lab == c
        if w[m].sum() > 0:
            w[m] *= (0.5 * len(E)) / w[m].sum()
    return w


def thr_for_cc_rejection(p_es, lab, target):
    """Smallest global threshold on P(ES) whose CC rejection reaches target."""
    cc = p_es[lab == 1]
    # rejection(t) = mean(cc <= t); take the target-quantile of the CC scores
    return float(np.quantile(cc, target))


# --------------------------------------------------------------------------
def table_marginal(models):
    emit('### Table A - marginal test metrics (balanced ES:CC = 1:1 test set)')
    emit()
    emit('| model | N(ES) | N(CC) | accuracy @0.5 | AUC | AUC after energy matching |')
    emit('|---|---|---|---|---|---|')
    for m in models:
        lab, p = m['label'], m['p_es']
        pred_cc = (p < 0.5).astype(int)
        acc = float((pred_cc == lab).mean())
        auc = roc_auc_score(lab, 1 - p)
        w = energy_match_weights(m['E'], lab)
        aucw = roc_auc_score(lab, 1 - p, sample_weight=w)
        emit(f"| {m['name']} | {(lab==0).sum()} | {(lab==1).sum()} | {acc:.4f} | "
             f'{auc:.4f} | {aucw:.4f} |')
    emit()
    emit('"AUC after energy matching" reweights both classes to the mean reconstructed-energy '
         'spectrum before computing the AUC, i.e. it is the separation that does NOT come from '
         'the different ES and CC energy spectra. It is the marginal analogue of Table B.')
    emit()


def table_conditional_auc(models):
    emit('### Table B - conditional AUC in bins of reconstructed main-cluster energy '
         '(0.5 = no separation)')
    emit()
    header = '| E [MeV] | ' + ' | '.join(f"N(ES)/N(CC) {m['name']}" for m in models[:1])
    emit('| E [MeV] | N(ES) | N(CC) | ' + ' | '.join(m['name'] for m in models) + ' |')
    emit('|---' * (3 + len(models)) + '|')
    rows = []
    for i, lb in enumerate(E_LABELS):
        cells, ns = [], None
        for m in models:
            b = ebin(m['E'])
            sel = b == i
            lab, p = m['label'][sel], m['p_es'][sel]
            nes, ncc = int((lab == 0).sum()), int((lab == 1).sum())
            if ns is None:
                ns = (nes, ncc)
            if nes < 5 or ncc < 5:
                cells.append('-')
            else:
                cells.append(f'{roc_auc_score(lab, 1 - p):.3f}')
        rows.append((lb, ns, cells))
        emit(f'| {lb} | {ns[0]} | {ns[1]} | ' + ' | '.join(cells) + ' |')
    marg = [f"{roc_auc_score(m['label'], 1 - m['p_es']):.3f}" for m in models]
    emit(f"| **all (marginal)** | {(models[0]['label']==0).sum()} | "
         f"{(models[0]['label']==1).sum()} | " + ' | '.join(f'**{v}**' for v in marg) + ' |')
    emit()
    emit('N(ES)/N(CC) are those of the first model; the other models have their own '
         'test draw, so their per-bin counts differ slightly (they are listed in Table C).')
    emit()
    return rows


def table_calibration(models):
    emit('### Table C - median P(ES) per class per energy bin (the calibration slide)')
    emit()
    cols = []
    for m in models:
        cols += [f"{m['name']} ES", f"{m['name']} CC"]
    emit('| E [MeV] | ' + ' | '.join(cols) + ' |')
    emit('|---' * (1 + len(cols)) + '|')
    for i, lb in enumerate(E_LABELS):
        cells = []
        for m in models:
            b = ebin(m['E'])
            for c in (0, 1):
                v = m['p_es'][(b == i) & (m['label'] == c)]
                cells.append(f'{np.median(v):.3f} ({len(v)})' if len(v) >= 5 else '-')
        emit(f'| {lb} | ' + ' | '.join(cells) + ' |')
    emit()
    emit('Numbers in brackets are the events in the bin. A model whose score is calibrated in '
         'energy has flat columns; v80 slides from ~0.89 to ~0.24 for ES.')
    emit()
    # slide magnitude
    emit('| model | median P(ES) ES, 0-3 MeV | median P(ES) ES, 15-25 MeV | slide | '
         'median P(ES) CC, 0-3 | median P(ES) CC, 15-25 | slide |')
    emit('|---|---|---|---|---|---|---|')
    for m in models:
        b = ebin(m['E'])
        out = []
        for c in (0, 1):
            lo = np.median(m['p_es'][(b == 0) & (m['label'] == c)])
            hi = np.median(m['p_es'][(b == 5) & (m['label'] == c)])
            out += [f'{lo:.3f}', f'{hi:.3f}', f'{lo-hi:+.3f}']
        emit(f"| {m['name']} | " + ' | '.join(out) + ' |')
    emit()


def table_working_points(models, targets=(0.945, 0.90)):
    emit('### Table D - global threshold at MATCHED CC rejection: ES efficiency per energy bin')
    emit()
    for tgt in targets:
        emit(f'**CC rejection = {tgt:.3f} (global threshold on P(ES), chosen on the test set)**')
        emit()
        emit('| model | threshold | ' + ' | '.join(f'{lb} MeV' for lb in E_LABELS) +
             ' | ES eff (E>10) | ES eff (all) | ES pointing info kept |')
        emit('|---' * (5 + len(E_LABELS)) + '|')
        for m in models:
            lab, p, E = m['label'], m['p_es'], m['E']
            thr = thr_for_cc_rejection(p, lab, tgt)
            keep = p > thr
            es = lab == 0
            b = ebin(E)
            cells = []
            for i in range(len(E_LABELS)):
                sel = es & (b == i)
                cells.append(f'{keep[sel].mean():.3f}' if sel.sum() >= 5 else '-')
            w = pointing_weights(E)
            hi = es & (E >= 10)
            info = w[es & keep].sum() / w[es].sum()
            emit(f"| {m['name']} | {thr:.4f} | " + ' | '.join(cells) +
                 f" | {keep[hi].mean():.3f} | {keep[es].mean():.3f} | {info:.3f} |")
        emit()


def table_summary_points(models, targets=(0.945, 0.90)):
    emit('### Table E - working-point summary (the v80 study Table 8 columns)')
    emit()
    emit('| model | selection | ES eff (all) | CC rej (all) | ES eff (E>10) | CC rej (E>10) '
         '| ES pointing information kept |')
    emit('|---|---|---|---|---|---|---|')
    for m in models:
        lab, p, E = m['label'], m['p_es'], m['E']
        es, cc = lab == 0, lab == 1
        w = pointing_weights(E)
        hi = E >= 10
        for tgt in targets:
            thr = thr_for_cc_rejection(p, lab, tgt)
            keep = p > thr
            emit(f"| {m['name']} | global P(ES) > {thr:.4f} (CC rej {tgt:.3f}) | "
                 f'{keep[es].mean():.3f} | {1-keep[cc].mean():.3f} | '
                 f'{keep[es & hi].mean():.3f} | {1-keep[cc & hi].mean():.3f} | '
                 f'{w[es & keep].sum()/w[es].sum():.3f} |')
        # also the deployed absolute threshold, for reference
        keep = p > 0.8
        emit(f"| {m['name']} | global P(ES) > 0.80 (deployed value) | "
             f'{keep[es].mean():.3f} | {1-keep[cc].mean():.3f} | '
             f'{keep[es & hi].mean():.3f} | {1-keep[cc & hi].mean():.3f} | '
             f'{w[es & keep].sum()/w[es].sum():.3f} |')
    emit()
    emit('"ES pointing information kept" = fraction of sum(1/theta68(E)^2) over ES events that '
         'survives the selection, with theta68 per energy bin fixed to Table 0 of the v80 study '
         f'({", ".join(f"{t:.1f}" for t in T68_STUDY)} deg) so that all models are scored on the '
         'same yardstick.')
    emit()


def table_t68_check(models):
    have = [m for m in models if m['theta'] is not None]
    if not have:
        return
    emit('### Table F - cross-check of the pointing yardstick on this test draw')
    emit()
    emit('| E [MeV] | theta68 study (Table 0) [deg] | ' +
         ' | '.join(f"theta68 {m['name']} [deg]" for m in have) + ' |')
    emit('|---' * (2 + len(have)) + '|')
    for i, lb in enumerate(E_LABELS):
        cells = []
        for m in have:
            b = ebin(m['E'])
            t = m['theta'][(b == i) & (m['label'] == 0) & np.isfinite(m['theta'])]
            cells.append(f'{np.percentile(t, 68):.1f} ({len(t)})' if len(t) >= 20 else '-')
        emit(f'| {lb} | {T68_STUDY[i]:.1f} | ' + ' | '.join(cells) + ' |')
    emit()


# --------------------------------------------------------------------------
def style(ax, xlabel=None, ylabel=None, title=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc='left')


def figures(models, figdir, targets=(0.945,)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(figdir, exist_ok=True)
    xc = [0.5 * (a + b) for a, b in zip(E_EDGES_FINE[:-1], E_EDGES_FINE[1:])]
    xc[-1] = 45

    # --- fig 1: the calibration slide ------------------------------------
    fig, axs = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.9), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for k, m in enumerate(models):
        ax = axs[0][k]
        for c, col, nm in ((0, C_A, 'ES'), (1, C_B, 'CC')):
            med, lo, hi = [], [], []
            for a, b in zip(E_EDGES_FINE[:-1], E_EDGES_FINE[1:]):
                v = m['p_es'][(m['label'] == c) & (m['E'] >= a) & (m['E'] < b)]
                if len(v) < 5:
                    med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
                med.append(np.median(v)); lo.append(np.percentile(v, 25))
                hi.append(np.percentile(v, 75))
            ax.fill_between(xc, lo, hi, color=col, alpha=0.16, lw=0)
            ax.plot(xc, med, color=col, lw=2, marker='o', ms=6, label=f'{nm} median (band: IQR)')
        ax.set_xscale('log'); ax.set_xticks([1, 2, 5, 10, 20, 40])
        ax.set_xticklabels(['1', '2', '5', '10', '20', '40'])
        ax.set_ylim(0, 1)
        style(ax, 'reconstructed main-cluster energy [MeV]', 'CT score P(ES)', m['name'])
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='lower left')
    fig.tight_layout()
    p = os.path.join(figdir, 'fig01_calibration_slide.png')
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f'  wrote {p}')

    # --- fig 2: ES efficiency vs energy at matched CC rejection ------------
    fig, axs = plt.subplots(1, len(targets), figsize=(6.2 * len(targets), 4.0), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for j, tgt in enumerate(targets):
        ax = axs[0][j]
        for k, m in enumerate(models):
            thr = thr_for_cc_rejection(m['p_es'], m['label'], tgt)
            keep = m['p_es'] > thr
            es = m['label'] == 0
            eff = []
            for a, b in zip(E_EDGES_FINE[:-1], E_EDGES_FINE[1:]):
                sel = es & (m['E'] >= a) & (m['E'] < b)
                eff.append(keep[sel].mean() if sel.sum() >= 10 else np.nan)
            ax.plot(xc, eff, color=MODEL_COLORS[k % len(MODEL_COLORS)], lw=2,
                    marker=['o', 's', 'D', '^'][k % 4], ms=6,
                    label=f"{m['name']} (thr {thr:.2f})")
        ax.set_xscale('log'); ax.set_xticks([1, 2, 5, 10, 20, 40])
        ax.set_xticklabels(['1', '2', '5', '10', '20', '40'])
        ax.set_ylim(-0.02, 1.02)
        style(ax, 'reconstructed main-cluster energy [MeV]', 'ES efficiency',
              f'ES efficiency at matched CC rejection {tgt:.3f}')
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc='upper right')
    fig.tight_layout()
    p = os.path.join(figdir, 'fig02_es_efficiency_matched_rejection.png')
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f'  wrote {p}')

    # --- fig 3: conditional AUC -------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    fig.patch.set_facecolor(SURFACE)
    xb = np.arange(len(E_LABELS))
    for k, m in enumerate(models):
        b = ebin(m['E'])
        vals = []
        for i in range(len(E_LABELS)):
            sel = b == i
            lab, p = m['label'][sel], m['p_es'][sel]
            vals.append(roc_auc_score(lab, 1 - p)
                        if ((lab == 0).sum() >= 5 and (lab == 1).sum() >= 5) else np.nan)
        ax.plot(xb, vals, color=MODEL_COLORS[k % len(MODEL_COLORS)], lw=2,
                marker=['o', 's', 'D', '^'][k % 4], ms=6, label=m['name'])
    ax.axhline(0.5, color=INK2, lw=1.0, ls=':')
    ax.set_xticks(xb); ax.set_xticklabels(E_LABELS)
    ax.set_ylim(0.45, 1.0)
    style(ax, 'reconstructed main-cluster energy [MeV]', 'conditional AUC',
          'Separation that survives conditioning on energy')
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc='lower left')
    fig.tight_layout()
    p = os.path.join(figdir, 'fig03_conditional_auc.png')
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f'  wrote {p}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', action='append', required=True,
                    help='NAME=/path/to/model_dir (repeatable, order preserved)')
    ap.add_argument('--tables', default=None, help='write the markdown tables here')
    ap.add_argument('--figdir', default=None)
    ap.add_argument('--emin', type=float, default=None,
                    help='drop events with reconstructed energy below this (MeV) from every '
                         'model, to compare models whose event selections differ at low energy')
    args = ap.parse_args()

    models = []
    for spec in args.model:
        name, _, path = spec.partition('=')
        models.append(load_model_dir(path, name))
        m = models[-1]
        print(f"loaded {name}: N={len(m['label'])} "
              f"(ES {int((m['label']==0).sum())}, CC {int((m['label']==1).sum())}) from {path}")

    if args.emin is not None:
        for m in models:
            keep = np.isfinite(m['E']) & (m['E'] >= args.emin)
            for k in ('p_es', 'label', 'E', 'total_adc'):
                m[k] = m[k][keep]
            for k in ('cat', 'theta'):
                if m[k] is not None:
                    m[k] = m[k][keep]
            print(f"  {m['name']}: kept {int(keep.sum())}/{len(keep)} events with "
                  f'E >= {args.emin} MeV')

    emit(f'*Generated by channel_tagging/ana/ct_v82_compare.py on the CT test cats'
         + (f', restricted to reconstructed energy >= {args.emin} MeV.*' if args.emin is not None
            else '.*'))
    emit()
    table_marginal(models)
    table_conditional_auc(models)
    table_calibration(models)
    table_working_points(models)
    table_summary_points(models)
    table_t68_check(models)

    if args.figdir:
        figures(models, args.figdir)
    if args.tables:
        with open(args.tables, 'w') as f:
            f.write('\n'.join(_TABLES) + '\n')
        print(f'wrote {args.tables}')


if __name__ == '__main__':
    main()
