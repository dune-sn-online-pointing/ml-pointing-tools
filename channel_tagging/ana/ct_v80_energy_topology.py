#!/usr/bin/env python3
"""CT v80 energy-vs-topology study: is the deployed channel tagger an energy cut?

Inputs
  --join        output of ct_v80_build_test_join.py (CT scores + full volume
                metadata for the 8000-volume balanced test set, cats 597-621)
  --truth-scan  output of ct_v80_truth_scan.py over the same cats (all volumes,
                metadata only) -- used for the high-statistics topology profiles

Outputs
  --figdir      PNG figures
  --tables      markdown tables (stdout is also written there)

Everything is read-only; nothing is written outside --figdir / --tables.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

# --- palette (dataviz reference instance, light mode; validated) --------------
C_ES = '#2a78d6'      # slot 1 blue
C_CC = '#eb6834'      # slot 2 orange
C_3 = '#1baf7a'       # slot 3 aqua
C_4 = '#eda100'       # slot 4 yellow
C_5 = '#e87ba4'       # slot 5 magenta
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#dedcd6'

E_EDGES = [0, 3, 5, 7, 10, 15, 25, 100]
E_EDGES_FINE = [0, 2, 4, 6, 8, 10, 12, 15, 18, 22, 30, 100]

_TABLES = []


def emit(text=''):
    print(text)
    _TABLES.append(text)


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


def newfig(*a, **kw):
    fig, ax = plt.subplots(*a, **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def save(fig, figdir, name):
    p = os.path.join(figdir, name)
    fig.tight_layout()
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f'  wrote {p}')


def bin_centers(edges, last=None):
    e = list(edges)
    if last is not None:
        e[-1] = last
    return [0.5 * (a + b) for a, b in zip(e[:-1], e[1:])]


def profile(x, y, edges, stat='mean'):
    """Return per-bin (n, value, error) of y in bins of x."""
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        n = int(m.sum())
        if n == 0:
            out.append((0, np.nan, np.nan))
            continue
        v = y[m]
        if stat == 'mean':
            out.append((n, float(v.mean()), float(v.std() / np.sqrt(n))))
        elif stat == 'frac':
            p = float(v.mean())
            out.append((n, p, float(np.sqrt(max(p * (1 - p), 1e-9) / n))))
        else:
            out.append((n, float(np.median(v)), np.nan))
    return out


# =============================================================================
# Q0: how much pointing information does each energy carry?
# =============================================================================
def pointing_weights(j):
    """Per-event weight ~ Fisher information of the ES kinematic scattering angle.

    The true ES electron direction and the true neutrino direction are both in
    the volume metadata, so the intrinsic (irreducible) kinematic smearing can
    be measured directly.  Weight = 1 / theta68(E)^2 using the 68th percentile
    of the opening angle in the event's energy bin.
    """
    E = j['particle_energy']
    lab = j['label'].astype(int)
    e = np.stack([j['main_track_momentum_x'], j['main_track_momentum_y'],
                  j['main_track_momentum_z']], 1)
    nu = np.stack([j['main_track_neutrino_momentum_x'], j['main_track_neutrino_momentum_y'],
                   j['main_track_neutrino_momentum_z']], 1)
    ne, nn = np.linalg.norm(e, axis=1), np.linalg.norm(nu, axis=1)
    good = (ne > 0) & (nn > 0)
    cos = np.ones(len(E))
    cos[good] = np.clip(np.sum(e[good] * nu[good], axis=1) / (ne[good] * nn[good]), -1, 1)
    theta = np.degrees(np.arccos(cos))
    w = np.zeros(len(E))
    t68 = {}
    for a, b in zip(E_EDGES[:-1], E_EDGES[1:]):
        m = (lab == 0) & good & (E >= a) & (E < b)
        if m.sum() < 20:
            continue
        t = float(np.percentile(theta[m], 68))
        t68[(a, b)] = t
        w[(E >= a) & (E < b)] = 1.0 / t ** 2
    return theta, w, t68


def q0_pointing(j, theta, t68, figdir):
    lab = j['label'].astype(int)
    E = j['particle_energy']
    es = lab == 0
    emit('### Table 0 - intrinsic ES kinematics: the pointing power lives at high energy')
    emit()
    emit('| E [MeV] | N(ES) | median angle(e-, nu) [deg] | 68% angle [deg] | relative pointing '
         'information per event |')
    emit('|---|---|---|---|---|')
    ref = None
    for (a, b), t in t68.items():
        m = es & (E >= a) & (E < b)
        if ref is None:
            ref = 1.0 / t ** 2
        emit(f'| {a}-{b} | {m.sum()} | {np.median(theta[m]):.1f} | {t:.1f} | '
             f'{(1.0/t**2)/ref:.1f}x |')
    emit()
    fig, ax = newfig(figsize=(6.2, 4.0))
    xc = bin_centers(E_EDGES_FINE, last=45)
    med, p68 = [], []
    for a, b in zip(E_EDGES_FINE[:-1], E_EDGES_FINE[1:]):
        m = es & (E >= a) & (E < b)
        if m.sum() < 10:
            med.append(np.nan); p68.append(np.nan); continue
        med.append(np.median(theta[m])); p68.append(np.percentile(theta[m], 68))
    ax.plot(xc, med, color=C_ES, lw=2, marker='o', ms=5, label='median opening angle')
    ax.plot(xc, p68, color=C_ES, lw=1.6, ls='--', marker='s', ms=4, alpha=0.8,
            label='68th percentile')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks([1, 2, 5, 10, 20, 40]); ax.set_xticklabels(['1', '2', '5', '10', '20', '40'])
    ax.set_yticks([2, 5, 10, 20, 40]); ax.set_yticklabels(['2', '5', '10', '20', '40'])
    style(ax, 'reconstructed main-cluster energy [MeV]', 'angle between electron and neutrino [deg]',
          'Irreducible ES kinematic smearing (truth, no reconstruction)')
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    ax.axvspan(1, 7, color=C_CC, alpha=0.08, lw=0)
    ax.text(1.15, 3.0, 'the band CT v80 keeps\nat P(ES)>0.8', color=C_CC, fontsize=8)
    save(fig, figdir, 'fig00_es_kinematics.png')


# =============================================================================
# Q1: truth-level topology vs energy
# =============================================================================
def q1_topology(scan, figdir):
    lab = scan['label']
    E = scan['particle_energy']
    es, cc = lab == 0, lab == 1
    ncl = scan['n_clusters_in_volume']
    nmar = scan['n_marley_clusters']
    nnon = scan['n_non_marley_clusters']
    davg = scan['avg_marley_cluster_distance_cm']
    dmax = scan['max_marley_cluster_distance_cm']

    emit('### Table 1 - truth-level volume topology vs reconstructed energy '
         '(test cats 597-621, all volumes)')
    emit()
    emit('| E [MeV] | N(ES) | n_clus ES | n_marley ES | n_nonmarley ES | f(n_mar>1) ES | med d_avg ES [cm] '
         '| N(CC) | n_clus CC | n_marley CC | n_nonmarley CC | f(n_mar>1) CC | med d_avg CC [cm] |')
    emit('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for a, b in zip(E_EDGES[:-1], E_EDGES[1:]):
        cells = [f'{a}-{b}' if b < 100 else f'>{a}']
        for m0 in (es, cc):
            m = m0 & (E >= a) & (E < b)
            n = int(m.sum())
            if n < 5:
                cells += [str(n), '-', '-', '-', '-', '-']
                continue
            dd = davg[m]
            dd = dd[dd > 0]
            cells += [f'{n}', f'{ncl[m].mean():.2f}', f'{nmar[m].mean():.2f}',
                      f'{nnon[m].mean():.2f}', f'{(nmar[m] > 1).mean():.3f}',
                      f'{np.median(dd):.1f}' if len(dd) > 5 else '-']
        emit('| ' + ' | '.join(cells) + ' |')
    emit()

    xc = bin_centers(E_EDGES_FINE, last=45)
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    fig.patch.set_facecolor(SURFACE)

    for ax, (var, ylab, stat, ttl) in zip(axs.ravel(), [
            (nmar, 'mean n_marley_clusters', 'mean', '(a) signal (MARLEY) cluster multiplicity'),
            (None, 'fraction with n_marley > 1', 'frac', '(b) fraction with a secondary signal cluster'),
            (nnon, 'mean n_non_marley_clusters', 'mean', '(c) radiological clusters: nearly identical for ES and CC'),
            (None, 'median distance [cm]', 'med', '(d) distance of secondary signal clusters')]):
        for m0, col, name in ((es, C_ES, 'ES'), (cc, C_CC, 'CC')):
            if ttl.startswith('(b)'):
                pr = profile(E[m0], (nmar[m0] > 1).astype(float), E_EDGES_FINE, 'frac')
            elif ttl.startswith('(d)'):
                sel = m0 & (davg > 0)
                pr = profile(E[sel], davg[sel], E_EDGES_FINE, 'median')
                pr2 = profile(E[m0 & (dmax > 0)], dmax[m0 & (dmax > 0)], E_EDGES_FINE, 'median')
                ax.plot(xc, [p[1] for p in pr2], color=col, lw=1.4, ls=':', marker='s', ms=4,
                        alpha=0.75, label=f'{name} max')
            else:
                pr = profile(E[m0], var[m0], E_EDGES_FINE, stat)
                if ttl.startswith('(c)'):
                    ax.set_ylim(0, 3.4)
            y = np.array([p[1] for p in pr])
            e = np.array([p[2] for p in pr])
            lb = f'{name} avg' if ttl.startswith('(d)') else name
            ax.errorbar(xc, y, yerr=e, color=col, lw=2, marker='o', ms=5, capsize=2, label=lb)
        style(ax, 'reconstructed main-cluster energy [MeV]', ylab, ttl)
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 40])
        ax.set_xticklabels(['1', '2', '5', '10', '20', '40'])
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle('CT v80 test cats: volume topology vs reconstructed energy  (ES = blue, CC = orange)',
                 color=INK, fontsize=11, x=0.01, ha='left')
    save(fig, figdir, 'fig01_topology_vs_energy.png')


# =============================================================================
# Q2: what drives the CT score
# =============================================================================
def q2_score(j, figdir):
    lab = j['label'].astype(int)
    E = j['particle_energy']
    p = j['p_es']
    z = np.log(np.clip(j['p_cc'], 1e-7, 1) / np.clip(j['p_es'], 1e-7, 1))
    adc = j['total_adc']
    npix = j['n_nonzero']
    ncl = j['n_clusters_in_volume']
    nmar = j['n_marley_clusters']
    es, cc = lab == 0, lab == 1
    lE, ladc, lpix = np.log(E), np.log1p(adc), np.log1p(npix)

    # ---- (a) score vs energy -------------------------------------------------
    emit('### Table 2 - CT score and working points vs reconstructed energy '
         '(balanced test set, 4000 ES + 4000 CC)')
    emit()
    emit('| E [MeV] | N(ES) | median P(ES) ES | eff(ES) @0.5 | eff(ES) @0.8 | N(CC) | median P(ES) CC '
         '| rej(CC) @0.5 | rej(CC) @0.8 |')
    emit('|---|---|---|---|---|---|---|---|---|')
    for a, b in zip(E_EDGES[:-1], E_EDGES[1:]):
        m = (E >= a) & (E < b)
        me, mc = m & es, m & cc
        if me.sum() < 5 or mc.sum() < 5:
            continue
        emit(f'| {a}-{b} | {me.sum()} | {np.median(p[me]):.3f} | {(p[me]>0.5).mean():.3f} | '
             f'{(p[me]>0.8).mean():.3f} | {mc.sum()} | {np.median(p[mc]):.3f} | '
             f'{(p[mc]<=0.5).mean():.3f} | {(p[mc]<=0.8).mean():.3f} |')
    emit()

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.patch.set_facecolor(SURFACE)
    xc = bin_centers(E_EDGES_FINE, last=45)
    ax = axs[0]
    for m0, col, name in ((es, C_ES, 'ES'), (cc, C_CC, 'CC')):
        med, lo, hi = [], [], []
        for a, b in zip(E_EDGES_FINE[:-1], E_EDGES_FINE[1:]):
            v = p[m0 & (E >= a) & (E < b)]
            if len(v) < 5:
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            med.append(np.median(v)); lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
        ax.fill_between(xc, lo, hi, color=col, alpha=0.16, lw=0)
        ax.plot(xc, med, color=col, lw=2, marker='o', ms=5, label=f'{name} median (band: IQR)')
    ax.axhline(0.8, color=INK2, lw=1.2, ls='--')
    ax.axhline(0.5, color=INK2, lw=1.0, ls=':')
    ax.text(1.05, 0.815, 'deployed threshold 0.80', color=INK2, fontsize=7.5)
    ax.text(1.05, 0.515, 'argmax 0.50', color=INK2, fontsize=7.5)
    ax.set_xscale('log'); ax.set_xticks([1, 2, 5, 10, 20, 40])
    ax.set_xticklabels(['1', '2', '5', '10', '20', '40'])
    ax.set_ylim(0, 1)
    style(ax, 'reconstructed main-cluster energy [MeV]', 'CT score P(ES)',
          '(a) the score slides with energy for BOTH classes')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='lower left')

    ax = axs[1]
    effs = {}
    for thr, ls, mk in ((0.5, ':', 's'), (0.8, '-', 'o')):
        pr = profile(E[es], (p[es] > thr).astype(float), E_EDGES_FINE, 'frac')
        ax.errorbar(xc, [q[1] for q in pr], yerr=[q[2] for q in pr], color=C_ES, lw=2, ls=ls,
                    marker=mk, ms=5, label=f'ES efficiency @ {thr}')
        pr = profile(E[cc], (p[cc] <= thr).astype(float), E_EDGES_FINE, 'frac')
        ax.errorbar(xc, [q[1] for q in pr], yerr=[q[2] for q in pr], color=C_CC, lw=2, ls=ls,
                    marker=mk, ms=5, label=f'CC rejection @ {thr}')
    ax.set_xscale('log'); ax.set_xticks([1, 2, 5, 10, 20, 40])
    ax.set_xticklabels(['1', '2', '5', '10', '20', '40'])
    ax.set_ylim(-0.02, 1.32)
    style(ax, 'reconstructed main-cluster energy [MeV]', 'efficiency / rejection',
          '(b) a fixed threshold acts as an energy cut')
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc='upper center', ncol=2)
    save(fig, figdir, 'fig02_score_and_efficiency_vs_energy.png')

    # ---- (b) conditional AUC -------------------------------------------------
    X4 = np.column_stack([lE, ncl, ladc, lpix])
    m4 = LogisticRegression(max_iter=5000).fit(X4, lab)
    p4 = m4.predict_proba(X4)[:, 1]

    emit('### Table 3 - conditional AUC in bins of reconstructed energy '
         '(CC vs ES; 0.5 = no separation)')
    emit()
    emit('| E [MeV] | N(ES) | N(CC) | CT v80 | n_clusters | total ADC | n_marley (truth) '
         '| logistic(logE, n_clus, ADC, n_pix) |')
    emit('|---|---|---|---|---|---|---|---|')
    rows = []
    for a, b in zip(E_EDGES[:-1], E_EDGES[1:]):
        m = (E >= a) & (E < b)
        if (m & es).sum() < 20 or (m & cc).sum() < 20:
            continue
        r = dict(lo=a, hi=b, nes=int((m & es).sum()), ncc=int((m & cc).sum()),
                 ct=roc_auc_score(lab[m], j['p_cc'][m]),
                 ncl=roc_auc_score(lab[m], ncl[m]),
                 adc=roc_auc_score(lab[m], adc[m]),
                 nmar=roc_auc_score(lab[m], nmar[m]),
                 log4=roc_auc_score(lab[m], p4[m]))
        rows.append(r)
        emit(f'| {a}-{b} | {r["nes"]} | {r["ncc"]} | {r["ct"]:.3f} | {r["ncl"]:.3f} | '
             f'{r["adc"]:.3f} | {r["nmar"]:.3f} | {r["log4"]:.3f} |')
    auc_marg = roc_auc_score(lab, j['p_cc'])
    emit(f'| **all (marginal)** | 4000 | 4000 | **{auc_marg:.3f}** | '
         f'{roc_auc_score(lab, ncl):.3f} | {roc_auc_score(lab, adc):.3f} | '
         f'{roc_auc_score(lab, nmar):.3f} | {roc_auc_score(lab, p4):.3f} |')
    emit()

    fig, ax = newfig(figsize=(7.2, 4.2))
    xs = np.arange(len(rows))
    series = [('CT v80', 'ct', C_ES, 'o', '-'),
              ('logistic(logE, n_clus, ADC, n_pix)', 'log4', C_CC, 's', '-'),
              ('n_marley_clusters (truth)', 'nmar', C_3, '^', '--'),
              ('total ADC', 'adc', C_4, 'D', '--'),
              ('n_clusters_in_volume', 'ncl', C_5, 'v', ':')]
    for name, key, col, mk, ls in series:
        ax.plot(xs, [r[key] for r in rows], color=col, lw=2, marker=mk, ms=6, ls=ls, label=name)
    ax.axhline(auc_marg, color=INK2, lw=1.2, ls='--')
    ax.text(0.05, auc_marg + 0.006, f'CT marginal AUC = {auc_marg:.3f} (all energies together)',
            color=INK2, fontsize=8)
    ax.axhline(0.5, color=GRID, lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{r["lo"]}-{r["hi"]}' if r['hi'] < 100 else f'>{r["lo"]}' for r in rows])
    ax.set_ylim(0.45, 0.90)
    style(ax, 'reconstructed main-cluster energy bin [MeV]', 'AUC within the energy bin',
          'At FIXED energy the CT still separates ES from CC (AUC ~ 0.75)')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='lower right', ncol=1)
    save(fig, figdir, 'fig03_conditional_auc.png')

    # conditioning on an OBSERVABLE energy proxy instead of truth energy
    emit('### Table 3b - conditional AUC in deciles of total ADC (an observable proxy for '
         'visible energy)')
    emit()
    emit('| total ADC range | N(ES) | N(CC) | CT v80 | n_clusters | n_marley (truth) |')
    emit('|---|---|---|---|---|---|')
    qa = np.quantile(adc, np.linspace(0, 1, 8))
    qa[-1] += 1
    for a, b in zip(qa[:-1], qa[1:]):
        m = (adc >= a) & (adc < b)
        if (m & es).sum() < 20 or (m & cc).sum() < 20:
            continue
        emit(f'| {a:.0f}-{b:.0f} | {int((m & es).sum())} | {int((m & cc).sum())} | '
             f'{roc_auc_score(lab[m], j["p_cc"][m]):.3f} | '
             f'{roc_auc_score(lab[m], ncl[m]):.3f} | {roc_auc_score(lab[m], nmar[m]):.3f} |')
    emit()

    # ---- (c) logistic models / deviance --------------------------------------
    emit('### Table 4 - simple models of the LABEL, fitted on the test set')
    emit()
    emit('| model | AUC | deviance explained | AUC gain vs CT (AUC-0.5 ratio) |')
    emit('|---|---|---|---|')
    ll0 = log_loss(lab, np.full(len(lab), 0.5))
    ct_gain = auc_marg - 0.5

    def fit_row(name, cols):
        X = np.column_stack(cols)
        mdl = LogisticRegression(max_iter=5000).fit(X, lab)
        pr = mdl.predict_proba(X)[:, 1]
        a = roc_auc_score(lab, pr)
        d = 1 - log_loss(lab, pr) / ll0
        emit(f'| {name} | {a:.4f} | {d:.4f} | {(a-0.5)/ct_gain:.2f} |')
        return a, d

    fit_row('label ~ log E', [lE])
    fit_row('label ~ log E + (log E)^2', [lE, lE ** 2])
    fit_row('label ~ n_clusters_in_volume', [ncl])
    fit_row('label ~ log(total ADC)', [ladc])
    fit_row('label ~ log E + n_clusters', [lE, ncl])
    fit_row('label ~ log E + n_clusters + log ADC + log n_pix', [lE, ncl, ladc, lpix])
    fit_row('label ~ n_marley_clusters (TRUTH, not available online)', [nmar])
    d_ct = 1 - log_loss(lab, np.clip(j['p_cc'], 1e-7, 1 - 1e-7)) / ll0
    emit(f'| **CT v80 (the CNN itself)** | **{auc_marg:.4f}** | **{d_ct:.4f}** | 1.00 |')
    emit()

    # ---- (d) how much of the CT SCORE is a function of a few scalars ---------
    def r2(y, cols):
        A = np.column_stack([np.ones(len(y))] + cols)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        return 1 - np.sum((y - A @ c) ** 2) / np.sum((y - y.mean()) ** 2)

    emit('### Table 5 - variance of the CT logit log[P(CC)/P(ES)] explained by simple variables (R^2)')
    emit()
    emit('| regressors | all | ES only | CC only |')
    emit('|---|---|---|---|')
    specs = [('cubic in log E', lambda m: [lE[m], lE[m] ** 2, lE[m] ** 3]),
             ('log(total ADC)', lambda m: [ladc[m]]),
             ('n_clusters_in_volume', lambda m: [ncl[m]]),
             ('n_marley_clusters (truth)', lambda m: [nmar[m]]),
             ('n_clusters + log ADC + log n_pix', lambda m: [ncl[m], ladc[m], lpix[m]]),
             ('cubic log E + n_clusters + log ADC + log n_pix',
              lambda m: [lE[m], lE[m] ** 2, lE[m] ** 3, ncl[m], ladc[m], lpix[m]])]
    r2vals = {}
    for name, f in specs:
        vals = [r2(z[m], f(m)) for m in (np.ones(len(z), bool), es, cc)]
        r2vals[name] = vals
        emit(f'| {name} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} |')
    emit()

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor(SURFACE)
    ax = axs[0]
    names = list(r2vals)
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, [r2vals[n][0] for n in names], color=C_ES, height=0.55)
    for yy, n in zip(ypos, names):
        ax.text(r2vals[n][0] + 0.012, yy, f'{r2vals[n][0]:.2f}', va='center', color=INK2, fontsize=8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([n.replace(' + ', '\n+ ') for n in names], fontsize=7.5, color=INK2)
    ax.set_xlim(0, 1)
    style(ax, 'R^2 of the CT logit', None, '(a) the CT score is ~80% four scalars')

    ax = axs[1]
    A = np.column_stack([np.ones(len(z)), lE, lE ** 2, lE ** 3, ncl, ladc, lpix])
    c, *_ = np.linalg.lstsq(A, z, rcond=None)
    zfit = A @ c
    for m0, col, name in ((es, C_ES, 'ES'), (cc, C_CC, 'CC')):
        ax.scatter(zfit[m0], z[m0], s=3, alpha=0.18, color=col, lw=0, label=name)
    lim = [min(z.min(), zfit.min()), max(z.max(), zfit.max())]
    ax.plot(lim, lim, color=INK2, lw=1, ls='--')
    style(ax, 'prediction from log E, n_clusters, ADC, n_pix', 'CT logit log[P(CC)/P(ES)]',
          f'(b) R^2 = {r2vals[names[-1]][0]:.2f}')
    lg = ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='upper left', markerscale=3)
    for h in lg.legend_handles:
        h.set_alpha(1)
    save(fig, figdir, 'fig04_logit_decomposition.png')

    return p4


# =============================================================================
# Q2/Q3: the bremsstrahlung test and the working-point fix
# =============================================================================
def q3_brems(j, figdir, w_point=None):
    lab = j['label'].astype(int)
    E = j['particle_energy']
    p = j['p_es']
    nmar = j['n_marley_clusters']
    nnon = j['n_non_marley_clusters']
    adc = j['total_adc']
    es, cc = lab == 0, lab == 1

    emit('### Table 6 - the bremsstrahlung test: ES events at FIXED reconstructed energy, split by secondary '
         'signal-cluster multiplicity')
    emit()
    emit('| E [MeV] | N(n_mar=1) | med P(ES) | eff@0.5 | eff@0.8 | N(n_mar>1) | med P(ES) | eff@0.5 '
         '| eff@0.8 | AUC(P(ES) tells them apart) |')
    emit('|---|---|---|---|---|---|---|---|---|---|')
    for a, b in zip(E_EDGES[:-1], E_EDGES[1:]):
        m = es & (E >= a) & (E < b)
        m1, m2 = m & (nmar == 1), m & (nmar > 1)
        if m1.sum() < 15 or m2.sum() < 15:
            continue
        y = np.concatenate([np.zeros(m1.sum()), np.ones(m2.sum())])
        sc = np.concatenate([p[m1], p[m2]])
        emit(f'| {a}-{b} | {m1.sum()} | {np.median(p[m1]):.3f} | {(p[m1]>0.5).mean():.3f} | '
             f'{(p[m1]>0.8).mean():.3f} | {m2.sum()} | {np.median(p[m2]):.3f} | '
             f'{(p[m2]>0.5).mean():.3f} | {(p[m2]>0.8).mean():.3f} | {1-roc_auc_score(y, sc):.3f} |')
    emit()

    emit('### Table 7 - the same for RADIOLOGICAL clusters (which carry no class information), '
         'clean ES (n_marley = 1) only')
    emit()
    emit('| E [MeV] | N(n_nonmar<=1) | med P(ES) | eff@0.5 | N(n_nonmar>=4) | med P(ES) | eff@0.5 |')
    emit('|---|---|---|---|---|---|---|')
    for a, b in zip(E_EDGES[:-1], E_EDGES[1:]):
        m = es & (E >= a) & (E < b) & (nmar == 1)
        m1, m2 = m & (nnon <= 1), m & (nnon >= 4)
        if m1.sum() < 15 or m2.sum() < 15:
            continue
        emit(f'| {a}-{b} | {m1.sum()} | {np.median(p[m1]):.3f} | {(p[m1]>0.5).mean():.3f} | '
             f'{m2.sum()} | {np.median(p[m2]):.3f} | {(p[m2]>0.5).mean():.3f} |')
    emit()

    # figure: median score vs energy split by multiplicity
    xc = bin_centers(E_EDGES_FINE, last=45)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor(SURFACE)
    ax = axs[0]
    for sel, col, ls, mk, name in (
            (es & (nmar == 1), C_ES, '-', 'o', 'ES, no secondary signal cluster'),
            (es & (nmar > 1), C_3, '--', '^', 'ES, >=1 secondary signal cluster'),
            (cc & (nmar > 1), C_CC, '-', 's', 'CC, >=1 de-excitation cluster')):
        med = []
        for a, b in zip(E_EDGES_FINE[:-1], E_EDGES_FINE[1:]):
            v = p[sel & (E >= a) & (E < b)]
            med.append(np.median(v) if len(v) >= 10 else np.nan)
        ax.plot(xc, med, color=col, lw=2, ls=ls, marker=mk, ms=5, label=name)
    ax.axhline(0.8, color=INK2, lw=1.2, ls='--')
    ax.axhline(0.5, color=INK2, lw=1.0, ls=':')
    ax.set_xscale('log'); ax.set_xticks([1, 2, 5, 10, 20, 40])
    ax.set_xticklabels(['1', '2', '5', '10', '20', '40'])
    ax.set_ylim(0, 1)
    style(ax, 'reconstructed energy [MeV]', 'median CT score P(ES)',
          '(a) a brems cluster costs an ES event ~0.2-0.3 in P(ES)')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='lower left')

    ax = axs[1]
    for sel, col, ls, mk, name in (
            (es & (nnon <= 1) & (nmar == 1), C_ES, '-', 'o', 'clean ES, <=1 radiological cluster'),
            (es & (nnon >= 4) & (nmar == 1), C_4, '--', 'D', 'clean ES, >=4 radiological clusters')):
        med = []
        for a, b in zip(E_EDGES_FINE[:-1], E_EDGES_FINE[1:]):
            v = p[sel & (E >= a) & (E < b)]
            med.append(np.median(v) if len(v) >= 10 else np.nan)
        ax.plot(xc, med, color=col, lw=2, ls=ls, marker=mk, ms=5, label=name)
    ax.axhline(0.8, color=INK2, lw=1.2, ls='--')
    ax.axhline(0.5, color=INK2, lw=1.0, ls=':')
    ax.set_xscale('log'); ax.set_xticks([1, 2, 5, 10, 20, 40])
    ax.set_xticklabels(['1', '2', '5', '10', '20', '40'])
    ax.set_ylim(0, 1)
    style(ax, 'reconstructed energy [MeV]', 'median CT score P(ES)',
          '(b) radiological pile-up also pushes ES towards CC')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='lower left')
    save(fig, figdir, 'fig05_brems_and_pileup.png')

    # ---- working points ------------------------------------------------------
    hi = E >= 10
    emit('### Table 8 - working points. "E>10 MeV" is the sample that carries the pointing power.')
    emit()
    emit('| selection | ES eff (all) | CC rej (all) | ES eff (E>10) | CC rej (E>10) '
         '| ES pointing information kept |')
    emit('|---|---|---|---|---|---|')
    wtot = float(w_point[es].sum()) if w_point is not None else np.nan

    def row(keep, name):
        wk = float(w_point[es & keep].sum()) / wtot if w_point is not None else np.nan
        emit(f'| {name} | {keep[es].mean():.3f} | {1-keep[cc].mean():.3f} | '
             f'{keep[es & hi].mean():.3f} | {1-keep[cc & hi].mean():.3f} | {wk:.3f} |')
        return (keep[es & hi].mean(), 1 - keep[cc & hi].mean(),
                keep[es].mean(), 1 - keep[cc].mean())

    glob_pts, flat_pts = [], []
    for thr in (0.5, 0.668, 0.8):
        glob_pts.append(row(p > thr, f'global P(ES) > {thr}'))
    # ADC-decile-flattened thresholds, thresholds derived on one half of the cats
    cats = np.array([str(c) for c in j['cat']])
    ucats = np.unique(cats)
    fold = np.isin(cats, ucats[::2])           # A/B split by burst
    q = np.quantile(adc, np.linspace(0, 1, 11))
    q[-1] += 1

    def flat_keep(target):
        keep = np.zeros(len(p), bool)
        for lo, up in zip(q[:-1], q[1:]):
            m = (adc >= lo) & (adc < up)
            for train, apply_ in ((fold, ~fold), (~fold, fold)):
                mt = m & train & es
                if mt.sum() < 10:
                    continue
                thr = np.quantile(p[mt], 1 - target)
                keep[m & apply_] = p[m & apply_] > thr
        return keep

    for target in (0.7, 0.5, 0.4, 0.3, 0.2, 0.15):
        flat_pts.append(row(flat_keep(target),
                            f'per-ADC-decile threshold, ES eff {target:.2f} in every decile'))
    emit()
    emit('(The per-decile thresholds are derived on half of the test bursts and applied to the other '
         'half, so they are not tuned on the events they select.)')
    emit()

    fig, ax = newfig(figsize=(6.4, 4.4))
    g = np.array(glob_pts)
    f_ = np.array(flat_pts)
    ax.plot(g[:, 1], g[:, 0], color=C_CC, lw=2, marker='o', ms=7, label='global threshold on P(ES)')
    ax.plot(f_[:, 1], f_[:, 0], color=C_ES, lw=2, marker='s', ms=7,
            label='threshold flattened in total ADC')
    for (x, y), t in zip(g[:, [1, 0]], ['0.50', '0.67', '0.80']):
        ax.annotate(t, (x, y), textcoords='offset points', xytext=(6, -10), fontsize=7.5, color=C_CC)
    for (x, y), t in zip(f_[:, [1, 0]], ['0.70', '0.50', '0.40', '0.30', '0.20', '0.15']):
        ax.annotate(t, (x, y), textcoords='offset points', xytext=(-22, 4), fontsize=7.5, color=C_ES)
    style(ax, 'CC rejection (E > 10 MeV)', 'ES efficiency (E > 10 MeV)',
          'Same network, same scores - only the threshold policy changes')
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc='lower left')
    save(fig, figdir, 'fig06_working_points.png')


# =============================================================================
# Q3: image-level look at the misclassifications
# =============================================================================
def q3_images(j, figdir, n_per_cat=200, n_examples=3, elo=10.0, ehi=25.0,
              compute_features=True):
    """Image-level comparison of the four (truth, tag) categories in one energy band."""
    from scipy import ndimage
    lab = j['label'].astype(int)
    E = j['particle_energy']
    p = j['p_es']
    es, cc = lab == 0, lab == 1
    band = (E >= elo) & (E < ehi)

    cats = {
        'ES tagged CC (P(ES)<0.2)': band & es & (p < 0.2),
        'ES tagged ES (P(ES)>0.5)': band & es & (p > 0.5),
        'CC tagged CC (P(ES)<0.2)': band & cc & (p < 0.2),
        'CC tagged ES (P(ES)>0.5)': band & cc & (p > 0.5),
    }
    rng = np.random.RandomState(11)
    sel_idx = {}
    for k, m in cats.items():
        idx = np.where(m)[0]
        rng.shuffle(idx)
        sel_idx[k] = idx[:n_per_cat]
        print(f'  {k}: {m.sum()} available, using {len(sel_idx[k])}')

    if not compute_features:
        _examples_figure(j, sel_idx, figdir, n_examples, elo, ehi)
        return

    # group by file so each npz is opened once
    need = {}
    for k, idx in sel_idx.items():
        for i in idx:
            need.setdefault(str(j['src_file'][i]), []).append(i)

    feats = {i: None for k in sel_idx for i in sel_idx[k]}
    struct = np.ones((3, 3), bool)
    for f, ii in need.items():
        arr = np.load(f, allow_pickle=True)['images']
        for i in ii:
            img = np.asarray(arr[int(j['src_index'][i])], dtype=np.float32)
            nz = img > 0
            if not nz.any():
                feats[i] = dict(wire=0, tick=0, ncomp=0, frac_out=0.0, adc=0.0, npix=0)
                continue
            ch, tk = np.where(nz)
            # dilate a little so hits of one cluster merge, then count components
            dil = ndimage.binary_dilation(nz, structure=struct, iterations=3)
            liml, ncomp = ndimage.label(dil, structure=struct)
            sums = ndimage.sum(img, liml, index=np.arange(1, ncomp + 1))
            frac_out = 1.0 - (sums.max() / sums.sum()) if sums.sum() > 0 else 0.0
            feats[i] = dict(wire=int(ch.max() - ch.min() + 1), tick=int(tk.max() - tk.min() + 1),
                            ncomp=int(ncomp), frac_out=float(frac_out),
                            adc=float(img.sum()), npix=int(nz.sum()))

    emit(f'### Table 9 - image-level comparison, reconstructed energy {elo:.0f}-{ehi:.0f} MeV '
         f'({n_per_cat} volumes sampled per category)')
    emit()
    emit('| category | N | med E [MeV] | med P(ES) | med n_clus | med n_marley | med n_nonmarley '
         '| med total ADC | med n_pix | med wire extent | med tick extent | med n blobs '
         '| med ADC fraction outside main blob |')
    emit('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for k, idx in sel_idx.items():
        if len(idx) == 0:
            continue
        g = lambda key: np.median([feats[i][key] for i in idx])
        emit(f'| {k} | {len(idx)} | {np.median(E[idx]):.1f} | {np.median(p[idx]):.3f} | '
             f'{np.median(j["n_clusters_in_volume"][idx]):.1f} | '
             f'{np.median(j["n_marley_clusters"][idx]):.1f} | '
             f'{np.median(j["n_non_marley_clusters"][idx]):.1f} | '
             f'{g("adc"):.0f} | {g("npix"):.0f} | {g("wire"):.0f} | {g("tick"):.0f} | '
             f'{g("ncomp"):.0f} | {g("frac_out"):.3f} |')
    emit()

    # distributions
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    fig.patch.set_facecolor(SURFACE)
    colors = [C_ES, C_3, C_CC, C_4]
    for ax, (key, xlab, bins) in zip(axs, [
            ('ncomp', 'number of separated blobs in the volume', np.arange(0.5, 12.5, 1)),
            ('frac_out', 'ADC fraction outside the main blob', np.linspace(0, 1, 21)),
            ('wire', 'extent along wires [channels]', np.linspace(0, 208, 27))]):
        for (k, idx), col in zip(sel_idx.items(), colors):
            if len(idx) == 0:
                continue
            v = np.array([feats[i][key] for i in idx], dtype=float)
            ax.hist(v, bins=bins, histtype='step', lw=2, color=col, density=True, label=k)
        style(ax, xlab, 'normalised', None)
    axs[0].legend(frameon=False, fontsize=6.8, labelcolor=INK2)
    fig.suptitle(f'Image-level properties, reconstructed energy {elo:.0f}-{ehi:.0f} MeV',
                 color=INK, fontsize=10, x=0.01, ha='left')
    save(fig, figdir, 'fig07_misclassification_image_properties.png')

    _examples_figure(j, sel_idx, figdir, n_examples, elo, ehi)


def _examples_figure(j, sel_idx, figdir, n_examples, elo, ehi):
    """Example volumes, all panels on the SAME axes so they can be compared."""
    from scipy import ndimage
    E = j['particle_energy']
    p = j['p_es']
    fig, axs = plt.subplots(len(sel_idx), n_examples,
                            figsize=(3.6 * n_examples, 1.9 * len(sel_idx)))
    fig.patch.set_facecolor(SURFACE)
    for r, (k, idx) in enumerate(sel_idx.items()):
        pick = idx[:n_examples]
        for c in range(n_examples):
            ax = axs[r, c]
            ax.set_facecolor('#000000')
            if c >= len(pick):
                ax.axis('off')
                continue
            i = pick[c]
            arr = np.load(str(j['src_file'][i]), allow_pickle=True)['images']
            img = np.asarray(arr[int(j['src_index'][i])], dtype=np.float32)
            # dilate so that single-pixel hits stay visible in a full-volume view
            vis = ndimage.grey_dilation(np.log1p(img), size=(3, 9))
            ax.imshow(vis, aspect='auto', origin='lower', cmap='inferno',
                      extent=[0, img.shape[1], 0, img.shape[0]],
                      vmin=0, vmax=max(vis.max(), 1e-3))
            ax.set_title(f'E={E[i]:.1f} MeV  P(ES)={p[i]:.2f}  '
                         f'n_mar={int(j["n_marley_clusters"][i])}  '
                         f'n_bkg={int(j["n_non_marley_clusters"][i])}',
                         fontsize=7, color=INK, loc='left')
            ax.tick_params(colors=INK2, labelsize=6)
            if c == 0:
                ax.set_ylabel(k.replace(' (', '\n('), fontsize=7, color=INK2)
            else:
                ax.set_yticklabels([])
            if r == len(sel_idx) - 1:
                ax.set_xlabel('time tick', fontsize=7, color=INK2)
            else:
                ax.set_xticklabels([])
    fig.suptitle(f'Example volumes, reconstructed energy {elo:.0f}-{ehi:.0f} MeV - full 208 x 1242 image, '
                 'identical axes (log1p ADC, dilated for visibility)',
                 color=INK, fontsize=9.5, x=0.01, ha='left')
    save(fig, figdir, 'fig08_example_volumes.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--join', required=True)
    ap.add_argument('--truth-scan', required=True)
    ap.add_argument('--figdir', required=True)
    ap.add_argument('--tables', required=True)
    ap.add_argument('--skip-images', action='store_true')
    ap.add_argument('--examples-only', action='store_true',
                    help='remake only the example-volume figure (skips the slow feature scan)')
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)

    j = dict(np.load(args.join, allow_pickle=True))
    scan = dict(np.load(args.truth_scan))
    print(f'join: {len(j["p_es"])} volumes; truth scan: {len(scan["label"])} volumes')

    emit('<!-- generated by channel_tagging/ana/ct_v80_energy_topology.py -->')
    emit()
    theta, w_point, t68 = pointing_weights(j)
    q0_pointing(j, theta, t68, args.figdir)
    q1_topology(scan, args.figdir)
    q2_score(j, args.figdir)
    q3_brems(j, args.figdir, w_point)
    if args.examples_only:
        q3_images(j, args.figdir, compute_features=False)
    elif not args.skip_images:
        q3_images(j, args.figdir)

    with open(args.tables, 'w') as f:
        f.write('\n'.join(_TABLES) + '\n')
    print(f'Tables written to {args.tables}')


if __name__ == '__main__':
    main()
