#!/usr/bin/env python3
"""
Generate a PDF summary of the CT improvement campaign (v79 baseline -> v80+).

Re-runnable: pulls results.json from each output directory it can find, so the
PDF stays current as new trainings finish.

Usage:
    python3 channel_tagging/ana/ct_improvement_summary.py [-o output.pdf]
"""

import os
import json
import glob
import argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

NN_BASE = '/eos/user/e/evilla/dune/sn-tps/neural_networks/channel_tagging'
DEFAULT_OUT = os.path.join(NN_BASE, 'CT_improvement_summary.pdf')

# Static entry for the v79 baseline (its results live in the old neural-networks
# area and its training data was deleted; numbers from condor log CT_v79_mem60_8542803).
V79_BASELINE = {
    'name': 'v79 (baseline)',
    'accuracy': 0.683,
    'auc': None,
    'es_recall': 0.61,
    'cc_recall': 0.75,
    'notes': 'e2p0 volumes (deleted), per-image max norm, random split (leaky)',
}


def collect_runs():
    """Find v80+ runs with results.json under NN_BASE."""
    runs = []
    for res_file in sorted(glob.glob(os.path.join(NN_BASE, 'ct_*_v8*_*/results.json'))) + \
                    sorted(glob.glob(os.path.join(NN_BASE, 'ct_volume_v8*_*/results.json'))) + \
                    sorted(glob.glob(os.path.join(NN_BASE, 'ct_points_v8*_*/results.json'))):
        try:
            with open(res_file) as f:
                r = json.load(f)
        except Exception:
            continue
        tm = r.get('test_metrics', {})
        cm = np.array(tm.get('confusion_matrix_normalized', [[np.nan] * 2] * 2))
        runs.append({
            'name': r.get('model_name', os.path.basename(os.path.dirname(res_file))),
            'dir': os.path.dirname(res_file),
            'timestamp': r.get('timestamp', ''),
            'accuracy': tm.get('accuracy'),
            'auc': tm.get('auc'),
            'es_recall': float(cm[0, 0]),
            'cc_recall': float(cm[1, 1]),
            'cm': cm,
            'n_train': r.get('data_summary', {}).get('n_train'),
            'epochs': r.get('epochs_trained'),
            'preprocessing': r.get('preprocessing', {}).get('image', '?'),
            'aux': bool(r.get('preprocessing', {}).get('aux_features')),
        })
    # dedup by dir
    seen, uniq = set(), []
    for r in runs:
        if r['dir'] not in seen:
            seen.add(r['dir'])
            uniq.append(r)
    return uniq


def text_page(pdf, title, lines, fontsize=11):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=16, weight='bold')
    fig.text(0.08, 0.90, '\n'.join(lines), fontsize=fontsize, va='top', family='monospace')
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output', default=DEFAULT_OUT)
    args = ap.parse_args()

    runs = collect_runs()

    with PdfPages(args.output) as pdf:
        # ---- page 1: motivation ----
        text_page(pdf, 'CT Improvement Campaign — Summary', [
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            'Branch: feature/ct-v80-burst-volumes (refactor-ml-for-pointing)',
            '',
            'GOAL',
            '  Improve ES-vs-CC channel tagging for SN online pointing.',
            '',
            'PROBLEMS FOUND IN THE v79 BASELINE (68.3% acc)',
            '  1. Domain shift: trained on e2p0 prod volumes (since DELETED from',
            '     EOS), while snop-pipeline runs inference on e3p0 burst-sample',
            '     volumes. Model never saw its deployment distribution.',
            '  2. Preprocessing mismatch: v79 trained with per-image max norm,',
            '     pipeline channel_tagger feeds raw ADC. Max-norm also erases',
            '     absolute amplitude (energy) information.',
            '  3. Split leakage: random sample-level split mixes correlated',
            '     volumes from the same burst across train/test.',
            '',
            'V80 DESIGN',
            '  - Data: sn-burst-samples cat*_volume_images_..._e3p0/X volumes',
            '    (EOS project space, 618 cats) = the production data product.',
            '  - Cat-level splits: train 400-571, val 572-596, test 597-621.',
            '    Cats 1-399 reserved untouched for unbiased pipeline evaluation.',
            '  - Preprocessing: log1p(ADC), recorded in results.json.',
            '  - Labels: ES=0, CC=1 (pipeline convention, P(ES)=softmax[:,0]).',
            '  - Variants:',
            '      v80      image-only CNN (BatchNorm, flip augment, early stop)',
            '      v80_aux  + truth-free aux branch: n_clusters_in_volume,',
            '               log1p(total ADC), log1p(n nonzero pixels).',
            '               Physics: CC has de-excitation gamma blips -> higher',
            '               cluster multiplicity than single-electron ES.',
            '',
            'SPARSE APPROACHES (in progress)',
            '  Volume images are ~99.9% empty (about 240 hit pixels / 258k).',
            '  - v81 (TF, no new deps): point-cloud model on nonzero pixels',
            '    (channel, tick, ADC) with DeepSets/PointNet-style encoder.',
            '    Avoids GAP dilution of small gamma blips; ~1000x smaller input.',
            '  - True sparse convs (spconv/MinkowskiEngine) require PyTorch',
            '    extensions not in LCG; possible follow-up via local_packages.',
            '',
            'HONEST-COMPARISON CAVEAT',
            '  v80+ numbers use a stricter leakage-free split and different data',
            '  than v79; the definitive comparison is the snop-pipeline scenario',
            '  analysis on held-out cats 1-399.',
        ], fontsize=10)

        # ---- page 2: results table + bar chart ----
        rows = [V79_BASELINE] + runs
        fig, (ax_t, ax_b) = plt.subplots(2, 1, figsize=(8.27, 11.69),
                                         gridspec_kw={'height_ratios': [1, 1.2]})
        ax_t.axis('off')
        col_labels = ['model', 'acc', 'AUC', 'ES recall', 'CC recall']
        cells = []
        for r in rows:
            cells.append([
                r['name'],
                f"{r['accuracy']:.3f}" if r.get('accuracy') is not None else '—',
                f"{r['auc']:.3f}" if r.get('auc') else '—',
                f"{r['es_recall']:.2f}" if r.get('es_recall') is not None else '—',
                f"{r['cc_recall']:.2f}" if r.get('cc_recall') is not None else '—',
            ])
        table = ax_t.table(cellText=cells, colLabels=col_labels, loc='center',
                           cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)
        ax_t.set_title('Test-set results', fontsize=14, weight='bold')

        names = [r['name'] for r in rows]
        accs = [r['accuracy'] or 0 for r in rows]
        colors = ['#888888'] + ['#2266aa'] * (len(rows) - 1)
        ax_b.barh(range(len(rows)), accs, color=colors)
        ax_b.set_yticks(range(len(rows)))
        ax_b.set_yticklabels(names)
        ax_b.set_xlim(0.5, 1.0)
        ax_b.axvline(V79_BASELINE['accuracy'], color='k', ls='--', lw=1,
                     label='v79 baseline')
        ax_b.set_xlabel('Test accuracy')
        ax_b.invert_yaxis()
        ax_b.legend()
        for i, a in enumerate(accs):
            if a:
                ax_b.text(a + 0.005, i, f'{a:.3f}', va='center', fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- one page per run: confusion matrix + details ----
        for r in runs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
            cm = r['cm']
            im = ax1.imshow(cm, vmin=0, vmax=1, cmap='Blues')
            for (i, j), v in np.ndenumerate(cm):
                ax1.text(j, i, f'{v:.2f}', ha='center', va='center',
                         color='white' if v > 0.5 else 'black')
            ax1.set_xticks([0, 1]); ax1.set_xticklabels(['ES', 'CC'])
            ax1.set_yticks([0, 1]); ax1.set_yticklabels(['ES', 'CC'])
            ax1.set_xlabel('Predicted'); ax1.set_ylabel('True')
            ax1.set_title('Confusion matrix (normalized)')
            fig.colorbar(im, ax=ax1, shrink=0.8)

            ax2.axis('off')
            info = [
                f"model:       {r['name']}",
                f"timestamp:   {r['timestamp']}",
                f"accuracy:    {r['accuracy']:.4f}" if r['accuracy'] else '',
                f"AUC:         {r['auc']:.4f}" if r['auc'] else '',
                f"n_train:     {r['n_train']}",
                f"epochs:      {r['epochs']}",
                f"preproc:     {r['preprocessing']}",
                f"aux branch:  {r['aux']}",
                '',
                'output dir:',
                f"  {r['dir']}",
            ]
            ax2.text(0, 0.95, '\n'.join(info), va='top', fontsize=10,
                     family='monospace')
            fig.suptitle(r['name'], fontsize=14, weight='bold')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f'Summary PDF written to {args.output} ({len(runs)} v80+ runs found)')


if __name__ == '__main__':
    main()
