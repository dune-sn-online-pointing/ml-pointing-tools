#!/usr/bin/env python3
"""Channel tagging training v85: the v83 recipe on radiological-MASKED volumes.

v85 = exactly the v83 configuration (same architecture, caps, cat splits, seed,
log1p preprocessing, no aux branch, common-energy-spectrum sample weights) trained
on the radiological-masked volume images
`<cat>_volume_images_tick3_ch2_min2_tot3_e3p0_radmask/X` (40 cm / 2 MeV image-level
blob mask) instead of the unmasked ones.

The one piece of machinery this file adds is **variant loading**, and it exists to
keep the comparison honest:

  The radmask product is missing one source file in each of cats 501 and 514 (both
  in the TRAIN range) because those files are corrupt.  If the loader globbed the
  radmask directories directly, the file lists would be 2 shorter, `rng.shuffle`
  would consume a different amount of randomness, and every downstream draw -
  including the VALIDATION and TEST selections - would differ from v80/v83/v84.
  The 2x2 model-x-test-images comparison would then not be paired.

  So selection is driven by the UNMASKED file lists (identical lengths to v80/v83,
  hence an identical RandomState stream), and each selected file is read from its
  radmask counterpart.  Files with no radmask counterpart are skipped with a
  warning, exactly as the original loader skipped the corrupt ones.  The result:
  v85's validation and test selections are bit-for-bit the v80/v83 selections, and
  its training set differs from v83's only by the volumes in those 2 corrupt files.

Everything else is imported from `train_ct_volume_v83.py` so the architecture, the
reweighting and the dataset plumbing cannot drift.

Labels: ES=0, CC=1.
"""

import sys
import os
import json
import glob
import argparse
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'lib'))

import numpy as np

from ct_volume_features import IMAGE_SHAPE, aux_feature_names, compute_aux_features, \
    preprocess_image
import train_ct_volume_v83 as base


def parse_args():
    p = argparse.ArgumentParser(description='Train CT v85 on radmask burst-sample volumes')
    p.add_argument('--json', '-j', type=str, required=True, help='JSON config file')
    p.add_argument('--test-local', action='store_true',
                   help='Tiny local run: few cats, few samples, 2 epochs')
    return p.parse_args()


def cat_dir_pairs(base_dir, primary_fmt, variant_fmt, cat_lo, cat_hi):
    """(primary_dir, variant_dir) for cats whose PRIMARY dir exists.

    The primary directory drives selection; the variant directory supplies pixels.
    """
    out = []
    for i in range(cat_lo, cat_hi + 1):
        cat = f'cat{i:06d}'
        pd = os.path.join(base_dir, cat, primary_fmt.format(cat=cat))
        vd = os.path.join(base_dir, cat, variant_fmt.format(cat=cat)) if variant_fmt else pd
        if os.path.isdir(pd):
            out.append((pd, vd))
    return out


def load_split_variant(dir_pairs, max_per_class, rng, split_name='', per_image_norm=None,
                       aux_include_energy=False):
    """v83's load_split, but selection uses the primary dirs and pixels come from
    the variant dirs.  RandomState consumption is identical to v83/v80."""
    files_per_class = {0: [], 1: []}
    prim_to_var = {}
    for pd, vd in dir_pairs:
        for label, pat in ((0, 'es_*.npz'), (1, 'cc_*.npz')):
            fs = sorted(glob.glob(os.path.join(pd, pat)))
            files_per_class[label].extend(fs)
            for f in fs:
                prim_to_var[f] = os.path.join(vd, os.path.basename(f))

    images_list, labels_list, aux_list, energy_list = [], [], [], []
    adc_list, cat_list, src_file_list, src_idx_list = [], [], [], []
    counts = {0: 0, 1: 0}
    n_missing = 0

    for label in (0, 1):
        files = files_per_class[label]
        rng.shuffle(files)                      # <- identical stream to v80/v83
        last_milestone = 0
        for f in files:
            if counts[label] >= max_per_class:
                break
            vf = prim_to_var[f]
            if not os.path.exists(vf):
                n_missing += 1
                print(f'  Warning: no variant image file for {os.path.basename(f)} '
                      f'({os.path.dirname(vf)}) - skipped', flush=True)
                continue
            try:
                data = np.load(vf, allow_pickle=True)
                imgs = data['images']
                metadata = data['metadata'] if 'metadata' in data else [None] * len(imgs)
            except Exception as e:
                print(f'  Warning: failed to load {vf}: {e}')
                continue
            cat = base._cat_of(f)
            for idx in range(len(imgs)):
                if counts[label] >= max_per_class:
                    break
                img = np.asarray(imgs[idx], dtype=np.float32)
                if img.shape != IMAGE_SHAPE:
                    continue
                meta = metadata[idx] if idx < len(metadata) else None
                aux = compute_aux_features(img, meta, include_energy=aux_include_energy)
                energy = float(meta.get('particle_energy', np.nan)) if isinstance(meta, dict) \
                    else np.nan
                images_list.append(preprocess_image(img, per_image_norm).astype(np.float16))
                labels_list.append(label)
                aux_list.append(aux)
                energy_list.append(energy)
                adc_list.append(float(img.sum()))
                cat_list.append(cat)
                src_file_list.append(vf)
                src_idx_list.append(idx)
                counts[label] += 1
            if counts[label] // 5000 > last_milestone:
                last_milestone = counts[label] // 5000
                print(f'  [{split_name}] class {label}: {counts[label]} samples...', flush=True)

    images = np.stack(images_list).astype(np.float16)[..., np.newaxis]
    labels = np.array(labels_list, dtype=np.int32)
    perm = rng.permutation(len(images))         # <- identical stream to v80/v83
    out = dict(images=images[perm], labels=labels[perm],
               aux=np.array(aux_list, dtype=np.float32)[perm],
               energies=np.array(energy_list, dtype=np.float32)[perm],
               total_adc=np.array(adc_list, dtype=np.float32)[perm],
               cats=np.array(cat_list)[perm],
               src_files=np.array(src_file_list)[perm],
               src_idx=np.array(src_idx_list, dtype=np.int32)[perm])
    print(f'[{split_name}] loaded: ES={counts[0]}, CC={counts[1]}, '
          f'shape={out["images"].shape}, {n_missing} files missing in variant', flush=True)
    return out, n_missing


def main():
    args = parse_args()
    with open(args.json) as f:
        config = json.load(f)

    dcfg, mcfg, tcfg, ocfg = config['data'], config['model'], config['training'], config['output']
    use_aux = bool(mcfg.get('use_aux_features', False))
    aux_include_energy = bool(mcfg.get('aux_include_energy', False))
    per_image_norm = dcfg.get('per_image_normalization', None)
    do_reweight = bool(tcfg.get('energy_reweight', False))
    reweight_edges = tcfg.get('reweight_bin_edges', base.DEFAULT_REWEIGHT_EDGES)
    model_name = config.get('model_name', 'ct_volume_v85')
    aux_names = aux_feature_names(include_energy=aux_include_energy)

    primary_fmt = dcfg['volume_subdir_fmt']
    variant_fmt = dcfg.get('image_subdir_fmt', None)

    max_dirs = None
    if args.test_local:
        print('*** TEST-LOCAL MODE: tiny caps, 2 epochs ***')
        dcfg['max_train_per_class'] = 40
        dcfg['max_val_per_class'] = 10
        dcfg['max_test_per_class'] = 10
        max_dirs = 2
        tcfg['epochs'] = 2

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(ocfg['base_dir'], f'{model_name}_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output directory: {out_dir}')
    print(f'Selection driven by : {primary_fmt}')
    print(f'Pixels loaded from  : {variant_fmt or primary_fmt}')

    rng = np.random.RandomState(tcfg.get('seed', 42))
    splits, missing = {}, {}
    for split, ck, cap in (('train', 'train_cats', 'max_train_per_class'),
                           ('val', 'val_cats', 'max_val_per_class'),
                           ('test', 'test_cats', 'max_test_per_class')):
        lo, hi = dcfg[ck]
        pairs = cat_dir_pairs(dcfg['burst_samples_dir'], primary_fmt, variant_fmt, lo, hi)
        if max_dirs:
            pairs = pairs[:max_dirs]
        print(f'\n[{split}] cats {lo}-{hi}: {len(pairs)} cat dirs found')
        if not pairs:
            raise RuntimeError(f'No cat dirs for split {split} in range {lo}-{hi}')
        splits[split], missing[split] = load_split_variant(
            pairs, dcfg[cap], rng, split_name=split, per_image_norm=per_image_norm,
            aux_include_energy=aux_include_energy)

    import tensorflow as tf
    from tensorflow import keras

    tr, va, te = splits['train'], splits['val'], splits['test']
    X_tr, y_tr, aux_tr = tr['images'], tr['labels'], tr['aux']
    X_va, y_va, aux_va = va['images'], va['labels'], va['aux']
    X_te, y_te, aux_te = te['images'], te['labels'], te['aux']

    aux_mean = aux_tr.mean(axis=0)
    aux_std = aux_tr.std(axis=0) + 1e-6
    aux_tr = (aux_tr - aux_mean) / aux_std
    aux_va = (aux_va - aux_mean) / aux_std
    aux_te = (aux_te - aux_mean) / aux_std

    reweight_info = {}
    w_tr = w_va = None
    if do_reweight:
        w_tr, reweight_info['train'] = base.energy_reweight(tr['energies'], y_tr,
                                                            reweight_edges, 'train')
        w_va, reweight_info['val'] = base.energy_reweight(va['energies'], y_va,
                                                          reweight_edges, 'val')

    bs = tcfg['batch_size']
    ds_tr = base.make_dataset(X_tr, y_tr, aux_tr, use_aux, bs,
                              augment=tcfg.get('augment', True), shuffle=True, weights=w_tr)
    ds_va = base.make_dataset(X_va, y_va, aux_va, use_aux, bs, augment=False, shuffle=False,
                              weights=w_va)
    ds_te = base.make_dataset(X_te, y_te, aux_te, use_aux, bs, augment=False, shuffle=False)

    model = base.build_model(use_aux, mcfg['filter_list'], mcfg['dense_units'],
                             mcfg['dropout_rate'], n_aux=len(aux_names))
    ck = dict(optimizer=keras.optimizers.Adam(learning_rate=tcfg['learning_rate']),
              loss=keras.losses.SparseCategoricalCrossentropy())
    ck['weighted_metrics' if do_reweight else 'metrics'] = ['accuracy']
    model.compile(**ck)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(out_dir, 'best_model.keras'),
                                        monitor='val_accuracy', save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=tcfg.get('patience', 10),
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,
                                          min_lr=1e-5, verbose=1),
        keras.callbacks.CSVLogger(os.path.join(out_dir, 'training_history.csv')),
    ]
    history = model.fit(ds_tr, validation_data=ds_va, epochs=tcfg['epochs'],
                        callbacks=callbacks, verbose=2)

    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
    y_prob = model.predict(ds_te, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    acc = float((y_pred == y_te).mean())
    auc = float(roc_auc_score(y_te, y_prob[:, 1]))
    cm = confusion_matrix(y_te, y_pred, normalize='true')
    report = classification_report(y_te, y_pred, target_names=['ES', 'CC'])
    print(f'\nTest accuracy: {acc:.4f}   AUC: {auc:.4f}')
    print(cm); print(report)

    np.savez_compressed(
        os.path.join(out_dir, 'test_predictions.npz'),
        predictions=y_prob, true_labels=y_te, energies=te['energies'],
        total_adc=te['total_adc'], cat=te['cats'],
        src_file=te['src_files'], src_index=te['src_idx'],
        aux_features=aux_te, aux_feature_names=np.array(aux_names))

    results = {
        'model_name': model_name, 'timestamp': timestamp, 'config': config,
        'label_convention': {'ES': 0, 'CC': 1},
        'preprocessing': {
            'image': 'log1p' if not per_image_norm else f'log1p_div_per_image_{per_image_norm}',
            'per_image_normalization': per_image_norm if per_image_norm else False,
            'image_variant': variant_fmt or primary_fmt,
            'selection_driven_by': primary_fmt,
            'aux_features': aux_names if use_aux else [],
            'aux_mean': aux_mean.tolist(), 'aux_std': aux_std.tolist(),
            'feature_module': 'channel_tagging/lib/ct_volume_features.py',
        },
        'energy_reweighting': {
            'enabled': do_reweight,
            'variable': 'particle_energy (reconstructed main-cluster energy, MeV)',
            'scheme': 'w_c(b) = target(b)/p_c(b), target = (p_ES + p_CC)/2, '
                      'renormalized to mean weight 1 per class; applied to train and val',
            'per_split': reweight_info,
        },
        'variant_loading': {
            'note': 'selection driven by the unmasked file lists so the RandomState '
                    'stream (and hence the val/test selections) match v80/v83/v84 exactly',
            'files_missing_in_variant': missing,
        },
        'data_summary': {
            'n_train': int(len(y_tr)), 'n_val': int(len(y_va)), 'n_test': int(len(y_te)),
            'train_cats': dcfg['train_cats'], 'val_cats': dcfg['val_cats'],
            'test_cats': dcfg['test_cats'],
        },
        'test_metrics': {'accuracy': acc, 'auc': auc,
                         'confusion_matrix_normalized': cm.tolist(),
                         'classification_report': report},
        'epochs_trained': len(history.history.get('loss', [])),
    }
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, vmin=0, vmax=1, cmap='Blues')
        for (i, jj), v in np.ndenumerate(cm):
            ax.text(jj, i, f'{v:.2f}', ha='center', va='center')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['ES', 'CC'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['ES', 'CC'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'{model_name} acc={acc:.3f} auc={auc:.3f}')
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150)
    except Exception as e:
        print(f'Warning: could not save confusion matrix plot: {e}')

    print(f'\nDone. Results in {out_dir}')


if __name__ == '__main__':
    main()
