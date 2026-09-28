#!/usr/bin/env python3
"""Channel tagging training v83 / v84: energy-decorrelated CT on burst-sample volumes.

This is a copy of `train_ct_volume_v80.py` with three new, config-gated options.
Everything else - architecture, data caps, cat splits, seed, augmentation,
optimizer, callbacks - is byte-for-byte the v80 behaviour, so v80 / v83 / v84 are
directly comparable.  In particular the `np.random.RandomState(seed)` stream is
consumed in exactly the same order and amount as in v80, so the train/val/test
samples are the *same volumes in the same order* as the ones v80 saw.

New options (json `training` / `data` blocks):

  training.energy_reweight : bool  (v83, v84)
      Per-sample weights that reweight the two classes to a COMMON spectrum of
      the reconstructed main-cluster energy `particle_energy`.  For each class c
      and energy bin b:  w_c(b) = target(b) / p_c(b),  target = (p_ES + p_CC)/2,
      with p_c the class-normalized histogram.  Weights are then renormalized so
      each class has mean weight 1 (hence equal total weight, the sample being
      balanced).  The weights are passed to `model.fit` through tf.data as
      sample weights, for training AND validation.

      Why energy and not total ADC: `particle_energy` is the observable the
      pointing selection would be binned in (it is the reconstructed main-cluster
      energy, already in the volume metadata and available online), it is the
      variable Table 8 of the v80 study is written in, and it is the variable the
      ES kinematic pointing weight 1/theta68(E)^2 depends on.  Flattening in
      total ADC would decorrelate the score from the CNN's dominant input but
      would leave a residual slide versus the quantity we actually select on.

  data.per_image_normalization : null | "max" | "sum"   (v84)
      Divide each log1p image by its own maximum ("max", used by v84) or sum.
      Removes the absolute calorimetric scale from the pixels.

  model.use_aux_features + model.aux_include_energy : bool  (v84)
      Auxiliary scalar branch (volume_v80_aux.json branch) =
      [n_clusters_in_volume, log1p(total ADC), log1p(n nonzero pixels)]
      plus log1p(particle_energy) when aux_include_energy.
      Aux features are always computed from the RAW image, so they are unaffected
      by per-image normalization - that is the point: the scale is taken out of
      the pixels and handed to the network explicitly.

The aux features and the image preprocessing come from
`channel_tagging/lib/ct_volume_features.py`, which is importable standalone so
the pipeline can reproduce them bit-for-bit.

Labels: ES=0, CC=1 (same convention as v52/v79/v80 and snop-pipeline channel_tagger).
"""

import sys
import os
import json
import glob
import argparse
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'python'))
sys.path.insert(0, os.path.join(_HERE, '..', 'lib'))

import numpy as np

from ct_volume_features import (IMAGE_SHAPE, aux_feature_names, compute_aux_features,
                                preprocess_image)

# energy-reweighting histogram: 1 MeV bins to 30 MeV, 5 MeV to 60, one overflow
DEFAULT_REWEIGHT_EDGES = (list(np.arange(0.0, 30.0, 1.0)) +
                          list(np.arange(30.0, 60.0, 5.0)) + [1.0e9])


def parse_args():
    parser = argparse.ArgumentParser(description='Train CT v83/v84 on burst-sample volumes')
    parser.add_argument('--json', '-j', type=str, required=True, help='JSON config file')
    parser.add_argument('--test-local', action='store_true',
                        help='Tiny local run: few cats, few samples, 2 epochs')
    return parser.parse_args()


def cat_dirs_for_range(base_dir, volume_subdir_fmt, cat_lo, cat_hi):
    """Return existing volume dirs for cats in [cat_lo, cat_hi]."""
    dirs = []
    for i in range(cat_lo, cat_hi + 1):
        cat = f'cat{i:06d}'
        d = os.path.join(base_dir, cat, volume_subdir_fmt.format(cat=cat))
        if os.path.isdir(d):
            dirs.append(d)
    return dirs


def _cat_of(path):
    """cat000597 from .../cat000597/cat000597_volume_images_.../X/es_000003_bg_planeX.npz"""
    try:
        return os.path.basename(path.split('/X/')[0]).split('_')[0]
    except Exception:
        return 'unknown'


def load_split(vol_dirs, max_per_class, rng, split_name='', per_image_norm=None,
               aux_include_energy=True):
    """Load ES (label 0) and CC (label 1) volumes from a list of cat volume dirs.

    IDENTICAL selection logic and RandomState consumption to v80: shuffle(es
    files), shuffle(cc files), permutation(n).  Only the returned quantities and
    the image preprocessing differ.
    """
    files_per_class = {0: [], 1: []}
    for d in vol_dirs:
        files_per_class[0].extend(sorted(glob.glob(os.path.join(d, 'es_*.npz'))))
        files_per_class[1].extend(sorted(glob.glob(os.path.join(d, 'cc_*.npz'))))

    images_list, labels_list, aux_list, energy_list = [], [], [], []
    adc_list, cat_list, src_file_list, src_idx_list = [], [], [], []
    counts = {0: 0, 1: 0}

    for label in (0, 1):
        files = files_per_class[label]
        rng.shuffle(files)
        last_milestone = 0
        for f in files:
            if counts[label] >= max_per_class:
                break
            try:
                data = np.load(f, allow_pickle=True)
                imgs = data['images']
                metadata = data['metadata'] if 'metadata' in data else [None] * len(imgs)
            except Exception as e:
                print(f'  Warning: failed to load {f}: {e}')
                continue
            cat = _cat_of(f)
            for idx in range(len(imgs)):
                if counts[label] >= max_per_class:
                    break
                img = np.asarray(imgs[idx], dtype=np.float32)
                if img.shape != IMAGE_SHAPE:
                    continue
                meta = metadata[idx] if idx < len(metadata) else None
                # aux features are computed from the RAW image, always
                aux = compute_aux_features(img, meta, include_energy=aux_include_energy)
                energy = float(meta.get('particle_energy', np.nan)) if isinstance(meta, dict) \
                    else np.nan
                images_list.append(preprocess_image(img, per_image_norm).astype(np.float16))
                labels_list.append(label)
                aux_list.append(aux)
                energy_list.append(energy)
                adc_list.append(float(img.sum()))
                cat_list.append(cat)
                src_file_list.append(f)
                src_idx_list.append(idx)
                counts[label] += 1
            if counts[label] // 5000 > last_milestone:
                last_milestone = counts[label] // 5000
                print(f'  [{split_name}] class {label}: {counts[label]} samples...', flush=True)

    images = np.stack(images_list).astype(np.float16)[..., np.newaxis]
    labels = np.array(labels_list, dtype=np.int32)
    aux = np.array(aux_list, dtype=np.float32)
    energies = np.array(energy_list, dtype=np.float32)
    total_adc = np.array(adc_list, dtype=np.float32)
    cats = np.array(cat_list)
    src_files = np.array(src_file_list)
    src_idx = np.array(src_idx_list, dtype=np.int32)

    perm = rng.permutation(len(images))
    out = dict(images=images[perm], labels=labels[perm], aux=aux[perm],
               energies=energies[perm], total_adc=total_adc[perm], cats=cats[perm],
               src_files=src_files[perm], src_idx=src_idx[perm])
    print(f'[{split_name}] loaded: ES={counts[0]}, CC={counts[1]}, shape={out["images"].shape}',
          flush=True)
    return out


# -----------------------------------------------------------------------------
# energy reweighting
# -----------------------------------------------------------------------------
def energy_reweight(energies, labels, edges, split_name=''):
    """Per-sample weights bringing both classes to a common energy spectrum.

    target(b) = (p_ES(b) + p_CC(b)) / 2 ;  w_c(b) = target(b) / p_c(b)
    then renormalized so each class has mean weight exactly 1.

    Returns (weights, info_dict).  Events with non-finite energy get weight 1 and
    are counted in info_dict['n_bad_energy'].
    """
    edges = np.asarray(edges, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    E = np.asarray(energies, dtype=np.float64)
    good = np.isfinite(E)

    idx = np.clip(np.digitize(np.where(good, E, 0.0), edges) - 1, 0, len(edges) - 2)
    nb = len(edges) - 1

    h = {}
    for c in (0, 1):
        m = good & (labels == c)
        h[c] = np.bincount(idx[m], minlength=nb).astype(np.float64)
    p = {c: (h[c] / h[c].sum() if h[c].sum() > 0 else h[c]) for c in (0, 1)}
    target = 0.5 * (p[0] + p[1])

    wbin = {}
    for c in (0, 1):
        wbin[c] = np.divide(target, p[c], out=np.zeros(nb), where=p[c] > 0)

    w = np.ones(len(E), dtype=np.float64)
    for c in (0, 1):
        m = good & (labels == c)
        w[m] = wbin[c][idx[m]]
    # renormalize each class to mean weight 1 -> equal total weight per class
    for c in (0, 1):
        m = labels == c
        mean = w[m].mean()
        if mean > 0:
            w[m] = w[m] / mean

    info = {
        'bin_edges': edges.tolist(),
        'hist_es': h[0].tolist(),
        'hist_cc': h[1].tolist(),
        'weight_bin_es': wbin[0].tolist(),
        'weight_bin_cc': wbin[1].tolist(),
        'n_bad_energy': int((~good).sum()),
        'weight_min': float(w.min()),
        'weight_max': float(w.max()),
        'weight_mean_es': float(w[labels == 0].mean()),
        'weight_mean_cc': float(w[labels == 1].mean()),
        'weight_sum_es': float(w[labels == 0].sum()),
        'weight_sum_cc': float(w[labels == 1].sum()),
    }
    print(f'[{split_name}] energy reweighting: w in [{w.min():.3f}, {w.max():.3f}], '
          f'sum(ES)={info["weight_sum_es"]:.1f}, sum(CC)={info["weight_sum_cc"]:.1f}, '
          f'{info["n_bad_energy"]} events with bad energy', flush=True)
    return w.astype(np.float32), info


def build_model(use_aux, filter_list, dense_units, dropout_rate, n_aux):
    from tensorflow import keras

    img_in = keras.layers.Input(shape=IMAGE_SHAPE + (1,), name='image')
    x = img_in
    for i, filters in enumerate(filter_list):
        x = keras.layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        if i < len(filter_list) - 1:
            x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    inputs = [img_in]
    if use_aux:
        aux_in = keras.layers.Input(shape=(n_aux,), name='aux')
        a = keras.layers.BatchNormalization()(aux_in)
        a = keras.layers.Dense(16, activation='relu')(a)
        x = keras.layers.Concatenate()([x, a])
        inputs.append(aux_in)

    for units in dense_units:
        x = keras.layers.Dense(units, activation='relu')(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    out = keras.layers.Dense(2, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=out)


def make_dataset(images, labels, aux, use_aux, batch_size, augment, shuffle, weights=None):
    import tensorflow as tf

    x = (images, aux) if use_aux else images
    if weights is None:
        ds = tf.data.Dataset.from_tensor_slices((x, labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices((x, labels, weights))
    if shuffle:
        ds = ds.shuffle(min(len(labels), 20000), reshuffle_each_iteration=True)

    def _prep_x(x):
        if use_aux:
            img, a = x
        else:
            img = x
        img = tf.cast(img, tf.float32)
        if augment:
            # flip along the wire-channel axis (detector left-right symmetry)
            img = tf.image.random_flip_up_down(img)
        return (img, a) if use_aux else img

    if weights is None:
        ds = ds.batch(batch_size).map(lambda x, y: (_prep_x(x), y),
                                      num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch_size).map(lambda x, y, w: (_prep_x(x), y, w),
                                      num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def main():
    args = parse_args()
    with open(args.json) as f:
        config = json.load(f)

    dcfg = config['data']
    mcfg = config['model']
    tcfg = config['training']
    ocfg = config['output']

    use_aux = bool(mcfg.get('use_aux_features', False))
    aux_include_energy = bool(mcfg.get('aux_include_energy', False))
    per_image_norm = dcfg.get('per_image_normalization', None)
    do_reweight = bool(tcfg.get('energy_reweight', False))
    reweight_edges = tcfg.get('reweight_bin_edges', DEFAULT_REWEIGHT_EDGES)
    model_name = config.get('model_name', 'ct_volume_v83')
    aux_names = aux_feature_names(include_energy=aux_include_energy)

    max_dirs_per_split = None
    if args.test_local:
        print('*** TEST-LOCAL MODE: tiny caps, 2 epochs ***')
        dcfg['max_train_per_class'] = 40
        dcfg['max_val_per_class'] = 10
        dcfg['max_test_per_class'] = 10
        max_dirs_per_split = 2
        tcfg['epochs'] = 2

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(ocfg['base_dir'], f'{model_name}_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output directory: {out_dir}')
    print(f'Options: use_aux={use_aux} aux_energy={aux_include_energy} '
          f'per_image_norm={per_image_norm!r} energy_reweight={do_reweight}')

    rng = np.random.RandomState(tcfg.get('seed', 42))
    base = dcfg['burst_samples_dir']
    subdir_fmt = dcfg['volume_subdir_fmt']

    splits = {}
    for split, cats_key, cap_key in (
            ('train', 'train_cats', 'max_train_per_class'),
            ('val', 'val_cats', 'max_val_per_class'),
            ('test', 'test_cats', 'max_test_per_class')):
        lo, hi = dcfg[cats_key]
        dirs = cat_dirs_for_range(base, subdir_fmt, lo, hi)
        if max_dirs_per_split:
            dirs = dirs[:max_dirs_per_split]
        print(f'\n[{split}] cats {lo}-{hi}: {len(dirs)} cat dirs found')
        if not dirs:
            raise RuntimeError(f'No cat dirs for split {split} in range {lo}-{hi}')
        splits[split] = load_split(dirs, dcfg[cap_key], rng, split_name=split,
                                   per_image_norm=per_image_norm,
                                   aux_include_energy=aux_include_energy)

    import tensorflow as tf
    from tensorflow import keras

    tr, va, te = splits['train'], splits['val'], splits['test']
    X_tr, y_tr, aux_tr = tr['images'], tr['labels'], tr['aux']
    X_va, y_va, aux_va = va['images'], va['labels'], va['aux']
    X_te, y_te, aux_te = te['images'], te['labels'], te['aux']

    # standardize aux features with train stats (recorded for deployment)
    aux_mean = aux_tr.mean(axis=0)
    aux_std = aux_tr.std(axis=0) + 1e-6
    aux_tr = (aux_tr - aux_mean) / aux_std
    aux_va = (aux_va - aux_mean) / aux_std
    aux_te = (aux_te - aux_mean) / aux_std

    # ---- sample weights ------------------------------------------------------
    reweight_info = {}
    w_tr = w_va = None
    if do_reweight:
        w_tr, reweight_info['train'] = energy_reweight(tr['energies'], y_tr, reweight_edges,
                                                       'train')
        w_va, reweight_info['val'] = energy_reweight(va['energies'], y_va, reweight_edges, 'val')

    batch_size = tcfg['batch_size']
    ds_tr = make_dataset(X_tr, y_tr, aux_tr, use_aux, batch_size,
                         augment=tcfg.get('augment', True), shuffle=True, weights=w_tr)
    ds_va = make_dataset(X_va, y_va, aux_va, use_aux, batch_size, augment=False, shuffle=False,
                         weights=w_va)
    ds_te = make_dataset(X_te, y_te, aux_te, use_aux, batch_size, augment=False, shuffle=False)

    model = build_model(use_aux, mcfg['filter_list'], mcfg['dense_units'], mcfg['dropout_rate'],
                        n_aux=len(aux_names))
    # With sample weights the metric that matters is the weighted one; Keras only
    # weights entries of `weighted_metrics`, so use that list when reweighting.
    # Either way the metric object is named 'accuracy' -> 'val_accuracy' in the logs.
    compile_kwargs = dict(
        optimizer=keras.optimizers.Adam(learning_rate=tcfg['learning_rate']),
        loss=keras.losses.SparseCategoricalCrossentropy())
    if do_reweight:
        compile_kwargs['weighted_metrics'] = ['accuracy']
    else:
        compile_kwargs['metrics'] = ['accuracy']
    model.compile(**compile_kwargs)
    model.summary()

    monitor = 'val_accuracy'
    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(out_dir, 'best_model.keras'),
                                        monitor=monitor, save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor=monitor, patience=tcfg.get('patience', 10),
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,
                                          min_lr=1e-5, verbose=1),
        keras.callbacks.CSVLogger(os.path.join(out_dir, 'training_history.csv')),
    ]

    history = model.fit(ds_tr, validation_data=ds_va, epochs=tcfg['epochs'],
                        callbacks=callbacks, verbose=2)

    # ---- evaluation (UNWEIGHTED, balanced test set, as v80) ------------------
    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

    y_prob = model.predict(ds_te, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    acc = float((y_pred == y_te).mean())
    auc = float(roc_auc_score(y_te, y_prob[:, 1]))
    cm = confusion_matrix(y_te, y_pred, normalize='true')
    report = classification_report(y_te, y_pred, target_names=['ES', 'CC'])

    print(f'\nTest accuracy: {acc:.4f}   AUC: {auc:.4f}')
    print('Confusion matrix (normalized):')
    print(cm)
    print(report)

    np.savez_compressed(
        os.path.join(out_dir, 'test_predictions.npz'),
        predictions=y_prob, true_labels=y_te, energies=te['energies'],
        total_adc=te['total_adc'], cat=te['cats'],
        src_file=te['src_files'], src_index=te['src_idx'],
        aux_features=aux_te, aux_feature_names=np.array(aux_names))

    results = {
        'model_name': model_name,
        'timestamp': timestamp,
        'config': config,
        'label_convention': {'ES': 0, 'CC': 1},
        'preprocessing': {
            # self-documenting string; the snop-pipeline's _resolve_preprocess_mode
            # only checks that it startswith 'log1p'
            'image': 'log1p' if not per_image_norm else f'log1p_div_per_image_{per_image_norm}',
            'per_image_normalization': per_image_norm if per_image_norm else False,
            'aux_features': aux_names if use_aux else [],
            'aux_mean': aux_mean.tolist(),
            'aux_std': aux_std.tolist(),
            'aux_computed_from': 'raw image (before log1p / per-image normalization)',
            'feature_module': 'channel_tagging/lib/ct_volume_features.py',
        },
        'energy_reweighting': {
            'enabled': do_reweight,
            'variable': 'particle_energy (reconstructed main-cluster energy, MeV)',
            'scheme': 'w_c(b) = target(b)/p_c(b), target = (p_ES + p_CC)/2, '
                      'renormalized to mean weight 1 per class; applied to train and val',
            'per_split': reweight_info,
        },
        'data_summary': {
            'n_train': int(len(y_tr)), 'n_val': int(len(y_va)), 'n_test': int(len(y_te)),
            'train_cats': dcfg['train_cats'], 'val_cats': dcfg['val_cats'],
            'test_cats': dcfg['test_cats'],
        },
        'test_metrics': {
            'accuracy': acc,
            'auc': auc,
            'confusion_matrix_normalized': cm.tolist(),
            'classification_report': report,
        },
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
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['ES', 'CC'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['ES', 'CC'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'{model_name} acc={acc:.3f} auc={auc:.3f}')
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150)
    except Exception as e:
        print(f'Warning: could not save confusion matrix plot: {e}')

    print(f'\nDone. Results in {out_dir}')


if __name__ == '__main__':
    main()
