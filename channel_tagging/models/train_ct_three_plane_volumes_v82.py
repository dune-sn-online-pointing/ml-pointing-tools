#!/usr/bin/env python3
"""
Channel tagging v82: three-plane volume images (U+V+X) from burst samples.

Extends v80 (X-plane only) by feeding the U and V induction-plane volume
images of the same interaction alongside X. Volumes are matched across planes
by (file basename, event): the burst-sample volume production creates at most
one volume per event per plane, so this matching is unambiguous and does not
depend on cluster match_id.

U/V volumes are produced by online-pointing-utils
python/app/create_volumes_uv_for_cats.py (X existed already).

Architecture: one small BatchNorm conv tower per plane (planes have different
wire pitch/geometry, so towers are not weight-shared), GAP each, concatenate,
dense head. Preprocessing and split conventions identical to v80
(log1p, cat-level splits, ES=0 / CC=1).
"""

import sys
import os
import json
import glob
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import numpy as np

IMAGE_SHAPE = (208, 1242)
PLANES = ['U', 'V', 'X']


def parse_args():
    parser = argparse.ArgumentParser(description='Train CT v82 three-plane volume model')
    parser.add_argument('--json', '-j', type=str, required=True, help='JSON config file')
    parser.add_argument('--test-local', action='store_true',
                        help='Tiny local run: few cats, few samples, 2 epochs')
    return parser.parse_args()


def cat_dirs_for_range(base_dir, volume_subdir_fmt, cat_lo, cat_hi):
    """Return existing volume dirs (containing all three plane subdirs)."""
    dirs = []
    for i in range(cat_lo, cat_hi + 1):
        cat = f'cat{i:06d}'
        d = os.path.join(base_dir, cat, volume_subdir_fmt.format(cat=cat))
        if all(os.path.isdir(os.path.join(d, p)) for p in PLANES):
            dirs.append(d)
    return dirs


def load_split(vol_dirs, max_per_class, rng, split_name=''):
    """Load matched (U, V, X) volume triplets. Label from filename prefix."""
    basenames_per_class = {0: [], 1: []}
    for d in vol_dirs:
        for f in sorted(glob.glob(os.path.join(d, 'X', 'es_*_planeX.npz'))):
            basenames_per_class[0].append((d, os.path.basename(f)[:-len('_planeX.npz')]))
        for f in sorted(glob.glob(os.path.join(d, 'X', 'cc_*_planeX.npz'))):
            basenames_per_class[1].append((d, os.path.basename(f)[:-len('_planeX.npz')]))

    images = {p: [] for p in PLANES}
    labels_list, energy_list = [], []
    counts = {0: 0, 1: 0}
    n_unmatched = 0

    for label in (0, 1):
        entries = basenames_per_class[label]
        rng.shuffle(entries)
        last_milestone = 0
        for d, base in entries:
            if counts[label] >= max_per_class:
                break
            plane_data = {}
            try:
                for p in PLANES:
                    f = os.path.join(d, p, f'{base}_plane{p}.npz')
                    if not os.path.isfile(f):
                        raise FileNotFoundError(f)
                    plane_data[p] = np.load(f, allow_pickle=True)
            except Exception:
                continue

            # index volumes by event per plane (max one volume/event/plane)
            ev_idx = {}
            for p in PLANES:
                meta = plane_data[p]['metadata']
                ev_idx[p] = {m['event']: i for i, m in enumerate(meta)
                             if isinstance(m, dict)}
            common = set(ev_idx['U']) & set(ev_idx['V']) & set(ev_idx['X'])
            n_unmatched += len(ev_idx['X']) - len(common)

            for ev in sorted(common):
                if counts[label] >= max_per_class:
                    break
                imgs = {}
                ok = True
                for p in PLANES:
                    im = np.asarray(plane_data[p]['images'][ev_idx[p][ev]],
                                    dtype=np.float32)
                    if im.shape != IMAGE_SHAPE:
                        ok = False
                        break
                    imgs[p] = np.log1p(im).astype(np.float16)
                if not ok:
                    continue
                for p in PLANES:
                    images[p].append(imgs[p])
                meta_x = plane_data['X']['metadata'][ev_idx['X'][ev]]
                energy_list.append(float(meta_x.get('particle_energy', np.nan)))
                labels_list.append(label)
                counts[label] += 1

            if counts[label] // 2000 > last_milestone:
                last_milestone = counts[label] // 2000
                print(f'  [{split_name}] class {label}: {counts[label]} samples...', flush=True)

    X = {p: np.stack(images[p]).astype(np.float16)[..., np.newaxis] for p in PLANES}
    labels = np.array(labels_list, dtype=np.int32)
    energies = np.array(energy_list, dtype=np.float32)

    perm = rng.permutation(len(labels))
    X = {p: X[p][perm] for p in PLANES}
    labels, energies = labels[perm], energies[perm]
    print(f'[{split_name}] loaded: ES={counts[0]}, CC={counts[1]}, '
          f'X-plane volumes without 3-plane match skipped: {n_unmatched}')
    return X, labels, energies


def build_model(filter_list, dense_units, dropout_rate):
    from tensorflow import keras

    inputs, towers = [], []
    for p in PLANES:
        inp = keras.layers.Input(shape=IMAGE_SHAPE + (1,), name=f'image_{p}')
        x = inp
        for i, filters in enumerate(filter_list):
            x = keras.layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            if i < len(filter_list) - 1:
                x = keras.layers.MaxPooling2D((2, 2))(x)
        towers.append(keras.layers.GlobalAveragePooling2D()(x))
        inputs.append(inp)

    h = keras.layers.Concatenate()(towers)
    for units in dense_units:
        h = keras.layers.Dense(units, activation='relu')(h)
        h = keras.layers.Dropout(dropout_rate)(h)
    out = keras.layers.Dense(2, activation='softmax')(h)
    return keras.Model(inputs=inputs, outputs=out)


def make_dataset(X, labels, batch_size, augment, shuffle):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices(
        ((X['U'], X['V'], X['X']), labels))
    if shuffle:
        ds = ds.shuffle(min(len(labels), 20000), reshuffle_each_iteration=True)

    def _prep(x, y):
        u, v, xx = (tf.cast(t, tf.float32) for t in x)
        if augment:
            # consistent channel-axis flip across planes
            do_flip = tf.random.uniform(()) > 0.5
            u, v, xx = (tf.cond(do_flip, lambda t=t: tf.reverse(t, axis=[1]), lambda t=t: t)
                        for t in (u, v, xx))
        return (u, v, xx), y

    ds = ds.batch(batch_size).map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def main():
    args = parse_args()
    with open(args.json) as f:
        config = json.load(f)

    dcfg = config['data']
    mcfg = config['model']
    tcfg = config['training']
    ocfg = config['output']
    model_name = config.get('model_name', 'ct_three_plane_vol_v82')

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
        print(f'\n[{split}] cats {lo}-{hi}: {len(dirs)} cat dirs with U/V/X found')
        if not dirs:
            raise RuntimeError(f'No 3-plane cat dirs for split {split} in range {lo}-{hi}')
        splits[split] = load_split(dirs, dcfg[cap_key], rng, split_name=split)

    import tensorflow as tf
    from tensorflow import keras

    X_tr, y_tr, _ = splits['train']
    X_va, y_va, _ = splits['val']
    X_te, y_te, E_te = splits['test']

    batch_size = tcfg['batch_size']
    ds_tr = make_dataset(X_tr, y_tr, batch_size,
                         augment=tcfg.get('augment', True), shuffle=True)
    ds_va = make_dataset(X_va, y_va, batch_size, augment=False, shuffle=False)
    ds_te = make_dataset(X_te, y_te, batch_size, augment=False, shuffle=False)

    model = build_model(mcfg['filter_list'], mcfg['dense_units'], mcfg['dropout_rate'])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=tcfg['learning_rate']),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
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
    print('Confusion matrix (normalized):')
    print(cm)
    print(report)

    np.savez_compressed(
        os.path.join(out_dir, 'test_predictions.npz'),
        predictions=y_prob, true_labels=y_te, energies=E_te)

    results = {
        'model_name': model_name,
        'timestamp': timestamp,
        'config': config,
        'label_convention': {'ES': 0, 'CC': 1},
        'preprocessing': {
            'image': 'log1p',
            'per_image_normalization': False,
            'planes': PLANES,
            'plane_matching': 'by (file basename, event)',
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

    print(f'\nDone. Results in {out_dir}')


if __name__ == '__main__':
    main()
