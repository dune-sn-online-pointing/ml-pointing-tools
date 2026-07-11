#!/usr/bin/env python3
"""
Channel tagging v81: sparse point-cloud model on burst-sample volumes.

Motivation: CT volume images are ~99.9% empty (typically ~250 hit pixels out
of 208x1242). Dense CNNs spend almost all compute on zeros, and global average
pooling dilutes small displaced gamma blips (the key CC signature). Instead of
true sparse convolutions (spconv/MinkowskiEngine, not available in LCG), this
trainer represents each volume as a point cloud of its nonzero pixels:

    point = (channel_rel, tick_rel, log1p(ADC))   relative to image center

and classifies with a DeepSets-style permutation-invariant encoder:
shared per-point MLP -> masked max+mean pooling -> dense head, concatenated
with the same truth-free aux features as v80_aux (n_clusters_in_volume,
log1p(total ADC), log1p(n nonzero pixels)).

Point clouds are ~1000x smaller than images, so this trainer can afford far
more samples in memory and trains in minutes.

Data/splits/labels identical to v80 (cat-level splits, ES=0, CC=1).
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
AUX_FEATURE_NAMES = ['n_clusters_in_volume', 'log1p_total_adc', 'log1p_n_nonzero']
N_POINT_FEATURES = 3


def parse_args():
    parser = argparse.ArgumentParser(description='Train CT v81 point-cloud model')
    parser.add_argument('--json', '-j', type=str, required=True, help='JSON config file')
    parser.add_argument('--test-local', action='store_true',
                        help='Tiny local run: few cats, few samples, 2 epochs')
    return parser.parse_args()


def cat_dirs_for_range(base_dir, volume_subdir_fmt, cat_lo, cat_hi):
    dirs = []
    for i in range(cat_lo, cat_hi + 1):
        cat = f'cat{i:06d}'
        d = os.path.join(base_dir, cat, volume_subdir_fmt.format(cat=cat))
        if os.path.isdir(d):
            dirs.append(d)
    return dirs


def image_to_points(img, max_points):
    """Convert a sparse image to (max_points, 3) float16 + n_real points.

    Points sorted by ADC descending, so truncation drops the faintest hits.
    Coordinates are relative to the image center, scaled to ~[-1, 1].
    """
    ch, tk = np.nonzero(img)
    adc = img[ch, tk]
    if len(ch) > max_points:
        keep = np.argsort(adc)[::-1][:max_points]
        ch, tk, adc = ch[keep], tk[keep], adc[keep]
    n = len(ch)
    pts = np.zeros((max_points, N_POINT_FEATURES), dtype=np.float16)
    pts[:n, 0] = (ch - IMAGE_SHAPE[0] / 2) / (IMAGE_SHAPE[0] / 2)
    pts[:n, 1] = (tk - IMAGE_SHAPE[1] / 2) / (IMAGE_SHAPE[1] / 2)
    pts[:n, 2] = np.log1p(adc)
    return pts, n


def load_split(vol_dirs, max_per_class, max_points, rng, split_name=''):
    files_per_class = {0: [], 1: []}
    for d in vol_dirs:
        files_per_class[0].extend(sorted(glob.glob(os.path.join(d, 'es_*.npz'))))
        files_per_class[1].extend(sorted(glob.glob(os.path.join(d, 'cc_*.npz'))))

    points_list, npts_list, labels_list, aux_list, energy_list = [], [], [], [], []
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
            for idx in range(len(imgs)):
                if counts[label] >= max_per_class:
                    break
                img = np.asarray(imgs[idx], dtype=np.float32)
                if img.shape != IMAGE_SHAPE:
                    continue
                meta = metadata[idx] if idx < len(metadata) else None
                pts, n = image_to_points(img, max_points)
                if n == 0:
                    continue
                n_clusters = float(meta.get('n_clusters_in_volume', 1)) if isinstance(meta, dict) else 1.0
                aux_list.append([n_clusters, np.log1p(img.sum()),
                                 np.log1p(np.count_nonzero(img))])
                energy_list.append(float(meta.get('particle_energy', np.nan))
                                   if isinstance(meta, dict) else np.nan)
                points_list.append(pts)
                npts_list.append(n)
                labels_list.append(label)
                counts[label] += 1
            if counts[label] // 10000 > last_milestone:
                last_milestone = counts[label] // 10000
                print(f'  [{split_name}] class {label}: {counts[label]} samples...', flush=True)

    points = np.stack(points_list).astype(np.float16)
    npts = np.array(npts_list, dtype=np.int32)
    labels = np.array(labels_list, dtype=np.int32)
    aux = np.array(aux_list, dtype=np.float32)
    energies = np.array(energy_list, dtype=np.float32)

    perm = rng.permutation(len(labels))
    points, npts, labels, aux, energies = (points[perm], npts[perm], labels[perm],
                                           aux[perm], energies[perm])
    print(f'[{split_name}] loaded: ES={counts[0]}, CC={counts[1]}, points={points.shape}, '
          f'median hits/img={int(np.median(npts))}, truncated={int((npts == points.shape[1]).sum())}')
    return points, labels, aux, energies


def build_model(max_points, point_mlp, head_units, dropout_rate,
                n_aux=len(AUX_FEATURE_NAMES)):
    import tensorflow as tf
    from tensorflow import keras

    pts_in = keras.layers.Input(shape=(max_points, N_POINT_FEATURES), name='points')
    aux_in = keras.layers.Input(shape=(n_aux,), name='aux')

    # mask: a real point has nonzero log1p(ADC)
    mask = keras.layers.Lambda(
        lambda p: tf.cast(tf.abs(p[..., 2:3]) > 0, tf.float32))(pts_in)

    x = pts_in
    for units in point_mlp:
        x = keras.layers.Dense(units, use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

    x_masked = keras.layers.Multiply()([x, mask])
    x_max = keras.layers.Lambda(lambda t: tf.reduce_max(t, axis=1))(x_masked)
    x_sum = keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x_masked)
    n_real = keras.layers.Lambda(
        lambda m: tf.reduce_sum(m, axis=1) + 1e-6)(mask)
    x_mean = keras.layers.Lambda(lambda t: t[0] / t[1])([x_sum, n_real])

    a = keras.layers.BatchNormalization()(aux_in)
    a = keras.layers.Dense(16, activation='relu')(a)

    h = keras.layers.Concatenate()([x_max, x_mean, a])
    for units in head_units:
        h = keras.layers.Dense(units, activation='relu')(h)
        h = keras.layers.Dropout(dropout_rate)(h)
    out = keras.layers.Dense(2, activation='softmax')(h)
    return keras.Model(inputs=[pts_in, aux_in], outputs=out)


def make_dataset(points, labels, aux, batch_size, augment, shuffle):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices(((points, aux), labels))
    if shuffle:
        ds = ds.shuffle(min(len(labels), 50000), reshuffle_each_iteration=True)

    def _prep(x, y):
        pts, a = x
        pts = tf.cast(pts, tf.float32)
        if augment:
            # channel-axis mirror symmetry (flip sign of relative channel coord)
            flip = tf.where(tf.random.uniform(()) > 0.5, -1.0, 1.0)
            pts = tf.concat([pts[..., 0:1] * flip, pts[..., 1:]], axis=-1)
        return (pts, a), y

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
    model_name = config.get('model_name', 'ct_points_v81')
    max_points = int(mcfg.get('max_points', 1024))

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
        print(f'\n[{split}] cats {lo}-{hi}: {len(dirs)} cat dirs found')
        if not dirs:
            raise RuntimeError(f'No cat dirs for split {split} in range {lo}-{hi}')
        splits[split] = load_split(dirs, dcfg[cap_key], max_points, rng, split_name=split)

    import tensorflow as tf
    from tensorflow import keras

    P_tr, y_tr, aux_tr, _ = splits['train']
    P_va, y_va, aux_va, _ = splits['val']
    P_te, y_te, aux_te, E_te = splits['test']

    aux_mean = aux_tr.mean(axis=0)
    aux_std = aux_tr.std(axis=0) + 1e-6
    aux_tr = (aux_tr - aux_mean) / aux_std
    aux_va = (aux_va - aux_mean) / aux_std
    aux_te = (aux_te - aux_mean) / aux_std

    batch_size = tcfg['batch_size']
    ds_tr = make_dataset(P_tr, y_tr, aux_tr, batch_size,
                         augment=tcfg.get('augment', True), shuffle=True)
    ds_va = make_dataset(P_va, y_va, aux_va, batch_size, augment=False, shuffle=False)
    ds_te = make_dataset(P_te, y_te, aux_te, batch_size, augment=False, shuffle=False)

    model = build_model(max_points, mcfg['point_mlp'], mcfg['head_units'],
                        mcfg['dropout_rate'])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=tcfg['learning_rate']),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(out_dir, 'best_model.keras'),
                                        monitor='val_accuracy', save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=tcfg.get('patience', 15),
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
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
        predictions=y_prob, true_labels=y_te, energies=E_te,
        aux_features=aux_te, aux_feature_names=np.array(AUX_FEATURE_NAMES))

    results = {
        'model_name': model_name,
        'timestamp': timestamp,
        'config': config,
        'label_convention': {'ES': 0, 'CC': 1},
        'preprocessing': {
            'image': f'point_cloud(max_points={max_points}, features=[ch_rel, tick_rel, log1p_adc])',
            'aux_features': AUX_FEATURE_NAMES,
            'aux_mean': aux_mean.tolist(),
            'aux_std': aux_std.tolist(),
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
