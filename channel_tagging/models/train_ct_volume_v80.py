#!/usr/bin/env python3
"""
Channel tagging training v80: burst-sample volume images at production conditions.

Key differences w.r.t. the v78/v79 volume trainers:
- Trains on the sn-burst-samples volume images (tick3_ch2_min2_tot3_e3p0),
  i.e. the SAME data product and clustering conditions the snop-pipeline
  runs CT inference on (the old prod_es/prod_cc e2p0 volumes are gone).
- Splits train/val/test at the CAT (burst) level, never mixing volumes from
  the same burst across splits.
- Deterministic preprocessing recorded in results.json (log1p), so inference
  code can reproduce it exactly. No per-image max normalization: the absolute
  ADC scale carries energy information.
- Optional auxiliary-feature branch (truth-free, available online):
  n_clusters_in_volume, log1p(total ADC), log1p(n nonzero pixels).
- BatchNorm conv blocks, channel-axis flip augmentation, early stopping.

Labels: ES=0, CC=1 (same convention as v52/v79 and snop-pipeline channel_tagger).
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


def parse_args():
    parser = argparse.ArgumentParser(description='Train CT v80 on burst-sample volumes')
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


def extract_aux_and_energy(img, meta):
    """Truth-free aux features + truth energy (for analysis only, never a model input)."""
    n_clusters = float(meta.get('n_clusters_in_volume', 1)) if isinstance(meta, dict) else 1.0
    total_adc = float(img.sum())
    n_nonzero = float(np.count_nonzero(img))
    aux = [n_clusters, np.log1p(total_adc), np.log1p(n_nonzero)]
    energy = float(meta.get('particle_energy', np.nan)) if isinstance(meta, dict) else np.nan
    return aux, energy


def load_split(vol_dirs, max_per_class, rng, split_name=''):
    """Load ES (label 0) and CC (label 1) volumes from a list of cat volume dirs.

    Files are interleaved across cats so the per-class cap draws from many bursts.
    Images stored as float16 after log1p to keep memory manageable.
    """
    files_per_class = {0: [], 1: []}
    for d in vol_dirs:
        files_per_class[0].extend(sorted(glob.glob(os.path.join(d, 'es_*.npz'))))
        files_per_class[1].extend(sorted(glob.glob(os.path.join(d, 'cc_*.npz'))))

    images_list, labels_list, aux_list, energy_list = [], [], [], []
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
                aux, energy = extract_aux_and_energy(img, meta)
                images_list.append(np.log1p(img).astype(np.float16))
                labels_list.append(label)
                aux_list.append(aux)
                energy_list.append(energy)
                counts[label] += 1
            if counts[label] // 5000 > last_milestone:
                last_milestone = counts[label] // 5000
                print(f'  [{split_name}] class {label}: {counts[label]} samples...', flush=True)

    images = np.stack(images_list).astype(np.float16)[..., np.newaxis]
    labels = np.array(labels_list, dtype=np.int32)
    aux = np.array(aux_list, dtype=np.float32)
    energies = np.array(energy_list, dtype=np.float32)

    perm = rng.permutation(len(images))
    images, labels, aux, energies = images[perm], labels[perm], aux[perm], energies[perm]
    print(f'[{split_name}] loaded: ES={counts[0]}, CC={counts[1]}, shape={images.shape}')
    return images, labels, aux, energies


def build_model(use_aux, filter_list, dense_units, dropout_rate, n_aux=len(AUX_FEATURE_NAMES)):
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


def make_dataset(images, labels, aux, use_aux, batch_size, augment, shuffle):
    import tensorflow as tf

    if use_aux:
        ds = tf.data.Dataset.from_tensor_slices(((images, aux), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(min(len(labels), 20000), reshuffle_each_iteration=True)

    def _prep(x, y):
        if use_aux:
            img, a = x
        else:
            img = x
        img = tf.cast(img, tf.float32)
        if augment:
            # flip along the wire-channel axis (detector left-right symmetry)
            img = tf.image.random_flip_up_down(img)
        return ((img, a) if use_aux else img), y

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

    use_aux = bool(mcfg.get('use_aux_features', False))
    model_name = config.get('model_name', 'ct_volume_v80')

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
        splits[split] = load_split(dirs, dcfg[cap_key], rng, split_name=split)

    import tensorflow as tf
    from tensorflow import keras

    X_tr, y_tr, aux_tr, _ = splits['train']
    X_va, y_va, aux_va, _ = splits['val']
    X_te, y_te, aux_te, E_te = splits['test']

    # standardize aux features with train stats (recorded for deployment)
    aux_mean = aux_tr.mean(axis=0)
    aux_std = aux_tr.std(axis=0) + 1e-6
    aux_tr = (aux_tr - aux_mean) / aux_std
    aux_va = (aux_va - aux_mean) / aux_std
    aux_te = (aux_te - aux_mean) / aux_std

    batch_size = tcfg['batch_size']
    ds_tr = make_dataset(X_tr, y_tr, aux_tr, use_aux, batch_size,
                         augment=tcfg.get('augment', True), shuffle=True)
    ds_va = make_dataset(X_va, y_va, aux_va, use_aux, batch_size, augment=False, shuffle=False)
    ds_te = make_dataset(X_te, y_te, aux_te, use_aux, batch_size, augment=False, shuffle=False)

    model = build_model(use_aux, mcfg['filter_list'], mcfg['dense_units'], mcfg['dropout_rate'])
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

    # ---- evaluation ----
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
            'image': 'log1p',
            'per_image_normalization': False,
            'aux_features': AUX_FEATURE_NAMES if use_aux else [],
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
