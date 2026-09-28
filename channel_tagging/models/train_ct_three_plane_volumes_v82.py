#!/usr/bin/env python3
"""
Channel tagging v82: three-plane volume images (U+V+X) from burst samples,
with optional energy reweighting of the two classes to a common reconstructed
main-cluster energy spectrum.

Extends v80 (X-plane only) in two ways:

1. Three planes. The U and V induction-plane volume images of the same
   interaction are fed alongside X. Volumes are matched across planes by
   (file basename, event): the burst-sample volume production creates at most
   one volume per event per plane, so this matching is unambiguous and does not
   depend on cluster match_id (which is a per-plane cluster label and does NOT
   agree across planes for ~1/3 of the volumes).

2. Energy reweighting (config `training.energy_reweight`). The v80 study
   (docs/CT_v80_energy_topology_study.md) showed that ~75% of the v80 AUC gain
   comes from P(class | E) rather than from topology, which makes the score
   uncalibrated in energy and turns a global threshold into an energy cut.
   Per-sample weights

       w_i = target(E_i) / h_{class(i)}(E_i),
       target(E) = 0.5 * (h_ES(E) + h_CC(E)),

   with h_c the class-normalised histogram of the reconstructed main-cluster
   energy, remove that information from the loss. Weights are renormalised so
   the two classes carry equal total weight. They are passed as sample weights
   to model.fit and (via weighted_metrics) used in the validation metric that
   drives early stopping and checkpointing.

U/V volumes are produced by online-pointing-utils
python/app/create_volumes_uv_for_cats.py (X existed already).

Architecture: one small BatchNorm conv tower per plane (planes have different
wire pitch/geometry, so towers are not weight-shared), GAP each, concatenate,
dense head. Image preprocessing and split conventions identical to v80
(log1p, no per-image normalisation, cat-level splits, ES=0 / CC=1).

Memory: the volume images are extremely sparse (mean 201 nonzero pixels out of
208x1242 on X, 73 on U, 82 on V), so they are held in a ragged sparse form
(flat pixel index + log1p value + per-sample offsets) and densified one batch at
a time inside the tf.data generator. Three planes at the v80 caps are 0.09 GB
this way instead of 62 GB dense (and 124 GB with the from_tensor_slices copy the
v80 trainer uses), which is what makes the job schedulable at all. The test
split is still loaded only after training.
"""

import sys
import os
import gc
import json
import glob
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import numpy as np

IMAGE_SHAPE = (208, 1242)
ALL_PLANES = ['U', 'V', 'X']
# reweighting binning: fine where the two spectra differ most, coarse in the tails
DEFAULT_REWEIGHT_EDGES = [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18,
                          20, 25, 30, 40, 60, 1.0e9]

META_SCALARS = ['n_clusters_in_volume', 'n_marley_clusters', 'n_non_marley_clusters']


def parse_args():
    parser = argparse.ArgumentParser(description='Train CT v82 three-plane volume model')
    parser.add_argument('--json', '-j', type=str, required=True, help='JSON config file')
    parser.add_argument('--test-local', action='store_true',
                        help='Tiny local run: few cats, few samples, 2 epochs')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Override output directory (used for local smoke tests)')
    return parser.parse_args()


def cat_dirs_for_range(base_dir, volume_subdir_fmt, cat_lo, cat_hi, planes):
    """Return (cat_number, dir) for cats whose volume dir has all requested planes."""
    dirs = []
    for i in range(cat_lo, cat_hi + 1):
        cat = f'cat{i:06d}'
        d = os.path.join(base_dir, cat, volume_subdir_fmt.format(cat=cat))
        if all(os.path.isdir(os.path.join(d, p)) for p in planes):
            dirs.append((i, d))
    return dirs


def _opening_angle_deg(meta):
    """Truth angle between the main track and the parent neutrino (analysis only)."""
    try:
        e = np.array([meta['main_track_momentum_x'], meta['main_track_momentum_y'],
                      meta['main_track_momentum_z']], dtype=np.float64)
        nu = np.array([meta['main_track_neutrino_momentum_x'],
                       meta['main_track_neutrino_momentum_y'],
                       meta['main_track_neutrino_momentum_z']], dtype=np.float64)
    except (KeyError, TypeError):
        return np.nan
    ne, nn = np.linalg.norm(e), np.linalg.norm(nu)
    if ne <= 0 or nn <= 0:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(e, nu) / (ne * nn), -1.0, 1.0))))


def load_split(cat_dirs, max_per_class, rng, split_name, match_planes, input_planes):
    """Load matched volume tuples. Label from filename prefix (es_ -> 0, cc_ -> 1).

    match_planes: planes that must all contain the event for it to be used
                  (defines the event selection; keep it fixed across ablations).
    input_planes: planes whose images are actually loaded and fed to the model.
                  'X' must be present (used for energy / total ADC / metadata).
    """
    if 'X' not in input_planes:
        raise ValueError("input_planes must contain 'X'")

    entries = {0: [], 1: []}
    for cat, d in cat_dirs:
        for prefix, label in (('es_', 0), ('cc_', 1)):
            for f in sorted(glob.glob(os.path.join(d, 'X', prefix + '*_planeX.npz'))):
                entries[label].append((cat, d, os.path.basename(f)[:-len('_planeX.npz')]))

    cap_total = 2 * max_per_class
    # ragged sparse image store: one (indices, values) pair per sample per plane
    sp_idx = {p: [] for p in input_planes}
    sp_val = {p: [] for p in input_planes}
    labels = np.empty(cap_total, dtype=np.int32)
    energies = np.full(cap_total, np.nan, dtype=np.float32)
    total_adc = {p: np.zeros(cap_total, dtype=np.float32) for p in input_planes}
    n_nonzero = np.zeros(cap_total, dtype=np.float32)
    cat_ids = np.zeros(cap_total, dtype=np.int32)
    event_ids = np.full(cap_total, -1, dtype=np.int32)
    basenames = np.empty(cap_total, dtype=object)
    theta = np.full(cap_total, np.nan, dtype=np.float32)
    meta_scalars = {k: np.full(cap_total, np.nan, dtype=np.float32) for k in META_SCALARS}

    n = 0
    counts = {0: 0, 1: 0}
    n_unmatched = 0
    n_bad_shape = 0
    n_failed_files = 0
    n_dup_events = 0

    for label in (0, 1):
        ent = entries[label]
        rng.shuffle(ent)
        last_milestone = 0
        for cat, d, base in ent:
            if counts[label] >= max_per_class:
                break

            # one open per plane: metadata always, images only for input planes
            # (the npz members are read lazily, so U/V images cost nothing in
            # the X-only ablation, but re-opening the file would)
            meta, imgs_all = {}, {}
            try:
                for p in match_planes:
                    f = os.path.join(d, p, f'{base}_plane{p}.npz')
                    with np.load(f, allow_pickle=True) as z:
                        meta[p] = list(z['metadata'])
                        if p in input_planes:
                            imgs_all[p] = z['images']
            except Exception:
                n_failed_files += 1
                continue

            # one volume per (file, event, plane) is the assumption behind the
            # match; drop any event that violates it rather than guessing
            ev_idx, dup = {}, set()
            for p in match_planes:
                idx = {}
                for i, m in enumerate(meta[p]):
                    if not isinstance(m, dict):
                        continue
                    ev = m['event']
                    if ev in idx:
                        dup.add(ev)
                    idx[ev] = i
                ev_idx[p] = idx
            if dup:
                n_dup_events += len(dup)
            common = set.intersection(*[set(ev_idx[p]) for p in match_planes]) - dup
            n_unmatched += len(ev_idx['X']) - len(common)
            if not common:
                continue

            for ev in sorted(common):
                if counts[label] >= max_per_class:
                    break
                raw = {}
                ok = True
                for p in input_planes:
                    im = np.asarray(imgs_all[p][ev_idx[p][ev]], dtype=np.float32)
                    if im.shape != IMAGE_SHAPE:
                        ok = False
                        break
                    raw[p] = im
                if not ok:
                    n_bad_shape += 1
                    continue

                for p in input_planes:
                    flat = raw[p].reshape(-1)
                    nz = np.flatnonzero(flat)
                    sp_idx[p].append(nz.astype(np.int32))
                    sp_val[p].append(np.log1p(flat[nz]).astype(np.float16))
                    total_adc[p][n] = float(flat.sum())
                mx = meta['X'][ev_idx['X'][ev]]
                n_nonzero[n] = float(np.count_nonzero(raw['X']))
                energies[n] = float(mx.get('particle_energy', np.nan))
                for k in META_SCALARS:
                    meta_scalars[k][n] = float(mx.get(k, np.nan))
                theta[n] = _opening_angle_deg(mx)
                labels[n] = label
                cat_ids[n] = cat
                event_ids[n] = int(ev)
                basenames[n] = base
                n += 1
                counts[label] += 1

            del imgs_all
            if counts[label] // 2000 > last_milestone:
                last_milestone = counts[label] // 2000
                print(f'  [{split_name}] class {label}: {counts[label]} samples...', flush=True)

    perm = rng.permutation(n)
    sparse = {}
    for p in input_planes:
        lens = np.array([len(sp_idx[p][i]) for i in perm], dtype=np.int64)
        ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(lens, out=ptr[1:])
        sparse[p] = {
            'idx': (np.concatenate([sp_idx[p][i] for i in perm]) if n else
                    np.zeros(0, np.int32)),
            'val': (np.concatenate([sp_val[p][i] for i in perm]) if n else
                    np.zeros(0, np.float16)),
            'ptr': ptr,
        }
    sp_idx.clear()
    sp_val.clear()

    out = {
        'images': sparse,
        'labels': labels[:n],
        'energies': energies[:n],
        'total_adc': {p: total_adc[p][:n] for p in input_planes},
        'n_nonzero': n_nonzero[:n],
        'cat_ids': cat_ids[:n],
        'event_ids': event_ids[:n],
        'basenames': np.array(list(basenames[:n]), dtype='<U64'),
        'theta_true_deg': theta[:n],
        'meta_scalars': {k: v[:n] for k, v in meta_scalars.items()},
    }

    for p in input_planes:
        out['total_adc'][p] = out['total_adc'][p][perm]
    for k in ('labels', 'energies', 'n_nonzero', 'cat_ids', 'event_ids',
              'basenames', 'theta_true_deg'):
        out[k] = out[k][perm]
    out['meta_scalars'] = {k: v[perm] for k, v in out['meta_scalars'].items()}

    print(f'[{split_name}] loaded: ES={counts[0]}, CC={counts[1]}, total={n}; '
          f'X-plane volumes without a full plane match skipped: {n_unmatched}; '
          f'duplicate-event volumes dropped: {n_dup_events}; '
          f'bad shape: {n_bad_shape}; unreadable files: {n_failed_files}', flush=True)
    return out


# --------------------------------------------------------------------------
# energy reweighting
# --------------------------------------------------------------------------
def compute_energy_weights(energies, labels, edges, clip=None):
    """w = target(E) / h_class(E), target = mean of the two class histograms.

    h_c is normalised to unit sum, so target is a proper density and, within a
    class, the mean weight is 1 by construction. Weights are then renormalised
    so that each class carries the same total weight. Events with a
    non-finite energy keep weight 1 and are excluded from the histograms.
    """
    edges = np.asarray(edges, dtype=np.float64)
    nb = len(edges) - 1
    good = np.isfinite(energies)
    b = np.clip(np.digitize(np.where(good, energies, 0.0), edges) - 1, 0, nb - 1)

    h = np.zeros((2, nb))
    for c in (0, 1):
        m = good & (labels == c)
        if m.sum() == 0:
            continue
        h[c] = np.bincount(b[m], minlength=nb).astype(np.float64)
        h[c] /= h[c].sum()
    target = 0.5 * (h[0] + h[1])

    w = np.ones(len(labels), dtype=np.float64)
    for c in (0, 1):
        m = good & (labels == c)
        den = h[c][b[m]]
        w[m] = np.where(den > 0, target[b[m]] / np.maximum(den, 1e-12), 0.0)
    if clip:
        w = np.minimum(w, float(clip))

    # equal total weight per class
    n_tot = len(labels)
    for c in (0, 1):
        m = labels == c
        s = w[m].sum()
        if s > 0:
            w[m] *= (0.5 * n_tot) / s

    info = {
        'edges': edges.tolist(),
        'h_es': h[0].tolist(),
        'h_cc': h[1].tolist(),
        'target': target.tolist(),
        'clip': clip,
        'w_min': float(w.min()), 'w_max': float(w.max()),
        'w_mean_es': float(w[labels == 0].mean()) if (labels == 0).any() else None,
        'w_mean_cc': float(w[labels == 1].mean()) if (labels == 1).any() else None,
        'n_nonfinite_energy': int((~good).sum()),
    }
    return w.astype(np.float32), info


# --------------------------------------------------------------------------
# model / data pipeline
# --------------------------------------------------------------------------
def build_model(planes, filter_list, dense_units, dropout_rate):
    from tensorflow import keras

    inputs, towers = {}, []
    for p in planes:
        inp = keras.layers.Input(shape=IMAGE_SHAPE + (1,), name=f'image_{p}')
        x = inp
        for i, filters in enumerate(filter_list):
            x = keras.layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            if i < len(filter_list) - 1:
                x = keras.layers.MaxPooling2D((2, 2))(x)
        towers.append(keras.layers.GlobalAveragePooling2D()(x))
        inputs[f'image_{p}'] = inp

    h = towers[0] if len(towers) == 1 else keras.layers.Concatenate()(towers)
    for units in dense_units:
        h = keras.layers.Dense(units, activation='relu')(h)
        h = keras.layers.Dropout(dropout_rate)(h)
    out = keras.layers.Dense(2, activation='softmax')(h)
    return keras.Model(inputs=inputs, outputs=out)


def n_steps(n, batch_size):
    return int(np.ceil(n / batch_size))


def densify(sp, sel):
    """Rebuild a dense (len(sel), H, W, 1) float16 batch from the sparse store."""
    out = np.zeros((len(sel), IMAGE_SHAPE[0] * IMAGE_SHAPE[1]), dtype=np.float16)
    idx, val, ptr = sp['idx'], sp['val'], sp['ptr']
    for j, i in enumerate(sel):
        a, b = ptr[i], ptr[i + 1]
        if b > a:
            out[j, idx[a:b]] = val[a:b]
    return out.reshape((len(sel),) + IMAGE_SHAPE + (1,))


def make_dataset(arrays, labels, weights, planes, batch_size, augment, shuffle, seed,
                 repeat=False):
    """tf.data pipeline over the pre-loaded sparse image store.

    The generator densifies one batch at a time from `arrays` (a per-plane
    {idx, val, ptr} sparse store) and never materialises the full dense set, nor
    copies it into a TF constant the way v80's from_tensor_slices does.

    repeat=True makes the generator loop forever, re-shuffling at every pass;
    combined with steps_per_epoch = ceil(n / batch_size) an epoch is exactly one
    pass over the data, and Keras never hits an end-of-sequence.
    """
    import tensorflow as tf

    n = len(labels)
    rs = np.random.RandomState(seed)

    def one_pass():
        order = np.arange(n)
        if shuffle:
            rs.shuffle(order)
        for s in range(0, n, batch_size):
            sel = order[s:s + batch_size]
            imgs = {f'image_{p}': densify(arrays[p], sel) for p in planes}
            if augment:
                # channel-axis flip, same decision for all planes of an event
                flip = rs.rand(len(sel)) < 0.5
                if flip.any():
                    for k in imgs:
                        imgs[k][flip] = imgs[k][flip][:, ::-1]
            yield imgs, labels[sel], weights[sel]

    def gen():
        if not repeat:
            yield from one_pass()
            return
        while True:
            yield from one_pass()

    sig = ({f'image_{p}': tf.TensorSpec(shape=(None,) + IMAGE_SHAPE + (1,), dtype=tf.float16)
            for p in planes},
           tf.TensorSpec(shape=(None,), dtype=tf.int32),
           tf.TensorSpec(shape=(None,), dtype=tf.float32))

    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    # bounded parallelism/prefetch: a three-plane float32 batch is ~100 MB
    ds = ds.map(lambda x, y, w: ({k: tf.cast(v, tf.float32) for k, v in x.items()}, y, w),
                num_parallel_calls=2)
    return ds.prefetch(4)


def main():
    args = parse_args()
    with open(args.json) as f:
        config = json.load(f)

    dcfg = config['data']
    mcfg = config['model']
    tcfg = config['training']
    ocfg = config['output']
    model_name = config.get('model_name', 'ct_three_plane_volumes_v82')

    match_planes = list(dcfg.get('match_planes', ALL_PLANES))
    input_planes = list(dcfg.get('input_planes', match_planes))
    for p in input_planes:
        if p not in match_planes:
            raise ValueError(f'input plane {p} not in match_planes {match_planes}')

    rw_cfg = tcfg.get('energy_reweight', {})
    rw_enabled = bool(rw_cfg.get('enabled', False))
    rw_edges = rw_cfg.get('bin_edges', DEFAULT_REWEIGHT_EDGES)
    rw_clip = rw_cfg.get('clip', None)

    max_dirs_per_split = None
    if args.test_local:
        print('*** TEST-LOCAL MODE: tiny caps, 2 epochs ***')
        dcfg['max_train_per_class'] = 30
        dcfg['max_val_per_class'] = 10
        dcfg['max_test_per_class'] = 10
        max_dirs_per_split = 2
        tcfg['epochs'] = 2

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(ocfg['base_dir'], f'{model_name}_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output directory: {out_dir}')
    print(f'match_planes={match_planes}  input_planes={input_planes}  '
          f'energy_reweight={rw_enabled}')

    rng = np.random.RandomState(tcfg.get('seed', 42))
    base = dcfg['burst_samples_dir']
    subdir_fmt = dcfg['volume_subdir_fmt']

    def get_dirs(cats_key):
        lo, hi = dcfg[cats_key]
        dirs = cat_dirs_for_range(base, subdir_fmt, lo, hi, match_planes)
        if max_dirs_per_split:
            dirs = dirs[:max_dirs_per_split]
        return (lo, hi), dirs

    # ---- train + val (test is loaded after training to keep the peak RSS down)
    splits = {}
    for split, cats_key, cap_key in (('train', 'train_cats', 'max_train_per_class'),
                                     ('val', 'val_cats', 'max_val_per_class')):
        (lo, hi), dirs = get_dirs(cats_key)
        print(f'\n[{split}] cats {lo}-{hi}: {len(dirs)} cat dirs with {match_planes}')
        if not dirs:
            raise RuntimeError(f'No cat dirs for split {split} in range {lo}-{hi}')
        splits[split] = load_split(dirs, dcfg[cap_key], rng, split, match_planes, input_planes)

    tr, va = splits['train'], splits['val']

    rw_info = {}
    if rw_enabled:
        w_tr, rw_info['train'] = compute_energy_weights(tr['energies'], tr['labels'],
                                                        rw_edges, rw_clip)
        w_va, rw_info['val'] = compute_energy_weights(va['energies'], va['labels'],
                                                      rw_edges, rw_clip)
        print(f"\nEnergy reweighting on. train w in [{rw_info['train']['w_min']:.3f}, "
              f"{rw_info['train']['w_max']:.3f}], mean ES {rw_info['train']['w_mean_es']:.3f}, "
              f"mean CC {rw_info['train']['w_mean_cc']:.3f}")
    else:
        w_tr = np.ones(len(tr['labels']), dtype=np.float32)
        w_va = np.ones(len(va['labels']), dtype=np.float32)

    import tensorflow as tf
    from tensorflow import keras

    batch_size = tcfg['batch_size']
    seed = int(tcfg.get('seed', 42))
    ds_tr = make_dataset(tr['images'], tr['labels'], w_tr, input_planes, batch_size,
                         augment=tcfg.get('augment', True), shuffle=True, seed=seed,
                         repeat=True)
    ds_va = make_dataset(va['images'], va['labels'], w_va, input_planes, batch_size,
                         augment=False, shuffle=False, seed=seed + 1, repeat=True)
    steps_tr = n_steps(len(tr['labels']), batch_size)
    steps_va = n_steps(len(va['labels']), batch_size)
    print(f'steps_per_epoch={steps_tr}  validation_steps={steps_va}')

    model = build_model(input_planes, mcfg['filter_list'], mcfg['dense_units'],
                        mcfg['dropout_rate'])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=tcfg['learning_rate']),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # weighted_metrics (not metrics): in Keras 3 only these receive the
        # sample weights, so val_accuracy is the reweighted accuracy.
        weighted_metrics=['accuracy'])
    model.summary()

    monitor = tcfg.get('monitor', 'val_accuracy')
    mode = tcfg.get('monitor_mode', 'max')
    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(out_dir, 'best_model.keras'),
                                        monitor=monitor, mode=mode,
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor=monitor, mode=mode,
                                      patience=tcfg.get('patience', 10),
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,
                                          min_lr=1e-5, verbose=1),
        keras.callbacks.CSVLogger(os.path.join(out_dir, 'training_history.csv')),
    ]

    history = model.fit(ds_tr, validation_data=ds_va, epochs=tcfg['epochs'],
                        steps_per_epoch=steps_tr, validation_steps=steps_va,
                        callbacks=callbacks, verbose=2)
    print('history keys:', list(history.history.keys()))

    model.save(os.path.join(out_dir, 'final_model.keras'))

    # ---- release the training images before loading the test split ----------
    n_train, n_val = len(tr['labels']), len(va['labels'])
    train_cats_used = sorted(set(int(c) for c in tr['cat_ids']))
    val_cats_used = sorted(set(int(c) for c in va['cat_ids']))
    del ds_tr, ds_va, tr, va, splits
    gc.collect()

    (lo, hi), dirs = get_dirs('test_cats')
    print(f'\n[test] cats {lo}-{hi}: {len(dirs)} cat dirs with {match_planes}')
    if not dirs:
        raise RuntimeError(f'No cat dirs for split test in range {lo}-{hi}')
    te = load_split(dirs, dcfg['max_test_per_class'], rng, 'test', match_planes, input_planes)

    if rw_enabled:
        w_te, rw_info['test'] = compute_energy_weights(te['energies'], te['labels'],
                                                       rw_edges, rw_clip)
    else:
        w_te = np.ones(len(te['labels']), dtype=np.float32)

    ds_te = make_dataset(te['images'], te['labels'], w_te, input_planes, batch_size,
                         augment=False, shuffle=False, seed=seed + 2, repeat=True)
    steps_te = n_steps(len(te['labels']), batch_size)

    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

    y_prob = model.predict(ds_te, steps=steps_te, verbose=0)
    y_te = te['labels']
    if len(y_prob) != len(y_te):
        raise RuntimeError(f'prediction length {len(y_prob)} != test size {len(y_te)}')
    y_pred = y_prob.argmax(axis=1)
    acc = float((y_pred == y_te).mean())
    auc = float(roc_auc_score(y_te, y_prob[:, 1]))
    acc_w = float(np.average((y_pred == y_te), weights=w_te))
    auc_w = float(roc_auc_score(y_te, y_prob[:, 1], sample_weight=w_te))
    cm = confusion_matrix(y_te, y_pred, normalize='true')
    report = classification_report(y_te, y_pred, target_names=['ES', 'CC'])

    print(f'\nTest accuracy: {acc:.4f}   AUC: {auc:.4f}')
    print(f'Test accuracy (energy-reweighted): {acc_w:.4f}   AUC (reweighted): {auc_w:.4f}')
    print('Confusion matrix (normalized):')
    print(cm)
    print(report)

    save_kw = dict(
        predictions=y_prob,
        p_es=y_prob[:, 0].astype(np.float32),
        true_labels=y_te,
        energies=te['energies'],
        total_adc=te['total_adc']['X'],
        n_nonzero=te['n_nonzero'],
        cat_ids=te['cat_ids'],
        event_ids=te['event_ids'],
        basenames=te['basenames'],
        theta_true_deg=te['theta_true_deg'],
        sample_weights=w_te,
        planes=np.array(input_planes),
    )
    for p in input_planes:
        save_kw[f'total_adc_{p}'] = te['total_adc'][p]
    for k, v in te['meta_scalars'].items():
        save_kw[k] = v
    np.savez_compressed(os.path.join(out_dir, 'test_predictions.npz'), **save_kw)

    results = {
        'model_name': model_name,
        'timestamp': timestamp,
        'config': config,
        'label_convention': {'ES': 0, 'CC': 1},
        'preprocessing': {
            'image': 'log1p',
            'per_image_normalization': False,
            'match_planes': match_planes,
            'input_planes': input_planes,
            'plane_matching': 'by (file basename, event number)',
        },
        'energy_reweighting': {
            'enabled': rw_enabled,
            'variable': 'particle_energy (reconstructed main-cluster energy, X plane)',
            'scheme': 'w = target(E)/h_class(E); target = mean of the two class '
                      'histograms; classes renormalised to equal total weight',
            'per_split': rw_info,
        },
        'data_summary': {
            'n_train': n_train, 'n_val': n_val, 'n_test': int(len(y_te)),
            'train_cats': dcfg['train_cats'], 'val_cats': dcfg['val_cats'],
            'test_cats': dcfg['test_cats'],
            'n_train_cats_used': len(train_cats_used),
            'n_val_cats_used': len(val_cats_used),
            'n_test_cats_used': len(sorted(set(int(c) for c in te['cat_ids']))),
        },
        'test_metrics': {
            'accuracy': acc,
            'auc': auc,
            'accuracy_reweighted': acc_w,
            'auc_reweighted': auc_w,
            'confusion_matrix_normalized': cm.tolist(),
            'classification_report': report,
        },
        'epochs_trained': len(history.history.get('loss', [])),
        'history_keys': list(history.history.keys()),
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
