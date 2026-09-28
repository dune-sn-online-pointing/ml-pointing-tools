#!/usr/bin/env python3
"""
ED v61: three-plane electron-direction training on sn-burst-samples ES clusters.

Same architecture / optimizer / schedule as v58 (three_plane_cnn, 4 conv x 64,
2 dense x 256, dropout 0.3, Adam lr 1e-3 clipnorm 0.5, ReduceLROnPlateau 0.5/10,
early stopping 30, best checkpoint by val_loss), with the differences:

- Data: sn-burst-samples cluster images at production conditions
  (tick3_ch2_min2_tot3_e3p0), ES only, selected exactly like the snop-pipeline
  three-plane loader (main-track match_id present in X, U, V).
- Cat-level split identical to CT v80: train 400-571, val 572-596, test 597-621
  (missing cats skipped). Cats 1-399 and >= 623 are never touched.
- Loss: `safe_angular_loss` = v58's angular loss (mean arccos(cos)) with the
  cosine clipped to [-1+eps, 1+eps] instead of [-1, 1]. v58 went NaN at epoch
  13; d/dx arccos(x) diverges at |x| = 1, which the hard clip does not prevent.
  Plus TerminateOnNaN. Set training.loss = "angular_loss" to use the original.
- Baseline: every model in `baseline_models` is evaluated on the SAME test
  arrays (apples to apples) and its per-energy table is stored in results.json.

Outputs (in <output.base_dir>/<model.name>_<timestamp>/):
  best_model.keras, checkpoints/model_epoch_XX_val_loss_X.XXXX.keras,
  results.json, training_history.csv, val_predictions.npz,
  test_predictions.npz, baseline_<tag>_test_predictions.npz,
  per_energy_tables_test.txt
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                   # three_plane_cnn, direction_losses
sys.path.insert(0, os.path.join(_HERE, '..', 'lib'))        # burstcats_es_loader
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'python'))

import burstcats_es_loader as bl  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='Train ED v61 on burst-sample ES clusters')
    p.add_argument('-j', '--json', required=True, help='JSON config')
    p.add_argument('--dry-run', action='store_true',
                   help='Load a few cats, build model, one train step on a tiny batch, '
                        'run baseline eval on the small test subset, no fit')
    p.add_argument('--max-cats-per-split', type=int, default=None,
                   help='Limit cats per split (debug)')
    p.add_argument('--output-dir', default=None,
                   help='Override output directory (debug); default <base_dir>/<name>_<ts>')
    return p.parse_args()


def make_safe_angular_loss(eps):
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package='DirectionLosses')
    def safe_angular_loss(y_true, y_pred):
        y_true_n = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred_n = tf.nn.l2_normalize(y_pred, axis=-1)
        cos = tf.reduce_sum(y_true_n * y_pred_n, axis=-1)
        cos = tf.clip_by_value(cos, -1.0 + eps, 1.0 - eps)
        return tf.reduce_mean(tf.acos(cos))

    return safe_angular_loss


def load_split(dcfg, split, max_cats=None):
    lo, hi = dcfg[f'{split}_cats']
    dirs = bl.cat_dirs_for_range(dcfg['burst_samples_dir'], lo, hi,
                                 dcfg.get('cluster_subdir_fmt', bl.DEFAULT_SUBDIR_FMT))
    if max_cats:
        dirs = dirs[:max_cats]
    expected = hi - lo + 1
    print(f'\n[{split}] cats {lo}-{hi}: {len(dirs)}/{expected} cat dirs with X/U/V present; '
          f'missing: {[i for i in range(lo, hi + 1) if i not in {c for c, _ in dirs}]}',
          flush=True)
    if not dirs:
        raise RuntimeError(f'No cat dirs for split {split} in {lo}-{hi}')
    t0 = time.time()
    data = bl.load_three_plane_es_from_cats(
        dirs, x_glob=dcfg.get('x_file_glob', bl.DEFAULT_X_GLOB), verbose=False,
        split_name=split)
    print(f'[{split}] loaded in {time.time() - t0:.0f}s', flush=True)
    data['cats_used'] = [c for c, _ in dirs]
    data['cats_missing'] = [i for i in range(lo, hi + 1) if i not in set(data['cats_used'])]
    return data


def split_summary(d):
    return {
        'n_samples': int(len(d['directions'])),
        'n_cats_used': int(d['n_cats']),
        'cats_used': [int(c) for c in d['cats_used']],
        'cats_missing': [int(c) for c in d['cats_missing']],
        'n_file_triplets': int(d['n_file_triplets']),
        'n_x_files_without_uv': int(d['missing_uv']),
        'clusters_per_file': float(len(d['directions']) / max(d['n_file_triplets'], 1)),
        'true_energy_MeV': {
            'min': float(d['true_energy'].min()), 'median': float(np.median(d['true_energy'])),
            'max': float(d['true_energy'].max())},
    }


def evaluate(model, d, batch_size, tag, out_dir, fname):
    """Predict on a loaded split, save npz, return (overall, table_true, table_reco)."""
    preds = model.predict([d['images_u'], d['images_v'], d['images_x']],
                          batch_size=batch_size, verbose=0)
    preds = np.asarray(preds, dtype=np.float32)
    if preds.ndim == 3 and preds.shape[1] == 1:
        preds = preds[:, 0, :]
    preds_n = bl.normalize_rows(preds)
    err = bl.angular_errors_deg(preds_n, d['directions'])
    np.savez_compressed(
        os.path.join(out_dir, fname),
        predictions=preds_n.astype(np.float32),
        predictions_raw=preds,
        true_directions=d['directions'],
        angular_errors=err.astype(np.float32),
        energies=d['cluster_energy'],          # col 10, same meaning as v58 val_predictions.npz
        cluster_energy=d['cluster_energy'],
        true_energy=d['true_energy'],
        cat_ids=d['cat_ids'],
        metadata=d['metadata'])
    overall = bl.overall_stats(err)
    t_true = bl.per_energy_table(err, d['true_energy'])
    t_reco = bl.per_energy_table(err, d['cluster_energy'])
    print(bl.format_overall(overall, f'{tag} overall'))
    print(bl.format_table(t_true, f'{tag}: per TRUE energy'))
    print(bl.format_table(t_reco, f'{tag}: per RECO cluster energy'))
    print(flush=True)
    return {'overall': overall, 'per_true_energy': t_true, 'per_reco_energy': t_reco}


def main():
    args = parse_args()
    with open(args.json) as f:
        config = json.load(f)
    dcfg, mcfg, tcfg, ocfg = config['data'], config['model'], config['training'], config['output']
    name = mcfg['name']
    seed = int(tcfg.get('seed', 42))
    rng = np.random.RandomState(seed)

    print('=' * 70)
    print(f'ED v61 THREE-PLANE TRAINING ON BURST-SAMPLE ES CLUSTERS: {name}')
    print('=' * 70)
    print(f'Config: {args.json}')
    print(json.dumps(config, indent=2))
    print(flush=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.output_dir or os.path.join(ocfg['base_dir'], f'{name}_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f'Output directory: {out_dir}', flush=True)

    # ------------------------------------------------------------------ data
    max_cats = args.max_cats_per_split or (2 if args.dry_run else None)
    t_load = time.time()
    train = load_split(dcfg, 'train', max_cats)
    val = load_split(dcfg, 'val', max_cats)
    test = load_split(dcfg, 'test', max_cats)
    load_seconds = time.time() - t_load
    bl.shuffle_split(train, rng)   # val/test keep file order (deterministic)
    counts = {s: split_summary(d) for s, d in (('train', train), ('val', val), ('test', test))}
    print('\nSample counts:')
    for s in counts:
        print(f"  {s:5s}: {counts[s]['n_samples']:6d} clusters from {counts[s]['n_cats_used']} cats "
              f"({counts[s]['n_file_triplets']} file triplets, "
              f"{counts[s]['clusters_per_file']:.1f}/file); missing cats {counts[s]['cats_missing']}")
    # leakage guard
    assert not (set(train['cats_used']) & set(val['cats_used'])), 'train/val cat overlap'
    assert not (set(train['cats_used']) & set(test['cats_used'])), 'train/test cat overlap'
    assert not (set(val['cats_used']) & set(test['cats_used'])), 'val/test cat overlap'
    for d in (train, val, test):
        assert d['cat_ids'].min() >= 400 and d['cat_ids'].max() <= 622, 'forbidden cat used'
    print(flush=True)

    # ----------------------------------------------------------------- model
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(seed)
    np.random.seed(seed)
    from three_plane_cnn import build_three_plane_cnn   # v58 architecture
    from direction_losses import angular_loss

    model = build_three_plane_cnn(
        input_shape=tuple(mcfg['input_shape']), output_dim=mcfg['output_dim'],
        n_conv_layers=mcfg['n_conv_layers'], n_filters=mcfg['n_filters'],
        kernel_size=mcfg['kernel_size'], n_dense_layers=mcfg['n_dense_layers'],
        n_dense_units=mcfg['n_dense_units'], learning_rate=tcfg['learning_rate'])

    loss_name = tcfg.get('loss', 'safe_angular_loss')
    if loss_name == 'safe_angular_loss':
        loss_fn = make_safe_angular_loss(float(tcfg.get('safe_angular_eps', 1e-6)))
    elif loss_name == 'angular_loss':
        loss_fn = angular_loss
    else:
        raise ValueError(f'Unsupported loss: {loss_name}')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=tcfg['learning_rate'],
                                        clipnorm=tcfg.get('clipnorm', 0.5)),
        loss=loss_fn, metrics=['mae'])
    model.summary()
    print(f'Loss: {loss_name}; clipnorm={tcfg.get("clipnorm", 0.5)}; '
          f'lr={tcfg["learning_rate"]}; batch={tcfg["batch_size"]}', flush=True)

    batch_size = int(tcfg['batch_size'])
    X_tr = [train['images_u'], train['images_v'], train['images_x']]
    X_va = [val['images_u'], val['images_v'], val['images_x']]

    results = {
        'model_name': name, 'version': config.get('version'), 'timestamp': timestamp,
        'output_dir': out_dir, 'config': config,
        'preprocessing': {'image': 'raw float32 ADC', 'per_image_normalization': False,
                          'input_order': ['U', 'V', 'X'], 'input_shape': mcfg['input_shape']},
        'selection': dcfg.get('selection'),
        'data_summary': counts, 'data_load_seconds': load_seconds,
        'environment': {'tensorflow': tf.__version__, 'keras': keras.__version__,
                        'host': os.uname().nodename,
                        'gpus': [g.name for g in tf.config.list_physical_devices('GPU')]},
    }

    # --------------------------------------------------------------- dry run
    if args.dry_run:
        print('*** DRY RUN: one train step on 8 samples, no fit ***')
        nb = min(8, len(train['directions']))
        l0 = model.train_on_batch([x[:nb] for x in X_tr], train['directions'][:nb])
        print(f'train_on_batch loss/mae: {l0}')
        assert np.all(np.isfinite(np.asarray(l0))), 'non-finite loss in dry run'
        results['history'] = {}
        results['epochs_trained'] = 0
    else:
        # ---------------------------------------------------------- training
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(ckpt_dir, 'model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras'),
                monitor='val_loss', save_best_only=True, verbose=1),
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=int(tcfg.get('early_stopping_patience', 30)),
                restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=float(tcfg.get('reduce_lr_factor', 0.5)),
                patience=int(tcfg.get('reduce_lr_patience', 10)),
                min_lr=float(tcfg.get('reduce_lr_min_lr', 1e-6)), verbose=1),
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.CSVLogger(os.path.join(out_dir, 'training_history.csv')),
        ]
        t_fit = time.time()
        history = model.fit(X_tr, train['directions'],
                            validation_data=(X_va, val['directions']),
                            epochs=int(tcfg['epochs']), batch_size=batch_size,
                            callbacks=callbacks, verbose=2, shuffle=True)
        fit_seconds = time.time() - t_fit
        hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        val_losses = np.asarray(hist.get('val_loss', []), dtype=float)
        finite = np.isfinite(val_losses)
        best_epoch = int(np.nanargmin(np.where(finite, val_losses, np.inf)) + 1) if finite.any() else None
        results['history'] = hist
        results['epochs_trained'] = int(len(val_losses))
        results['best_epoch'] = best_epoch
        results['best_val_loss'] = float(val_losses[best_epoch - 1]) if best_epoch else None
        results['best_val_loss_deg'] = float(np.degrees(val_losses[best_epoch - 1])) if best_epoch else None
        results['final_lr'] = float(hist['learning_rate'][-1]) if 'learning_rate' in hist else None
        results['nan_encountered'] = bool((~np.isfinite(np.asarray(hist.get('loss', [])))).any())
        results['fit_seconds'] = fit_seconds
        print(f'\nTraining done: {results["epochs_trained"]} epochs, best epoch {best_epoch}, '
              f'best val_loss {results["best_val_loss"]} rad '
              f'({results["best_val_loss_deg"]:.2f} deg mean angular error), '
              f'{fit_seconds / 60:.1f} min', flush=True)

    # EarlyStopping restored best weights (or dry-run weights); persist them.
    model.save(os.path.join(out_dir, 'best_model.keras'))

    # ------------------------------------------------------------ evaluation
    print('\n' + '=' * 70 + '\nEVALUATION (model weights = best val_loss epoch)\n' + '=' * 70)
    results['val_metrics'] = evaluate(model, val, batch_size, f'{name} VAL', out_dir, 'val_predictions.npz')
    results['test_metrics'] = evaluate(model, test, batch_size, f'{name} TEST', out_dir, 'test_predictions.npz')

    # --------------------------------------------------------------- baselines
    results['baselines'] = {}
    for b in config.get('baseline_models', []):
        tag, path = b['tag'], b['path']
        print('=' * 70 + f'\nBASELINE {tag}: {path}\n' + '=' * 70)
        try:
            bm = keras.models.load_model(path, compile=False)
            bshape = [tuple(i.shape[1:]) for i in bm.inputs]
            print(f'  loaded; inputs {bshape}')
            r = evaluate(bm, test, batch_size, f'{tag} TEST', out_dir, f'baseline_{tag}_test_predictions.npz')
            r['path'] = path
            results['baselines'][tag] = r
            del bm
        except Exception as e:  # noqa: BLE001
            print(f'  baseline {tag} FAILED: {e!r}')
            results['baselines'][tag] = {'path': path, 'error': repr(e)}

    # ------------------------------------------------------------ write out
    lines = [f'{name}  ({out_dir})', f'TEST cats {dcfg["test_cats"]}: {counts["test"]["n_samples"]} clusters '
             f'from {counts["test"]["n_cats_used"]} cats', '']
    lines.append(bl.format_overall(results['test_metrics']['overall'], f'{name} TEST'))
    lines.append(bl.format_table(results['test_metrics']['per_true_energy'], f'{name} TEST per TRUE energy'))
    lines.append('')
    lines.append(bl.format_table(results['test_metrics']['per_reco_energy'], f'{name} TEST per RECO energy'))
    for tag, r in results['baselines'].items():
        if 'overall' in r:
            lines += ['', bl.format_overall(r['overall'], f'{tag} TEST'),
                      bl.format_table(r['per_true_energy'], f'{tag} TEST per TRUE energy'), '',
                      bl.format_table(r['per_reco_energy'], f'{tag} TEST per RECO energy')]
    with open(os.path.join(out_dir, 'per_energy_tables_test.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    shutil.copy(args.json, os.path.join(out_dir, os.path.basename(args.json)))
    print(f'\nDone. Results in {out_dir}', flush=True)


if __name__ == '__main__':
    main()
