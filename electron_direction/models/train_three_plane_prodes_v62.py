#!/usr/bin/env python3
"""
ED v62 / v63: three-plane electron-direction training on the ES PRODUCTION pool
rebuilt with the FIXED 3-plane matcher.

v62 = the v58 recipe, retrained from scratch on the matcher-fixed pool.
v63 = same data, initialised from the deployed v58 weights at a reduced learning
      rate (set `training.init_from` and `training.learning_rate`).

Reproduced from v58 (models/train_three_plane_simple.py + json of results.json):
  architecture three_plane_cnn 4 conv x 64 (doubling), kernel 3, 2 dense x 256,
  dropout 0.3, input (128, 32, 1); Adam lr 1e-3 with clipnorm 0.5 (the v58
  trainer hard-coded clipnorm=0.5 and ignored the 0.4 in its json); loss
  angular_loss = mean arccos(cos); batch 64; epochs 200; EarlyStopping(30,
  restore_best); ReduceLROnPlateau(0.5, 10, min_lr 1e-6); best-val checkpointing;
  images raw float32 ADC, no normalization.

Deliberate deviations from v58, all documented in results.json:
  * loss defaults to `safe_angular_loss` (cos clipped to [-1+eps, 1-eps]).
    v58's run went NaN at epoch 13 and its deployed weights are the epoch-12
    checkpoint; d/dx arccos diverges at |x|=1 and the hard clip does not help.
    Set training.loss = "angular_loss" for the bit-exact v58 loss.
  * proper train/val/test split (0.70/0.15/0.15) BY SOURCE FILE with a fixed
    seed.  v58 used train_split 0.7 and evaluated on the remaining 30% ("val"),
    with an unseeded shuffle and no held-out test set.
  * TerminateOnNaN + CSVLogger.

Outputs in <output.base_dir>/<model.name>_<timestamp>/:
  best_model.keras, checkpoints/, results.json, training_history.csv,
  val_predictions.npz, test_predictions.npz, baseline_<tag>_test_predictions.npz
Each *_predictions.npz carries, per event: reco dir, true dir, true energy,
cluster energies per plane, r = min(E_U,E_V)/E_X, partners_main, and the
provenance (file index into the split manifest, event, match_id) that the
SAME/NEW legacy-rule flag is joined on.
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
sys.path.insert(0, _HERE)                                # three_plane_cnn, direction_losses
sys.path.insert(0, os.path.join(_HERE, '..', 'lib'))     # prodes_matchfix_loader

import prodes_matchfix_loader as pl  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='Train ED v62/v63 on the matchfix production pool')
    p.add_argument('-j', '--json', required=True)
    p.add_argument('--dry-run', action='store_true',
                   help='load a small slice, build, one train step, no fit')
    p.add_argument('--max-samples-per-split', type=int, default=None)
    p.add_argument('--output-dir', default=None)
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


def load_split(packed_dir, split, limit=None):
    t0 = time.time()
    d = pl.load_packed(packed_dir, split, mmap=True)
    n = len(d['e_x'])
    if limit:
        n = min(n, limit)
    for k in list(d):
        d[k] = np.ascontiguousarray(np.asarray(d[k][:n]))
    gb = sum(d[f'images_{p}'].nbytes for p in 'uvx') / 1e9
    print(f'[{split}] {n} samples, images {gb:.2f} GB in RAM, {time.time() - t0:.0f}s',
          flush=True)
    return d


def split_summary(d):
    return {
        'n_samples': int(len(d['e_x'])),
        'n_files': int(len(np.unique(d['file_idx']))),
        'frac_partners_main': float(np.mean(d['partners_main'])),
        'cluster_energy_MeV': {'min': float(d['e_x'].min()),
                               'median': float(np.median(d['e_x'])),
                               'max': float(d['e_x'].max())},
        'true_energy_MeV': {'min': float(d['true_energy'].min()),
                            'median': float(np.median(d['true_energy'])),
                            'max': float(d['true_energy'].max())},
        'r_median': float(np.nanmedian(d['r'])),
    }


def evaluate(model, d, batch_size, tag, out_dir, fname):
    preds = model.predict([d['images_u'], d['images_v'], d['images_x']],
                          batch_size=batch_size, verbose=0)
    preds = np.asarray(preds, dtype=np.float32)
    if preds.ndim == 3 and preds.shape[1] == 1:
        preds = preds[:, 0, :]
    preds_n = pl.normalize_rows(preds).astype(np.float32)
    err = pl.angular_errors_deg(preds_n, d['directions'])
    np.savez(os.path.join(out_dir, fname),
             predictions=preds_n,
             predictions_raw=preds,
             true_directions=d['directions'],
             angular_errors=err.astype(np.float32),
             cos_reco_true=pl.cos_from_err(err).astype(np.float32),
             true_energy=d['true_energy'],
             energies=d['e_x'],          # same meaning as v58's val_predictions.npz
             e_x=d['e_x'], e_u=d['e_u'], e_v=d['e_v'], r=d['r'],
             partners_main=d['partners_main'],
             file_idx=d['file_idx'], event=d['event'], match_id=d['match_id'],
             x_row=d['x_row'], metadata=d['metadata'])
    overall = pl.overall_stats(err)
    t_true = pl.per_energy_table(err, d['true_energy'])
    t_reco = pl.per_energy_table(err, d['e_x'])
    print(pl.format_overall(overall, f'{tag} overall'))
    print(pl.format_table(t_true, f'{tag}: per TRUE energy'))
    print(pl.format_table(t_reco, f'{tag}: per RECO cluster energy'))
    print(flush=True)
    return {'overall': overall, 'per_true_energy': t_true, 'per_reco_energy': t_reco}


def main():
    args = parse_args()
    with open(args.json) as f:
        config = json.load(f)
    dcfg, mcfg, tcfg, ocfg = config['data'], config['model'], config['training'], config['output']
    name = mcfg['name']
    seed = int(tcfg.get('seed', 42))

    print('=' * 70)
    print(f'ED TRAINING ON THE MATCHFIX ES PRODUCTION POOL: {name}')
    print('=' * 70)
    print(json.dumps(config, indent=2), flush=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.output_dir or os.path.join(ocfg['base_dir'], f'{name}_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f'Output directory: {out_dir}', flush=True)

    # ------------------------------------------------------------------ data
    packed = dcfg['packed_dir']
    manifest = pl.load_manifest(packed)
    limit = args.max_samples_per_split or (2000 if args.dry_run else None)
    t_load = time.time()
    train = load_split(packed, 'train', limit)
    val = load_split(packed, 'val', limit)
    test = load_split(packed, 'test', limit)
    load_seconds = time.time() - t_load
    counts = {s: split_summary(d) for s, d in (('train', train), ('val', val), ('test', test))}
    print('\nSample counts: ' + json.dumps({s: counts[s]['n_samples'] for s in counts}), flush=True)

    # ----------------------------------------------------------------- model
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(seed)
    np.random.seed(seed)
    from three_plane_cnn import build_three_plane_cnn
    from direction_losses import angular_loss

    loss_name = tcfg.get('loss', 'safe_angular_loss')
    if loss_name == 'safe_angular_loss':
        loss_fn = make_safe_angular_loss(float(tcfg.get('safe_angular_eps', 1e-6)))
    elif loss_name == 'angular_loss':
        loss_fn = angular_loss
    else:
        raise ValueError(f'Unsupported loss: {loss_name}')

    init_from = tcfg.get('init_from')
    if init_from:
        print(f'Initialising weights from {init_from}', flush=True)
        model = keras.models.load_model(init_from, compile=False)
    else:
        model = build_three_plane_cnn(
            input_shape=tuple(mcfg['input_shape']), output_dim=mcfg['output_dim'],
            n_conv_layers=mcfg['n_conv_layers'], n_filters=mcfg['n_filters'],
            kernel_size=mcfg['kernel_size'], n_dense_layers=mcfg['n_dense_layers'],
            n_dense_units=mcfg['n_dense_units'], learning_rate=tcfg['learning_rate'])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=tcfg['learning_rate'],
                                        clipnorm=tcfg.get('clipnorm', 0.5)),
        loss=loss_fn, metrics=['mae'])
    model.summary()
    print(f'Loss: {loss_name}; clipnorm={tcfg.get("clipnorm", 0.5)}; '
          f'lr={tcfg["learning_rate"]}; batch={tcfg["batch_size"]}; '
          f'init_from={init_from}', flush=True)

    batch_size = int(tcfg['batch_size'])
    X_tr = [train['images_u'], train['images_v'], train['images_x']]
    X_va = [val['images_u'], val['images_v'], val['images_x']]

    results = {
        'model_name': name, 'version': config.get('version'), 'timestamp': timestamp,
        'output_dir': out_dir, 'config': config,
        'pool_manifest': {k: v for k, v in manifest.items() if k != 'files'},
        'preprocessing': {'image': 'raw float32 ADC', 'per_image_normalization': False,
                          'input_order': ['U', 'V', 'X'], 'input_shape': mcfg['input_shape']},
        'data_summary': counts, 'data_load_seconds': load_seconds,
        'environment': {'tensorflow': tf.__version__, 'keras': keras.__version__,
                        'host': os.uname().nodename,
                        'gpus': [g.name for g in tf.config.list_physical_devices('GPU')]},
    }

    if args.dry_run:
        print('*** DRY RUN ***')
        nb = min(8, len(train['e_x']))
        l0 = model.train_on_batch([x[:nb] for x in X_tr], train['directions'][:nb])
        print(f'train_on_batch loss/mae: {l0}')
        assert np.all(np.isfinite(np.asarray(l0))), 'non-finite loss in dry run'
        results['history'] = {}
        results['epochs_trained'] = 0
    else:
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
        results.update({
            'history': hist,
            'epochs_trained': int(len(val_losses)),
            'best_epoch': best_epoch,
            'best_val_loss': float(val_losses[best_epoch - 1]) if best_epoch else None,
            'best_val_loss_deg': float(np.degrees(val_losses[best_epoch - 1])) if best_epoch else None,
            'nan_encountered': bool((~np.isfinite(np.asarray(hist.get('loss', [])))).any()),
            'fit_seconds': fit_seconds,
        })
        print(f'\nTraining done: {results["epochs_trained"]} epochs, best epoch {best_epoch}, '
              f'best val_loss {results["best_val_loss"]} rad, {fit_seconds / 60:.1f} min',
              flush=True)

    model.save(os.path.join(out_dir, 'best_model.keras'))

    print('\n' + '=' * 70 + '\nEVALUATION (weights = best val_loss epoch)\n' + '=' * 70)
    results['val_metrics'] = evaluate(model, val, batch_size, f'{name} VAL', out_dir,
                                      'val_predictions.npz')
    results['test_metrics'] = evaluate(model, test, batch_size, f'{name} TEST', out_dir,
                                       'test_predictions.npz')

    results['baselines'] = {}
    for b in config.get('baseline_models', []):
        tag, path = b['tag'], b['path']
        print('=' * 70 + f'\nBASELINE {tag}: {path}\n' + '=' * 70)
        try:
            bm = keras.models.load_model(path, compile=False)
            r = evaluate(bm, test, batch_size, f'{tag} TEST', out_dir,
                         f'baseline_{tag}_test_predictions.npz')
            r['path'] = path
            results['baselines'][tag] = r
            del bm
        except Exception as e:  # noqa: BLE001
            print(f'  baseline {tag} FAILED: {e!r}')
            results['baselines'][tag] = {'path': path, 'error': repr(e)}

    lines = [f'{name}  ({out_dir})',
             f'TEST: {counts["test"]["n_samples"]} clusters from {counts["test"]["n_files"]} files', '']
    lines.append(pl.format_overall(results['test_metrics']['overall'], f'{name} TEST'))
    lines.append(pl.format_table(results['test_metrics']['per_true_energy'],
                                 f'{name} TEST per TRUE energy'))
    lines.append('')
    lines.append(pl.format_table(results['test_metrics']['per_reco_energy'],
                                 f'{name} TEST per RECO energy'))
    for tag, r in results['baselines'].items():
        if 'overall' in r:
            lines += ['', pl.format_overall(r['overall'], f'{tag} TEST'),
                      pl.format_table(r['per_true_energy'], f'{tag} TEST per TRUE energy'), '',
                      pl.format_table(r['per_reco_energy'], f'{tag} TEST per RECO energy')]
    with open(os.path.join(out_dir, 'per_energy_tables_test.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    shutil.copy(args.json, os.path.join(out_dir, os.path.basename(args.json)))
    print(f'\nDone. Results in {out_dir}', flush=True)


if __name__ == '__main__':
    main()
