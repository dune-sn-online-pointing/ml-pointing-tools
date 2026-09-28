"""
Three-plane ES cluster loader for sn-burst-samples cats (electron direction).

Mirrors EXACTLY the selection used by the snop-pipeline three-plane loader
(refactor-snop-pipeline/python/lib/sample_loader.py::_load_samples_from_folder,
load_all_planes=True):

  * only main-track clusters (metadata col 2 == 1)
  * only clusters with a match_id (col 13) != -1
  * the match_id must be present as a main-track match_id in ALL of X, U, V
  * the three plane images with that match_id are the three inputs
  * the X-plane metadata row is the reference (direction target = normalized
    px,py,pz from cols 7-9; cluster_energy col 10; true energy col 11)

Images are fed RAW (float32 ADC, channel dim appended), exactly as the
pipeline's ed_inference.py feeds the deployed v58 model. No normalization.

Metadata columns (18):
  0 event, 1 is_marley, 2 is_main_track, 3 is_es_interaction, 4-6 true_pos,
  7-9 true particle momentum (px,py,pz), 10 cluster_energy (MeV, reco),
  11 true particle energy (MeV), 12 plane, 13 match_id, 14 nu_energy,
  15-17 nu momentum.
"""

import glob
import os

import numpy as np

DEFAULT_SUBDIR_FMT = '{cat}_cluster_images_tick3_ch2_min2_tot3_e3p0'
DEFAULT_X_GLOB = 'es_*_bg_matched_planeX.npz'
ENERGY_BINS = [2, 3, 5, 7, 10, 15, 20, 30, 50, 70]


def cat_dirs_for_range(base_dir, cat_lo, cat_hi, subdir_fmt=DEFAULT_SUBDIR_FMT):
    """Return [(cat_id, cluster_images_dir)] for existing cats in [cat_lo, cat_hi]."""
    out = []
    for i in range(cat_lo, cat_hi + 1):
        cat = f'cat{i:06d}'
        d = os.path.join(base_dir, cat, subdir_fmt.format(cat=cat))
        if os.path.isdir(os.path.join(d, 'X')) and os.path.isdir(os.path.join(d, 'U')) \
                and os.path.isdir(os.path.join(d, 'V')):
            out.append((i, d))
    return out


def _main_track_match_index(meta):
    """{match_id: row index} for main-track rows with match_id != -1 (pipeline logic)."""
    is_main = meta[:, 2] == 1
    return {int(m): i for i, m in enumerate(meta[:, 13]) if is_main[i] and m != -1}


def load_three_plane_es_from_cats(cat_dirs, x_glob=DEFAULT_X_GLOB, verbose=True,
                                  split_name='', max_files=None):
    """Load pipeline-selected three-plane ES clusters from a list of (cat_id, dir).

    Returns dict with:
      images_u, images_v, images_x : (N, 128, 32, 1) float32 raw ADC
      directions                   : (N, 3) unit vectors from X metadata cols 7-9
      cluster_energy               : (N,) reco cluster energy (MeV), col 10
      true_energy                  : (N,) true particle energy (MeV), col 11
      metadata                     : (N, 18) X-plane metadata
      cat_ids                      : (N,) int cat id per sample
      n_cats, n_file_triplets, n_x_files_seen, missing_uv
    """
    imgs_u, imgs_v, imgs_x, metas, cats = [], [], [], [], []
    n_triplets = 0
    n_x_seen = 0
    n_missing_uv = 0
    n_files_total = 0

    for cat_id, d in cat_dirs:
        x_files = sorted(glob.glob(os.path.join(d, 'X', x_glob)))
        for fx in x_files:
            if max_files is not None and n_files_total >= max_files:
                break
            n_files_total += 1
            n_x_seen += 1
            name = os.path.basename(fx)
            fu = os.path.join(d, 'U', name.replace('planeX', 'planeU'))
            fv = os.path.join(d, 'V', name.replace('planeX', 'planeV'))
            if not (os.path.exists(fu) and os.path.exists(fv)):
                n_missing_uv += 1
                continue
            try:
                dx = np.load(fx, allow_pickle=True)
                du = np.load(fu, allow_pickle=True)
                dv = np.load(fv, allow_pickle=True)
                ix_img, mx = dx['images'], dx['metadata']
                iu_img, mu = du['images'], du['metadata']
                iv_img, mv = dv['images'], dv['metadata']
            except Exception as e:  # noqa: BLE001
                print(f'  Warning: failed to load {name}: {e}')
                continue

            idx_x = _main_track_match_index(mx)
            idx_u = _main_track_match_index(mu)
            idx_v = _main_track_match_index(mv)
            common = set(idx_x) & set(idx_u) & set(idx_v)
            for mid in sorted(common):
                imgs_x.append(np.asarray(ix_img[idx_x[mid]], dtype=np.float32))
                imgs_u.append(np.asarray(iu_img[idx_u[mid]], dtype=np.float32))
                imgs_v.append(np.asarray(iv_img[idx_v[mid]], dtype=np.float32))
                metas.append(np.asarray(mx[idx_x[mid]], dtype=np.float32))
                cats.append(cat_id)
            n_triplets += 1
        if verbose:
            print(f'  [{split_name}] cat{cat_id:06d}: {len(x_files)} X files, '
                  f'running total {len(metas)} clusters', flush=True)

    if not metas:
        raise RuntimeError(f'No usable three-plane ES clusters found for split {split_name}')

    images_u = np.stack(imgs_u)[..., np.newaxis]
    images_v = np.stack(imgs_v)[..., np.newaxis]
    images_x = np.stack(imgs_x)[..., np.newaxis]
    metadata = np.stack(metas)
    cat_ids = np.asarray(cats, dtype=np.int32)

    mom = metadata[:, 7:10].astype(np.float32)
    norms = np.linalg.norm(mom, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    directions = (mom / norms).astype(np.float32)

    out = {
        'images_u': images_u, 'images_v': images_v, 'images_x': images_x,
        'directions': directions,
        'cluster_energy': metadata[:, 10].astype(np.float32),
        'true_energy': metadata[:, 11].astype(np.float32),
        'metadata': metadata,
        'cat_ids': cat_ids,
        'n_cats': len(cat_dirs),
        'n_file_triplets': n_triplets,
        'n_x_files_seen': n_x_seen,
        'missing_uv': n_missing_uv,
    }
    if verbose:
        print(f'[{split_name}] {out["n_cats"]} cats, {n_triplets} file triplets '
              f'({n_missing_uv} X files without U/V), {len(metadata)} clusters '
              f'({len(metadata) / max(n_triplets, 1):.1f}/file); '
              f'images {images_x.shape} {images_x.dtype}', flush=True)
    return out


def shuffle_split(data, rng):
    """In-place consistent shuffle of all per-sample arrays in a loaded split."""
    n = len(data['directions'])
    perm = rng.permutation(n)
    for k in ('images_u', 'images_v', 'images_x', 'directions', 'cluster_energy',
              'true_energy', 'metadata', 'cat_ids'):
        data[k] = data[k][perm]
    return data


def normalize_rows(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (n + 1e-8)


def angular_errors_deg(pred, true):
    p = normalize_rows(np.asarray(pred, dtype=np.float64))
    t = normalize_rows(np.asarray(true, dtype=np.float64))
    cos = np.clip(np.sum(p * t, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def overall_stats(err):
    err = np.asarray(err, dtype=np.float64)
    if err.size == 0:
        return {'N': 0}
    return {
        'N': int(err.size),
        'median_deg': float(np.median(err)),
        'mean_deg': float(np.mean(err)),
        'q68_deg': float(np.percentile(err, 68)),
        'q25_deg': float(np.percentile(err, 25)),
        'q75_deg': float(np.percentile(err, 75)),
        'std_deg': float(np.std(err)),
        'frac_lt_90deg': float(np.mean(err < 90.0)),
        'mean_cos': float(np.mean(np.cos(np.radians(err)))),
    }


def per_energy_table(err, energy, bins=ENERGY_BINS):
    """Rows: bin label, N, median, mean, Q68 of angular error (deg), frac < 90 deg."""
    err = np.asarray(err, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (energy >= lo) & (energy < hi)
        n = int(m.sum())
        row = {'bin': f'{lo}-{hi}', 'e_lo': lo, 'e_hi': hi, 'N': n}
        if n > 0:
            e = err[m]
            row.update({
                'median_deg': float(np.median(e)),
                'mean_deg': float(np.mean(e)),
                'q68_deg': float(np.percentile(e, 68)),
                'frac_lt_90deg': float(np.mean(e < 90.0)),
            })
        else:
            row.update({'median_deg': None, 'mean_deg': None, 'q68_deg': None,
                        'frac_lt_90deg': None})
        rows.append(row)
    return rows


def format_table(rows, title=''):
    lines = []
    if title:
        lines.append(title)
    lines.append(f"{'E bin [MeV]':>12} {'N':>6} {'median':>8} {'mean':>8} {'Q68':>8} {'f<90':>6}")
    for r in rows:
        if r['N'] > 0:
            lines.append(f"{r['bin']:>12} {r['N']:>6d} {r['median_deg']:>8.2f} "
                         f"{r['mean_deg']:>8.2f} {r['q68_deg']:>8.2f} {r['frac_lt_90deg']:>6.3f}")
        else:
            lines.append(f"{r['bin']:>12} {0:>6d} {'-':>8} {'-':>8} {'-':>8} {'-':>6}")
    return '\n'.join(lines)


def format_overall(stats, title=''):
    if stats.get('N', 0) == 0:
        return f'{title}: no samples'
    return (f"{title}: N={stats['N']}  median={stats['median_deg']:.2f}  "
            f"mean={stats['mean_deg']:.2f}  Q68={stats['q68_deg']:.2f}  "
            f"f<90={stats['frac_lt_90deg']:.3f}  <cos>={stats['mean_cos']:.3f}")
