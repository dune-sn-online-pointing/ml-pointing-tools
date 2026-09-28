"""
Loader / packer for the ES production three-plane ED training pool.

This is the pool that trained ED v58 (`es_production_cluster_images_tick3_ch2_min2_tot3_e2p0`),
rebuilt with the FIXED 3-plane matcher into
`/eos/project-e/ep-nu/evilla/sn-online-pointing/prod_es_matchfix/`.

SELECTION — reproduced from what v58 actually used, i.e.
`refactor-ml-for-pointing/python/data_loader.py::load_three_plane_matched`
(called by `models/train_three_plane_simple.py`, the trainer of v58):

  * X-plane rows with match_id (col 13) != -1  (match_clusters only assigns a
    match_id to MAIN X clusters, so this is implicitly "X main track")
  * the pair (event, match_id) must also exist in the U and V metadata with
    match_id != -1.  NOTE: no main-track requirement on the U/V partners --
    this is the v58 training selection, and it is LOOSER than the deployed
    pipeline selection (sample_loader.py requires main-track partners).
    `partners_main` is stored per sample so the deployed subset can be selected.
  * direction target = normalized (px, py, pz) from X metadata cols 7-9
  * images fed RAW (float32 ADC, channel dim appended), as v58 / ed_inference.py

SPLIT — v58's own split is NOT recoverable (train_three_plane_simple.py shuffles
with an unseeded global numpy RNG and stores no file provenance), so this module
uses a fixed-seed split BY SOURCE FILE: the sorted list of X-plane npz files is
permuted with numpy.random.RandomState(seed) and cut 70/15/15.  A file-level
split also guarantees that no event can leak between splits.

Metadata columns of the current products (18):
  0 event, 1 is_marley, 2 is_main_track, 3 is_es_interaction, 4-6 true_pos,
  7-9 true particle momentum (px,py,pz), 10 cluster_energy (MeV, reco),
  11 true particle energy (MeV), 12 plane, 13 match_id, 14 nu_energy,
  15-17 nu momentum.   (v58's products had only cols 0-13; 0-13 are identical.)
"""

import glob
import json
import os

import numpy as np

SPLITS = ('train', 'val', 'test')
IMG_SHAPE = (128, 32, 1)


# --------------------------------------------------------------------------- #
# file discovery and split
# --------------------------------------------------------------------------- #
def list_x_files(images_dir, x_glob='*_planeX.npz'):
    """Sorted list of X-plane npz files that have both U and V partners on disk."""
    x_files = sorted(glob.glob(os.path.join(images_dir, 'X', x_glob)))
    keep = []
    for fx in x_files:
        name = os.path.basename(fx)
        fu = os.path.join(images_dir, 'U', name.replace('planeX', 'planeU'))
        fv = os.path.join(images_dir, 'V', name.replace('planeX', 'planeV'))
        if os.path.exists(fu) and os.path.exists(fv):
            keep.append(fx)
    return keep


def split_files(x_files, seed=42, fractions=(0.70, 0.15, 0.15)):
    """Fixed-seed file-level split. Returns {split: [file paths]} (sorted inside)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(x_files))
    n = len(x_files)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    idx = {'train': perm[:n_train],
           'val': perm[n_train:n_train + n_val],
           'test': perm[n_train + n_val:]}
    return {s: sorted(x_files[i] for i in v) for s, v in idx.items()}


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def _uv_index(meta, main_only=False):
    """{(event, match_id): row} for rows with match_id != -1 (optionally main-track)."""
    out = {}
    for i, m in enumerate(meta):
        if m[13] != -1 and (not main_only or m[2] == 1):
            out[(m[0], m[13])] = i
    return out


def select_file(fx, fu, fv, with_images=True, require_main_partners=False,
                es_main_only=False):
    """Apply the v58 three-plane selection to one (X, U, V) npz triplet.

    require_main_partners=True switches to the DEPLOYED pipeline selection
    (sample_loader.py): the U and V partners must themselves be main-track
    clusters.  es_main_only=True additionally requires the X row to be a
    true-ES MARLEY main track (metadata cols 1, 2, 3), as the pipeline's
    perfect-CT scenario does.

    Returns dict of per-sample arrays; images only if with_images.
    """
    dx = np.load(fx, allow_pickle=False)
    mx = dx['metadata']
    du = np.load(fu, allow_pickle=False)
    mu = du['metadata']
    dv = np.load(fv, allow_pickle=False)
    mv = dv['metadata']

    idx_u = _uv_index(mu, main_only=require_main_partners)
    idx_v = _uv_index(mv, main_only=require_main_partners)

    rows_x, rows_u, rows_v = [], [], []
    for ix, m in enumerate(mx):
        if m[13] == -1:
            continue
        if es_main_only and not (m[1] == 1 and m[2] == 1 and m[3] == 1):
            continue
        key = (m[0], m[13])
        iu = idx_u.get(key)
        iv = idx_v.get(key)
        if iu is None or iv is None:
            continue
        rows_x.append(ix)
        rows_u.append(iu)
        rows_v.append(iv)

    rows_x = np.asarray(rows_x, dtype=np.int32)
    rows_u = np.asarray(rows_u, dtype=np.int32)
    rows_v = np.asarray(rows_v, dtype=np.int32)

    if rows_x.size == 0:
        out = {'n': 0}
        if with_images:
            out.update({'images_u': np.zeros((0,) + IMG_SHAPE, np.float32),
                        'images_v': np.zeros((0,) + IMG_SHAPE, np.float32),
                        'images_x': np.zeros((0,) + IMG_SHAPE, np.float32)})
        for k in ('metadata',):
            out[k] = np.zeros((0, mx.shape[1]), np.float32)
        for k in ('e_u', 'e_v', 'e_x', 'r', 'true_energy', 'event', 'match_id',
                  'x_row', 'partners_main'):
            out[k] = np.zeros(0, np.float32)
        out['directions'] = np.zeros((0, 3), np.float32)
        return out

    meta_x = mx[rows_x].astype(np.float32)
    e_x = meta_x[:, 10]
    e_u = mu[rows_u, 10].astype(np.float32)
    e_v = mv[rows_v, 10].astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(e_x > 0, np.minimum(e_u, e_v) / e_x, np.nan).astype(np.float32)
    partners_main = ((mu[rows_u, 2] == 1) & (mv[rows_v, 2] == 1)).astype(np.float32)

    mom = meta_x[:, 7:10]
    norms = np.linalg.norm(mom, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    directions = (mom / norms).astype(np.float32)

    out = {
        'n': int(rows_x.size),
        'metadata': meta_x,
        'directions': directions,
        'e_x': e_x, 'e_u': e_u, 'e_v': e_v, 'r': r,
        'true_energy': meta_x[:, 11],
        'event': meta_x[:, 0],
        'match_id': meta_x[:, 13],
        'x_row': rows_x.astype(np.float32),
        'partners_main': partners_main,
    }
    if with_images:
        out['images_x'] = dx['images'][rows_x].astype(np.float32)[..., np.newaxis]
        out['images_u'] = du['images'][rows_u].astype(np.float32)[..., np.newaxis]
        out['images_v'] = dv['images'][rows_v].astype(np.float32)[..., np.newaxis]
    dx.close(); du.close(); dv.close()
    return out


def uv_paths(fx):
    name = os.path.basename(fx)
    d = os.path.dirname(os.path.dirname(fx))
    return (os.path.join(d, 'U', name.replace('planeX', 'planeU')),
            os.path.join(d, 'V', name.replace('planeX', 'planeV')))


# --------------------------------------------------------------------------- #
# packed-pool IO
# --------------------------------------------------------------------------- #
SCALARS = ('directions', 'e_x', 'e_u', 'e_v', 'r', 'true_energy', 'event',
           'match_id', 'x_row', 'partners_main', 'metadata')


def load_packed(packed_dir, split, mmap=True):
    """Load one packed split. Images come back as memmaps unless mmap=False."""
    mode = 'r' if mmap else None
    out = {}
    for p in ('u', 'v', 'x'):
        out[f'images_{p}'] = np.load(os.path.join(packed_dir, f'{split}_{p}.npy'),
                                     mmap_mode=mode)
    z = np.load(os.path.join(packed_dir, f'{split}_meta.npz'), allow_pickle=False)
    for k in z.files:
        out[k] = z[k]
    z.close()
    out['cluster_energy'] = out['e_x']
    return out


def load_manifest(packed_dir):
    with open(os.path.join(packed_dir, 'split_manifest.json')) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# metrics helpers (same definitions as burstcats_es_loader, kept local)
# --------------------------------------------------------------------------- #
def normalize_rows(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (n + 1e-8)


def angular_errors_deg(pred, true):
    p = normalize_rows(np.asarray(pred, dtype=np.float64))
    t = normalize_rows(np.asarray(true, dtype=np.float64))
    cos = np.clip(np.sum(p * t, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def cos_from_err(err_deg):
    return np.cos(np.radians(np.asarray(err_deg, dtype=np.float64)))


def overall_stats(err):
    err = np.asarray(err, dtype=np.float64)
    if err.size == 0:
        return {'N': 0}
    return {
        'N': int(err.size),
        'median_deg': float(np.median(err)),
        'mean_deg': float(np.mean(err)),
        'q68_deg': float(np.percentile(err, 68)),
        'std_deg': float(np.std(err)),
        'frac_lt_90deg': float(np.mean(err < 90.0)),
        'mean_cos': float(np.mean(np.cos(np.radians(err)))),
    }


ENERGY_BINS = [3, 5, 10, 20, 30, 1e9]
ENERGY_LABELS = ['3-5', '5-10', '10-20', '20-30', '30+']


def per_energy_table(err, energy, bins=ENERGY_BINS, labels=ENERGY_LABELS):
    err = np.asarray(err, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    rows = []
    for lab, lo, hi in zip(labels, bins[:-1], bins[1:]):
        m = (energy >= lo) & (energy < hi)
        s = overall_stats(err[m])
        s['bin'] = lab
        rows.append(s)
    return rows


def format_table(rows, title=''):
    lines = []
    if title:
        lines.append(title)
    lines.append(f"{'E bin [MeV]':>12} {'N':>7} {'<cos>':>7} {'median':>8} {'Q68':>8}")
    for r in rows:
        if r['N'] > 0:
            lines.append(f"{r['bin']:>12} {r['N']:>7d} {r['mean_cos']:>7.3f} "
                         f"{r['median_deg']:>8.2f} {r['q68_deg']:>8.2f}")
        else:
            lines.append(f"{r['bin']:>12} {0:>7d} {'-':>7} {'-':>8} {'-':>8}")
    return '\n'.join(lines)


def format_overall(stats, title=''):
    if stats.get('N', 0) == 0:
        return f'{title}: no samples'
    return (f"{title}: N={stats['N']}  <cos>={stats['mean_cos']:.3f}  "
            f"median={stats['median_deg']:.2f}  Q68={stats['q68_deg']:.2f}  "
            f"mean={stats['mean_deg']:.2f}  f<90={stats['frac_lt_90deg']:.3f}")
