#!/usr/bin/env python3
"""Standalone volume preprocessing + auxiliary-feature extraction for CT v83/v84.

This module is the single source of truth for

  * the auxiliary scalar features fed to the CT v84 aux branch, and
  * the image preprocessing (log1p, optional per-image normalization)

so that the snop-pipeline can reproduce the training-time inputs bit-for-bit.
`channel_tagging/models/train_ct_volume_v83.py` imports these functions; it does
not reimplement them.  Anything that changes here changes training too.

Label convention (unchanged from v52/v79/v80): ES = 0, CC = 1, so P(ES) is
`model.predict(...)[:, 0]`.

Typical inference use (v84):

    from ct_volume_features import compute_aux_features, preprocess_image
    x   = preprocess_image(img, per_image_norm='max')       # (208, 1242) float32
    aux = compute_aux_features(img, meta)                    # (4,) float32, RAW
    aux = (aux - aux_mean) / aux_std                         # from results.json
    p   = model.predict([x[None, ..., None], aux[None, :]])
    p_es = p[0, 0]

`aux_mean` / `aux_std` are stored in the model's `results.json` under
`preprocessing.aux_mean` / `preprocessing.aux_std`, in the same order as
`aux_feature_names()`.

For v83 (no aux branch) only `preprocess_image(img, per_image_norm=None)` is
needed and the model takes a single input, exactly like v80.
"""

import numpy as np

IMAGE_SHAPE = (208, 1242)

# v80/v80_aux aux branch, plus the reconstructed main-cluster energy (v84)
AUX_FEATURE_NAMES_BASE = ['n_clusters_in_volume', 'log1p_total_adc', 'log1p_n_nonzero']
AUX_FEATURE_NAMES_ENERGY = AUX_FEATURE_NAMES_BASE + ['log1p_particle_energy']

_EPS = 1e-6


def aux_feature_names(include_energy=True):
    """Names of the aux features, in the order `compute_aux_features` returns them."""
    return list(AUX_FEATURE_NAMES_ENERGY if include_energy else AUX_FEATURE_NAMES_BASE)


def compute_aux_features(image, metadata, include_energy=True):
    """Truth-free auxiliary scalars for one volume image.

    Parameters
    ----------
    image : array_like, shape (208, 1242)
        RAW volume image (ADC integrals per pixel), i.e. exactly what is stored
        in `<cat>_volume_images_*/X/*.npz['images'][i]`.  NOT log1p'd, NOT
        normalized.  The aux features are always computed from the raw image,
        whatever image preprocessing the model uses.
    metadata : dict or None
        The matching entry of `...npz['metadata']`.  Only `n_clusters_in_volume`
        and `particle_energy` are read.  `particle_energy` is the RECONSTRUCTED
        main-cluster energy (main-cluster ADC integral / calibration constant),
        an online observable - not a truth quantity.
    include_energy : bool
        If True (v84) append log1p(particle_energy) as a 4th scalar.

    Returns
    -------
    np.ndarray, shape (3,) or (4,), float32 -- RAW (un-standardized) features.
    """
    img = np.asarray(image, dtype=np.float32)
    n_clusters = 1.0
    energy = 0.0
    if isinstance(metadata, dict):
        v = metadata.get('n_clusters_in_volume', 1)
        n_clusters = float(v) if v is not None else 1.0
        v = metadata.get('particle_energy', None)
        energy = float(v) if v is not None else 0.0
    if not np.isfinite(energy):
        energy = 0.0

    total_adc = float(img.sum())
    n_nonzero = float(np.count_nonzero(img))

    feats = [n_clusters, np.log1p(total_adc), np.log1p(n_nonzero)]
    if include_energy:
        feats.append(np.log1p(max(energy, 0.0)))
    return np.asarray(feats, dtype=np.float32)


def preprocess_image(image, per_image_norm=None):
    """Image preprocessing shared by v80/v83 (per_image_norm=None) and v84.

    v80/v83 : x = log1p(raw ADC)                       (absolute scale kept)
    v84     : x = log1p(raw ADC) / max(log1p(raw ADC)) (absolute level removed)

    Parameters
    ----------
    per_image_norm : None | 'max' | 'sum'
        'max' divides the log1p image by its own maximum pixel (v84).
        'sum' divides it by its own sum.  None leaves it alone (v80/v83).

    Returns
    -------
    np.ndarray, shape (208, 1242), float32
    """
    img = np.log1p(np.asarray(image, dtype=np.float32))
    if per_image_norm in (None, False, 'none'):
        return img
    if per_image_norm == 'max':
        denom = float(img.max())
    elif per_image_norm == 'sum':
        denom = float(img.sum())
    else:
        raise ValueError(f'unknown per_image_norm: {per_image_norm!r}')
    return img / (denom + _EPS)


def standardize_aux(aux, aux_mean, aux_std):
    """Apply the training-set standardization recorded in results.json."""
    return (np.asarray(aux, dtype=np.float32) - np.asarray(aux_mean, dtype=np.float32)) / \
        np.asarray(aux_std, dtype=np.float32)
