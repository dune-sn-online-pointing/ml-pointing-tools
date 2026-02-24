"""
Data loader for three-plane electron direction training.
Loads U, V, and X plane images simultaneously and their corresponding direction labels.

Metadata format (13 columns per sample):
  0: event number
  1: is_marley flag (0 or 1)
  2: is_main_track flag (0 or 1)
  3: is_es_interaction flag (0 or 1)
  4-6: true_pos (x, y, z) in cm
  7-9: true_particle_mom (px, py, pz) in GeV/c
  10: cluster_energy in MeV
  11: true_particle_energy in GeV
  12: plane_number (0=U, 1=V, 2=X)

Direction extraction: Normalize momentum vector (cols 7-9) to get unit direction.
"""

import numpy as np
import glob
import os
from typing import List, Tuple


def load_three_plane_data(
    data_directories: List[str],
    max_samples: int = None,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    shuffle: bool = True,
    filter_main_tracks: bool = True,
    random_seed: int = 42
):
    """
    Load data from all three planes (U, V, X) simultaneously.
    
    Args:
        data_directories: List of directories containing cluster images
        max_samples: Maximum number of samples to load
        train_fraction: Fraction for training
        val_fraction: Fraction for validation
        test_fraction: Fraction for testing
        shuffle: Whether to shuffle data
        filter_main_tracks: Only include main track clusters
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of ((train_u, train_v, train_x, train_y),
                  (val_u, val_v, val_x, val_y),
                  (test_u, test_v, test_x, test_y))
    """
    
    np.random.seed(random_seed)
    
    print("=" * 60)
    print("THREE-PLANE DATA LOADER")
    print("=" * 60)
    
    # Find all files for each plane
    all_files = {'U': [], 'V': [], 'X': []}
    
    for data_dir in data_directories:
        for plane in ['U', 'V', 'X']:
            pattern = os.path.join(data_dir, f'*_plane{plane}.npz')
            files = glob.glob(pattern)
            all_files[plane].extend(files)
    
    print(f"Found {len(all_files['U'])} U files")
    print(f"Found {len(all_files['V'])} V files")
    print(f"Found {len(all_files['X'])} X files")
    
    # Group files by base name (without plane suffix)
    file_groups = {}
    for plane in ['U', 'V', 'X']:
        for file_path in all_files[plane]:
            base_name = file_path.replace(f'_plane{plane}.npz', '')
            if base_name not in file_groups:
                file_groups[base_name] = {}
            file_groups[base_name][plane] = file_path
    
    # Only keep groups that have all three planes
    complete_groups = {k: v for k, v in file_groups.items() 
                       if len(v) == 3 and 'U' in v and 'V' in v and 'X' in v}
    
    print(f"\nFound {len(complete_groups)} complete file groups (all 3 planes)")
    
    if len(complete_groups) == 0:
        raise ValueError("No complete file groups found with all three planes!")
    
    # Load data from all planes
    images_u, images_v, images_x = [], [], []
    directions = []
    
    print("\nLoading data...")
    for i, (base_name, plane_files) in enumerate(complete_groups.items()):
        if i % 100 == 0 and i > 0:
            print(f"  Processed {i}/{len(complete_groups)} file groups...")
        
        try:
            # Load all three planes
            data_u = np.load(plane_files['U'], allow_pickle=True)
            data_v = np.load(plane_files['V'], allow_pickle=True)
            data_x = np.load(plane_files['X'], allow_pickle=True)
            
            imgs_u = data_u['images']
            imgs_v = data_v['images']
            imgs_x = data_x['images']
            
            meta_u = data_u['metadata']
            
            # Extract direction from metadata
            # Metadata format (13 columns):
            # 0: event, 1: is_marley, 2: is_main_track, 3: is_es_interaction,
            # 4-6: true_pos (x,y,z), 7-9: true_particle_mom (px,py,pz),
            # 10: cluster_energy, 11: true_particle_energy, 12: plane_number
            if len(meta_u.shape) > 1:
                n_cols = meta_u.shape[1]
                if n_cols == 13:
                    # Columns 7-9 are momentum (px, py, pz) - need to normalize to get direction
                    momentum = meta_u[:, 7:10].astype(np.float32)
                    # Normalize to get direction
                    mom_mag = np.linalg.norm(momentum, axis=1, keepdims=True)
                    # Avoid division by zero
                    mom_mag = np.where(mom_mag == 0, 1.0, mom_mag)
                    dirs = momentum / mom_mag
                elif n_cols in [11, 12]:
                    # Old format fallback
                    dirs = meta_u[:, 3:6].astype(np.float32)
                else:
                    print(f"  Warning: Unexpected metadata columns: {n_cols}")
                    continue
            else:
                # Object array with dicts
                if isinstance(meta_u[0], dict) and 'direction' in meta_u[0]:
                    dirs = np.array([m['direction'] for m in meta_u])
                else:
                    continue
            
            # Filter main tracks if requested
            if filter_main_tracks:
                # Only filter if metadata has is_main_track field
                if isinstance(meta_u[0], dict) and 'is_main_track' in meta_u[0]:
                    is_main = np.array([m.get('is_main_track', False) for m in meta_u])
                    imgs_u = imgs_u[is_main]
                    imgs_v = imgs_v[is_main]
                    imgs_x = imgs_x[is_main]
                    dirs = dirs[is_main]
                elif len(meta_u.shape) > 1 and meta_u.shape[1] >= 1:
                    # Old format: first column after offset might be is_main_track
                    # For production_es data, all are main tracks, so skip filtering
                    pass
            
            if len(imgs_u) > 0:
                images_u.append(imgs_u)
                images_v.append(imgs_v)
                images_x.append(imgs_x)
                directions.append(dirs)
                
        except Exception as e:
            print(f"  Warning: Error loading {base_name}: {e}")
            continue
    
    # Concatenate all data
    images_u = np.concatenate(images_u, axis=0)
    images_v = np.concatenate(images_v, axis=0)
    images_x = np.concatenate(images_x, axis=0)
    directions = np.concatenate(directions, axis=0)
    
    print(f"\nTotal samples loaded: {len(images_u)}")
    print(f"Image U shape: {images_u.shape}")
    print(f"Image V shape: {images_v.shape}")
    print(f"Image X shape: {images_x.shape}")
    print(f"Directions shape: {directions.shape}")
    
    # Limit samples if requested
    if max_samples and len(images_u) > max_samples:
        indices = np.random.choice(len(images_u), max_samples, replace=False)
        images_u = images_u[indices]
        images_v = images_v[indices]
        images_x = images_x[indices]
        directions = directions[indices]
        print(f"Limited to {max_samples} samples")
    
    # Normalize images
    images_u = images_u.astype(np.float32)
    images_v = images_v.astype(np.float32)
    images_x = images_x.astype(np.float32)
    
    # Add channel dimension if needed
    if len(images_u.shape) == 3:
        images_u = np.expand_dims(images_u, axis=-1)
        images_v = np.expand_dims(images_v, axis=-1)
        images_x = np.expand_dims(images_x, axis=-1)
    
    # Shuffle if requested
    if shuffle:
        indices = np.random.permutation(len(images_u))
        images_u = images_u[indices]
        images_v = images_v[indices]
        images_x = images_x[indices]
        directions = directions[indices]
    
    # Split into train/val/test
    n_samples = len(images_u)
    n_train = int(n_samples * train_fraction)
    n_val = int(n_samples * val_fraction)
    
    train_u = images_u[:n_train]
    train_v = images_v[:n_train]
    train_x = images_x[:n_train]
    train_y = directions[:n_train]
    
    val_u = images_u[n_train:n_train+n_val]
    val_v = images_v[n_train:n_train+n_val]
    val_x = images_x[n_train:n_train+n_val]
    val_y = directions[n_train:n_train+n_val]
    
    test_u = images_u[n_train+n_val:]
    test_v = images_v[n_train+n_val:]
    test_x = images_x[n_train+n_val:]
    test_y = directions[n_train+n_val:]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_u)}")
    print(f"  Val: {len(val_u)}")
    print(f"  Test: {len(test_u)}")
    print("=" * 60)
    
    return ((train_u, train_v, train_x, train_y),
            (val_u, val_v, val_x, val_y),
            (test_u, test_v, test_x, test_y))
