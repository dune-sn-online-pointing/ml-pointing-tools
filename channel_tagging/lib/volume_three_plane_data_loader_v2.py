#!/usr/bin/env python3
"""
Data loader for three-plane volume images for channel tagging (Updated V2).
Loads volume images from BOTH ES and CC directories and combines them.
"""

import os
import numpy as np


def load_three_plane_volumes_from_two_dirs(es_data_dir, cc_data_dir, max_samples_per_class=None, 
                                           train_frac=0.7, val_frac=0.15, shuffle=True, verbose=True):
    """
    Load three-plane matched volume data for channel tagging from BOTH ES and CC directories.
    
    Args:
        es_data_dir: ES directory containing U/, V/, X/ subfolders with NPZ files
        cc_data_dir: CC directory containing U/, V/, X/ subfolders with NPZ files
        max_samples_per_class: Maximum number of samples to load PER CLASS (None = all)
        train_frac: Fraction for training (default 0.7)
        val_frac: Fraction for validation (default 0.15)
        shuffle: Whether to shuffle data (default True)
        verbose: Print loading info (default True)
        
    Returns:
        (train_data, val_data, test_data) where each is:
            ((u_images, v_images, x_images), labels, metadata)
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print("THREE-PLANE VOLUME DATA LOADER (ES + CC)")
        print(f"{'='*70}")
        print(f"ES directory: {es_data_dir}")
        print(f"CC directory: {cc_data_dir}")
    
    # Load ES data
    if verbose:
        print(f"\n--- Loading ES data ---")
    es_data = _load_single_class_data(es_data_dir, max_samples_per_class, verbose=verbose)
    
    # Load CC data
    if verbose:
        print(f"\n--- Loading CC data ---")
    cc_data = _load_single_class_data(cc_data_dir, max_samples_per_class, verbose=verbose)
    
    # Combine data
    if verbose:
        print(f"\n--- Combining datasets ---")
    
    images_u = np.concatenate([es_data['images_u'], cc_data['images_u']], axis=0)
    images_v = np.concatenate([es_data['images_v'], cc_data['images_v']], axis=0)
    images_x = np.concatenate([es_data['images_x'], cc_data['images_x']], axis=0)
    labels = np.concatenate([es_data['labels'], cc_data['labels']], axis=0)
    metadata = np.concatenate([es_data['metadata'], cc_data['metadata']], axis=0)
    
    if verbose:
        print(f"Total combined: {len(labels)} samples")
        print(f"  ES: {np.sum(labels == 0)} ({100*np.sum(labels == 0)/len(labels):.1f}%)")
        print(f"  CC: {np.sum(labels == 1)} ({100*np.sum(labels == 1)/len(labels):.1f}%)")
    
    # Shuffle if requested
    if shuffle:
        if verbose:
            print("\nShuffling data...")
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        images_u = images_u[indices]
        images_v = images_v[indices]
        images_x = images_x[indices]
        labels = labels[indices]
        metadata = metadata[indices]
    
    # Split into train/val/test
    n = len(labels)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    train_data = (
        (images_u[:n_train], images_v[:n_train], images_x[:n_train]),
        labels[:n_train],
        metadata[:n_train]
    )
    
    val_data = (
        (images_u[n_train:n_train+n_val], images_v[n_train:n_train+n_val], images_x[n_train:n_train+n_val]),
        labels[n_train:n_train+n_val],
        metadata[n_train:n_train+n_val]
    )
    
    test_data = (
        (images_u[n_train+n_val:], images_v[n_train+n_val:], images_x[n_train+n_val:]),
        labels[n_train+n_val:],
        metadata[n_train+n_val:]
    )
    
    if verbose:
        print(f"\nDataset splits:")
        print(f"  Train: {len(train_data[1])} samples")
        print(f"  Val:   {len(val_data[1])} samples")
        print(f"  Test:  {len(test_data[1])} samples")
    
    return train_data, val_data, test_data


def _load_single_class_data(data_dir, max_samples=None, verbose=True):
    """Load data from a single class directory (ES or CC)."""
    
    # Check for U/V/X subdirectories
    u_dir = os.path.join(data_dir, 'U')
    v_dir = os.path.join(data_dir, 'V')
    x_dir = os.path.join(data_dir, 'X')
    
    if not all(os.path.exists(d) for d in [u_dir, v_dir, x_dir]):
        raise ValueError(f"Missing U/V/X subdirectories in {data_dir}")
    
    # Get list of files from U plane (all planes should have same files)
    u_files = sorted([f for f in os.listdir(u_dir) if f.endswith('.npz')])
    
    if not u_files:
        raise ValueError(f"No NPZ files found in {u_dir}")
    
    if verbose:
        print(f"Found {len(u_files)} NPZ files per plane")
    
    # Load and match volumes by main_cluster_match_id
    all_u_volumes = []
    all_v_volumes = []
    all_x_volumes = []
    all_labels = []
    all_metadata = []
    
    total_loaded = 0
    
    for filename in u_files:
        if max_samples and total_loaded >= max_samples:
            break
        
        # Load all three planes for this file
        try:
            u_data = np.load(os.path.join(u_dir, filename), allow_pickle=True)
            v_data = np.load(os.path.join(v_dir, filename.replace('_planeU', '_planeV')), allow_pickle=True)
            x_data = np.load(os.path.join(x_dir, filename.replace('_planeU', '_planeX')), allow_pickle=True)
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
            continue
        
        u_images = u_data['images']
        u_metadata = u_data['metadata']
        v_images = v_data['images']
        v_metadata = v_data['metadata']
        x_images = x_data['images']
        x_metadata = x_data['metadata']
        
        # Match volumes by match_id
        # For each U volume, find corresponding V and X volumes
        for i, u_meta in enumerate(u_metadata):
            if max_samples and total_loaded >= max_samples:
                break
            
            match_id = u_meta.get('main_cluster_match_id', -1)
            
            # Find matching V and X volumes
            v_idx = None
            x_idx = None
            
            for j, v_meta in enumerate(v_metadata):
                if v_meta.get('main_cluster_match_id', -1) == match_id and \
                   v_meta.get('event') == u_meta.get('event'):
                    v_idx = j
                    break
            
            for j, x_meta in enumerate(x_metadata):
                if x_meta.get('main_cluster_match_id', -1) == match_id and \
                   x_meta.get('event') == u_meta.get('event'):
                    x_idx = j
                    break
            
            # Only include if all three planes match
            if v_idx is not None and x_idx is not None:
                all_u_volumes.append(u_images[i])
                all_v_volumes.append(v_images[v_idx])
                all_x_volumes.append(x_images[x_idx])
                
                # Determine label from interaction type
                interaction_type = u_meta.get('interaction_type', 'Background')
                if interaction_type == 'ES':
                    label = 0
                elif interaction_type == 'CC':
                    label = 1
                else:  # Background or NC
                    label = 2
                
                all_labels.append(label)
                all_metadata.append(u_meta)
                
                total_loaded += 1
    
    if verbose:
        print(f"Matched {total_loaded} three-plane volumes")
    
    # Convert to numpy arrays
    images_u = np.array(all_u_volumes, dtype=object)
    images_v = np.array(all_v_volumes, dtype=object)
    images_x = np.array(all_x_volumes, dtype=object)
    labels = np.array(all_labels, dtype=np.int32)
    metadata = np.array(all_metadata, dtype=object)
    
    # Normalize each image independently
    if verbose:
        print("Normalizing images...")
    
    normalized_u = []
    normalized_v = []
    normalized_x = []
    
    for i in range(len(images_u)):
        u_img = images_u[i].astype(np.float32)
        v_img = images_v[i].astype(np.float32)
        x_img = images_x[i].astype(np.float32)
        
        # Normalize each plane
        u_max = np.max(u_img)
        if u_max > 0:
            u_img /= u_max
        
        v_max = np.max(v_img)
        if v_max > 0:
            v_img /= v_max
        
        x_max = np.max(x_img)
        if x_max > 0:
            x_img /= x_max
        
        # Add channel dimension
        normalized_u.append(np.expand_dims(u_img, axis=-1))
        normalized_v.append(np.expand_dims(v_img, axis=-1))
        normalized_x.append(np.expand_dims(x_img, axis=-1))
    
    images_u = np.array(normalized_u, dtype=np.float32)
    images_v = np.array(normalized_v, dtype=np.float32)
    images_x = np.array(normalized_x, dtype=np.float32)
    
    if verbose:
        print(f"Final dataset:")
        print(f"  U plane: {images_u.shape}")
        print(f"  V plane: {images_v.shape}")
        print(f"  X plane: {images_x.shape}")
        print(f"  Labels: {labels.shape}")
    
    return {
        'images_u': images_u,
        'images_v': images_v,
        'images_x': images_x,
        'labels': labels,
        'metadata': metadata
    }
