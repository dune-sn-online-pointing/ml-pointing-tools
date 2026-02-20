#!/usr/bin/env python3
"""
Data loader for three-plane volume images for electron direction prediction.
Loads 1m x 1m volume images from ES production data.
"""

import os
import numpy as np


def load_ed_volumes(data_dir, max_samples=None, max_files=None, shuffle=True, verbose=True):
    """
    Load three-plane volume data for electron direction prediction.
    
    Args:
        data_dir: Base directory containing U/, V/, X/ subfolders with NPZ files
        max_samples: Maximum total samples to load (None = all)
        max_files: Maximum number of NPZ files to process (None = all)
        shuffle: Whether to shuffle data
        verbose: Print loading info
        
    Returns:
        (images_u, images_v, images_x, directions, energies, metadata)
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print("LOADING VOLUME DATA FOR ELECTRON DIRECTION")
        print(f"{'='*70}")
        print(f"Data directory: {data_dir}")
    
    # Check for U/V/X subdirectories
    u_dir = os.path.join(data_dir, 'U')
    v_dir = os.path.join(data_dir, 'V')
    x_dir = os.path.join(data_dir, 'X')
    
    if not all(os.path.exists(d) for d in [u_dir, v_dir, x_dir]):
        raise ValueError(f"Missing U/V/X subdirectories in {data_dir}")
    
    # Get list of NPZ files (use X plane as reference)
    x_files = sorted([f for f in os.listdir(x_dir) if f.endswith('_planeX.npz')])
    
    if not x_files:
        raise ValueError(f"No NPZ files found in {x_dir}")
    
    if max_files:
        x_files = x_files[:max_files]
    
    if verbose:
        print(f"Found {len(x_files)} NPZ files to process")
    
    # Load all volumes
    all_u_volumes = []
    all_v_volumes = []
    all_x_volumes = []
    all_directions = []
    all_energies = []
    all_metadata = []
    
    total_loaded = 0
    
    for file_idx, x_filename in enumerate(x_files):
        if max_samples and total_loaded >= max_samples:
            break
        
        # Construct corresponding U and V filenames
        base_name = x_filename.replace('_planeX.npz', '')
        u_filename = base_name + '_planeU.npz'
        v_filename = base_name + '_planeV.npz'
        
        try:
            u_data = np.load(os.path.join(u_dir, u_filename), allow_pickle=True)
            v_data = np.load(os.path.join(v_dir, v_filename), allow_pickle=True)
            x_data = np.load(os.path.join(x_dir, x_filename), allow_pickle=True)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not load file {file_idx+1}/{len(x_files)}: {e}")
            continue
        
        u_images = u_data['images']
        u_metadata = u_data['metadata']
        v_images = v_data['images']
        v_metadata = v_data['metadata']
        x_images = x_data['images']
        x_metadata = x_data['metadata']
        
        # Match volumes by event and main_cluster_match_id
        for i, x_meta in enumerate(x_metadata):
            if max_samples and total_loaded >= max_samples:
                break
            
            event = x_meta.get('event', -1)
            match_id = x_meta.get('main_cluster_match_id', -1)
            
            # Find matching U and V volumes
            u_idx = None
            v_idx = None
            
            for j, u_meta in enumerate(u_metadata):
                if (u_meta.get('event') == event and 
                    u_meta.get('main_cluster_match_id') == match_id):
                    u_idx = j
                    break
            
            for j, v_meta in enumerate(v_metadata):
                if (v_meta.get('event') == event and 
                    v_meta.get('main_cluster_match_id') == match_id):
                    v_idx = j
                    break
            
            if u_idx is None or v_idx is None:
                continue
            
            # Extract momentum as direction
            mom_x = x_meta.get('main_track_momentum_x', 0)
            mom_y = x_meta.get('main_track_momentum_y', 0)
            mom_z = x_meta.get('main_track_momentum_z', 0)
            
            momentum = np.array([mom_x, mom_y, mom_z], dtype=np.float32)
            momentum_norm = np.linalg.norm(momentum)
            
            if momentum_norm < 1e-6:
                continue  # Skip if no momentum
            
            direction = momentum / momentum_norm
            energy = x_meta.get('particle_energy', 0)
            
            # Store volumes and labels
            all_u_volumes.append(u_images[u_idx])
            all_v_volumes.append(v_images[v_idx])
            all_x_volumes.append(x_images[i])  # i is the x_idx from the enumerate loop
            all_directions.append(direction)
            all_energies.append(energy)
            all_metadata.append(x_meta)
            
            total_loaded += 1
        
        if verbose and (file_idx + 1) % 100 == 0:
            print(f"  Processed {file_idx+1}/{len(x_files)} files, loaded {total_loaded} volumes")
    
    if verbose:
        print(f"\nTotal matched: {total_loaded} three-plane volumes")
    
    # Normalize volumes
    if verbose:
        print("Normalizing volumes...")
    
    normalized_u = []
    normalized_v = []
    normalized_x = []
    
    for i in range(len(all_u_volumes)):
        u_img = all_u_volumes[i].astype(np.float32)
        v_img = all_v_volumes[i].astype(np.float32)
        x_img = all_x_volumes[i].astype(np.float32)
        
        # Normalize each plane independently
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
    directions = np.array(all_directions, dtype=np.float32)
    energies = np.array(all_energies, dtype=np.float32)
    metadata = np.array(all_metadata, dtype=object)
    
    # Shuffle if requested
    if shuffle:
        if verbose:
            print("Shuffling data...")
        indices = np.arange(len(directions))
        np.random.shuffle(indices)
        images_u = images_u[indices]
        images_v = images_v[indices]
        images_x = images_x[indices]
        directions = directions[indices]
        energies = energies[indices]
        metadata = metadata[indices]
    
    if verbose:
        print(f"\nFinal dataset:")
        print(f"  U volumes: {images_u.shape}")
        print(f"  V volumes: {images_v.shape}")
        print(f"  X volumes: {images_x.shape}")
        print(f"  Directions: {directions.shape}")
        print(f"  Direction norms: min={np.min(np.linalg.norm(directions, axis=1)):.3f}, "
              f"max={np.max(np.linalg.norm(directions, axis=1)):.3f}")
        print(f"  Energy range: {np.min(energies):.1f} - {np.max(energies):.1f} MeV")
    
    return images_u, images_v, images_x, directions, energies, metadata


if __name__ == '__main__':
    # Test loading
    data_dir = '/eos/user/e/evilla/dune/sn-tps/prod_es/es_production_volume_images_tick3_ch2_min2_tot3_e2p0'
    images_u, images_v, images_x, directions, energies, metadata = load_ed_volumes(
        data_dir=data_dir,
        max_samples=100,
        max_files=2,
        verbose=True
    )
    print("\n✓ Data loader test successful!")
