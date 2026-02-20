"""
Data loader for pre-matched three-plane dataset.

Loads from the preprocessed NPZ file containing matched triplets.
Much faster than on-the-fly matching during training.

File structure:
  - images_u: U plane images (N, 128, 16)
  - images_v: V plane images (N, 128, 16)
  - images_x: X plane images (N, 128, 16)
  - metadata: Metadata array (N, 14)
  - match_ids: Match ID for each triplet (N,)

Metadata columns (14):
  [0]: event
  [1]: is_marley
  [2]: is_main_track
  [3]: is_es_interaction
  [4-6]: true_pos (x, y, z)
  [7-9]: true_particle_mom (px, py, pz)
  [10]: cluster_energy
  [11]: true_particle_energy
  [12]: plane_id
  [13]: match_id
"""

import numpy as np
from typing import Tuple


def load_matched_three_plane_data(
    matched_file_path: str,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    shuffle: bool = True,
    random_seed: int = 42,
    min_energy: float = None
) -> Tuple:
    """
    Load pre-matched three-plane dataset and split into train/val/test.
    
    Args:
        matched_file_path: Path to pre-matched NPZ file
        train_fraction: Fraction for training
        val_fraction: Fraction for validation
        test_fraction: Fraction for testing
        shuffle: Whether to shuffle data before splitting
        random_seed: Random seed for reproducibility
        min_energy: Minimum energy threshold in MeV (optional)
        
    Returns:
        Tuple of ((train_u, train_v, train_x, train_y),
                  (val_u, val_v, val_x, val_y),
                  (test_u, test_v, test_x, test_y))
    """
    
    np.random.seed(random_seed)
    
    print("=" * 70)
    print("MATCHED THREE-PLANE DATA LOADER")
    print("=" * 70)
    print(f"Loading from: {matched_file_path}")
    
    # Load pre-matched data
    data = np.load(matched_file_path)
    
    images_u = data['images_u']
    images_v = data['images_v']
    images_x = data['images_x']
    metadata = data['metadata']
    
    print(f"\nLoaded {len(images_u)} matched triplets")
    print(f"  U plane: {images_u.shape}")
    print(f"  V plane: {images_v.shape}")
    print(f"  X plane: {images_x.shape}")
    print(f"  Metadata: {metadata.shape}")
    
    # Apply ES main track filtering
    is_marley = metadata[:, 1] == 1
    is_main_track = metadata[:, 2] == 1
    is_es_interaction = metadata[:, 3] == 1
    es_main_mask = is_marley & is_main_track & is_es_interaction
    
    print(f"\nApplying filters:")
    print(f"  Before: {len(images_u)} samples")
    print(f"  ES main tracks: {np.sum(es_main_mask)} ({100*np.mean(es_main_mask):.1f}%)")
    
    # Apply energy filter if specified
    if min_energy is not None:
        true_energy = metadata[:, 11]  # Column 11 is true_particle_energy
        energy_mask = true_energy >= min_energy
        combined_mask = es_main_mask & energy_mask
        print(f"  Energy >= {min_energy} MeV: {np.sum(energy_mask)} ({100*np.mean(energy_mask):.1f}%)")
        print(f"  Combined (ES + Energy): {np.sum(combined_mask)} ({100*np.mean(combined_mask):.1f}%)")
    else:
        combined_mask = es_main_mask
        print(f"  No energy filter applied")
    
    # Filter all arrays
    images_u = images_u[combined_mask]
    images_v = images_v[combined_mask]
    images_x = images_x[combined_mask]
    metadata = metadata[combined_mask]
    
    print(f"  After: {len(images_u)} samples")
    
    # Extract direction labels from momentum (columns 7-9)
    momentum = metadata[:, 7:10].astype(np.float32)
    
    # Normalize to get direction
    mom_mag = np.linalg.norm(momentum, axis=1, keepdims=True)
    mom_mag = np.where(mom_mag == 0, 1.0, mom_mag)
    directions = momentum / mom_mag
    
    print(f"\nDirection labels: {directions.shape}")
    print(f"  Mean |px|/|p|: {np.abs(directions[:, 0]).mean():.3f}")
    print(f"  Mean |py|/|p|: {np.abs(directions[:, 1]).mean():.3f}")
    print(f"  Mean |pz|/|p|: {np.abs(directions[:, 2]).mean():.3f}")
    
    # Normalize images to float32
    images_u = images_u.astype(np.float32)
    images_v = images_v.astype(np.float32)
    images_x = images_x.astype(np.float32)
    
    # Add channel dimension if needed
    if len(images_u.shape) == 3:
        images_u = np.expand_dims(images_u, axis=-1)
        images_v = np.expand_dims(images_v, axis=-1)
        images_x = np.expand_dims(images_x, axis=-1)
    
    print(f"\nFinal shapes (with channel):")
    print(f"  U plane: {images_u.shape}")
    print(f"  V plane: {images_v.shape}")
    print(f"  X plane: {images_x.shape}")
    
    # Shuffle if requested
    if shuffle:
        indices = np.random.permutation(len(images_u))
        images_u = images_u[indices]
        images_v = images_v[indices]
        images_x = images_x[indices]
        directions = directions[indices]
        print("\n✓ Data shuffled")
    
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
    
    print(f"\nDataset split:")
    print(f"  Training:   {len(train_u)} samples ({train_fraction*100:.0f}%)")
    print(f"  Validation: {len(val_u)} samples ({val_fraction*100:.0f}%)")
    print(f"  Test:       {len(test_u)} samples ({test_fraction*100:.0f}%)")
    
    print("\n✓ Data loaded and ready for training!")
    
    return (
        (train_u, train_v, train_x, train_y),
        (val_u, val_v, val_x, val_y),
        (test_u, test_v, test_x, test_y)
    )


if __name__ == '__main__':
    # Test the loader
    matched_file = '/eos/user/e/evilla/dune/sn-tps/production_es/three_plane_matched_50k.npz'
    
    (train_u, train_v, train_x, train_y), \
    (val_u, val_v, val_x, val_y), \
    (test_u, test_v, test_x, test_y) = load_matched_three_plane_data(
        matched_file_path=matched_file,
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        shuffle=True,
        random_seed=42
    )
    
    print("\n" + "=" * 70)
    print("TEST SUCCESSFUL")
    print("=" * 70)
