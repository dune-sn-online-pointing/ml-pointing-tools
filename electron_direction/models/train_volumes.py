#!/usr/bin/env python3
"""
Three-plane electron direction training using 1m x 1m volume images.
Adapts the 2D pentagon architecture for larger volume inputs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import custom modules
from direction_losses import cosine_similarity_loss
from volume_ed_data_loader import load_ed_volumes


def build_three_plane_volume_cnn(
    input_shape=(208, 1242, 1),  # Actual volume size from data
    output_dim=3,
    n_conv_layers=5,  # More layers for large input
    n_filters=32,
    kernel_size=3,
    n_dense_layers=2,
    n_dense_units=256,
    learning_rate=0.001
):
    """
    Build three-plane CNN for electron direction using volumes.
    
    Volumes are (208, 1242) pixels representing ~1m x 1m physical space.
    Uses progressive downsampling to handle the large input size.
    
    Args:
        input_shape: Shape of volume input (channels, time_bins, 1)
        output_dim: Output dimension (3 for x, y, z direction)
        n_conv_layers: Number of convolutional layers per branch
        n_filters: Number of filters in first conv layer
        kernel_size: Size of convolutional kernels
        n_dense_layers: Number of dense layers after concatenation
        n_dense_units: Number of units in dense layers
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    
    def create_volume_branch(name_prefix):
        """Create CNN branch for one plane's volume"""
        branch_input = keras.Input(shape=input_shape, name=f'{name_prefix}_input')
        x = branch_input
        
        # Progressive downsampling with increasing filters
        # Input: (208, 1242, 1)
        for i in range(n_conv_layers):
            filters = n_filters * (2 ** i)  # 32, 64, 128, 256, 512
            x = layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'{name_prefix}_conv_{i+1}'
            )(x)
            x = layers.MaxPooling2D(
                pool_size=(2, 2),
                name=f'{name_prefix}_pool_{i+1}'
            )(x)
            x = layers.BatchNormalization(
                name=f'{name_prefix}_bn_{i+1}'
            )(x)
            # After 5 poolings: 208->104->52->26->13->6, 1242->621->310->155->77->38
        
        # Global average pooling to reduce to vector
        x = layers.GlobalAveragePooling2D(name=f'{name_prefix}_gap')(x)
        
        return branch_input, x
    
    # Create three branches
    input_u, features_u = create_volume_branch('volume_u')
    input_v, features_v = create_volume_branch('volume_v')
    input_x, features_x = create_volume_branch('volume_x')
    
    # Concatenate features from all planes
    concatenated = layers.Concatenate(name='concatenate')([features_u, features_v, features_x])
    
    # Dense layers
    x = concatenated
    for i in range(n_dense_layers):
        x = layers.Dense(
            n_dense_units,
            activation='relu',
            name=f'dense_{i+1}'
        )(x)
        x = layers.Dropout(0.3, name=f'dropout_{i+1}')(x)
    
    # Output layer (direction vector)
    output = layers.Dense(output_dim, activation='linear', name='output')(x)
    
    # Normalize to unit vector
    output_normalized = layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=-1),
        name='normalize'
    )(output)
    
    # Create model
    model = keras.Model(
        inputs=[input_u, input_v, input_x],
        outputs=output_normalized,
        name='three_plane_volume_cnn'
    )
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=cosine_similarity_loss,
        metrics=['mae']
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train 3-plane electron direction CNN using volumes')
    parser.add_argument('-j', '--json', type=str, required=True,
                       help='JSON configuration file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.json, 'r') as f:
        config = json.load(f)
    
    print("=" * 70)
    print("THREE-PLANE ELECTRON DIRECTION TRAINING (VOLUMES)")
    print("=" * 70)
    print(f"Config: {args.json}")
    print(f"Version: {config['model'].get('name', 'unknown')}")
    print()
    
    # Extract config parameters
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    output_config = config['output']
    
    # Data parameters
    data_dir = data_config['data_directory']
    train_split = data_config.get('train_split', 0.8)
    max_samples = data_config.get('max_samples', None)
    max_files = data_config.get('max_files', None)
    shuffle = data_config.get('shuffle', True)
    
    # Model parameters
    input_shape = tuple(model_config.get('input_shape', (208, 1242, 1)))
    output_dim = model_config['output_dim']
    
    # Training parameters
    epochs = training_config.get('epochs', 100)
    batch_size = training_config.get('batch_size', 8)  # Small batch for large inputs
    learning_rate = training_config.get('learning_rate', 1e-3)
    
    # Architecture hyperparameters
    n_conv_layers = model_config.get('n_conv_layers', 5)
    n_filters = model_config.get('n_filters', 32)
    kernel_size = model_config.get('kernel_size', 3)
    n_dense_layers = model_config.get('n_dense_layers', 2)
    n_dense_units = model_config.get('n_dense_units', 256)
    
    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print()
    
    # Load data
    images_u, images_v, images_x, directions, energies, metadata = load_ed_volumes(
        data_dir=data_dir,
        max_samples=max_samples,
        max_files=max_files,
        shuffle=shuffle,
        verbose=True
    )
    
    print(f"\nTotal loaded: {len(directions)} samples")
    
    # Train/val split
    n_train = int(len(directions) * train_split)
    
    train_x = [images_u[:n_train], images_v[:n_train], images_x[:n_train]]
    train_y = directions[:n_train]
    val_x = [images_u[n_train:], images_v[n_train:], images_x[n_train:]]
    val_y = directions[n_train:]
    val_energies = energies[n_train:]
    val_metadata = metadata[n_train:]
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_y)} samples")
    print(f"  Val:   {len(val_y)} samples")
    
    # Build model
    print("\n" + "=" * 70)
    print("STEP 2: BUILDING MODEL")
    print("=" * 70)
    
    model = build_three_plane_volume_cnn(
        input_shape=input_shape,
        output_dim=output_dim,
        n_conv_layers=n_conv_layers,
        n_filters=n_filters,
        kernel_size=kernel_size,
        n_dense_layers=n_dense_layers,
        n_dense_units=n_dense_units,
        learning_rate=learning_rate
    )
    
    model.summary()
    
    # Setup output directory
    output_dir = Path(output_config['output_folder'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING")
    print("=" * 70)
    
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(output_dir / 'final_model.keras')
    print(f"\nModel saved to {output_dir / 'final_model.keras'}")
    
    # Generate validation predictions
    print("\n" + "=" * 70)
    print("STEP 4: VALIDATION PREDICTIONS")
    print("=" * 70)
    
    val_predictions = model.predict(val_x, batch_size=batch_size, verbose=1)
    
    # Save predictions
    np.savez(
        output_dir / 'val_predictions.npz',
        predictions=val_predictions,
        true_directions=val_y,
        energies=val_energies,
        metadata=val_metadata
    )
    
    # Compute metrics
    cosine_similarities = np.sum(val_predictions * val_y, axis=1)
    angles_deg = np.rad2deg(np.arccos(np.clip(cosine_similarities, -1, 1)))
    
    print(f"\nValidation Results:")
    print(f"  Median angular error: {np.median(angles_deg):.2f}°")
    print(f"  Mean angular error: {np.mean(angles_deg):.2f}°")
    print(f"  68% quantile: {np.percentile(angles_deg, 68):.2f}°")
    print(f"  95% quantile: {np.percentile(angles_deg, 95):.2f}°")
    
    # Save results
    results = {
        'config': config,
        'median_angle_error': float(np.median(angles_deg)),
        'mean_angle_error': float(np.mean(angles_deg)),
        'quantile_68': float(np.percentile(angles_deg, 68)),
        'quantile_95': float(np.percentile(angles_deg, 95)),
        'n_train': len(train_y),
        'n_val': len(val_y)
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'results.json'}")
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
