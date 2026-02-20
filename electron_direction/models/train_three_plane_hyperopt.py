"""
Three-plane electron direction training with Hyperopt optimization.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

import json
import argparse
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from direction_losses import cosine_similarity_loss

try:
    from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    print("WARNING: hyperopt not available")

import matched_three_plane_data_loader as data_loader


def build_three_plane_model(params, input_shape=(128, 16, 1)):
    """Build three-plane CNN with given hyperparameters."""
    # Three input branches
    input_u = layers.Input(shape=input_shape, name='input_u')
    input_v = layers.Input(shape=input_shape, name='input_v')
    input_x = layers.Input(shape=input_shape, name='input_x')
    
    def conv_branch(inputs, name_prefix):
        x = inputs
        # Limit to 4 conv layers max for (128, 16) input
        n_layers = min(params['n_conv_layers'], 4)
        
        for i in range(n_layers):
            x = layers.Conv2D(
                params['n_filters'],
                params['kernel_size'],
                padding='same',
                activation='relu',
                name=f'{name_prefix}_conv{i}'
            )(x)
            # Only pool if we have enough spatial dimensions left
            current_h = input_shape[0] // (2 ** i)
            current_w = input_shape[1] // (2 ** i)
            if current_h >= 4 and current_w >= 2:
                x = layers.MaxPooling2D(2, name=f'{name_prefix}_pool{i}')(x)
        x = layers.Flatten(name=f'{name_prefix}_flatten')(x)
        return x
    
    # Process each plane
    feat_u = conv_branch(input_u, 'u')
    feat_v = conv_branch(input_v, 'v')
    feat_x = conv_branch(input_x, 'x')
    
    # Concatenate features
    combined = layers.Concatenate(name='concat')([feat_u, feat_v, feat_x])
    
    # Dense layers
    x = combined
    for i in range(params['n_dense_layers']):
        units = params['n_dense_units'] // (2 ** i)
        x = layers.Dense(units, activation='relu', name=f'dense{i}')(x)
        x = layers.Dropout(params['dropout_rate'], name=f'dropout{i}')(x)
    
    # Output layer
    outputs = layers.Dense(3, activation='linear', name='direction')(x)
    
    model = keras.Model(inputs=[input_u, input_v, input_x], outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss=cosine_similarity_loss,
        metrics=['mae']
    )
    
    return model


def objective(params, train_data, val_data, config):
    """Objective function for hyperopt."""
    print("\n" + "=" * 70)
    print(f"TRIAL: {params}")
    print("=" * 70)
    
    # Build model
    model = build_three_plane_model(params)
    
    # Train
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=config['early_stopping_patience'],
            restore_best_weights=True,
            monitor='val_loss'
        )
    ]
    
    train_u, train_v, train_x, train_y = train_data
    val_u, val_v, val_x, val_y = val_data
    
    history = model.fit(
        [train_u, train_v, train_x], train_y,
        validation_data=([val_u, val_v, val_x], val_y),
        epochs=config['epochs_per_trial'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=0
    )
    
    # Get best validation loss
    best_val_loss = min(history.history['val_loss'])
    
    print(f"Best val_loss: {best_val_loss:.4f}")
    
    return {'loss': best_val_loss, 'status': STATUS_OK, 'model': model}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.input_json, 'r') as f:
        config = json.load(f)
    
    if not HYPEROPT_AVAILABLE:
        print("ERROR: hyperopt not available!")
        return
    
    print("=" * 70)
    print("THREE-PLANE HYPEROPT OPTIMIZATION")
    print("=" * 70)
    print(f"Max trials: {config['hyperopt_parameters']['max_trials']}")
    print(f"Epochs per trial: {config['hyperopt_parameters']['epochs_per_trial']}")
    
    # Load data
    data_file = config.get('data_file') or config.get('matched_data_file')
    (train_u, train_v, train_x, train_y), \
    (val_u, val_v, val_x, val_y), \
    (test_u, test_v, test_x, test_y) = data_loader.load_matched_three_plane_data(
        matched_file_path=data_file,
        train_fraction=config.get('train_fraction', 0.7),
        val_fraction=config.get('val_fraction', 0.15),
        shuffle=True,
        random_seed=42,
        min_energy=config.get('min_energy', None)
    )
    
    train_data = (train_u, train_v, train_x, train_y)
    val_data = (val_u, val_v, val_x, val_y)
    
    # Define search space
    search_space = config['hyperopt_parameters']['search_space']
    space = {
        'n_conv_layers': hp.choice('n_conv_layers', search_space['n_conv_layers']),
        'n_filters': hp.choice('n_filters', search_space['n_filters']),
        'kernel_size': hp.choice('kernel_size', search_space['kernel_size']),
        'n_dense_layers': hp.choice('n_dense_layers', search_space['n_dense_layers']),
        'n_dense_units': hp.choice('n_dense_units', search_space['n_dense_units']),
        'learning_rate': hp.choice('learning_rate', search_space['learning_rate']),
        'dropout_rate': hp.choice('dropout_rate', search_space['dropout_rate'])
    }
    
    # Run hyperopt
    trials = Trials()
    
    best = fmin(
        fn=lambda params: objective(params, train_data, val_data, config['hyperopt_parameters']),
        space=space,
        algo=tpe.suggest,
        max_evals=config['hyperopt_parameters']['max_trials'],
        trials=trials
    )
    
    print("\n" + "=" * 70)
    print("HYPEROPT OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Best parameters: {best}")
    
    # Convert best indices to actual values
    best_params = {
        'n_conv_layers': search_space['n_conv_layers'][best['n_conv_layers']],
        'n_filters': search_space['n_filters'][best['n_filters']],
        'kernel_size': search_space['kernel_size'][best['kernel_size']],
        'n_dense_layers': search_space['n_dense_layers'][best['n_dense_layers']],
        'n_dense_units': search_space['n_dense_units'][best['n_dense_units']],
        'learning_rate': search_space['learning_rate'][best['learning_rate']],
        'dropout_rate': search_space['dropout_rate'][best['dropout_rate']]
    }
    
    print(f"Best parameters (actual values): {best_params}")
    
    # Train final model with best parameters
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
    print("=" * 70)
    
    final_model = build_three_plane_model(best_params)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(
        config['output_folder'], 
        'electron_direction',
        f'three_plane_hyperopt_{timestamp}'
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=15,
            restore_best_weights=True,
            monitor='val_loss'
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=5,
            factor=0.5,
            monitor='val_loss'
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'checkpoints', 'best_model.keras'),
            save_best_only=True,
            monitor='val_loss'
        )
    ]
    
    history = final_model.fit(
        [train_u, train_v, train_x], train_y,
        validation_data=([val_u, val_v, val_x], val_y),
        epochs=100,
        batch_size=config['hyperopt_parameters']['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Test
    print("\n" + "=" * 70)
    print("TESTING FINAL MODEL")
    print("=" * 70)
    
    predictions = final_model.predict([test_u, test_v, test_x], verbose=0)
    pred_norm = predictions / (np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8)
    
    dot_products = np.sum(pred_norm * test_y, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angular_errors = np.arccos(dot_products) * 180.0 / np.pi
    
    print(f"\nAngular Error Statistics:")
    print(f"  Mean:   {angular_errors.mean():.2f}°")
    print(f"  Median: {np.median(angular_errors):.2f}°")
    print(f"  Std:    {angular_errors.std():.2f}°")
    print(f"  25th:   {np.percentile(angular_errors, 25):.2f}°")
    print(f"  75th:   {np.percentile(angular_errors, 75):.2f}°")
    
    # Save results
    with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    np.savez(
        os.path.join(output_dir, 'test_results.npz'),
        predictions=predictions,
        true_directions=test_y,
        angular_errors=angular_errors
    )
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"✓ Best angular error: {angular_errors.mean():.2f}° (mean), {np.median(angular_errors):.2f}° (median)")


if __name__ == '__main__':
    main()
