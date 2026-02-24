"""
Training script for three-plane electron direction CNN
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import json
import argparse
import numpy as np
from datetime import datetime

# Import custom modules
import three_plane_data_loader as data_loader
from three_plane_cnn import build_three_plane_cnn, train_three_plane_model


def main():
    parser = argparse.ArgumentParser(description='Train three-plane electron direction CNN')
    parser.add_argument('--input_json', type=str, required=True, help='Input JSON configuration')
    parser.add_argument('--output_folder', type=str, help='Override output folder')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.input_json, 'r') as f:
        config = json.load(f)
    
    print("=" * 70)
    print("THREE-PLANE ELECTRON DIRECTION TRAINING")
    print("=" * 70)
    print(f"Config: {args.input_json}")
    print()
    
    # Get parameters
    data_dirs = config['data_directories']
    output_folder = args.output_folder if args.output_folder else config['output_folder']
    
    dataset_params = config.get('dataset_parameters', {})
    model_params = config.get('model_parameters', {})
    
    # Load data
    print("STEP 1: LOADING DATA")
    print("-" * 70)
    
    (train_u, train_v, train_x, train_y), \
    (val_u, val_v, val_x, val_y), \
    (test_u, test_v, test_x, test_y) = data_loader.load_three_plane_data(
        data_directories=data_dirs,
        max_samples=dataset_params.get('max_samples', None),
        train_fraction=dataset_params.get('train_fraction', 0.7),
        val_fraction=dataset_params.get('val_fraction', 0.15),
        test_fraction=dataset_params.get('test_fraction', 0.15),
        shuffle=dataset_params.get('shuffle_data', True),
        filter_main_tracks=config.get('filter_main_tracks', True),
        random_seed=dataset_params.get('random_seed', 42)
    )
    
    # Build model
    print("\nSTEP 2: BUILDING MODEL")
    print("-" * 70)
    
    model = build_three_plane_cnn(
        input_shape=tuple(model_params.get('input_shape', [128, 16, 1])),
        output_dim=model_params.get('output_dim', 3),
        n_conv_layers=model_params.get('n_conv_layers', 2),
        n_filters=model_params.get('n_filters', 64),
        kernel_size=model_params.get('kernel_size', 3),
        n_dense_layers=model_params.get('n_dense_layers', 2),
        n_dense_units=model_params.get('n_dense_units', 128),
        learning_rate=model_params.get('learning_rate', 0.001),
        decay_rate=model_params.get('decay_rate', 0.95)
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(output_folder, 'electron_direction', 'three_plane_cnn', timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nOutput directory: {save_dir}")
    
    # Save configuration
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train model
    print("\nSTEP 3: TRAINING MODEL")
    print("-" * 70)
    
    # Prepare data in correct format for multi-input model
    train_data = ([train_u, train_v, train_x], train_y)
    val_data = ([val_u, val_v, val_x], val_y)
    
    history = train_three_plane_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        epochs=model_params.get('epochs', 50),
        batch_size=dataset_params.get('batch_size', 32),
        output_folder=save_dir,
        early_stopping_patience=model_params.get('early_stopping_patience', 10)
    )
    
    # Save model
    print("\nSTEP 4: SAVING MODEL")
    print("-" * 70)
    
    model_path = os.path.join(save_dir, 'three_plane_cnn.h5')
    model.save(model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        json.dump(history_dict, f, indent=2)
    print(f"✓ History saved to: {history_path}")
    
    # Evaluate on test set
    print("\nSTEP 5: EVALUATING MODEL")
    print("-" * 70)
    
    test_loss, test_mae = model.evaluate([test_u, test_v, test_x], test_y, verbose=1)
    print(f"\nTest Results:")
    print(f"  Loss (MSE): {test_loss:.6f}")
    print(f"  MAE: {test_mae:.6f}")
    
    # Save test results
    test_results = {
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'n_test_samples': len(test_u)
    }
    
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    predictions = model.predict([test_u, test_v, test_x], verbose=1)
    
    # Save predictions
    np.savez(
        os.path.join(save_dir, 'test_predictions.npz'),
        predictions=predictions,
        true_values=test_y
    )
    print(f"✓ Predictions saved")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Results saved in: {save_dir}")
    print(f"Model file: {model_path}")
    print()


if __name__ == '__main__':
    main()
