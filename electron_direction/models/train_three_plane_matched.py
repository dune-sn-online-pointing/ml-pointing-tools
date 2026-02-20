"""
Training script for three-plane electron direction CNN using pre-matched data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

import json
import argparse
import numpy as np
from datetime import datetime

# Import custom modules
import matched_three_plane_data_loader as data_loader
from three_plane_cnn import build_three_plane_cnn, train_three_plane_model


def main():
    parser = argparse.ArgumentParser(description='Train three-plane electron direction CNN')
    parser.add_argument('--input_json', type=str, required=True,
                       help='Input JSON configuration')
    parser.add_argument('--output_folder', type=str,
                       help='Override output folder')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.input_json, 'r') as f:
        config = json.load(f)
    
    print("=" * 70)
    print("THREE-PLANE ELECTRON DIRECTION TRAINING (PRE-MATCHED DATA)")
    print("=" * 70)
    print(f"Config: {args.input_json}")
    print()
    
    # Get parameters
    matched_file = config['matched_data_file']
    output_folder = args.output_folder if args.output_folder else config['output_folder']
    
    dataset_params = config.get('dataset_parameters', {})
    model_params = config.get('model_parameters', {})
    training_params = config.get('training_parameters', {})
    
    # Load data
    print("STEP 1: LOADING DATA")
    print("-" * 70)
    
    (train_u, train_v, train_x, train_y), \
    (val_u, val_v, val_x, val_y), \
    (test_u, test_v, test_x, test_y) = data_loader.load_matched_three_plane_data(
        matched_file_path=matched_file,
        train_fraction=dataset_params.get('train_fraction', 0.7),
        val_fraction=dataset_params.get('val_fraction', 0.15),
        test_fraction=dataset_params.get('test_fraction', 0.15),
        shuffle=dataset_params.get('shuffle_data', True),
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
    save_dir = os.path.join(output_folder, 'electron_direction', 'three_plane_matched', timestamp)
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
        epochs=training_params.get('epochs', 50),
        batch_size=training_params.get('batch_size', 32),
        output_folder=save_dir
    )
    
    # Evaluate on test set
    print("\nSTEP 4: EVALUATING ON TEST SET")
    print("-" * 70)
    
    test_results = model.evaluate([test_u, test_v, test_x], test_y, verbose=1)
    test_loss = test_results[0] if isinstance(test_results, list) else test_results
    print(f"\nTest Loss (MSE): {test_loss:.6f}")
    
    # Calculate angular error
    predictions = model.predict([test_u, test_v, test_x])
    
    # Normalize predictions
    pred_mag = np.linalg.norm(predictions, axis=1, keepdims=True)
    pred_mag = np.where(pred_mag == 0, 1.0, pred_mag)
    predictions_normalized = predictions / pred_mag
    
    # Calculate angular error (in degrees)
    dot_products = np.sum(predictions_normalized * test_y, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angular_errors = np.arccos(dot_products) * 180.0 / np.pi
    
    mean_angular_error = np.mean(angular_errors)
    median_angular_error = np.median(angular_errors)
    
    print(f"\nAngular Error Statistics:")
    print(f"  Mean:   {mean_angular_error:.2f}°")
    print(f"  Median: {median_angular_error:.2f}°")
    print(f"  Std:    {np.std(angular_errors):.2f}°")
    print(f"  Min:    {np.min(angular_errors):.2f}°")
    print(f"  Max:    {np.max(angular_errors):.2f}°")
    
    # Save results
    results = {
        'test_loss': float(test_loss),
        'mean_angular_error': float(mean_angular_error),
        'median_angular_error': float(median_angular_error),
        'std_angular_error': float(np.std(angular_errors)),
        'min_angular_error': float(np.min(angular_errors)),
        'max_angular_error': float(np.max(angular_errors))
    }
    
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"✓ Results saved to: {save_dir}")
    print(f"✓ Test angular error: {mean_angular_error:.2f}° (mean), {median_angular_error:.2f}° (median)")


if __name__ == '__main__':
    main()
