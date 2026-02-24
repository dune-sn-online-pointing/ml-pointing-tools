#!/usr/bin/env python3
"""
Three-Plane Channel Tagging Training Script

Trains a three-plane CNN for ES/CC/NC classification using volume images from U/V/X folders.
"""

import sys
import os
import json
import argparse
from datetime import datetime

# Add shared libs and CT-specific libs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import local modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.three_plane_ct_cnn import build_three_plane_ct_cnn

import general_purpose_libs as gpl
import volume_three_plane_data_loader as data_loader


def parse_args():
    parser = argparse.ArgumentParser(description='Train three-plane CT classifier')
    parser.add_argument('--input_json', '-j', type=str, required=True,
                        help='JSON config file')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("THREE-PLANE CHANNEL TAGGING TRAINING")
    print("=" * 70)
    
    # Load config
    with open(args.input_json, 'r') as f:
        config = json.load(f)
    
    # Extract parameters
    data_dir = config.get('data', {}).get('data_dir')
    if not data_dir:
        raise ValueError("data_dir is required in config['data']")
    
    output_base = config.get('output', {}).get('base_dir', 'training_output/channel_tagging')
    model_name = config.get('model', {}).get('name', 'ct_three_plane_v60')
    
    max_samples = config.get('data', {}).get('max_samples', None)
    train_split = config.get('data', {}).get('train_split', 0.7)
    val_split = config.get('data', {}).get('val_split', 0.15)
    
    batch_size = config.get('training', {}).get('batch_size', 32)
    epochs = config.get('training', {}).get('epochs', 100)
    learning_rate = config.get('training', {}).get('learning_rate', 0.001)
    early_stop_patience = config.get('training', {}).get('early_stopping_patience', 20)
    lr_patience = config.get('training', {}).get('reduce_lr_patience', 10)
    
    n_conv_layers = config.get('model', {}).get('n_conv_layers', 3)
    n_filters = config.get('model', {}).get('n_filters', 64)
    n_dense_units = config.get('model', {}).get('n_dense_units', 256)
    dropout_rate = config.get('model', {}).get('dropout_rate', 0.3)
    input_shape = tuple(config.get('model', {}).get('input_shape', [208, 1242, 1]))
    
    # Create output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_base, f"{model_name}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Output folder: {output_folder}")
    
    # Save config
    with open(os.path.join(output_folder, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load data
    train_data, val_data, test_data = data_loader.load_three_plane_volumes(
        data_dir=data_dir,
        max_samples=max_samples,
        train_frac=train_split,
        val_frac=val_split,
        shuffle=True,
        verbose=True
    )
    
    # Unpack data
    (train_u, train_v, train_x), train_labels, train_meta = train_data
    (val_u, val_v, val_x), val_labels, val_meta = val_data
    (test_u, test_v, test_x), test_labels, test_meta = test_data
    
    # Build model
    print(f"\n{'='*70}")
    print("BUILDING MODEL")
    print(f"{'='*70}")
    
    model = build_three_plane_ct_cnn(
        input_shape=input_shape,
        n_classes=3,
        n_conv_layers=n_conv_layers,
        n_filters=n_filters,
        n_dense_units=n_dense_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    print("Model created:")
    model.summary()
    
    # Train model
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stop_patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=lr_patience,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_folder, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        [train_u, train_v, train_x],
        train_labels,
        validation_data=([val_u, val_v, val_x], val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    gpl.save_history(history, output_folder)
    
    # Evaluate on test set
    print(f"\n{'='*70}")
    print("EVALUATION ON TEST SET")
    print(f"{'='*70}")
    
    test_loss, test_acc = model.evaluate([test_u, test_v, test_x], test_labels, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Predictions
    test_pred_probs = model.predict([test_u, test_v, test_x], verbose=0)
    test_pred = np.argmax(test_pred_probs, axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['ES', 'CC', 'NC'],
                yticklabels=['ES', 'CC', 'NC'])
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_pred, target_names=['ES', 'CC', 'NC']))
    
    # Save results
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'confusion_matrix': cm.tolist(),
        'model_name': model_name,
        'timestamp': timestamp,
        'config': config
    }
    
    gpl.write_results_json(results, output_folder)
    
    print(f"\n{'='*70}")
    print(f"Training complete! Results saved to: {output_folder}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
