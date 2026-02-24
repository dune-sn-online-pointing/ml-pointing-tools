#!/usr/bin/env python3
"""
Three-Plane Channel Tagging Training Script (V2 - Updated for ES+CC)
Trains a three-plane CNN for ES/CC classification using volume images from both directories.
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
import volume_three_plane_data_loader_v2 as data_loader

def parse_args():
    parser = argparse.ArgumentParser(description='Train three-plane CT classifier')
    parser.add_argument('--input_json', '-j', type=str, required=True,
                        help='JSON config file')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("THREE-PLANE CHANNEL TAGGING TRAINING (V2)")
    print("=" * 70)
    
    # Load config
    with open(args.input_json, 'r') as f:
        config = json.load(f)
    
    # Extract parameters
    es_data_dir = config.get('data', {}).get('es_data_dir')
    cc_data_dir = config.get('data', {}).get('cc_data_dir')
    
    if not es_data_dir or not cc_data_dir:
        raise ValueError("Both es_data_dir and cc_data_dir are required in config['data']")
    
    output_base = config.get('output', {}).get('base_dir', 'training_output/channel_tagging')
    model_name = config.get('model', {}).get('name', 'ct_three_plane_v2')
    
    max_samples_per_class = config.get('data', {}).get('max_samples_per_class', None)
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
    
    # Load data from BOTH ES and CC directories
    train_data, val_data, test_data = data_loader.load_three_plane_volumes_from_two_dirs(
        es_data_dir=es_data_dir,
        cc_data_dir=cc_data_dir,
        max_samples_per_class=max_samples_per_class,
        train_frac=train_split,
        val_frac=val_split,
        shuffle=True,
        verbose=True
    )
    
    # Unpack data
    (train_u, train_v, train_x), train_labels, train_meta = train_data
    (val_u, val_v, val_x), val_labels, val_meta = val_data
    (test_u, test_v, test_x), test_labels, test_meta = test_data
    
    # Build model (n_classes=2 for ES vs CC binary classification)
    print(f"\n{'='*70}")
    print("BUILDING MODEL")
    print(f"{'='*70}")
    
    model = build_three_plane_ct_cnn(
        input_shape=input_shape,
        n_classes=2,  # ES vs CC only
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
    
    # Save history
    np.save(os.path.join(output_folder, 'history.npy'), history.history)
    
    # Evaluate on test set
    print(f"\n{'='*70}")
    print("EVALUATION")
    print(f"{'='*70}")
    
    test_loss, test_acc = model.evaluate([test_u, test_v, test_x], test_labels, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Predictions
    y_pred_proba = model.predict([test_u, test_v, test_x], verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, y_pred, target_names=['ES', 'CC']))
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training History - Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training History - Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training_history.png'), dpi=300, bbox_inches='tight')
    print(f"\nTraining history saved to {output_folder}/training_history.png")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['ES', 'CC'], yticklabels=['ES', 'CC'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_folder}/confusion_matrix.png")
    
    # Save summary
    summary = {
        'model_name': model_name,
        'timestamp': timestamp,
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'n_train': len(train_labels),
        'n_val': len(val_labels),
        'n_test': len(test_labels),
        'config': config
    }
    
    with open(os.path.join(output_folder, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Output folder: {output_folder}")


if __name__ == '__main__':
    main()
