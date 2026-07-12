#!/usr/bin/env python3
"""
Channel tagging training with volume images - NO hyperopt, simple fixed parameters.
Loads all ES and CC volume images, trains a simple CNN.
"""

import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("CHANNEL TAGGING TRAINING - VOLUME IMAGES (SIMPLE CNN)")
print("=" * 80)

def parse_args():
    parser = argparse.ArgumentParser(description='Train CT with volume images')
    parser.add_argument('--plane', '-p', type=str, default='X', choices=['U', 'V', 'X'],
                        help='Plane to use')
    parser.add_argument('--max-samples', '-m', type=int, default=50000,
                        help='Maximum samples to load (per class)')
    parser.add_argument('--json', '-j', type=str, required=True,
                        help='JSON config file')
    return parser.parse_args()

def load_config(json_file):
    """Load configuration from JSON file."""
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

def create_simple_cnn(input_shape=(208, 1242, 1), n_filters=32, n_conv_layers=3, 
                      dropout_rate=0.3, dense_units=128, n_classes=2):
    """Create a simple CNN model."""
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape)
    ])
    
    # Convolutional layers
    for i in range(n_conv_layers):
        filters = n_filters * (2 ** i)
        model.add(keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
    
    # Dense layers
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(dense_units, activation='relu'))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(n_classes, activation='softmax'))
    
    return model

def load_volume_data(es_directory, cc_directory, plane='X', max_samples_per_class=25000):
    """Load volume images from ES and CC directories."""
    print(f"\nLoading volume images for plane {plane}...")
    print(f"Maximum {max_samples_per_class} samples per class")
    print(f"ES directory: {es_directory}")
    print(f"CC directory: {cc_directory}")
    
    es_pattern = f'{es_directory}*plane{plane}.npz'
    cc_pattern = f'{cc_directory}*plane{plane}.npz'
    
    es_files = sorted(glob.glob(es_pattern))
    cc_files = sorted(glob.glob(cc_pattern))
    
    print(f"Found {len(es_files)} ES files, {len(cc_files)} CC files")
    
    images_list = []
    labels_list = []
    
    # Load ES samples (label=0)
    es_count = 0
    for f in es_files:
        if es_count >= max_samples_per_class:
            break
        try:
            data = np.load(f, allow_pickle=True)
            imgs = data['images']
            for img in imgs:
                if es_count >= max_samples_per_class:
                    break
                img_array = np.array(img, dtype=np.float32)
                if img_array.shape == (208, 1242):
                    images_list.append(img_array)
                    labels_list.append(0)  # ES
                    es_count += 1
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue
        
        if es_count % 5000 == 0:
            print(f"  Loaded {es_count} ES samples...")
    
    # Load CC samples (label=1)
    cc_count = 0
    for f in cc_files:
        if cc_count >= max_samples_per_class:
            break
        try:
            data = np.load(f, allow_pickle=True)
            imgs = data['images']
            for img in imgs:
                if cc_count >= max_samples_per_class:
                    break
                img_array = np.array(img, dtype=np.float32)
                if img_array.shape == (208, 1242):
                    images_list.append(img_array)
                    labels_list.append(1)  # CC
                    cc_count += 1
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue
        
        if cc_count % 5000 == 0:
            print(f"  Loaded {cc_count} CC samples...")
    
    print(f"\nTotal loaded: {es_count} ES samples, {cc_count} CC samples")
    
    # Convert to arrays
    images = np.array(images_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    
    # Normalize images (per-image normalization)
    print("Normalizing images...")
    for i in range(len(images)):
        img_max = np.max(images[i])
        if img_max > 0:
            images[i] = images[i] / img_max
    
    # Add channel dimension
    images = images[..., np.newaxis]
    
    # Shuffle
    print("Shuffling data...")
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    
    print(f"Final dataset: {images.shape}, labels: {labels.shape}")
    print(f"Label distribution: ES={np.sum(labels==0)}, CC={np.sum(labels==1)}")
    
    return images, labels

def main():
    args = parse_args()
    config = load_config(args.json)
    
    print(f"\nConfiguration: {args.json}")
    print(f"Plane: {args.plane}")
    print(f"Max samples per class: {args.max_samples // 2}")
    
    # Get directories from config
    es_directory = config['data']['es_directory']
    cc_directory = config['data']['cc_directory']
    plane = config['data']['plane']
    
    
    # Check streaming mode
    use_streaming = config.get('data', {}).get('use_streaming', False)
    
    if use_streaming:
        print("\n>>> STREAMING MODE: Loading data in batches <<<")
        # In streaming mode, we load all data but process in smaller chunks during training
        # This reduces peak memory usage during data loading
        print("Note: True streaming (incremental loading) not yet implemented")
        print("Using standard loading for now...")
    
    # Load data
    images, labels = load_volume_data(
        es_directory=es_directory,
        cc_directory=cc_directory,
        plane=plane,
        max_samples_per_class=args.max_samples // 2
    )
    
    # Split data: 75% train, 15% val, 10% test
    n = len(images)
    n_train = int(0.75 * n)
    n_val = int(0.15 * n)
    
    X_train = images[:n_train]
    y_train = labels[:n_train]
    X_val = images[n_train:n_train+n_val]
    y_val = labels[n_train:n_train+n_val]
    X_test = images[n_train+n_val:]
    y_test = labels[n_train+n_val:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(X_train)} ({np.sum(y_train==0)} ES, {np.sum(y_train==1)} CC)")
    print(f"  Val:   {len(X_val)} ({np.sum(y_val==0)} ES, {np.sum(y_val==1)} CC)")
    print(f"  Test:  {len(X_test)} ({np.sum(y_test==0)} ES, {np.sum(y_test==1)} CC)")
    
    # Create datasets
    print("\nCreating TensorFlow datasets...")
    batch_size = config.get('batch_size', 32)
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create model
    print("\nCreating model...")
    model_params = config.get('model_params', {})
    model = create_simple_cnn(
        input_shape=(208, 1242, 1),
        n_filters=model_params.get('n_filters', 32),
        n_conv_layers=model_params.get('n_conv_layers', 3),
        dropout_rate=model_params.get('dropout_rate', 0.3),
        dense_units=model_params.get('dense_units', 128)
    )
    
    learning_rate = model_params.get('learning_rate', 0.001)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.get('model_name', 'ct_volume_simple')
    version = config.get('version', 'unknown')
    output_dir = os.path.join(
        config.get('output', {}).get('base_dir', 'training_output/channel_tagging'),
        version,
        timestamp
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config['timestamp'] = timestamp
    config['plane'] = args.plane
    config['actual_samples'] = {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)}
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.get('early_stopping_patience', 10),
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'checkpoint_epoch_{epoch:03d}.keras'),
            save_freq='epoch',
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'training_history.csv')
        )
    ]
    
    # Train
    print("\nTraining...")
    history = model.fit(
        train_ds,
        epochs=config.get('epochs', 50),
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(
        X_test, y_test, batch_size=batch_size, verbose=0
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")

    # Generate predictions for confusion matrix
    print("\nGenerating predictions and confusion matrix...")
    y_pred_probs = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test
    
    # Save predictions
    # Keep legacy keys (y_true/y_pred/y_prob) and also add the keys expected by
    # the CT analysis tools (true_labels/predictions).
    np.savez(
        os.path.join(output_dir, 'test_predictions.npz'),
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_pred_probs,
        true_labels=y_true,
        predictions=y_pred_probs,
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['ES', 'CC']
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax, annot_kws={'size': 14})
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {config.get("version", "CT")}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate classification report. Pass explicit labels so it stays valid
    # even when a (tiny) test split happens to contain a single class.
    report = classification_report(y_true, y_pred,
                                   labels=list(range(len(class_names))),
                                   target_names=class_names)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    print(f"✓ Confusion matrix saved to: {output_dir}/confusion_matrix.png")
    print(f"✓ Predictions saved to: {output_dir}/test_predictions.npz")
    print(f"\nClassification Report:")
    print(report)
    
    # Save final results
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_loss': float(min(history.history['val_loss'])),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'config': config
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"✓ Results saved to: {output_dir}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
