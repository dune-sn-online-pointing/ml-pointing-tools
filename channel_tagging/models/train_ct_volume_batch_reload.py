#!/usr/bin/env python3
"""
Channel tagging training with volume images using batch data reloading.
Reloads a fresh batch of data every N epochs to reduce memory pressure.
"""

import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import general_purpose_libs as gpl

print("=" * 80)
print("CHANNEL TAGGING TRAINING - VOLUME IMAGES (BATCH RELOAD)")
print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description='Train CT with volume images (batch reload)')
    parser.add_argument('--plane', '-p', type=str, default='X', choices=['U', 'V', 'X'],
                        help='Plane to use')
    parser.add_argument('--max-samples', '-m', type=int, default=50000,
                        help='Maximum samples to load per batch (per class)')
    parser.add_argument('--json', '-j', type=str, required=True,
                        help='JSON config file')
    parser.add_argument('--reload-epochs', '-r', type=int, default=5,
                        help='Reload data every N epochs')
    parser.add_argument('--test-local', action='store_true',
                        help='Test locally with tiny dataset')
    return parser.parse_args()


def load_config(json_file):
    """Load configuration from JSON file."""
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config


def create_deep_cnn(input_shape=(208, 1242, 1), filter_list=[28, 28, 29, 47, 48, 48], 
                    kernel_sizes=[3, 3, 3, 3, 3, 3], dropout_rate=0.3, 
                    dense_units=[96, 32], n_classes=2):
    """Create a deep CNN model with custom architecture.

    Default parameters are the stable CT baseline used around the v57-era
    production studies (same shape used in v52/v63 configs) and are the
    reference defaults for v78 retraining unless overridden by JSON.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape)
    ])
    
    # Convolutional blocks
    for i, (filters, kernel_size) in enumerate(zip(filter_list, kernel_sizes)):
        model.add(keras.layers.Conv2D(filters, (kernel_size, kernel_size), 
                                     activation='relu', padding='same'))
        if i < len(filter_list) - 1:  # Don't pool after last conv
            model.add(keras.layers.MaxPooling2D((2, 2)))
    
    # Dense layers
    model.add(keras.layers.GlobalAveragePooling2D())
    for units in dense_units:
        model.add(keras.layers.Dense(units, activation='relu'))
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(n_classes, activation='softmax'))
    
    return model


def load_volume_batch(es_directory, cc_directory, plane='X', 
                      max_samples_per_class=25000, seed=None):
    """
    Load a batch of volume images from ES and CC directories.
    Uses random sampling with optional seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"\nLoading batch for plane {plane} (seed={seed})...")
    print(f"Maximum {max_samples_per_class} samples per class")
    
    def discover_plane_files(base_directory, plane_name):
        patterns = [
            f'{base_directory}/*plane{plane_name}.npz',
            f'{base_directory}/{plane_name}/*plane{plane_name}.npz',
            f'{base_directory}/{plane_name}/*.npz',
            f'{base_directory}/*.npz',
        ]

        for pattern in patterns:
            files = sorted(glob.glob(pattern))
            if files:
                return files, pattern

        return [], patterns[0]

    es_files, es_pattern = discover_plane_files(es_directory, plane)
    cc_files, cc_pattern = discover_plane_files(cc_directory, plane)
    
    if not es_files or not cc_files:
        raise ValueError(f"No NPZ files found! ES: {es_pattern}, CC: {cc_pattern}")
    
    print(f"Found {len(es_files)} ES files, {len(cc_files)} CC files")
    
    # Shuffle files for randomness
    np.random.shuffle(es_files)
    np.random.shuffle(cc_files)
    
    images_list = []
    labels_list = []
    energies_list = []
    
    # Load ES samples (label=0)
    es_count = 0
    for f in es_files:
        if es_count >= max_samples_per_class:
            break
        try:
            data = np.load(f, allow_pickle=True)
            imgs = data['images']
            metadata = data['metadata'] if 'metadata' in data else None
            
            # Randomly sample from this file
            indices = np.arange(len(imgs))
            np.random.shuffle(indices)
            
            for idx in indices:
                if es_count >= max_samples_per_class:
                    break
                img = imgs[idx]
                img_array = np.array(img, dtype=np.float32)
                if img_array.shape == (208, 1242):
                    # Normalize
                    img_max = np.max(img_array)
                    if img_max > 0:
                        img_array = img_array / img_max
                    images_list.append(img_array)
                    labels_list.append(0)  # ES
                    if metadata is not None and idx < len(metadata) and isinstance(metadata[idx], dict):
                        energies_list.append(float(metadata[idx].get('particle_energy', np.nan)))
                    else:
                        energies_list.append(np.nan)
                    es_count += 1
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue
        
        if es_count % 5000 == 0 and es_count > 0:
            print(f"  Loaded {es_count} ES samples...")
    
    # Load CC samples (label=1)
    cc_count = 0
    for f in cc_files:
        if cc_count >= max_samples_per_class:
            break
        try:
            data = np.load(f, allow_pickle=True)
            imgs = data['images']
            metadata = data['metadata'] if 'metadata' in data else None
            
            # Randomly sample from this file
            indices = np.arange(len(imgs))
            np.random.shuffle(indices)
            
            for idx in indices:
                if cc_count >= max_samples_per_class:
                    break
                img = imgs[idx]
                img_array = np.array(img, dtype=np.float32)
                if img_array.shape == (208, 1242):
                    # Normalize
                    img_max = np.max(img_array)
                    if img_max > 0:
                        img_array = img_array / img_max
                    images_list.append(img_array)
                    labels_list.append(1)  # CC
                    if metadata is not None and idx < len(metadata) and isinstance(metadata[idx], dict):
                        energies_list.append(float(metadata[idx].get('particle_energy', np.nan)))
                    else:
                        energies_list.append(np.nan)
                    cc_count += 1
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue
        
        if cc_count % 5000 == 0 and cc_count > 0:
            print(f"  Loaded {cc_count} CC samples...")
    
    print(f"Total loaded: {es_count} ES samples, {cc_count} CC samples")
    
    # Convert to arrays
    images = np.array(images_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    energies = np.array(energies_list, dtype=np.float32)
    
    # Add channel dimension
    images = np.expand_dims(images, axis=-1)
    
    # Shuffle the combined dataset
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]
    energies = energies[indices]
    
    print(f"Final batch shape: {images.shape}, labels: {labels.shape}")
    return images, labels, energies


def split_data(images, labels, energies, train_frac=0.7, val_frac=0.15):
    """Split data into train/val/test sets."""
    n = len(images)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    train_images = images[:n_train]
    train_labels = labels[:n_train]
    train_energies = energies[:n_train]
    
    val_images = images[n_train:n_train + n_val]
    val_labels = labels[n_train:n_train + n_val]
    val_energies = energies[n_train:n_train + n_val]
    
    test_images = images[n_train + n_val:]
    test_labels = labels[n_train + n_val:]
    test_energies = energies[n_train + n_val:]
    
    print(f"Split: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")
    return (train_images, train_labels, train_energies), (val_images, val_labels, val_energies), (test_images, test_labels, test_energies)


class BatchReloadCallback(keras.callbacks.Callback):
    """Callback to reload data every N epochs."""
    
    def __init__(self, reload_every_n_epochs, data_loader_fn, split_fn, 
                 batch_size=32, verbose=1):
        super().__init__()
        self.reload_every_n_epochs = reload_every_n_epochs
        self.data_loader_fn = data_loader_fn
        self.split_fn = split_fn
        self.batch_size = batch_size
        self.verbose = verbose
        self.epoch_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        
        # Check if it's time to reload
        if self.epoch_count % self.reload_every_n_epochs == 0:
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"🔄 Reloading data at epoch {self.epoch_count}...")
                print(f"{'='*70}")
            
            try:
                # Clear old data from memory
                import gc
                gc.collect()
                tf.keras.backend.clear_session()
                
                # Load new batch with different seed
                seed = self.epoch_count * 42  # Use epoch-based seed
                images, labels = self.data_loader_fn(seed=seed)
                
                # Split into train/val/test
                train_data, val_data, test_data = self.split_fn(images, labels)
                train_images, train_labels = train_data
                val_images, val_labels = val_data
                
                # Update model's training data by reconstructing the dataset
                # We'll store this in the model for the next epoch
                self.model._train_data = (train_images, train_labels)
                self.model._val_data = (val_images, val_labels)
                self.model._test_data = test_data
                
                if self.verbose:
                    print(f"✓ Data reloaded: {len(train_images)} train, {len(val_images)} val samples")
                    print(f"{'='*70}\n")
                    
            except Exception as e:
                print(f"⚠ Warning: Failed to reload data: {e}")
                print("Continuing with current data...")


def train_with_batch_reload(model, initial_train, initial_val, test_data,
                            data_loader_fn, split_fn, epochs=50, 
                            batch_size=32, reload_every_n_epochs=5,
                            output_folder=None, learning_rate=0.001):
    """
    Train model with periodic data reloading.
    
    Strategy: Train for reload_every_n_epochs at a time, then manually reload data.
    """
    print("\n" + "="*70)
    print("TRAINING WITH BATCH RELOAD")
    print("="*70)
    print(f"Total epochs: {epochs}")
    print(f"Reload every: {reload_every_n_epochs} epochs")
    print(f"Batch size: {batch_size}")
    print("="*70 + "\n")
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Prepare callbacks
    callbacks = []
    
    if output_folder:
        checkpoint_path = os.path.join(output_folder, 'best_model.keras')
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )
    
# DISABLED:     callbacks.append(
# DISABLED:         keras.callbacks.ReduceLROnPlateau(
# DISABLED:             monitor='val_loss',
# DISABLED:             factor=0.5,
# DISABLED:             patience=3,
# DISABLED:             verbose=1,
# DISABLED:             min_lr=1e-6
# DISABLED:         )
# DISABLED:     )
# DISABLED:     
    # Store data in model for callback access
    train_images, train_labels, train_energies = initial_train
    val_images, val_labels, val_energies = initial_val
    
    # Training loop with manual reloading
    history_all = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    num_reload_cycles = (epochs + reload_every_n_epochs - 1) // reload_every_n_epochs
    
    for cycle in range(num_reload_cycles):
        cycle_start_epoch = cycle * reload_every_n_epochs
        cycle_epochs = min(reload_every_n_epochs, epochs - cycle_start_epoch)
        
        print(f"\n{'='*70}")
        print(f"TRAINING CYCLE {cycle + 1}/{num_reload_cycles}")
        print(f"Epochs {cycle_start_epoch + 1} to {cycle_start_epoch + cycle_epochs}")
        print(f"{'='*70}\n")
        
        # Train for this cycle
        history = model.fit(
            train_images, train_labels,
            validation_data=(val_images, val_labels),
            epochs=cycle_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Accumulate history
        for key in history.history:
            if key in history_all:
                history_all[key].extend(history.history[key])
        
        # Reload data for next cycle (unless this is the last cycle)
        if cycle < num_reload_cycles - 1:
            print(f"\n{'='*70}")
            print(f"🔄 RELOADING DATA FOR NEXT CYCLE...")
            print(f"{'='*70}")
            
            # Clear memory
            import gc
            del train_images, train_labels, train_energies, val_images, val_labels, val_energies
            gc.collect()
            
            # Load new batch
            seed = (cycle + 1) * 42
            images, labels, energies = data_loader_fn(seed=seed)
            
            # Split
            train_data, val_data, _ = split_fn(images, labels, energies)
            train_images, train_labels, train_energies = train_data
            val_images, val_labels, val_energies = val_data
            
            print(f"✓ New batch loaded: {len(train_images)} train, {len(val_images)} val")
            print(f"{'='*70}\n")
    
    # Create history object
    class History:
        def __init__(self, history_dict):
            self.history = history_dict
    
    return model, History(history_all), test_data


def evaluate_model(model, test_images, test_labels, test_energies, output_folder):
    """Evaluate model on test set."""
    print("\n" + "="*70)
    print("EVALUATING MODEL")
    print("="*70)
    
    # Predict
    predictions = model.predict(test_images, verbose=1)
    pred_labels = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    
    print(f"\nTest Results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, pred_labels, normalize='true')
    
    print("\nConfusion Matrix (normalized):")
    print(cm)
    
    # Classification report
    target_names = ['ES', 'CC']
    print("\nClassification Report:")
    print(classification_report(test_labels, pred_labels, target_names=target_names))
    
    # Save confusion matrix plot
    if output_folder:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix (Test Set)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'), dpi=150)
        plt.close()
        print(f"✓ Saved confusion matrix to {output_folder}/confusion_matrix.png")
        
        # Save predictions for comprehensive analysis
        pred_file = os.path.join(output_folder, 'test_predictions.npz')
        np.savez(pred_file,
                 predictions=predictions,
                 true_labels=test_labels,
                 test_images=test_images,
                 energies=test_energies)
        print(f"✓ Saved predictions to {pred_file}")
    
    return {
        'test_loss': float(loss),
        'test_accuracy': float(accuracy),
        'confusion_matrix': cm.tolist()
    }


def main():
    args = parse_args()
    config = load_config(args.json)
    
    # Extract config
    plane = args.plane
    max_samples = config.get('max_samples_per_class', args.max_samples)
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 32)
    reload_epochs = args.reload_epochs
    learning_rate = config.get('learning_rate', 0.001)
    
    # Data directories
    es_dir = config.get('es_directory')
    cc_dir = config.get('cc_directory')
    
    if not es_dir or not cc_dir:
        raise ValueError("Config must contain 'es_directory' and 'cc_directory'")
    
    # Output folder
    output_folder = config.get('output_folder', 'training_output/channel_tagging')
    model_name = config.get('model_name', 'ct_volume_batch_reload')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_folder, f"{model_name}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Output folder: {output_folder}")
    
    # Save config
    with open(os.path.join(output_folder, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Test mode with tiny dataset
    if args.test_local:
        print("\n⚠ TEST MODE: Using tiny dataset")
        max_samples = 100
        epochs = 10
        reload_epochs = 3
    
    # Create data loader function
    def data_loader_fn(seed=None):
        return load_volume_batch(
            es_dir, cc_dir, plane=plane,
            max_samples_per_class=max_samples,
            seed=seed
        )
    
    # Load initial batch
    print("\n" + "="*70)
    print("STEP 1: LOADING INITIAL DATA")
    print("="*70)
    images, labels, energies = data_loader_fn(seed=42)
    
    # Split data
    train_data, val_data, test_data = split_data(images, labels, energies)
    
    # Create model
    print("\n" + "="*70)
    print("STEP 2: CREATING MODEL")
    print("="*70)
    
    input_shape = (208, 1242, 1)
    model_params = config.get('model_parameters', {})
    
    # Architecture note for v78 retraining:
    # Keep the historical v57-era CT baseline defaults via model_parameters,
    # while only swapping the input sample directories in the JSON config.
    model = create_deep_cnn(
        input_shape=input_shape,
        filter_list=model_params.get('filter_list', [28, 28, 29, 47, 48, 48]),
        kernel_sizes=model_params.get('kernel_sizes', [3, 3, 3, 3, 3, 3]),
        dropout_rate=model_params.get('dropout_rate', 0.3),
        dense_units=model_params.get('dense_units', [96, 32]),
        n_classes=2
    )
    
    print("Model created:")
    model.summary()
    
    # Train with batch reload
    print("\n" + "="*70)
    print("STEP 3: TRAINING")
    print("="*70)
    
    model, history, test_data = train_with_batch_reload(
        model=model,
        initial_train=train_data,
        initial_val=val_data,
        test_data=test_data,
        data_loader_fn=data_loader_fn,
        split_fn=split_data,
        epochs=epochs,
        batch_size=batch_size,
        reload_every_n_epochs=reload_epochs,
        output_folder=output_folder,
        learning_rate=learning_rate
    )
    
    print("\n✓ Training completed")
    
    # Save final model
    model_path = os.path.join(output_folder, f'{model_name}.keras')
    model.save(model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save history
    gpl.save_history(history, output_folder)
    print(f"✓ History saved")
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("STEP 4: EVALUATION")
    print("="*70)
    
    test_images, test_labels, test_energies = test_data
    metrics = evaluate_model(model, test_images, test_labels, test_energies, output_folder)
    
    # Save results
    results = {
        'config': config,
        'metrics': metrics,
        'history': gpl.history_to_serializable(history)
    }
    
    results_path = os.path.join(output_folder, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Output folder: {output_folder}")
    

if __name__ == '__main__':
    main()
