#!/usr/bin/env python3
"""Test CT v72 deeper model that was held before completion."""

import numpy as np
import tensorflow as tf
from pathlib import Path
import json

def load_ct_data(data_dir, plane='X', max_samples=500, seed=42):
    """Load CT volume images for testing."""
    np.random.seed(seed)
    
    data_path = Path(data_dir) / plane
    all_files = sorted(data_path.glob('*.npz'))
    
    print(f"  Total files available: {len(all_files)}")
    
    # Use different files than training (use last N samples)
    if len(all_files) > max_samples:
        test_files = all_files[-max_samples:]
    else:
        test_files = all_files[:max_samples]
    
    images = []
    labels = []
    
    print(f"Loading {len(test_files)} test samples from {plane} plane...")
    
    for i, file_path in enumerate(test_files):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(test_files)}")
        
        data = np.load(file_path, allow_pickle=True)
        imgs = data['images']
        metadata = data['metadata']
        
        for img, meta in zip(imgs, metadata):
            images.append(img)
            label = 1 if meta['interaction_type'] == 'ES' else 0
            labels.append(label)
    
    images = np.array(images, dtype=np.float32)[..., np.newaxis]
    labels = np.array(labels, dtype=np.int32)
    
    return images, labels

def main():
    # Load model
    model_dir = Path('/afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/training_output/channel_tagging/ct_volume_v72_deeper_100k_20251119_161206')
    model_path = model_dir / 'best_model.keras'
    
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"✓ Model loaded")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Parameters: {model.count_params():,}")
    
    # Load config
    with open(model_dir / 'config.json') as f:
        config = json.load(f)
    
    # Load test data
    data_dir = '/eos/home-e/evilla/dune/sn-tps/prod_es/es_production_volume_images_tick3_ch2_min2_tot3_e2p0'
    test_images, test_labels = load_ct_data(data_dir, plane='X', max_samples=500, seed=99)
    
    print(f"\n✓ Test data loaded")
    print(f"  Shape: {test_images.shape}")
    print(f"  ES samples: {np.sum(test_labels)} ({100*np.mean(test_labels):.1f}%)")
    print(f"  CC samples: {len(test_labels) - np.sum(test_labels)} ({100*(1-np.mean(test_labels)):.1f}%)")
    
    # Evaluate
    print(f"\nEvaluating on test set...")
    predictions = model.predict(test_images, batch_size=32, verbose=1)
    pred_labels = (predictions[:, 1] > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = np.mean(pred_labels == test_labels)
    
    # Confusion matrix
    tp = np.sum((pred_labels == 1) & (test_labels == 1))
    tn = np.sum((pred_labels == 0) & (test_labels == 0))
    fp = np.sum((pred_labels == 1) & (test_labels == 0))
    fn = np.sum((pred_labels == 0) & (test_labels == 1))
    
    es_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    es_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    cc_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    cc_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Save results
    results = {
        'test_accuracy': float(accuracy),
        'test_es_recall': float(es_recall),
        'test_es_precision': float(es_precision),
        'test_cc_recall': float(cc_recall),
        'test_cc_precision': float(cc_precision),
        'confusion_matrix': {
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        },
        'n_test_samples': len(test_labels)
    }
    
    output_path = model_dir / 'test_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"CT v72 deeper - Test Results ({len(test_labels)} samples)")
    print(f"{'='*60}")
    print(f"Accuracy:      {100*accuracy:.2f}%")
    print(f"\nES (class 1):")
    print(f"  Recall:      {100*es_recall:.2f}%")
    print(f"  Precision:   {100*es_precision:.2f}%")
    print(f"\nCC (class 0):")
    print(f"  Recall:      {100*cc_recall:.2f}%")
    print(f"  Precision:   {100*cc_precision:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:4d}  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  TN: {tn:4d}")
    print(f"\n✓ Results saved to: {output_path}")

if __name__ == '__main__':
    main()
