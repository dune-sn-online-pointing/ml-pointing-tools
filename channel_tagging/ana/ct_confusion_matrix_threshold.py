#!/usr/bin/env python3
"""
Generate confusion matrix for CT model with confidence threshold.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import argparse
from pathlib import Path

def load_predictions(results_dir):
    """Load predictions from model results."""
    predictions_path = Path(results_dir) / 'test_predictions.npz'
    data = np.load(predictions_path)
    
    return {
        'true_labels': data['y_true'],
        'predictions': data['y_pred']
    }

def apply_threshold(y_prob, threshold=0.8):
    """Apply confidence threshold to predictions."""
    max_probs = y_prob.max(axis=1)
    y_pred = y_prob.argmax(axis=1)
    
    # Mask predictions below threshold
    mask = max_probs >= threshold
    
    return y_pred, mask, max_probs

def plot_confusion_matrices(y_true, y_pred_no_thresh, y_pred_thresh, mask, threshold, output_path):
    """Plot confusion matrices with and without threshold."""
    class_names = ['0-ES', '1-Induction', '2-Collection', '3-Other', '4-NC', '5-BKG', '6-Pileup']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # No threshold
    cm_no_thresh = confusion_matrix(y_true, y_pred_no_thresh, labels=range(7))
    acc_no_thresh = accuracy_score(y_true, y_pred_no_thresh)
    
    sns.heatmap(cm_no_thresh, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True', fontsize=12, fontweight='bold')
    axes[0].set_title(f'No Threshold\nAccuracy: {acc_no_thresh:.3f}\nSamples: {len(y_true)}', 
                     fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)
    
    # With threshold
    y_true_thresh = y_true[mask]
    y_pred_thresh_filtered = y_pred_thresh[mask]
    cm_thresh = confusion_matrix(y_true_thresh, y_pred_thresh_filtered, labels=range(7))
    acc_thresh = accuracy_score(y_true_thresh, y_pred_thresh_filtered)
    kept_pct = 100 * mask.sum() / len(mask)
    
    sns.heatmap(cm_thresh, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Count'})
    axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Threshold ≥ {threshold}\nAccuracy: {acc_thresh:.3f}\n'
                     f'Samples: {mask.sum()} ({kept_pct:.1f}%)', 
                     fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)
    
    plt.suptitle('CT Model Confusion Matrices', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to: {output_path}")
    
    # Print detailed stats
    print(f"\n{'='*70}")
    print(f"NO THRESHOLD:")
    print(f"{'='*70}")
    print(f"Accuracy: {acc_no_thresh:.4f}")
    print(f"Total samples: {len(y_true)}")
    print(f"\n{classification_report(y_true, y_pred_no_thresh, target_names=class_names, zero_division=0)}")
    
    print(f"\n{'='*70}")
    print(f"THRESHOLD ≥ {threshold}:")
    print(f"{'='*70}")
    print(f"Accuracy: {acc_thresh:.4f}")
    print(f"Samples kept: {mask.sum()} / {len(y_true)} ({kept_pct:.1f}%)")
    print(f"Samples rejected: {(~mask).sum()} ({100-kept_pct:.1f}%)")
    print(f"\n{classification_report(y_true_thresh, y_pred_thresh_filtered, target_names=class_names, zero_division=0)}")

def main():
    parser = argparse.ArgumentParser(description='Generate CT confusion matrix with threshold')
    parser.add_argument('results_dir', help='Path to model results directory')
    parser.add_argument('-t', '--threshold', type=float, default=0.8, 
                       help='Confidence threshold (default: 0.8)')
    parser.add_argument('-o', '--output', default=None,
                       help='Output image path (default: ct_confusion_threshold_XX.png)')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading predictions from: {args.results_dir}")
    predictions = load_predictions(args.results_dir)
    
    y_true = predictions['true_labels'].astype(int)
    y_prob = predictions['predictions']
    
    # Apply threshold
    y_pred_thresh, mask, max_probs = apply_threshold(y_prob, args.threshold)
    y_pred_no_thresh = y_prob.argmax(axis=1)
    
    # Output path
    if args.output is None:
        thresh_str = str(int(args.threshold * 100))
        args.output = f"ct_confusion_threshold_{thresh_str}.png"
    
    # Plot
    plot_confusion_matrices(y_true, y_pred_no_thresh, y_pred_thresh, mask, 
                           args.threshold, args.output)

if __name__ == '__main__':
    main()
