#!/usr/bin/env python3
"""
Comprehensive CT Model Analysis PDF Report
Generates multi-page PDF with:
- Confusion matrices at different thresholds (0.6, 0.7, 0.8, 0.9)
- ES prediction probability distributions by true label
- Per-class metrics
- Training history
- ROC curves
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Class labels for CT task
CLASS_NAMES = ['0-ES', '1-Induction', '2-Collection', '3-Other', '4-NC', '5-BKG', '6-Pileup']
N_CLASSES = 7

def load_data(results_dir):
    """Load predictions and results from model directory."""
    results_dir = Path(results_dir)
    
    # Load predictions
    pred_files = list(results_dir.glob('*predictions*.npz'))
    if not pred_files:
        raise FileNotFoundError(f"No predictions found in {results_dir}")
    
    predictions = np.load(pred_files[0], allow_pickle=True)
    y_true = predictions['true_labels'].astype(int) if 'true_labels' in predictions else predictions['y_true'].astype(int)
    y_prob = predictions['predictions'] if 'predictions' in predictions else predictions['y_pred']
    
    # Load results/metrics JSON
    result_files = list(results_dir.glob('*results*.json')) + list(results_dir.glob('*metrics*.json'))
    results = {}
    if result_files:
        with open(result_files[0], 'r') as f:
            results = json.load(f)
    
    # Load training history if available
    history = results.get('history', {})
    if not history:
        history_files = list(results_dir.glob('training_history.json'))
        if history_files:
            with open(history_files[0], 'r') as f:
                history = json.load(f)
    
    return y_true, y_prob, results, history, results_dir.name


def plot_confusion_matrices_page(pdf, y_true, y_prob, model_name):
    """Page 1: Confusion matrices at different thresholds."""
    thresholds = [0.6, 0.7, 0.8, 0.9]
    fig = plt.figure(figsize=(17, 11))
    fig.suptitle(f'CT Model: {model_name}\nConfusion Matrices at Different Confidence Thresholds', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    for idx, threshold in enumerate(thresholds):
        ax = plt.subplot(2, 2, idx + 1)
        
        # Apply threshold
        max_probs = y_prob.max(axis=1)
        y_pred = y_prob.argmax(axis=1)
        mask = max_probs >= threshold
        
        # Filter data
        y_true_filt = y_true[mask]
        y_pred_filt = y_pred[mask]
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true_filt, y_pred_filt, labels=range(N_CLASSES))
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        # Plot
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        
        # Add percentages
        thresh_color = cm_norm.max() / 2.
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                color = "white" if cm_norm[i, j] > thresh_color else "black"
                text = f'{cm_norm[i, j]:.1%}'
                ax.text(j, i, text, ha="center", va="center",
                       color=color, fontsize=10, fontweight='bold')
        
        # Formatting
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
        
        # Metrics
        acc = accuracy_score(y_true_filt, y_pred_filt)
        kept_pct = 100 * mask.sum() / len(mask)
        n_samples = mask.sum()
        
        title_text = f'Threshold ≥{threshold:.1f}\n'
        title_text += f'Acc={acc:.1%}, N={n_samples:,} ({kept_pct:.1f}%)'
        ax.set_title(title_text, fontsize=12, fontweight='bold')
        
        # Colorbar
        if idx % 2 == 1:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Recall', rotation=270, labelpad=15, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Page 1: Confusion matrices at thresholds")


def plot_prediction_distributions_page(pdf, y_true, y_prob, model_name):
    """Page 2: Distribution of ES prediction probabilities by true label."""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'CT Model {model_name}: Distribution of ES Prediction Probabilities by True Label',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # ES probabilities (class 0)
    es_probs = y_prob[:, 0]
    
    # Plot histogram for each true class
    thresholds = [0.6, 0.7, 0.8, 0.9]
    colors_cc = ['purple', 'blue']
    colors_es = ['orange', 'brown']
    
    # Define true class labels mapping (True CC vs True ES)
    # CC = class 1 (Induction) + class 2 (Collection)
    # ES = class 0 (ES)
    true_cc_mask = np.isin(y_true, [1, 2])
    true_es_mask = (y_true == 0)
    
    bins = np.linspace(0, 1, 51)
    
    # Plot overlapping histograms
    ax = plt.subplot(1, 1, 1)
    
    ax.hist(es_probs[true_cc_mask], bins=bins, alpha=0.6, label='True CC', 
            color='blue', edgecolor='darkblue')
    ax.hist(es_probs[true_es_mask], bins=bins, alpha=0.6, label='True ES',
            color='orange', edgecolor='darkorange')
    
    # Add threshold lines
    for threshold in thresholds:
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(threshold, ax.get_ylim()[1] * 0.95, f'{threshold:.1f}',
               ha='center', va='top', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
    
    ax.set_xlabel('ES Prediction Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    # Add statistics text box
    n_cc = true_cc_mask.sum()
    n_es = true_es_mask.sum()
    mean_cc = es_probs[true_cc_mask].mean()
    mean_es = es_probs[true_es_mask].mean()
    
    stats_text = f'True CC: N={n_cc:,}, Mean ES prob={mean_cc:.3f}\n'
    stats_text += f'True ES: N={n_es:,}, Mean ES prob={mean_es:.3f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Page 2: ES prediction probability distributions")


def plot_metrics_page(pdf, y_true, y_prob, model_name):
    """Page 3: Per-class metrics."""
    y_pred = y_prob.argmax(axis=1)
    
    # Calculate metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(N_CLASSES), zero_division=0
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'CT Model: {model_name}\nPer-Class Performance Metrics',
                 fontsize=14, fontweight='bold')
    
    x = np.arange(N_CLASSES)
    width = 0.6
    
    # Precision
    axes[0, 0].bar(x, precision, width, color='steelblue', edgecolor='black')
    axes[0, 0].set_ylabel('Precision', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Precision by Class', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(precision):
        axes[0, 0].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
    
    # Recall
    axes[0, 1].bar(x, recall, width, color='coral', edgecolor='black')
    axes[0, 1].set_ylabel('Recall', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Recall by Class', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(recall):
        axes[0, 1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
    
    # F1 Score
    axes[1, 0].bar(x, f1, width, color='mediumseagreen', edgecolor='black')
    axes[1, 0].set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('F1 Score by Class', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(f1):
        axes[1, 0].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
    
    # Support
    axes[1, 1].bar(x, support, width, color='mediumpurple', edgecolor='black')
    axes[1, 1].set_ylabel('Support (# samples)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Support by Class', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(support):
        axes[1, 1].text(i, v + 0.02, f'{int(v)}', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Page 3: Per-class metrics")


def plot_training_history_page(pdf, history, model_name):
    """Page 4: Training history."""
    if not history:
        print("  ⚠ No training history available, skipping")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'CT Model: {model_name}\nTraining History',
                 fontsize=14, fontweight='bold')
    
    epochs = range(1, len(history.get('loss', [])) + 1)
    
    # Loss
    if 'loss' in history:
        axes[0].plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss')
        if 'val_loss' in history:
            axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
        axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=11, fontweight='bold')
        axes[0].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    if 'accuracy' in history or 'acc' in history:
        acc_key = 'accuracy' if 'accuracy' in history else 'acc'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'
        
        axes[1].plot(epochs, history[acc_key], 'b-', linewidth=2, label='Training Accuracy')
        if val_acc_key in history:
            axes[1].plot(epochs, history[val_acc_key], 'r-', linewidth=2, label='Validation Accuracy')
        axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        axes[1].set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Page 4: Training history")


def plot_roc_curves_page(pdf, y_true, y_prob, model_name):
    """Page 5: ROC curves (one-vs-rest)."""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'CT Model: {model_name}\nROC Curves (One-vs-Rest)',
                 fontsize=14, fontweight='bold')
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(N_CLASSES))
    
    # Plot ROC for each class
    for i in range(N_CLASSES):
        ax = plt.subplot(3, 3, i + 1)
        
        if len(np.unique(y_true_bin[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=9)
            ax.set_ylabel('True Positive Rate', fontsize=9)
            ax.set_title(f'{CLASS_NAMES[i]}', fontsize=11, fontweight='bold')
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No positive samples', ha='center', va='center',
                   fontsize=10, transform=ax.transAxes)
            ax.set_title(f'{CLASS_NAMES[i]}', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Page 5: ROC curves")


def generate_report(results_dir, output_pdf=None):
    """Generate comprehensive PDF report."""
    print(f"\n{'='*70}")
    print(f"Generating CT Model Analysis Report")
    print(f"{'='*70}")
    print(f"Model directory: {results_dir}")
    
    # Load data
    y_true, y_prob, results, history, model_name = load_data(results_dir)
    print(f"Loaded: {len(y_true)} test samples, {y_prob.shape[1]} classes")
    
    # Output PDF
    if output_pdf is None:
        output_pdf = Path(results_dir) / f'{model_name}_comprehensive_report.pdf'
    else:
        output_pdf = Path(output_pdf)
    
    print(f"\nGenerating PDF: {output_pdf}")
    print(f"{'='*70}\n")
    
    with PdfPages(output_pdf) as pdf:
        plot_confusion_matrices_page(pdf, y_true, y_prob, model_name)
        plot_prediction_distributions_page(pdf, y_true, y_prob, model_name)
        plot_metrics_page(pdf, y_true, y_prob, model_name)
        plot_training_history_page(pdf, history, model_name)
        plot_roc_curves_page(pdf, y_true, y_prob, model_name)
    
    print(f"\n{'='*70}")
    print(f"✓ PDF report saved: {output_pdf}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive CT model PDF report')
    parser.add_argument('results_dir', help='Path to model results directory')
    parser.add_argument('-o', '--output', default=None, help='Output PDF path')
    args = parser.parse_args()
    
    generate_report(args.results_dir, args.output)


if __name__ == '__main__':
    main()
