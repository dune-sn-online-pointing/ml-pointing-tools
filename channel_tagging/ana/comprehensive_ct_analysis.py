#!/usr/bin/env python3
"""
Comprehensive Channel Tagging Analysis
Generates a multi-page PDF with all analysis plots including:
- Multi-class confusion matrix
- Per-class precision, recall, F1
- ROC curves (one-vs-rest)
- Training history
- Energy-dependent performance
- Prediction distributions
- Best and worst predictions per class
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import json
import csv
import argparse
import os
import sys
from pathlib import Path

# Channel labels mapping
CHANNEL_LABELS = {
    0: 'ES',
    1: 'CC'
}

def load_results(results_dir):
    """Load results.json and predictions."""
    results_dir = Path(results_dir)
    results_path = results_dir / 'results.json'
    metrics_path = results_dir / 'v77_dario_batch_5k_metrics.json'
    pred_path = results_dir / 'test_predictions.npz'
    pred_path_alt = results_dir / 'v77_dario_batch_5k_predictions.npz'
    pred_energy_path = results_dir / 'test_predictions_with_energy.npz'
    history_path = results_dir / 'training_history.json'
    
    # Try different result file names
    if results_path.exists():
        results_file = results_path
    elif metrics_path.exists():
        results_file = metrics_path
    else:
        raise FileNotFoundError(f"No results.json or metrics.json found in {results_dir}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Load training history if available
    if 'history' not in results:
        if history_path.exists():
            with open(history_path, 'r') as f:
                results['history'] = json.load(f)
        else:
            # Try CSV format
            csv_path = results_dir / f"{results_dir.name}_training_log.csv"
            if not csv_path.exists():
                # Try finding any training_log.csv
                csv_files = list(results_dir.glob('*training_log.csv'))
                if csv_files:
                    csv_path = csv_files[0]
            if csv_path.exists():
                import csv as csv_module
                history = {}
                with open(csv_path, 'r') as f:
                    reader = csv_module.DictReader(f)
                    for row in reader:
                        for key, value in row.items():
                            if not key or key.strip().lower() in ['epoch', 'cycle']:
                                continue
                            key = key.strip()
                            history.setdefault(key, [])
                            try:
                                history[key].append(float(value))
                            except (TypeError, ValueError):
                                history[key].append(np.nan)
                if history:
                    results['history'] = history
    
    # Try loading predictions with energy first, fall back to regular predictions
    if pred_energy_path.exists():
        predictions = np.load(pred_energy_path, allow_pickle=True)
    elif pred_path.exists():
        predictions = np.load(pred_path, allow_pickle=True)
    elif pred_path_alt.exists():
        predictions = np.load(pred_path_alt, allow_pickle=True)
    else:
        predictions = None

    # Normalize prediction file schemas across different trainers.
    # Expected by downstream plots:
    #   - predictions['predictions'] : (N, n_classes) probabilities
    #   - predictions['true_labels'] : (N,) integer labels
    if predictions is not None:
        keys = set(predictions.files) if hasattr(predictions, "files") else set(predictions.keys())

        # Legacy CT volume trainer format
        if 'predictions' not in keys and 'y_prob' in keys:
            prob = predictions['y_prob']
            pred_dict = {k: predictions[k] for k in keys}
            pred_dict['predictions'] = prob
            predictions = pred_dict
            keys = set(pred_dict.keys())

        if 'true_labels' not in keys:
            if 'y_true' in keys:
                pred_dict = dict(predictions)
                pred_dict['true_labels'] = pred_dict['y_true']
                predictions = pred_dict
                keys = set(pred_dict.keys())
            elif 'true_labels' not in keys and 'true' in keys:
                pred_dict = dict(predictions)
                pred_dict['true_labels'] = pred_dict['true']
                predictions = pred_dict
                keys = set(pred_dict.keys())
    
    # Convert binary predictions to 2-class format if needed
    if predictions is not None and 'predictions' in predictions:
        preds = predictions['predictions']
        if len(preds.shape) == 2 and preds.shape[1] == 1:
            # Binary classification with single sigmoid output
            # Convert to 2-class probabilities: [1-p, p]
            preds_2class = np.hstack([1 - preds, preds])
            # Create new dict with updated predictions
            predictions = dict(predictions)
            predictions['predictions'] = preds_2class
    
    return results, predictions


def calculate_metrics(y_true, y_pred, y_prob, n_classes=7):
    """Calculate multi-class classification metrics."""
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                  confusion_matrix, roc_auc_score)
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(n_classes), zero_division=0
    )
    
    # Weighted averages
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'precision_weighted': precision_w,
        'recall_weighted': recall_w,
        'f1_weighted': f1_w,
        'confusion_matrix': cm
    }
    
    # ROC AUC (one-vs-rest)
    try:
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # label_binarize returns only one column for binary tasks, but the
        # downstream plots expect one column per class. Expand the negative
        # class explicitly so the arrays always have shape (N, n_classes).
        if y_true_bin.shape[1] != n_classes:
            if n_classes == 2 and y_true_bin.shape[1] == 1:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
            else:
                raise ValueError(
                    f"Unexpected binarized shape {y_true_bin.shape} for {n_classes} classes"
                )
        auc_per_class = []
        for i in range(n_classes):
            if len(np.unique(y_true_bin[:, i])) > 1:
                auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                auc_per_class.append(auc)
            else:
                auc_per_class.append(np.nan)
        metrics['auc_per_class'] = auc_per_class
    except:
        metrics['auc_per_class'] = [np.nan] * n_classes
    
    return metrics


def plot_confusion_matrix(predictions, results, fig):
    """Page 1: Confusion matrix and overall metrics."""
    y_true = predictions['true_labels'].astype(int)
    y_pred = predictions['predictions'].argmax(axis=1).astype(int)
    y_prob = predictions['predictions']
    
    n_classes = y_prob.shape[1]
    metrics = calculate_metrics(y_true, y_pred, y_prob, n_classes)
    cm = metrics['confusion_matrix']
    
    gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3, 
                          width_ratios=[2, 1])
    
    # Plot 1: Confusion matrix
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Normalize by row (true labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    im = ax1.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    cbar = ax1.figure.colorbar(im, ax=ax1)
    cbar.set_label('Recall (Fraction)', rotation=270, labelpad=20, fontsize=11, fontweight='bold')
    
    # Add text annotations (percentages only)
    thresh = cm_norm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm_norm[i, j] > thresh else "black"
            # Show percentage only
            text = f'{cm_norm[i, j]:.1%}'
            ax1.text(j, i, text, ha="center", va="center",
                    color=color, fontsize=12, fontweight='bold')
    
    # Labels
    labels = [CHANNEL_LABELS.get(i, f'Class {i}') for i in range(n_classes)]
    ax1.set_xticks(range(n_classes))
    ax1.set_yticks(range(n_classes))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_ylabel('True Channel', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Channel', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix (Normalized by True Class)', 
                 fontsize=13, fontweight='bold')
    
    # Plot 2: Metrics summary
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    total = len(y_true)
    
    metrics_text = f"""OVERALL METRICS

Total Samples: {total:,}

Weighted Averages:
  Accuracy:  {metrics['accuracy']:.4f}
  Precision: {metrics['precision_weighted']:.4f}
  Recall:    {metrics['recall_weighted']:.4f}
  F1 Score:  {metrics['f1_weighted']:.4f}

Class Distribution:"""
    
    # Add class counts
    for i in range(n_classes):
        count = metrics['support'][i]
        pct = 100 * count / total
        label = CHANNEL_LABELS.get(i, f'C{i}')
        metrics_text += f"\n  {label:12s}: {count:5d} ({pct:4.1f}%)"
    
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5))


def plot_per_class_metrics(predictions, fig):
    """Page 2: Per-class precision, recall, F1, and AUC."""
    y_true = predictions['true_labels'].astype(int)
    y_pred = predictions['predictions'].argmax(axis=1).astype(int)
    y_prob = predictions['predictions']
    
    n_classes = y_prob.shape[1]
    metrics = calculate_metrics(y_true, y_pred, y_prob, n_classes)
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    labels = [CHANNEL_LABELS.get(i, f'C{i}') for i in range(n_classes)]
    x_pos = np.arange(n_classes)
    
    # Plot 1: Precision per class
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(x_pos, metrics['precision'], alpha=0.7, edgecolor='black')
    ax1.axhline(metrics['precision_weighted'], color='red', linestyle='--', 
               linewidth=2, label=f'Weighted: {metrics["precision_weighted"]:.3f}')
    ax1.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax1.set_title('Precision per Channel', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.set_ylim([0, 1])
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, axis='y')
    
    # Color bars by performance
    for i, bar in enumerate(bars1):
        if metrics['precision'][i] >= 0.8:
            bar.set_facecolor('green')
        elif metrics['precision'][i] >= 0.6:
            bar.set_facecolor('orange')
        else:
            bar.set_facecolor('red')
    
    # Plot 2: Recall per class
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(x_pos, metrics['recall'], alpha=0.7, edgecolor='black')
    ax2.axhline(metrics['recall_weighted'], color='red', linestyle='--', 
               linewidth=2, label=f'Weighted: {metrics["recall_weighted"]:.3f}')
    ax2.set_ylabel('Recall', fontsize=11, fontweight='bold')
    ax2.set_title('Recall per Channel', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylim([0, 1])
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars2):
        if metrics['recall'][i] >= 0.8:
            bar.set_facecolor('green')
        elif metrics['recall'][i] >= 0.6:
            bar.set_facecolor('orange')
        else:
            bar.set_facecolor('red')
    
    # Plot 3: F1 score per class
    ax3 = fig.add_subplot(gs[1, 0])
    bars3 = ax3.bar(x_pos, metrics['f1'], alpha=0.7, edgecolor='black')
    ax3.axhline(metrics['f1_weighted'], color='red', linestyle='--', 
               linewidth=2, label=f'Weighted: {metrics["f1_weighted"]:.3f}')
    ax3.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax3.set_title('F1 Score per Channel', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax3.set_ylim([0, 1])
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars3):
        if metrics['f1'][i] >= 0.8:
            bar.set_facecolor('green')
        elif metrics['f1'][i] >= 0.6:
            bar.set_facecolor('orange')
        else:
            bar.set_facecolor('red')
    
    # Plot 4: ROC AUC per class
    ax4 = fig.add_subplot(gs[1, 1])
    auc_values = metrics['auc_per_class']
    bars4 = ax4.bar(x_pos, auc_values, alpha=0.7, edgecolor='black')
    mean_auc = np.nanmean(auc_values)
    ax4.axhline(mean_auc, color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {mean_auc:.3f}')
    ax4.set_ylabel('ROC AUC (OvR)', fontsize=11, fontweight='bold')
    ax4.set_title('ROC AUC per Channel (One-vs-Rest)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax4.set_ylim([0, 1])
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars4):
        if not np.isnan(auc_values[i]):
            if auc_values[i] >= 0.9:
                bar.set_facecolor('green')
            elif auc_values[i] >= 0.75:
                bar.set_facecolor('orange')
            else:
                bar.set_facecolor('red')


def plot_roc_curves(predictions, fig):
    """Page 3: ROC curves for each class (one-vs-rest)."""
    y_true = predictions['true_labels'].astype(int)
    y_prob = predictions['predictions']
    
    n_classes = y_prob.shape[1]
    
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    if y_true_bin.shape[1] != n_classes:
        if n_classes == 2 and y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        else:
            raise ValueError(
                f"Unexpected binarized shape {y_true_bin.shape} for {n_classes} classes"
            )
    
    # Calculate ROC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        if len(np.unique(y_true_bin[:, i])) > 1:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            fpr[i] = None
            tpr[i] = None
            roc_auc[i] = np.nan
    
    # Create subplots - 3 rows, 3 columns (for 7 classes + micro average)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    for i in range(n_classes):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        if fpr[i] is not None:
            ax.plot(fpr[i], tpr[i], linewidth=2, 
                   label=f'AUC = {roc_auc[i]:.3f}')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlabel('FPR', fontsize=10, fontweight='bold')
        ax.set_ylabel('TPR', fontsize=10, fontweight='bold')
        ax.set_title(f'{CHANNEL_LABELS.get(i, f"Class {i}")}', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    # Micro-average ROC (last subplot)
    ax = fig.add_subplot(gs[2, 1])
    
    # Compute micro-average
    y_true_flat = y_true_bin.ravel()
    y_prob_flat = y_prob.ravel()
    
    if len(np.unique(y_true_flat)) > 1:
        fpr_micro, tpr_micro, _ = roc_curve(y_true_flat, y_prob_flat)
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        ax.plot(fpr_micro, tpr_micro, linewidth=3, 
               label=f'Micro-avg AUC = {roc_auc_micro:.3f}', color='deeppink')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('FPR', fontsize=10, fontweight='bold')
    ax.set_ylabel('TPR', fontsize=10, fontweight='bold')
    ax.set_title('Micro-Average ROC', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Hide last subplot
    ax_hide = fig.add_subplot(gs[2, 2])
    ax_hide.axis('off')


def plot_training_history(results, fig):
    """Page 4: Training history analysis."""
    history = results['history']
    
    # Filter out NaN values
    epochs_loss = [i+1 for i, x in enumerate(history['loss']) if not (isinstance(x, float) and np.isnan(x))]
    loss = [x for x in history['loss'] if not (isinstance(x, float) and np.isnan(x))]
    val_loss = [x for x in history['val_loss'] if not (isinstance(x, float) and np.isnan(x))]
    
    # Check for accuracy metrics
    has_accuracy = 'accuracy' in history or 'acc' in history
    acc_key = 'accuracy' if 'accuracy' in history else 'acc' if 'acc' in history else None
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc' if 'val_acc' in history else None
    
    if has_accuracy and acc_key and val_acc_key:
        gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.3)
    else:
        gs = gridspec.GridSpec(1, 1, figure=fig)
    
    # Plot 1: Loss curves
    ax1 = fig.add_subplot(gs[0, 0] if has_accuracy else gs[0])
    ax1.plot(epochs_loss, loss, label='Training Loss', linewidth=2, marker='o', markersize=3)
    ax1.plot(epochs_loss, val_loss, label='Validation Loss', linewidth=2, marker='s', markersize=3)
    
    # Mark best epoch
    best_epoch = np.argmin(val_loss) + 1
    best_val_loss = np.min(val_loss)
    ax1.axvline(best_epoch, color='red', linestyle='--', alpha=0.5, 
               label=f'Best epoch: {best_epoch}')
    ax1.scatter([best_epoch], [best_val_loss], color='red', s=100, zorder=5, marker='*')
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss (Categorical Crossentropy)', fontsize=12, fontweight='bold')
    ax1.set_title('Training History - Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Accuracy curves (if available)
    if has_accuracy and acc_key and val_acc_key:
        ax2 = fig.add_subplot(gs[1, 0])
        acc = [x for x in history[acc_key] if not (isinstance(x, float) and np.isnan(x))]
        val_acc = [x for x in history[val_acc_key] if not (isinstance(x, float) and np.isnan(x))]
        epochs_acc = [i+1 for i, x in enumerate(history[acc_key]) if not (isinstance(x, float) and np.isnan(x))]
        
        ax2.plot(epochs_acc, acc, label='Training Accuracy', linewidth=2, marker='o', markersize=3)
        ax2.plot(epochs_acc, val_acc, label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
        ax2.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Training History - Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)


def plot_prediction_distribution(predictions, fig):
    """Page 5: Prediction confidence distributions."""
    y_true = predictions['true_labels'].astype(int)
    y_prob = predictions['predictions']
    y_pred = y_prob.argmax(axis=1)
    
    n_classes = y_prob.shape[1]
    
    # Max probability (confidence)
    max_probs = y_prob.max(axis=1)
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # Plot 1: Confidence distribution for correct vs incorrect
    ax1 = fig.add_subplot(gs[0, :])
    
    correct_mask = (y_pred == y_true)
    correct_probs = max_probs[correct_mask]
    incorrect_probs = max_probs[~correct_mask]
    
    bins = np.linspace(0, 1, 40)
    ax1.hist(correct_probs, bins=bins, alpha=0.6, label=f'Correct ({len(correct_probs):,})', 
            color='green', edgecolor='black')
    ax1.hist(incorrect_probs, bins=bins, alpha=0.6, label=f'Incorrect ({len(incorrect_probs):,})', 
            color='red', edgecolor='black')
    ax1.set_xlabel('Prediction Confidence (Max Probability)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Confidence Distribution: Correct vs Incorrect', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.axvline(correct_probs.mean(), color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(incorrect_probs.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Plot 2: Mean confidence per class
    ax2 = fig.add_subplot(gs[1, 0])
    
    labels = [CHANNEL_LABELS.get(i, f'C{i}') for i in range(n_classes)]
    mean_conf_per_class = []
    
    for i in range(n_classes):
        mask = (y_true == i)
        if np.sum(mask) > 0:
            mean_conf = max_probs[mask].mean()
            mean_conf_per_class.append(mean_conf)
        else:
            mean_conf_per_class.append(0)
    
    x_pos = np.arange(n_classes)
    bars = ax2.bar(x_pos, mean_conf_per_class, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Mean Confidence', fontsize=11, fontweight='bold')
    ax2.set_title('Mean Confidence per True Channel', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylim([0, 1])
    ax2.grid(alpha=0.3, axis='y')
    ax2.axhline(max_probs.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    for bar in bars:
        bar.set_facecolor('steelblue')
    
    # Plot 3: Accuracy vs confidence threshold
    ax3 = fig.add_subplot(gs[1, 1])
    
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    kept_samples = []
    
    for thresh in thresholds:
        mask = max_probs >= thresh
        if np.sum(mask) > 0:
            acc = np.mean(y_pred[mask] == y_true[mask])
            accuracies.append(acc)
            kept_samples.append(100 * np.sum(mask) / len(y_true))
        else:
            accuracies.append(np.nan)
            kept_samples.append(0)
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(thresholds, accuracies, linewidth=2, color='blue', label='Accuracy')
    line2 = ax3_twin.plot(thresholds, kept_samples, linewidth=2, color='orange', 
                          linestyle='--', label='% Kept')
    
    ax3.set_xlabel('Confidence Threshold', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Accuracy', fontsize=11, fontweight='bold', color='blue')
    ax3_twin.set_ylabel('% Samples Kept', fontsize=11, fontweight='bold', color='orange')
    ax3.set_title('Accuracy vs Confidence Threshold', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.set_ylim([0, 1])
    ax3_twin.set_ylim([0, 100])
    
    # Combined legend
    lines = line1 + line2
    labels_leg = [l.get_label() for l in lines]
    ax3.legend(lines, labels_leg, fontsize=10)




def plot_confusion_matrices_thresholds(predictions, fig):
    """Page with confusion matrices at different confidence thresholds."""
    y_true = predictions['true_labels'].astype(int)
    y_prob = predictions['predictions']
    y_pred = y_prob.argmax(axis=1)
    
    n_classes = y_prob.shape[1]
    labels = [CHANNEL_LABELS.get(i, f'C{i}') for i in range(n_classes)]
    
    # Define thresholds to test
    thresholds = [0.6, 0.7, 0.8, 0.9]
    
    # Create 2x2 subplot grid
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    for idx, threshold in enumerate(thresholds):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        
        # Apply threshold
        max_probs = y_prob.max(axis=1)
        mask = max_probs >= threshold
        
        # Filter predictions
        y_true_filt = y_true[mask]
        y_pred_filt = y_pred[mask]
        
        if len(y_true_filt) == 0:
            ax.text(0.5, 0.5, f'No samples above threshold {threshold}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, fontweight='bold')
            continue
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_true_filt, y_pred_filt, labels=range(n_classes))
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        cm_norm = np.nan_to_num(cm_norm)
        
        # Plot confusion matrix
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        
        # Add percentage annotations
        thresh_color = cm_norm.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                color = "white" if cm_norm[i, j] > thresh_color else "black"
                text = f'{cm_norm[i, j]:.1%}'
                ax.text(j, i, text, ha="center", va="center",
                       color=color, fontsize=9 if n_classes > 3 else 11, fontweight='bold')
        
        # Labels
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8 if n_classes > 3 else 9)
        ax.set_yticklabels(labels, fontsize=8 if n_classes > 3 else 9)
        ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
        
        # Calculate metrics
        acc = accuracy_score(y_true_filt, y_pred_filt)
        kept_pct = 100 * mask.sum() / len(mask)
        n_samples = mask.sum()
        
        title_text = f'Threshold ≥ {threshold:.1f}\n'
        title_text += f'Acc={acc:.1%}, N={n_samples:,} ({kept_pct:.1f}%)'
        ax.set_title(title_text, fontsize=11, fontweight='bold')
        
        # Add colorbar
        if idx % 2 == 1:  # Right column
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Recall', rotation=270, labelpad=15, fontsize=9, fontweight='bold')


def plot_es_probability_distribution(predictions, fig):
    """Page with ES prediction probability distribution by true label."""
    y_true = predictions['true_labels'].astype(int)
    y_prob = predictions['predictions']
    
    n_classes = y_prob.shape[1]
    
    # ES probabilities (class 0)
    es_probs = y_prob[:, 0]
    
    # Define true class groupings
    # For 7-class: CC = classes 1,2 (Induction, Collection), ES = class 0
    # For 2-class: CC = class 1, ES = class 0
    if n_classes == 2:
        true_cc_mask = (y_true == 1)
        true_es_mask = (y_true == 0)
        title_suffix = "ES vs CC"
    elif n_classes == 7:
        true_cc_mask = np.isin(y_true, [1, 2])  # Induction + Collection
        true_es_mask = (y_true == 0)  # ES
        title_suffix = "ES vs CC (Ind+Col)"
    else:
        # Fallback: just show ES vs others
        true_es_mask = (y_true == 0)
        true_cc_mask = (y_true != 0)
        title_suffix = "ES vs Others"
    
    # Create single large plot
    ax = fig.add_subplot(1, 1, 1)
    
    # Histogram bins
    bins = np.linspace(0, 1, 51)
    
    # Plot overlapping histograms
    n_cc = true_cc_mask.sum()
    n_es = true_es_mask.sum()
    
    if n_cc > 0:
        ax.hist(es_probs[true_cc_mask], bins=bins, alpha=0.6, label=f'True CC (N={n_cc:,})', 
                color='blue', edgecolor='darkblue', linewidth=1.5)
    
    if n_es > 0:
        ax.hist(es_probs[true_es_mask], bins=bins, alpha=0.6, label=f'True ES (N={n_es:,})',
                color='orange', edgecolor='darkorange', linewidth=1.5)
    
    # Add threshold lines
    thresholds = [0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(threshold, ax.get_ylim()[1] * 0.95, f'{threshold:.1f}',
               ha='center', va='top', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    # Labels and formatting
    ax.set_xlabel('ES Prediction Probability', fontsize=13, fontweight='bold')
    ax.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax.set_title(f'Distribution of ES Prediction Probabilities: {title_suffix}',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=12, loc='upper center')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    # Add statistics text box
    if n_cc > 0:
        mean_cc = es_probs[true_cc_mask].mean()
        median_cc = np.median(es_probs[true_cc_mask])
        std_cc = es_probs[true_cc_mask].std()
    else:
        mean_cc = median_cc = std_cc = 0
    
    if n_es > 0:
        mean_es = es_probs[true_es_mask].mean()
        median_es = np.median(es_probs[true_es_mask])
        std_es = es_probs[true_es_mask].std()
    else:
        mean_es = median_es = std_es = 0
    
    stats_text = 'Statistics:\n\n'
    if n_cc > 0:
        stats_text += f'True CC:\n'
        stats_text += f'  Mean:   {mean_cc:.3f}\n'
        stats_text += f'  Median: {median_cc:.3f}\n'
        stats_text += f'  Std:    {std_cc:.3f}\n\n'
    if n_es > 0:
        stats_text += f'True ES:\n'
        stats_text += f'  Mean:   {mean_es:.3f}\n'
        stats_text += f'  Median: {median_es:.3f}\n'
        stats_text += f'  Std:    {std_es:.3f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.7))
    
    # Add separation quality metric
    if n_cc > 0 and n_es > 0:
        # Calculate overlap at different thresholds
        separation_text = '\nThreshold Performance:\n'
        for thresh in thresholds:
            es_kept = (es_probs[true_es_mask] >= thresh).sum()
            cc_rejected = (es_probs[true_cc_mask] < thresh).sum()
            es_purity = es_kept / (es_kept + (es_probs[true_cc_mask] >= thresh).sum() + 1e-10)
            separation_text += f'  ≥{thresh:.1f}: ES eff={100*es_kept/n_es:.1f}%, CC rej={100*cc_rejected/n_cc:.1f}%\n'
        
        ax.text(0.98, 0.98, separation_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=0.7))


def plot_energy_analysis(predictions, fig):
    """Page 6: Performance vs energy analysis."""
    # Check if energies exist in predictions
    try:
        energies = predictions['energies']
        if energies is None or (hasattr(energies, 'item') and energies.item() is None):
            raise ValueError("No energy data")
    except (KeyError, ValueError, TypeError):
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Energy data not available\n\nRun: python extract_test_energies.py <results_dir>', 
               ha='center', va='center', fontsize=12)
        ax.axis('off')
        return
    
    y_true = predictions['true_labels'].astype(int)
    y_pred = predictions['predictions'].argmax(axis=1)
    
    # Filter out invalid energies
    valid_mask = (energies > 0) & (energies < 1000)
    energies_valid = energies[valid_mask]
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if len(energies_valid) == 0:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No valid energy data available', 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # Plot 1: Energy distribution by channel
    ax1 = fig.add_subplot(gs[0, :])
    
    n_classes = predictions['predictions'].shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    bins = np.linspace(energies_valid.min(), energies_valid.max(), 25)
    
    for i in range(n_classes):
        mask = (y_true_valid == i)
        if np.sum(mask) > 0:
            label = CHANNEL_LABELS.get(i, f'C{i}')
            ax1.hist(energies_valid[mask], bins=bins, alpha=0.5, 
                    label=label, color=colors[i], edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Neutrino Energy (MeV)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Energy Distribution by True Channel', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, ncol=4, loc='upper right')
    ax1.grid(alpha=0.3, axis='y')
    
    # Plot 2: Accuracy vs energy
    ax2 = fig.add_subplot(gs[1, 0])
    
    n_bins = 15
    energy_bins = np.linspace(energies_valid.min(), energies_valid.max(), n_bins + 1)
    bin_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
    
    accuracies = []
    counts = []
    
    for i in range(n_bins):
        mask = (energies_valid >= energy_bins[i]) & (energies_valid < energy_bins[i+1])
        if np.sum(mask) > 0:
            acc = np.mean(y_pred_valid[mask] == y_true_valid[mask])
            accuracies.append(acc)
            counts.append(np.sum(mask))
        else:
            accuracies.append(np.nan)
            counts.append(0)
    
    ax2.plot(bin_centers, accuracies, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('Neutrino Energy (MeV)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title('Accuracy vs Energy', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Plot 3: Sample counts per bin
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(bin_centers, counts, width=(energy_bins[1]-energy_bins[0])*0.8, 
           alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Neutrino Energy (MeV)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
    ax3.set_title('Samples per Energy Bin', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')


def plot_example_predictions(predictions, fig, results_dir=None):
    """Page 7: Best and worst prediction examples per class with volume images."""
    y_true = predictions['true_labels'].astype(int)
    y_prob = predictions['predictions']
    y_pred = y_prob.argmax(axis=1)
    
    n_classes = y_prob.shape[1]
    
    # Try to load test images if available
    test_images = predictions.get('test_images', None)
    
    # Find best and worst for each class (limit to 4 examples to fit page)
    examples = []
    
    # Pick 4 most common classes
    class_counts = [(i, np.sum(y_true == i)) for i in range(n_classes)]
    class_counts.sort(key=lambda x: x[1], reverse=True)
    top_classes = [c[0] for c in class_counts[:4]]
    
    for class_id in top_classes:
        # Best: correctly predicted with highest confidence
        correct_mask = (y_true == class_id) & (y_pred == class_id)
        if np.sum(correct_mask) > 0:
            best_idx_in_correct = np.argmax(y_prob[correct_mask, class_id])
            best_idx = np.where(correct_mask)[0][best_idx_in_correct]
            examples.append(('Best', class_id, best_idx, 'green'))
        
        # Worst: incorrectly predicted with lowest confidence in true class
        incorrect_mask = (y_true == class_id) & (y_pred != class_id)
        if np.sum(incorrect_mask) > 0:
            worst_idx_in_incorrect = np.argmin(y_prob[incorrect_mask, class_id])
            worst_idx = np.where(incorrect_mask)[0][worst_idx_in_incorrect]
            examples.append(('Worst', class_id, worst_idx, 'red'))
    
    if len(examples) == 0:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No examples available', 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return
    
    # Layout: image on left, probability + details on right
    gs = gridspec.GridSpec(len(examples), 3, figure=fig, hspace=0.5, wspace=0.3,
                          width_ratios=[1.5, 1, 1])
    
    for row, (ex_type, true_class, idx, color) in enumerate(examples):
        probs = y_prob[idx]
        pred_class = y_pred[idx]
        
        # Plot 1: Volume image (if available)
        ax_img = fig.add_subplot(gs[row, 0])
        if test_images is not None and idx < len(test_images):
            img = test_images[idx]
            # Handle different image formats
            if len(img.shape) == 3 and img.shape[-1] == 1:
                img = img[:, :, 0]  # Remove channel dimension
            ax_img.imshow(img, cmap='viridis', aspect='auto')
            ax_img.set_title(f'{ex_type} - True: {CHANNEL_LABELS.get(true_class)}', 
                           fontsize=11, fontweight='bold')
            ax_img.axis('off')
        else:
            ax_img.text(0.5, 0.5, 'Image not\navailable', 
                       ha='center', va='center', fontsize=10)
            ax_img.set_title(f'{ex_type} - True: {CHANNEL_LABELS.get(true_class)}', 
                           fontsize=11, fontweight='bold')
            ax_img.axis('off')
        
        # Plot 2: Probability distribution
        ax_prob = fig.add_subplot(gs[row, 1])
        
        x_pos = np.arange(n_classes)
        labels = [CHANNEL_LABELS.get(i, f'C{i}') for i in range(n_classes)]
        
        bars = ax_prob.bar(x_pos, probs, alpha=0.7, edgecolor='black')
        
        # Color the bars
        for i, bar in enumerate(bars):
            if i == true_class:
                bar.set_facecolor('green')
                bar.set_linewidth(3)
            elif i == pred_class:
                bar.set_facecolor('red')
                bar.set_linewidth(3)
            else:
                bar.set_facecolor('lightgray')
        
        ax_prob.set_ylabel('Probability', fontsize=10, fontweight='bold')
        ax_prob.set_title('Prediction Confidence', fontsize=10, fontweight='bold')
        ax_prob.set_xticks(x_pos)
        ax_prob.set_xticklabels(labels, fontsize=9)
        ax_prob.set_ylim([0, 1])
        ax_prob.grid(alpha=0.3, axis='y')
        ax_prob.axhline(0.5, color='black', linestyle=':', alpha=0.3)
        
        # Plot 3: Details
        ax_info = fig.add_subplot(gs[row, 2])
        ax_info.axis('off')
        
        true_label = CHANNEL_LABELS.get(true_class, f'C{true_class}')
        pred_label = CHANNEL_LABELS.get(pred_class, f'C{pred_class}')
        
        info_text = f"""True: {true_label}
Predicted: {pred_label}

ES prob: {probs[0]:.3f}
CC prob: {probs[1]:.3f}

Confidence: {probs[pred_class]:.3f}

Correct: {'✓ Yes' if pred_class == true_class else '✗ No'}"""
        
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.3, pad=0.5))


def generate_comprehensive_analysis(results_dir, output_pdf=None):
    """Generate comprehensive multi-page PDF analysis."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE CHANNEL TAGGING ANALYSIS")
    print("="*80 + "\n")
    
    # Load results
    print("📊 Loading results...")
    results, predictions = load_results(results_dir)
    results_path = Path(results_dir)

    # Older trainings stored history only in CSV form. Load it if missing so the
    # training history plots can still be generated.
    if 'history' not in results:
        csv_path = results_path / 'training_history.csv'
        if csv_path.exists():
            history = {}
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key, value in row.items():
                        if not key or key.strip().lower() == 'epoch':
                            continue
                        key = key.strip()
                        history.setdefault(key, [])
                        try:
                            history[key].append(float(value))
                        except (TypeError, ValueError):
                            history[key].append(np.nan)
            if history:
                results['history'] = history
    
    if predictions is None:
        print("❌ No predictions found. Run training with save_predictions=True")
        return
    
    # Handle different config structures
    config = results.get('config', {})
    if 'model' in config:
        # New format: nested structure
        model_name = config['model'].get('name', 'ct_model')
    else:
        # Old format: flat structure
        model_name = config.get('model_name', 'ct_model')
    
    print(f"✓ Model: {model_name}")
    print(f"✓ Predictions: {len(predictions['predictions']):,} samples")
    print(f"✓ Classes: {predictions['predictions'].shape[1]}")
    
    # Determine output path
    if output_pdf is None:
        output_pdf = results_path / f"{model_name}_comprehensive_analysis.pdf"
    else:
        output_pdf = Path(output_pdf)
    
    print(f"✓ Output: {output_pdf}\n")
    
    # Create multi-page PDF
    with PdfPages(output_pdf) as pdf:
        # Page 1: Confusion matrix
        print("📈 Generating page 1/9: Confusion Matrix...")
        fig = plt.figure(figsize=(14, 10))
        plot_confusion_matrix(predictions, results, fig)
        fig.suptitle(f'{model_name.upper()} - Confusion Matrix', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Per-class metrics
        print("📈 Generating page 2/9: Per-Class Metrics...")
        fig = plt.figure(figsize=(14, 10))
        plot_per_class_metrics(predictions, fig)
        fig.suptitle(f'{model_name.upper()} - Per-Class Performance', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3: ROC curves
        print("📈 Generating page 3/9: ROC Curves...")
        fig = plt.figure(figsize=(14, 12))
        plot_roc_curves(predictions, fig)
        fig.suptitle(f'{model_name.upper()} - ROC Curves (One-vs-Rest)', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 4: Training history
        print("📈 Generating page 4/9: Training History...")
        fig = plt.figure(figsize=(14, 10))
        plot_training_history(results, fig)
        fig.suptitle(f'{model_name.upper()} - Training History', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 5: Prediction distributions
        print("📈 Generating page 5/9: Prediction Distributions...")
        fig = plt.figure(figsize=(14, 10))
        plot_prediction_distribution(predictions, fig)
        fig.suptitle(f'{model_name.upper()} - Prediction Confidence', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 6: Confusion matrices at different thresholds (NEW!)
        print("�� Generating page 6/9: Confidence Threshold Analysis...")
        fig = plt.figure(figsize=(17, 11))
        plot_confusion_matrices_thresholds(predictions, fig)
        fig.suptitle(f'{model_name.upper()} - Confusion Matrices at Different Confidence Thresholds', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 7: ES probability distribution (NEW!)
        print("📈 Generating page 7/9: ES Probability Distribution...")
        fig = plt.figure(figsize=(14, 10))
        plot_es_probability_distribution(predictions, fig)
        fig.suptitle(f'{model_name.upper()} - ES Prediction Probability Distribution', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 8: Energy analysis
        print("📈 Generating page 8/9: Energy-Dependent Performance...")
        fig = plt.figure(figsize=(14, 10))
        plot_energy_analysis(predictions, fig)
        fig.suptitle(f'{model_name.upper()} - Energy-Dependent Performance', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 9: Example predictions
        print("📈 Generating page 9/9: Example Predictions...")
        fig = plt.figure(figsize=(16, 10))
        plot_example_predictions(predictions, fig, results_dir)
        fig.suptitle(f'{model_name.upper()} - Example Predictions', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    print(f"\n✅ Analysis complete! Saved to: {output_pdf}")
    print("="*80 + "\n")
    return output_pdf


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive CT analysis PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/results/dir
  %(prog)s /path/to/results/dir -o custom_name.pdf
        """
    )
    parser.add_argument('results_dir', help='Path to model results directory')
    parser.add_argument('-o', '--output', help='Output PDF path (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"❌ Directory not found: {args.results_dir}")
        return 1
    
    try:
        generate_comprehensive_analysis(args.results_dir, args.output)
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
