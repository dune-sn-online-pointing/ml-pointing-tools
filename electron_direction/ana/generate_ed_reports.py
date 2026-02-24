#!/usr/bin/env python3
"""
Generate comprehensive ED PDF reports for training outputs.
Adapts the output format to work with comprehensive_ed_analysis.py
"""

import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
import subprocess

def prepare_results_for_analysis(training_dir):
    """
    Prepare training outputs for comprehensive analysis.
    Converts metrics.json -> results.json and test_predictions.npz -> val_predictions.npz
    """
    training_path = Path(training_dir)
    
    # Check for required files
    metrics_path = training_path / 'metrics.json'
    test_pred_path = training_path / 'test_predictions.npz'
    
    if not metrics_path.exists():
        print(f"❌ No metrics.json found in {training_dir}")
        return False
    
    if not test_pred_path.exists():
        print(f"❌ No test_predictions.npz found in {training_dir}")
        return False
    
    # Load metrics and convert to results format
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Load config if available
    config_path = training_path / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {'model_name': training_path.name}
    
    # Calculate std from test predictions if available
    angular_errors = None
    if test_pred_path.exists():
        pred_data = np.load(test_pred_path, allow_pickle=True)
        if 'angular_errors' in pred_data:
            angular_errors = pred_data['angular_errors']
        elif 'predictions' in pred_data and 'true_directions' in pred_data:
            # Calculate angular errors
            preds = pred_data['predictions']
            true_dirs = pred_data['true_directions']
            # Normalize
            preds_norm = preds / np.linalg.norm(preds, axis=1, keepdims=True)
            true_norm = true_dirs / np.linalg.norm(true_dirs, axis=1, keepdims=True)
            # Dot product
            dot_products = np.sum(preds_norm * true_norm, axis=1)
            dot_products = np.clip(dot_products, -1.0, 1.0)
            # Angular error
            angular_errors = np.degrees(np.arccos(dot_products))
    
    angular_error_std = float(np.std(angular_errors)) if angular_errors is not None else 0.0
    
    # Create minimal history (not available from saved files)
    # Just include final metrics as single-epoch history
    history = {
        'loss': [float(metrics.get('test_loss', 0))],
        'val_loss': [float(metrics.get('test_loss', 0))],
        'mae': [float(metrics.get('test_mae', 0))],
        'val_mae': [float(metrics.get('test_mae', 0))]
    }
    
    # Create results.json in expected format
    results = {
        'val_loss': float(metrics.get('test_loss', 0)),
        'val_mae': float(metrics.get('test_mae', 0)),
        'angular_error_mean': float(metrics.get('mean_angular_error', 0)),
        'angular_error_median': float(metrics.get('median_angular_error', 0)),
        'angular_error_std': angular_error_std,
        'angular_error_68th': float(metrics.get('angular_error_68th', 0)),
        'angular_error_25th': float(metrics.get('angular_error_25th', 0)),
        'angular_error_75th': float(metrics.get('angular_error_75th', 0)),
        'history': history,
        'config': config
    }
    
    results_path = training_path / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Created results.json")
    
    # Load test predictions and create symlink as val_predictions
    val_pred_path = training_path / 'val_predictions.npz'
    
    # If val_predictions already exists, remove it
    if val_pred_path.exists() or val_pred_path.is_symlink():
        val_pred_path.unlink()
    
    # Create symlink
    val_pred_path.symlink_to('test_predictions.npz')
    print(f"✓ Created val_predictions.npz symlink")
    
    return True


def generate_report(training_dir):
    """Generate comprehensive PDF report."""
    training_path = Path(training_dir)
    
    print(f"\n{'='*70}")
    print(f"Generating report for: {training_path.name}")
    print(f"{'='*70}\n")
    
    # Prepare results
    if not prepare_results_for_analysis(training_dir):
        return False
    
    # Run comprehensive analysis
    analysis_script = Path(__file__).parent / 'comprehensive_ed_analysis.py'
    
    if not analysis_script.exists():
        # Try alternative location
        analysis_script = Path('/afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/electron_direction/ana/comprehensive_ed_analysis.py')
    
    if not analysis_script.exists():
        print(f"❌ Cannot find comprehensive_ed_analysis.py")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(analysis_script), str(training_dir)],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running analysis: {e}")
        print(e.stdout)
        print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate ED comprehensive reports')
    parser.add_argument('training_dirs', nargs='+', help='Training output directories')
    args = parser.parse_args()
    
    success_count = 0
    fail_count = 0
    
    for training_dir in args.training_dirs:
        if generate_report(training_dir):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: {success_count} succeeded, {fail_count} failed")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
