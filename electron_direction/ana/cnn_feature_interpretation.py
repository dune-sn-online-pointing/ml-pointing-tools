#!/usr/bin/env python3
"""
CNN Feature Interpretation for Electron Direction Reconstruction

Techniques to understand what the CNN is learning:
1. Gradient-based saliency maps (what pixels matter most)
2. Integrated gradients (attribution to input features)
3. Occlusion analysis (systematic masking to find important regions)
4. Plane importance (which plane contributes most to direction)
5. Energy-dependent feature importance
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
from pathlib import Path
import json
import argparse
from tqdm import tqdm

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CNNFeatureInterpreter:
    """Interpret what features a CNN is using for electron direction prediction."""
    
    def __init__(self, model_path, load_model=False):
        """Load trained model and predictions."""
        self.model_path = Path(model_path)
        
        # Only load model if explicitly requested (optional for most analyses)
        self.model = None
        if load_model:
            model_file = self.model_path / 'best_model.keras'
            if not model_file.exists():
                # Try checkpoints directory
                model_file = self.model_path / 'checkpoints' / 'best_model.keras'
            
            if model_file.exists():
                self.model = tf.keras.models.load_model(
                    model_file,
                    custom_objects={'angular_loss': self._angular_loss}
                )
                print(f"✓ Loaded model from: {model_file}")
            else:
                print(f"⚠️  Model file not found, skipping model-dependent analyses")
        
        # Load predictions and validation data
        pred_data = np.load(self.model_path / 'val_predictions.npz')
        self.predictions = pred_data['predictions']
        self.true_directions = pred_data['true_directions']
        self.energies = pred_data['energies']
        self.angular_errors = pred_data['angular_errors']
        
        # Load configuration
        with open(self.model_path / 'results.json', 'r') as f:
            self.config = json.load(f)['config']
        
        print(f"Loaded model: {self.model_path.name}")
        print(f"  Validation samples: {len(self.predictions)}")
        model_cfg = self.config.get('model', {}) if isinstance(self.config, dict) else {}
        arch_desc = (
            model_cfg.get('architecture')
            or model_cfg.get('name')
            or f"input_shape={model_cfg.get('input_shape', 'unknown')}"
        )
        print(f"  Model description: {arch_desc}")
    
    @staticmethod
    def _angular_loss(y_true, y_pred):
        """Custom angular loss function."""
        y_pred = tf.nn.l2_normalize(y_pred, axis=-1)
        cosine_sim = tf.reduce_sum(y_true * y_pred, axis=-1)
        cosine_sim = tf.clip_by_value(cosine_sim, -1.0, 1.0)
        return tf.acos(cosine_sim)
    
    def compute_saliency_maps(self, n_samples=100, sample_type='random'):
        """
        Compute gradient-based saliency maps.
        
        Shows which input pixels have the largest gradient with respect to output.
        High gradient = small change in pixel causes large change in prediction.
        
        Args:
            n_samples: Number of samples to analyze
            sample_type: 'random', 'best', or 'worst'
        """
        print("\n" + "="*80)
        print("COMPUTING SALIENCY MAPS")
        print("="*80)
        
        # Select samples
        indices = self._select_samples(n_samples, sample_type)
        
        # We need to reload the actual images (not stored in predictions file)
        # This is a limitation - we'd need to modify the prediction saving to include images
        print("⚠️  Note: Saliency maps require access to validation images")
        print("    Predictions file only contains directions, not input images")
        print("    Need to modify training script to save validation data")
        
        return None
    
    def analyze_plane_importance(self, n_samples=1000):
        """
        Analyze which plane (U, V, X) contributes most to direction prediction.
        
        Method: Zero out each plane individually and measure performance drop.
        Larger performance drop = more important plane.
        """
        print("\n" + "="*80)
        print("ANALYZING PLANE IMPORTANCE")
        print("="*80)
        
        print("⚠️  Note: Requires access to validation images")
        print("    This analysis needs the 3-plane input data structure")
        
        # Conceptual approach (needs implementation with actual data):
        # 1. Load validation images (U, V, X planes)
        # 2. For each sample:
        #    - Get baseline prediction with all planes
        #    - Get prediction with U plane = 0
        #    - Get prediction with V plane = 0
        #    - Get prediction with X plane = 0
        # 3. Compute angular error increase for each masking
        # 4. Average across samples
        
        return None
    
    def occlusion_analysis(self, patch_size=8, stride=4):
        """
        Systematically mask regions of input and measure impact on prediction.
        
        Creates a 2D heatmap showing which spatial regions are most important.
        """
        print("\n" + "="*80)
        print("OCCLUSION ANALYSIS")
        print("="*80)
        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Stride: {stride}")
        
        print("⚠️  Note: Requires access to validation images")
        
        return None
    
    def analyze_energy_dependence(self):
        """
        Analyze how model performance varies with energy.
        
        This can reveal if certain energy ranges have better features.
        """
        print("\n" + "="*80)
        print("ENERGY-DEPENDENT PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Energy bins
        energy_bins = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 
                               22, 24, 26, 28, 30, 35, 40, 50, 70])
        
        bin_centers = []
        median_errors = []
        mean_cosines = []
        n_samples = []
        
        for i in range(len(energy_bins) - 1):
            e_min, e_max = energy_bins[i], energy_bins[i+1]
            mask = (self.energies >= e_min) & (self.energies < e_max)
            
            if np.sum(mask) > 0:
                bin_centers.append((e_min + e_max) / 2)
                median_errors.append(np.median(self.angular_errors[mask]))
                
                # Compute cosine similarity
                cosines = np.sum(self.predictions[mask] * self.true_directions[mask], axis=1)
                mean_cosines.append(np.mean(cosines))
                n_samples.append(np.sum(mask))
        
        # Print results
        print("\nEnergy Bin | Median Error | Mean Cosine | N Samples")
        print("-" * 60)
        for ec, me, mc, ns in zip(bin_centers, median_errors, mean_cosines, n_samples):
            print(f"{ec:6.1f} MeV | {me:11.2f}° | {mc:11.3f} | {ns:8d}")
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(bin_centers, median_errors, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Energy (MeV)', fontsize=12)
        ax1.set_ylabel('Median Angular Error (°)', fontsize=12)
        ax1.set_title('Performance vs Energy', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        ax2.plot(bin_centers, mean_cosines, 'o-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Energy (MeV)', fontsize=12)
        ax2.set_ylabel('Mean Cosine Similarity', fontsize=12)
        ax2.set_title('Direction Alignment vs Energy', fontsize=14, fontweight='bold')
        ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        return fig, {
            'energy_bins': bin_centers,
            'median_errors': median_errors,
            'mean_cosines': mean_cosines,
            'n_samples': n_samples
        }
    
    def analyze_error_modes(self, n_bins=20):
        """
        Analyze the distribution of errors and identify patterns.
        
        - Angular error distribution
        - Cosine similarity distribution
        - Correlation between error and event properties
        """
        print("\n" + "="*80)
        print("ERROR MODE ANALYSIS")
        print("="*80)
        
        # Compute cosine similarities
        cosines = np.sum(self.predictions * self.true_directions, axis=1)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Angular error distribution
        ax = axes[0, 0]
        ax.hist(self.angular_errors, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.median(self.angular_errors), color='r', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(self.angular_errors):.1f}°')
        ax.axvline(np.mean(self.angular_errors), color='g', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(self.angular_errors):.1f}°')
        ax.set_xlabel('Angular Error (°)', fontsize=12)
        ax.set_ylabel('Counts', fontsize=12)
        ax.set_title('Angular Error Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Cosine similarity distribution
        ax = axes[0, 1]
        ax.hist(cosines, bins=50, alpha=0.7, edgecolor='black', color='green')
        ax.axvline(1, color='g', linestyle='--', alpha=0.5, label='Perfect (+1)')
        ax.axvline(-1, color='r', linestyle='--', alpha=0.5, label='Reversed (-1)')
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5, label='Orthogonal (0)')
        ax.set_xlabel('Cosine Similarity', fontsize=12)
        ax.set_ylabel('Counts', fontsize=12)
        ax.set_title('Cosine Similarity Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Print statistics
        print(f"\nAngular Error Statistics:")
        print(f"  Mean:   {np.mean(self.angular_errors):.2f}°")
        print(f"  Median: {np.median(self.angular_errors):.2f}°")
        print(f"  Std:    {np.std(self.angular_errors):.2f}°")
        print(f"  25th:   {np.percentile(self.angular_errors, 25):.2f}°")
        print(f"  75th:   {np.percentile(self.angular_errors, 75):.2f}°")
        
        print(f"\nCosine Similarity Statistics:")
        print(f"  Mean:   {np.mean(cosines):.4f}")
        print(f"  Median: {np.median(cosines):.4f}")
        print(f"  Std:    {np.std(cosines):.4f}")
        
        print(f"\nDirectional Categories:")
        excellent = np.sum(cosines > 0.9)
        good = np.sum((cosines > 0.7) & (cosines <= 0.9))
        mediocre = np.sum((cosines > -0.7) & (cosines <= 0.7))
        reversed = np.sum(cosines <= -0.7)
        
        print(f"  Excellent (cos > 0.9):  {excellent:6d} ({100*excellent/len(cosines):5.2f}%)")
        print(f"  Good (0.7 < cos ≤ 0.9): {good:6d} ({100*good/len(cosines):5.2f}%)")
        print(f"  Mediocre (±0.7):        {mediocre:6d} ({100*mediocre/len(cosines):5.2f}%)")
        print(f"  Reversed (cos ≤ -0.7):  {reversed:6d} ({100*reversed/len(cosines):5.2f}%)")
        
        # 3. Error vs Energy
        ax = axes[1, 0]
        ax.hexbin(self.energies, self.angular_errors, gridsize=30, cmap='YlOrRd', mincnt=1)
        ax.set_xlabel('Energy (MeV)', fontsize=12)
        ax.set_ylabel('Angular Error (°)', fontsize=12)
        ax.set_title('Error vs Energy', fontsize=14, fontweight='bold')
        plt.colorbar(ax.collections[0], ax=ax, label='Counts')
        
        # 4. Cosine vs Energy
        ax = axes[1, 1]
        ax.hexbin(self.energies, cosines, gridsize=30, cmap='RdYlGn', mincnt=1)
        ax.set_xlabel('Energy (MeV)', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title('Alignment vs Energy', fontsize=14, fontweight='bold')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        plt.colorbar(ax.collections[0], ax=ax, label='Counts')
        
        plt.tight_layout()
        
        return fig
    
    def analyze_component_predictions(self):
        """
        Analyze predictions in terms of direction components (x, y, z).
        
        Shows if model has biases toward certain directions.
        """
        print("\n" + "="*80)
        print("COMPONENT PREDICTION ANALYSIS")
        print("="*80)
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        components = ['X', 'Y', 'Z']
        
        for i, comp in enumerate(components):
            # True vs Predicted scatter
            ax = axes[0, i]
            ax.hexbin(self.true_directions[:, i], self.predictions[:, i], 
                     gridsize=40, cmap='Blues', mincnt=1)
            ax.plot([-1, 1], [-1, 1], 'r--', alpha=0.5, linewidth=2)
            ax.set_xlabel(f'True {comp}', fontsize=12)
            ax.set_ylabel(f'Predicted {comp}', fontsize=12)
            ax.set_title(f'{comp} Component: True vs Predicted', fontsize=12, fontweight='bold')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.grid(alpha=0.3)
            
            # Residuals
            ax = axes[1, i]
            residuals = self.predictions[:, i] - self.true_directions[:, i]
            ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='r', linestyle='--', linewidth=2)
            ax.axvline(np.mean(residuals), color='g', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(residuals):.3f}')
            ax.set_xlabel(f'{comp} Residual (Pred - True)', fontsize=12)
            ax.set_ylabel('Counts', fontsize=12)
            ax.set_title(f'{comp} Component Residuals', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Print statistics
            print(f"\n{comp} Component:")
            print(f"  Mean residual: {np.mean(residuals):7.4f}")
            print(f"  RMS residual:  {np.sqrt(np.mean(residuals**2)):7.4f}")
            print(f"  Std residual:  {np.std(residuals):7.4f}")
        
        plt.tight_layout()
        return fig
    
    def _select_samples(self, n_samples, sample_type):
        """Select samples based on type."""
        if sample_type == 'best':
            # Smallest angular errors
            indices = np.argsort(self.angular_errors)[:n_samples]
        elif sample_type == 'worst':
            # Largest angular errors
            indices = np.argsort(self.angular_errors)[-n_samples:]
        else:  # random
            indices = np.random.choice(len(self.angular_errors), n_samples, replace=False)
        
        return indices
    
    def generate_report(self, output_file):
        """Generate comprehensive interpretation report."""
        print("\n" + "="*80)
        print(f"GENERATING INTERPRETATION REPORT: {output_file}")
        print("="*80)
        
        with PdfPages(output_file) as pdf:
            # Page 1: Energy-dependent performance
            fig, _ = self.analyze_energy_dependence()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 2: Error modes
            fig = self.analyze_error_modes()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 3: Component predictions
            fig = self.analyze_component_predictions()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        print(f"\n✅ Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Interpret CNN features for electron direction reconstruction"
    )
    parser.add_argument('model_dir', type=str,
                       help='Path to model directory (contains best_model.keras and val_predictions.npz)')
    parser.add_argument('-o', '--output', type=str, default='cnn_interpretation_report.pdf',
                       help='Output PDF file name')
    
    args = parser.parse_args()
    
    # Create interpreter
    interpreter = CNNFeatureInterpreter(args.model_dir)
    
    # Generate report
    interpreter.generate_report(args.output)
    
    print("\n" + "="*80)
    print("INTERPRETATION COMPLETE")
    print("="*80)
    print("\nNote: Full feature interpretation requires:")
    print("  1. Access to validation images (not just predictions)")
    print("  2. Saliency map computation (needs gradient backprop)")
    print("  3. Occlusion analysis (systematic masking)")
    print("  4. Plane importance analysis (ablation studies)")
    print("\nCurrent analysis provides:")
    print("  ✓ Energy-dependent performance trends")
    print("  ✓ Error mode characterization")
    print("  ✓ Component-wise prediction analysis")
    print("  ✓ Cosine similarity distributions")


if __name__ == '__main__':
    main()
