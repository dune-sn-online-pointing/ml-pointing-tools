#!/usr/bin/env python3
"""
Comprehensive Channel Tagging Analysis - Version 2
Updated with improved visualizations (Nov 24, 2025)
"""

# Import the original script's functions
import sys
sys.path.insert(0, '/afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/channel_tagging/ana')
sys.path.insert(0, '/tmp')

from comprehensive_ct_analysis import *

# Import improved plotting functions
from improved_plot_functions import plot_roc_curves_v2, plot_es_probability_distribution_v2

# Import additional needed for energy estimation  
from pathlib import Path

def estimate_energy_from_image(predictions, results_dir):
    """Estimate energy from image ADC values when energy metadata is missing."""
    try:
        results_path = Path(results_dir)
        
        # Look for test data NPZ files in the model directory
        test_data_files = []
        for pattern in ['*test*.npz', '*predictions*.npz']:
            test_data_files.extend(list(results_path.glob(pattern)))
        
        if not test_data_files:
            return None
        
        # Load and find image data
        test_data = np.load(test_data_files[0], allow_pickle=True)
        image_keys = [k for k in test_data.keys() if 'image' in k.lower() or k.upper() in ['X', 'DATA']]
        
        if not image_keys:
            return None
        
        images = test_data[image_keys[0]]
        
        # Sum pixels and convert to MeV
        if len(images.shape) == 3:
            total_adc = images.sum(axis=(1, 2))
        elif len(images.shape) == 4:
            total_adc = images.sum(axis=(1, 2, 3))
        else:
            return None
        
        energies = total_adc / 3600.0
        return energies
        
    except Exception as e:
        return None


# Override generate_comprehensive_analysis to use new page order
def generate_comprehensive_analysis_v2(results_dir, output_pdf=None):
    """Generate comprehensive multi-page PDF analysis - V2 with updated page order."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE CHANNEL TAGGING ANALYSIS - V2")
    print("="*80 + "\n")
    
    # Load results
    print("📊 Loading results...")
    results, predictions = load_results(results_dir)
    results_path = Path(results_dir)

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
        print("❌ No predictions found")
        return
    
    config = results.get('config', {})
    if 'model' in config:
        model_name = config['model'].get('name', 'ct_model')
    else:
        model_name = config.get('model_name', 'ct_model')
    
    print(f"✓ Model: {model_name}")
    print(f"✓ Predictions: {len(predictions['predictions']):,} samples")
    print(f"✓ Classes: {predictions['predictions'].shape[1]}")
    
    if output_pdf is None:
        output_pdf = results_path / f'{model_name}_comprehensive_analysis_v2.pdf'
    else:
        output_pdf = Path(output_pdf)
    
    print(f"✓ Output: {output_pdf}\n")
    
    with PdfPages(output_pdf) as pdf:
        # Page 1: Confusion matrices at thresholds (WAS PAGE 6, NOW FIRST!)
        print("�� Generating page 1/7: Confusion Matrices at Thresholds...")
        fig = plt.figure(figsize=(17, 11))
        plot_confusion_matrices_thresholds(predictions, fig)
        fig.suptitle(f'{model_name.upper()} - Confusion Matrices at Confidence Thresholds', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: ROC curves - BIGGER (WAS PAGE 3)
        print("📈 Generating page 2/7: ROC Curves...")
        fig = plt.figure(figsize=(16, 12))
        plot_roc_curves_v2(predictions, fig)
        fig.suptitle(f'{model_name.upper()} - ROC Curves (One-vs-Rest)', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3: Training history (WAS PAGE 4)
        print("📈 Generating page 3/7: Training History...")
        fig = plt.figure(figsize=(14, 10))
        plot_training_history(results, fig)
        fig.suptitle(f'{model_name.upper()} - Training History', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 4: Prediction distributions (WAS PAGE 5)
        print("📈 Generating page 4/7: Prediction Distributions...")
        fig = plt.figure(figsize=(14, 10))
        plot_prediction_distribution(predictions, fig)
        fig.suptitle(f'{model_name.upper()} - Prediction Confidence', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 5: ES probability distribution - FIXED OVERLAPS (WAS PAGE 7)
        print("📈 Generating page 5/7: ES Probability Distribution...")
        fig = plt.figure(figsize=(15, 10))
        plot_es_probability_distribution_v2(predictions, fig)
        fig.suptitle(f'{model_name.upper()} - ES Prediction Probability Distribution', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 6: Energy analysis WITH ADC ESTIMATION (WAS PAGE 8)
        print("📈 Generating page 6/7: Energy-Dependent Performance...")
        
        # Try to estimate energy from images if not available
        if 'energies' not in predictions or predictions.get('energies') is None:
            print("  ⚙️  Estimating energy from ADC values...")
            energies = estimate_energy_from_image(predictions, results_dir)
            if energies is not None:
                predictions = dict(predictions)
                predictions['energies'] = energies
                print(f"  ✓ Estimated energies (range: {energies.min():.1f}-{energies.max():.1f} MeV)")
        
        fig = plt.figure(figsize=(14, 10))
        plot_energy_analysis(predictions, fig)
        fig.suptitle(f'{model_name.upper()} - Energy-Dependent Performance', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 7: Example predictions (WAS PAGE 9)
        print("📈 Generating page 7/7: Example Predictions...")
        fig = plt.figure(figsize=(16, 10))
        plot_example_predictions(predictions, fig, results_dir)
        fig.suptitle(f'{model_name.upper()} - Example Predictions', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    print(f"\n✅ Analysis complete! Saved to: {output_pdf}")
    print("="*80 + "\n")
    
    return output_pdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate comprehensive CT analysis PDF - V2')
    parser.add_argument('results_dir', help='Path to model results directory')
    parser.add_argument('-o', '--output', help='Output PDF path (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"❌ Directory not found: {args.results_dir}")
        sys.exit(1)
    
    try:
        generate_comprehensive_analysis_v2(args.results_dir, args.output)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
