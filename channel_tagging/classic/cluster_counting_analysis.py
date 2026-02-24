#!/usr/bin/env python3
"""
Classic (non-ML) approach for Channel Tagging based on cluster counting.
Count total clusters within 50cm radius across U, V, X planes for ES vs CC discrimination.
"""

import sys
import os
import numpy as np
import uproot
import matplotlib.pyplot as plt
from pathlib import Path

# Conversion factors
MM_PER_TICK = 0.8      # mm per tick (time dimension)
MM_PER_CHANNEL = 4.5   # mm per channel (spatial dimension)
CM_PER_TICK = MM_PER_TICK / 10.0      # cm per tick
CM_PER_CHANNEL = MM_PER_CHANNEL / 10.0  # cm per channel

def count_nearby_clusters(file_obj, radius_cm=50):
    """
    Count clusters within radius_cm of each other across all three planes.
    For each match_id, collect all cluster positions from U, V, X planes,
    then for each cluster count how many others are within the radius.
    Returns average count per match_id.
    
    Args:
        file_obj: uproot file object with separate trees per plane
        radius_cm: Radius in cm to search for nearby clusters
        
    Returns:
        List of average nearby cluster counts (one per unique match_id)
    """
    planes = ['U', 'V', 'X']
    
    # Collect all unique match_ids
    all_match_ids = set()
    for plane in planes:
        tree_name = f'clusters/clusters_tree_{plane}'
        if tree_name not in file_obj:
            continue
        tree = file_obj[tree_name]
        match_ids = tree['match_id'].array(library='np')
        all_match_ids.update(match_ids)
    
    counts = []
    
    # For each match_id, collect all cluster positions from all planes
    for match_id in all_match_ids:
        all_positions = []  # List of (time_cm, channel_cm) tuples
        
        for plane in planes:
            tree_name = f'clusters/clusters_tree_{plane}'
            if tree_name not in file_obj:
                continue
            
            tree = file_obj[tree_name]
            arrays = tree.arrays(['match_id', 'tp_time_start', 'tp_detector_channel'], library='np')
            
            # Find entries for this match_id
            for i in range(len(arrays['match_id'])):
                if arrays['match_id'][i] == match_id:
                    times = arrays['tp_time_start'][i]
                    channels = arrays['tp_detector_channel'][i]
                    
                    # Convert each cluster position to cm
                    for t, ch in zip(times, channels):
                        time_cm = t * CM_PER_TICK
                        channel_cm = ch * CM_PER_CHANNEL
                        all_positions.append((time_cm, channel_cm))
        
        if len(all_positions) < 2:
            continue
        
        # For each cluster, count how many others are within radius_cm
        nearby_counts = []
        for i, (t1, ch1) in enumerate(all_positions):
            count_within_radius = 0
            for j, (t2, ch2) in enumerate(all_positions):
                if i == j:
                    continue
                # Calculate 2D distance
                dist = np.sqrt((t1 - t2)**2 + (ch1 - ch2)**2)
                if dist <= radius_cm:
                    count_within_radius += 1
            nearby_counts.append(count_within_radius)
        
        # Average count for this match_id
        if nearby_counts:
            counts.append(np.mean(nearby_counts))
    
    return np.array(counts)


def main():
    print("=" * 70)
    print("CLASSIC CHANNEL TAGGING: Cluster Counting Analysis")
    print("=" * 70)
    
    # Paths
    es_dir = Path("/eos/user/e/evilla/dune/sn-tps/prod_es/es_production_matched_clusters_tick3_ch2_min2_tot3_e2p0")
    cc_dir = Path("/eos/user/e/evilla/dune/sn-tps/prod_cc/cc_production_matched_clusters_tick3_ch2_min2_tot3_e2p0")
    
    output_dir = Path("/afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/channel_tagging/classic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get files (100 per class for better statistics)
    es_files = sorted(list(es_dir.glob("*.root")))[:100]
    cc_files = sorted(list(cc_dir.glob("*.root")))[:100]
    
    print(f"\nFound {len(es_files)} ES files")
    print(f"Found {len(cc_files)} CC files")
    
    # Process ES files
    print("\nProcessing ES files...")
    es_counts = []
    for i, file_path in enumerate(es_files):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(es_files)} ES files")
        try:
            with uproot.open(file_path) as f:
                counts = count_nearby_clusters(f, radius_cm=50)
                es_counts.extend(counts)
        except Exception as e:
            print(f"  Warning: Error processing {file_path.name}: {e}")
            continue
    
    es_counts = np.array(es_counts)
    print(f"Total ES events: {len(es_counts)}")
    
    # Process CC files
    print("\nProcessing CC files...")
    cc_counts = []
    for i, file_path in enumerate(cc_files):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(cc_files)} CC files")
        try:
            with uproot.open(file_path) as f:
                counts = count_nearby_clusters(f, radius_cm=50)
                cc_counts.extend(counts)
        except Exception as e:
            print(f"  Warning: Error processing {file_path.name}: {e}")
            continue
    
    cc_counts = np.array(cc_counts)
    print(f"Total CC events: {len(cc_counts)}")
    
    # Statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"ES - Mean: {np.mean(es_counts):.2f}, Median: {np.median(es_counts):.2f}, "
          f"Std: {np.std(es_counts):.2f}, Min: {np.min(es_counts)}, Max: {np.max(es_counts)}")
    print(f"CC - Mean: {np.mean(cc_counts):.2f}, Median: {np.median(cc_counts):.2f}, "
          f"Std: {np.std(cc_counts):.2f}, Min: {np.min(cc_counts)}, Max: {np.max(cc_counts)}")
    
    # Create plots (0-20 range)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ========== CLUSTER COUNTS ==========
    # Histogram with range 0-20
    bins = np.linspace(0, 20, 41)  # 0.5 cluster bins up to 20
    
    axes[0].hist(es_counts, bins=bins, alpha=0.6, label=f'ES (n={len(es_counts)})', 
                 color='blue', density=True)
    axes[0].hist(cc_counts, bins=bins, alpha=0.6, label=f'CC (n={len(cc_counts)})', 
                 color='red', density=True)
    axes[0].set_xlabel('Total Clusters (U+V+X)')
    axes[0].set_ylabel('Normalized Frequency')
    axes[0].set_title('Cluster Count Distribution (Linear, 0-20)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Log scale histogram
    axes[1].hist(es_counts, bins=bins, alpha=0.6, label=f'ES (n={len(es_counts)})', 
                 color='blue', density=True)
    axes[1].hist(cc_counts, bins=bins, alpha=0.6, label=f'CC (n={len(cc_counts)})', 
                 color='red', density=True)
    axes[1].set_xlabel('Total Clusters (U+V+X)')
    axes[1].set_ylabel('Normalized Frequency (log)')
    axes[1].set_title('Cluster Count Distribution (Log Scale, 0-20)')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Cumulative distribution
    es_sorted = np.sort(es_counts)
    cc_sorted = np.sort(cc_counts)
    es_cdf = np.arange(1, len(es_sorted) + 1) / len(es_sorted)
    cc_cdf = np.arange(1, len(cc_sorted) + 1) / len(cc_sorted)
    
    axes[2].plot(es_sorted, es_cdf, label='ES', color='blue', linewidth=2)
    axes[2].plot(cc_sorted, cc_cdf, label='CC', color='red', linewidth=2)
    axes[2].set_xlabel('Total Clusters')
    axes[2].set_ylabel('Cumulative Probability')
    axes[2].set_title('CDF - Cluster Counts')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 20)
    
    plt.tight_layout()
    output_file = output_dir / "cluster_counting_es_vs_cc.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Calculate separation metrics
    from scipy.stats import ks_2samp
    
    # Cluster counts
    ks_stat, ks_pval = ks_2samp(es_counts, cc_counts)
    print(f"\nKolmogorov-Smirnov test:")
    print(f"  Statistic: {ks_stat:.4f}")
    print(f"  p-value: {ks_pval:.4e}")
    
    # Simple threshold analysis for counts
    thresholds = np.linspace(np.min([es_counts.min(), cc_counts.min()]), 
                             np.max([es_counts.max(), cc_counts.max()]), 100)
    
    best_accuracy = 0
    best_threshold = 0
    
    for thresh in thresholds:
        es_correct = np.sum(es_counts <= thresh)  # ES should have fewer clusters
        cc_correct = np.sum(cc_counts > thresh)   # CC should have more clusters
        accuracy = (es_correct + cc_correct) / (len(es_counts) + len(cc_counts))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh
    
    print(f"\nBest threshold: {best_threshold:.1f} clusters")
    print(f"Best accuracy: {best_accuracy*100:.2f}%")
    
    # Save results
    results = {
        'es_mean': float(np.mean(es_counts)),
        'es_median': float(np.median(es_counts)),
        'es_std': float(np.std(es_counts)),
        'cc_mean': float(np.mean(cc_counts)),
        'cc_median': float(np.median(cc_counts)),
        'cc_std': float(np.std(cc_counts)),
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'best_threshold': float(best_threshold),
        'best_accuracy': float(best_accuracy),
        'n_es_events': len(es_counts),
        'n_cc_events': len(cc_counts)
    }
    
    import json
    results_file = output_dir / "cluster_counting_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
