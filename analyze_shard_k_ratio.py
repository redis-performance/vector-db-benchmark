#!/usr/bin/env python3
"""
Script to analyze shard_k_ratio feature impact on performance and recall.

This script generates graphs showing the relationship between shard_k_ratio values
and performance/recall metrics for different database configurations.
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import math
# Configuration
RESULTS_DIR = "/home/ubuntu/vector-db-benchmark-1/results/final"
OUTPUT_DIR = "./graphs"

def find_summary_files() -> List[str]:
    """Find all summary JSON files in the results directory."""
    pattern = os.path.join(RESULTS_DIR, "*-summary.json")
    files = glob.glob(pattern)
    print(f"Found {len(files)} summary files:")
    for file in files:
        print(f"  - {os.path.basename(file)}")
    return files

def get_wrokers_from_filename(filename: str) -> int:
    """
    Parse filename to extract workers.
    Expected format: *workers-*-summary.json
    """
    basename = os.path.basename(filename)

    # Check if it's the new format with workers
    if "workers_" in basename:
        # New format: ...k_ratio_1.0-workers_8-summary.json
        workers_part = basename.split("workers_")[1]  # "8-summary.json"
        workers = workers_part.split("-")[0]  # "8"
    else:
        # Old format: ...k_ratio_1.0-summary.json
        workers = "8"  # or "unknown" - your choice for default

    # Remove -summary.json suffix
    name_parts = basename.replace("-summary.json", "").split("-")

    return int(workers)

def load_benchmark_data(file_path: str) -> Dict:
    """Load and parse a benchmark summary JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def calculate_effective_k(k_ratio, shard_count, k):
    num_shards = shard_count
    k_ratio_config = k_ratio  # The configured k_ratio

    k_min = math.ceil(k / num_shards)
    k_per_shard_req = math.ceil(k * k_ratio_config)
    effective_k = max(k_min, k_per_shard_req)

    return effective_k

def calculatr_actual_k_ratio(k_ratio, shard_count, k):
    effective_k = calculate_effective_k(k_ratio, shard_count, k)
    k_ratio_actual = effective_k / k

    return k_ratio_actual

def extract_summary_data(data: Dict, filename: str) -> Optional[Dict]:
    """Extract summary data from a benchmark JSON file."""
    if 'precision_summary' not in data or 'precision' not in data:
        print(f"Warning: No precision_summary or precision found in {filename}")
        return None

    # Get k_ratio and shard_count from the precision section
    precision_data = data['precision']
    if not precision_data:
        print(f"Warning: Empty precision section in {filename}")
        return None

    # Get the first precision entry (they should all have the same k_ratio and shard_count)
    first_precision_key = list(precision_data.keys())[0]
    config = precision_data[first_precision_key].get('config', {})

    k_ratio = config.get('k_ratio', 'unknown')
    shard_count = config.get('shard_count', 'unknown')
    workers = get_wrokers_from_filename(filename)

    if k_ratio == 'unknown' or shard_count == 'unknown':
        print(f"Warning: Missing k_ratio or shard_count in {filename}")
        print(f"  Found k_ratio: {k_ratio}, shard_count: {shard_count}")
        return None

    # Get K value from the search section (since it's not in config)
    k_value = 'unknown'
    if 'search' in data and data['search']:
        # Get the first search experiment to extract the 'top' value
        first_search_key = list(data['search'].keys())[0]
        search_params = data['search'][first_search_key].get('params', {})
        k_value = search_params.get('top', 'unknown')

    # Extract performance data from precision_summary
    precision_summary = data['precision_summary']

    # Get the best precision point (highest precision available)
    best_precision = max(precision_summary.keys(), key=float)
    best_data = precision_summary[best_precision]

    return {
        'filename': filename,
        'k_ratio': calculatr_actual_k_ratio(k_ratio, shard_count, k_value),
        'shard_count': shard_count,
        'workers': workers,
        'k_value': k_value,
        'precision': float(best_precision),
        'qps': best_data.get('qps', 0),
        'p50': best_data.get('p50', 0),
        'p95': best_data.get('p95', 0),
    }

def organize_data_by_k_and_shards(summaries: List[Dict]) -> Dict[tuple, List[Dict]]:
    """Organize summary data by (k_value, shard_count) combination."""
    by_k_and_shards = {}

    for summary in summaries:
        k_value = summary['k_value']
        shard_count = summary['shard_count']
        workers = summary['workers']
        key = (k_value, shard_count, workers)

        if key not in by_k_and_shards:
            by_k_and_shards[key] = []
        by_k_and_shards[key].append(summary)

    return by_k_and_shards

def create_graph_for_shard_count(shard_count: int, summaries: List[Dict], output_dir: str):
    """Create a graph for a specific shard count showing ratio vs performance/recall."""
    if not summaries:
        print(f"No data for shard count {shard_count}")
        return

    # Sort by k_ratio for proper line plotting
    summaries = sorted(summaries, key=lambda x: x['k_ratio'])

    # Extract data for plotting
    effective_k_values = [calculate_effective_k(s['k_ratio'], shard_count, s['k_value']) for s in summaries]
    ratios = [s['k_ratio'] for s in summaries]
    qps_values = [s['qps'] for s in summaries]
    precision_values = [s['precision'] for s in summaries]

    # Get workers (should be the same for all experiments in this shard count)
    workers = set(s['workers'] for s in summaries)
    workers = list(workers)[0] if len(workers) == 1 else f"Workers={sorted(workers)}"

    # Get K value (should be the same for all experiments in this shard count)
    k_values = set(s.get('k_value', 'unknown') for s in summaries)
    k_value = list(k_values)[0] if len(k_values) == 1 else f"K={sorted(k_values)}"

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot QPS (performance) on left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Effective K (Ratio)')
    ax1.set_ylabel('QPS (Queries Per Second)', color=color1)
    line1 = ax1.plot(effective_k_values, qps_values, 'o-', color=color1, label='QPS', linewidth=2, markersize=6)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Create consistent x-axis with fixed intervals from lowest ratio to 1.0
    min_ratio = min(ratios)
    # Create x-axis values from min_ratio to 1.0 in 0.1 increments
    x_axis_ratios = []
    current_ratio = min_ratio
    while current_ratio <= 1.0:
        x_axis_ratios.append(round(current_ratio, 1))
        current_ratio += 0.1

    # Calculate corresponding effective K values for x-axis
    x_axis_effective_k = [calculate_effective_k(ratio, shard_count, k_value) for ratio in x_axis_ratios]

    # Create x-axis labels
    x_labels = [f'{int(eff_k)}\n({ratio:.1f})' for eff_k, ratio in zip(x_axis_effective_k, x_axis_ratios)]

    # Set consistent x-axis
    ax1.set_xticks(x_axis_effective_k)
    ax1.set_xticklabels(x_labels, rotation=0, ha='center')
    ax1.set_xlim(min(x_axis_effective_k) * 0.95, max(x_axis_effective_k) * 1.05)

    # Create second y-axis for precision (recall)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Precision (Recall)', color=color2)
    line2 = ax2.plot(effective_k_values, precision_values, 's-', color=color2, label='Precision', linewidth=2, markersize=6)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Format precision y-axis to show max 3 decimal places
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))

    # Add labels for QPS values (blue line) - positioned closer to points
    for i, (x, y) in enumerate(zip(effective_k_values, qps_values)):
        # Alternate positioning: odd indices go slightly higher to avoid overlap
        offset_y = 8 if i % 2 == 0 else 12
        ax1.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, offset_y), ha='center',
                    fontsize=8, color=color1, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7, edgecolor=color1))

    # Add labels for Precision values (red line) - positioned closer below points
    for i, (x, y) in enumerate(zip(effective_k_values, precision_values)):
        # Alternate positioning: odd indices go slightly lower to avoid overlap
        offset_y = -12 if i % 2 == 0 else -16
        ax2.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, offset_y), ha='center',
                    fontsize=8, color=color2, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7, edgecolor=color2))

    # Add selective improvement annotations (Option C)
    if len(summaries) >= 2:
        # Find the point with ratio 1.0 (baseline) and lowest ratio
        baseline_idx = None
        lowest_ratio_idx = 0

        for i, ratio in enumerate(ratios):
            if abs(ratio - 1.0) < 0.01:  # ratio ≈ 1.0
                baseline_idx = i
            if ratios[i] < ratios[lowest_ratio_idx]:
                lowest_ratio_idx = i

        # Add annotation if we have both baseline and lowest ratio points, and they're different
        if baseline_idx is not None and lowest_ratio_idx != baseline_idx:
            baseline_qps = qps_values[baseline_idx]
            lowest_qps = qps_values[lowest_ratio_idx]
            baseline_precision = precision_values[baseline_idx]
            lowest_precision = precision_values[lowest_ratio_idx]

            qps_improvement = ((lowest_qps - baseline_qps) / baseline_qps) * 100
            precision_change = ((lowest_precision - baseline_precision) / baseline_precision) * 100

            # Option A: Place annotation outside graph area, bottom-left corner
            lowest_ratio = ratios[lowest_ratio_idx]
            annotation_text = f"Ratio: {lowest_ratio:.1f} vs 1.0\n{qps_improvement:+.0f}% QPS\n{precision_change:+.1f}% Precision"

            # Position outside the plot area in bottom-left, below x-axis labels and legend
            ax1.text(0.02, -0.13, annotation_text, transform=ax1.transAxes,
                    fontsize=12, ha='left', va='top',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', alpha=0.9, edgecolor='orange'))

    # Add title and legend
    plt.title(f'Performance vs EffectiveK - {shard_count} Shards (K={k_value})', fontsize=14, fontweight='bold')

    # Create proper legend below x-axis, centered in one line
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, bbox_to_anchor=(0.5, -0.15), loc='upper center',
               ncol=2, frameon=False, fontsize=10)

    # Adjust plot margins to prevent label truncation
    plt.subplots_adjust(top=0.85, bottom=0.15)

    # Improve layout
    plt.tight_layout()

    # Save the graph
    filename = f"shard_count_{shard_count}_k_{k_value}_workers_{workers}_performance.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Graph saved: {filename}")

    # Print data summary
    print(f"  Data points: {len(summaries)}")
    print(f"  Effective K values: {effective_k_values}")
    print(f"  QPS range: {min(qps_values):.1f} - {max(qps_values):.1f}")
    print(f"  Precision range: {min(precision_values):.3f} - {max(precision_values):.3f}")

def main():
    """Main function to orchestrate the analysis."""
    print("Starting shard_k_ratio analysis...")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Looking for result files in: {RESULTS_DIR}")
    print(f"Graphs will be saved to: {OUTPUT_DIR}")

    # Find and parse summary files
    summary_files = find_summary_files()

    if not summary_files:
        print("No summary files found! Please check the results directory.")
        return

    # Process all files and extract summary data
    all_summaries = []

    for file_path in summary_files:
        print(f"\nProcessing: {os.path.basename(file_path)}")
        data = load_benchmark_data(file_path)

        if data:
            summary = extract_summary_data(data, os.path.basename(file_path))
            if summary:
                all_summaries.append(summary)
                print(f"  Extracted summary: k_ratio={summary['k_ratio']}, shard_count={summary['shard_count']}, precision={summary['precision']:.3f}, workers={summary['workers']}")

    print(f"\nTotal valid summaries found: {len(all_summaries)}")

    if not all_summaries:
        print("No valid summary data found! Please check your result files.")
        return

    # Organize by shard count
    by_k_and_shards = organize_data_by_k_and_shards(all_summaries)
    print(f"Data organized by shard counts: {sorted(by_k_and_shards.keys())}")

    # Show summary of what we found
    for (k_value, shard_count, workers) in sorted(by_k_and_shards.keys()):
        summaries = by_k_and_shards[(k_value, shard_count, workers)]
        ratios = sorted(set(s['k_ratio'] for s in summaries))
        print(f"  K={k_value}, Shard count {shard_count}: {len(summaries)} summaries")
        print(f"    K ratios found: {ratios}")

    # Create graphs for each shard count
    for (k_value, shard_count, workers) in sorted(by_k_and_shards.keys()):
        summaries = by_k_and_shards[(k_value, shard_count, workers)]
        print(f"\nCreating graph for K={k_value}, {shard_count} shards, {workers} workers:")
        create_graph_for_shard_count(shard_count, summaries, OUTPUT_DIR)

    print(f"\nAll graphs saved to: {OUTPUT_DIR}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()
