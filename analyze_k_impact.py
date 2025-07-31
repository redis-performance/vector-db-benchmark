#!/usr/bin/env python3
"""
Script to analyze K value impact on performance for same effective K configurations.

This script generates graphs showing how different K values affect QPS performance
when they result in the same effective K per shard.
"""

import json
import os
import glob
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import math

# Configuration
RESULTS_DIR = "/home/ubuntu/vector-db-benchmark/results/final"
OUTPUT_DIR = "./graphs_k_impact"

def find_summary_files() -> List[str]:
    """Find all summary JSON files in the results directory."""
    pattern = os.path.join(RESULTS_DIR, "*-summary.json")
    files = glob.glob(pattern)
    print(f"Found {len(files)} summary files:")
    for file in files:
        print(f"  - {os.path.basename(file)}")
    return files

def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse filename to extract database configuration info.
    Handles both old and new formats with workers.
    """
    basename = os.path.basename(filename)

    # Check if it's the new format with workers
    if "workers_" in basename:
        # New format: ...k_ratio_1.0-workers_8-summary.json
        workers_part = basename.split("workers_")[1]  # "8-summary.json"
        workers = workers_part.split("-")[0]  # "8"
    else:
        # Old format: ...k_ratio_1.0-summary.json
        workers = "8"  # Default for old files

    # Remove -summary.json suffix
    name_parts = basename.replace("-summary.json", "").split("-")

    # Extract key information
    parsed = {
        "engine": name_parts[0] if len(name_parts) > 0 else "unknown",
        "algorithm": name_parts[1] if len(name_parts) > 1 else "unknown",
        "filename": basename,
        "full_name": basename.replace("-summary.json", ""),
        "workers": int(workers)
    }

    return parsed

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
    """Calculate the effective K per shard."""
    num_shards = shard_count
    k_ratio_config = k_ratio

    k_min = math.ceil(k / num_shards)
    k_per_shard_req = math.ceil(k * k_ratio_config)
    effective_k = max(k_min, k_per_shard_req)

    return effective_k

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

    # Get the first precision entry
    first_precision_key = list(precision_data.keys())[0]
    config = precision_data[first_precision_key].get('config', {})

    k_ratio = config.get('k_ratio')
    shard_count = config.get('shard_count')
    workers = parse_filename(filename)['workers']

    # Get K value from search section
    if 'search' not in data or not data['search']:
        return None

    first_search_key = list(data['search'].keys())[0]
    search_params = data['search'][first_search_key].get('params', {})
    k_value = search_params.get('top')

    if k_ratio is None or shard_count is None or k_value is None:
        print(f"Warning: Missing required data in {filename}")
        return None

    # Calculate effective K
    effective_k = calculate_effective_k(k_ratio, shard_count, k_value)

    # Extract performance data from precision_summary
    precision_summary = data['precision_summary']
    best_precision = max(precision_summary.keys(), key=float)
    qps = precision_summary[best_precision].get('qps', 0)

    return {
        'filename': filename,
        'k_ratio': k_ratio,
        'shard_count': shard_count,
        'workers': workers,
        'k_value': k_value,
        'effective_k': effective_k,
        'qps': qps,
        'precision': float(best_precision)
    }

def organize_data_by_effective_k_and_shards(summaries: List[Dict]) -> Dict[tuple, List[Dict]]:
    """Organize summary data by (effective_k, shard_count, workers) combination."""
    by_config = {}

    for summary in summaries:
        effective_k = summary['effective_k']
        shard_count = summary['shard_count']
        workers = summary['workers']
        key = (effective_k, shard_count, workers)

        if key not in by_config:
            by_config[key] = []
        by_config[key].append(summary)

    return by_config

def create_k_impact_graph(effective_k: int, shard_count: int, workers: int, summaries: List[Dict], output_dir: str):
    """Create a graph showing K value impact on QPS for same effective K configuration."""
    if len(summaries) < 2:
        print(f"Skipping graph for effective_k={effective_k}, shards={shard_count}, workers={workers} - need at least 2 K values")
        return

    # Sort by K value for proper line plotting
    summaries = sorted(summaries, key=lambda x: x['k_value'])

    # Extract data for plotting
    k_values = [s['k_value'] for s in summaries]
    qps_values = [s['qps'] for s in summaries]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot QPS vs K values
    color = 'tab:blue'
    ax.set_xlabel('K Value')
    ax.set_ylabel('QPS (Queries Per Second)', color=color)
    line = ax.plot(k_values, qps_values, 'o-', color=color, linewidth=2, markersize=8)
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(True, alpha=0.3)

    # Add labels for QPS values
    for i, (x, y) in enumerate(zip(k_values, qps_values)):
        offset_y = 8 if i % 2 == 0 else 12
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, offset_y), ha='center',
                   fontsize=9, color=color, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7, edgecolor=color))

    # Add title
    plt.title(f'QPS vs K Value - EffectiveK={effective_k}, {shard_count} Shards, {workers} Workers',
              fontsize=14, fontweight='bold')

    # Set x-axis to show all K values
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])

    # Improve layout
    plt.tight_layout()

    # Save the graph
    filename = f"effective_k_{effective_k}_shards_{shard_count}_workers_{workers}_k_impact.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Graph saved: {filename}")
    print(f"  K values: {k_values}")
    print(f"  QPS values: {[f'{qps:.1f}' for qps in qps_values]}")

def create_combined_k_impact_graph(shard_count: int, workers: int, configs_data: Dict[int, List[Dict]], output_dir: str):
    """Create a combined graph with multiple lines, each representing an effective K value."""

    # Filter configs that have multiple K values
    valid_configs = {eff_k: summaries for eff_k, summaries in configs_data.items() if len(summaries) >= 2}

    if not valid_configs:
        print(f"Skipping combined graph for {shard_count} shards, {workers} workers - no configs with multiple K values")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palette for different effective K lines
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

    # Plot each effective K as a separate line
    for i, (effective_k, summaries) in enumerate(sorted(valid_configs.items())):
        # Sort by K value
        summaries = sorted(summaries, key=lambda x: x['k_value'])

        k_values = [s['k_value'] for s in summaries]
        qps_values = [s['qps'] for s in summaries]

        color = colors[i % len(colors)]

        # Plot line
        ax.plot(k_values, qps_values, 'o-', color=color, linewidth=2, markersize=6,
                label=f'Effective K = {effective_k}')

        # Add value labels
        for j, (x, y) in enumerate(zip(k_values, qps_values)):
            offset_y = 8 if j % 2 == 0 else 12
            ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, offset_y),
                       ha='center', fontsize=8, color=color, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7, edgecolor=color))

    # Formatting
    ax.set_xlabel('K Value')
    ax.set_ylabel('QPS (Queries Per Second)')
    ax.set_title(f'QPS vs K Value by Effective K - {shard_count} Shards, {workers} Workers',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Set x-axis to show all K values
    all_k_values = sorted(set(s['k_value'] for summaries in valid_configs.values() for s in summaries))
    ax.set_xticks(all_k_values)
    ax.set_xticklabels([str(k) for k in all_k_values])

    # Improve layout
    plt.tight_layout()

    # Save the graph
    filename = f"combined_k_impact_shards_{shard_count}_workers_{workers}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Combined graph saved: {filename}")
    print(f"  Effective K values: {sorted(valid_configs.keys())}")
    print(f"  K values range: {min(all_k_values)} - {max(all_k_values)}")

def main():
    """Main function to orchestrate the K impact analysis."""
    print("Starting K Impact Analysis...")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Looking for result files in: {RESULTS_DIR}")
    print(f"Graphs will be saved to: {OUTPUT_DIR}")

    # Find and process summary files
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
                print(f"  Extracted: K={summary['k_value']}, EffectiveK={summary['effective_k']}, Shards={summary['shard_count']}, Workers={summary['workers']}")

    print(f"\nTotal valid summaries found: {len(all_summaries)}")

    if not all_summaries:
        print("No valid summary data found! Please check your result files.")
        return

    # Organize by effective K and shard configuration
    by_config = organize_data_by_effective_k_and_shards(all_summaries)
    print(f"Data organized by configurations: {len(by_config)} unique configs")

    # Show summary of what we found
    for (effective_k, shard_count, workers), summaries in sorted(by_config.items()):
        k_values = sorted(set(s['k_value'] for s in summaries))
        print(f"  EffectiveK={effective_k}, Shards={shard_count}, Workers={workers}: {len(summaries)} summaries")
        print(f"    K values: {k_values}")

    # Create individual graphs for each configuration
    print(f"\nCreating individual K impact graphs...")
    individual_graphs_created = 0

    for (effective_k, shard_count, workers), summaries in sorted(by_config.items()):
        print(f"\nProcessing EffectiveK={effective_k}, Shards={shard_count}, Workers={workers}:")
        create_k_impact_graph(effective_k, shard_count, workers, summaries, OUTPUT_DIR)
        individual_graphs_created += 1

    # Create combined graphs by shard count and workers
    print(f"\nCreating combined K impact graphs...")
    combined_graphs_created = 0

    # Group by (shard_count, workers) for combined graphs
    by_shard_workers = {}
    for (effective_k, shard_count, workers), summaries in by_config.items():
        key = (shard_count, workers)
        if key not in by_shard_workers:
            by_shard_workers[key] = {}
        by_shard_workers[key][effective_k] = summaries

    for (shard_count, workers), configs_data in sorted(by_shard_workers.items()):
        print(f"\nCreating combined graph for {shard_count} shards, {workers} workers:")
        create_combined_k_impact_graph(shard_count, workers, configs_data, OUTPUT_DIR)
        combined_graphs_created += 1

    print(f"\nAnalysis complete!")
    print(f"Created {individual_graphs_created} individual graphs")
    print(f"Created {combined_graphs_created} combined graphs")
    print(f"All graphs saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
