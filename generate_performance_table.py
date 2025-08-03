#!/usr/bin/env python3
"""
Generate performance comparison table showing improvements from baseline (ratio=1.0)
to best performing ratio for each shard/K configuration.
"""

import json
import os
import glob
import math
from typing import Dict, List, Optional, Tuple
import csv
from io import StringIO

# Configuration
RESULTS_DIR = "results/final"

def get_workers_from_filename(filename: str) -> int:
    """Parse filename to extract workers count."""
    basename = os.path.basename(filename)
    if "workers_" in basename:
        workers_part = basename.split("workers_")[1]
        workers = workers_part.split("-")[0]
    else:
        workers = "8"  # default
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
    """Calculate effective K per shard."""
    k_min = math.ceil(k / shard_count)
    k_per_shard_req = math.ceil(k * k_ratio)
    effective_k = max(k_min, k_per_shard_req)
    return effective_k

def calculate_actual_k_ratio(k_ratio, shard_count, k):
    """Calculate actual K ratio based on effective K."""
    effective_k = calculate_effective_k(k_ratio, shard_count, k)
    return effective_k / k

def extract_summary_data(data: Dict, filename: str) -> Optional[Dict]:
    """Extract summary data from a benchmark JSON file."""
    if 'precision_summary' not in data or 'precision' not in data:
        return None

    precision_data = data['precision']
    if not precision_data:
        return None

    # Get config from first precision entry
    first_precision_key = list(precision_data.keys())[0]
    config = precision_data[first_precision_key].get('config', {})

    k_ratio = config.get('k_ratio')
    shard_count = config.get('shard_count')
    workers = get_workers_from_filename(filename)

    if k_ratio is None or shard_count is None:
        return None

    # Get K value from search section
    k_value = None
    if 'search' in data and data['search']:
        first_search_key = list(data['search'].keys())[0]
        search_params = data['search'][first_search_key].get('params', {})
        k_value = search_params.get('top')

    if k_value is None:
        return None

    # Extract performance data from precision_summary
    precision_summary = data['precision_summary']
    best_precision = max(precision_summary.keys(), key=float)
    best_data = precision_summary[best_precision]

    return {
        'filename': filename,
        'k_ratio_config': k_ratio,
        'k_ratio_actual': calculate_actual_k_ratio(k_ratio, shard_count, k_value),
        'shard_count': shard_count,
        'workers': workers,
        'k_value': k_value,
        'precision': float(best_precision),
        'qps': best_data.get('qps', 0),
        'p50': best_data.get('p50', 0),
        'p95': best_data.get('p95', 0),
    }

def find_baseline_and_best(summaries: List[Dict]) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Find baseline (ratio=1.0) and best performing configuration."""
    baseline = None
    best_qps = None

    for summary in summaries:
        # Find baseline (closest to ratio 1.0)
        if abs(summary['k_ratio_actual'] - 1.0) < 0.01:
            baseline = summary

        # Find best QPS
        if best_qps is None or summary['qps'] > best_qps['qps']:
            best_qps = summary

    return baseline, best_qps

def calculate_improvements(baseline: Dict, best: Dict) -> Dict:
    """Calculate percentage improvements from baseline to best."""
    if not baseline or not best:
        return {}

    qps_improvement = ((best['qps'] - baseline['qps']) / baseline['qps']) * 100

    # For latency, lower is better, so improvement is negative change
    p95_improvement = ((baseline['p95'] - best['p95']) / baseline['p95']) * 100

    # For accuracy, calculate loss (negative improvement)
    accuracy_change = ((best['precision'] - baseline['precision']) / baseline['precision']) * 100

    return {
        'qps_improvement': qps_improvement,
        'latency_improvement': p95_improvement,
        'accuracy_change': accuracy_change,
        'ratio_change': f"{baseline['k_ratio_actual']:.2f} → {best['k_ratio_actual']:.2f}"
    }

def main():
    """Main function to generate the performance table."""
    print("Generating performance comparison table...")

    # Find all summary files
    pattern = os.path.join(RESULTS_DIR, "*-summary.json")
    files = glob.glob(pattern)
    print(f"Found {len(files)} summary files")

    # Process all files
    all_summaries = []
    for file_path in files:
        data = load_benchmark_data(file_path)
        if data:
            summary = extract_summary_data(data, os.path.basename(file_path))
            if summary:
                all_summaries.append(summary)

    print(f"Extracted {len(all_summaries)} valid summaries")

    # Group by (shard_count, k_value, workers)
    by_config = {}
    for summary in all_summaries:
        key = (summary['shard_count'], summary['k_value'], summary['workers'])
        if key not in by_config:
            by_config[key] = []
        by_config[key].append(summary)

    # Generate table data
    table_rows = []

    for (shard_count, k_value, workers), summaries in sorted(by_config.items()):
        baseline, best = find_baseline_and_best(summaries)

        if baseline and best and baseline != best:
            improvements = calculate_improvements(baseline, best)
            if improvements:
                table_rows.append({
                    'shard_count': shard_count,
                    'k_value': k_value,
                    'workers': workers,
                    **improvements
                })

    # Print formatted table
    print("\n" + "="*100)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*100)
    print(f"{'Shard Count':<12} {'K Value':<8} {'Workers':<8} {'Ratio Change':<15} {'QPS Improvement':<16} {'Latency Improvement':<20} {'Accuracy Change':<15}")
    print("-"*100)

    for row in table_rows:
        print(f"{row['shard_count']:<12} {row['k_value']:<8} {row['workers']:<8} "
              f"{row['ratio_change']:<15} {row['qps_improvement']:+.0f}%{'':<11} "
              f"{row['latency_improvement']:+.0f}%{'':<15} {row['accuracy_change']:+.1f}%")

    # Generate tab-separated table (easy to copy)
    print("\n" + "="*100)
    print("TAB-SEPARATED TABLE (copy-paste ready):")
    print("="*100)
    print("Shard Count\tK Value\tWorkers\tRatio Change\tQPS Improvement (%)\tLatency Improvement (%)\tAccuracy Change (%)")

    for row in table_rows:
        print(f"{row['shard_count']}\t{row['k_value']}\t{row['workers']}\t"
              f"{row['ratio_change']}\t{row['qps_improvement']:+.0f}\t"
              f"{row['latency_improvement']:+.0f}\t{row['accuracy_change']:+.1f}")

    # Generate CSV output
    print("\n" + "="*100)
    print("CSV FORMAT (copy-paste ready):")
    print("="*100)

    csv_output = StringIO()
    csv_writer = csv.writer(csv_output)

    # Write header
    csv_writer.writerow(['Shard Count', 'K Value', 'Workers', 'Ratio Change', 'QPS Improvement (%)', 'Latency Improvement (%)', 'Accuracy Change (%)'])

    # Write data rows
    for row in table_rows:
        csv_writer.writerow([
            row['shard_count'],
            row['k_value'],
            row['workers'],
            row['ratio_change'],
            f"{row['qps_improvement']:+.0f}",
            f"{row['latency_improvement']:+.0f}",
            f"{row['accuracy_change']:+.1f}"
        ])

    print(csv_output.getvalue())

    # Save to file
    csv_filename = "performance_comparison.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        csvfile.write(csv_output.getvalue())

    print(f"\nResults saved to: {csv_filename}")
    print(f"Total configurations analyzed: {len(table_rows)}")

if __name__ == "__main__":
    main()
