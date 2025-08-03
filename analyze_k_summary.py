#!/usr/bin/env python3
"""
Script to generate K value analysis summary from benchmark results.

This script analyzes the results and generates a summary showing QPS improvements
for different shard configurations and K ratios compared to baseline (ratio 1.0).
"""

import json
import os
import glob
import math
from typing import Dict, List, Optional
from collections import defaultdict

# Configuration
RESULTS_DIR = "results/final"

def find_summary_files() -> List[str]:
    """Find all summary JSON files in the results directory."""
    pattern = os.path.join(RESULTS_DIR, "*-summary.json")
    files = glob.glob(pattern)
    return files

def get_workers_from_filename(filename: str) -> int:
    """Parse filename to extract workers."""
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
    """Calculate effective K value based on ratio and shard configuration."""
    num_shards = shard_count
    k_ratio_config = k_ratio
    
    k_min = math.ceil(k / num_shards)
    k_per_shard_req = math.ceil(k * k_ratio_config)
    effective_k = max(k_min, k_per_shard_req)
    
    return effective_k

def calculate_actual_k_ratio(k_ratio, shard_count, k):
    """Calculate actual K ratio based on effective K."""
    effective_k = calculate_effective_k(k_ratio, shard_count, k)
    k_ratio_actual = effective_k / k
    return k_ratio_actual

def extract_summary_data(data: Dict, filename: str) -> Optional[Dict]:
    """Extract summary data from a benchmark JSON file."""
    if 'precision_summary' not in data or 'precision' not in data:
        return None
    
    precision_data = data['precision']
    if not precision_data:
        return None
    
    # Get configuration from first precision entry
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
        'k_ratio': k_ratio,
        'actual_k_ratio': calculate_actual_k_ratio(k_ratio, shard_count, k_value),
        'shard_count': shard_count,
        'workers': workers,
        'k_value': k_value,
        'precision': float(best_precision),
        'qps': best_data.get('qps', 0),
        'p50': best_data.get('p50', 0),
        'p95': best_data.get('p95', 0),
    }

def organize_data_by_k_value(summaries: List[Dict]) -> Dict[int, Dict[int, List[Dict]]]:
    """Organize data by K value, then by shard count."""
    by_k_value = defaultdict(lambda: defaultdict(list))
    
    for summary in summaries:
        k_value = summary['k_value']
        shard_count = summary['shard_count']
        by_k_value[k_value][shard_count].append(summary)
    
    return dict(by_k_value)

def generate_k_analysis_summary(data_by_k: Dict[int, Dict[int, List[Dict]]]):
    """Generate and print the K analysis summary."""
    
    for k_value in sorted(data_by_k.keys()):
        print(f"K={k_value} Data Analysis:")
        print()
        
        shard_data = data_by_k[k_value]
        
        for shard_count in sorted(shard_data.keys()):
            shard_summaries = shard_data[shard_count]
            
            # Sort by k_ratio to ensure baseline (1.0) comes first
            shard_summaries = sorted(shard_summaries, key=lambda x: x['k_ratio'], reverse=True)
            
            print(f"{shard_count} Shards:")
            print()
            
            # Find baseline (ratio 1.0) QPS
            baseline_qps = None
            for summary in shard_summaries:
                if abs(summary['k_ratio'] - 1.0) < 0.001:  # Handle floating point comparison
                    baseline_qps = summary['qps']
                    print(f"Ratio {summary['k_ratio']:.2f}: {summary['qps']:.1f} QPS")
                    break
            
            # Print other ratios with improvement percentages
            for summary in shard_summaries:
                if abs(summary['k_ratio'] - 1.0) >= 0.001:  # Not baseline
                    qps = summary['qps']
                    if baseline_qps and baseline_qps > 0:
                        improvement = ((qps - baseline_qps) / baseline_qps) * 100
                        print(f"Ratio {summary['k_ratio']:.2f}: {qps:.1f} QPS → {improvement:+.0f}% improvement")
                    else:
                        print(f"Ratio {summary['k_ratio']:.2f}: {qps:.1f} QPS")
            
            print()

def main():
    """Main function to run the analysis."""
    print("K Value Analysis Summary")
    print("=" * 50)
    print()
    
    # Find and load all summary files
    files = find_summary_files()
    print(f"Processing {len(files)} summary files...")
    print()
    
    # Extract data from all files
    all_summaries = []
    for file_path in files:
        data = load_benchmark_data(file_path)
        if data:
            summary = extract_summary_data(data, file_path)
            if summary:
                all_summaries.append(summary)
    
    print(f"Successfully processed {len(all_summaries)} files")
    print()
    
    # Organize data by K value and shard count
    data_by_k = organize_data_by_k_value(all_summaries)
    
    # Generate and print the summary
    generate_k_analysis_summary(data_by_k)

if __name__ == "__main__":
    main()
