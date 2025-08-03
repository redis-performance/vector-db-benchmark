#!/usr/bin/env python3
"""
Extract raw data table from benchmark results for Confluence.
"""

import json
import os
import glob
import math

RESULTS_DIR = "results/final"

def extract_data_from_file(file_path):
    """Extract data from a single summary file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Get data from precision section
        if 'precision' not in data or not data['precision']:
            return None

        first_precision_key = list(data['precision'].keys())[0]
        config = data['precision'][first_precision_key].get('config', {})

        k_ratio_config = config.get('k_ratio')
        shard_count = config.get('shard_count')

        # Get K from search section
        if 'search' not in data or not data['search']:
            return None

        first_search_key = list(data['search'].keys())[0]
        search_params = data['search'][first_search_key].get('params', {})
        k_value = search_params.get('top')

        # Get QPS from precision_summary
        if 'precision_summary' not in data:
            return None

        precision_summary = data['precision_summary']
        best_precision = max(precision_summary.keys(), key=float)
        qps = precision_summary[best_precision].get('qps', 0)

        # Calculate effective K
        k_min = math.ceil(k_value / shard_count)
        k_per_shard_req = math.ceil(k_value * k_ratio_config)
        effective_k = max(k_min, k_per_shard_req)
        actual_ratio = effective_k / k_value

        return {
            'shard_count': shard_count,
            'k': k_value,
            'effective_k': effective_k,
            'ratio': actual_ratio,
            'qps': qps,
            'filename': os.path.basename(file_path)
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    """Extract all data and create table."""
    pattern = os.path.join(RESULTS_DIR, "*-summary.json")
    files = glob.glob(pattern)

    print(f"Found {len(files)} files")

    all_data = []
    for file_path in files:
        data = extract_data_from_file(file_path)
        if data:
            all_data.append(data)

    # Sort by shard count, then K, then effective K
    all_data.sort(key=lambda x: (x['shard_count'], x['k'], x['effective_k']))

    print("\n" + "="*70)
    print("RAW DATA TABLE FOR CONFLUENCE:")
    print("="*70)
    print("| Shard Count | K | Effective K | Ratio | QPS |")
    print("|-------------|---|-------------|-------|-----|")

    for row in all_data:
        print(f"| {row['shard_count']} | {row['k']} | {row['effective_k']} | {row['ratio']:.2f} | {row['qps']:.1f} |")

    print("="*70)
    print(f"\nTotal rows: {len(all_data)}")

if __name__ == "__main__":
    main()
