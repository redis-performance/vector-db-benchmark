#!/usr/bin/env python3
"""
Extract benchmark results table from final summary files.
"""

import os
import json
import glob
import re
from datetime import datetime

def list_final_files():
    """List all final summary files and print basic info."""
    final_dir = "results/final"

    if not os.path.exists(final_dir):
        print(f"Error: {final_dir} directory not found")
        return []

    # Get all JSON files in final directory
    pattern = os.path.join(final_dir, "*.json")
    files = glob.glob(pattern)

    print(f"Found {len(files)} final summary files:")
    for i, file_path in enumerate(files, 1):
        filename = os.path.basename(file_path)
        print(f"{i:2d}. {filename}")

    return files

def examine_file_structure(file_path):
    """Examine the structure of a single final file."""
    print(f"\n=== Examining: {os.path.basename(file_path)} ===")

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        print("Top-level keys:", list(data.keys()))

        # Check precision section
        if 'precision' in data:
            precision_keys = list(data['precision'].keys())
            print(f"Precision levels: {precision_keys}")

            if precision_keys:
                first_precision = precision_keys[0]
                config = data['precision'][first_precision].get('config', {})
                print(f"Config keys: {list(config.keys())}")
                print(f"Sample config: {config}")

        # Check search section for referenced files
        if 'search' in data:
            search_keys = list(data['search'].keys())
            print(f"Number of search results: {len(search_keys)}")
            if search_keys:
                print(f"Sample search key: {search_keys[0]}")

    except Exception as e:
        print(f"Error reading file: {e}")

def parse_timestamp_from_filename(filename):
    """Extract date and time from a filename with timestamp."""
    # Pattern: redis-hnsw-m-16-ef-128-arxiv-titles-384-angular-filters-search-4-403801-2025-07-31-09-44-01
    # We want the last part: 2025-07-31-09-44-01
    pattern = r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})$'
    match = re.search(pattern, filename)

    if match:
        timestamp_str = match.group(1)
        try:
            # Parse the timestamp
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M-%S')
            return dt.strftime('%Y-%m-%d'), dt.strftime('%H:%M:%S')
        except ValueError as e:
            print(f"Error parsing timestamp {timestamp_str}: {e}")
            return None, None
    else:
        print(f"No timestamp found in filename: {filename}")
        return None, None

def extract_config_from_final_file(file_path):
    """Extract configuration parameters from a final summary file."""
    print(f"\n=== Extracting config from: {os.path.basename(file_path)} ===")

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Get config from precision section
        if 'precision' not in data or not data['precision']:
            print("No precision data found")
            return None

        # Get the first precision level
        first_precision_key = list(data['precision'].keys())[0]
        config = data['precision'][first_precision_key].get('config', {})

        # Extract parameters
        shard_count = config.get('shard_count')
        workers = config.get('workers')
        k_ratio = config.get('k_ratio')

        print(f"Config extracted: shard_count={shard_count}, workers={workers}, k_ratio={k_ratio}")

        # Get k value from search section
        k_value = None
        if 'search' in data and data['search']:
            first_search_key = list(data['search'].keys())[0]
            search_params = data['search'][first_search_key].get('params', {})
            k_value = search_params.get('top')
            print(f"K value from search params: {k_value}")

        # Get timestamps from search keys
        timestamps = []
        if 'search' in data:
            for search_key in data['search'].keys():
                date, time = parse_timestamp_from_filename(search_key)
                if date and time:
                    timestamps.append((date, time))
                    print(f"Found timestamp: {date} {time}")

        return {
            'shard_count': shard_count,
            'workers': workers,
            'k': k_value,
            'ratio': k_ratio,
            'timestamps': timestamps,
            'filename': os.path.basename(file_path)
        }

    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def generate_table():
    """Generate the complete table from all final files."""
    print("\n" + "=" * 80)
    print("Step 2c: Generating complete table")
    print("=" * 80)

    files = list_final_files()
    all_rows = []

    for file_path in files:
        print(f"\nProcessing: {os.path.basename(file_path)}")
        config_data = extract_config_from_final_file(file_path)

        if config_data and config_data['timestamps']:
            # Create a row for each timestamp in this file
            for date, time in config_data['timestamps']:
                row = {
                    'date': date,
                    'time': time,
                    'shard_count': config_data['shard_count'],
                    'workers': config_data['workers'],
                    'k': config_data['k'],
                    'ratio': config_data['ratio']
                }
                all_rows.append(row)
                print(f"  Added row: {date} {time} | shards={row['shard_count']} | workers={row['workers']} | k={row['k']} | ratio={row['ratio']}")

    # Sort by date and time
    all_rows.sort(key=lambda x: (x['date'], x['time']))

    print(f"\n" + "=" * 80)
    print("FINAL TABLE")
    print("=" * 80)
    print(f"{'Date':<12} {'Time':<10} {'Shard Count':<12} {'Workers':<8} {'K':<6} {'Ratio':<6}")
    print("-" * 80)

    for row in all_rows:
        print(f"{row['date']:<12} {row['time']:<10} {row['shard_count']:<12} {row['workers']:<8} {row['k']:<6} {row['ratio']:<6}")

    print(f"\nTotal rows: {len(all_rows)}")
    return all_rows

if __name__ == "__main__":
    # Generate the complete table
    table_data = generate_table()
