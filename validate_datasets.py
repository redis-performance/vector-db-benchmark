#!/usr/bin/env python3
"""
Dataset validation script for vector-db-benchmark

This script validates the datasets.json file to ensure:
1. All required fields are present
2. Field types are correct
3. Values are reasonable
4. Dataset names are unique
5. The --describe functionality works

Usage:
    python validate_datasets.py
    python validate_datasets.py --strict  # Exit on warnings too
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Any


def load_datasets() -> List[dict]:
    """Load and parse datasets.json file."""
    datasets_file = Path('datasets/datasets.json')
    if not datasets_file.exists():
        print('❌ datasets/datasets.json not found')
        sys.exit(1)
        
    try:
        with open(datasets_file, 'r') as f:
            datasets = json.load(f)
    except json.JSONDecodeError as e:
        print(f'❌ Invalid JSON in datasets.json: {e}')
        sys.exit(1)
        
    if not isinstance(datasets, list):
        print('❌ datasets.json must contain a list of datasets')
        sys.exit(1)
        
    return datasets


def validate_dataset_structure(datasets: List[dict]) -> Tuple[List[str], List[str]]:
    """Validate dataset structure and required fields."""
    required_fields = {
        'name': str,
        'vector_size': int,
        'distance': str,
        'type': str,
        'path': (str, dict),  # Can be string or dict for h5-multi type
        'vector_count': (int, type(None)),
        'description': (str, type(None))
    }
    
    valid_distances = ['cosine', 'l2', 'dot', 'euclidean']
    valid_types = ['h5', 'tar', 'jsonl', 'h5-multi']
    
    errors = []
    warnings = []
    
    for i, dataset in enumerate(datasets):
        dataset_name = dataset.get('name', f'Dataset #{i+1}')
        
        # Check required fields
        for field, expected_type in required_fields.items():
            if field not in dataset:
                errors.append(f'❌ {dataset_name}: Missing required field "{field}"')
                continue
                
            value = dataset[field]
            
            # Handle tuple types (multiple allowed types)
            if isinstance(expected_type, tuple):
                if not isinstance(value, expected_type):
                    type_names = [t.__name__ if t != type(None) else 'null' for t in expected_type]
                    errors.append(f'❌ {dataset_name}: Field "{field}" must be {" or ".join(type_names)}, got {type(value).__name__}')
            elif not isinstance(value, expected_type):
                errors.append(f'❌ {dataset_name}: Field "{field}" must be {expected_type.__name__}, got {type(value).__name__}')
        
        # Validate specific field values
        distance = dataset.get('distance')
        if distance and distance not in valid_distances:
            errors.append(f'❌ {dataset_name}: Invalid distance "{distance}", must be one of {valid_distances}')
            
        data_type = dataset.get('type')
        if data_type and data_type not in valid_types:
            warnings.append(f'⚠️  {dataset_name}: Unusual type "{data_type}", expected one of {valid_types}')
            
        # Validate numeric values
        vector_size = dataset.get('vector_size')
        if isinstance(vector_size, int):
            if vector_size <= 0:
                errors.append(f'❌ {dataset_name}: vector_size must be positive, got {vector_size}')
            elif vector_size > 4096:
                warnings.append(f'⚠️  {dataset_name}: Very large vector_size ({vector_size}) - verify this is correct')
                
        vector_count = dataset.get('vector_count')
        if isinstance(vector_count, int):
            if vector_count <= 0:
                errors.append(f'❌ {dataset_name}: vector_count must be positive, got {vector_count}')
            elif vector_count >= 1000000 and vector_count % 1000000 == 0:
                warnings.append(f'⚠️  {dataset_name}: vector_count {vector_count} looks like an estimate (very round number)')
        elif vector_count is None:
            warnings.append(f'⚠️  {dataset_name}: vector_count is None - consider adding actual count')
            
        # Check for missing descriptions
        if dataset.get('description') is None:
            warnings.append(f'⚠️  {dataset_name}: description is None - consider adding description')
            
        # Check for missing links on downloadable datasets
        if not dataset.get('link') and data_type in ['h5', 'tar', 'h5-multi']:
            warnings.append(f'⚠️  {dataset_name}: No download link provided for {data_type} dataset')
    
    return errors, warnings


def validate_unique_names(datasets: List[dict]) -> List[str]:
    """Check that all dataset names are unique."""
    names = [d.get('name') for d in datasets if d.get('name')]
    duplicates = [name for name in set(names) if names.count(name) > 1]
    
    errors = []
    if duplicates:
        errors.append(f'❌ Found duplicate dataset names: {duplicates}')
    
    return errors


def test_describe_functionality() -> List[str]:
    """Test that the --describe functionality works."""
    import subprocess

    errors = []

    try:
        # Test describe datasets
        print("🔍 Testing --describe datasets functionality...")
        result = subprocess.run(
            [sys.executable, 'run.py', '--describe', 'datasets'],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            errors.append(f'❌ --describe datasets failed with exit code {result.returncode}')
            if result.stderr:
                errors.append(f'   Error: {result.stderr.strip()}')
        elif not result.stdout or 'Available Datasets' not in result.stdout:
            errors.append('❌ --describe datasets output missing expected content')
        else:
            print("✅ --describe datasets works correctly")

        # Test describe engines
        print("🔍 Testing --describe engines functionality...")
        result = subprocess.run(
            [sys.executable, 'run.py', '--describe', 'engines'],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            errors.append(f'❌ --describe engines failed with exit code {result.returncode}')
            if result.stderr:
                errors.append(f'   Error: {result.stderr.strip()}')
        elif not result.stdout or 'Available Engines' not in result.stdout:
            errors.append('❌ --describe engines output missing expected content')
        else:
            print("✅ --describe engines works correctly")

    except subprocess.TimeoutExpired:
        errors.append('❌ --describe command timed out')
    except Exception as e:
        errors.append(f'❌ Error testing --describe functionality: {e}')

    return errors


def main():
    parser = argparse.ArgumentParser(description='Validate datasets.json file')
    parser.add_argument('--strict', action='store_true', 
                       help='Exit with error code on warnings too')
    args = parser.parse_args()
    
    print("🔍 Validating datasets.json...")
    
    # Load datasets
    datasets = load_datasets()
    print(f"✅ Loaded {len(datasets)} datasets from datasets.json")
    
    # Validate structure
    structure_errors, structure_warnings = validate_dataset_structure(datasets)
    
    # Validate unique names
    name_errors = validate_unique_names(datasets)
    
    # Test describe functionality
    describe_errors = test_describe_functionality()
    
    # Combine all errors and warnings
    all_errors = structure_errors + name_errors + describe_errors
    all_warnings = structure_warnings
    
    # Print results
    if all_warnings:
        print("\n⚠️  Warnings:")
        for warning in all_warnings:
            print(f"   {warning}")
    
    if all_errors:
        print("\n❌ Errors:")
        for error in all_errors:
            print(f"   {error}")
        print(f"\n❌ Validation failed with {len(all_errors)} errors")
        sys.exit(1)
    else:
        print(f"\n✅ All validations passed!")
        if all_warnings:
            print(f"⚠️  Found {len(all_warnings)} warnings (non-blocking)")
            if args.strict:
                print("❌ Strict mode: treating warnings as errors")
                sys.exit(1)
        
        print(f"📊 Summary: {len(datasets)} datasets validated successfully")


if __name__ == "__main__":
    main()
