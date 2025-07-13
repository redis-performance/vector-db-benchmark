#!/usr/bin/env python3
"""
Script to add missing vector_count and description fields to datasets.json
"""

import json
import re

def estimate_vector_count(name):
    """Estimate vector count from dataset name patterns"""
    name_lower = name.lower()
    
    # Direct patterns
    if '1b' in name_lower or '1billion' in name_lower or '1g' in name_lower:
        return 1000000000
    elif '400m' in name_lower:
        return 400000000
    elif '200m' in name_lower:
        return 200000000
    elif '100m' in name_lower:
        return 100000000
    elif '40m' in name_lower:
        return 40000000
    elif '20m' in name_lower:
        return 20000000
    elif '10m' in name_lower:
        return 10000000
    elif '1m' in name_lower:
        return 1000000
    elif '100k' in name_lower:
        return 100000
    elif '10k' in name_lower:
        return 10000
    elif '1k' in name_lower:
        return 1000
    elif 'random-100' in name_lower:
        return 100
    
    # Special cases
    if 'glove' in name_lower:
        return 1183514  # Standard GloVe size
    elif 'deep-image' in name_lower:
        return 9990000  # Standard deep image size
    elif 'gist' in name_lower:
        return 1000000  # Standard GIST size
    elif 'yandex' in name_lower and '100k' in name_lower:
        return 100000
    elif 'dbpedia' in name_lower:
        return 1000000
    elif 'h-and-m' in name_lower:
        return 105542
    elif 'arxiv' in name_lower:
        return 2205995
    elif 'laion-small-clip' in name_lower:
        return 100000
    elif 'random-match' in name_lower or 'random-range' in name_lower or 'random-geo' in name_lower:
        if '2048' in name_lower:
            return 100000  # 2048D synthetic datasets
        else:
            return 1000000  # 100D synthetic datasets
    elif 'random-100-match' in name_lower:
        return 100  # Small vocab datasets

    # Default for unknown patterns
    return None

def generate_description(name):
    """Generate description from dataset name patterns"""
    name_lower = name.lower()
    
    if 'laion' in name_lower:
        return 'Image embeddings'
    elif 'glove' in name_lower:
        return 'Word vectors'
    elif 'deep-image' in name_lower:
        return 'CNN image features'
    elif 'gist' in name_lower:
        return 'Image descriptors'
    elif 'dbpedia' in name_lower:
        return 'Knowledge embeddings'
    elif 'yandex' in name_lower:
        return 'Text-to-image embeddings'
    elif 'arxiv' in name_lower:
        return 'Academic paper embeddings'
    elif 'h-and-m' in name_lower:
        return 'Fashion product embeddings'
    elif 'random' in name_lower:
        if 'match' in name_lower and 'keyword' in name_lower:
            return 'Synthetic keyword matching'
        elif 'match' in name_lower and 'int' in name_lower:
            return 'Synthetic integer matching'
        elif 'range' in name_lower:
            return 'Synthetic range queries'
        elif 'geo' in name_lower:
            return 'Synthetic geo queries'
        else:
            return 'Synthetic data'
    else:
        return None

def main():
    # Read the datasets.json file
    with open('datasets/datasets.json', 'r') as f:
        datasets = json.load(f)
    
    updated_count = 0
    
    for dataset in datasets:
        updated = False
        
        # Add vector_count if missing
        if 'vector_count' not in dataset:
            vector_count = estimate_vector_count(dataset['name'])
            dataset['vector_count'] = vector_count
            updated = True
            print(f"Added vector_count {vector_count} to {dataset['name']}")
        
        # Add description if missing
        if 'description' not in dataset:
            description = generate_description(dataset['name'])
            dataset['description'] = description
            updated = True
            print(f"Added description '{description}' to {dataset['name']}")
        
        if updated:
            updated_count += 1
    
    # Write back the updated datasets.json
    with open('datasets/datasets.json', 'w') as f:
        json.dump(datasets, f, indent=2)
    
    print(f"\nUpdated {updated_count} datasets")
    print("datasets.json has been updated with missing vector_count and description fields")

if __name__ == "__main__":
    main()
