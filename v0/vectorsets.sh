#!/bin/bash

# Check if hostname is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <hostname>"
    exit 1
fi

hostname=$1

# Define experiments array
experiments=(
    "vectorsets-bin-default"
    "vectorsets-bin-m-32-ef-128"
    "vectorsets-bin-m-32-ef-256"
    "vectorsets-bin-m-32-ef-512"
    "vectorsets-bin-m-64-ef-256"
    "vectorsets-bin-m-64-ef-512"
    
    "vectorsets-q8-default"
    "vectorsets-q8-m-32-ef-128"
    "vectorsets-q8-m-32-ef-256"
    "vectorsets-q8-m-32-ef-512"
    "vectorsets-q8-m-64-ef-256"
    "vectorsets-q8-m-64-ef-512"
    
    "vectorsets-fp32-default"
    "vectorsets-fp32-m-32-ef-128"
    "vectorsets-fp32-m-32-ef-256"
    "vectorsets-fp32-m-32-ef-512"
    "vectorsets-fp32-m-64-ef-256"
    "vectorsets-fp32-m-64-ef-512"
)

# Run command for each experiment
for experiment in "${experiments[@]}"; do
    echo "Running experiment: $experiment"
    python run.py --engines "$experiment" --datasets dbpedia-openai-1M-1536-angular --host "$hostname"
    echo "Completed experiment: $experiment"
    echo "-----------------------------------"
done

echo "All experiments completed!"