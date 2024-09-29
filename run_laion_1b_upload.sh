#!/bin/bash

set -x

part_size=10000000  # 10 million elements per part

# Create the output directory if it doesn't exist

for i in {0..99}; do
  # Calculate the start and end indices for each part
  start_idx=$((i * part_size))
  end_idx=$(((i + 1) * part_size))
  
  # Launch each process in a new screen session
  screen -dmS split_process_$i bash -c "python3 run.py --engines redis-intel-float16-hnsw-m-16-ef-32 --dataset laion-img-emb-768d-1Billion-cosine  --skip-search --upload_start_idx $start_idx --upload_end_idx $end_idx --part $((i + 1))"
done

