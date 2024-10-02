#!/bin/bash

set -x

part_size=10000000  # 10 million elements per part

# Create the output directory if it doesn't exist

for i in {0..99}; do
  # Calculate the start and end indices for each part
  start_idx=$((i * part_size))
  end_idx=$(((i + 1) * part_size))

  # Launch each process in a new screen session
  screen -dmS loader_$i bash -c "REDIS_PORT=30001 REDIS_CLUSTER=1  python3 run.py --host 192.168.2.6  --engines redis-intel-float16-hnsw-m-16-ef-32 --datasets laion-img-emb-768d-1Billion-cosine  --skip-search --upload-start-idx $start_idx --upload-end-idx $end_idx "
done
