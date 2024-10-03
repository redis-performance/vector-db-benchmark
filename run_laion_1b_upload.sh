#!/bin/bash

set -x

part_size=10000000  # 10 million elements per part
max_screens=50       # Maximum number of screens running simultaneously

# Function to wait until the number of running screens is below the limit
wait_for_available_screen_slot() {
    echo "waiting for avaiable screen."
    while [ "$(screen -ls | grep -c loader_)" -ge "$max_screens" ]; do
        sleep 15  # Wait for 15 seconds before checking again
    done
}

# Create the output directory if it doesn't exist

for i in {10..99}; do
  # Wait until there's an available screen slot
  wait_for_available_screen_slot

  # Calculate the start and end indices for each part
  start_idx=$((i * part_size))
  end_idx=$(((i + 1) * part_size))

  # Launch each process in a new screen session
  screen -dmS loader_$i bash -c "REDIS_PORT=30001 REDIS_CLUSTER=1 python3 run.py --host 192.168.2.6 --engines redis-intel-float16-hnsw-m-16-ef-32 --datasets laion-img-emb-768d-1Billion-cosine --skip-search --upload-start-idx $start_idx --upload-end-idx $end_idx"

  # Print progress
  echo "Started screen loader_$i: uploading indices $start_idx to $end_idx"
  echo "$((i+1))/100 processes started"
done
