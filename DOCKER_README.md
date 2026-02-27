# vector-db-benchmark

A Rust-based benchmarking tool for vector databases. Measures upload throughput, search QPS, latency percentiles, and recall for various vector search engines.

## Supported Engines

- **Redis** (RediSearch) — HNSW via `FT.CREATE` / `FT.SEARCH`
- **VectorSets** — Redis VectorSets via `VADD` / `VSIM`
- **Elasticsearch** — dense vector with HNSW

## Quick Start

```bash
# Show help
docker run --rm redis/vector-db-benchmark --help

# List available datasets
docker run --rm redis/vector-db-benchmark --describe datasets

# List available engine configurations
docker run --rm redis/vector-db-benchmark --describe engines
```

## Running a Benchmark

```bash
# Start Redis
docker run -d --name redis -p 6379:6379 redis:8.6.0

# Run benchmark against localhost Redis
docker run --rm --network host \
  -v $(pwd)/datasets:/code/datasets \
  -v $(pwd)/results:/code/results \
  redis/vector-db-benchmark \
  --host localhost --engines 'redis*' --datasets 'glove-25-angular'
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--engines <PATTERN>` | Engine patterns to run (supports wildcards) | `*` |
| `--datasets <PATTERN>` | Dataset patterns to run (supports wildcards) | `*` |
| `--host <HOST>` | Redis / engine host | `localhost` |
| `--skip-upload` | Skip the upload phase | — |
| `--skip-search` | Skip the search phase | — |
| `--skip-if-exists` | Skip if results already exist | — |
| `--parallels <N,N,...>` | Filter by parallel thread counts | — |
| `--ef-runtime <N,N,...>` | Filter by ef runtime values | — |
| `--describe <TYPE>` | Describe `datasets` or `engines` | — |
| `-v, --verbose` | Verbose output for `--describe` | — |
| `--exit-on-error` | Stop on first error | — |
| `--timeout <SECS>` | Timeout in seconds | `86400` |

## Volume Mounts

| Container Path | Purpose |
|----------------|---------|
| `/code/datasets` | Dataset files (HDF5, NPY, JSONL) |
| `/code/results` | Benchmark result JSON files |

The image includes the `random-100` dataset (228KB) for quick smoke tests. For larger datasets, mount your local `datasets/` directory.

## Docker Compose Example

```yaml
services:
  redis:
    image: redis:8.6.0
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 1s
      timeout: 3s
      retries: 10

  benchmark:
    image: redis/vector-db-benchmark:latest
    depends_on:
      redis:
        condition: service_healthy
    volumes:
      - ./datasets:/code/datasets
      - ./results:/code/results
    command:
      - "--engines"
      - "redis*"
      - "--datasets"
      - "glove-25-angular"
      - "--host"
      - "redis"
```

## Source Code

[github.com/RedisLabs/vector-db-benchmark](https://github.com/RedisLabs/vector-db-benchmark)
