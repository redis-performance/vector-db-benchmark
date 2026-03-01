# vector-db-benchmark

A benchmarking tool for vector databases, written in Rust. Measures upload throughput, search QPS, latency percentiles (p50/p95/p99), and recall for vector search engines.

## Supported Engines

| Engine | Client Library | Protocol | Distance Metrics | Metadata Filters |
|--------|---------------|----------|-----------------|-----------------|
| **Redis** (RediSearch) | `redis` 0.27 | Redis protocol | L2, Cosine, IP | Yes |
| **VectorSets** | `redis` 0.27 | Redis protocol | L2, Cosine | Yes |
| **Elasticsearch** | `elasticsearch` 8.15 | HTTP/REST | L2, Cosine | Yes |
| **OpenSearch** | `opensearch` 2.3 | HTTP/REST | L2, Cosine | Yes |
| **Qdrant** | `qdrant-client` 1.13 | gRPC | L2, Cosine, Dot | Yes |
| **PgVector** | `postgres` 0.19 + `pgvector` 0.4 | PostgreSQL | L2, Cosine | Yes |
| **Weaviate** | `reqwest` (REST API) | HTTP/REST + GraphQL | L2, Cosine, Dot | Yes |
| **Milvus** | `reqwest` (REST API v2) | HTTP/REST | L2, Cosine, IP | Yes |
| **MongoDB** (Atlas Search) | `mongodb` 3 (sync) | MongoDB protocol | Euclidean, Cosine, Dot | Yes |
| **Valkey** (Valkey Search) | `redis` 0.27 | RESP protocol | L2, Cosine, IP | Yes |

```
docker run --rm --network=host redis/vector-db-benchmark:latest \
  --host localhost --engines 'redis-single*' --datasets glove-25-angular
(...)
============================================================
Running experiment: redis-single-node - glove-25-angular
============================================================
Experiment stage: Configure
Using algorithm hnsw with config {'M': 16, 'EF_CONSTRUCTION': 128}
Experiment stage: Upload
Reading dataset from datasets/glove-25-angular/...
Read 1183514 vectors (25d) in 0.82s
Upload time: 12.3s (96,220 records/sec)
Experiment stage: Search
  Running search 0: ef=128, parallel=4
  → QPS: 3214.5, Precision: 0.9785
  Running search 1: ef=128, parallel=8
  → QPS: 5891.2, Precision: 0.9785
Experiment stage: Done
```

> [View published results](https://redis.io/blog/benchmarking-results-for-vector-databases/)

## Quick Start

### Docker (recommended)

```bash
# Show help
docker run --rm redis/vector-db-benchmark:latest --help

# List available datasets and engines
docker run --rm redis/vector-db-benchmark:latest --describe datasets
docker run --rm redis/vector-db-benchmark:latest --describe engines

# Run a benchmark against a local Redis instance
docker run --rm --network=host \
  -v $(pwd)/datasets:/code/datasets \
  -v $(pwd)/results:/code/results \
  redis/vector-db-benchmark:latest \
  --host localhost --engines 'redis-single*' --datasets glove-25-angular
```

### Using with Redis

```bash
# Start Redis
docker run -d --name redis -p 6379:6379 redis:8.6.0

# Run benchmark
docker run --rm --network=host \
  -v $(pwd)/datasets:/code/datasets \
  -v $(pwd)/results:/code/results \
  redis/vector-db-benchmark:latest \
  --host localhost --engines redis-docker-test --datasets random-100

# Clean up
docker stop redis && docker rm redis
```

### Using Docker Compose

```bash
# Full integration test (downloads h-and-m dataset ~200MB)
make docker-integration

# Fast smoke test (uses random-100 dataset baked into image, 228KB)
make docker-integration-fast
```

### Build from source

```bash
# Prerequisites: Rust toolchain, libhdf5-dev, pkg-config
cargo build --release --bin vector-db-benchmark

# Run
./target/release/vector-db-benchmark --help
./target/release/vector-db-benchmark --describe datasets
./target/release/vector-db-benchmark \
  --host localhost --engines 'redis-single*' --datasets glove-25-angular
```

## CLI Options

```
Usage: vector-db-benchmark [OPTIONS]

Options:
    --engines <PATTERN>        Engine config patterns (wildcards supported) [default: *]
    --engines-file <PATH>      Path to JSON file with custom engine configs
    --datasets <PATTERN>       Dataset patterns (wildcards supported) [default: *]
    --host <HOST>              Redis/engine host [default: localhost]
    --parallels <N,N,...>      Filter by parallel thread counts
    --ef-runtime <N,N,...>     Filter by ef runtime values
    --skip-upload              Skip the upload phase
    --skip-search              Skip the search phase
    --skip-if-exists           Skip if results already exist
    --exit-on-error            Stop on first error
    --timeout <SECS>           Timeout in seconds [default: 86400]
    --describe <TYPE>          Describe available 'datasets' or 'engines'
    -v, --verbose              Verbose output for --describe
    -h, --help                 Print help
```

## Datasets

All datasets are automatically downloaded on first use. The image includes `random-100` (228KB) for quick smoke tests.

| Dataset                                                                                                     | Dimensions |  Train size | Test size | Neighbors | Distance  |
| ----------------------------------------------------------------------------------------------------------- | ---------: |  ---------: | --------: | --------: | --------- |
| **LAION Image Embeddings (512D)**                                                                          |            |             |           |           |           |
| [LAION-1M: subset of LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/)   |        512 |   1,000,000 |    10,000 |       100 | Cosine    |
| [LAION-10M: subset of LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/)  |        512 |  10,000,000 |    10,000 |       100 | Cosine    |
| [LAION-20M: subset of LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/)  |        512 |  20,000,000 |    10,000 |       100 | Cosine    |
| [LAION-40M: subset of LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/)  |        512 |  40,000,000 |    10,000 |       100 | Cosine    |
| [LAION-100M: subset of LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/) |        512 | 100,000,000 |    10,000 |       100 | Cosine    |
| [LAION-200M: subset of LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/) |        512 | 200,000,000 |    10,000 |       100 | Cosine    |
| [LAION-400M: from LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/)      |        512 | 400,000,000 |    10,000 |       100 | Cosine    |
| **LAION Image Embeddings (768D)**                                                                          |            |             |           |           |           |
| [LAION-1M: 768D image embeddings](https://laion.ai/blog/laion-400-open-dataset/)                           |        768 |   1,000,000 |    10,000 |       100 | Cosine    |
| [LAION-1B: 768D image embeddings](https://laion.ai/blog/laion-400-open-dataset/)                           |        768 | 1,000,000,000|   10,000 |       100 | Cosine    |
| **Standard Benchmarks**                                                                                    |            |             |           |           |           |
| [GloVe-25: Word vectors](http://ann-benchmarks.com)                                                        |         25 |   1,183,514 |    10,000 |       100 | Cosine    |
| [GloVe-100: Word vectors](http://ann-benchmarks.com)                                                       |        100 |   1,183,514 |    10,000 |       100 | Cosine    |
| [Deep Image-96: CNN image features](http://ann-benchmarks.com)                                             |         96 |   9,990,000 |    10,000 |       100 | Cosine    |
| [GIST-960: Image descriptors](http://ann-benchmarks.com)                                                   |        960 |   1,000,000 |     1,000 |       100 | L2        |
| **Text and Knowledge Embeddings**                                                                          |            |             |           |           |           |
| [DBpedia OpenAI-1M: Knowledge embeddings](https://www.dbpedia.org/)                                       |      1,536 |   1,000,000 |    10,000 |       100 | Cosine    |
| [LAION Small CLIP: Small CLIP embeddings](https://laion.ai/blog/laion-400-open-dataset/)                   |        512 |     100,000 |     1,000 |       100 | Cosine    |
| **Yandex Datasets**                                                                                        |            |             |           |           |           |
| [Yandex T2I: Text-to-image embeddings](https://research.yandex.com/)                                      |        200 |   1,000,000 |   100,000 |       100 | Dot       |
| **Random and Synthetic**                                                                                   |            |             |           |           |           |
| Random-100: Small synthetic dataset                                                                        |        100 |         100 |         9 |         9 | Cosine    |
| Random-100-Euclidean: Small synthetic dataset                                                              |        100 |         100 |         9 |         9 | L2        |
| **Filtered Search Datasets**                                                                               |            |             |           |           |           |
| H&M-2048: Fashion product embeddings (with filters)                                                        |      2,048 |     105,542 |     2,000 |       100 | Cosine    |
| H&M-2048: Fashion product embeddings (no filters)                                                          |      2,048 |     105,542 |     2,000 |       100 | Cosine    |
| ArXiv-384: Academic paper embeddings (with filters)                                                        |        384 |   2,205,995 |    10,000 |       100 | Cosine    |
| ArXiv-384: Academic paper embeddings (no filters)                                                          |        384 |   2,205,995 |    10,000 |       100 | Cosine    |
| Random Match Keyword-100: Synthetic keyword matching (with filters)                                        |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Match Keyword-100: Synthetic keyword matching (no filters)                                          |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Match Int-100: Synthetic integer matching (with filters)                                            |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Match Int-100: Synthetic integer matching (no filters)                                              |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Range-100: Synthetic range queries (with filters)                                                   |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Range-100: Synthetic range queries (no filters)                                                     |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Geo Radius-100: Synthetic geo queries (with filters)                                                |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Geo Radius-100: Synthetic geo queries (no filters)                                                  |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Match Keyword-2048: Large synthetic keyword matching (with filters)                                 |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Match Keyword-2048: Large synthetic keyword matching (no filters)                                   |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Match Int-2048: Large synthetic integer matching (with filters)                                     |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Match Int-2048: Large synthetic integer matching (no filters)                                       |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Range-2048: Large synthetic range queries (with filters)                                            |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Range-2048: Large synthetic range queries (no filters)                                              |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Geo Radius-2048: Large synthetic geo queries (with filters)                                         |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Geo Radius-2048: Large synthetic geo queries (no filters)                                           |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Match Keyword Small Vocab-256: Small vocabulary keyword matching (with filters)                     |        256 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Match Keyword Small Vocab-256: Small vocabulary keyword matching (no filters)                       |        256 |   1,000,000 |    10,000 |       100 | Cosine    |

## Engine Configurations

Engine configurations live in [`experiments/configurations/`](./experiments/configurations/). Each JSON file defines one or more experiment configurations specifying the engine, index parameters, search parameters, and upload parallelism.

Example (`redis-docker-test.json`):

```json
[
  {
    "name": "redis-docker-test",
    "engine": "redis",
    "connection_params": {},
    "collection_params": {
      "hnsw_config": { "M": 16, "EF_CONSTRUCTION": 128 }
    },
    "search_params": [
      { "parallel": 1, "search_params": { "ef": 128 } }
    ],
    "upload_params": { "parallel": 8 }
  }
]
```

Use `--engines` with wildcard patterns to select configurations:

```bash
vector-db-benchmark --engines 'redis-single*' --datasets 'glove*'
vector-db-benchmark --engines 'vectorsets*' --datasets 'h-and-m*'
vector-db-benchmark --engines 'elasticsearch*' --datasets 'glove*'
vector-db-benchmark --engines 'qdrant*' --datasets 'deep-image*'
```

Or provide a custom file with `--engines-file`:

```bash
vector-db-benchmark --engines-file my_engines.json --datasets glove-25-angular
```

## How to register a dataset?

Datasets are configured in [`datasets/datasets.json`](./datasets/datasets.json). The tool automatically downloads datasets on first use if a download link is provided.

## Development

### Prerequisites

- **Rust toolchain** (install via [rustup](https://rustup.rs/))
- **libhdf5-dev** and **pkg-config**

```bash
# Ubuntu/Debian
sudo apt-get install libhdf5-dev pkg-config

# macOS
brew install hdf5 pkg-config
```

### Build and test

```bash
make build              # Build release binary
make test               # Run unit tests
make check              # Clippy + rustfmt
make docker-build       # Build Docker image
```

### Integration tests

Each engine has a dedicated integration test that runs against a Docker container:

```bash
make integration-test                  # Redis (default)
make integration-test-elasticsearch    # Elasticsearch 8.10.2
make integration-test-opensearch       # OpenSearch 2.19.2
make integration-test-pgvector         # PgVector (PostgreSQL 16)
make integration-test-qdrant           # Qdrant v1.13.4
make integration-test-weaviate         # Weaviate 1.28.9
make integration-test-milvus           # Milvus 2.5.6
make integration-test-mongodb          # MongoDB Atlas Local 8.0.4
make integration-test-valkey           # Valkey Bundle (latest)
```

Each target starts the engine via `docker compose -f tests/docker-compose.test.yml`, runs the tests, then stops the container.

### Project structure

```
src/
  lib.rs                              # Library: readers, data formats
  readers/                            # HDF5, NPY, JSONL, compound readers
  bin/
    vector_db_benchmark/
      main.rs                         # CLI entry point
      config.rs                       # Configuration loading
      dataset.rs                      # Dataset resolution and reading
      experiment.rs                   # Experiment runner with calibration
      engine/
        mod.rs                        # Engine trait and factory
        redis.rs                      # Redis (RediSearch) engine
        vectorsets.rs                 # VectorSets engine
        elasticsearch.rs              # Elasticsearch engine
        opensearch.rs                 # OpenSearch engine
        qdrant.rs                     # Qdrant engine (gRPC)
        pgvector.rs                   # PgVector engine (PostgreSQL)
        weaviate.rs                   # Weaviate engine (REST)
        milvus.rs                     # Milvus engine (REST)
        mongodb_engine.rs             # MongoDB Atlas Search engine
        valkey.rs                     # Valkey engine (RESP protocol)
experiments/configurations/           # Engine configuration JSON files
datasets/datasets.json                # Dataset definitions
tests/
  docker-compose.test.yml             # Docker services for integration tests
  integration_redis.rs                # Redis integration tests
  integration_elasticsearch.rs        # Elasticsearch integration tests
  integration_opensearch.rs           # OpenSearch integration tests
  integration_pgvector.rs             # PgVector integration tests
  integration_qdrant.rs               # Qdrant integration tests
  integration_weaviate.rs             # Weaviate integration tests
  integration_milvus.rs               # Milvus integration tests
  integration_mongodb.rs              # MongoDB integration tests
  integration_valkey.rs               # Valkey integration tests
```
