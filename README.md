# vector-db-benchmark

A benchmarking tool for vector databases, written in Rust. Measures upload throughput, search QPS, latency percentiles (p50/p95/p99), and recall for vector search engines.

## Supported Engines

| Engine | Client Library | Protocol | Distance Metrics | Metadata Filters |
|--------|---------------|----------|-----------------|-----------------|
| **Redis** (RediSearch) | `redis` 1.3 | Redis protocol | L2, Cosine, IP | Yes |
| **VectorSets** | `redis` 1.3 | Redis protocol | L2, Cosine | Yes |
| **Elasticsearch** | `elasticsearch` 8.15 | HTTP/REST | L2, Cosine | Yes |
| **OpenSearch** | `opensearch` 2.4 | HTTP/REST | L2, Cosine | Yes |
| **Qdrant** | `qdrant-client` 1.18 | gRPC | L2, Cosine, Dot | Yes |
| **PgVector** | `postgres` 0.19 + `pgvector` 0.4 | PostgreSQL | L2, Cosine | Yes |
| **Weaviate** | `tonic` 0.12 / `prost` 0.13 (gRPC) + `reqwest` (REST) | gRPC (search) + HTTP/REST (schema) [\*\*](#weaviate-protocol-note) | L2, Cosine, Dot | Yes |
| **Milvus** | `reqwest` (REST API v2) | HTTP/REST | L2, Cosine, IP | Yes |
| **MongoDB** (Atlas Search) | `mongodb` 3 (sync) | MongoDB protocol | Euclidean, Cosine, Dot | Yes |
| **Valkey** (Valkey Search) | `redis` 1.3 [\*](#valkey-client-note) | RESP protocol | L2, Cosine, IP | Yes |
| **Turbopuffer** | `turbopuffer-client` 0.0.4 | HTTP/REST (cloud) | Cosine, Euclidean | Yes |
| **Dragonfly** (Dragonfly Search) | `redis` 1.3 | RESP protocol | L2, Cosine, IP | No [\*\*\*](#dragonfly-note) |

<a id="valkey-client-note"></a>
\* **Valkey client note:** Valkey GLIDE has no published Rust crate ([valkey-io/valkey-glide#828](https://github.com/valkey-io/valkey-glide/issues/828), closed NOT_PLANNED). The GLIDE maintainers recommend using `redis-rs` for Rust and upstream their improvements to it. The `redis` crate works with Valkey since it speaks the same RESP protocol.

<a id="weaviate-protocol-note"></a>
\*\* **Weaviate protocol note:** Vector search runs over Weaviate's **gRPC** API (port 50051) by default — the high-throughput query path used by the official clients (packed binary vectors). Schema management, upload, and search-time `ef` tuning use the REST API (v1). Filtered searches also run over gRPC — metadata filter conditions are translated to the gRPC `Filters` message. The tool falls back to the slower GraphQL-over-HTTP search path only when a filter condition can't be expressed in gRPC or when `WEAVIATE_USE_GRAPHQL` is set. Override the gRPC port with `WEAVIATE_GRPC_PORT`.

<a id="dragonfly-note"></a>
\*\*\* **Dragonfly note:** Uses **Dragonfly Search** (Beta), the RediSearch-compatible `FT.*` subset Dragonfly ships (`FT.CREATE`/`FT.SEARCH`/`FT.INFO`/`FT.DROPINDEX`, `VECTOR` FLAT/HNSW, `*=>[KNN k @field $blob AS score]`). This engine implements **pure vector KNN only** — no metadata filters, no full-text, no mixed workload, no quantization. Dragonfly Search supports only the **float32** vector type, so vectors are always encoded as FLOAT32. Runs over the RESP protocol via Docker; set the host port with `DRAGONFLY_PORT` (default `6385`). Upload concurrency/batch come from the engine config but can be overridden at runtime with `DRAGONFLY_UPLOAD_PARALLEL` / `DRAGONFLY_UPLOAD_BATCH_SIZE` (env takes precedence) — managed **Dragonfly Cloud** resets connections under the default 100-thread upload burst on larger-dimensional datasets, so cloud runs set `DRAGONFLY_UPLOAD_PARALLEL=16`; search throughput is unaffected by upload concurrency.

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
docker run -d --name redis -p 6379:6379 redis:8.8.0

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
    --update-search-ratio <U:S> Mixed benchmark: interleave U updates per S searches
    --describe <TYPE>          Describe available 'datasets' or 'engines'
    -v, --verbose              Verbose output for --describe
    --plot <OUTPUT.svg>        Render a QPS-vs-precision trade-off chart from results/
    -h, --help                 Print help
```

### Charts

Render a QPS-vs-precision trade-off plot (SVG, no dependencies) from existing `*-summary.json` results — one colored series per engine, filtered by `--engines`/`--datasets`:

```bash
# Compare all engines on one dataset
vector-db-benchmark --plot tradeoff.svg --engines '*' --datasets glove-100-angular
```

## Mixed Benchmarks (Update + Search)

The `--update-search-ratio` flag enables mixed workload benchmarks that interleave vector updates with searches. This measures how search performance is affected by concurrent write operations.

```bash
# 1 update per 10 searches
vector-db-benchmark --engines redis-docker-test --datasets random-100 \
  --update-search-ratio 1:10

# 1 update per 5 searches (heavier write load)
vector-db-benchmark --engines vectorsets-docker-test --datasets h-and-m-2048-angular \
  --update-search-ratio 1:5
```

The ratio format is `U:S` where U = number of updates and S = number of searches per cycle. Each worker thread performs S searches followed by U updates in a loop.

**Supported engines**: Redis, VectorSets, Valkey

Results JSON includes separate metrics for both operation types:

```json
{
  "results": {
    "rps": 5891.2,
    "precision": 0.9785,
    "p50_time": 0.00032,
    "p95_time": 0.00089,
    "p99_time": 0.00142,
    "update_rps": 589.1,
    "update_mean_time": 0.00045,
    "update_p50_time": 0.00041,
    "update_p95_time": 0.00098,
    "update_p99_time": 0.00156,
    "update_search_ratio": "1:10"
  }
}
```

Omitting the flag preserves the standard search-only benchmark behavior.

## Multi-tenancy

Multi-tenancy benchmarks model many tenants sharing **one** index: every search is scoped to a single tenant via an exact keyword-equality filter on a `tenant` field (`schema: { "tenant": "keyword" }`), and recall is measured against the nearest neighbours **within that tenant only**. This reuses the standard keyword-TAG filter path (no engine-specific code) and mirrors upstream qdrant/vector-db-benchmark's `random-768-*-tenants` scenario (registered here as `random-768-25-tenants`). The per-query filter looks like `{"and":[{"tenant":{"match":{"value":"tenant_7"}}}]}`. Because ground truth is tenant-local, recall is a strong isolation signal — a leaked cross-tenant document displaces a correct neighbour and lowers recall — and the tests assert **exact** per-query recall (`== 1.0` against an exact search, one query per tenant), so any single cross-tenant leak fails the check. Redis and Valkey are covered end-to-end (over both RESP2 and RESP3) by the `test_binary_{redis,valkey}_tenancy` integration tests. (Recall is necessary but not by itself sufficient to *prove* zero leakage; strict per-id membership checking is a possible future hardening.)

## Datasets

Most datasets are automatically downloaded on first use. The image includes `random-100` (228KB) for quick smoke tests. (Exception: `random-768-25-tenants` is a locally-generated placeholder with no public download link yet — see the Multi-tenancy section.)

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

### Generating local datasets

The sparse-vector, hybrid (dense+sparse fusion), and multi-datatype filter code
paths ship with **locally-generated** synthetic datasets — small, deterministic
(fixed-seed) fixtures with **no public download link**. Generate them once with:

```bash
cargo run --release --bin generate-dataset          # writes into ./datasets
# or a subset / custom location:
cargo run --release --bin generate-dataset -- --only sparse --out-dir /tmp/ds
```

This writes three datasets under `datasets/` (each in the exact on-disk layout
its reader expects), registered in [`datasets/datasets.json`](./datasets/datasets.json):

| Dataset                | Type     | Dims | Distance | Layout                                                                                  |
| ---------------------- | -------- | ---: | -------- | --------------------------------------------------------------------------------------- |
| `synthetic-sparse-300` | `sparse` |  300 | dot      | `data.csr` + `queries.csr` + `neighbours.jsonl` (dot/MIPS ground truth)                 |
| `synthetic-hybrid-16`  | `hybrid` |   16 | l2       | `vectors.npy` + `queries.npy` + `data.csr` + `queries.csr` + shared `neighbours.jsonl`  |
| `synthetic-filter-32`  | `tar`    |   32 | l2       | `vectors.npy` + `payloads.jsonl` + `tests.jsonl` (per-query `conditions` + filtered GT) |

`synthetic-filter-32`'s per-query `conditions` rotate through **keyword**, **int**,
**bool** and **datetime** filters (schema `color:keyword, size:int, flag:bool, ts:datetime`),
each with ground truth brute-forced over only the matching documents, so a high
recall proves the engine actually applied the filter. The generated files are
git-ignored — regenerate them on any checkout with the command above.

Example runs against a generated dataset (start the engine first, e.g.
`docker compose -f tests/docker-compose.test.yml up -d qdrant redis`):

```bash
# Sparse (Qdrant):
cargo run --release --bin vector-db-benchmark -- \
  --engines qdrant-default --datasets synthetic-sparse-300
# Hybrid dense+sparse fusion (Qdrant):
cargo run --release --bin vector-db-benchmark -- \
  --engines qdrant-hybrid --datasets synthetic-hybrid-16
# Filter datatypes (Redis):
cargo run --release --bin vector-db-benchmark -- \
  --engines redis-docker-test --datasets synthetic-filter-32
```

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

### Qdrant hybrid (dense + sparse) search

`experiments/configurations/qdrant-hybrid.json` runs Qdrant's server-side
reciprocal-rank fusion (RRF) of a dense-vector prefetch and a sparse-vector
prefetch. It **requires a `type: "hybrid"` dataset** — running it against an
ordinary dense dataset silently degrades to a plain dense search (there is no
sparse vector to fuse). A hybrid dataset directory must contain all of:

```
vectors.npy      # dense document vectors  (npy, row i == point id i)
queries.npy      # dense query vectors      (npy)
data.csr         # sparse document vectors  (binary CSR, row-aligned with vectors.npy)
queries.csr      # sparse query vectors     (binary CSR, row-aligned with queries.npy)
neighbours.jsonl # fused ground truth: one JSON array of ids per query line
```

Register it in `datasets/datasets.json` with `"type": "hybrid"` and the dense
`vector_size`/`distance`. The end-to-end hybrid path (collection with a named
`dense` + named `sparse` vector, dual-vector upsert, and RRF fusion) is covered
by `tests/integration_qdrant.rs::test_binary_qdrant_hybrid`, which also
generates a tiny hybrid fixture you can consult for the exact layout.

## How to register a dataset?

Datasets are configured in [`datasets/datasets.json`](./datasets/datasets.json). The tool automatically downloads datasets on first use if a download link is provided.

## Development

### Prerequisites

The quickest way to install all dependencies (Linux/macOS):

```bash
make setup    # installs libhdf5, pkg-config, and Rust toolchain
```

Or install manually:

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
make integration-test                  # Redis 8.8.0 (default)
make integration-test-elasticsearch    # Elasticsearch 9.4.3
make integration-test-opensearch       # OpenSearch 3.7.0
make integration-test-pgvector         # PgVector (PostgreSQL 18)
make integration-test-qdrant           # Qdrant v1.18.2
make integration-test-weaviate         # Weaviate 1.38.2
make integration-test-milvus           # Milvus v2.6.19
make integration-test-mongodb          # MongoDB Atlas Local 8.0.17
make integration-test-valkey           # Valkey Bundle (latest)
```

Each target starts the engine via `docker compose -f tests/docker-compose.test.yml`, runs the tests, then stops the container.

### Fuzzing

The untrusted dataset parsers (sparse CSR, NPY, JSONL, and metadata/JSON readers) are fuzzed with [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz) / libFuzzer to ensure malformed input returns `Err` instead of panicking/overflowing/OOMing. Run locally with a nightly toolchain:

```bash
cargo +nightly fuzz run sparse_reader -- -max_total_time=60 -rss_limit_mb=2048
```

A nightly GitHub Actions workflow (`.github/workflows/fuzz.yml`) fuzzes each parser at higher effort. See [`fuzz/README.md`](fuzz/README.md) for details.

**Turbopuffer** is cloud-only and requires an API key:
```bash
TURBOPUFFER_API_KEY=your-key ./target/release/vector-db-benchmark \
  --engines 'turbopuffer*' --datasets random-100
```

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
        weaviate_grpc.rs              # Weaviate gRPC client (generated from vendored v1 protos)
        weaviate.rs                   # Weaviate engine (gRPC search + REST schema)
        milvus.rs                     # Milvus engine (REST)
        mongodb_engine.rs             # MongoDB Atlas Search engine
        valkey.rs                     # Valkey engine (RESP protocol)
        turbopuffer.rs                # Turbopuffer engine (cloud API)
        redis_utils.rs                # Shared utils for Redis-protocol engines
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
