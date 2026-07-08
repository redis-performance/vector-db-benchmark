# Tech Stack

## Overview
Rust CLI tool for benchmarking vector database engines. Designed for performance-critical workloads with parallel execution and large dataset handling.

## Languages
Rust (edition 2021)

Chosen for performance, memory safety, and strong type system — critical for a benchmarking tool where measurement overhead must be minimal.

## Framework
clap 4.5 (derive macros) for CLI argument parsing.

Multiple binary targets:
- `vector-db-benchmark` — main entry point
- `bench_jsonl` — JSONL format benchmarks
- `bench_npy` — NPY format benchmarks

## Key Libraries

| Library | Purpose |
|---------|---------|
| redis 0.27 (cluster) | Redis/RediSearch client with cluster support |
| rayon 1.10 | Data parallelism for concurrent operations |
| hdf5 0.8.1 | HDF5 dataset reading |
| ndarray 0.16 / ndarray-npy 0.9 | N-dimensional arrays and NPY format support |
| serde / serde_json | Configuration and data serialization |
| indicatif 0.18 | Progress bars for long-running operations |
| reqwest 0.12 (blocking) | HTTP client for dataset downloads + Weaviate/Milvus engines |
| flate2 / tar | Archive extraction for datasets |
| half 2.4 | Half-precision float support |
| chrono 0.4 | Timestamps |
| glob 0.3 | File pattern matching |
| tokio 1 (rt-multi-thread) | Async runtime for async-only engine clients |
| qdrant-client 1.13 | Official Qdrant gRPC client |
| elasticsearch 8.15.0-alpha.1 | Official Elasticsearch client |
| opensearch 2.3.0 | Official OpenSearch client |
| postgres 0.19 | PostgreSQL sync client |
| pgvector 0.4 | pgvector extension support for postgres crate |
| mongodb 3 (sync) | Official MongoDB sync client |

## Infrastructure & Deployment
Runs locally against database instances managed via Docker Compose. No cloud deployment — the tool is executed on the benchmarking machine directly.

## Package Manager
Cargo (Rust standard)

## Decision Relationships
- Rust + rayon enables low-overhead parallel benchmark execution
- hdf5 + ndarray provides efficient large dataset handling (standard ANN benchmark formats)
- redis crate with cluster feature supports both standalone and clustered Redis configurations
