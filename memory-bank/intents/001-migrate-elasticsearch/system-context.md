---
intent: 001-migrate-elasticsearch
phase: inception
status: context-defined
updated: 2026-02-27T00:00:00Z
---

# Migrate Elasticsearch - System Context

## System Overview

Add Elasticsearch as a benchmark engine to the Rust vector-db-benchmark CLI. The engine connects to an Elasticsearch instance via REST API to create vector indices, upload vectors, and execute KNN searches.

## Context Diagram

```mermaid
C4Context
    title System Context - Elasticsearch Engine Migration

    Person(dev, "Benchmark Operator", "Runs benchmarks via CLI")
    System(bench, "vector-db-benchmark", "Rust CLI benchmarking tool")
    System_Ext(es, "Elasticsearch 8.x", "Vector search engine via REST API")
    System_Ext(datasets, "Dataset Files", "HDF5, JSONL, NPY, compound formats")

    Rel(dev, bench, "Runs benchmark commands")
    Rel(bench, es, "REST API: index create, bulk upload, KNN search, delete")
    Rel(bench, datasets, "Reads vectors, queries, neighbors")
```

## External Integrations

- **Elasticsearch 8.x**: Target vector database via REST API (port 9200). Operations: index management, bulk document upload, KNN vector search, force merge.
- **Dataset Files**: Local dataset files in HDF5/JSONL/NPY/compound formats providing vectors, queries, and ground truth neighbors.

## High-Level Constraints

- Must connect to Elasticsearch via HTTP REST (not transport protocol)
- Must use same index name convention as Python v0 (`bench` default)
- Must produce results JSON compatible with existing precision comparison tooling

## Key NFR Goals

- Search precision must match Python v0 for identical datasets and parameters
- Upload and search must support parallel execution via Rust threads
