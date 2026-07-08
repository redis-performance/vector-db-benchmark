---
intent: 011-mixed-benchmark
phase: inception
status: context-defined
updated: 2026-03-05T10:00:00Z
---

# Mixed Benchmark - System Context

## System Overview

Extends the vector-db-benchmark CLI to support mixed workloads where vector updates are interleaved with searches. This measures how concurrent writes impact search latency and throughput — a critical real-world scenario for production vector databases.

## Context Diagram

```text
                    ┌─────────────────────────────────┐
                    │     vector-db-benchmark CLI      │
                    │                                  │
  --update-search   │  ┌──────────┐  ┌─────────────┐  │
  -ratio 1:10  ───> │  │ CLI Args │  │ Experiment  │  │
                    │  │ (cli.rs) │─>│  Runner      │  │
  --engines redis   │  └──────────┘  │(experiment.rs│  │
  --datasets glove  │                └──────┬───────┘  │
                    │                       │          │
                    │        ┌───────────────┼──────┐   │
                    │        │    Engine Trait       │   │
                    │        │  search() + update()  │   │
                    │        └───┬───────┬───────┬──┘   │
                    │            │       │       │      │
                    │     ┌──────┴┐ ┌────┴───┐ ┌─┴────┐ │
                    │     │Redis  │ │VSetsets│ │Valkey│ │
                    │     │Engine │ │Engine  │ │Engine│ │
                    │     └──┬───┘ └───┬────┘ └──┬───┘ │
                    └────────┼─────────┼─────────┼─────┘
                             │         │         │
                     ┌───────┴──┐  ┌───┴──┐  ┌───┴──┐
                     │ Redis    │  │Redis │  │Valkey│
                     │ Server   │  │Server│  │Server│
                     │(FT.*/HSET│  │(VADD/│  │(FT.*/│
                     │  )       │  │VSIM) │  │ HSET)│
                     └──────────┘  └──────┘  └──────┘
```

## External Integrations

- **Redis Server**: HSET for updates, FT.SEARCH for search (existing)
- **VectorSets (Redis module)**: VADD for upsert, VSIM for search (existing)
- **Valkey Server**: HSET for updates, FT.SEARCH for search (existing)
- **Dataset files**: HDF5/parquet files with vectors, metadata, queries, neighbors (existing)

## High-Level Constraints

- Must not break existing search-only benchmark behavior (flag is optional)
- Must reuse existing parallel worker thread infrastructure (`thread::scope` + `AtomicUsize`)
- Update data comes from the same dataset used for ingestion (no separate dataset needed)
- Initial engine support: Redis, VectorSets, Valkey only

## Key NFR Goals

- Framework overhead for mixed mode < 5% vs search-only (excluding actual update I/O)
- Deterministic update sequence (seeded PRNG, reproducible across runs)
- Separate metrics reporting (search vs update latencies)
