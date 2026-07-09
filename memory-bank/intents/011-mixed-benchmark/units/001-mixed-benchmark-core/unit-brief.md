---
unit: 001-mixed-benchmark-core
intent: 011-mixed-benchmark
phase: inception
status: ready
created: 2026-03-05T10:00:00Z
updated: 2026-03-05T10:00:00Z
---

# Unit Brief: Mixed Benchmark Core

## Purpose

Add mixed workload benchmark capability — interleaved vector updates with searches at a configurable ratio — to measure search performance under write pressure.

## Scope

### In Scope
- CLI flag `--update-search-ratio U:S` parsing and validation
- Engine trait `update()` method with default "not supported" implementation
- Interleaved worker loop that alternates S searches and U updates per cycle
- Deterministic update sequence via seeded PRNG (seed 42)
- Separate update/search metrics in results JSON
- `update()` implementations for Redis, VectorSets, Valkey
- Precision invariance validation (same recall with or without updates)

### Out of Scope
- Other engines (Qdrant, ES, OpenSearch, PgVector, Milvus, MongoDB, Weaviate, Turbopuffer)
- Update with different/new vectors (only re-insert same data)
- Separate update dataset files
- Parallel update-only benchmarks (updates are interleaved with searches)

---

## Assigned Requirements

| FR | Requirement | Priority |
|----|-------------|----------|
| FR-1 | CLI Flag `--update-search-ratio` | Must |
| FR-2 | Interleaved Execution | Must |
| FR-3 | Update Operation (Vector + Metadata) | Must |
| FR-4 | Deterministic Update Sequence | Must |
| FR-5 | Separate Metrics Reporting | Must |
| FR-6 | Engine Support — Redis | Must |
| FR-7 | Engine Support — VectorSets | Must |
| FR-8 | Engine Support — Valkey | Must |
| FR-9 | Results JSON Schema Extension | Must |
| FR-10 | Engine Trait Extension | Must |
| FR-11 | Precision Invariance | Must |

---

## Domain Concepts

### Key Entities
| Entity | Description | Attributes |
|--------|-------------|------------|
| UpdateSearchRatio | Parsed ratio from CLI | updates: u64, searches: u64 |
| UpdateSequence | Seeded PRNG sequence of vector IDs | seed: u64, rng: StdRng |
| MixedSearchResults | Extended SearchResults | search metrics + update metrics |

### Key Operations
| Operation | Description | Inputs | Outputs |
|-----------|-------------|--------|---------|
| parse_ratio | Parse "U:S" string | "1:10" | (1, 10) |
| update | Upsert vector + metadata by ID | id, vector, metadata | Ok/Err |
| interleaved_loop | Worker loop alternating S searches + U updates | ratio, queries, vectors | MixedSearchResults |

---

## Story Summary

| Metric | Count |
|--------|-------|
| Total Stories | 8 |
| Must Have | 8 |
| Should Have | 0 |
| Could Have | 0 |

### Stories

| Story ID | Title | Priority | Status |
|----------|-------|----------|--------|
| 001-cli-flag | CLI flag parsing | Must | Planned |
| 002-engine-trait | Engine trait update method | Must | Planned |
| 003-interleaved-loop | Interleaved worker loop | Must | Planned |
| 004-metrics-reporting | Separate metrics and results JSON | Must | Planned |
| 005-redis-update | Redis engine update() | Must | Planned |
| 006-vectorsets-update | VectorSets engine update() | Must | Planned |
| 007-valkey-update | Valkey engine update() | Must | Planned |
| 008-precision-invariance | Precision invariance validation | Must | Planned |

---

## Dependencies

### Depends On
None — all target engines already exist and work.

### Depended By
None.

### External Dependencies
| System | Purpose | Risk |
|--------|---------|------|
| Redis 8.6.0 | HSET upsert for updates | Low (proven) |
| VectorSets module | VADD upsert semantics | Low (confirmed) |
| Valkey Server | HSET upsert for updates | Low (same as Redis) |

---

## Technical Context

### Suggested Technology
- `rand` crate with `SeedableRng` for deterministic update sequence
- Existing `thread::scope` + `AtomicUsize` pattern for parallel workers
- `clap` for CLI flag parsing (already used)

### Integration Points
| Integration | Type | Protocol |
|-------------|------|----------|
| Redis/Valkey | RESP | HSET (vector blob + metadata fields) |
| VectorSets | RESP | VADD (FP32 bytes + SETATTR JSON) |

---

## Constraints

- Update vectors are the same data as ingestion (re-insert, not new data)
- Per-thread interleaving (not global) — actual ratio may vary slightly under parallelism
- `rand` crate may need to be added to Cargo.toml

---

## Success Criteria

### Functional
- [ ] `--update-search-ratio 1:10` works on Redis, VectorSets, Valkey
- [ ] Omitting the flag preserves existing search-only behavior
- [ ] Results JSON includes separate search/update metrics when ratio is set
- [ ] Precision matches search-only baseline (FR-11)

### Non-Functional
- [ ] Framework overhead < 5% vs search-only
- [ ] Deterministic: same seed + ratio = same results

### Quality
- [ ] `cargo build --release` compiles cleanly
- [ ] `cargo clippy` no warnings
- [ ] Tested on random-100 and h-and-m-2048-angular datasets

---

## Bolt Suggestions

| Bolt | Type | Stories | Objective |
|------|------|---------|-----------|
| 006-mixed-benchmark-core | simple-construction-bolt | 001-008 | Full mixed benchmark implementation |

---

## Notes

All 8 stories are tightly coupled (CLI feeds into trait, trait feeds into loop, loop uses engine impls, metrics wrap everything). A single bolt is appropriate.
