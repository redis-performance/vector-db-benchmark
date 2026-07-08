---
intent: 011-mixed-benchmark
phase: inception
status: units-decomposed
updated: 2026-03-05T10:00:00Z
---

# Mixed Benchmark - Unit Decomposition

## Units Overview

This intent decomposes into 1 unit of work. The scope is focused: add an `update()` method to the Engine trait, extend the CLI, modify the search loop to interleave updates, and extend results reporting. All changes land in existing files with one clear integration point.

### Unit 1: 001-mixed-benchmark-core

**Description**: Core mixed benchmark implementation — CLI flag, Engine trait update method, interleaved worker loop, separate metrics, and engine implementations for Redis/VectorSets/Valkey.

**Stories**:

- Story 001: CLI flag parsing (`--update-search-ratio`)
- Story 002: Engine trait `update()` method
- Story 003: Interleaved worker loop in experiment runner
- Story 004: Separate update/search metrics and results JSON
- Story 005: Redis engine `update()` implementation
- Story 006: VectorSets engine `update()` implementation
- Story 007: Valkey engine `update()` implementation
- Story 008: Precision invariance validation

**Deliverables**:

- Modified `cli.rs` with new flag
- Modified `engine/mod.rs` with `update()` trait method and extended `SearchResults`
- Modified `experiment.rs` with interleaved search/update loop
- Modified `engine/redis.rs`, `engine/vectorsets.rs`, `engine/valkey.rs` with `update()` implementations
- Extended results JSON schema

**Dependencies**:

- Depends on: None (all engines already exist and work)
- Depended by: None

**Estimated Complexity**: M

## Unit Dependency Graph

```text
[001-mixed-benchmark-core]  (standalone, no dependencies)
```

## Execution Order

Single unit — one bolt covers all stories sequentially.
