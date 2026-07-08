---
id: 006-mixed-benchmark-core
unit: 001-mixed-benchmark-core
intent: 011-mixed-benchmark
type: simple-construction-bolt
status: complete
stories:
  - 001-cli-flag
  - 002-engine-trait
  - 003-interleaved-loop
  - 004-metrics-reporting
  - 005-redis-update
  - 006-vectorsets-update
  - 007-valkey-update
  - 008-precision-invariance
created: 2026-03-05T10:00:00Z
started: 2026-03-05T10:30:00Z
completed: 2026-03-05T12:30:00Z
current_stage: complete
stages_completed:
  - name: plan
    completed: 2026-03-05T10:45:00Z
    artifact: implementation-plan.md
  - name: implement
    completed: 2026-03-05T12:00:00Z
    artifact: implementation-walkthrough.md
  - name: test
    completed: 2026-03-05T12:30:00Z
    notes: Unit tests pass. Precision invariance requires running Redis/VectorSets/Valkey servers.

requires_bolts: []
enables_bolts: []
requires_units: []
blocks: false

complexity:
  avg_complexity: 2
  avg_uncertainty: 1
  max_dependencies: 2
  testing_scope: 2
---

# Bolt: 006-mixed-benchmark-core

## Overview

Single bolt implementing the full mixed benchmark feature — CLI parsing, Engine trait update method, interleaved worker loop, separate metrics, and update implementations for Redis/VectorSets/Valkey.

## Objective

Enable `--update-search-ratio U:S` to run mixed workload benchmarks that interleave vector updates with searches, measuring both operation types independently.

## Stories Included

- **001-cli-flag**: CLI flag parsing `--update-search-ratio U:S` (Must)
- **002-engine-trait**: Engine trait `update()` method + SearchResults extension (Must)
- **003-interleaved-loop**: Interleaved worker loop in experiment runner (Must)
- **004-metrics-reporting**: Separate search/update metrics in results JSON (Must)
- **005-redis-update**: Redis engine `update()` — HSET upsert (Must)
- **006-vectorsets-update**: VectorSets engine `update()` — VADD upsert + SETATTR (Must)
- **007-valkey-update**: Valkey engine `update()` — HSET upsert (Must)
- **008-precision-invariance**: Validate precision matches search-only baseline (Must)

## Bolt Type

**Type**: Simple Construction Bolt
**Definition**: `.specsmd/aidlc/templates/construction/bolt-types/simple-construction-bolt.md`

## Stages

- [ ] **1. implement**: Code all stories → modified cli.rs, mod.rs, experiment.rs, redis.rs, vectorsets.rs, valkey.rs
- [ ] **2. test**: Validate precision invariance on random-100 and h-and-m datasets

## Dependencies

### Requires
- None (all target engines already exist)

### Enables
- Future: additional engine support for mixed benchmarks

## Success Criteria

- [ ] `--update-search-ratio 1:10` runs on Redis, VectorSets, Valkey
- [ ] Omitting flag preserves search-only behavior
- [ ] Results JSON has separate search/update metrics
- [ ] Precision matches search-only baseline (within 0.001)
- [ ] `cargo build --release` clean
- [ ] `cargo clippy` no warnings

## Notes

Implementation order within the bolt:
1. CLI flag + ratio struct (001)
2. Engine trait update() + SearchResults extension (002)
3. Redis update() impl (005) — test immediately
4. VectorSets update() impl (006) — test immediately
5. Valkey update() impl (007) — test immediately
6. Interleaved loop in experiment runner (003)
7. Metrics reporting (004)
8. Precision invariance validation (008)
