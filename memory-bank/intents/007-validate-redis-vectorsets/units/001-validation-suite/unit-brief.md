---
unit: 001-validation-suite
intent: 007-validate-redis-vectorsets
phase: inception
status: complete
unit_type: cli
default_bolt_type: simple-construction-bolt
created: 2026-02-27T00:00:00.000Z
updated: 2026-02-27T00:00:00.000Z
---

# Unit Brief: Validation Suite

## Purpose

Expand test coverage and v0-check comparisons to confirm correctness of existing Redis (RediSearch) and VectorSets Rust engine implementations. No engine code changes — validation only.

## Scope

### In Scope
- Expand v0_check.sh to test multiple datasets and VectorSets
- Add integration tests for metadata filter search
- Add integration tests for distance metrics (L2, COSINE)
- Add integration tests for parallel execution correctness
- Add VectorSets precision test

### Out of Scope
- Engine implementation changes (bug fixes found go into separate intents)
- Performance optimization
- New engine migrations

## Assigned Requirements

| FR | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Expand v0-check dataset coverage | Must |
| FR-2 | v0-check for VectorSets | Must |
| FR-3 | Integration tests for Redis metadata filters | Must |
| FR-4 | Integration tests for distance metrics | Must |
| FR-5 | Integration tests for parallel upload/search | Should |
| FR-6 | VectorSets precision integration test | Must |

## Story Summary

| Metric | Count |
|--------|-------|
| Total Stories | 6 |
| Must Have | 5 |
| Should Have | 1 |
| Could Have | 0 |

### Stories

| Story ID | Title | Priority | Status |
|----------|-------|----------|--------|
| 001-expand-v0-check | Multi-dataset v0-check coverage | Must | Planned |
| 002-v0-check-vectorsets | VectorSets v0-check support | Must | Planned |
| 003-redis-filter-tests | Redis metadata filter integration tests | Must | Planned |
| 004-distance-metric-tests | L2 and COSINE distance metric tests | Must | Planned |
| 005-parallel-execution-tests | Parallel upload/search correctness tests | Should | Planned |
| 006-vectorsets-precision | VectorSets precision integration test | Must | Planned |

## Dependencies

### Depends On
None.

### External Dependencies
| System | Purpose | Risk |
|--------|---------|------|
| Redis 8.6.0 Docker | Test target | Low |
| Python v0 environment | Reference for v0-check | Low |

## Success Criteria

### Functional
- [ ] v0-check passes for Redis across 3+ datasets
- [ ] v0-check passes for VectorSets
- [ ] All new integration tests pass
- [ ] No precision regressions found

### Quality
- [ ] `make check` passes
- [ ] `make integration-test` includes new tests
