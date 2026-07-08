---
intent: 007-validate-redis-vectorsets
phase: inception
status: complete
created: 2026-02-27T00:00:00Z
updated: 2026-02-27T00:00:00Z
---

# Requirements: Validate Redis & VectorSets Migration

## Intent Overview

Systematically validate that the existing Rust Redis (RediSearch) and VectorSets engine implementations produce correct results by expanding v0-check coverage across multiple datasets/configs and adding more integration tests for edge cases, metadata filters, and parallel execution.

## Business Goals

| Goal | Success Metric | Priority |
|------|----------------|----------|
| Precision parity confirmed | v0-check passes for all dataset/config combinations | Must |
| Edge case coverage | Integration tests cover metadata filters, distance metrics, parallel modes | Must |
| Regression safety net | New tests catch future regressions during engine refactoring | Should |

## Functional Requirements

### FR-1: Expand v0-check Dataset Coverage
- **Description**: Extend `scripts/v0_check.sh` to run against multiple datasets (not just h-and-m-2048-angular-filters) and multiple engine configs. Support running all combinations or a specific one.
- **Acceptance Criteria**: v0-check can run against at least 3 different datasets; all pass with precision match
- **Priority**: Must

### FR-2: v0-check for VectorSets
- **Description**: Add VectorSets engine to v0-check comparison. Currently only Redis (RediSearch) is tested.
- **Acceptance Criteria**: v0-check runs VectorSets configs against Python v0 and Rust; precision matches
- **Priority**: Must

### FR-3: Integration Tests for Redis Metadata Filters
- **Description**: Add integration tests that exercise the Redis engine's metadata filter parsing through FT.SEARCH with actual filter expressions (exact match, range, numeric, tag).
- **Acceptance Criteria**: Tests verify correct filtered search results against known data
- **Priority**: Must

### FR-4: Integration Tests for Distance Metrics
- **Description**: Add integration tests covering L2 and COSINE distance metrics for both Redis and VectorSets engines.
- **Acceptance Criteria**: Both distance metrics produce correct neighbor ordering
- **Priority**: Must

### FR-5: Integration Tests for Parallel Upload/Search
- **Description**: Add integration tests that verify parallel upload and parallel search produce correct results (no race conditions, no data loss).
- **Acceptance Criteria**: Parallel upload stores all vectors; parallel search returns correct precision
- **Priority**: Should

### FR-6: VectorSets Precision Integration Test
- **Description**: Add a precision test for VectorSets (similar to test_redis_knn_precision) that computes brute-force neighbors and compares against VSIM results.
- **Acceptance Criteria**: VectorSets precision >= 0.8 for test dataset with 200+ vectors
- **Priority**: Must

## Non-Functional Requirements

### NFR-1: Test Execution
- All new integration tests run within existing `make integration-test` target
- v0-check remains a separate `make v0-check` target

## Constraints

- Tests must work with redis:8.6.0 Docker image (same as existing)
- v0-check requires Python v0 environment (poetry install in v0/)
- No changes to engine implementation code — validation only
