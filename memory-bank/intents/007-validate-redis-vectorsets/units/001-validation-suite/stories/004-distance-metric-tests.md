---
id: 004-distance-metric-tests
unit: 001-validation-suite
intent: 007-validate-redis-vectorsets
status: complete
priority: must
created: 2026-02-27T00:00:00.000Z
assigned_bolt: 004-validation-suite
implemented: true
---

# Story: 004-distance-metric-tests

## User Story

**As a** developer
**I want** integration tests for both L2 and COSINE distance metrics
**So that** I can confirm neighbor ordering is correct for each metric

## Acceptance Criteria

- [ ] **Given** known vectors, **When** searching with L2 metric, **Then** nearest neighbors match brute-force L2 computation
- [ ] **Given** known vectors, **When** searching with COSINE metric, **Then** nearest neighbors match brute-force cosine computation
- [ ] **Given** both metrics, **When** comparing Redis and VectorSets, **Then** both produce correct orderings

## Technical Notes

- Existing test_redis_knn_precision uses L2 only
- Add COSINE variant with brute_force_neighbors adapted for cosine distance
- For VectorSets: VSIM uses cosine similarity by default (score = cosine similarity)

## Dependencies

### Requires
- None

### Enables
- None
