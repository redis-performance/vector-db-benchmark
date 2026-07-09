---
id: 006-vectorsets-precision
unit: 001-validation-suite
intent: 007-validate-redis-vectorsets
status: complete
priority: must
created: 2026-02-27T00:00:00.000Z
assigned_bolt: 004-validation-suite
implemented: true
---

# Story: 006-vectorsets-precision

## User Story

**As a** developer
**I want** a precision integration test for VectorSets
**So that** I can confirm VSIM returns correct nearest neighbors at scale

## Acceptance Criteria

- [ ] **Given** 200+ random vectors uploaded via VADD, **When** querying with VSIM, **Then** precision >= 0.8 against brute-force cosine neighbors
- [ ] **Given** the query vector exists in the dataset, **When** searching, **Then** it appears as the top-1 result
- [ ] **Given** score conversion (1-score), **When** computing distance, **Then** self-distance is ~0

## Technical Notes

- Mirror test_redis_knn_precision structure but for VectorSets
- Use cosine similarity for brute-force (VectorSets default)
- Adapt brute_force_neighbors to support cosine distance

## Dependencies

### Requires
- None

### Enables
- None
