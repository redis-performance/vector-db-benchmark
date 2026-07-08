---
id: 003-redis-filter-tests
unit: 001-validation-suite
intent: 007-validate-redis-vectorsets
status: complete
priority: must
created: 2026-02-27T00:00:00.000Z
assigned_bolt: 004-validation-suite
implemented: true
---

# Story: 003-redis-filter-tests

## User Story

**As a** developer
**I want** integration tests for Redis metadata-filtered KNN search
**So that** I can confirm filter parsing produces correct results

## Acceptance Criteria

- [ ] **Given** vectors with TAG metadata, **When** searching with exact match filter, **Then** only matching documents are returned
- [ ] **Given** vectors with NUMERIC metadata, **When** searching with range filter, **Then** only documents within range are returned
- [ ] **Given** AND conditions, **When** searching, **Then** all conditions must be satisfied
- [ ] **Given** no filter, **When** searching same dataset, **Then** all vectors are candidates

## Technical Notes

- Create test vectors with known metadata (e.g., category=A/B, price=10/20/30)
- Create FT.CREATE schema with TAG and NUMERIC fields
- Run FT.SEARCH with filter expressions and verify result IDs
- Reference: existing test_redis_metadata_upload (currently only tests upload, not filtered search)

## Dependencies

### Requires
- None

### Enables
- None
