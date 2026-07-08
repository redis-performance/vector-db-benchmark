---
id: 005-redis-update
unit: 001-mixed-benchmark-core
intent: 011-mixed-benchmark
status: draft
priority: must
created: 2026-03-05T10:00:00Z
assigned_bolt: null
implemented: false
---

# Story: 005-redis-update

## User Story

**As a** benchmark operator
**I want** the Redis (RediSearch) engine to support vector updates in mixed mode
**So that** I can measure RediSearch search performance under write pressure

## Acceptance Criteria

- [ ] **Given** a Redis engine with uploaded data, **When** `update()` is called with an ID, vector, and metadata, **Then** the vector is re-written via `HSET` with the vector blob and all metadata fields
- [ ] **Given** an update for an existing ID, **When** the HSET completes, **Then** the RediSearch index automatically reflects the updated data
- [ ] **Given** mixed mode on Redis, **When** running h-and-m-2048-angular, **Then** search precision matches the search-only baseline

## Technical Notes

- Implement `update()` on `RedisEngine` — similar to the upload path but for a single vector
- Use `HSET key field1 val1 field2 val2 ...` with the vector blob field and metadata fields
- The hash key format follows the existing upload pattern (e.g., `doc:{id}`)
- Connection: each worker thread already has its own `redis::Connection` — the update uses the same connection as searches
- Consider a free function `hset_update(conn, id, vector, metadata, config)` for clarity

## Dependencies

### Requires
- 002-engine-trait (trait method signature)

### Enables
- 008-precision-invariance (Redis precision validation)

## Edge Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| Update a non-existent ID | HSET creates a new hash key (upsert semantics) |
| Update while FT.SEARCH is running | Redis handles concurrently (single-threaded server serializes) |

## Out of Scope

- Batch updates (single vector per call)
- Pipeline updates
