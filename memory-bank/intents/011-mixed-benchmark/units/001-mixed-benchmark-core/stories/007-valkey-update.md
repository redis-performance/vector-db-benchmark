---
id: 007-valkey-update
unit: 001-mixed-benchmark-core
intent: 011-mixed-benchmark
status: draft
priority: must
created: 2026-03-05T10:00:00Z
assigned_bolt: null
implemented: false
---

# Story: 007-valkey-update

## User Story

**As a** benchmark operator
**I want** the Valkey engine to support vector updates in mixed mode
**So that** I can measure Valkey Search performance under write pressure

## Acceptance Criteria

- [ ] **Given** a Valkey engine with uploaded data, **When** `update()` is called with an ID, vector, and metadata, **Then** the vector is re-written via `HSET` with the vector blob and all metadata fields
- [ ] **Given** an update for an existing ID, **When** the HSET completes, **Then** the Valkey Search index reflects the updated data
- [ ] **Given** mixed mode on Valkey, **When** running h-and-m-2048-angular, **Then** search precision matches the search-only baseline

## Technical Notes

- Implementation is nearly identical to Redis (story 005) — both use HSET on hash keys
- Implement `update()` on `ValkeyEngine` using the same `HSET` pattern
- The hash key format follows Valkey's existing upload pattern
- Valkey Search indexes updates automatically (same as RediSearch)
- Can likely extract a shared helper used by both Redis and Valkey engines

## Dependencies

### Requires
- 002-engine-trait (trait method signature)

### Enables
- 008-precision-invariance (Valkey precision validation)

## Edge Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| Same as Redis (005) | HSET upsert semantics identical |

## Out of Scope

- Valkey-specific optimizations
