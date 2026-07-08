---
id: 006-vectorsets-update
unit: 001-mixed-benchmark-core
intent: 011-mixed-benchmark
status: draft
priority: must
created: 2026-03-05T10:00:00Z
assigned_bolt: null
implemented: false
---

# Story: 006-vectorsets-update

## User Story

**As a** benchmark operator
**I want** the VectorSets engine to support vector updates in mixed mode
**So that** I can measure VectorSets search performance under write pressure

## Acceptance Criteria

- [ ] **Given** a VectorSets engine with uploaded data, **When** `update()` is called with an ID, vector, and metadata, **Then** the vector is re-inserted via `VADD` with the same element name (upsert) and `SETATTR` with JSON metadata
- [ ] **Given** an update for an existing element, **When** the VADD completes, **Then** the HNSW index reflects the updated vector
- [ ] **Given** mixed mode on VectorSets, **When** running h-and-m-2048-angular, **Then** search precision matches the search-only baseline

## Technical Notes

- Implement `update()` on `VectorSetsEngine`
- Use `VADD idx FP32 <vec_bytes> <id> <quant> [SETATTR '<json>']` — same as upload but single element, no CAS needed for update
- The element name is the string ID (same as upload)
- Metadata serialization: reuse the JSON serialization pattern from `vadd_batch()`
- Connection: worker thread's existing connection

## Dependencies

### Requires
- 002-engine-trait (trait method signature)

### Enables
- 008-precision-invariance (VectorSets precision validation)

## Edge Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| VADD with existing element name | Upsert — vector is replaced in the HNSW graph |
| VADD while VSIM is running | Redis handles concurrently (single-threaded server serializes) |

## Out of Scope

- VSETATTR-only updates (we do full vector + metadata)
- Batch VADD for updates
