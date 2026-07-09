---
id: 008-precision-invariance
unit: 001-mixed-benchmark-core
intent: 011-mixed-benchmark
status: draft
priority: must
created: 2026-03-05T10:00:00Z
assigned_bolt: null
implemented: false
---

# Story: 008-precision-invariance

## User Story

**As a** benchmark operator
**I want** confidence that interleaved updates don't degrade search recall
**So that** I know the mixed benchmark is measuring latency impact, not correctness issues

## Acceptance Criteria

- [ ] **Given** a search-only run on random-100 with precision X, **When** I run the same config with `--update-search-ratio 1:10`, **Then** the precision is X (within floating-point tolerance of 0.001)
- [ ] **Given** a search-only run on h-and-m-2048-angular with precision Y, **When** I run with `--update-search-ratio 1:1`, **Then** the precision is Y (within 0.001)
- [ ] **Given** ratio 1:1 (heavy updates), **When** running on all three engines (Redis, VectorSets, Valkey), **Then** precision matches search-only for each

## Technical Notes

- This is a validation story, not a code story — it produces test runs, not new code
- Run search-only baseline, then mixed mode, compare `mean_precision` from both results JSON files
- Since updates re-insert identical vectors, the index should be semantically unchanged
- If precision differs, it indicates a bug in the update path (wrong data, corrupted index, or timing issue)
- Can be automated as a CI check or manual validation

## Dependencies

### Requires
- 005-redis-update, 006-vectorsets-update, 007-valkey-update (engine implementations)
- 004-metrics-reporting (precision in results JSON)

### Enables
- None (final validation)

## Edge Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| VectorSets HNSW graph mutation during VSIM | VADD upsert should be atomic per element |
| Very high update ratio (10:1) | Precision still matches — updates are same data |

## Out of Scope

- Performance regression testing (that's what the QPS/latency metrics are for)
- Testing with different update data (out of scope for this intent)
