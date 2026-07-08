---
id: 003-interleaved-loop
unit: 001-mixed-benchmark-core
intent: 011-mixed-benchmark
status: draft
priority: must
created: 2026-03-05T10:00:00Z
assigned_bolt: null
implemented: false
---

# Story: 003-interleaved-loop

## User Story

**As a** benchmark operator
**I want** worker threads to interleave searches and updates based on my ratio
**So that** the benchmark simulates real-world mixed workloads

## Acceptance Criteria

- [ ] **Given** ratio 1:10 with 100 queries, **When** the benchmark runs, **Then** each worker performs 10 searches then 1 update, repeating (~10 updates total across all workers)
- [ ] **Given** ratio 1:1, **When** running, **Then** workers alternate: 1 search, 1 update, 1 search, 1 update...
- [ ] **Given** the same seed and ratio, **When** run twice, **Then** the update sequence (which vector IDs are updated) is identical
- [ ] **Given** more updates needed than vectors exist, **When** the sequence exhausts all IDs, **Then** it wraps around (modulo vector count)
- [ ] **Given** parallel=100 workers, **When** running, **Then** updates are distributed across workers via the same AtomicUsize pattern used for queries

## Technical Notes

- The interleaved loop lives in `experiment.rs` (or a new `mixed_search` function called from experiment runner)
- Two atomic counters: `query_idx` (existing) and `update_idx` (new)
- Per worker cycle: execute `S` searches, then `U` updates, repeat
- Update sequence: pre-generate a shuffled permutation of all vector IDs using `rand::SeedableRng` with seed 42. Workers claim next update index atomically.
- The update needs access to the original vectors and metadata from the dataset — pass as shared references to worker threads
- Each worker thread has its own Redis connection (existing pattern) — updates use the same connection

## Dependencies

### Requires
- 001-cli-flag (ratio value)
- 002-engine-trait (update() method signature)

### Enables
- 004-metrics-reporting (feeds search_times and update_times)

## Edge Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| ratio 1:10 with only 5 queries | 5 searches, 0 updates (not enough searches to trigger an update cycle) |
| ratio 10:1 | 1 search then 10 updates per cycle — heavy write load |
| All workers finish queries before completing update cycle | Remaining updates in current cycle are skipped |

## Out of Scope

- Global ordering guarantees across workers (each worker interleaves independently)
- Separate update thread pool
