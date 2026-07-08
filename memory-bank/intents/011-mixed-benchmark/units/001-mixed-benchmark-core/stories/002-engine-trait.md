---
id: 002-engine-trait
unit: 001-mixed-benchmark-core
intent: 011-mixed-benchmark
status: draft
priority: must
created: 2026-03-05T10:00:00Z
assigned_bolt: null
implemented: false
---

# Story: 002-engine-trait

## User Story

**As a** developer extending the benchmark
**I want** an `update()` method on the Engine trait
**So that** engines can implement upsert logic for mixed benchmarks

## Acceptance Criteria

- [ ] **Given** the Engine trait, **When** I look at its methods, **Then** there is `fn update(&mut self, id: i64, vector: &[f32], metadata: Option<&MetadataItem>) -> Result<(), String>`
- [ ] **Given** an engine that doesn't implement `update()`, **When** mixed mode is requested, **Then** the default implementation returns `Err("mixed benchmark not supported for {engine}")`
- [ ] **Given** the `SearchResults` struct, **When** mixed mode runs, **Then** it includes `update_count`, `update_rps`, `update_mean_time`, `update_p50`, `update_p95`, `update_p99`, `update_latencies`

## Technical Notes

- Add `update()` with a default implementation in `engine/mod.rs`
- Extend `SearchResults` with optional update fields (use `Option<>` to keep backward compat)
- The `update()` method takes a single vector — it's called once per update in the interleaved loop
- Connection management is per-thread (each worker already has its own connection)
- Consider a separate `fn update_single(conn, id, vector, metadata)` free function per engine (matching the `ft_search_knn` / `vsim_search` pattern)

## Dependencies

### Requires
- None

### Enables
- 005-redis-update, 006-vectorsets-update, 007-valkey-update (engine implementations)
- 003-interleaved-loop (calls update())

## Edge Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| Engine doesn't override update() | Default returns error, experiment skips mixed mode with clear message |
| update() called without prior upload | Engine-specific behavior (likely succeeds as upsert creates new entry) |

## Out of Scope

- Batch update operations (single-vector update only for now)
- Connection pooling changes
