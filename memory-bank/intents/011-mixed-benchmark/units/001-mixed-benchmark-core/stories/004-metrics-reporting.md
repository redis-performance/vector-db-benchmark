---
id: 004-metrics-reporting
unit: 001-mixed-benchmark-core
intent: 011-mixed-benchmark
status: draft
priority: must
created: 2026-03-05T10:00:00Z
assigned_bolt: null
implemented: false
---

# Story: 004-metrics-reporting

## User Story

**As a** benchmark operator
**I want** separate search and update metrics in the results
**So that** I can analyze performance degradation per operation type

## Acceptance Criteria

- [ ] **Given** mixed mode results, **When** I look at the JSON, **Then** I see `update_count`, `update_rps`, `update_mean_time`, `update_p50_time`, `update_p95_time`, `update_p99_time`, `update_latencies`
- [ ] **Given** mixed mode results, **When** I look at the JSON, **Then** search metrics (`rps`, `mean_time`, `p50_time`, etc.) reflect search operations only
- [ ] **Given** mixed mode results, **When** I look at the JSON, **Then** `update_search_ratio` field records the configured ratio string (e.g., "1:10")
- [ ] **Given** search-only mode (no ratio flag), **When** I look at the JSON, **Then** no update fields are present (backward compatible)
- [ ] **Given** mixed mode, **When** the summary table prints, **Then** both search and update metrics are displayed

## Technical Notes

- Extend `SearchResults` in `engine/mod.rs` with optional update fields
- In the interleaved loop, collect `update_times: Arc<Mutex<Vec<f64>>>` alongside `search_times`
- Compute update stats (mean, p50, p95, p99) the same way as search stats
- In `experiment.rs` `save_search_results()`, conditionally include update fields in the JSON
- Summary print: add a second line for updates when present

## Dependencies

### Requires
- 003-interleaved-loop (produces the raw timing data)

### Enables
- 008-precision-invariance (uses search precision from results)

## Edge Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| 0 updates completed (ratio 1:10 with few queries) | update fields present but with 0 count, no latency stats |
| update_latencies empty | update_p50/p95/p99 = 0.0 |

## Out of Scope

- Graphical comparison of mixed vs search-only
- CSV export
