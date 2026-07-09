---
id: 001-cli-flag
unit: 001-mixed-benchmark-core
intent: 011-mixed-benchmark
status: draft
priority: must
created: 2026-03-05T10:00:00Z
assigned_bolt: null
implemented: false
---

# Story: 001-cli-flag

## User Story

**As a** benchmark operator
**I want** to specify `--update-search-ratio U:S` on the CLI
**So that** I can configure how many updates are interleaved per S searches

## Acceptance Criteria

- [ ] **Given** `--update-search-ratio 1:10`, **When** args are parsed, **Then** ratio is stored as (updates=1, searches=10)
- [ ] **Given** `--update-search-ratio 0:0` or negative values, **When** args are parsed, **Then** a clear error message is printed and the process exits
- [ ] **Given** the flag is omitted, **When** the benchmark runs, **Then** search-only behavior is unchanged (no updates)
- [ ] **Given** `--update-search-ratio 1:1`, **When** parsed, **Then** ratio is (1, 1) — equal updates and searches

## Technical Notes

- Add to `Args` struct in `cli.rs` as `Option<String>` (e.g., `--update-search-ratio "1:10"`)
- Parse the `U:S` format into a tuple `(u64, u64)` — store as a new struct `UpdateSearchRatio { updates: u64, searches: u64 }`
- Validation: both values must be positive integers, at least one must be > 0
- Pass through to `experiment.rs` as `Option<UpdateSearchRatio>`

## Dependencies

### Requires
- None (first story)

### Enables
- 003-interleaved-loop (needs the parsed ratio)

## Edge Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| `--update-search-ratio 0:10` | Valid — search-only (0 updates), same as omitting flag |
| `--update-search-ratio 10:0` | Error — 0 searches makes no sense for a benchmark |
| `--update-search-ratio 1:1:1` | Error — invalid format |
| `--update-search-ratio abc` | Error — non-numeric |

## Out of Scope

- Validating engine support for updates (done at runtime in experiment runner)
