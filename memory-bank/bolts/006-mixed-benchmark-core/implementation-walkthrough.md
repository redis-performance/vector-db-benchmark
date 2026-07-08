---
stage: implement
bolt: 006-mixed-benchmark-core
created: 2026-03-05T12:00:00Z
---

## Implementation Walkthrough: Mixed Benchmark Core

### Files Modified

| File | Stories | Description |
|------|---------|-------------|
| `Cargo.toml` | — | Moved `rand` from dev-dependencies to dependencies |
| `cli.rs` | 001 | Added `--update-search-ratio` CLI flag |
| `engine/mod.rs` | 002 | `UpdateSearchRatio` struct, `SearchResults` update fields, `search_mixed()` default |
| `experiment.rs` | 003, 004 | Ratio parsing, dispatch to `search_mixed()`, extended JSON output |
| `engine/redis.rs` | 005 | `hset_single()` + `search_mixed()` with interleaved FT.SEARCH + HSET |
| `engine/vectorsets.rs` | 006 | `vadd_single()` + `search_mixed()` with interleaved VSIM + VADD |
| `engine/valkey.rs` | 007 | `hset_single()` + `search_mixed()` with interleaved FT.SEARCH + HSET |
| All other engines | — | Added `..Default::default()` for new optional fields |

### Architecture

Each engine's `search_mixed()` follows the same pattern:

1. Read queries + ground truth (same as `search()`)
2. Read upload vectors + ids + metadata (for updates)
3. Create deterministic shuffled update sequence (seed=42)
4. Per-thread interleaved loop: S searches → U updates → repeat
5. Two atomic counters: `search_idx` (stops at num_queries), `update_idx` (wraps)
6. Separate timing collections for search and update latencies
7. Return `SearchResults` with both search metrics and optional update metrics

### Backward Compatibility

- Omitting `--update-search-ratio` preserves exact existing behavior
- All optional update fields default to `None` via `..Default::default()`
- Results JSON only includes update fields when ratio is set

### Verification

- `cargo build --release` — clean
- `cargo clippy` — no new warnings (14 pre-existing)
- `cargo test --bin vector-db-benchmark` — 15/15 pass
