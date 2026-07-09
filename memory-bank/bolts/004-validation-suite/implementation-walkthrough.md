---
stage: implement
bolt: 004-validation-suite
created: 2026-02-27T14:30:00Z
---

## Implementation Walkthrough: Validation Suite (Integration Tests)

### Summary

Added 6 new integration tests, 3 helper functions, and a subprocess test harness to `tests/integration_redis.rs`. Tests cover protocol-level validation (filtered KNN, COSINE precision, parallel correctness, VectorSets precision) plus end-to-end binary subprocess tests for both Redis and VectorSets engines.

### Structure Overview

Two layers of testing: raw Redis command tests for protocol-level coverage, plus subprocess tests that run the actual `vector-db-benchmark` binary against synthetic JSONL datasets to validate the full production code path.

### Completed Work

- [x] `tests/integration_redis.rs` — Added `brute_force_neighbors_cosine` helper for cosine distance computation
- [x] `tests/integration_redis.rs` — Added `parse_ft_search_ids` helper to extract IDs from FT.SEARCH responses
- [x] `tests/integration_redis.rs` — Added `create_test_project` helper to scaffold a temporary project directory with JSONL datasets, datasets.json, and engine configs
- [x] `tests/integration_redis.rs` — Added `test_redis_filtered_knn_search` (TAG, NUMERIC range, AND, unfiltered baseline)
- [x] `tests/integration_redis.rs` — Added `test_redis_cosine_precision` (COSINE metric brute-force comparison)
- [x] `tests/integration_redis.rs` — Added `test_redis_parallel_upload_search` (4-thread upload + search)
- [x] `tests/integration_redis.rs` — Added `test_vectorsets_knn_precision` (VSIM precision at 200 vectors)
- [x] `tests/integration_redis.rs` — Added `test_binary_redis_end_to_end` (subprocess: binary configure+upload+search via synthetic JSONL)
- [x] `tests/integration_redis.rs` — Added `test_binary_vectorsets_end_to_end` (subprocess: binary VectorSets end-to-end)

### Key Decisions

- **Two-layer testing**: Raw Redis tests for protocol-level validation + subprocess tests for production code path coverage
- **Subprocess harness**: Creates a temporary directory mimicking the project layout (v0/datasets/datasets.json, datasets/, experiments/configurations/) so the binary's `project_root()` discovery works correctly
- **Unique index names per test**: Each test uses a distinct index name to avoid conflicts
- **Parallel precision threshold at 0.6**: Lower than standard 0.8 because the parallel test focuses on thread safety
- **REDIS_PORT env var**: Subprocess tests pass `REDIS_PORT=6399` to target the test container

### Deviations from Plan

- Added 2 subprocess tests (test_binary_redis_end_to_end, test_binary_vectorsets_end_to_end) to test through the actual binary — not in original plan, added per user request

### Dependencies Added

- `std::fs` and `std::process::Command` — for subprocess tests (standard library, no new crate deps)

### Developer Notes

- Tests must run single-threaded (`--test-threads=1`) because they share a Redis instance
- Subprocess tests require `cargo build --release` to have run first (the binary must exist)
- The `create_test_project` function leaks the TempDir to keep it alive until explicit cleanup at test end
