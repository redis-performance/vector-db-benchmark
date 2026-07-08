---
stage: test
bolt: 004-validation-suite
created: 2026-02-27T16:00:00Z
---

## Test Walkthrough: Validation Suite (Integration Tests)

### Test Execution

Command: `cargo test --test integration_redis -- --test-threads=1`

**Result: 13/13 PASS** (2 consecutive clean runs)

### Test Matrix

| Test | Story | Type | Status |
|------|-------|------|--------|
| test_redis_filtered_knn_search | 003 | Protocol | PASS |
| test_redis_cosine_precision | 004 | Protocol | PASS |
| test_redis_parallel_upload_search | 005 | Protocol | PASS |
| test_vectorsets_knn_precision | 006 | Protocol | PASS |
| test_binary_redis_end_to_end | — | Subprocess | PASS |
| test_binary_vectorsets_end_to_end | — | Subprocess | PASS |
| test_redis_upload_and_retrieve | — | Existing | PASS (regression) |
| test_redis_knn_search | — | Existing | PASS (regression) |
| test_redis_knn_precision | — | Existing | PASS (regression) |
| test_redis_metadata_upload | — | Existing | PASS (regression) |
| test_vectorsets_vadd_and_vsim | — | Existing | PASS (regression) |
| test_vectorsets_pipeline_batch | — | Existing | PASS (regression) |
| test_vectorsets_score_conversion | — | Existing | PASS (regression) |

### Bugs Found and Fixed

1. **AND filter query syntax** (test_redis_filtered_knn_search)
   - **Symptom**: "Syntax error at offset 39 near >["
   - **Cause**: RediSearch parser fails on `@field1:{val} @field2:[a b]=>[KNN...]` without grouping parens
   - **Fix**: Wrapped compound prefilter in parentheses: `(@field1:{val} @field2:[a b])=>[KNN...]`

2. **FLUSHALL does not drop FT indexes in Redis 8** (test_redis_knn_search, test_redis_knn_precision)
   - **Symptom**: "No such index idx" — previous test's FT.CREATE leaves stale index definitions
   - **Cause**: Redis 8.x FLUSHALL removes data but preserves FT index metadata
   - **Fix**: Updated `flush_db()` to call `FT._LIST` + `FT.DROPINDEX DD` before FLUSHALL

3. **FT.INFO RESP3 Map response format** (test_redis_parallel_upload_search)
   - **Symptom**: `num_docs` extraction returned None instead of Some(500)
   - **Cause**: redis crate 0.27 may return `redis::Value::Map` (RESP3) instead of flat `Array` for FT.INFO
   - **Fix**: Added `extract_ft_info_value()` helper that handles both `Array` (key/value pairs) and `Map` formats, plus `SimpleString`/`BulkString`/`Int`/`Double` value types

### Quality Checks

- `cargo fmt -- --check`: PASS (no formatting issues)
- `cargo clippy --all-targets`: No new errors (pre-existing warnings only)
- No precision regressions in existing tests
