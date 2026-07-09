---
stage: plan
bolt: 004-validation-suite
created: 2026-02-27T14:15:00Z
---

## Implementation Plan: Validation Suite (Integration Tests)

### Objective

Add 4 new integration tests to `tests/integration_redis.rs` covering: metadata-filtered KNN search, COSINE distance metric precision, parallel upload/search correctness, and VectorSets brute-force precision validation.

### Deliverables

1. **`test_redis_filtered_knn_search`** — Story 003: TAG + NUMERIC filter tests
2. **`test_redis_cosine_precision`** — Story 004: COSINE metric brute-force precision
3. **`test_redis_parallel_upload_search`** — Story 005: Parallel thread safety
4. **`test_vectorsets_knn_precision`** — Story 006: VectorSets precision at scale
5. **`brute_force_neighbors_cosine`** helper — Reusable for stories 004 + 006

### Dependencies

- Redis 8.6.0 Docker (already in `tests/docker-compose.test.yml`)
- `rand` crate (already a dev dependency)
- No new crate dependencies needed

### Technical Approach

#### Helper: `brute_force_neighbors_cosine`

Add a cosine-distance variant of the existing `brute_force_neighbors`. Cosine distance = 1 - cosine_similarity, where cosine_similarity = dot(a,b) / (||a|| * ||b||). Needed by tests 004 and 006.

#### Test 1: `test_redis_filtered_knn_search` (Story 003)

**Setup**: 20 vectors (dim=8) with metadata:
- `category` (TAG): alternating "electronics" / "clothing"
- `price` (NUMERIC): 10 * i (values 0, 10, 20, ..., 190)

**FT.CREATE schema**: vector HNSW + `category TAG SEPARATOR ; SORTABLE` + `price NUMERIC SORTABLE`

**Test cases**:
1. **TAG filter**: `@category:{electronics}` → only even-indexed IDs returned
2. **NUMERIC range**: `@price:[50 100]` → only IDs 5-10 returned
3. **AND (TAG + NUMERIC)**: `@category:{electronics} @price:[0 80]` → only even IDs 0-8
4. **No filter**: `*` → all vectors are candidates (baseline comparison)

**Verification**: Assert result IDs are subsets of expected filtered IDs. Verify filtered results have fewer hits than unfiltered.

#### Test 2: `test_redis_cosine_precision` (Story 004)

Mirror existing `test_redis_knn_precision` but with `DISTANCE_METRIC COSINE`:
- 200 vectors, dim=16, top=5
- Create index with COSINE metric
- Query with vector[42], compare against `brute_force_neighbors_cosine`
- Assert precision >= 0.8
- Verify query vector is its own nearest neighbor (score ~1.0)

#### Test 3: `test_redis_parallel_upload_search` (Story 005)

**Upload phase**: 500 vectors (dim=8), uploaded across 4 threads using `std::thread::scope` + chunking (same pattern as engine/redis.rs). Each thread pipelines its chunk via HSET.

**Verification**:
1. After upload, `FT.INFO idx` → `num_docs` == 500
2. Run 20 queries across 4 threads (AtomicUsize index), each thread does FT.SEARCH
3. All 20 results returned without error
4. Precision check on first query against brute-force

#### Test 4: `test_vectorsets_knn_precision` (Story 006)

Mirror `test_redis_knn_precision` structure but for VectorSets:
- 200 vectors, dim=16, top=5
- Upload via VADD with NOQUANT, M=16, EF=200, CAS
- Query with vector[42] via VSIM WITHSCORES COUNT 5 EF 64
- Convert scores: distance = 1 - score
- Compare against `brute_force_neighbors_cosine` (VectorSets default is cosine)
- Assert precision >= 0.8
- Verify self-query top-1 and self-distance ~0

### Acceptance Criteria

- [ ] All 4 new tests pass with `cargo test --test integration_redis --release -- --nocapture --test-threads=1`
- [ ] `make integration-test` runs all tests (existing + new)
- [ ] `make check` passes (no new Rust code formatting/lint issues)
- [ ] No precision regressions in existing tests
- [ ] `brute_force_neighbors_cosine` helper correctly computes cosine distance
