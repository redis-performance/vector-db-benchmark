---
stage: plan
bolt: 002-elasticsearch-engine
created: 2026-02-27T12:00:00Z
---

## Implementation Plan: Elasticsearch Engine — Search & Filters (Bolt 002)

### Objective

Complete the Elasticsearch Engine trait by replacing the search stub with a working KNN search implementation, adding metadata filter parsing (match, range, geo), and comprehensive integration tests for the full configure/upload/search/delete cycle.

### Deliverables

- `num_candidates` field added to `SearchParams` in `config.rs`
- KNN search implementation in `elasticsearch.rs` (replaces the stub)
- Elasticsearch condition parser (filter → ES bool query) in `elasticsearch.rs`
- Integration tests for search and filters in `tests/integration_elasticsearch.rs`

### Dependencies

- Bolt 001 (complete): Connection, config, index creation, upload, force merge all exist

### Technical Approach

**Config change**: Add `num_candidates: Option<i64>` to `SearchParams`. The ES config JSON has `{ "parallel": 1, "num_candidates": 128 }` at the top level of each search params entry.

**Filter parser** (`parse_es_conditions`):
Converts the benchmark internal filter format to ES bool queries. Input is `Option<serde_json::Value>`, output is `Option<serde_json::Value>`.

Pattern from Python v0:
- Input: `{"and": [{"field": {"match": {"value": X}}}], "or": [...]}`
- Output: `{"bool": {"must": [...], "should": [...]}}`

Filter types:
- `match` → `{"match": {"field_name": value}}`
- `range` → `{"range": {"field_name": {"gt": N, "lt": M, ...}}}` (omit null bounds)
- `geo` → `{"geo_distance": {"distance": "Rm", "field_name": {"lat": Y, "lon": X}}}`

**KNN search implementation**:
- POST `/{index}/_search` with body:
  ```json
  {
    "knn": {
      "field": "vector",
      "query_vector": [...],
      "k": top,
      "num_candidates": N,
      "filter": <es_bool_query>  // optional
    },
    "size": top
  }
  ```
- Parse response: `hits.hits[*]._id` (UUID hex → int via `Uuid::parse_str().as_u128() as i64`), `hits.hits[*]._score`
- Parallel search: `thread::scope` + `AtomicUsize` query index counter (same pattern as redis.rs)
- Each thread creates its own `reqwest::blocking::Client`
- Collect `(query_time, precision)` per query in `Arc<Mutex<Vec<_>>>`
- Compute statistics: mean/std/min/max/p50/p95/p99 latency, mean precision, RPS

**UUID-to-int conversion**: `Uuid::parse_str(hex_str)?.as_u128() as i64` — reverse of `id_to_uuid_hex`

**Delete**: Already implemented in bolt 001

### Acceptance Criteria

- [ ] `num_candidates` field added to `SearchParams`
- [ ] KNN search returns correct nearest neighbors
- [ ] UUID hex IDs converted back to integer IDs
- [ ] num_candidates parameter passed to KNN query
- [ ] Parallel search via thread::scope
- [ ] Precision calculation matches Python v0
- [ ] Match filter produces `{"match": {field: value}}`
- [ ] Range filter produces `{"range": {field: {gt/lt/gte/lte}}}`
- [ ] Geo filter produces `{"geo_distance": {...}}`
- [ ] AND → `{"bool": {"must": [...]}}`, OR → `{"bool": {"should": [...]}}`
- [ ] No filter → no filter key in KNN query
- [ ] Integration tests: search returns correct results
- [ ] Integration tests: precision = 1.0 for small exact dataset
- [ ] `make check` passes
