---
id: 005-knn-search
unit: 001-elasticsearch-engine
intent: 001-migrate-elasticsearch
status: complete
priority: must
created: 2026-02-27T00:00:00.000Z
assigned_bolt: 002-elasticsearch-engine
implemented: true
---

# Story: 005-knn-search

## User Story

**As a** benchmark operator
**I want** KNN vector search executed against Elasticsearch with configurable parallelism
**So that** I can measure search throughput and precision

## Acceptance Criteria

- [ ] **Given** a query vector and top=10, **When** searching, **Then** returns top-10 nearest neighbors with scores
- [ ] **Given** `num_candidates: 256` in search_params, **When** searching, **Then** KNN uses 256 candidates
- [ ] **Given** `parallel: 100` in search_params, **When** searching, **Then** search uses 100 threads via thread::scope
- [ ] **Given** search results with UUID hex IDs, **When** processing results, **Then** UUIDs are converted back to integer IDs
- [ ] **Given** ground truth neighbors, **When** computing precision, **Then** precision matches Python v0 calculation

## Technical Notes

- KNN query: `{"field": "vector", "query_vector": [...], "k": top, "num_candidates": N}`
- Execute via `client.search(index, knn, size=top)`
- Result parsing: extract `hits.hits[*]._id` (UUID hex -> int) and `hits.hits[*]._score`
- Parallel pattern: same as redis.rs (thread::scope + Arc<Mutex> for results collection)
- Reference: `v0/engine/clients/elasticsearch/search.py`

## Dependencies

### Requires
- 001-connection-and-config
- 004-force-merge (conceptually — search runs after upload+merge)

### Enables
- 006-filter-parsing (search with filters builds on basic search)

## Edge Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| No results returned | Empty result set, precision = 0 |
| Fewer results than top-k | Return what's available |
| Single thread (parallel: 1) | Sequential search |

## Out of Scope

- Approximate search tuning beyond num_candidates
- Multi-field vector search
