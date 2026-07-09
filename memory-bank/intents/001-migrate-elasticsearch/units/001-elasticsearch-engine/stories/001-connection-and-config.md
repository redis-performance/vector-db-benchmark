---
id: 001-connection-and-config
unit: 001-elasticsearch-engine
intent: 001-migrate-elasticsearch
status: complete
priority: must
created: 2026-02-27T00:00:00.000Z
assigned_bolt: 001-elasticsearch-engine
implemented: true
---

# Story: 001-connection-and-config

## User Story

**As a** benchmark operator
**I want** the Elasticsearch engine to parse config JSON and connect to an ES instance
**So that** I can run benchmarks with different HNSW parameter configurations

## Acceptance Criteria

- [ ] **Given** a config JSON with `engine: "elasticsearch"`, **When** parsed, **Then** `ElasticsearchEngine` is created with correct m, ef_construction, batch_size, parallel, num_candidates
- [ ] **Given** env vars `ELASTIC_PORT`, `ELASTIC_USER`, `ELASTIC_PASSWORD` are set, **When** connecting, **Then** client connects with those credentials
- [ ] **Given** `ELASTIC_API_KEY` is set, **When** connecting, **Then** API key auth is used instead of basic auth
- [ ] **Given** the engine name `"elasticsearch"` in config, **When** the engine factory is called, **Then** an `ElasticsearchEngine` instance is returned

## Technical Notes

- Add `elasticsearch` and `uuid` crates to Cargo.toml
- Create `src/bin/vector_db_benchmark/engine/elasticsearch.rs`
- Register engine in `src/bin/vector_db_benchmark/engine/mod.rs` factory
- Env vars: ELASTIC_PORT (9200), ELASTIC_USER ("elastic"), ELASTIC_PASSWORD ("passwd"), ELASTIC_API_KEY (None), ELASTIC_TIMEOUT (300), ELASTIC_INDEX ("bench")
- Reference: `v0/engine/clients/elasticsearch/config.py`

## Dependencies

### Requires
- None (first story)

### Enables
- 002-index-configuration
- 003-bulk-upload
- 005-knn-search

## Edge Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| ES not reachable | Return error with descriptive message |
| Invalid credentials | Return auth error |
| Missing ELASTIC_PORT env | Use default 9200 |

## Out of Scope

- Cluster health monitoring beyond initial ping
- TLS certificate validation
