---
unit: 001-elasticsearch-engine
intent: 001-migrate-elasticsearch
phase: inception
status: complete
unit_type: cli
default_bolt_type: simple-construction-bolt
created: 2026-02-27T00:00:00.000Z
updated: 2026-02-27T00:00:00.000Z
---

# Unit Brief: Elasticsearch Engine

## Purpose

Implement the Elasticsearch vector search engine as a Rust module, conforming to the `Engine` trait. Provides configure, upload, search, and delete operations against Elasticsearch 8.x via the official elasticsearch-rs crate.

## Scope

### In Scope
- Elasticsearch client connection management (auth, timeouts, env vars)
- Dense vector index creation with HNSW parameters
- Bulk upload with UUID ID conversion and parallel threading
- Force merge post-upload with retry logic
- KNN search with num_candidates parameter
- Metadata filter parsing (exact match, range, geo)
- Config JSON parsing (collection_params, search_params, upload_params)
- Integration tests with Docker container
- Engine registration in the factory

### Out of Scope
- Elasticsearch cluster management
- Non-vector search features (full-text, aggregations)
- Monitoring/alerting integration

## Assigned Requirements

| FR | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Index Configuration (HNSW, distance mapping, schema fields) | Must |
| FR-2 | Bulk Upload (bulk API, UUID conversion, force merge, parallel) | Must |
| FR-3 | KNN Search (num_candidates, parallel, UUID-to-int conversion) | Must |
| FR-4 | Metadata Filter Parsing (bool query: match, range, geo_distance) | Must |
| FR-5 | Delete/Cleanup (drop index, ignore NotFound) | Must |
| FR-6 | Config Parsing (JSON config, engine factory registration) | Must |
| FR-7 | Integration Tests (Docker ES container, Makefile targets) | Must |

## Domain Concepts

### Key Entities
| Entity | Description | Attributes |
|--------|-------------|------------|
| ElasticsearchEngine | Engine trait impl | host, config, client |
| ElasticsearchConfig | Parsed config | m, ef_construction, batch_size, parallel, num_candidates |
| ElasticConditionParser | Filter parser | Converts meta_conditions to bool queries |

### Key Operations
| Operation | Description | Inputs | Outputs |
|-----------|-------------|--------|---------|
| configure | Create index with HNSW vector mapping | Dataset, collection_params | Result<()> |
| upload | Bulk upload vectors with UUID IDs | Dataset vectors | UploadStats |
| search | KNN vector search | Query vectors, search_params | SearchResults |
| delete | Drop benchmark index | - | Result<()> |
| parse_filters | Convert meta_conditions to ES bool query | JSON conditions | serde_json::Value |

## Story Summary

| Metric | Count |
|--------|-------|
| Total Stories | 7 |
| Must Have | 7 |
| Should Have | 0 |
| Could Have | 0 |

### Stories

| Story ID | Title | Priority | Status |
|----------|-------|----------|--------|
| 001-connection-and-config | Connection management and config parsing | Must | Planned |
| 002-index-configuration | Create dense vector index with HNSW | Must | Planned |
| 003-bulk-upload | Parallel bulk upload with UUID conversion | Must | Planned |
| 004-force-merge | Post-upload force merge with retry | Must | Planned |
| 005-knn-search | KNN search with parallel execution | Must | Planned |
| 006-filter-parsing | Metadata filter to bool query conversion | Must | Planned |
| 007-integration-tests | Docker-based integration tests | Must | Planned |

## Dependencies

### Depends On
None — self-contained engine module.

### External Dependencies
| System | Purpose | Risk |
|--------|---------|------|
| Elasticsearch 8.x | Target database | Low (stable REST API) |
| elasticsearch-rs crate | Rust client | Medium (verify API coverage) |

## Technical Context

### Suggested Technology
- `elasticsearch` crate for REST API client
- `uuid` crate for ID conversion (int <-> UUID hex)
- `serde_json` for filter query building

### Integration Points
| Integration | Type | Protocol |
|-------------|------|----------|
| Elasticsearch | REST API | HTTP (port 9200) |

## Constraints

- DOT product distance not supported (return error)
- Vector size > 2048 not supported (return error)
- Must follow same Rust patterns as redis.rs and vectorsets.rs engines

## Success Criteria

### Functional
- [ ] All 7 config variants in elasticsearch-single-node.json parseable
- [ ] Upload produces same document count as Python v0
- [ ] Search precision matches Python v0 within float tolerance
- [ ] All filter types (match, range, geo) work correctly

### Non-Functional
- [ ] Parallel upload/search via thread::scope
- [ ] Progress bars via indicatif

### Quality
- [ ] Integration tests pass with Docker Elasticsearch
- [ ] `make check` passes (rustfmt + clippy)

## Bolt Suggestions

| Bolt | Type | Stories | Objective |
|------|------|---------|-----------|
| 001-elasticsearch-engine | simple | 001-004 | Core engine: connection, config, index, upload, merge |
| 002-elasticsearch-engine | simple | 005-007 | Search, filters, and integration tests |

## Notes

- Reference implementation: `v0/engine/clients/elasticsearch/` (Python)
- Existing Rust pattern: `src/bin/vector_db_benchmark/engine/redis.rs` (~1128 lines)
- Config JSON already exists at `experiments/configurations/elasticsearch-single-node.json`
- Docker compose reference: `v0/engine/servers/elasticsearch-single-node/docker-compose.yaml` (ES 8.10.2)
