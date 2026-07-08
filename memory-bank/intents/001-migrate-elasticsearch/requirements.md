---
intent: 001-migrate-elasticsearch
phase: inception
status: complete
created: 2026-02-27T00:00:00Z
updated: 2026-02-27T00:00:00Z
---

# Requirements: Migrate Elasticsearch Engine

## Intent Overview

Migrate the Elasticsearch vector search engine client from Python (v0/engine/clients/elasticsearch/) to Rust, implementing the Engine trait. Uses the official elasticsearch-rs crate. Full feature parity including metadata filter parsing.

## Business Goals

| Goal | Success Metric | Priority |
|------|----------------|----------|
| Feature parity with Python v0 | All configure/upload/search operations work identically | Must |
| Precision match | mean_precision matches Python v0 within float tolerance | Must |
| Integration testing | Automated tests with Elasticsearch container | Must |

## Functional Requirements

### FR-1: Index Configuration
- **Description**: Create Elasticsearch dense vector index with HNSW parameters (m, ef_construction from config JSON), distance metric mapping (L2->l2_norm, COSINE->cosine), metadata field indexing (int->long, geo->geo_point), and `_source.excludes: ["vector"]`
- **Acceptance Criteria**: Index created with correct mappings matching Python v0 behavior; incompatibility errors raised for DOT product and vector_size > 2048
- **Priority**: Must

### FR-2: Bulk Upload
- **Description**: Upload vectors using Elasticsearch bulk API. Convert integer IDs to UUID hex strings. Support parallel upload via `thread::scope`. Post-upload: force merge to 1 segment with retry logic (up to 30 retries). Wait for cluster yellow status.
- **Acceptance Criteria**: All vectors uploaded; force merge completes; progress bar shown via indicatif; upload count matches dataset size
- **Priority**: Must

### FR-3: KNN Search
- **Description**: Execute KNN vector search with configurable top-k and `num_candidates` (from search_params). Support parallel search via thread::scope. Convert UUID hex results back to integer IDs.
- **Acceptance Criteria**: Search returns correct neighbors; precision matches Python v0 within float tolerance
- **Priority**: Must

### FR-4: Metadata Filter Parsing
- **Description**: Convert JSON meta_conditions to Elasticsearch bool queries: exact match (`{"match": ...}`), range (`{"range": ...}`), geo (`{"geo_distance": ...}`), with AND (must) / OR (should) composition
- **Acceptance Criteria**: All filter types supported; filters produce same search results as Python v0
- **Priority**: Must

### FR-5: Delete/Cleanup
- **Description**: Delete the benchmark index (`bench` by default) on cleanup
- **Acceptance Criteria**: Index removed; NotFoundError silently ignored
- **Priority**: Must

### FR-6: Config Parsing
- **Description**: Parse `experiments/configurations/elasticsearch-single-node.json` — extract collection_params.index_options (m, ef_construction), upload_params (parallel), search_params (parallel, num_candidates), connection_params (request_timeout)
- **Acceptance Criteria**: All 7 config variants parseable; engine registered in engine factory
- **Priority**: Must

### FR-7: Integration Tests
- **Description**: Add Elasticsearch 8.10.2 container to `tests/docker-compose.test.yml`. Write integration tests covering configure, upload, search, and delete operations.
- **Acceptance Criteria**: Tests pass with containerized Elasticsearch; added to Makefile targets
- **Priority**: Must

## Non-Functional Requirements

### NFR-1: Dependency
- Crate: `elasticsearch` (official elasticsearch-rs)
- Must work with Elasticsearch 8.x REST API

### NFR-2: Environment Variables
- `ELASTIC_PORT` (default: 9200), `ELASTIC_USER` (default: "elastic"), `ELASTIC_PASSWORD` (default: "passwd"), `ELASTIC_API_KEY` (optional), `ELASTIC_TIMEOUT` (default: 300s), `ELASTIC_INDEX` (default: "bench")

### NFR-3: Compatibility
- Same JSON config format as Python v0 (exists at `experiments/configurations/elasticsearch-single-node.json`)
- Same results JSON format for precision comparison

## Constraints

### Technical Constraints
- DOT product distance: not supported (reject with error)
- Vector size limit: 2048 dimensions (reject larger)
- Uses Rust threading model (thread::scope + AtomicUsize), not multiprocessing

## Assumptions

| Assumption | Risk if Invalid | Mitigation |
|------------|-----------------|------------|
| elasticsearch-rs crate provides bulk API and KNN search | Missing features | Fall back to raw HTTP via reqwest |
| Elasticsearch 8.10.2 Docker image available | Image unavailable | Use latest 8.x |
