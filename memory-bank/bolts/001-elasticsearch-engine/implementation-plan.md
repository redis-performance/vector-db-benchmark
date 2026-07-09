---
stage: plan
bolt: 001-elasticsearch-engine
created: 2026-02-27T10:00:00Z
---

## Implementation Plan: Elasticsearch Engine (Bolt 001)

### Objective

Implement the core Elasticsearch engine module: connection management, config parsing, dense vector index creation with HNSW, parallel bulk upload with UUID IDs, force merge post-upload, and engine factory registration. After this bolt, the engine can configure and upload but not yet search.

### Deliverables

- `elasticsearch` and `uuid` crates added to `Cargo.toml`
- `src/bin/vector_db_benchmark/engine/elasticsearch.rs` — full module
- Engine registered in `src/bin/vector_db_benchmark/engine/mod.rs` factory for `"elasticsearch"`
- `CollectionParams` extended with `index_options` field in `src/bin/vector_db_benchmark/config.rs`

### Dependencies

- `reqwest` (blocking): Already in Cargo.toml — will use for all ES REST API calls instead of the `elasticsearch` crate, keeping dependencies minimal
- `uuid`: New — for converting integer IDs to UUID hex strings (matching Python v0 `uuid.UUID(int=idx).hex`)
- `serde_json`: Already available — for building ES request bodies

### Technical Approach

**HTTP Client (not elasticsearch crate)**: Use `reqwest::blocking::Client` directly against the ES REST API. The operations are simple JSON-over-HTTP and don't warrant a dedicated ES client crate. This matches the project's existing approach of minimal dependencies.

**Config Parsing**: The ES config JSON uses `collection_params.index_options` with lowercase keys (`m`, `ef_construction`), differing from Redis's `hnsw_config` with uppercase keys. Add an `IndexOptions` struct and `index_options` field to `CollectionParams`. The `ElasticsearchEngine` constructor extracts `m`, `ef_construction`, `batch_size`, and `parallel` from the `EngineConfig`.

**ElasticsearchEngine struct**:
```
ElasticsearchEngine {
    name: String,
    host: String,         // from --host arg
    port: u16,            // ELASTIC_PORT env (default 9200)
    index_name: String,   // ELASTIC_INDEX env (default "bench")
    timeout: u64,         // ELASTIC_TIMEOUT env (default 300)
    client: reqwest::blocking::Client,
    config: ElasticsearchConfig,
    search_params: Vec<SearchParams>,
}

ElasticsearchConfig {
    m: i64,               // from collection_params.index_options.m (default 16)
    ef_construction: i64, // from collection_params.index_options.ef_construction (default 100)
    batch_size: usize,    // from upload_params.batch_size (default 500)
    parallel: usize,      // from upload_params.parallel (default 16)
}
```

**Authentication**: Check env vars in order:
1. `ELASTIC_API_KEY` → use `Authorization: ApiKey {key}` header
2. `ELASTIC_USER` + `ELASTIC_PASSWORD` → use HTTP Basic auth
3. Neither → connect without auth

**Configure (create index)**:
1. Delete existing index (ignore 404)
2. Validate: return error if distance=DOT or vector_size > 2048
3. PUT /{index} with settings (shards=1, replicas=0, refresh=10s) and mappings (dense_vector + schema fields)
4. Distance mapping: L2 → "l2_norm", COSINE → "cosine"
5. Schema field type mapping: int → "long", geo → "geo_point", others pass through

**Upload (bulk API)**:
1. Follow redis.rs parallel pattern: `thread::scope` + `AtomicUsize` batch counter
2. Each thread gets its own `reqwest::blocking::Client` with auth headers
3. Build bulk body: alternating `{"index": {"_id": uuid_hex}}` and `{"vector": [...], ...metadata}` NDJSON lines
4. POST /_bulk with Content-Type: application/x-ndjson
5. Progress bar via indicatif (same style as redis.rs)

**Force Merge (post-upload)**:
1. POST /{index}/_forcemerge?wait_for_completion=true&max_num_segments=1
2. Retry up to 30 times on error
3. After merge, poll GET /_cluster/health?wait_for_status=yellow (up to 100 attempts, 100ms sleep)

**Delete**: DELETE /{index} — ignore 404

**Search**: Stub returning error "search not implemented (see bolt 002)" — bolt 002 will add KNN search

**Engine trait**: Implement all 6 methods (`name`, `configure`, `upload`, `search`, `delete`, `search_params`)

### Config changes to `config.rs`

Add `IndexOptions` struct to `CollectionParams`:
```rust
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct IndexOptions {
    pub m: Option<i64>,
    pub ef_construction: Option<i64>,
}

pub struct CollectionParams {
    pub hnsw_config: Option<HnswConfig>,
    pub index_options: Option<IndexOptions>,  // NEW — for Elasticsearch
}
```

### Acceptance Criteria

- [ ] `uuid` crate added to Cargo.toml
- [ ] `IndexOptions` added to `CollectionParams` in config.rs
- [ ] `src/bin/vector_db_benchmark/engine/elasticsearch.rs` created with full Engine trait impl
- [ ] Engine registered in factory for `"elasticsearch"` engine name
- [ ] Configure creates index with correct HNSW settings and distance mapping
- [ ] DOT distance and vector_size > 2048 return errors
- [ ] Schema fields mapped correctly (int→long, geo→geo_point)
- [ ] Upload inserts all vectors with UUID hex IDs
- [ ] Parallel upload uses thread::scope pattern matching redis.rs
- [ ] Progress bar shown during upload
- [ ] Force merge to 1 segment with 30 retries
- [ ] Cluster health wait (yellow) after merge
- [ ] Delete ignores 404
- [ ] `make check` passes (rustfmt + clippy)
