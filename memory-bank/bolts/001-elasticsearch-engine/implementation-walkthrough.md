---
stage: implement
bolt: 001-elasticsearch-engine
created: 2026-02-27T11:00:00Z
---

## Implementation Walkthrough: Elasticsearch Engine (Bolt 001)

### Summary

Implemented the Elasticsearch vector database engine as a Rust module using reqwest for HTTP calls against the ES REST API. The engine supports index creation with HNSW parameters, parallel bulk upload with UUID IDs, force merge post-upload, and engine factory registration.

### Structure Overview

Single-file engine module following the same pattern as redis.rs. Uses reqwest::blocking::Client for all Elasticsearch REST API calls. Parallel upload uses the same thread::scope + AtomicUsize batch distribution pattern established in redis.rs.

### Completed Work

- [x] `Cargo.toml` - Added `uuid = "1"` dependency, added `json` feature to `reqwest`
- [x] `src/bin/vector_db_benchmark/config.rs` - Added `IndexOptions` struct and field to `CollectionParams` for ES config parsing
- [x] `src/bin/vector_db_benchmark/engine/elasticsearch.rs` - Full Engine trait implementation (~400 lines)
- [x] `src/bin/vector_db_benchmark/engine/mod.rs` - Registered `"elasticsearch"` in engine factory, added module and pub use

### Key Decisions

- **reqwest over elasticsearch crate**: Used existing `reqwest::blocking` dependency instead of adding the heavy `elasticsearch` crate. All ES operations are simple HTTP JSON calls (PUT, POST, DELETE).
- **Basic auth in URL**: Embedded credentials in the URL (`http://user:pass@host:port`) rather than per-request headers, matching the simple auth pattern.
- **UUID conversion**: Used `Uuid::from_u128(id as u128).as_simple().to_string()` to match Python v0's `uuid.UUID(int=idx).hex`.
- **Search stub**: Implemented `search()` as an error stub since search is bolt 002 scope.

### Deviations from Plan

- Removed `host` and `port` fields from struct (only `base_url` needed, avoids clippy warnings)
- Added `json` feature to reqwest (needed for `.json()` on request builder and response)

### Dependencies Added

- [x] `uuid = "1"` - Integer-to-UUID hex conversion matching Python v0 behavior
- [x] `reqwest` `json` feature - JSON serialization/deserialization on HTTP requests and responses

### Developer Notes

- Bulk API uses NDJSON format (newline-delimited JSON) with Content-Type `application/x-ndjson`
- Force merge retries up to 30 times, matching Python v0 behavior exactly
- Cluster health polling waits for yellow status with 100 attempts at 100ms intervals
- DOT product distance and vector_size > 2048 are rejected with errors per ES limitations
