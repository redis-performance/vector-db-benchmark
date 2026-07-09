---
intent: 009-add-mongodb
phase: complete
status: complete
created: 2026-03-01T00:00:00Z
updated: 2026-03-01T00:00:00Z
---

# Requirements: Add MongoDB Atlas Vector Search Engine

## Intent Overview

Add a new MongoDB Atlas vector search engine client using the official `mongodb` Rust crate (v3, sync). MongoDB uses Atlas Search with `$vectorSearch` aggregation pipeline for approximate nearest neighbor queries.

## Business Goals

| Goal | Success Metric | Priority |
|------|----------------|----------|
| Feature parity with Python v0 | All configure/upload/search operations work identically | Must |
| Precision match | mean_precision matches expected within float tolerance | Must |
| Official client | Use mongodb crate (sync feature) | Must |

## Functional Requirements

### FR-1: Collection & Index Configuration
- **Description**: Create MongoDB collection and vector search index via `createSearchIndexes` command. Support euclidean, cosine, and dotProduct similarity. Poll `listSearchIndexes` for index readiness.
- **Acceptance Criteria**: Collection and vector search index created; index reaches READY status
- **Priority**: Must
- **Status**: Complete

### FR-2: Batch Upload
- **Description**: Upload vectors using `insert_many` in configurable batch sizes. Vectors stored as f64 BSON arrays.
- **Acceptance Criteria**: All vectors uploaded with progress bar
- **Priority**: Must
- **Status**: Complete

### FR-3: Vector Search
- **Description**: Execute `$vectorSearch` aggregation pipeline with configurable numCandidates and limit. Project `_id` and `vectorSearchScore`.
- **Acceptance Criteria**: Search returns correct neighbors; precision validated
- **Priority**: Must
- **Status**: Complete

### FR-4: Metadata Filter Parsing
- **Description**: Convert JSON meta_conditions to MongoDB query operators ($eq, $gt, $lt, $gte, $lte, $ne, $in, $and, $or).
- **Acceptance Criteria**: All filter types supported via `filter` field in $vectorSearch
- **Priority**: Must
- **Status**: Complete

### FR-5: Delete/Cleanup
- **Description**: Drop the benchmark collection on cleanup
- **Acceptance Criteria**: Collection removed after benchmark
- **Priority**: Must
- **Status**: Complete

## Implementation Details

- **Crate**: `mongodb = { version = "3", features = ["sync"] }`
- **Engine file**: `src/bin/vector_db_benchmark/engine/mongodb_engine.rs`
- **Integration tests**: `tests/integration_mongodb.rs`
- **Docker**: `mongodb/mongodb-atlas-local:8.0.4` on port 27018
- **Key pattern**: All mongodb v3 operations use action builder `.run()` pattern

## Non-Functional Requirements

### Constraints
- MongoDB 8.x with Atlas Search required for $vectorSearch
- Vectors stored as f64 BSON arrays (f32 converted to f64)
- Same JSON config format as other engines

## Open Questions

| Question | Owner | Due Date | Resolution |
|----------|-------|----------|------------|
| Which MongoDB image supports vector search? | TBD | Before construction | Resolved: `mongodb/mongodb-atlas-local:8.0.4` includes Atlas Search built-in |
| Sync or async mongodb driver? | TBD | Before construction | Resolved: Sync feature of mongodb v3 crate |
