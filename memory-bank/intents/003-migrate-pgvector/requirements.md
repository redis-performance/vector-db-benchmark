---
intent: 003-migrate-pgvector
phase: complete
status: complete
created: 2026-02-27T00:00:00Z
updated: 2026-03-01T00:00:00Z
---

# Requirements: Migrate pgvector Engine

## Intent Overview

Migrate the pgvector (PostgreSQL) engine client from Python (v0/engine/clients/pgvector/) to Rust. This is a SQL-based engine with a different pattern from the HTTP-based engines.

## Business Goals

| Goal | Success Metric | Priority |
|------|----------------|----------|
| Feature parity with Python v0 | All configure/upload/search operations work identically | Must |
| Precision match | mean_precision matches Python v0 within float tolerance | Must |
| Efficient bulk insertion | Use COPY or pipelined INSERT for upload performance | Should |

## Functional Requirements

### FR-1: Table & Index Configuration
- **Description**: Create PostgreSQL table with pgvector extension, vector column, and HNSW index with distance operator (vector_l2_ops, vector_cosine_ops). Configurable m, ef_construction.
- **Acceptance Criteria**: Table and HNSW index created; pgvector extension registered
- **Priority**: Must

### FR-2: Bulk Upload (COPY)
- **Description**: Upload vectors using PostgreSQL COPY protocol for performance. Python v0 uses single-threaded COPY.
- **Acceptance Criteria**: All vectors uploaded via COPY; progress bar shown
- **Priority**: Must

### FR-3: Vector Search
- **Description**: Execute SQL queries with distance operators (<-> for L2, <=> for cosine). Support SET hnsw.ef_search parameter.
- **Acceptance Criteria**: Search returns correct neighbors; precision matches Python v0
- **Priority**: Must

### FR-4: Metadata Filter Parsing
- **Description**: Convert JSON meta_conditions to SQL WHERE clauses (AND, OR, comparison operators)
- **Acceptance Criteria**: Filter expressions generate valid SQL
- **Priority**: Must

### FR-5: Delete/Cleanup
- **Description**: Drop the benchmark table on cleanup
- **Acceptance Criteria**: Table removed after benchmark
- **Priority**: Must

## Non-Functional Requirements

### Constraints
- DOT product distance: not supported (reject)
- Single-threaded upload (COPY limitation in Python v0 — evaluate if Rust can parallelize)
- Same JSON config format as Python v0

## Assumptions

| Assumption | Risk if Invalid | Mitigation |
|------------|-----------------|------------|
| tokio-postgres or postgres crate supports COPY | May need different upload strategy | Fall back to batched INSERT |
| pgvector extension available on target PostgreSQL | Missing extension | Document prerequisite |

## Open Questions

| Question | Owner | Due Date | Resolution |
|----------|-------|----------|------------|
| Use sync postgres crate or async tokio-postgres? | TBD | Before construction | Resolved: Using sync `postgres` crate v0.19 with `pgvector` crate v0.4 for Vector type. |
| Can we parallelize upload unlike Python v0? | TBD | During construction | Resolved: Upload uses single-threaded COPY protocol (same as Python v0) for correctness. |
