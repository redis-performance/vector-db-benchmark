---
intent: 005-migrate-milvus
phase: complete
status: complete
created: 2026-02-27T00:00:00Z
updated: 2026-03-01T00:00:00Z
---

# Requirements: Migrate Milvus Engine

## Intent Overview

Migrate the Milvus vector database engine client from Python (v0/engine/clients/milvus/) to Rust. Milvus uses gRPC with a typed schema model and requires index building synchronization with backoff.

## Business Goals

| Goal | Success Metric | Priority |
|------|----------------|----------|
| Feature parity with Python v0 | All configure/upload/search operations work identically | Must |
| Precision match | mean_precision matches Python v0 within float tolerance | Must |
| Robust index building | Wait for index completion with backoff, matching Python behavior | Must |

## Functional Requirements

### FR-1: Collection Configuration
- **Description**: Create Milvus collection with typed schema (INT64 primary key, VARCHAR, DOUBLE, FLOAT_VECTOR). Drop existing indices. Support cosine only on normalized vectors.
- **Acceptance Criteria**: Collection created with correct field types; existing indices dropped
- **Priority**: Must

### FR-2: Batch Upload with Index Build
- **Description**: Upload vectors using insert API with batch support. Post-upload: create HNSW index (configurable type: HNSW, IVF_FLAT, IVF_SQ8, IVF_PQ). Implement exponential backoff retry and wait_for_index_building_complete synchronization.
- **Acceptance Criteria**: All vectors uploaded; index built; wait for completion before search
- **Priority**: Must

### FR-3: Vector Search
- **Description**: Execute collection.search() with parametrized search params and optional filter expressions
- **Acceptance Criteria**: Search returns correct neighbors; precision matches Python v0
- **Priority**: Must

### FR-4: Metadata Filter Parsing
- **Description**: Convert JSON meta_conditions to Milvus expression language (&&, ||, ==, <, >, <=, >=)
- **Acceptance Criteria**: All filter types from Python parser supported
- **Priority**: Must

### FR-5: Delete/Cleanup
- **Description**: Drop the benchmark collection on cleanup
- **Acceptance Criteria**: Collection removed after benchmark
- **Priority**: Must

## Non-Functional Requirements

### Constraints
- Milvus uses gRPC on port 19530
- Index types: HNSW (default), IVF_FLAT, IVF_SQ8, IVF_PQ
- Cosine distance requires normalized vectors
- Same JSON config format as Python v0

## Assumptions

| Assumption | Risk if Invalid | Mitigation |
|------------|-----------------|------------|
| milvus-sdk-rust or tonic-based client available | Must generate gRPC stubs from proto | Use tonic + Milvus proto definitions |
| Milvus gRPC API is stable | Breaking changes | Pin to specific Milvus version |

## Open Questions

| Question | Owner | Due Date | Resolution |
|----------|-------|----------|------------|
| Use milvus-sdk-rust or generate gRPC client from proto? | TBD | Before construction | Resolved: No official Rust client. Using reqwest with Milvus REST API v2. |
| Which Milvus version to target? | TBD | Before construction | Resolved: Milvus v2.5.6 (Docker image for integration tests). |
