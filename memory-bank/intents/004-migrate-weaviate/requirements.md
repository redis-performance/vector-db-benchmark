---
intent: 004-migrate-weaviate
phase: complete
status: complete
created: 2026-02-27T00:00:00Z
updated: 2026-03-01T00:00:00Z
---

# Requirements: Migrate Weaviate Engine

## Intent Overview

Migrate the Weaviate vector search engine client from Python (v0/engine/clients/weaviate/) to Rust. Weaviate uses HTTP REST + gRPC APIs with a class/property data model.

## Business Goals

| Goal | Success Metric | Priority |
|------|----------------|----------|
| Feature parity with Python v0 | All configure/upload/search operations work identically | Must |
| Precision match | mean_precision matches Python v0 within float tolerance | Must |

## Functional Requirements

### FR-1: Collection Configuration
- **Description**: Create Weaviate collection (class) with vector index config (HNSW), distance metric (l2-squared, cosine, dot), and property schema for metadata. Disable vectorizer (vectors provided manually).
- **Acceptance Criteria**: Collection created with correct schema and vectorIndexConfig
- **Priority**: Must

### FR-2: Batch Upload
- **Description**: Upload vectors using insert_many batch API with properties and vectors as DataObject instances
- **Acceptance Criteria**: All vectors uploaded; progress bar shown
- **Priority**: Must

### FR-3: Near Vector Search
- **Description**: Execute near_vector() queries with configurable top-k, optional metadata filters, and return_metadata for distance
- **Acceptance Criteria**: Search returns correct neighbors; precision matches Python v0
- **Priority**: Must

### FR-4: Metadata Filter Parsing
- **Description**: Convert JSON meta_conditions to Weaviate filter expressions (AND, OR with comparison operators)
- **Acceptance Criteria**: Filter expressions generate valid Weaviate queries; geo support included
- **Priority**: Must

### FR-5: Delete/Cleanup
- **Description**: Delete the benchmark collection on cleanup
- **Acceptance Criteria**: Collection removed after benchmark
- **Priority**: Must

## Non-Functional Requirements

### Constraints
- Weaviate uses HTTP port 8080 and gRPC port 50051
- Distance metrics: l2-squared, cosine, dot
- Same JSON config format as Python v0

## Assumptions

| Assumption | Risk if Invalid | Mitigation |
|------------|-----------------|------------|
| Weaviate REST API is sufficient (no need for gRPC) | Performance loss | Add gRPC via tonic if needed |
| No official Rust SDK exists | Must build HTTP client | Use reqwest with Weaviate REST API |

## Open Questions

| Question | Owner | Due Date | Resolution |
|----------|-------|----------|------------|
| Use REST API only or also gRPC? | TBD | Before construction | Resolved: REST API only via reqwest. Sufficient for all operations. |
| Is there a community Rust SDK for Weaviate? | TBD | Before construction | Resolved: No official Rust client exists. Using reqwest with REST + GraphQL APIs. |
