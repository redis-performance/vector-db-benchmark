---
intent: 006-migrate-qdrant
phase: complete
status: complete
created: 2026-02-27T00:00:00Z
updated: 2026-03-01T00:00:00Z
---

# Requirements: Migrate Qdrant Engine

## Intent Overview

Migrate the Qdrant vector search engine client from Python (v0/engine/clients/qdrant/) to Rust. Qdrant has the most sophisticated client with cloud integration, status synchronization, and a rich filter model.

## Business Goals

| Goal | Success Metric | Priority |
|------|----------------|----------|
| Feature parity with Python v0 | All configure/upload/search operations work identically | Must |
| Precision match | mean_precision matches Python v0 within float tolerance | Must |
| Cloud support | Support both local and Qdrant Cloud deployments | Should |

## Functional Requirements

### FR-1: Collection Configuration
- **Description**: Recreate collection with vector config (EUCLID, COSINE, DOT distances). Create payload indices for metadata fields. Implement exponential backoff retry logic.
- **Acceptance Criteria**: Collection created with correct vector params; payload indices created
- **Priority**: Must

### FR-2: Batch Upload with Sync
- **Description**: Upload vectors using batch upsert. Implement wait_collection_green() synchronization to ensure all data is indexed before search.
- **Acceptance Criteria**: All vectors uploaded; collection green before search; progress bar shown
- **Priority**: Must

### FR-3: Vector Search
- **Description**: Execute search() with optional query filters. Set gRPC preferences for performance.
- **Acceptance Criteria**: Search returns correct neighbors; precision matches Python v0
- **Priority**: Must

### FR-4: Metadata Filter Parsing
- **Description**: Convert JSON meta_conditions to Qdrant filter model (FieldCondition, Range, GeoRadius, Match). Support nested AND/OR conditions.
- **Acceptance Criteria**: All filter types from Python parser supported (the cleanest filter implementation)
- **Priority**: Must

### FR-5: Delete/Cleanup
- **Description**: Delete the benchmark collection on cleanup
- **Acceptance Criteria**: Collection removed after benchmark
- **Priority**: Must

### FR-6: Cloud Integration (Optional)
- **Description**: Support Qdrant Cloud deployments (QDRANT_URL, API key auth). Collect cloud usage metrics.
- **Acceptance Criteria**: Can benchmark against Qdrant Cloud instances
- **Priority**: Should

## Non-Functional Requirements

### Constraints
- Supports both gRPC and REST APIs
- Distance metrics: EUCLID, COSINE, DOT (all three supported)
- Same JSON config format as Python v0

## Assumptions

| Assumption | Risk if Invalid | Mitigation |
|------------|-----------------|------------|
| qdrant-client Rust crate is mature and maintained | Missing features | Use REST API via reqwest as fallback |
| Qdrant gRPC API is stable | Breaking changes | Pin to specific qdrant-client version |

## Open Questions

| Question | Owner | Due Date | Resolution |
|----------|-------|----------|------------|
| Use official qdrant-client Rust SDK or build custom? | TBD | Before construction | Resolved: Using official `qdrant-client` v1.13 crate with gRPC transport. tokio::runtime::Runtime for async wrapping. |
| Include cloud metrics collection or skip? | TBD | Before construction | Resolved: Skipped for initial implementation. Can add later via collection_info API. |
