---
intent: 002-migrate-opensearch
phase: complete
status: complete
created: 2026-02-27T00:00:00Z
updated: 2026-03-01T00:00:00Z
---

# Requirements: Migrate OpenSearch Engine

## Intent Overview

Migrate the OpenSearch vector search engine client from Python (v0/engine/clients/opensearch/) to Rust. OpenSearch is nearly identical to Elasticsearch — this intent leverages shared code from the Elasticsearch migration.

## Business Goals

| Goal | Success Metric | Priority |
|------|----------------|----------|
| Feature parity with Python v0 | All configure/upload/search operations work identically | Must |
| Precision match | mean_precision matches Python v0 within float tolerance | Must |
| Code reuse from Elasticsearch | Shared HTTP client / query builder where possible | Should |

## Functional Requirements

### FR-1: Index Configuration
- **Description**: Create knn_vector index with HNSW method, space type mapping (l2, cosinesimil, innerproduct)
- **Acceptance Criteria**: Index created with correct knn settings
- **Priority**: Must

### FR-2: Bulk Upload
- **Description**: Upload vectors using bulk API with UUID conversion. Post-upload force merge.
- **Acceptance Criteria**: All vectors uploaded; force merge completes
- **Priority**: Must

### FR-3: KNN Search
- **Description**: Execute KNN query with optional metadata filter wrapping. Includes setup_search for index settings.
- **Acceptance Criteria**: Search returns correct neighbors; precision matches Python v0
- **Priority**: Must

### FR-4: Metadata Filter Parsing
- **Description**: Convert JSON meta_conditions to OpenSearch bool queries (identical to Elasticsearch)
- **Acceptance Criteria**: All filter types supported
- **Priority**: Must

### FR-5: Delete/Cleanup
- **Description**: Delete the benchmark index on cleanup
- **Acceptance Criteria**: Index removed after benchmark
- **Priority**: Must

## Non-Functional Requirements

### Constraints
- Vector size limit: 2048 dimensions (reject larger)
- DOT product distance: not supported (reject)
- Same JSON config format as Python v0

## Assumptions

| Assumption | Risk if Invalid | Mitigation |
|------------|-----------------|------------|
| OpenSearch REST API is compatible enough with ES to share code | Divergent APIs | Separate implementation if needed |
| opensearch-rs crate exists and is usable | May not exist | Use reqwest with REST API directly |

## Open Questions

| Question | Owner | Due Date | Resolution |
|----------|-------|----------|------------|
| How much code can be shared with Elasticsearch? | TBD | After ES migration | Resolved: opensearch crate mirrors elasticsearch crate API closely. Separate implementation but identical patterns (JsonBody, IndicesForcemergeParts, TransportBuilder). |
