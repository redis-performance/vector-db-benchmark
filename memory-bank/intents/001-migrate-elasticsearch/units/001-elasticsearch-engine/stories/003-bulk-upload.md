---
id: 003-bulk-upload
unit: 001-elasticsearch-engine
intent: 001-migrate-elasticsearch
status: complete
priority: must
created: 2026-02-27T00:00:00.000Z
assigned_bolt: 001-elasticsearch-engine
implemented: true
---

# Story: 003-bulk-upload

## User Story

**As a** benchmark operator
**I want** vectors uploaded to Elasticsearch via bulk API with parallel threading
**So that** large datasets are ingested efficiently

## Acceptance Criteria

- [ ] **Given** a dataset of N vectors, **When** uploading, **Then** all N documents are indexed in Elasticsearch
- [ ] **Given** integer IDs (0..N), **When** uploading, **Then** IDs are converted to UUID hex strings (matching Python v0 `uuid.UUID(int=idx).hex`)
- [ ] **Given** metadata for each vector, **When** uploading, **Then** metadata fields are included in the document alongside the vector
- [ ] **Given** `upload_params.parallel: 16`, **When** uploading, **Then** upload uses thread::scope with 16 threads
- [ ] **Given** upload in progress, **When** observing CLI, **Then** an indicatif progress bar shows upload rate

## Technical Notes

- Bulk API format: alternating `{"index": {"_id": uuid_hex}}` and `{"vector": [...], ...metadata}` lines
- Use `thread::scope` + `AtomicUsize` batch distribution pattern from redis.rs
- Batch size from upload_params (default from config)
- Reference: `v0/engine/clients/elasticsearch/upload.py`

## Dependencies

### Requires
- 001-connection-and-config
- 002-index-configuration

### Enables
- 004-force-merge

## Edge Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| Empty metadata | Upload document with only vector field |
| Bulk API partial failure | Report error with failed document count |
| Single thread mode (parallel: 1) | Sequential upload, same logic |

## Out of Scope

- Streaming/chunked upload
- Resume from partial upload
