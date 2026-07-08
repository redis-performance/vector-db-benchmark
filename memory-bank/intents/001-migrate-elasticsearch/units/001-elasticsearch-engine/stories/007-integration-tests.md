---
id: 007-integration-tests
unit: 001-elasticsearch-engine
intent: 001-migrate-elasticsearch
status: complete
priority: must
created: 2026-02-27T00:00:00.000Z
assigned_bolt: 002-elasticsearch-engine
implemented: true
---

# Story: 007-integration-tests

## User Story

**As a** developer
**I want** automated integration tests for the Elasticsearch engine
**So that** regressions are caught before merge

## Acceptance Criteria

- [ ] **Given** `tests/docker-compose.test.yml`, **When** updated, **Then** includes Elasticsearch 8.10.2 service on a test port
- [ ] **Given** Elasticsearch container is running, **When** running integration tests, **Then** configure/upload/search/delete cycle completes successfully
- [ ] **Given** a small test dataset, **When** uploading and searching, **Then** precision is 1.0 for exact match
- [ ] **Given** the Makefile, **When** checking targets, **Then** `make integration-test` includes Elasticsearch tests
- [ ] **Given** `make check`, **When** run, **Then** new code passes rustfmt and clippy

## Technical Notes

- Add ES service to `tests/docker-compose.test.yml` (use different port, e.g., 9200 or 9201 to avoid conflict)
- ES settings: `discovery.type: single-node`, `xpack.security.enabled: false`
- Test pattern: follow `tests/integration_redis.rs` structure
- Create small in-memory test vectors (e.g., 100 vectors, 128 dims)
- Test both with and without metadata filters

## Dependencies

### Requires
- 001-connection-and-config
- 002-index-configuration
- 003-bulk-upload
- 005-knn-search
- 006-filter-parsing

### Enables
- None (validation story)

## Edge Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| ES container not running | Test skipped or fails with clear message |
| Port conflict | Use dedicated test port |

## Out of Scope

- Performance benchmarking in tests (only correctness)
- CI pipeline changes beyond Makefile
