---
id: 004-force-merge
unit: 001-elasticsearch-engine
intent: 001-migrate-elasticsearch
status: complete
priority: must
created: 2026-02-27T00:00:00.000Z
assigned_bolt: 001-elasticsearch-engine
implemented: true
---

# Story: 004-force-merge

## User Story

**As a** benchmark operator
**I want** a force merge to 1 segment after upload completes
**So that** search performance is optimal and consistent for benchmarking

## Acceptance Criteria

- [ ] **Given** upload is complete, **When** post-upload runs, **Then** force merge to max_num_segments=1 is executed
- [ ] **Given** force merge fails with a transient error, **When** retrying, **Then** up to 30 retries are attempted
- [ ] **Given** force merge succeeds, **When** checking cluster, **Then** cluster health is at least yellow before returning

## Technical Notes

- Call `_forcemerge` API with `wait_for_completion=true, max_num_segments=1`
- Retry on TLS errors and API errors (match Python v0 behavior)
- After merge, poll cluster health for yellow status (up to 100 attempts, 0.1s sleep)
- Reference: `v0/engine/clients/elasticsearch/upload.py` `post_upload()` method

## Dependencies

### Requires
- 003-bulk-upload

### Enables
- 005-knn-search (search should only run after merge)

## Edge Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| Merge times out after 30 retries | Return error |
| Cluster never reaches yellow | Return error after 100 attempts |

## Out of Scope

- Configurable merge parameters
- Segment monitoring
