---
id: 001-elasticsearch-engine
unit: 001-elasticsearch-engine
intent: 001-migrate-elasticsearch
type: simple-construction-bolt
status: complete
stories:
  - 001-connection-and-config
  - 002-index-configuration
  - 003-bulk-upload
  - 004-force-merge
created: 2026-02-27T00:00:00.000Z
started: 2026-02-27T10:00:00.000Z
completed: "2026-02-27T11:14:53Z"
current_stage: null
stages_completed:
  - name: plan
    completed: 2026-02-27T10:30:00.000Z
    artifact: implementation-plan.md
  - name: implement
    completed: 2026-02-27T11:00:00.000Z
    artifact: implementation-walkthrough.md
  - name: test
    completed: 2026-02-27T11:30:00.000Z
    artifact: test-walkthrough.md
requires_bolts: []
enables_bolts:
  - 002-elasticsearch-engine
requires_units: []
blocks: false
complexity:
  avg_complexity: 2
  avg_uncertainty: 1
  max_dependencies: 2
  testing_scope: 2
---

# Bolt: 001-elasticsearch-engine

## Overview

First bolt for Elasticsearch engine — establishes the module, connection, config parsing, index creation, bulk upload, and force merge.

## Objective

Implement the core Elasticsearch engine: connect to ES, create HNSW vector index, upload vectors via bulk API with parallel threading, and force merge post-upload. After this bolt, the engine can configure and upload but not yet search.

## Stories Included

- **001-connection-and-config**: Connection management, env vars, config JSON parsing, engine factory registration (Must)
- **002-index-configuration**: Dense vector index creation with HNSW, distance mapping, schema fields (Must)
- **003-bulk-upload**: Parallel bulk upload with UUID conversion, progress bar (Must)
- **004-force-merge**: Post-upload force merge to 1 segment with retry and cluster health wait (Must)

## Bolt Type

**Type**: Simple Construction Bolt
**Definition**: `.specsmd/aidlc/templates/construction/bolt-types/simple-construction-bolt.md`

## Stages

- [ ] **1. Plan**: Define implementation approach, file structure, dependencies
- [ ] **2. Implement**: Write elasticsearch.rs module with configure + upload
- [ ] **3. Test**: Verify upload against containerized ES

## Dependencies

### Requires
- None (first bolt)

### Enables
- 002-elasticsearch-engine (search, filters, integration tests)

## Success Criteria

- [ ] `elasticsearch` and `uuid` crates added to Cargo.toml
- [ ] `src/bin/vector_db_benchmark/engine/elasticsearch.rs` created
- [ ] Engine registered in factory for `"elasticsearch"` engine name
- [ ] Configure creates index with correct HNSW settings
- [ ] Upload inserts all vectors with UUID IDs
- [ ] Force merge completes with retry logic
- [ ] `make check` passes

## Notes

- Follow redis.rs patterns for threading, progress bars, and error handling
- Python reference: v0/engine/clients/elasticsearch/{config,configure,upload}.py
