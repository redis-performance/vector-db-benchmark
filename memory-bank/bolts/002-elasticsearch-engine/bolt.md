---
id: 002-elasticsearch-engine
unit: 001-elasticsearch-engine
intent: 001-migrate-elasticsearch
type: simple-construction-bolt
status: complete
stories:
  - 005-knn-search
  - 006-filter-parsing
  - 007-integration-tests
created: 2026-02-27T00:00:00.000Z
started: 2026-02-27T12:00:00.000Z
completed: "2026-02-27T11:31:08Z"
current_stage: null
stages_completed:
  - name: plan
    completed: 2026-02-27T12:15:00.000Z
    artifact: implementation-plan.md
  - name: implement
    completed: 2026-02-27T14:15:00.000Z
    artifact: implementation-walkthrough.md
requires_bolts:
  - 001-elasticsearch-engine
enables_bolts: []
requires_units: []
blocks: false
complexity:
  avg_complexity: 2
  avg_uncertainty: 1
  max_dependencies: 2
  testing_scope: 3
---

# Bolt: 002-elasticsearch-engine

## Overview

Second bolt for Elasticsearch engine — adds KNN search with parallel execution, metadata filter parsing, and comprehensive integration tests.

## Objective

Complete the Elasticsearch Engine trait implementation by adding search functionality with filter support, then validate everything with Docker-based integration tests.

## Stories Included

- **005-knn-search**: KNN vector search with num_candidates, parallel threading, UUID-to-int conversion (Must)
- **006-filter-parsing**: Metadata filter to ES bool query conversion (match, range, geo_distance) (Must)
- **007-integration-tests**: Docker ES container, configure/upload/search/delete test cycle (Must)

## Bolt Type

**Type**: Simple Construction Bolt
**Definition**: `.specsmd/aidlc/templates/construction/bolt-types/simple-construction-bolt.md`

## Stages

- [ ] **1. Plan**: Define search implementation, filter parser structure, test strategy
- [ ] **2. Implement**: Add search + filter parsing to elasticsearch.rs, update docker-compose and Makefile
- [ ] **3. Test**: Run integration tests against containerized ES, verify precision matches Python v0

## Dependencies

### Requires
- 001-elasticsearch-engine (connection, config, upload must exist first)

### Enables
- None (completes the Elasticsearch migration)

## Success Criteria

- [ ] KNN search returns correct results
- [ ] All filter types work (match, range, geo)
- [ ] Precision matches Python v0 within float tolerance
- [ ] `tests/docker-compose.test.yml` includes ES service
- [ ] Integration tests pass
- [ ] `make check` and `make integration-test` pass
- [ ] Delete/cleanup works correctly

## Notes

- Filter parser builds serde_json::Value objects (no typed ES query structs needed)
- Python reference: v0/engine/clients/elasticsearch/{search,parser}.py
- Test reference: tests/integration_redis.rs
