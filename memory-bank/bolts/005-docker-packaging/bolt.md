---
id: 005-docker-packaging
unit: 001-docker-packaging
intent: 008-docker-build
type: simple-construction-bolt
status: complete
stories:
  - 001-multi-stage-dockerfile
  - 002-docker-compose-services
  - 003-make-targets
created: 2026-02-27T15:10:00.000Z
started: 2026-02-27T15:15:00.000Z
completed: "2026-02-27T17:17:37Z"
current_stage: null
stages_completed:
  - name: plan
    completed: 2026-02-27T15:20:00.000Z
    artifact: implementation-plan.md
  - name: implement
    completed: 2026-02-27T15:30:00.000Z
    artifact: implementation-walkthrough.md
requires_bolts: []
enables_bolts: []
requires_units: []
blocks: false
complexity:
  avg_complexity: 1
  avg_uncertainty: 1
  max_dependencies: 2
  testing_scope: 3
---

# Bolt: 005-docker-packaging

## Overview

Single bolt covering the entire Docker packaging intent — Dockerfile, docker-compose, and Make targets. Small scope, tightly coupled artifacts.

## Objective

Replace the Python Dockerfile with a multi-stage Rust build, add docker-compose orchestration for benchmark + Redis, and create Make targets that build the image and run `h-and-m-2048-angular-filters` as a Docker-based integration test.

## Stories Included

- **001-multi-stage-dockerfile**: Multi-stage Rust Dockerfile with HDF5 support, dependency caching, slim runtime (Must)
- **002-docker-compose-services**: Docker-compose with benchmark + Redis on shared network (Must)
- **003-make-targets**: `make docker-build` and `make docker-integration` targets (Should)

## Bolt Type

**Type**: Simple Construction Bolt
**Definition**: `.specsmd/aidlc/templates/construction/bolt-types/simple-construction-bolt.md`

## Stages

- [ ] **1. Plan**: Define Dockerfile structure, compose layout, Make target flow
- [ ] **2. Implement**: Write Dockerfile, docker-compose, Makefile additions
- [ ] **3. Test**: Run `make docker-integration` end-to-end, verify pass

## Dependencies

### Requires
- None (standalone infrastructure bolt)

### Enables
- None

## Success Criteria

- [ ] `docker build -t vector-db-benchmark .` succeeds
- [ ] `docker run vector-db-benchmark --help` works
- [ ] `docker run vector-db-benchmark --describe datasets` works
- [ ] Final image < 150MB
- [ ] `make docker-integration` runs h-and-m-2048-angular-filters against Redis and exits 0
- [ ] Containers cleaned up after run
- [ ] `make check` still passes (no regressions)

## Notes

- Existing `Dockerfile` is Python-era (Poetry, maturin, PyO3) — replace entirely
- Follow existing Makefile patterns from `integration-test` target
- Use `redis-default-simple` config from `redis-single-node.json`
- h-and-m-2048-angular-filters: 105K vectors, 2048d, cosine, tar format
