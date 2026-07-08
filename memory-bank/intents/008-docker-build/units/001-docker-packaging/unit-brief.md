---
unit: 001-docker-packaging
intent: 008-docker-build
phase: inception
status: complete
created: 2026-02-27T15:10:00.000Z
updated: 2026-02-27T15:10:00.000Z
---

# Unit Brief: Docker Packaging

## Purpose

Package the Rust vector-db-benchmark binary into a small Docker image and provide Make targets for building the image and running an end-to-end integration test against Redis inside Docker.

## Scope

### In Scope
- Multi-stage Dockerfile replacing the Python version
- Cargo dependency layer caching
- HDF5 build-time and runtime library support
- Docker-compose file with benchmark + Redis services
- `make docker-build` and `make docker-integration` targets
- Volume mount support for datasets, experiments, results

### Out of Scope
- CI/CD pipeline integration
- Docker image publishing to a registry
- Support for engines other than Redis in the docker-integration test

---

## Assigned Requirements

| FR | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Rust Multi-Stage Dockerfile | Must |
| FR-2 | Volume Mounts for Datasets, Experiments, and Results | Must |
| FR-3 | Docker-Compose Integration Test Service | Must |
| FR-4 | `make docker-integration` Target | Must |
| FR-5 | `make docker-build` Target | Should |

---

## Domain Concepts

### Key Entities
| Entity | Description | Attributes |
|--------|-------------|------------|
| Dockerfile | Multi-stage build definition | builder stage, runtime stage |
| Docker Compose | Service orchestration | benchmark service, redis service, network |
| Makefile targets | Build/test automation | docker-build, docker-integration |

### Key Operations
| Operation | Description | Inputs | Outputs |
|-----------|-------------|--------|---------|
| docker-build | Build container image | Cargo.toml, src/, Dockerfile | vector-db-benchmark:latest image |
| docker-integration | End-to-end test in Docker | Image, datasets, redis config | Pass/fail exit code |

---

## Story Summary

| Metric | Count |
|--------|-------|
| Total Stories | 3 |
| Must Have | 2 |
| Should Have | 1 |
| Could Have | 0 |

### Stories

| Story ID | Title | Priority | Status |
|----------|-------|----------|--------|
| 001-multi-stage-dockerfile | Multi-stage Rust Dockerfile | Must | Planned |
| 002-docker-compose-services | Docker-compose integration services | Must | Planned |
| 003-make-targets | Make targets for docker build and integration | Should | Planned |

---

## Dependencies

### Depends On
None

### Depended By
None

### External Dependencies
| System | Purpose | Risk |
|--------|---------|------|
| Docker Hub | Base images (rust, debian, redis) | Low — standard public images |
| apt repos | libhdf5-dev / libhdf5-103-1 | Low — standard Debian packages |

---

## Technical Context

### Suggested Technology
- Multi-stage Docker build (rust:bookworm → debian:bookworm-slim)
- Docker Compose v2
- Make

### Integration Points
| Integration | Type | Protocol |
|-------------|------|----------|
| Redis 8.6.0 | Docker network | TCP (port 6379) |
| Host filesystem | Volume mount | bind mount |

---

## Constraints

- Must replace existing `Dockerfile`, not add a parallel one
- Builder and runtime stages must use the same Debian release (bookworm) for glibc compatibility
- Cargo.lock must be copied early for dependency layer caching

---

## Success Criteria

### Functional
- [ ] `docker build -t vector-db-benchmark .` succeeds
- [ ] `docker run vector-db-benchmark --help` works
- [ ] `docker run vector-db-benchmark --describe datasets` works
- [ ] `make docker-integration` runs h-and-m-2048-angular-filters against Redis and exits 0

### Non-Functional
- [ ] Final image < 150MB
- [ ] Cargo dependency layer is cached (rebuild on source-only change doesn't recompile deps)

---

## Bolt Suggestions

| Bolt | Type | Stories | Objective |
|------|------|---------|-----------|
| 005-docker-packaging | simple-construction-bolt | 001, 002, 003 | All stories in single bolt — tightly coupled, small scope |

---

## Notes

- Existing `Dockerfile` builds Python v0 with Poetry/maturin — completely different from what we need
- The `h-and-m-2048-angular-filters` dataset is 105K vectors, 2048d, cosine distance, tar format
- Use `redis-default-simple` config from `experiments/configurations/redis-single-node.json` for integration test
- Reference existing `tests/docker-compose.test.yml` for Redis service patterns
