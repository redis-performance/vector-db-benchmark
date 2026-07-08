---
intent: 008-docker-build
phase: inception
status: complete
created: 2026-02-27T15:00:00Z
updated: 2026-02-27T15:10:00Z
---

# Requirements: Docker Build

## Intent Overview

Build a proper Docker container for the Rust version of vector-db-benchmark and add a `make docker-integration` target that runs the containerized tool against Redis on the h-and-m-2048-angular-filters dataset.

## Business Goals

| Goal | Success Metric | Priority |
|------|----------------|----------|
| Containerized Rust binary | `docker run vector-db-benchmark --help` works | Must |
| Docker-based integration test | `make docker-integration` passes end-to-end | Must |
| Small image footprint | Final image < 150MB | Should |

---

## Functional Requirements

### FR-1: Rust Multi-Stage Dockerfile
- **Description**: Replace the existing Python-based `Dockerfile` with a new multi-stage Dockerfile that compiles the Rust binary (with HDF5 support) in a builder stage, then copies only the binary and runtime dependencies to a slim final image.
- **Acceptance Criteria**:
  - Builder stage compiles `vector-db-benchmark` binary in release mode with `libhdf5-dev`
  - Final image is based on a slim base (e.g., `debian:bookworm-slim`) with only `libhdf5` runtime lib
  - `vector-db-benchmark --help` runs successfully in the container
  - `vector-db-benchmark --describe datasets` works
- **Priority**: Must
- **Related Stories**: TBD

### FR-2: Volume Mounts for Datasets, Experiments, and Results
- **Description**: The container must support host-mounted volumes for `datasets/`, `experiments/`, and `results/` directories so datasets are not baked into the image.
- **Acceptance Criteria**:
  - Datasets can be mounted at `/code/datasets`
  - Experiment configs can be mounted at `/code/experiments`
  - Results are written to `/code/results` and visible on host
  - Default `ENTRYPOINT` is the `vector-db-benchmark` binary
- **Priority**: Must
- **Related Stories**: TBD

### FR-3: Docker-Compose Integration Test Service
- **Description**: Add a docker-compose file that defines the benchmark container and a Redis service, networked together so the benchmark can reach Redis by hostname.
- **Acceptance Criteria**:
  - Redis 8.6.0 service on default port
  - Benchmark container on same Docker network as Redis
  - Benchmark can connect to Redis via `--host redis` (docker service name)
- **Priority**: Must
- **Related Stories**: TBD

### FR-4: `make docker-integration` Target
- **Description**: Add a Makefile target that builds the Docker image, starts Redis + benchmark via docker-compose, runs `h-and-m-2048-angular-filters` dataset with a single redis config, and reports pass/fail.
- **Acceptance Criteria**:
  - `make docker-integration` builds the image, downloads the dataset if missing, starts services, runs benchmark, stops services
  - Uses `h-and-m-2048-angular-filters` dataset with a simple redis config
  - Exits with code 0 on success, non-zero on failure
  - Cleans up containers after run (success or failure)
- **Priority**: Must
- **Related Stories**: TBD

### FR-5: `make docker-build` Target
- **Description**: Add a standalone `make docker-build` target that builds the Docker image with appropriate tag.
- **Acceptance Criteria**:
  - Builds image tagged as `vector-db-benchmark:latest`
  - Supports optional `IMAGE_TAG` variable override
- **Priority**: Should
- **Related Stories**: TBD

---

## Non-Functional Requirements

### Performance
| Requirement | Metric | Target |
|-------------|--------|--------|
| Image size | Final image size | < 150MB |
| Build cache | Layer cache efficiency | Cargo dependency layer cached separately from source |

### Reliability
| Requirement | Metric | Target |
|-------------|--------|--------|
| Cleanup | Container cleanup on failure | Always stops containers via trap/finally |

---

## Constraints

### Technical Constraints

**Project-wide standards**: Required standards will be loaded from memory-bank standards folder by Construction Agent.

**Intent-specific constraints**:
- Must replace existing `Dockerfile` (not a separate file)
- HDF5 dev headers needed at build time, HDF5 shared lib needed at runtime
- Dataset `h-and-m-2048-angular-filters` is a `.tgz` (105K vectors, 2048d, cosine) — auto-downloaded if not present

### Business Constraints
- None

---

## Assumptions

| Assumption | Risk if Invalid | Mitigation |
|------------|-----------------|------------|
| `debian:bookworm-slim` has compatible glibc for Rust binary | Binary won't run | Use same base OS in builder and runtime stages |
| `libhdf5` runtime available via apt in slim image | HDF5 datasets won't load | Install `libhdf5-103` or equivalent runtime package |
| Docker network allows hostname-based service discovery | Benchmark can't reach Redis | Use explicit `--host` flag pointing to service name |

---

## Open Questions

| Question | Owner | Due Date | Resolution |
|----------|-------|----------|------------|
| None | - | - | - |
