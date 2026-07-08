---
stage: implement
bolt: 005-docker-packaging
created: 2026-02-27T15:25:00Z
---

## Implementation Walkthrough: docker-packaging

### Summary

Replaced the Python-era Dockerfile with a multi-stage Rust build, created a docker-compose file for orchestrating benchmark+Redis, and added `docker-build` and `docker-integration` Make targets.

### Structure Overview

Three files changed/created: one replaced (Dockerfile), one new (docker-compose), one modified (Makefile). The Dockerfile uses a two-stage build with Cargo dependency caching to keep rebuilds fast and the final image small.

### Completed Work

- [x] `Dockerfile` - Multi-stage Rust build: rust:bookworm builder with libhdf5-dev, debian:bookworm-slim runtime with libhdf5-103-1, dependency caching via dummy source trick
- [x] `tests/docker-compose.docker-test.yml` - Docker Compose with Redis 8.6.0 + benchmark service, shared network, volume mounts for datasets and results
- [x] `Makefile` - Added `docker-build` and `docker-integration` targets with IMAGE_TAG variable, exit code capture, and container cleanup

### Key Decisions

- **Dependency caching via dummy sources**: Copies Cargo.toml/Cargo.lock first, creates minimal dummy source files to compile dependencies, then replaces with real source. This means only source changes trigger a quick incremental build, not a full dependency recompile.
- **`docker compose run --rm` instead of `up`**: The benchmark service runs as a one-shot command rather than a long-running service. `run --rm` executes the command and returns the exit code, which is exactly what we need for pass/fail testing.
- **Separate compose file**: `docker-compose.docker-test.yml` is separate from `docker-compose.test.yml` to avoid mixing Docker-image-based tests with cargo-based integration tests.
- **Image reference by tag**: The compose file uses `image: vector-db-benchmark:${IMAGE_TAG:-latest}` rather than `build:` so `docker-build` is a separate explicit step.

### Deviations from Plan

None.

### Dependencies Added

- No new Rust crate dependencies
- Docker base images: `rust:bookworm`, `debian:bookworm-slim`, `redis:8.6.0`
- Apt packages: `libhdf5-dev` (build), `libhdf5-103-1` + `ca-certificates` (runtime)

### Developer Notes

- First `make docker-integration` run will be slow because the dataset auto-downloads inside the container (~200MB)
- Subsequent runs reuse the downloaded dataset from the host volume mount
- `ca-certificates` is needed in runtime image for HTTPS dataset downloads
