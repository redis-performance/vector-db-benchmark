---
story: 001-multi-stage-dockerfile
unit: 001-docker-packaging
intent: 008-docker-build
status: complete
priority: Must
complexity: 2
uncertainty: 1
dependencies: 0
implemented: true
---

# Story: Multi-Stage Rust Dockerfile

## User Story

**As a** developer
**I want** a Dockerfile that builds the Rust binary in a builder stage and copies it to a slim runtime image
**So that** I can run vector-db-benchmark in a container with minimal image size

## Acceptance Criteria

- [ ] Builder stage uses `rust:bookworm` with `libhdf5-dev` installed
- [ ] Cargo.toml and Cargo.lock copied first for dependency layer caching
- [ ] `cargo build --release --bin vector-db-benchmark` compiles successfully
- [ ] Runtime stage uses `debian:bookworm-slim` with `libhdf5-103-1` (or equivalent)
- [ ] Binary copied from builder to runtime at `/usr/local/bin/vector-db-benchmark`
- [ ] `datasets/datasets.json` and `experiments/` bundled into image for discoverability
- [ ] `ENTRYPOINT ["vector-db-benchmark"]` set as default
- [ ] Working directories created: `/code/datasets`, `/code/experiments`, `/code/results`
- [ ] `docker run vector-db-benchmark --help` succeeds
- [ ] `docker run vector-db-benchmark --describe datasets` succeeds
- [ ] Final image size < 150MB

## Technical Notes

- Use dummy `main.rs` + real `Cargo.toml`/`Cargo.lock` trick for dependency caching
- Copy `datasets/datasets.json` to `/code/datasets/datasets.json` so `--describe datasets` works without mounts
- Copy `experiments/configurations/` so engine configs are discoverable
- HDF5 shared library path may need `LD_LIBRARY_PATH` or `ldconfig` in runtime image
