---
stage: plan
bolt: 005-docker-packaging
created: 2026-02-27T15:15:00Z
---

## Implementation Plan: docker-packaging

### Objective

Replace the Python-era Dockerfile with a multi-stage Rust build, add a docker-compose file for benchmark+Redis orchestration, and create Makefile targets for building and running Docker-based integration tests.

### Deliverables

1. `Dockerfile` — replaced with multi-stage Rust build
2. `tests/docker-compose.docker-test.yml` — new compose file for Docker integration
3. `Makefile` — new `docker-build` and `docker-integration` targets + help updates

### Dependencies

- `rust:bookworm` base image (builder)
- `debian:bookworm-slim` base image (runtime)
- `libhdf5-dev` apt package (builder) / `libhdf5-103-1` (runtime)
- `redis:8.6.0` Docker image

### Technical Approach

#### 1. Dockerfile (replaces existing)

**Builder stage** (`rust:bookworm`):
- Install `libhdf5-dev` and `pkg-config` via apt
- Set `HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial`
- Dependency caching trick: copy `Cargo.toml` + `Cargo.lock`, create dummy `src/` structure, `cargo build --release`, then remove dummy and copy real source
- `cargo build --release --bin vector-db-benchmark`

**Runtime stage** (`debian:bookworm-slim`):
- Install `libhdf5-103-1` (HDF5 runtime shared lib)
- Copy binary from builder to `/usr/local/bin/vector-db-benchmark`
- Copy `datasets/datasets.json` to `/code/datasets/datasets.json`
- Copy `experiments/configurations/` to `/code/experiments/configurations/`
- Create `/code/datasets`, `/code/experiments`, `/code/results` directories
- `WORKDIR /code`
- `ENTRYPOINT ["vector-db-benchmark"]`
- `CMD ["--help"]`

**Layer cache strategy**:
```
COPY Cargo.toml Cargo.lock ./
# Create dummy lib.rs and bin stubs for dependency compilation
RUN mkdir src && echo "" > src/lib.rs && mkdir -p src/bin && ...
RUN cargo build --release
# Remove dummy, copy real source
RUN rm -rf src
COPY src ./src
RUN cargo build --release --bin vector-db-benchmark
```

#### 2. docker-compose.docker-test.yml

```yaml
services:
  redis:
    image: redis:8.6.0
    healthcheck: (same as existing test compose)

  benchmark:
    build:
      context: ..
      dockerfile: Dockerfile
    depends_on:
      redis:
        condition: service_healthy
    volumes:
      - ../datasets:/code/datasets
      - ../results:/code/results
    command: ["--engines", "redis-default-simple", "--datasets", "h-and-m-2048-angular-filters", "--host", "redis"]
```

#### 3. Makefile targets

```makefile
IMAGE_TAG ?= latest

docker-build:
    docker build -t vector-db-benchmark:$(IMAGE_TAG) .

docker-integration: docker-build
    # Start Redis + run benchmark via compose
    docker compose -f tests/docker-compose.docker-test.yml up -d redis --wait
    docker compose -f tests/docker-compose.docker-test.yml run --rm benchmark; \
    EXIT_CODE=$$?; \
    docker compose -f tests/docker-compose.docker-test.yml down; \
    exit $$EXIT_CODE
```

### Acceptance Criteria

- [ ] `docker build -t vector-db-benchmark .` succeeds
- [ ] `docker run --rm vector-db-benchmark --help` works
- [ ] `docker run --rm vector-db-benchmark --describe datasets` works
- [ ] Final image < 150MB
- [ ] `make docker-integration` builds image, starts Redis, runs h-and-m-2048-angular-filters benchmark, cleans up
- [ ] `make docker-integration` exits 0 on success, non-zero on failure
- [ ] `make check` still passes (no Rust regressions)
- [ ] Help target lists new targets

### Risks and Mitigations

- **HDF5 library not found at runtime**: Run `ldconfig` after installing `libhdf5-103-1` to update linker cache
- **Dependency caching breaks on Cargo.toml change**: Expected — only avoids recompiling deps when only `src/` changes
- **Dataset not downloaded**: The Rust binary auto-downloads datasets, so first run may be slow but will work
