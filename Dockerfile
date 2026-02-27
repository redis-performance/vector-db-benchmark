# Multi-stage Dockerfile for vector-db-benchmark (Rust)
#
# Build: docker build -t vector-db-benchmark .
# Run:   docker run --rm vector-db-benchmark --help

# ============================================================
# Stage 1: Build the Rust binary
# ============================================================
FROM rust:bookworm AS builder

# Install HDF5 development libraries and pkg-config
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

ENV HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial

WORKDIR /build

# --- Dependency caching layer ---
# Copy only Cargo manifests and lock file first, then build dependencies
# with dummy source files. This layer is cached unless Cargo.toml/lock change.
COPY Cargo.toml Cargo.lock ./

# Create dummy source structure matching all targets
RUN mkdir -p src/bin/vector_db_benchmark/engine src/readers src/redisearch src/vectorsets \
    && echo "fn main() {}" > src/bin/bench_hdf5.rs \
    && echo "fn main() {}" > src/bin/bench_jsonl.rs \
    && echo "fn main() {}" > src/bin/bench_npy.rs \
    && echo "fn main() {}" > src/bin/vector_db_benchmark/main.rs \
    && touch src/lib.rs \
    && touch src/bin/vector_db_benchmark/engine/mod.rs

# Build dependencies only (this layer is cached)
RUN cargo build --release --bin vector-db-benchmark 2>/dev/null || true
# Remove dummy source and binary fingerprints, keep compiled dependencies
RUN rm -rf src target/release/vector-db-benchmark target/release/.fingerprint/vector_db_benchmark-*

# --- Source build layer ---
# Copy real source code
COPY src ./src

# Build the actual binary (dependencies are cached, only project code recompiles)
RUN cargo build --release --bin vector-db-benchmark

# ============================================================
# Stage 2: Slim runtime image
# ============================================================
FROM debian:bookworm-slim

# Install HDF5 runtime library
RUN apt-get update && apt-get install -y \
    libhdf5-103-1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

WORKDIR /code

# Copy the binary from builder
COPY --from=builder /build/target/release/vector-db-benchmark /usr/local/bin/vector-db-benchmark

# Copy dataset definitions and experiment configurations into the image.
# project_root() searches for datasets/datasets.json as its marker.
COPY datasets/datasets.json /code/datasets/datasets.json
COPY experiments/configurations /code/experiments/configurations

# Create mount point directories.
# Downloaded datasets go to project_root/datasets/ (checked first by get_path).
RUN mkdir -p /code/datasets /code/results

ENTRYPOINT ["vector-db-benchmark"]
CMD ["--help"]
