# Intent 010: Add Valkey Engine

**Status**: complete
**Created**: 2026-03-01

## Summary
Add Valkey (Redis fork) as a supported vector database engine using the `redis` crate, since valkey-glide has no published Rust client (GitHub issue #828 closed as NOT_PLANNED). Valkey speaks the same RESP protocol and supports identical FT.* search commands via Valkey Search module.

## Functional Requirements

- **FR-1**: ValkeyEngine implements Engine trait (configure, upload, search, delete, get_memory_usage)
- **FR-2**: Uses VALKEY_PORT, VALKEY_AUTH, VALKEY_USER, VALKEY_HYBRID_POLICY, VALKEY_QUERY_TIMEOUT env vars
- **FR-3**: Supports HNSW index creation via FT.CREATE with configurable M, EF_CONSTRUCTION
- **FR-4**: Parallel upload via HSET pipeline batching (same as Redis engine)
- **FR-5**: KNN search via FT.SEARCH with prefilter support (match, range, geo)
- **FR-6**: Metadata filter parsing identical to Redis engine
- **FR-7**: Wait for indexing via FT.INFO polling
- **FR-8**: Engine configs in experiments/configurations/valkey-*.json
- **FR-9**: Integration tests using valkey/valkey-bundle Docker image
- **FR-10**: CI job, Makefile target, docker-compose service
- **FR-11**: README and DOCKER_README updated

## Non-Functional Requirements

- **NFR-1**: Uses `redis` crate 0.27 (already a dependency) — Valkey is RESP-compatible
- **NFR-2**: No new Cargo dependencies needed
- **NFR-3**: valkey-glide Rust client does not exist; redis crate is maintainer-recommended alternative

## Technical Decisions

- **Client**: `redis` crate (not valkey-glide — no Rust client available)
- **Docker**: `valkey/valkey-bundle:latest` (includes Valkey Search module for FT.* commands)
- **Test port**: 6380 (host) → 6379 (container)
- **Protocol**: Identical to Redis — same FT.CREATE/FT.SEARCH/FT.DROPINDEX/FT.INFO commands
