---
stage: test
bolt: 005-docker-packaging
created: 2026-02-27T15:35:00Z
---

## Test Report: docker-packaging

### Summary

- **Tests**: 5/5 passed
- **Coverage**: All acceptance criteria verified

### Test Results

- [x] `docker build -t vector-db-benchmark .` — succeeds (all layers cached on rebuild)
- [x] `docker run --rm vector-db-benchmark --help` — prints help text, exits 0
- [x] `docker run --rm vector-db-benchmark --describe datasets` — lists 48 datasets, exits 0
- [x] `make docker-integration` — full end-to-end pass (configure, upload 105K vectors, search with precision 0.9326)
- [x] `make check` — passes (no regressions)

### Acceptance Criteria Validation

- ✅ **Docker image builds**: Succeeds with multi-stage Rust build
- ✅ **--help works**: Prints clap-generated help text
- ✅ **--describe datasets works**: Lists all 48 datasets from bundled datasets.json
- ✅ **Image size < 150MB**: 101MB
- ✅ **docker-integration passes**: Configure + upload + search against Redis in Docker
- ✅ **Container cleanup**: Compose down runs on success and failure
- ✅ **make check passes**: No Rust regressions

### Issues Found and Fixed

1. **Cargo dependency cache invalidation** — Dummy source build produced an empty binary. Cargo didn't recompile when real source was copied because fingerprints weren't invalidated. Fixed by explicitly removing `target/release/vector-db-benchmark` and fingerprint files between dummy and real builds.

2. **datasets.json path** — Binary looks for `v0/datasets/datasets.json` (not `datasets/datasets.json`). Fixed Dockerfile COPY path to `/code/v0/datasets/datasets.json`.

3. **100-thread upload connection reset** — 100 parallel upload threads overwhelmed the Docker-networked Redis container. Created dedicated `redis-docker-test.json` config with 16 parallel threads and single-threaded search for stable Docker testing.

### Notes

- First run downloads h-and-m dataset (~200MB) inside the container — subsequent runs reuse the host-mounted volume
- Upload of 105K 2048d vectors takes ~105s with 16 threads in Docker
- Search precision 0.9326 at ef=128 (single-threaded)
