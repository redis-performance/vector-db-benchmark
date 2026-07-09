# Global Story Index

## Overview
- **Total stories**: 24
- **Generated**: 24
- **Last updated**: 2026-03-05

---

## Stories by Intent

### 001-migrate-elasticsearch

#### Unit: 001-elasticsearch-engine

- [x] **001-connection-and-config** ✅ GENERATED — Connection management and config parsing — Must — Bolt: 001
- [x] **002-index-configuration** ✅ GENERATED — Dense vector index creation with HNSW — Must — Bolt: 001
- [x] **003-bulk-upload** ✅ GENERATED — Parallel bulk upload with UUID conversion — Must — Bolt: 001
- [x] **004-force-merge** ✅ GENERATED — Post-upload force merge with retry — Must — Bolt: 001
- [x] **005-knn-search** ✅ GENERATED — KNN search with parallel execution — Must — Bolt: 002
- [x] **006-filter-parsing** ✅ GENERATED — Metadata filter to bool query conversion — Must — Bolt: 002
- [x] **007-integration-tests** ✅ GENERATED — Docker-based integration tests — Must — Bolt: 002

### 007-validate-redis-vectorsets

#### Unit: 001-validation-suite

- [x] **001-expand-v0-check** ✅ GENERATED — Multi-dataset v0-check coverage — Must — Bolt: 003
- [x] **002-v0-check-vectorsets** ✅ GENERATED — VectorSets v0-check support — Must — Bolt: 003
- [x] **003-redis-filter-tests** ✅ GENERATED — Redis metadata filter integration tests — Must — Bolt: 004
- [x] **004-distance-metric-tests** ✅ GENERATED — L2 and COSINE distance metric tests — Must — Bolt: 004
- [x] **005-parallel-execution-tests** ✅ GENERATED — Parallel upload/search correctness tests — Should — Bolt: 004
- [x] **006-vectorsets-precision** ✅ GENERATED — VectorSets precision integration test — Must — Bolt: 004

### 008-docker-build

#### Unit: 001-docker-packaging

- [x] **001-multi-stage-dockerfile** GENERATED — Multi-stage Rust Dockerfile with HDF5 and dependency caching — Must — Bolt: 005
- [x] **002-docker-compose-services** GENERATED — Docker-compose benchmark + Redis services — Must — Bolt: 005
- [x] **003-make-targets** GENERATED — Make targets for docker-build and docker-integration — Should — Bolt: 005

### 011-mixed-benchmark

#### Unit: 001-mixed-benchmark-core

- [ ] **001-cli-flag** — CLI flag `--update-search-ratio U:S` parsing — Must — Bolt: 006
- [ ] **002-engine-trait** — Engine trait `update()` method + SearchResults extension — Must — Bolt: 006
- [ ] **003-interleaved-loop** — Interleaved worker loop (S searches, U updates per cycle) — Must — Bolt: 006
- [ ] **004-metrics-reporting** — Separate search/update metrics in results JSON — Must — Bolt: 006
- [ ] **005-redis-update** — Redis engine `update()` via HSET — Must — Bolt: 006
- [ ] **006-vectorsets-update** — VectorSets engine `update()` via VADD + SETATTR — Must — Bolt: 006
- [ ] **007-valkey-update** — Valkey engine `update()` via HSET — Must — Bolt: 006
- [ ] **008-precision-invariance** — Validate precision matches search-only baseline — Must — Bolt: 006

---

## Stories by Status

- **Generated**: 24
- **In Progress**: 0
- **Completed**: 0
