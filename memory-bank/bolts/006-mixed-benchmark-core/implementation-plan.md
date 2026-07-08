---
stage: plan
bolt: 006-mixed-benchmark-core
created: 2026-03-05T10:30:00Z
---

## Implementation Plan: Mixed Benchmark Core

### Objective

Add `--update-search-ratio U:S` CLI flag to run mixed workload benchmarks that interleave vector updates with searches. Measure both operations independently. Initial engines: Redis, VectorSets, Valkey.

### Deliverables

- Modified `cli.rs` — new `--update-search-ratio` flag
- Modified `engine/mod.rs` — `update()` trait method + `SearchResults` update fields
- Modified `experiment.rs` — interleaved search/update loop + extended results JSON
- Modified `engine/redis.rs` — `update()` implementation (single HSET)
- Modified `engine/vectorsets.rs` — `update()` implementation (single VADD + SETATTR)
- Modified `engine/valkey.rs` — `update()` implementation (single HSET)

### Dependencies

- `rand` 0.8 — already in Cargo.toml (for `SeedableRng`, `SliceRandom`)
- No new crate dependencies needed

### Technical Approach

#### 1. CLI flag (`cli.rs`)

Add `--update-search-ratio` as `Option<String>`. Parse `"U:S"` format into a struct:

```rust
pub struct UpdateSearchRatio {
    pub updates: u64,
    pub searches: u64,
}
```

Parse in `Args` or in experiment.rs before the search loop. Validate: both positive, searches > 0.

#### 2. Engine trait (`engine/mod.rs`)

Add default method to `Engine`:

```rust
fn update(&self, conn: &mut redis::Connection, id: i64, vector: &[f32], metadata: Option<&MetadataItem>) -> Result<(), String> {
    Err(format!("mixed benchmark not supported for {}", self.name()))
}
```

**Problem**: The trait is engine-agnostic — it shouldn't depend on `redis::Connection`. Instead, use a different approach: each engine's `search()` method already creates per-thread connections. The update logic will be a free function inside each engine module (like `ft_search_knn` and `vsim_search`), called from within the interleaved loop.

**Revised approach**: Add to Engine trait:

```rust
fn supports_update(&self) -> bool { false }
```

The actual update call happens inside each engine's search method (or a new `mixed_search` method). But this couples the interleaving logic into each engine.

**Simplest approach**: Add `update_single()` free functions in each engine module. The interleaving loop lives in `experiment.rs` and calls `engine.search()` with an extra parameter, OR we refactor the search loop out of individual engines.

**Final decision**: The interleaved loop will live in `experiment.rs`. We add a new Engine trait method:

```rust
fn search_mixed(
    &mut self,
    dataset: &Dataset,
    search_params: &SearchParams,
    num_queries: i64,
    ratio: &UpdateSearchRatio,
) -> Result<SearchResults, String> {
    Err(format!("mixed benchmark not supported for {}", self.name()))
}
```

Each engine that supports mixed mode overrides `search_mixed()`. The method is similar to `search()` but with an interleaved loop that alternates between search and update operations.

Extend `SearchResults` with optional update fields:

```rust
pub update_count: Option<usize>,
pub update_rps: Option<f64>,
pub update_mean_time: Option<f64>,
pub update_p50_time: Option<f64>,
pub update_p95_time: Option<f64>,
pub update_p99_time: Option<f64>,
pub update_latencies: Option<Vec<f64>>,
pub update_search_ratio: Option<String>,
```

#### 3. Interleaved loop (inside each engine's `search_mixed`)

The worker thread loop pattern per engine:

```
Pre-compute: shuffled update sequence via SeededRng(42)
  → Vec<usize> of indices into (ids, vectors, metadata) arrays

Per worker thread:
  loop {
    // Search phase: do S searches
    for _ in 0..ratio.searches {
      idx = search_counter.fetch_add(1)
      if idx >= num_queries: break outer
      search(query[idx]) → push to search_times
    }
    // Update phase: do U updates
    for _ in 0..ratio.updates {
      uidx = update_counter.fetch_add(1)
      update(vectors[update_seq[uidx % len]]) → push to update_times
    }
  }
```

Two atomic counters: `search_idx` (stops at num_queries) and `update_idx` (wraps via modulo).

#### 4. Results JSON extension (`experiment.rs`)

In `save_search_results`, conditionally add update fields:

```json
"results": {
  ...existing search fields...,
  "update_search_ratio": "1:10",
  "update_count": 1000,
  "update_rps": 5432.1,
  "update_mean_time": 0.000184,
  "update_p50_time": 0.000172,
  "update_p95_time": 0.000312,
  "update_p99_time": 0.000456,
  "update_latencies": [...]
}
```

#### 5. Engine implementations

**Redis** (`redis.rs`): Single HSET with vector blob + metadata fields. Key = `id.to_string()`. Same vector encoding as upload (FLOAT32/FLOAT16/BFLOAT16 based on config).

**VectorSets** (`vectorsets.rs`): Single VADD with FP32 bytes + SETATTR JSON. Element name = `id.to_string()`. Same as upload path but for one vector.

**Valkey** (`valkey.rs`): Same as Redis — single HSET. Nearly identical code.

#### 6. experiment.rs changes

In `run_single_experiment`, after the search loop:
- Parse `args.update_search_ratio`
- If set, call `engine.search_mixed()` instead of `engine.search()`
- Pass the update data (ids, vectors, metadata) from the dataset

The dataset vectors need to be read and kept available for both search queries and updates. Currently `search()` reads queries internally. For mixed mode, `search_mixed()` will also need access to the upload vectors.

**Key insight**: `search_mixed()` needs two dataset reads — queries (for search) and vectors (for updates). The vectors are already read during upload, but not stored. Solution: `search_mixed()` reads both queries and vectors from the dataset.

### Implementation Order

1. `engine/mod.rs` — `SearchResults` extension + `search_mixed()` default
2. `cli.rs` — `UpdateSearchRatio` struct + `--update-search-ratio` flag
3. `experiment.rs` — dispatch to `search_mixed()` when ratio is set, extend JSON saving
4. `engine/redis.rs` — `search_mixed()` override
5. `engine/vectorsets.rs` — `search_mixed()` override
6. `engine/valkey.rs` — `search_mixed()` override
7. Build + test on random-100
8. Precision invariance on h-and-m-2048-angular

### Acceptance Criteria

- [ ] `--update-search-ratio 1:10` works on Redis, VectorSets, Valkey
- [ ] Omitting the flag preserves existing search-only behavior (no code path changes)
- [ ] Results JSON includes separate search/update metrics when ratio is set
- [ ] Precision matches search-only baseline within 0.001
- [ ] `cargo build --release` compiles cleanly
- [ ] `cargo clippy` no warnings
- [ ] Deterministic: same seed produces same update sequence
