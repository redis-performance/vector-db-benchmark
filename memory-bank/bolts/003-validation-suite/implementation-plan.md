---
stage: plan
bolt: 003-validation-suite
created: 2026-02-27T12:00:00Z
---

## Implementation Plan: Validation Suite (v0-check expansion)

### Objective

Expand `scripts/v0_check.sh` from a single-combination Redis-only check into a comprehensive multi-engine, multi-dataset validation tool that also covers VectorSets.

### Deliverables

1. **Updated `scripts/v0_check.sh`** — Refactored with "run all" mode, VectorSets support, per-combination reporting, and summary
2. **Updated `Makefile`** — `make v0-check` passes args through; add `make v0-check-all` for full matrix

### Dependencies

- Python v0 environment (`cd v0 && poetry install`) — already required
- Redis 8.6.0 Docker container — already used
- Datasets must be downloaded locally — same as today
- VectorSets Python config (`v0/experiments/configurations/vectorsets-NOQUANT.json`) — already exists
- VectorSets Rust config (`experiments/configurations/vectorsets-rs-NOQUANT.json`) — already exists

### Technical Approach

#### Current State

`v0_check.sh` takes `ENGINE` and `DATASET` as positional args (defaults: `redis-m-16-ef-128` / `h-and-m-2048-angular-filters`). It starts Redis, runs Python v0, runs Rust, compares precision/QPS/latency for that single combination, then tears down.

#### Changes

**1. Extract comparison into a function**

Refactor the current run+compare logic into a `run_comparison()` function that takes `ENGINE` and `DATASET` as parameters and returns pass/fail. This function handles:
- Cleaning old results for that combination
- Running Python v0
- Running Rust
- Comparing results
- Printing per-combination pass/fail

**2. Add "run all" mode**

When invoked with no args (or `--all`), iterate over a predefined matrix:

| Engine Config | Dataset | Engine Type |
|---|---|---|
| `redis-m-16-ef-128` | `h-and-m-2048-angular-filters` | redis |
| `redis-m-16-ef-128` | `glove-25-angular` | redis |
| `redis-m-16-ef-128` | `random-100k` | redis |
| `vectorsets-fp32-default` | `glove-25-angular` | vectorsets |
| `vectorsets-fp32-default` | `random-100k` | vectorsets |

Note: VectorSets does not support metadata filters, so `h-and-m-2048-angular-filters` is excluded for VectorSets.

**3. VectorSets engine support**

The key difference for VectorSets vs Redis in v0_check:
- Both use the same Redis Docker container (VectorSets is a Redis module)
- Python v0 picks up configs from `v0/experiments/configurations/vectorsets-NOQUANT.json`
- Rust picks up configs from `experiments/configurations/vectorsets-rs-NOQUANT.json`
- The config name (`vectorsets-fp32-default`) is the same in both, so `--engines` arg works for both
- No separate Docker setup needed — same Redis 8.6.0 container serves both

**4. Per-combination reporting**

- Track pass/fail per combination in arrays
- Print a summary table at the end showing all results
- Exit code: 0 if all pass, 1 if any fail

**5. Keep single-combination mode**

`./scripts/v0_check.sh redis-m-16-ef-128 h-and-m-2048-angular-filters` still works as before for quick dev iteration.

#### Script Structure (pseudocode)

```
#!/usr/bin/env bash
# ... existing setup (colors, cleanup, etc.)

run_comparison() {
    local engine="$1" dataset="$2"
    # clean old results
    # run python v0
    # run rust
    # compare results
    # return 0=pass, 1=fail
}

if [ $# -eq 0 ] || [ "$1" = "--all" ]; then
    # Run all mode
    COMBOS=(
        "redis-m-16-ef-128:h-and-m-2048-angular-filters"
        "redis-m-16-ef-128:glove-25-angular"
        "redis-m-16-ef-128:random-100k"
        "vectorsets-fp32-default:glove-25-angular"
        "vectorsets-fp32-default:random-100k"
    )
    # Start Redis once
    # Loop through combos, track pass/fail
    # Print summary
else
    # Single combination mode (existing behavior)
    run_comparison "$1" "$2"
fi
```

#### Makefile Changes

```makefile
.PHONY: v0-check
v0-check: vector-db-benchmark
	./scripts/v0_check.sh $(ARGS)

.PHONY: v0-check-all
v0-check-all: vector-db-benchmark
	./scripts/v0_check.sh --all
```

### Acceptance Criteria

- [ ] `./scripts/v0_check.sh` (no args) runs all 5 combinations from the matrix
- [ ] `./scripts/v0_check.sh redis-m-16-ef-128 h-and-m-2048-angular-filters` runs single combo (backward compat)
- [ ] `./scripts/v0_check.sh vectorsets-fp32-default glove-25-angular` runs VectorSets comparison
- [ ] Per-combination pass/fail printed with clear PASS/FAIL indicators
- [ ] Final summary shows overall result
- [ ] Exit code 0 only if all combinations pass
- [ ] `make v0-check` still works (single default)
- [ ] `make v0-check-all` runs full matrix
- [ ] `make check` passes (no Rust code changes, but verify)
