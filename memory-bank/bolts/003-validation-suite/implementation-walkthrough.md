---
stage: implement
bolt: 003-validation-suite
created: 2026-02-27T12:30:00Z
---

## Implementation Walkthrough: Validation Suite (v0-check expansion)

### Summary

Refactored `scripts/v0_check.sh` from a single-combination Redis-only validation script into a multi-engine, multi-dataset comparison tool. Added VectorSets support, a "run all" mode with 5 predefined combinations, per-combination PASS/FAIL reporting, and a summary table. Updated `Makefile` with `v0-check-all` target.

### Structure Overview

The script was restructured around two extracted functions (`compare_results` and `run_comparison`) with a dispatch block that handles both single-combo and run-all modes. Redis is started once and FLUSHALL'd between combinations.

### Completed Work

- [x] `scripts/v0_check.sh` - Refactored with run_comparison function, --all mode, VectorSets support, per-combo reporting, summary table
- [x] `Makefile` - Added `v0-check-all` target, `v0-check` now accepts ENGINE/DATASET make vars

### Key Decisions

- **FLUSHALL between combos**: Each combination gets a clean Redis state to avoid index/data conflicts
- **Same Docker container for VectorSets**: VectorSets is a Redis module, no separate container needed
- **5-combo matrix**: 3 Redis combos (h-and-m-filters, glove-25, random-100k) + 2 VectorSets combos (glove-25, random-100k) — VectorSets excluded from h-and-m-filters because it has no metadata filter support
- **`|| true` on run_comparison**: Prevents `set -e` from aborting the loop on first failure — all combos run regardless

### Deviations from Plan

None.

### Dependencies Added

None — pure shell script changes.

### Developer Notes

- Pre-existing `make check` failure in elasticsearch.rs/integration_elasticsearch.rs formatting is from bolt 001 (unrelated)
- `make v0-check ENGINE=redis-m-16-ef-128 DATASET=glove-25-angular` works for single-combo via make vars
- Bash syntax validated with `bash -n`
