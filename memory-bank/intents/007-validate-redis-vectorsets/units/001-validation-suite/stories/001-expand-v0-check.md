---
id: 001-expand-v0-check
unit: 001-validation-suite
intent: 007-validate-redis-vectorsets
status: complete
priority: must
created: 2026-02-27T00:00:00.000Z
assigned_bolt: 003-validation-suite
implemented: true
---

# Story: 001-expand-v0-check

## User Story

**As a** developer
**I want** v0-check to run across multiple datasets and engine configs
**So that** I can confirm precision parity is not limited to a single test case

## Acceptance Criteria

- [ ] **Given** `v0_check.sh` with no args, **When** run, **Then** it tests at least 3 dataset/engine combinations
- [ ] **Given** `v0_check.sh ENGINE DATASET`, **When** run with specific args, **Then** it tests only that combination
- [ ] **Given** all combinations, **When** v0-check completes, **Then** precision matches Python v0 for each
- [ ] **Given** a combination fails, **When** v0-check reports, **Then** it shows which specific combination failed

## Technical Notes

- Add a "run all" mode to v0_check.sh that iterates over dataset/engine pairs
- Candidate datasets: glove-25-angular (HDF5), random-100k (JSONL), h-and-m-2048-angular-filters (compound/TAR)
- Candidate configs: redis-m-16-ef-128, redis-m-32-ef-128
- Keep single-combo mode for quick dev iteration

## Dependencies

### Requires
- None

### Enables
- 002-v0-check-vectorsets
