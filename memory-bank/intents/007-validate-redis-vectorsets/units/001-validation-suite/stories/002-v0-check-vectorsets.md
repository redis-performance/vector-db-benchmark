---
id: 002-v0-check-vectorsets
unit: 001-validation-suite
intent: 007-validate-redis-vectorsets
status: complete
priority: must
created: 2026-02-27T00:00:00.000Z
assigned_bolt: 003-validation-suite
implemented: true
---

# Story: 002-v0-check-vectorsets

## User Story

**As a** developer
**I want** v0-check to compare VectorSets Rust vs Python v0
**So that** I can confirm the VectorSets migration is correct

## Acceptance Criteria

- [ ] **Given** v0_check.sh with a VectorSets engine config, **When** run, **Then** it executes both Python and Rust VectorSets benchmarks
- [ ] **Given** VectorSets results, **When** compared, **Then** precision matches within tolerance
- [ ] **Given** the "run all" mode, **When** run, **Then** VectorSets configs are included alongside Redis configs

## Technical Notes

- VectorSets config files: `experiments/configurations/vectorsets-rs-NOQUANT.json`
- May need to check if VectorSets Python config exists in v0/
- Score conversion (1-score) must be consistent between Rust and Python

## Dependencies

### Requires
- 001-expand-v0-check

### Enables
- None
