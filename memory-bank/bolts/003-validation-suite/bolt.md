---
id: 003-validation-suite
unit: 001-validation-suite
intent: 007-validate-redis-vectorsets
type: simple-construction-bolt
status: complete
stories:
  - 001-expand-v0-check
  - 002-v0-check-vectorsets
created: 2026-02-27T00:00:00.000Z
started: 2026-02-27T12:00:00.000Z
completed: "2026-02-27T11:28:42Z"
current_stage: null
stages_completed:
  - name: plan
    completed: 2026-02-27T12:00:00.000Z
    artifact: implementation-plan.md
  - name: implement
    completed: 2026-02-27T12:30:00.000Z
    artifact: implementation-walkthrough.md
  - name: test
    completed: 2026-02-27T14:00:00.000Z
    artifact: test-walkthrough.md
requires_bolts: []
enables_bolts:
  - 004-validation-suite
requires_units: []
blocks: false
complexity:
  avg_complexity: 1
  avg_uncertainty: 1
  max_dependencies: 1
  testing_scope: 3
---

# Bolt: 003-validation-suite

## Overview

Expand v0-check script to cover multiple datasets and add VectorSets engine comparison.

## Objective

Make v0_check.sh a comprehensive validation tool: multi-dataset "run all" mode, VectorSets support, clear per-combination pass/fail reporting.

## Stories Included

- **001-expand-v0-check**: Multi-dataset v0-check with "run all" mode (Must)
- **002-v0-check-vectorsets**: Add VectorSets to v0-check comparisons (Must)

## Bolt Type

**Type**: Simple Construction Bolt
**Definition**: `.specsmd/aidlc/templates/construction/bolt-types/simple-construction-bolt.md`

## Stages

- [ ] **1. Plan**: Define dataset/config matrix, script changes
- [ ] **2. Implement**: Update v0_check.sh with multi-dataset and VectorSets support
- [ ] **3. Test**: Run v0-check against all combinations, verify pass

## Dependencies

### Requires
- None

### Enables
- 004-validation-suite (integration tests)

## Success Criteria

- [ ] v0-check runs against 3+ dataset/engine combinations
- [ ] VectorSets precision matches Python v0
- [ ] Clear pass/fail per combination
