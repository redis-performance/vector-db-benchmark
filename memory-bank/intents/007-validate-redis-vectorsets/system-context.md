---
intent: 007-validate-redis-vectorsets
phase: inception
status: context-defined
updated: 2026-02-27T00:00:00Z
---

# Validate Redis & VectorSets - System Context

## System Overview

Validation intent — no new engine code. Extends test coverage and v0-check comparisons to confirm the existing Rust Redis and VectorSets engine implementations are correct.

## Context Diagram

```mermaid
C4Context
    title System Context - Validation

    Person(dev, "Developer", "Runs validation suite")
    System(tests, "Integration Tests", "Rust cargo test suite")
    System(v0check, "v0-check Script", "Precision comparison tool")
    System_Ext(redis, "Redis 8.6.0", "Docker test instance on port 6399")
    System_Ext(pyv0, "Python v0", "Reference implementation")

    Rel(dev, tests, "make integration-test")
    Rel(dev, v0check, "make v0-check")
    Rel(tests, redis, "FT.*, VADD, VSIM commands")
    Rel(v0check, redis, "Runs both Py and Rust benchmarks")
    Rel(v0check, pyv0, "Runs Python v0 benchmarks")
```

## External Integrations

- **Redis 8.6.0**: Docker test container (port 6399) for integration tests
- **Python v0**: Reference implementation for precision comparison via v0-check

## High-Level Constraints

- No changes to engine implementation code
- Must use existing Docker test infrastructure
- v0-check requires Python v0 environment
