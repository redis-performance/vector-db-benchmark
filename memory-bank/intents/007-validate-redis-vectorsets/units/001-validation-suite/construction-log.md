---
unit: 001-validation-suite
intent: 007-validate-redis-vectorsets
phase: construction
created: 2026-02-27T12:00:00Z
updated: 2026-02-27T12:00:00Z
---

# Construction Log: Validation Suite

## Active Bolt

None — all bolts complete.

## Bolt History

| Bolt | Status | Started | Completed |
|------|--------|---------|-----------|
| 003-validation-suite | complete | 2026-02-27 | 2026-02-27 |
| 004-validation-suite | complete | 2026-02-27 | 2026-02-27 |

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-27 | Extract comparison into function, add --all mode | Keeps backward compat while enabling multi-combo |
| 2026-02-27 | 5-combo matrix: 3 Redis + 2 VectorSets | Covers HDF5/JSONL/compound formats, both engines |
| 2026-02-27 | VectorSets excluded from h-and-m-filters dataset | VectorSets has no metadata filter support |
| 2026-02-27 | Subprocess testing for binary path coverage | Engine trait in binary crate, can't import from tests |
| 2026-02-27 | FLUSHALL + FT.DROPINDEX in flush_db() | Redis 8 FLUSHALL doesn't drop FT indexes |
| 2026-02-27 | extract_ft_info_value handles RESP2+RESP3 | Redis crate may return Map or Array for FT.INFO |
