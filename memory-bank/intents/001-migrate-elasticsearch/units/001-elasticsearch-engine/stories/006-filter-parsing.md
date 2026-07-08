---
id: 006-filter-parsing
unit: 001-elasticsearch-engine
intent: 001-migrate-elasticsearch
status: complete
priority: must
created: 2026-02-27T00:00:00.000Z
assigned_bolt: 002-elasticsearch-engine
implemented: true
---

# Story: 006-filter-parsing

## User Story

**As a** benchmark operator
**I want** metadata filters applied during KNN search
**So that** I can benchmark filtered vector search performance

## Acceptance Criteria

- [ ] **Given** an exact match condition `{"field": {"match": {"value": "X"}}}`, **When** parsing, **Then** produces `{"match": {"field": "X"}}`
- [ ] **Given** a range condition `{"field": {"range": {"gt": 5, "lt": 10}}}`, **When** parsing, **Then** produces `{"range": {"field": {"gt": 5, "lt": 10}}}`
- [ ] **Given** a geo condition `{"field": {"geo": {"lat": 1.0, "lon": 2.0, "radius": 1000}}}`, **When** parsing, **Then** produces `{"geo_distance": {"distance": "1000m", "field": {"lat": 1.0, "lon": 2.0}}}`
- [ ] **Given** AND conditions, **When** parsing, **Then** produces `{"bool": {"must": [...]}}`
- [ ] **Given** OR conditions, **When** parsing, **Then** produces `{"bool": {"should": [...]}}`
- [ ] **Given** combined AND+OR conditions, **When** parsing, **Then** produces `{"bool": {"must": [...], "should": [...]}}`
- [ ] **Given** no meta_conditions, **When** searching, **Then** no filter is applied to KNN query

## Technical Notes

- Build filter as `serde_json::Value` objects
- Attach filter to KNN query as `knn["filter"] = conditions`
- Reference: `v0/engine/clients/elasticsearch/parser.py`
- The base parser pattern: `parse()` method splits into `and_subfilters` and `or_subfilters`, calls `build_condition()`

## Dependencies

### Requires
- 005-knn-search

### Enables
- None (final search feature)

## Edge Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| Empty meta_conditions dict | No filter applied |
| Null values in range (e.g., no "lt") | Omit null bounds from range query |
| Unknown condition type | Skip or return error |

## Out of Scope

- Nested object filters
- Script-based filters
