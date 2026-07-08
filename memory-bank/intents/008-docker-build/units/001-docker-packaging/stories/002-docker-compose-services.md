---
story: 002-docker-compose-services
unit: 001-docker-packaging
intent: 008-docker-build
status: complete
priority: Must
complexity: 1
uncertainty: 1
dependencies: 1
implemented: true
---

# Story: Docker-Compose Integration Services

## User Story

**As a** developer
**I want** a docker-compose file that orchestrates the benchmark container and a Redis instance on the same network
**So that** I can run end-to-end benchmarks entirely inside Docker

## Acceptance Criteria

- [ ] New file `tests/docker-compose.docker-test.yml` created
- [ ] Redis 8.6.0 service defined with healthcheck
- [ ] Benchmark service builds from repo root `Dockerfile`
- [ ] Benchmark service on same network as Redis, can reach it via hostname `redis`
- [ ] Benchmark service mounts `./datasets:/code/datasets` and `./results:/code/results`
- [ ] Benchmark command runs: `--engines 'redis-default-simple' --datasets 'h-and-m-2048-angular-filters' --host redis`
- [ ] Benchmark service depends_on Redis (healthcheck)

## Technical Notes

- Reference existing `tests/docker-compose.test.yml` for Redis service pattern
- The benchmark container needs the dataset downloaded on the host first (or auto-downloads it)
- Use `profiles` or separate compose file to avoid mixing with unit test infrastructure
