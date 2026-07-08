---
story: 003-make-targets
unit: 001-docker-packaging
intent: 008-docker-build
status: complete
priority: Should
complexity: 1
uncertainty: 1
dependencies: 2
implemented: true
---

# Story: Make Targets for Docker Build and Integration

## User Story

**As a** developer
**I want** `make docker-build` and `make docker-integration` targets
**So that** I can build the Docker image and run Docker-based integration tests with a single command

## Acceptance Criteria

- [ ] `make docker-build` builds image tagged `vector-db-benchmark:latest`
- [ ] `make docker-build` supports `IMAGE_TAG` variable override
- [ ] `make docker-integration` depends on `docker-build`
- [ ] `make docker-integration` ensures dataset `h-and-m-2048-angular-filters` is downloaded (runs benchmark with auto-download or pre-downloads)
- [ ] `make docker-integration` starts Redis + benchmark via docker-compose
- [ ] `make docker-integration` exits 0 on success, non-zero on failure
- [ ] `make docker-integration` cleans up containers on success and failure (trap or compose down in finally block)
- [ ] Help target updated to list new targets

## Technical Notes

- Follow existing Makefile pattern (integration-test target uses docker compose up/down with exit code capture)
- Dataset auto-download: the Rust binary downloads datasets automatically when missing, so the container just needs the volume mount
- `IMAGE_TAG ?= latest` as Make variable default
