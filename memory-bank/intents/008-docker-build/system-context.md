---
intent: 008-docker-build
phase: inception
status: context-defined
updated: 2026-02-27T15:10:00Z
---

# Docker Build - System Context

## System Overview

Replace the Python-era Dockerfile with a multi-stage Rust build that produces a small container image. Add docker-compose orchestration and a `make docker-integration` target that runs the benchmark tool inside Docker against a Redis container using the h-and-m-2048-angular-filters dataset.

## Context Diagram

```
┌─────────────────────────────────────────────────┐
│              Docker Compose Network              │
│                                                  │
│  ┌──────────────────┐    ┌──────────────────┐   │
│  │  vector-db-bench  │──▶│   redis:8.6.0    │   │
│  │  (Rust binary)    │   │   (port 6379)    │   │
│  └──────────────────┘    └──────────────────┘   │
│         │                                        │
│    volumes:                                      │
│    - datasets/                                   │
│    - experiments/                                │
│    - results/                                    │
└─────────────────────────────────────────────────┘
         │
    Host filesystem
    (mounted volumes)
```

## External Integrations

- **Redis 8.6.0**: Target database for benchmark (Docker service)
- **Host filesystem**: Dataset files, experiment configs, and results via volume mounts
- **Docker Hub / Registry**: Base images (rust, debian:bookworm-slim, redis)

## High-Level Constraints

- Must replace existing `Dockerfile` in repo root
- Final image must be small (multi-stage, slim base)
- HDF5 support required (build-time headers + runtime shared lib)
- Cargo dependency caching in Docker layer for fast rebuilds

## Key NFR Goals

- Image size < 150MB
- Docker layer caching for Cargo dependencies (avoid full rebuild on source changes)
- Reliable cleanup of containers on test success or failure
