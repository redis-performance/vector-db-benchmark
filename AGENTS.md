# Agent guidelines

Instructions for AI coding agents (Claude Code, Copilot, Cursor, etc.) working in this repo.

## Project overview

`vector-db-benchmark` is a benchmarking framework for vector databases. It measures upload throughput and query performance (QPS, P50/P95 latency, recall/precision) across multiple engines — Redis (RediSearch and Vector Sets), Qdrant, Weaviate, Milvus, OpenSearch, Elasticsearch, PostgreSQL with pgvector, and others. Benchmarks are driven by `run.py` and configured through JSON files in `experiments/configurations/`. Engine client implementations live in `engine/clients/`, datasets are declared in `datasets/datasets.json`, and results are written to `results/`.

## Local setup

```bash
git clone git@github.com:redis-performance/vector-db-benchmark.git
cd vector-db-benchmark

# Install Poetry (if not already installed)
pip install poetry

# Install all dependencies
poetry install
```

Python 3.9–3.13 is supported. Poetry manages the virtualenv; no manual venv is needed.

## Branch naming

Same as human contributors: `<type>/<short-description>` (e.g. `fix/off-by-one-in-pipeline`).

## Coding standards

- Match the style already in the file you are editing.
- Prefer clear, minimal changes over large refactors unless explicitly asked.
- Do not add comments that describe *what* the code does — only add comments when the *why* is non-obvious.
- Do not introduce new dependencies without checking with the maintainer.
- Code is formatted with Black and imports sorted with isort (enforced by pre-commit hooks).

## Running tests

Start a local Redis instance, then run the smoke test and unit tests:

```bash
# Start Redis
docker run -d --name redis-test -p 6379:6379 redis:8.2-bookworm

# Smoke test — runs a full upload+search cycle against the synthetic dataset
poetry run python run.py --host localhost --engines redis-default-simple --datasets random-100 --queries 10

# Unit tests
poetry run pytest tests/

# Tear down
docker stop redis-test && docker rm redis-test
rm -rf results/ datasets/random-100/
```

Always run tests before declaring a task complete.

## How to submit changes

1. Create a branch: `git checkout -b <type>/<description>`.
2. Commit with a clear message focused on *why*, not *what*.
3. Open a pull request against `update.redisearch` (the default branch).
4. Do **not** push directly to `update.redisearch`.

## What to avoid

- Do not reformat files unrelated to your change.
- Do not remove error handling or tests.
- Do not commit secrets, credentials, or large binary files.
- Do not amend published commits.
- Do not add or change engine configurations in `experiments/configurations/` without a corresponding test or justification — these configurations directly affect published benchmark results.
- Do not change dataset registration in `datasets/datasets.json` without verifying the dataset URL and ground-truth format.
