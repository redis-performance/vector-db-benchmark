# Contributing

We treat this repo as "Open Source" within Redis: anyone who clears the bar below is welcome to contribute.

## Local setup

```bash
git clone git@github.com:redis-performance/vector-db-benchmark.git
cd vector-db-benchmark

# Install Poetry (if not already installed)
pip install poetry

# Install all dependencies
poetry install

# Activate the virtual environment
poetry shell
```

Python 3.9–3.13 are supported. Poetry manages the virtualenv automatically; no manual venv creation is needed.

## Branch naming

```
<type>/<short-description>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

Example: `feat/add-pipeline-mode`

## Coding standards

- Keep changes focused; one logical change per PR.
- Follow the conventions already present in the codebase (formatting, naming, error handling).
- No dead code, no commented-out blocks.
- Code is formatted with [Black](https://black.readthedocs.io/) and imports are sorted with [isort](https://pycqa.github.io/isort/). Run `pre-commit install` after cloning so these run automatically on every commit.

## Submitting changes

1. Fork or create a branch from `update.redisearch` (the default branch).
2. Make your changes with clear, atomic commits.
3. Open a pull request against `update.redisearch` with a descriptive title and summary.
4. Address review comments promptly; force-push to the same branch to update.

## Testing

- All new behaviour must be covered by tests.
- Existing tests must pass: run the test suite locally before opening a PR.
- Coverage should not decrease.

Run the test suite (requires a local Redis instance on port 6379):

```bash
# Start Redis
docker run -d --name redis-test -p 6379:6379 redis:8.2-bookworm

# Run the benchmark against the synthetic random-100 dataset as a smoke test
poetry run python run.py --host localhost --engines redis-default-simple --datasets random-100 --queries 10

# Verify results were written
ls results/

# Tear down
docker stop redis-test && docker rm redis-test
rm -rf results/ datasets/random-100/
```

For engine-level unit tests:

```bash
poetry run pytest tests/
```

## Review process

- At least one maintainer approval is required before merge.
- CI must be green.
- Maintainers may request changes or close PRs that don't meet the bar — this is normal and not personal.
