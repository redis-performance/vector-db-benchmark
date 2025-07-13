# Multi-stage Dockerfile for vector-db-benchmark
# Stage 1: Build environment
FROM python:3.10-slim AS builder

# Build arguments for Git metadata
ARG GIT_SHA
ARG GIT_DIRTY

# Environment variables for Python
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.5.1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Set working directory
WORKDIR /code

# Copy dependency files first for better caching
COPY poetry.lock pyproject.toml /code/

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Install additional dependencies
RUN pip install "boto3"

# Copy source code
COPY . /code

# Store Git information
RUN if [ -z "$GIT_SHA" ]; then \
        GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown"); \
    fi && \
    if [ -z "$GIT_DIRTY" ]; then \
        GIT_DIRTY=$(git diff --no-ext-diff 2>/dev/null | wc -l || echo "0"); \
    fi && \
    echo "Built with GIT_SHA=${GIT_SHA}, GIT_DIRTY=${GIT_DIRTY}" > /code/build_info.txt

# Stage 2: Runtime environment
FROM python:3.10-slim

# Environment variables for Python
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1001 -r appgroup && \
    useradd -u 1001 -r -g appgroup appuser

# Set working directory
WORKDIR /app

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /code /app

# Create directories with proper permissions
RUN mkdir -p /app/results /app/datasets && \
    chown -R appuser:appgroup /app && \
    chmod -R 755 /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Expose common ports (for documentation purposes)
EXPOSE 6379 6380

# Set entrypoint
ENTRYPOINT ["python"]

# Default command (show help)
CMD ["run.py", "--help"]

