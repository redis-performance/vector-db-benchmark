# Docker Setup and Publishing Guide

This guide explains how to set up Docker publishing for the `vector-db-benchmark` project to Docker Hub repository `redis-performance/vector-db-benchmark`.

## 🔐 Required GitHub Secrets

To enable automated Docker publishing, you need to configure the following secrets in your GitHub repository:

### Setting up Docker Hub Secrets

1. **Go to your GitHub repository** → Settings → Secrets and variables → Actions

2. **Add the following repository secrets:**

   - **`DOCKER_USERNAME`**: Your Docker Hub username
   - **`DOCKER_PASSWORD`**: Your Docker Hub access token (NOT your password)

### Creating a Docker Hub Access Token

1. **Log in to Docker Hub** at https://hub.docker.com
2. **Go to Account Settings** → Security → Access Tokens
3. **Click "New Access Token"**
4. **Configure the token:**
   - Name: `GitHub Actions - vector-db-benchmark`
   - Permissions: `Read, Write, Delete`
5. **Copy the generated token** and use it as `DOCKER_PASSWORD` secret

⚠️ **Important**: Use an access token, not your Docker Hub password, for better security.

### Credential Validation

All Docker publishing workflows include automatic credential validation:

- **PR Validation**: Checks if credentials are available but continues without them (expected for forks)
- **Master/Release Publishing**: **Requires** credentials and fails if not configured
- **Local Testing**: Warns if credentials are missing but continues validation

This ensures that:
- External contributors can still validate Docker builds in PRs
- Publishing workflows fail fast if credentials are misconfigured
- Local development works regardless of credential status

## 🚀 Automated Publishing

Once secrets are configured, Docker images will be automatically published:

### Master Branch Commits
- **Trigger**: Every push to `master` branch
- **Tags**: `latest`, `master-{sha}`, `master-{timestamp}`
- **Platforms**: `linux/amd64`, `linux/arm64`

### Releases
- **Trigger**: When a GitHub release is published
- **Tags**: `{version}`, `{major}.{minor}`, `{major}`, `latest`
- **Platforms**: `linux/amd64`, `linux/arm64`
- **Security**: Includes Trivy vulnerability scanning

### Example Tags for Release v1.2.3
```
redis-performance/vector-db-benchmark:v1.2.3
redis-performance/vector-db-benchmark:1.2.3
redis-performance/vector-db-benchmark:1.2
redis-performance/vector-db-benchmark:1
redis-performance/vector-db-benchmark:latest
```

## 🛠️ Manual Building and Publishing

### Local Build
```bash
# Build only
./docker-build.sh

# Build with custom tag
./docker-build.sh -t v1.0.0

# Build and push to Docker Hub
./docker-build.sh -t v1.0.0 --push

# Multi-platform build and push
./docker-build.sh -p linux/amd64,linux/arm64 --push

# Run local validation tests (mimics GitHub Action)
./docker-test.sh
```

### Prerequisites for Manual Push
```bash
# Login to Docker Hub
docker login

# Or use environment variables
export DOCKER_USERNAME=your_username
export DOCKER_PASSWORD=your_access_token
./docker-build.sh --push
```

## 📦 Using the Docker Images

### Pull and Run
```bash
# Latest version
docker pull redis-performance/vector-db-benchmark:latest
docker run --rm redis-performance/vector-db-benchmark:latest run.py --help

# Specific version
docker pull redis-performance/vector-db-benchmark:v1.2.3
docker run --rm redis-performance/vector-db-benchmark:v1.2.3 run.py --help
```

### Example Usage
```bash
# Basic Redis benchmark
docker run --rm --network=host redis-performance/vector-db-benchmark:latest \
  run.py --host localhost --engines redis --dataset random-100 --experiment redis-m-16-ef-64

# With custom Redis host
docker run --rm redis-performance/vector-db-benchmark:latest \
  run.py --host redis-server --engines redis --dataset random-100 --experiment redis-m-16-ef-64

# With results output (mount current directory)
docker run --rm -v $(pwd)/results:/app/results --network=host \
  redis-performance/vector-db-benchmark:latest \
  run.py --host localhost --engines redis --dataset random-100 --experiment redis-m-16-ef-64

# Using docker-compose for full setup
docker-compose up redis
docker-compose run --rm vector-db-benchmark run.py --host redis --engines redis --experiment redis-m-16-ef-64
```

## 🔍 Monitoring and Troubleshooting

### GitHub Actions
- Check workflow runs in **Actions** tab
- View build logs and summaries
- Monitor security scan results

### PR Validation
- **Automatic Docker build validation** on all pull requests
- Tests both single-platform (`linux/amd64`) and multi-platform builds
- Validates basic functionality and Redis connectivity
- Provides PR comments with build status and details
- Prevents merging PRs with broken Docker builds

### Docker Hub
- View images at: https://hub.docker.com/r/redis-performance/vector-db-benchmark
- Check image sizes and platforms
- Review vulnerability scan results

### Common Issues

1. **Authentication Failed**
   - Verify `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets
   - Ensure access token has correct permissions

2. **Build Failed**
   - Check Python version compatibility
   - Verify Dockerfile syntax
   - Review build logs in Actions

3. **Push Failed**
   - Confirm repository exists on Docker Hub
   - Check network connectivity
   - Verify permissions

## 🏗️ Architecture

### Multi-Stage Build
- **Stage 1**: Python build environment with full toolchain
- **Stage 2**: Minimal runtime with only necessary components
- **Result**: Optimized image size with security best practices

### Security Features
- Non-root user execution
- Minimal attack surface
- Vulnerability scanning with Trivy
- Signed images (when configured)

### Performance Optimizations
- Layer caching for faster builds
- Multi-platform support
- Efficient .dockerignore
- Build argument optimization
